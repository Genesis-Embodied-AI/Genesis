from functools import wraps
from pathlib import Path

import igl
import numpy as np
import quadrants as qd
import torch

import genesis as gs
import genesis.utils.element as eu
import genesis.utils.geom as gu
import genesis.utils.mesh as mu
from genesis.engine.entities.rigid_entity import RigidLink
from genesis.engine.couplers import SAPCoupler
from genesis.engine.states.cache import QueriedStates
from genesis.engine.states.entities import FEMEntityState
from genesis.utils.misc import to_gs_tensor, tensor_to_array, broadcast_tensor

from .base_entity import Entity


def assert_muscle(method):
    @wraps(method)
    def wrapper(self, *args, **kwargs):
        if not isinstance(self.material, gs.materials.FEM.Muscle):
            gs.raise_exception("This method is only supported by entities with 'FEM.Muscle' material.")
        return method(self, *args, **kwargs)

    return wrapper


@qd.data_oriented
class FEMEntity(Entity):
    """
    A finite element method (FEM)-based entity for deformable simulation.

    This class represents a deformable object using tetrahedral elements. It interfaces with
    the physics solver to handle state updates, checkpointing, gradients, and actuation
    for physics-based simulation in batched environments.

    Parameters
    ----------
    scene : Scene
        The simulation scene that this entity belongs to.
    solver : Solver
        The physics solver instance used for simulation.
    material : Material
        The material properties defining elasticity, density, etc.
    morph : Morph
        The morph specification that defines the entity's shape.
    surface : Surface
        The surface mesh associated with the entity (for rendering or collision).
    idx : int
        Unique identifier of the entity within the scene.
    v_start : int, optional
        Starting index of this entity's vertices in the global vertex array (default is 0).
    el_start : int, optional
        Starting index of this entity's elements in the global element array (default is 0).
    s_start : int, optional
        Starting index of this entity's surface triangles in the global surface array (default is 0).
    """

    def __init__(
        self, scene, solver, material, morph, surface, idx, v_start=0, el_start=0, s_start=0, name: str | None = None
    ):
        super().__init__(idx, scene, morph, solver, material, surface, name=name)

        self._v_start = v_start  # offset for vertex index of elements
        self._el_start = el_start  # offset for element index
        self._s_start = s_start  # offset for surface triangles
        self._step_global_added = None

        self._surface.update_texture()

        self.sample()

        # Check if this is cloth (elements are already triangles)
        from genesis.engine.materials.FEM.cloth import Cloth as ClothMaterial

        is_cloth = isinstance(self.material, ClothMaterial)

        if is_cloth:
            # For cloth, elements are already surface triangles
            self._surface_tri_np = self.elems
            self._n_surfaces = len(self._surface_tri_np)
            if self._n_surfaces > 0:
                self._n_surface_vertices = len(np.unique(self._surface_tri_np))
            else:
                self._n_surface_vertices = 0
            # For cloth, each triangle is its own "element"
            self._surface_el_np = np.arange(self.elems.shape[0], dtype=gs.np_int)
        else:
            # For volumetric FEM, extract surface triangles from tetrahedral elements
            el2tri = np.array(
                [  # follow the order with correct normal
                    [[v[0], v[2], v[1]], [v[1], v[2], v[3]], [v[0], v[1], v[3]], [v[0], v[3], v[2]]] for v in self.elems
                ],
                dtype=gs.np_int,
            )
            all_tri = el2tri.reshape((-1, 3))
            all_tri_sorted = np.sort(all_tri, axis=1)
            _, unique_idcs, cnt = np.unique(all_tri_sorted, axis=0, return_counts=True, return_index=True)
            unique_tri = all_tri[unique_idcs]
            surface_tri = unique_tri[cnt == 1]

            self._surface_tri_np = surface_tri
            self._n_surfaces = len(self._surface_tri_np)

            if self._n_surfaces > 0:
                self._n_surface_vertices = len(np.unique(self._surface_tri_np))
            else:
                self._n_surface_vertices = 0

            tri2el = np.repeat(np.arange(self.elems.shape[0], dtype=gs.np_int)[:, np.newaxis], 4, axis=1)
            unique_el = tri2el.flat[unique_idcs]
            self._surface_el_np = unique_el[cnt == 1]

        if isinstance(self.sim.coupler, SAPCoupler):
            self.compute_pressure_field()

        self.init_tgt_vars()
        self.init_ckpt()

        self._queried_states = QueriedStates()

        self.active = False  # This attribute is only used in forward pass. It should NOT be used during backward pass.

    # ------------------------------------------------------------------------------------
    # ----------------------------------- basic entity ops -------------------------------
    # ------------------------------------------------------------------------------------

    def _sanitize_verts_idx_local(self, verts_idx_local=None, envs_idx=None):
        if verts_idx_local is None:
            verts_idx_local = range(self.n_vertices)

        if envs_idx is None:
            verts_idx_local_ = broadcast_tensor(verts_idx_local, gs.tc_int, (-1,), ("verts_idx",))
        else:
            verts_idx_local_ = broadcast_tensor(
                verts_idx_local, gs.tc_int, (len(envs_idx), -1), ("envs_idx", "verts_idx")
            )

        # FIXME: This check is too expensive
        # if not (0 <= verts_idx_local_ & verts_idx_local_ < self.n_vertices).all():
        #     gs.raise_exception("Elements of `verts_idx_local' are out-of-range.")

        return verts_idx_local_.contiguous()

    def _sanitize_verts_tensor(self, tensor, dtype, verts_idx=None, envs_idx=None, element_shape=(), *, batched=True):
        n_vertices = verts_idx.shape[-1] if verts_idx is not None else self.n_vertices
        if batched:
            assert envs_idx is not None
            batch_shape = (len(envs_idx), n_vertices)
            dim_names = ("envs_idx", "verts_idx", *("" for _ in element_shape))
        else:
            batch_shape = (n_vertices,)
            dim_names = ("verts_idx", *("" for _ in element_shape))
        tensor_shape = (*batch_shape, *element_shape)

        return broadcast_tensor(tensor, dtype, tensor_shape, dim_names).contiguous()

    def set_position(self, pos):
        """
        Set the target position(s) for the FEM entity.

        Parameters
        ----------
        pos : torch.Tensor or array-like
            The desired position(s). Can be:
            - (3,): a single COM offset vector.
            - (n_vertices, 3): per-vertex positions for all vertices.
            - (n_envs, 3): per-environment COM offsets.
            - (n_envs, n_vertices, 3): full batched per-vertex positions.

        Raises
        ------
        Exception
            If the tensor shape is not supported.
        """
        self._assert_active()
        gs.logger.warning("Manually setting element positions. This is not recommended and could break gradient flow.")

        pos = to_gs_tensor(pos)

        is_valid = False
        if pos.ndim == 1:
            if pos.shape == (3,):
                pos = self.init_positions_COM_offset + pos
                self._tgt["pos"] = pos[None].tile((self._sim._B, 1, 1))
                is_valid = True
        elif pos.ndim == 2:
            if pos.shape == (self.n_vertices, 3):
                self._tgt["pos"] = pos[None].tile((self._sim._B, 1, 1))
                is_valid = True
            elif pos.shape == (self._sim._B, 3):
                pos = self.init_positions_COM_offset[None] + pos[:, None]
                self._tgt["pos"] = pos
                is_valid = True
        elif pos.ndim == 3:
            if pos.shape == (self._sim._B, self.n_vertices, 3):
                self._tgt["pos"] = pos
                is_valid = True
        if not is_valid:
            gs.raise_exception("Tensor shape not supported.")

        # Immediately flush to the solver's internal elements_v so that the
        # visualizer can render the updated positions without scene.step().
        if is_valid and self._tgt["pos"] is not None:
            self.set_pos(self._sim.cur_substep_local, self._tgt["pos"])

    def set_velocity(self, vel):
        """
        Set the target velocity(ies) for the FEM entity.

        Parameters
        ----------
        vel : torch.Tensor or array-like
            The desired velocity(ies). Can be:
            - (3,): a global velocity vector for all vertices.
            - (n_vertices, 3): per-vertex velocities.
            - (n_envs, 3): per-environment velocities broadcast to all vertices.
            - (n_envs, n_vertices, 3): full batched per-vertex velocities.

        Raises
        ------
        Exception
            If the tensor shape is not supported.
        """
        self._assert_active()
        gs.logger.warning("Manually setting element velocities. This is not recommended and could break gradient flow.")

        vel = to_gs_tensor(vel)

        is_valid = False
        if vel.ndim == 1:
            if vel.shape == (3,):
                self._tgt["vel"] = vel.tile((self._sim._B, self.n_vertices, 1))
                is_valid = True
        elif vel.ndim == 2:
            if vel.shape == (self.n_vertices, 3):
                self._tgt["vel"] = vel[None].tile((self._sim._B, 1, 1))
                is_valid = True
            elif vel.shape == (self._sim._B, 3):
                self._tgt["vel"] = vel[:, None].tile((1, self.n_vertices, 1))
                is_valid = True
        elif vel.ndim == 3:
            if vel.shape == (self._sim._B, self.n_vertices, 3):
                self._tgt["vel"] = vel
                is_valid = True
        if not is_valid:
            gs.raise_exception("Tensor shape not supported.")

    @assert_muscle
    def set_actuation(self, actu):
        """
        Set the actuation signal for the FEM entity.

        Parameters
        ----------
        actu : torch.Tensor or array-like
            The actuation tensor. Can be:
            - (): a single scalar for all groups.
            - (n_groups,): group-level actuation.
            - (n_envs, n_groups): batch of group-level actuation signals.

        Raises
        ------
        Exception
            If the tensor shape is not supported or per-element actuation is attempted.
        """
        self._assert_active()

        actu = to_gs_tensor(actu)

        is_valid = False
        n_groups = self.material.n_groups
        if actu.ndim == 0:
            self._tgt["actu"] = actu.tile((self._sim._B, n_groups))
            is_valid = True
        elif actu.ndim == 1:
            if actu.shape == (n_groups,):
                self._tgt["actu"] = actu[None].tile((self._sim._B, 1))
                is_valid = True
            elif actu.shape == (self.n_elements,):
                gs.raise_exception("Cannot set per-element actuation.")
        elif actu.ndim == 2:
            if actu.shape == (self._sim._B, n_groups):
                self._tgt["actu"] = actu
                is_valid = True
        if not is_valid:
            gs.raise_exception("Tensor shape not supported.")

    def set_muscle(self, muscle_group=None, muscle_direction=None):
        """
        Set the muscle group and/or muscle direction for the FEM entity.

        Parameters
        ----------
        muscle_group : torch.Tensor or array-like, optional
            Tensor of shape (n_elements,) specifying the muscle group ID for each element.

        muscle_direction : torch.Tensor or array-like, optional
            Tensor of shape (n_elements, 3) specifying unit direction vectors for muscle forces.

        Raises
        ------
        AssertionError
            If tensor shapes are incorrect or normalization fails.
        """
        self._assert_active()

        n_groups = self.material.n_groups
        max_group_id = muscle_group.max().item()

        muscle_group = to_gs_tensor(muscle_group)

        assert muscle_group.shape == (self.n_elements,)
        assert isinstance(max_group_id, int) and max_group_id < n_groups

        self.set_muscle_group(muscle_group)

        if muscle_direction is not None:
            muscle_direction = to_gs_tensor(muscle_direction)
            assert muscle_direction.shape == (self.n_elements, 3)
            assert ((1.0 - muscle_direction.norm(dim=-1)).abs() < gs.EPS).all()

            self.set_muscle_direction(muscle_direction)

    def get_state(self):
        state = FEMEntityState(self, self._sim.cur_step_global)
        self.get_frame(self._sim.cur_substep_local, state.pos, state.vel, state.active)

        # we store all queried states to track gradient flow
        self._queried_states.append(state)

        return state

    def deactivate(self):
        gs.logger.info(f"{self.__class__.__name__} <{self.id}> deactivated.")
        self._tgt["act"] = gs.INACTIVE
        self.active = False

    def activate(self):
        gs.logger.info(f"{self.__class__.__name__} <{self.id}> activated.")
        self._tgt["act"] = gs.ACTIVE
        self.active = True

    # ------------------------------------------------------------------------------------
    # ----------------------------------- instantiation ----------------------------------
    # ------------------------------------------------------------------------------------

    def instantiate(self, verts, elems):
        """
        Initialize FEM entity with given vertices and elements.

        Parameters
        ----------
        verts : np.ndarray
            Array of vertex positions with shape (n_vertices, 3).

        elems : np.ndarray
            Array of tetrahedral elements with shape (n_elements, 4), indexing into verts.

        Raises
        ------
        Exception
            If no vertices are provided.
        """
        verts = verts.astype(gs.np_float, copy=False)
        elems = elems.astype(gs.np_int, copy=False)

        # rotate
        R = gu.quat_to_R(np.array(self.morph.quat, dtype=gs.np_float))
        verts_COM = verts.mean(axis=0)
        init_positions = (verts - verts_COM) @ R.T + verts_COM

        if not init_positions.shape[0] > 0:
            gs.raise_exception("Entity has zero vertices.")

        self.init_positions = gs.tensor(init_positions)
        self.init_positions_COM_offset = self.init_positions - gs.tensor(verts_COM)

        self.elems = elems

    def sample(self):
        """
        Sample mesh and elements based on the entity's morph type.

        For Cloth material, loads surface mesh directly without tetrahedralization.
        For regular FEM materials, tetrahedralizes the mesh.

        Raises
        ------
        Exception
            If the morph type is unsupported.
        """
        from genesis.engine.materials.FEM.cloth import Cloth as ClothMaterial

        is_cloth = isinstance(self.material, ClothMaterial)
        self._uvs = None

        if is_cloth:
            # Cloth: load surface mesh directly (no tetrahedralization)
            if isinstance(self.morph, gs.options.morphs.Mesh):
                import trimesh

                mesh = trimesh.load_mesh(self._morph.file)
                verts = mesh.vertices * self._morph.scale + np.array(self._morph.pos)
                faces = mesh.faces
                # For cloth, we store faces as "elements" (treating them as surface elements)
                self.instantiate(verts, faces)

                # Load UVs from mesh (1:1 mapping for cloth).
                # UVs are not always available in 3D file, in case they are missing we set the entity UVs to None when UVs are None,
                # the solver will use 0 UVs for rendering. A mesh with 0 UVs means that no tangent directions can be recomputed,
                # thus texture mapping and anisotropic surfaces will not work properly.
                self._uvs = None
                if isinstance(mesh.visual, trimesh.visual.texture.TextureVisuals) and mesh.visual.uv is not None:
                    self._uvs = mesh.visual.uv.astype(gs.np_float, copy=False)
            else:
                gs.raise_exception(f"Cloth material only supports Mesh morph. Got: {self.morph}.")
        else:
            # Regular FEM: tetrahedralize mesh
            if isinstance(self.morph, gs.options.morphs.Sphere):
                verts, elems = eu.sphere_to_elements(
                    pos=self._morph.pos,
                    radius=self._morph.radius,
                    tet_cfg=self.tet_cfg,
                )
            elif isinstance(self.morph, gs.options.morphs.Box):
                verts, elems = eu.box_to_elements(
                    pos=self._morph.pos,
                    size=self._morph.size,
                    tet_cfg=self.tet_cfg,
                )
            elif isinstance(self.morph, gs.options.morphs.Cylinder):
                verts, elems = eu.cylinder_to_elements()
            elif isinstance(self.morph, gs.options.morphs.Mesh):
                # We don't need to proces UVs here because the tetrahedralization process append new vertices
                # and faces at the end of the vertex list, thus the original UVs are preserved at the beginning.
                # We can't generate UVs for newly created internal vertices as it doesn't make sense but they're
                # not used for rendering so it's fine.
                verts, elems, self._uvs = eu.mesh_to_elements(
                    file=self._morph.file,
                    pos=self._morph.pos,
                    scale=self._morph.scale,
                    tet_cfg=self.tet_cfg,
                )
            else:
                gs.raise_exception(f"Unsupported morph: {self.morph}.")

            self.instantiate(*eu.split_all_surface_tets(verts, elems))

    def _add_to_solver(self, in_backward=False):
        from genesis.engine.materials.FEM.cloth import Cloth as ClothMaterial

        is_cloth = isinstance(self.material, ClothMaterial)

        if not in_backward:
            self._step_global_added = self._sim.cur_step_global
            gs.logger.info(
                f"Entity {self.uid} added. class: {self.__class__.__name__}, morph: {self.morph.__class__.__name__}, size: ({self.n_elements}, {self.n_vertices}), material: {self.material}."
            )

        # Convert to appropriate numpy array types
        verts_numpy = tensor_to_array(self.init_positions, dtype=gs.np_float)
        uvs_np = self._uvs if self._uvs is not None else np.zeros((0, 2), dtype=gs.np_float)

        if is_cloth:
            # Cloth: add only vertices and surfaces for rendering (no physics computation)
            gs.logger.info(
                f"Entity {self.uid} is cloth - adding to FEM solver for rendering only (physics managed by IPC)"
            )
            self._solver._kernel_add_cloth_for_rendering(
                f=self._sim.cur_substep_local,
                n_surfaces=self._n_surfaces,
                v_start=self._v_start,
                s_start=self._s_start,
                verts=verts_numpy,
                tri2v=self._surface_tri_np,
                uvs=uvs_np,
            )
        else:
            # Regular FEM: add vertices, elements, and surfaces for physics and rendering
            elems_np = self.elems.astype(gs.np_int, copy=False)
            self._solver._kernel_add_elements(
                f=self._sim.cur_substep_local,
                mat_idx=self._material.idx,
                mat_mu=self._material.mu,
                mat_lam=self._material.lam,
                mat_rho=self._material.rho,
                mat_friction_mu=self._material.friction_mu,
                n_surfaces=self._n_surfaces,
                v_start=self._v_start,
                el_start=self._el_start,
                s_start=self._s_start,
                verts=verts_numpy,
                elems=elems_np,
                tri2v=self._surface_tri_np,
                tri2el=self._surface_el_np,
                uvs=uvs_np,
            )

        self.active = True

    def compute_pressure_field(self):
        """
        Compute the pressure field for the FEM entity based on its tetrahedral elements.

        For hydroelastic contact: https://drake.mit.edu/doxygen_cxx/group__hydroelastic__user__guide.html

        Notes
        -----
        https://github.com/RobotLocomotion/drake/blob/master/geometry/proximity/make_mesh_field.cc
        TODO: Add margin support
        Drake's implementation of margin seems buggy.
        """
        init_positions = tensor_to_array(self.init_positions)
        signed_distance, *_ = igl.signed_distance(init_positions, init_positions, self._surface_tri_np)
        signed_distance = signed_distance.astype(gs.np_float, copy=False)

        unsigned_distance = np.abs(signed_distance)
        max_distance = np.max(unsigned_distance)
        if max_distance < gs.EPS:
            gs.raise_exception(
                f"Pressure field max distance is too small: {max_distance}. "
                "This might be due to a mesh having no internal vertices."
            )
        self.pressure_field_np = unsigned_distance / max_distance * self.material._hydroelastic_modulus  # normalize

    # ------------------------------------------------------------------------------------
    # ---------------------------- checkpoint and buffer ---------------------------------
    # ------------------------------------------------------------------------------------

    def init_tgt_keys(self):
        """
        Initialize the keys used in target state management.

        This defines which physical properties (e.g., position, velocity) will be tracked for checkpointing and buffering.
        """
        self._tgt_keys = ["vel", "pos", "act", "actu"]

    def init_tgt_vars(self):
        """
        Initialize the target state variables and their buffers.

        This sets up internal dictionaries to store per-step target values for properties like velocity, position, actuation, and activation.
        """

        # temp variable to store targets for next step
        self._tgt = dict()
        self._tgt_buffer = dict()
        self.init_tgt_keys()

        for key in self._tgt_keys:
            self._tgt[key] = None
            self._tgt_buffer[key] = list()

    def init_ckpt(self):
        """
        Initialize the checkpoint storage dictionary.

        Creates an empty container for storing simulation checkpoints.
        """

        self._ckpt = dict()

    def save_ckpt(self, ckpt_name):
        """
        Save the current target state buffers to a named checkpoint.

        Parameters
        ----------
        ckpt_name : str
            The name to identify the checkpoint.

        Notes
        -----
        After saving, the internal target buffers are cleared to prepare for new input.
        """

        if ckpt_name not in self._ckpt:
            self._ckpt[ckpt_name] = {
                "_tgt_buffer": dict(),
            }

        for key in self._tgt_keys:
            self._ckpt[ckpt_name]["_tgt_buffer"][key] = list(self._tgt_buffer[key])
            self._tgt_buffer[key].clear()

    def load_ckpt(self, ckpt_name):
        """
        Load a previously saved target state buffer from a named checkpoint.

        Parameters
        ----------
        ckpt_name : str
            The name of the checkpoint to load.

        Raises
        ------
        KeyError
            If the checkpoint name is not found.
        """

        for key in self._tgt_keys:
            self._tgt_buffer[key] = list(self._ckpt[ckpt_name]["_tgt_buffer"][key])

    def reset_grad(self):
        """
        Clear all stored gradient-related buffers.

        This resets the target buffer and clears any queried states used for gradient tracking.
        """

        for key in self._tgt_keys:
            self._tgt_buffer[key].clear()
        self._queried_states.clear()

    def process_input(self, in_backward=False):
        """
        Push position, velocity, and activation target states into the simulator.

        Parameters
        ----------
        in_backward : bool, default=False
            Whether the simulation is in the backward (gradient) pass.
        """
        if in_backward:
            # use negative index because buffer length might not be full
            index = self._sim.cur_step_local - self._sim.max_steps_local
            for key in self._tgt_keys:
                self._tgt[key] = self._tgt_buffer[key][index]

        else:
            for key in self._tgt_keys:
                self._tgt_buffer[key].append(self._tgt[key])

        # set_pos followed by set_vel, because set_pos resets velocity.
        if self._tgt["pos"] is not None:
            self._tgt["pos"].assert_contiguous()
            self._tgt["pos"].assert_sceneless()
            self.set_pos(self._sim.cur_substep_local, self._tgt["pos"])

        if self._tgt["vel"] is not None:
            self._tgt["vel"].assert_contiguous()
            self._tgt["vel"].assert_sceneless()
            self.set_vel(self._sim.cur_substep_local, self._tgt["vel"])

        if self._tgt["act"] is not None:
            assert self._tgt["act"] in (gs.ACTIVE, gs.INACTIVE)
            self.set_active(self._sim.cur_substep_local, self._tgt["act"])

        if self._tgt["actu"] is not None:
            self._tgt["actu"].assert_contiguous()
            self._tgt["actu"].assert_sceneless()
            self.set_actu(self._sim.cur_substep_local, self._tgt["actu"])

        for key in self._tgt_keys:
            self._tgt[key] = None

    def process_input_grad(self):
        """
        Process gradients of input states and propagate them backward.

        Notes
        -----
        Automatically applies the backward hooks for position, velocity, and actuation tensors.
        Clears the gradients in the solver to avoid double accumulation.
        """
        _tgt_actu = self._tgt_buffer["actu"].pop()
        _tgt_vel = self._tgt_buffer["vel"].pop()
        _tgt_pos = self._tgt_buffer["pos"].pop()

        if _tgt_actu is not None and _tgt_actu.requires_grad:
            _tgt_actu._backward_from_qd(self.set_actu_grad, self._sim.cur_substep_local)

        if _tgt_vel is not None and _tgt_vel.requires_grad:
            _tgt_vel._backward_from_qd(self.set_vel_grad, self._sim.cur_substep_local)

        if _tgt_pos is not None and _tgt_pos.requires_grad:
            _tgt_pos._backward_from_qd(self.set_pos_grad, self._sim.cur_substep_local)

        if _tgt_vel is not None or _tgt_pos is not None or _tgt_actu is not None:
            # manually zero the grad since manually setting state breaks gradient flow
            self.clear_grad(self._sim.cur_substep_local)

    def _assert_active(self):
        if not self.active:
            gs.raise_exception(f"{self.__class__.__name__} is inactive. Call `entity.activate()` first.")

    # ------------------------------------------------------------------------------------
    # ---------------------------- interfacing with solver -------------------------------
    # ------------------------------------------------------------------------------------

    def set_pos(self, f, pos):
        """
        Set element positions in the solver.

        Parameters
        ----------
        f : int
            Current substep/frame index.

        pos : gs.Tensor
            Tensor of shape (n_envs, n_vertices, 3) containing new positions.
        """

        self._solver._kernel_set_elements_pos(
            f=f,
            element_v_start=self._v_start,
            n_vertices=self.n_vertices,
            pos=pos,
        )

    def set_pos_grad(self, f, pos_grad):
        """
        Set gradient of element positions in the solver.

        Parameters
        ----------
        f : int
            Current substep/frame index.

        pos_grad : gs.Tensor
            Tensor of shape (n_envs, n_vertices, 3) containing gradients of positions.
        """

        self._solver._kernel_set_elements_pos_grad(
            f=f,
            element_v_start=self._v_start,
            n_vertices=self.n_vertices,
            pos_grad=pos_grad,
        )

    def set_vel(self, f, vel):
        """
        Set element velocities in the solver.

        Parameters
        ----------
        f : int
            Current substep/frame index.

        vel : gs.Tensor
            Tensor of shape (n_envs, n_vertices, 3) containing velocities.
        """

        self._solver._kernel_set_elements_vel(
            f=f,
            element_v_start=self._v_start,
            n_vertices=self.n_vertices,
            vel=vel,
        )

    def set_vel_grad(self, f, vel_grad):
        """
        Set gradient of element velocities in the solver.

        Parameters
        ----------
        f : int
            Current substep/frame index.

        vel_grad : gs.Tensor
            Tensor of shape (n_envs, n_vertices, 3) containing gradients of velocities.
        """

        self._solver._kernel_set_elements_vel_grad(
            f=f,
            element_v_start=self._v_start,
            n_vertices=self.n_vertices,
            vel_grad=vel_grad,
        )

    def set_actu(self, f, actu):
        """
        Set actuation values for elements in the solver.

        Parameters
        ----------
        f : int
            Current substep/frame index.

        actu : gs.Tensor
            Tensor of shape (n_envs, n_groups) specifying actuation values.
        """

        self._solver._kernel_set_elements_actu(
            f=f,
            element_el_start=self._el_start,
            n_elements=self.n_elements,
            n_groups=self.material.n_groups,
            actu=actu,
        )

    def set_actu_grad(self, f, actu_grad):
        """
        Set gradient of actuation values in the solver.

        Parameters
        ----------
        f : int
            Current substep/frame index.

        actu_grad : gs.Tensor
            Tensor of shape (n_envs, n_groups) specifying gradients of actuation.
        """

        self._solver._kernel_set_elements_actu(
            f=f,
            element_el_start=self._el_start,
            n_elements=self.n_elements,
            actu_grad=actu_grad,
        )

    def set_active(self, f, active):
        """
        Set the active status of each element.

        Parameters
        ----------
        f : int
            Current substep/frame index.

        active : int
            Activity flag (gs.ACTIVE or gs.INACTIVE).
        """

        self._solver._kernel_set_active(
            f=f,
            element_el_start=self._el_start,
            n_elements=self.n_elements,
            active=active,
        )

    @assert_muscle
    def set_muscle_group(self, muscle_group):
        """
        Set muscle group index for each element.

        Parameters
        ----------
        muscle_group : torch.Tensor
            Tensor of shape (n_elements,) specifying muscle group IDs.
        """

        self._solver._kernel_set_muscle_group(
            element_el_start=self._el_start,
            n_elements=self.n_elements,
            muscle_group=muscle_group,
        )

    @assert_muscle
    def set_muscle_direction(self, muscle_direction):
        """
        Set muscle force direction for each element.

        Parameters
        ----------
        muscle_direction : torch.Tensor
            Tensor of shape (n_elements, 3) with unit direction vectors.
        """

        self._solver._kernel_set_muscle_direction(
            element_el_start=self._el_start,
            n_elements=self.n_elements,
            muscle_direction=muscle_direction,
        )

    def set_vertex_constraints(
        self, verts_idx_local, target_poss=None, link=None, is_soft_constraint=False, stiffness=0.0, envs_idx=None
    ):
        """
        Set vertex constraints for specified vertices.

        Parameters
        ----------
            verts_idx_local : array_like
                List of local vertex indices to constrain.
            target_poss : array_like, shape (len(verts_idx), 3), optional
                List of target positions [x, y, z] for each vertex. If not provided, the initial positions are used.
            link : RigidLink
                Optional rigid link for the vertices to follow, maintaining relative position.
            is_soft_constraint: bool
                By default, use a hard constraint directly sets position and zero velocity.
                A soft constraint uses a spring force to pull the vertex towards the target position.
            stiffness : float
                Specify a spring stiffness for a soft constraint. Critical damping is applied.
            envs_idx : array_like, optional
                List of environment indices to apply the constraints to. If None, applies to all environments.
        """
        if self._solver._use_implicit_solver and not self._solver._enable_vertex_constraints:
            gs.raise_exception(
                "This feature is disabled. Please set 'enable_vertex_constraints=True' when using FEM implicit solver."
            )

        if not self._solver._constraints_initialized:
            self._solver.init_constraints()

        use_current_poss = target_poss is None
        envs_idx = self._scene._sanitize_envs_idx(envs_idx)
        verts_idx_local = self._sanitize_verts_idx_local(verts_idx_local, envs_idx)
        verts_idx = verts_idx_local + self._v_start
        target_poss = self._sanitize_verts_tensor(target_poss, gs.tc_float, verts_idx, envs_idx, (3,))

        if use_current_poss:
            self._kernel_get_verts_pos(self._sim.cur_substep_local, target_poss, verts_idx)

        if link is None:
            link_idx = -1
            link_init_pos = torch.zeros((self._sim._B, 3), dtype=gs.tc_float, device=gs.device)
            link_init_quat = torch.zeros((self._sim._B, 4), dtype=gs.tc_float, device=gs.device)
        else:
            assert isinstance(link, RigidLink), "Only RigidLink is supported for vertex constraints."
            link_idx = link.idx
            link_init_pos = link.get_pos()
            link_init_quat = link.get_quat()
            if self._scene.n_envs == 0:
                link_init_pos = link_init_pos[None]
                link_init_quat = link_init_quat[None]

        self._solver._kernel_set_vertex_constraints(
            self._sim.cur_substep_local,
            verts_idx,
            target_poss,
            is_soft_constraint,
            stiffness,
            link_idx,
            link_init_pos,
            link_init_quat,
            envs_idx,
        )

    def update_constraint_targets(self, verts_idx_local, target_poss, envs_idx=None):
        """Update target positions for existing constraints."""
        if not self._solver._constraints_initialized:
            gs.logger.warning("Ignoring update_constraint_targets; constraints have not been initialized.")
            return

        assert target_poss is not None
        envs_idx = self._scene._sanitize_envs_idx(envs_idx)
        verts_idx_local = self._sanitize_verts_idx_local(verts_idx_local, envs_idx)
        verts_idx = verts_idx_local + self._v_start
        target_poss = self._sanitize_verts_tensor(target_poss, gs.tc_float, verts_idx, envs_idx, (3,))

        self._solver._kernel_update_constraint_targets(verts_idx, target_poss, envs_idx)

    def remove_vertex_constraints(self, verts_idx_local=None, envs_idx=None):
        """Remove constraints from specified vertices, or all if None."""
        if not self._solver._constraints_initialized:
            gs.logger.warning("Ignoring remove_vertex_constraints; constraints have not been initialized.")
            return

        # FIXME: Quadrants 'fill' method is very inefficient. Try using zero-copy if possible.
        if verts_idx_local is None:
            self._solver.vertex_constraints.is_constrained.fill(0)
            return

        envs_idx = self._scene._sanitize_envs_idx(envs_idx)
        verts_idx_local = self._sanitize_verts_idx_local(verts_idx_local, envs_idx)
        verts_idx = verts_idx_local + self._v_start

        self._solver._kernel_remove_specific_constraints(verts_idx, envs_idx)

    @qd.kernel
    def _kernel_get_verts_pos(self, f: qd.i32, pos: qd.types.ndarray(), verts_idx: qd.types.ndarray()):
        # get current position of vertices
        for i_v, i_b in qd.ndrange(verts_idx.shape[0], verts_idx.shape[1]):
            i_global = verts_idx[i_v, i_b] + self.v_start
            for j in qd.static(range(3)):
                pos[i_b, i_v, j] = self._solver.elements_v[f, i_global, i_b].pos[j]

    def get_el2v(self):
        """
        Retrieve the element-to-vertex mapping.

        Returns
        -------
        el2v : gs.Tensor
            Tensor of shape (n_elements, 4) mapping each element to its vertex indices.
        """
        el2v = gs.zeros((self.n_elements, 4), dtype=int, requires_grad=False, scene=self.scene)
        self._solver._kernel_get_el2v(element_el_start=self._el_start, n_elements=self.n_elements, el2v=el2v)
        return el2v

    @qd.kernel
    def get_frame(self, f: qd.i32, pos: qd.types.ndarray(), vel: qd.types.ndarray(), active: qd.types.ndarray()):
        """
        Fetch the position, velocity, and activation state of the FEM entity at a specific substep.

        Parameters
        ----------
        f : int
            The substep/frame index to fetch the state from.

        pos : np.ndarray
            Output array of shape (n_envs, n_vertices, 3) to store positions.

        vel : np.ndarray
            Output array of shape (n_envs, n_vertices, 3) to store velocities.

        active : np.ndarray
            Output array of shape (n_envs, n_elements) to store active flags.
        """

        for i_v, i_b in qd.ndrange(self.n_vertices, self._sim._B):
            i_global = i_v + self.v_start
            for j in qd.static(range(3)):
                pos[i_b, i_v, j] = self._solver.elements_v[f, i_global, i_b].pos[j]
                vel[i_b, i_v, j] = self._solver.elements_v[f, i_global, i_b].vel[j]

        for i_v, i_b in qd.ndrange(self.n_elements, self._sim._B):
            i_global = i_v + self.el_start
            active[i_b, i_v] = self._solver.elements_el_ng[f, i_global, i_b].active

    @qd.kernel
    def clear_grad(self, f: qd.i32):
        """
        Zero out the gradients of position, velocity, and actuation for the current substep.

        Parameters
        ----------
        f : int
            The substep/frame index for which to clear gradients.

        Notes
        -----
        This method is primarily used during backward passes to manually reset gradients
        that may be corrupted by explicit state setting.
        """
        # TODO: not well-tested
        for i_v, i_b in qd.ndrange(self.n_vertices, self._sim._B):
            i_global = i_v + self.v_start
            self._solver.elements_v.grad[f, i_global, i_b].pos = 0
            self._solver.elements_v.grad[f, i_global, i_b].vel = 0

        for i_v, i_b in qd.ndrange(self.n_elements, self._sim._B):
            i_global = i_v + self.el_start
            self._solver.elements_el.grad[f, i_global, i_b].actu = 0

    # ------------------------------------------------------------------------------------
    # --------------------------------- naming methods -----------------------------------
    # ------------------------------------------------------------------------------------

    def _get_morph_identifier(self) -> str:
        morph = self._morph

        if isinstance(morph, gs.morphs.Box):
            return "fem_box"
        if isinstance(morph, gs.morphs.Sphere):
            return "fem_sphere"
        if isinstance(morph, gs.morphs.Cylinder):
            return "fem_cylinder"
        if isinstance(morph, gs.morphs.Mesh):
            return f"fem_{Path(morph.file).stem}"
        return "fem_entity"

    # ------------------------------------------------------------------------------------
    # ----------------------------------- properties -------------------------------------
    # ------------------------------------------------------------------------------------

    @property
    def n_vertices(self):
        """Number of vertices in the FEM entity."""
        return len(self.init_positions)

    @property
    def n_elements(self):
        """Number of tetrahedral elements in the FEM entity."""
        return len(self.elems)

    @property
    def n_surfaces(self):
        """Number of surface triangles extracted from the FEM mesh."""
        return self._n_surfaces

    @property
    def v_start(self):
        """Global vertex index offset for this entity."""
        return self._v_start

    @property
    def el_start(self):
        """Global element index offset for this entity."""
        return self._el_start

    @property
    def s_start(self):
        """Global surface triangle index offset for this entity."""
        return self._s_start

    @property
    def morph(self):
        """Morph specification used to generate the FEM mesh."""
        return self._morph

    @property
    def material(self):
        """Material properties of the FEM entity."""
        return self._material

    @property
    def surface(self):
        """Surface for rendering."""
        return self._surface

    @property
    def n_surface_vertices(self):
        """Number of unique vertices involved in surface triangles."""
        return self._n_surface_vertices

    @property
    def surface_triangles(self):
        """Surface triangles of the FEM mesh."""
        return self._surface_tri_np

    @property
    def uvs(self):
        """UV coordinates for this entity's vertices, or None if not available."""
        return self._uvs

    @property
    def tet_cfg(self):
        """Configuration of tetrahedralization."""
        tet_cfg = mu.generate_tetgen_config_from_morph(self.morph)
        return tet_cfg
