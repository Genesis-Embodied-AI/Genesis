import igl
import numpy as np
import gstaichi as ti
import torch

import genesis as gs
import genesis.utils.element as eu
import genesis.utils.geom as gu
import genesis.utils.mesh as mu
from genesis.engine.entities.rigid_entity import RigidLink
from genesis.engine.couplers import SAPCoupler
from genesis.engine.states.cache import QueriedStates
from genesis.engine.states.entities import FEMEntityState
from genesis.utils.misc import ALLOCATE_TENSOR_WARNING, to_gs_tensor, tensor_to_array

from .base_entity import Entity


@ti.data_oriented
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

    def __init__(self, scene, solver, material, morph, surface, idx, v_start=0, el_start=0, s_start=0):
        super().__init__(idx, scene, morph, solver, material, surface)

        self._v_start = v_start  # offset for vertex index of elements
        self._el_start = el_start  # offset for element index
        self._s_start = s_start  # offset for surface triangles
        self._step_global_added = None

        self._surface.update_texture()

        self.sample()

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
        gs.logger.warning("Manally setting element positions. This is not recommended and could break gradient flow.")

        pos = to_gs_tensor(pos)

        is_valid = False
        if pos.ndim == 1:
            if pos.shape == (3,):
                pos = self.init_positions_COM_offset + pos
                self._tgt["pos"] = pos.unsqueeze(0).tile((self._sim._B, 1, 1))
                is_valid = True
        elif pos.ndim == 2:
            if pos.shape == (self.n_vertices, 3):
                self._tgt["pos"] = pos.unsqueeze(0).tile((self._sim._B, 1, 1))
                is_valid = True
            elif pos.shape == (self._sim._B, 3):
                pos = self.init_positions_COM_offset.unsqueeze(0) + pos.unsqueeze(1)
                self._tgt["pos"] = pos
                is_valid = True
        elif pos.ndim == 3:
            if pos.shape == (self._sim._B, self.n_vertices, 3):
                self._tgt["pos"] = pos
                is_valid = True
        if not is_valid:
            gs.raise_exception("Tensor shape not supported.")

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
        gs.logger.warning("Manally setting element velocities. This is not recommended and could break gradient flow.")

        vel = to_gs_tensor(vel)

        is_valid = False
        if vel.ndim == 1:
            if vel.shape == (3,):
                self._tgt["vel"] = vel.tile((self._sim._B, self.n_vertices, 1))
                is_valid = True
        elif vel.ndim == 2:
            if vel.shape == (self.n_vertices, 3):
                self._tgt["vel"] = vel.unsqueeze(0).tile((self._sim._B, 1, 1))
                is_valid = True
            elif vel.shape == (self._sim._B, 3):
                self._tgt["vel"] = vel.unsqueeze(1).tile((1, self.n_vertices, 1))
                is_valid = True
        elif vel.ndim == 3:
            if vel.shape == (self._sim._B, self.n_vertices, 3):
                self._tgt["vel"] = vel
                is_valid = True
        if not is_valid:
            gs.raise_exception("Tensor shape not supported.")

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

        n_groups = getattr(self.material, "n_groups", 1)

        is_valid = False
        if actu.ndim == 0:
            self._tgt["actu"] = actu.tile((self._sim._B, n_groups))
            is_valid = True
        elif actu.ndim == 1:
            if actu.shape == (n_groups,):
                self._tgt["actu"] = actu.unsqueeze(0).tile((self._sim._B, 1))
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

        if muscle_group is not None:
            n_groups = getattr(self.material, "n_groups", 1)
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
        self.get_frame(
            self._sim.cur_substep_local,
            state.pos,
            state.vel,
            state.active,
        )

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
            gs.raise_exception(f"Entity has zero vertices.")

        self.init_positions = gs.tensor(init_positions)
        self.init_positions_COM_offset = self.init_positions - gs.tensor(verts_COM)

        self.elems = elems

    def sample(self):
        """
        Sample mesh and elements based on the entity's morph type.

        Raises
        ------
        Exception
            If the morph type is unsupported.
        """

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
            verts, elems = eu.mesh_to_elements(
                file=self._morph.file,
                pos=self._morph.pos,
                scale=self._morph.scale,
                tet_cfg=self.tet_cfg,
            )
        else:
            gs.raise_exception(f"Unsupported morph: {self.morph}.")

        self.instantiate(*eu.split_all_surface_tets(verts, elems))

    def _add_to_solver(self, in_backward=False):
        if not in_backward:
            self._step_global_added = self._sim.cur_step_global
            gs.logger.info(
                f"Entity {self.uid} added. class: {self.__class__.__name__}, morph: {self.morph.__class__.__name__}, size: ({self.n_elements}, {self.n_vertices}), material: {self.material}."
            )

        # Convert to appropriate numpy array types
        elems_np = self.elems.astype(gs.np_int, copy=False)
        verts_numpy = tensor_to_array(self.init_positions, dtype=gs.np_float)

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

        if not ckpt_name in self._ckpt:
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
            assert self._tgt["act"] in [gs.ACTIVE, gs.INACTIVE]
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
            _tgt_actu._backward_from_ti(self.set_actu_grad, self._sim.cur_substep_local)

        if _tgt_vel is not None and _tgt_vel.requires_grad:
            _tgt_vel._backward_from_ti(self.set_vel_grad, self._sim.cur_substep_local)

        if _tgt_pos is not None and _tgt_pos.requires_grad:
            _tgt_pos._backward_from_ti(self.set_pos_grad, self._sim.cur_substep_local)

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

    def _sanitize_input_tensor(self, tensor, dtype, unbatched_ndim=1):
        _tensor = torch.as_tensor(tensor, dtype=dtype, device=gs.device)

        if _tensor.ndim < unbatched_ndim + 1:
            _tensor = _tensor.repeat((self._sim._B, *((1,) * max(1, _tensor.ndim))))
            if self._sim._B > 1:
                gs.logger.debug(ALLOCATE_TENSOR_WARNING)
        else:
            _tensor = _tensor.contiguous()
            if _tensor is not tensor:
                gs.logger.debug(ALLOCATE_TENSOR_WARNING)

            if len(_tensor) != self._sim._B:
                gs.raise_exception("Input tensor batch size must match the number of environments.")

        if _tensor.ndim != unbatched_ndim + 1:
            gs.raise_exception(f"Input tensor ndim is {_tensor.ndim}, should be {unbatched_ndim + 1}.")

        return _tensor

    def _sanitize_input_verts_idx(self, verts_idx_local):
        verts_idx = self._sanitize_input_tensor(verts_idx_local, dtype=gs.tc_int, unbatched_ndim=1) + self._v_start
        assert ((verts_idx >= 0) & (verts_idx < self._solver.n_vertices)).all(), "Vertex indices out of bounds."
        return verts_idx

    def _sanitize_input_poss(self, poss):
        poss = self._sanitize_input_tensor(poss, dtype=gs.tc_float, unbatched_ndim=2)
        assert poss.ndim == 3 and poss.shape[2] == 3, "Position tensor must have shape (B, num_verts, 3)."
        return poss

    def set_vertex_constraints(
        self, verts_idx, target_poss=None, link=None, is_soft_constraint=False, stiffness=0.0, envs_idx=None
    ):
        """
        Set vertex constraints for specified vertices.

        Parameters
        ----------
            verts_idx : array_like
                List of vertex indices to constrain.
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
        if self._solver._use_implicit_solver:
            if not self._solver._enable_vertex_constraints:
                gs.logger.warning("Ignoring vertex constraint; FEM implicit solver needs to enable vertex constraints.")
                return

        if not self._solver._constraints_initialized:
            self._solver.init_constraints()

        envs_idx = self._scene._sanitize_envs_idx(envs_idx)
        verts_idx = self._sanitize_input_verts_idx(verts_idx)

        if target_poss is None:
            target_poss = torch.zeros(
                (verts_idx.shape[0], verts_idx.shape[1], 3), dtype=gs.tc_float, device=gs.device, requires_grad=False
            )
            self._kernel_get_verts_pos(self._sim.cur_substep_local, target_poss, verts_idx)
        target_poss = self._sanitize_input_poss(target_poss)

        assert (
            len(envs_idx) == len(target_poss) == len(verts_idx)
        ), "First dimension should match number of environments."
        assert target_poss.shape[1] == verts_idx.shape[1], "Target position should be provided for each vertex."

        if link is None:
            link_init_pos = torch.zeros((self._sim._B, 3), dtype=gs.tc_float, device=gs.device)
            link_init_quat = torch.zeros((self._sim._B, 4), dtype=gs.tc_float, device=gs.device)
            link_idx = -1
        else:
            assert isinstance(link, RigidLink), "Only RigidLink is supported for vertex constraints."
            link_init_pos = self._sanitize_input_tensor(link.get_pos(), dtype=gs.tc_float)
            link_init_quat = self._sanitize_input_tensor(link.get_quat(), dtype=gs.tc_float)
            link_idx = link.idx

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

    def update_constraint_targets(self, verts_idx, target_poss, envs_idx=None):
        """Update target positions for existing constraints."""
        if not self._solver._constraints_initialized:
            gs.logger.warning("Ignoring update_constraint_targets; constraints have not been initialized.")
            return

        envs_idx = self._scene._sanitize_envs_idx(envs_idx)
        verts_idx = self._sanitize_input_verts_idx(verts_idx)
        target_poss = self._sanitize_input_poss(target_poss)
        assert target_poss.shape[1] == verts_idx.shape[1], "Target position should be provided for each vertex."

        self._solver._kernel_update_constraint_targets(verts_idx, target_poss, envs_idx)

    def remove_vertex_constraints(self, verts_idx=None, envs_idx=None):
        """Remove constraints from specified vertices, or all if None."""
        if not self._solver._constraints_initialized:
            gs.logger.warning("Ignoring remove_vertex_constraints; constraints have not been initialized.")
            return

        if verts_idx is None:
            self._solver.vertex_constraints.is_constrained.fill(0)
        else:
            verts_idx = self._sanitize_input_verts_idx(verts_idx)
            envs_idx = self._scene._sanitize_envs_idx(envs_idx)
            self._solver._kernel_remove_specific_constraints(verts_idx, envs_idx)

    @ti.kernel
    def _kernel_get_verts_pos(self, f: ti.i32, pos: ti.types.ndarray(), verts_idx: ti.types.ndarray()):
        # get current position of vertices
        for i_v, i_b in ti.ndrange(verts_idx.shape[0], verts_idx.shape[1]):
            i_global = verts_idx[i_v, i_b] + self.v_start
            for j in ti.static(range(3)):
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
        self._solver._kernel_get_el2v(
            element_el_start=self._el_start,
            n_elements=self.n_elements,
            el2v=el2v,
        )

        return el2v

    @ti.kernel
    def get_frame(self, f: ti.i32, pos: ti.types.ndarray(), vel: ti.types.ndarray(), active: ti.types.ndarray()):
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

        for i_v, i_b in ti.ndrange(self.n_vertices, self._sim._B):
            i_global = i_v + self.v_start
            for j in ti.static(range(3)):
                pos[i_b, i_v, j] = self._solver.elements_v[f, i_global, i_b].pos[j]
                vel[i_b, i_v, j] = self._solver.elements_v[f, i_global, i_b].vel[j]

        for i_v, i_b in ti.ndrange(self.n_elements, self._sim._B):
            i_global = i_v + self.el_start
            active[i_b, i_v] = self._solver.elements_el_ng[f, i_global, i_b].active

    @ti.kernel
    def clear_grad(self, f: ti.i32):
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
        for i_v, i_b in ti.ndrange(self.n_vertices, self._sim._B):
            i_global = i_v + self.v_start
            self._solver.elements_v.grad[f, i_global, i_b].pos = 0
            self._solver.elements_v.grad[f, i_global, i_b].vel = 0

        for i_v, i_b in ti.ndrange(self.n_elements, self._sim._B):
            i_global = i_v + self.el_start
            self._solver.elements_el.grad[f, i_global, i_b].actu = 0

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
    def tet_cfg(self):
        """Configuration of tetrahedralization."""
        tet_cfg = mu.generate_tetgen_config_from_morph(self.morph)
        return tet_cfg
