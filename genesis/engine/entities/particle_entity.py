import numpy as np
import gstaichi as ti
import torch
import trimesh
from scipy.spatial import KDTree

import genesis as gs
import genesis.utils.geom as gu
import genesis.utils.mesh as mu
import genesis.utils.particle as pu
from genesis.engine.states.cache import QueriedStates
from genesis.utils.misc import to_gs_tensor

from .base_entity import Entity


@ti.data_oriented
class ParticleEntity(Entity):
    """
    Base class for particle-based entity.

    Parameters
    ----------
    scene : Scene
        The scene object that this entity belongs to.
    solver : Solver
        The physics solver responsible for simulating the entity's particles.
    material : Material
        The material definition, including sampling strategy and physical properties.
    morph : Morph
        Geometry or volumetric shape used for sampling particles (e.g., mesh, primitive).
    surface : Surface
        Surface material or texture information associated with the entity.
    particle_size : float
        Size of each particle, determining resolution and sampling density.
    idx : int
        Index of this entity in the simulation.
    particle_start : int
        Global index offset for this entity’s particles in the solver.
    vvert_start : int, optional
        Global index offset for vertex-based rendering, used for skinning.
    vface_start : int, optional
        Global index offset for visual faces, used for skinning.
    need_skinning : bool, default=True
        Whether to enable skinning for rendering this entity's mesh.
    """

    def __init__(
        self,
        scene,
        solver,
        material,
        morph,
        surface,
        particle_size,
        idx,
        particle_start,
        vvert_start=None,
        vface_start=None,
        need_skinning=True,
    ):

        super().__init__(idx, scene, morph, solver, material, surface)

        self._particle_size = particle_size
        self._particle_start = particle_start

        self._need_skinning = need_skinning
        if need_skinning:
            self._vvert_start = vvert_start
            self._vface_start = vface_start
        else:
            self._vvert_start = -1
            self._vface_start = -1

        # visual mesh
        if isinstance(self._morph, gs.options.morphs.MeshSet):
            self._vmesh = gs.Mesh.from_morph_surface(self.morph, self.surface)
            self._surface = self._vmesh[0].surface

        elif isinstance(self._morph, (gs.options.morphs.Primitive, gs.options.morphs.Mesh)):
            self._vmesh = gs.Mesh.from_morph_surface(self.morph, self.surface)
            if isinstance(self._vmesh, list):
                if len(self._vmesh) > 1:
                    gs.raise_exception("Mesh file with multiple sub-meshes are not supported.")
                else:
                    self._vmesh = self._vmesh[0]
            self._surface = self._vmesh.surface

        else:
            surface.update_texture()
            self._vmesh = None

        self.sample()

        self.init_tgt_vars()
        self.init_ckpt()
        self._queried_states = QueriedStates()

        self.active = False  # This attribute is only used in forward pass. It should NOT be used during backward pass.

    def init_sampler(self):
        """
        Initialize the particle sampling strategy based on the material's sampler field.

        Raises
        ------
        GenesisException
            If the specified sampler is not supported or incorrectly formatted.
        """
        self.sampler = self._material.sampler

        valid = True
        if self.sampler == "regular":
            pass
        elif self.sampler == "random":
            pass
        elif "pbs" in self.sampler:
            splits = self.sampler.split("-")
            if len(splits) == 1:  # using default sdf_res=32
                self.sampler += "-32"
            elif len(splits) == 2 and splits[0] == "pbs" and splits[1].isnumeric():
                pass
            else:
                valid = False
        else:
            valid = False

        if not valid:
            gs.raise_exception(
                f"Only one of the following samplers is supported: [`regular`, `random`, `pbs`, `pbs-sdf_res`]. Got: {self.sampler}."
            )

    def init_tgt_keys(self):
        """
        Initialize the list of keys used for controlling entity state (position, velocity, activation).
        """
        self._tgt_keys = ["vel", "pos", "act"]

    def _add_to_solver(self):
        if isinstance(self._morph, gs.options.morphs.Nowhere):
            # for nowhere entity, it will be activated when particles are added later.
            # Note that self.active is not informative for nowhere entity, and such entity is not supported in differentiable mode.
            self.active = False
        else:
            self.active = True

        self._add_to_solver_()

        if self._need_skinning:
            self._add_vverts_to_solver()

        # TODO: make more elegant when merge with PBD
        if hasattr(self, "set_muscle_group") and hasattr(self, "mesh_set_group_ids"):
            self.set_muscle_group(self.mesh_set_group_ids)

    def _add_vverts_to_solver(self):
        # compute supports for rendering vverts using neighboring particles
        # need to put this kernel here because type hint for P_invs requires access to self.solver._n_vvert_supports
        @ti.kernel
        def _kernel_add_vverts_to_solver(
            vverts: ti.types.ndarray(dtype=ti.math.vec3),
            particles: ti.types.ndarray(dtype=ti.math.vec3),
            P_invs: ti.types.ndarray(dtype=ti.types.matrix(self.solver._n_vvert_supports - 1, 3, gs.ti_float)),
            support_idxs_local: ti.types.ndarray(dtype=ti.i32),
        ):
            for i_vv_ in range(self.n_vverts):
                i_vv = i_vv_ + self._vvert_start
                v = vverts[i_vv_]
                P_inv = P_invs[i_vv_]
                b = P_inv @ (v - particles[support_idxs_local[i_vv_, self.solver._n_vvert_supports - 1]])
                for j in range(self.solver._n_vvert_supports - 1):
                    self.solver.vverts_info.support_weights[i_vv][j] = b[j]
                self.solver.vverts_info.support_weights[i_vv][self.solver._n_vvert_supports - 1] = 1.0 - b.sum()
                for j in range(self.solver._n_vvert_supports):
                    self.solver.vverts_info.support_idxs[i_vv][j] = support_idxs_local[i_vv_, j] + self._particle_start

        kdtree = KDTree(self._particles)
        _, support_idxs = kdtree.query(self._vverts, k=self.solver._n_vvert_supports)
        support_idxs = np.clip(support_idxs.astype(gs.np_int, copy=False), 0, len(self._particles) - 1)
        all_ps = self._particles[support_idxs]
        Ps = np.stack(
            [all_ps[:, i, :] for i in range(self.solver._n_vvert_supports - 1)], axis=2, dtype=gs.np_float
        ) - np.expand_dims(all_ps[:, -1, :], axis=2)

        _kernel_add_vverts_to_solver(
            vverts=self._vverts,
            particles=self._particles,
            P_invs=np.linalg.pinv(Ps),
            support_idxs_local=support_idxs,
        )

    def sample(self):
        """
        Sample particles from the morph based on its type and the specified sampler.

        Raises
        ------
        GenesisException
            If no particles are sampled, or sampled particles lie outside the solver's domain.
        """
        self.init_sampler()

        if isinstance(self._morph, gs.options.morphs.MeshSet):
            particles = []
            for i, file in enumerate(self._morph.files):
                morph_i = self._morph.model_copy()
                morph_i.file = file
                morph_i.pos = morph_i.poss[i]
                morph_i.euler = morph_i.eulers[i]
                mesh_i = morph_i.file.copy()
                mesh_i.vertices = mesh_i.vertices * morph_i.scale

                sampler = self.sampler
                if "pbs" in sampler:
                    try:
                        particles_i = pu.trimesh_to_particles_pbs(
                            mesh=mesh_i,
                            p_size=self._particle_size,
                            sampler=sampler,
                        )
                    except gs.GenesisException:
                        sampler = "random"
                if "pbs" not in sampler:
                    particles_i = pu.trimesh_to_particles_simple(
                        mesh=mesh_i,
                        p_size=self._particle_size,
                        sampler=sampler,
                    )

                particles_i += np.array(morph_i.pos)
                particles.append(particles_i)
        elif isinstance(self._morph, (gs.options.morphs.Primitive, gs.options.morphs.Mesh)):
            particles = self._vmesh.particlize(self._particle_size, self.sampler)
            particles = particles.astype(gs.np_float, order="C", copy=False)
        elif isinstance(self._morph, gs.options.morphs.Nowhere):
            particles = pu.nowhere_particles(self._morph.n_particles)
        else:
            gs.raise_exception(f"Unsupported morph: {self._morph}.")

        if not isinstance(self._morph, gs.options.morphs.MeshSet) and particles.shape[0] == 0:
            gs.raise_exception("Entity has zero particles.")

        if isinstance(self._morph, gs.options.morphs.Nowhere):
            origin = gu.nowhere()
            self._vverts = np.array([], dtype=gs.np_float)
            self._vfaces = np.array([], dtype=gs.np_float)
        elif isinstance(self._morph, gs.options.morphs.MeshSet):
            for i in range(len(self._morph.files)):
                pos_i = np.asarray(self._morph.poss[i], dtype=gs.np_float)
                euler_i = np.asarray(self._morph.eulers[i], dtype=gs.np_float)
                quat_i = gs.utils.geom.xyz_to_quat(euler_i, rpy=True, degrees=True)
                self._vmesh[i].apply_transform(gu.trans_quat_to_T(pos_i, quat_i))

                # NOTE: particles are transformed already
                # particles[i] = gu.transform_by_trans_quat(particles[i], pos_i, quat_i)

            self.mesh_set_group_ids = np.concatenate(
                [np.full((len(v),), fill_value=i, dtype=gs.np_int) for i, v in enumerate(particles)]
            )
            particles = np.concatenate(particles, dtype=gs.np_float)
            if not self._solver.boundary.is_inside(particles):  # HACK no check
                gs.raise_exception(
                    f"Entity has particles outside solver boundary. Note that for MPMSolver, boundary is slightly tighter than the specified domain due to safety padding.\n\nCurrent boundary:\n{self._solver.boundary}\n\nEntity to be added:\nmin: {particles.min(0)}\nmax: {particles.max(0)}\n"
                )

            combined_verts, combined_faces = trimesh.util.append_faces(
                [v.verts for v in self._vmesh],
                [v.faces for v in self._vmesh],
            )
            combined_vert_normals = np.concatenate([v.trimesh.vertex_normals for v in self._vmesh])
            combined_tmesh = trimesh.Trimesh(
                vertices=combined_verts,
                faces=combined_faces,
                vertex_normals=combined_vert_normals,
            )
            self._vmesh = mu.trimesh_to_mesh(combined_tmesh, 1, self._surface)

            if self._need_skinning:
                self._vverts = np.asarray(self._vmesh.verts, dtype=gs.np_float)
                self._vfaces = np.asarray(self._vmesh.faces, dtype=gs.np_float)
            else:
                self._vverts = np.array([], dtype=gs.np_float)
                self._vfaces = np.array([], dtype=gs.np_float)
            origin = np.mean(self._morph.poss, dtype=gs.np_float)
        else:
            # transform vmesh
            pos = np.asarray(self._morph.pos, dtype=gs.np_float)
            quat = np.asarray(self._morph.quat, dtype=gs.np_float)
            self._vmesh.apply_transform(gu.trans_quat_to_T(pos, quat))
            # transform particles
            particles = gu.transform_by_trans_quat(
                particles,
                np.asarray(self._morph.pos, dtype=gs.np_float),
                np.asarray(self._morph.quat, dtype=gs.np_float),
            )

            if not self._solver.boundary.is_inside(particles):
                gs.raise_exception(
                    "Entity has particles outside solver boundary. Note that for MPMSolver, boundary is slightly "
                    "tighter than the specified domain due to safety padding.\n\n"
                    f"Current boundary:\n{self._solver.boundary}\n\nEntity to be added:\nmin: {particles.min(0)}\n"
                    f"max: {particles.max(0)}\n"
                )

            if self._need_skinning:
                self._vverts = np.asarray(self._vmesh.verts, dtype=gs.np_float)
                self._vfaces = np.asarray(self._vmesh.faces, dtype=gs.np_float)
            else:
                self._vverts = np.array([], dtype=gs.np_float)
                self._vfaces = np.array([], dtype=gs.np_float)
            origin = np.array(self._morph.pos, dtype=gs.np_float)

        self._particles = particles
        self._init_particles_offset = gs.tensor(self._particles) - gs.tensor(origin)
        self._n_particles = len(self._particles)

        gs.logger.info(f"Sampled ~~<{self._n_particles:,}>~~ particles.")

    def init_tgt_vars(self):
        """
        Initialize target buffers used for controlling and differentiating entity state over time.
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
        Initialize checkpoint storage for simulation state.
        """
        self._ckpt = dict()

    def save_ckpt(self, ckpt_name):
        """
        Save the current target state buffers to a checkpoint.

        Parameters
        ----------
        ckpt_name : str
            Name of the checkpoint to save.
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
        Restore target state buffers from a previously saved checkpoint.

        Parameters
        ----------
        ckpt_name : str
            Name of the checkpoint to load.
        """
        for key in self._tgt_keys:
            self._tgt_buffer[key] = list(self._ckpt[ckpt_name]["_tgt_buffer"][key])

    def reset_grad(self):
        """
        Clear target buffers and any externally queried simulation states.

        Used before backpropagation to reset gradients.
        """
        for key in self._tgt_keys:
            self._tgt_buffer[key].clear()
        self._queried_states.clear()

    def set_position(self, pos):
        """
        Set target particle position or center-of-mass (COM) position.

        Parameters
        ----------
        pos : torch.Tensor
            Desired position. Accepted shapes:
            - (3,) to reposition COM
            - (n_particles, 3)
            - (B, n_particles, 3)
            - (B, 3)

        Raises
        ------
        GenesisException
            If the shape of `pos` is not supported.
        """
        self._assert_active()
        if self.sim.requires_grad:
            gs.logger.warning(
                "Manually setting particle positions. This is not recommended and could break gradient flow. This also resets particle stress and velocities."
            )

        pos = to_gs_tensor(pos)

        is_valid = False
        if pos.ndim == 1:
            if pos.shape == (3,):
                pos = self._init_particles_offset + pos
                self._tgt["pos"] = pos.unsqueeze(0).tile((self._sim._B, 1, 1))
                is_valid = True
        elif pos.ndim == 2:
            if pos.shape == (self._n_particles, 3):
                self._tgt["pos"] = pos.unsqueeze(0).tile((self._sim._B, 1, 1))
                is_valid = True
            elif pos.shape == (self._sim._B, 3):
                pos = self._init_particles_offset.unsqueeze(0) + pos.unsqueeze(1)
                self._tgt["pos"] = pos
                is_valid = True
        elif pos.ndim == 3:
            if pos.shape == (self._sim._B, self._n_particles, 3):
                self._tgt["pos"] = pos
                is_valid = True
        if not is_valid:
            gs.raise_exception("Tensor shape not supported.")

    def set_velocity(self, vel):
        """
        Set target particle velocity.

        Parameters
        ----------
        vel : torch.Tensor
            Desired velocity. Accepted shapes:
            - (3,)
            - (n_particles, 3)
            - (B, n_particles, 3)
            - (B, 3)

        Raises
        ------
        GenesisException
            If the shape of `vel` is not supported.
        """

        self._assert_active()
        if self.sim.requires_grad:
            gs.logger.warning(
                "Manually setting particle velocities. This is not recommended and could break gradient flow."
            )

        vel = to_gs_tensor(vel)

        is_valid = False
        if vel.ndim == 1:
            if vel.shape == (3,):
                self._tgt["vel"] = vel.tile((self._sim._B, self._n_particles, 1))
                is_valid = True
        elif vel.ndim == 2:
            if vel.shape == (self._n_particles, 3):
                self._tgt["vel"] = vel.unsqueeze(0).tile((self._sim._B, 1, 1))
                is_valid = True
            elif vel.shape == (self._sim._B, 3):
                self._tgt["vel"] = vel.unsqueeze(1).tile((1, self._n_particles, 1))
                is_valid = True
        elif vel.ndim == 3:
            if vel.shape == (self._sim._B, self._n_particles, 3):
                self._tgt["vel"] = vel
                is_valid = True
        if not is_valid:
            gs.raise_exception("Tensor shape not supported.")

    def get_mass(self):
        """
        Return the total mass of the entity.

        Returns
        -------
        mass : float
            The computed total mass.
        """
        mass = np.zeros((1,), dtype=gs.np_float)
        self._kernel_get_mass(mass)
        return float(mass)

    def deactivate(self):
        """
        Deactivate the entity in simulation (will not receive updates).
        """
        gs.logger.info(f"{self.__class__.__name__} <{self._uid}> deactivated.")
        self._tgt["act"] = gs.INACTIVE
        self.active = False

    def activate(self):
        """
        Activate the entity in simulation (eligible for updates).
        """
        gs.logger.info(f"{self.__class__.__name__} <{self._uid}> activated.")
        self._tgt["act"] = gs.ACTIVE
        self.active = True

    def _assert_active(self):
        if not self.active:
            gs.raise_exception(f"{self.__class__.__name__} is inactive. Call `entity.activate()` first.")

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
        _tgt_vel = self._tgt_buffer["vel"].pop()
        _tgt_pos = self._tgt_buffer["pos"].pop()

        if _tgt_vel is not None and _tgt_vel.requires_grad:
            _tgt_vel._backward_from_ti(self.set_vel_grad, self._sim.cur_substep_local)

        if _tgt_pos is not None and _tgt_pos.requires_grad:
            _tgt_pos._backward_from_ti(self.set_pos_grad, self._sim.cur_substep_local)

        if _tgt_vel is not None or _tgt_pos is not None:
            # manually zero the grad since manually setting state breaks gradient flow
            self.clear_grad(self._sim.cur_substep_local)

    def collect_output_grads(self):
        """
        Collect gradients from external queried states.
        """
        if self._sim.cur_step_global in self._queried_states:
            # one step could have multiple states
            for state in self._queried_states[self._sim.cur_step_global]:
                self.add_grad_from_state(state)

    # ------------------------------------------------------------------------------------
    # ---------------------------------- io & control ------------------------------------
    # ------------------------------------------------------------------------------------

    @gs.assert_built
    def get_particles(self):
        """
        Get the current particle positions from the simulation.

        Returns
        -------
        particles : np.ndarray
            Particle positions with shape (B, n_particles, 3).
        """
        particles = np.empty((self._sim._B, self.n_particles, 3), dtype=gs.np_float)
        self._kernel_get_particles(particles)
        return particles

    @property
    def uid(self):
        """Unique identifier for the entity."""
        return self._uid

    @property
    def idx(self):
        """Index of the entity within the simulation."""
        return self._idx

    @property
    def morph(self):
        """Morphological representation used for particle sampling."""
        return self._morph

    @property
    def vmesh(self):
        """Visual mesh used for skinning and rendering."""
        return self._vmesh

    @property
    def n_vverts(self):
        """Number of visual mesh vertices."""
        return len(self._vverts)

    @property
    def n_vfaces(self):
        """Number of visual mesh faces."""
        return len(self._vfaces)

    @property
    def n_particles(self):
        """Number of particles"""
        return self._n_particles

    @property
    def particle_start(self):
        """Starting index of the entity's particles in the global buffer."""
        return self._particle_start

    @property
    def particle_end(self):
        """Ending index (exclusive) of the entity's particles."""
        return self._particle_start + self._n_particles

    @property
    def vvert_start(self):
        """Starting index for visual mesh vertices."""
        return self._vvert_start

    @property
    def vvert_end(self):
        """Ending index (exclusive) for visual mesh vertices."""
        return self._vvert_start + self.n_vverts

    @property
    def vface_start(self):
        """Starting index for visual mesh faces."""
        return self._vface_start

    @property
    def vface_end(self):
        """Ending index (exclusive) for visual mesh faces."""
        return self._vface_start + self.n_vfaces

    @property
    def particle_size(self):
        """Diameter of individual particles."""
        return self._particle_size

    @property
    def init_particles(self):
        """Initial sampled particle positions."""
        return self._particles

    @property
    def material(self):
        """Material of this entity."""
        return self._material

    @property
    def surface(self):
        """Surface for rendering."""
        return self._surface
