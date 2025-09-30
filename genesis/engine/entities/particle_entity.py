import functools

import numpy as np
import gstaichi as ti
import torch
import trimesh
from scipy.spatial import KDTree

import genesis as gs
import genesis.utils.geom as gu
import genesis.utils.mesh as mu
import genesis.utils.particle as pu
from genesis.utils.misc import ALLOCATE_TENSOR_WARNING
from genesis.engine.states.cache import QueriedStates
from genesis.utils.misc import to_gs_tensor

from .base_entity import Entity


def assert_active(method):
    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        if not self.active:
            gs.raise_exception(f"'{self.__class__.__name__}' is not active. Call `entity.activate()` first.")
        return method(self, *args, **kwargs)

    return wrapper


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

        # Note that this attribute must only be used in forward pass
        self.active = False

    def _sanitize_particles_idx_local(self, particles_idx_local, envs_idx=None, *, unsafe=False):
        if particles_idx_local is None:
            particles_idx_local = range(self._n_particles)
        elif isinstance(particles_idx_local, slice):
            particles_idx_local = range(
                particles_idx_local.start or 0,
                particles_idx_local.stop if particles_idx_local.stop is not None else self._n_particles,
                particles_idx_local.step or 1,
            )
        elif isinstance(particles_idx_local, (int, np.integer)):
            particles_idx_local = [particles_idx_local]

        if unsafe:
            return particles_idx_local

        _particles_idx_local = torch.as_tensor(particles_idx_local, dtype=gs.tc_int, device=gs.device).contiguous()
        if _particles_idx_local is not particles_idx_local or (envs_idx is not None and _particles_idx_local.ndim < 2):
            gs.logger.debug(ALLOCATE_TENSOR_WARNING)
        if envs_idx is not None:
            if _particles_idx_local.ndim < 2:
                _particles_idx_local = _particles_idx_local.reshape((1, -1)).tile((len(envs_idx), 1))
                if _particles_idx_local.ndim != 2:
                    gs.raise_exception("Expecting 0D, 1D or 2D tensor for `particles_idx_local`.")
        else:
            _particles_idx_local = torch.atleast_1d(_particles_idx_local)
            if _particles_idx_local.ndim != 1:
                gs.raise_exception("Expecting 0D or 1D tensor for `particles_idx_local`.")
        if not ((0 <= _particles_idx_local).all() or (_particles_idx_local < input_size).all()):
            gs.raise_exception("Elements of `particles_idx_local' are out-of-range.")

        return _particles_idx_local

    def _sanitize_particles_tensor(
        self, element_shape, dtype, tensor, particles_idx=None, envs_idx=None, *, batched=True
    ):
        n_particles = particles_idx.shape[-1] if particles_idx is not None else self._n_particles
        if batched:
            batch_shape = (len(envs_idx), n_particles)
        else:
            batch_shape = (n_particles,)
        tensor_shape = (*batch_shape, *element_shape)

        tensor = to_gs_tensor(tensor, dtype)
        if tensor.ndim == len(element_shape):
            tensor = tensor.reshape((*((1,) * len(batch_shape)), *element_shape)).expand(tensor_shape)
        elif batched and tensor.ndim == len(tensor_shape) - 1:
            for i in range(len(batch_shape)):
                if len(tensor) != tensor_shape[i]:
                    tensor = tensor.reshape((*tensor_shape[:i], 1, *tensor_shape[(i + 1) :])).expand(tensor_shape)
                    break
        if tensor.shape != tensor_shape:
            gs.raise_exception(f"Invalid tensor shape {tensor.shape} (expected {tensor_shape}).")

        _tensor = tensor.contiguous()
        if _tensor is not tensor:
            gs.logger.debug(ALLOCATE_TENSOR_WARNING)
        return _tensor

    def init_sampler(self):
        """
        Initialize the particle sampling strategy based on the material's sampler field.

        Raises
        ------
        GenesisException
            If the specified sampler is not supported or incorrectly formatted.
        """
        self.sampler = self._material.sampler

        match self.sampler.split("-"):
            case ["regular"]:
                pass
            case ["random"]:
                pass
            case ["pbs"]:
                # using default sdf_res=32
                self.sampler += "-32"
            case ["pbs", num] if num.isnumeric():
                pass
            case _:
                gs.raise_exception(
                    "Only one of the following samplers is supported: [`regular`, `random`, `pbs`, `pbs-sdf_res`]. "
                    f"Got: {self.sampler}."
                )

    def init_tgt_keys(self):
        """
        Initialize the list of keys used for controlling entity state (position, velocity, activation).
        """
        self._tgt_keys = ("pos", "vel", "act")

    def _add_to_solver(self):
        # For nowhere entity, it will be activated when particles are later added.
        # Note that 'self.active' is not informative for nowhere entity, and such entity is not supported in
        # differentiable mode.
        self.active = not isinstance(self._morph, gs.options.morphs.Nowhere)

        self._add_particles_to_solver()
        if self._need_skinning:
            self._add_vverts_to_solver()

    def _add_particles_to_solver(self):
        raise NotImplementedError

    def _add_vverts_to_solver(self):
        # Compute supports for rendering vverts using neighboring particles
        kdtree = KDTree(self._particles)
        _, support_idxs = kdtree.query(self._vverts, k=self.solver._n_vvert_supports)
        support_idxs = support_idxs.astype(gs.np_int)
        support_idxs = np.clip(support_idxs, 0, len(self._particles) - 1)
        all_ps = self._particles[support_idxs]
        Ps = all_ps[:, :-1].swapaxes(-2, -1) - np.expand_dims(all_ps[:, -1], axis=-1)
        P_invs = np.linalg.pinv(Ps)

        self._kernel_add_vverts_to_solver(
            vverts=self._vverts,
            particles=self._particles,
            P_invs=P_invs,
            support_idxs_local=support_idxs,
        )

    @ti.kernel
    def _kernel_add_vverts_to_solver(
        self,
        vverts: ti.types.ndarray(element_dim=1),
        particles: ti.types.ndarray(element_dim=1),
        P_invs: ti.types.ndarray(element_dim=2),
        support_idxs_local: ti.types.ndarray(),
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
                    particles_i = pu.trimesh_to_particles_pbs(
                        mesh=mesh_i,
                        p_size=self._particle_size,
                        sampler=sampler,
                    )
                else:
                    particles_i = pu.trimesh_to_particles_simple(
                        mesh=mesh_i,
                        p_size=self._particle_size,
                        sampler=sampler,
                    )

                particles_i += np.asarray(morph_i.pos, dtype=gs.np_float)
                particles.append(particles_i)
        elif isinstance(self._morph, (gs.options.morphs.Primitive, gs.options.morphs.Mesh)):
            particles = self._vmesh.particlize(self._particle_size, self.sampler)
            particles = particles.astype(gs.np_float, order="C", copy=False)
        elif isinstance(self._morph, gs.options.morphs.Nowhere):
            particles = pu.nowhere_particles(self._morph.n_particles)
        else:
            gs.raise_exception(f"Unsupported morph: {self._morph}.")

        if not isinstance(self._morph, gs.options.morphs.MeshSet) and particles.size == 0:
            gs.raise_exception("Entity has zero particles.")

        if isinstance(self._morph, gs.options.morphs.Nowhere):
            self._vverts = np.zeros((0, 3), dtype=gs.np_float)
            self._vfaces = np.zeros((0, 3), dtype=gs.np_float)
            origin = gu.nowhere()
        elif isinstance(self._morph, gs.options.morphs.MeshSet):
            for i in range(len(self._morph.files)):
                # Note that particles are already transformed at this point
                pos_i = np.asarray(self._morph.poss[i], dtype=gs.np_float)
                euler_i = np.asarray(self._morph.eulers[i], dtype=gs.np_float)
                quat_i = gs.utils.geom.xyz_to_quat(euler_i, rpy=True, degrees=True)
                self._vmesh[i].apply_transform(gu.trans_quat_to_T(pos_i, quat_i))

            self.mesh_set_group_ids = np.concatenate(
                [np.full((len(v),), fill_value=i, dtype=gs.np_int) for i, v in enumerate(particles)]
            )
            particles = np.concatenate(particles, dtype=gs.np_float)
            if not self._solver.boundary.is_inside(particles):
                gs.raise_exception(
                    "Entity has particles outside solver boundary. Note that for MPMSolver, boundary is slightly "
                    "tighter than the specified domain due to safety padding.\n\nCurrent boundary:\n"
                    f"{self._solver.boundary}\n\nEntity to be added:\nmin: {particles.min(0)}\nmax: {particles.max(0)}\n"
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
                self._vverts = np.zeros((0, 3), dtype=gs.np_float)
                self._vfaces = np.zeros((0, 3), dtype=gs.np_float)
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
                self._vverts = np.zeros((0, 3), dtype=gs.np_float)
                self._vfaces = np.zeros((0, 3), dtype=gs.np_float)
            origin = np.asarray(self._morph.pos, dtype=gs.np_float)

        self._particles = np.asarray(particles, dtype=gs.np_float, order="C")
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
            self._ckpt[ckpt_name] = {"_tgt_buffer": dict()}

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

    def _reset_grad(self):
        """
        Clear gradients.
        """
        raise NotImplementedError

    def add_grad_from_state(state):
        raise NotImplementedError

    def process_input(self, in_backward: bool = False):
        """
        Push position, velocity, and activation target states into the simulator.

        Parameters
        ----------
        in_backward : bool, default=False
            Whether the simulation is in the backward gradient pass.
        """
        if in_backward:
            # use negative index because buffer length might not be full
            index = self._sim.cur_step_local - self._sim._steps_local
            for key in self._tgt_keys:
                self._tgt[key] = self._tgt_buffer[key][index]
        else:
            for key in self._tgt_keys:
                self._tgt_buffer[key].append(self._tgt[key])

        if any(self._tgt[key] is not None for key in self._tgt_keys):
            particles_idx_local = self._sanitize_particles_idx_local(None, self._scene._envs_idx)

        # Note that setting positions resets velocities to zero, so it must be done BEFORE setting velocities
        if self._tgt["pos"] is not None:
            self._tgt["pos"].assert_contiguous()
            self._tgt["pos"].assert_sceneless()
            self.set_particles_pos(self._tgt["pos"], particles_idx_local)

        if self._tgt["vel"] is not None:
            self._tgt["vel"].assert_contiguous()
            self._tgt["vel"].assert_sceneless()
            self.set_particles_vel(self._tgt["vel"], particles_idx_local)

        if self._tgt["act"] is not None:
            self._tgt["act"].assert_contiguous()
            self._tgt["act"].assert_sceneless()
            act_values = torch.tensor((gs.ACTIVE, gs.INACTIVE), dtype=gs.tc_int, device=gs.device)
            assert torch.isin(self._tgt["act"], act_values).all()
            self.set_particles_active(self._tgt["act"], particles_idx_local)

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
        _tgt_pos = self._tgt_buffer["pos"].pop()
        if _tgt_pos is not None and _tgt_pos.requires_grad:
            _tgt_pos._backward_from_ti(self._set_particles_pos_grad)

        _tgt_vel = self._tgt_buffer["vel"].pop()
        if _tgt_vel is not None and _tgt_vel.requires_grad:
            _tgt_vel._backward_from_ti(self._set_particles_vel_grad)

        # Manually zero the grad since manually setting state breaks gradient flow
        if _tgt_vel is not None or _tgt_pos is not None:
            self._reset_grad()

    def collect_output_grads(self):
        """
        Collect gradients from external queried states.
        """
        if self._sim.cur_step_global in self._queried_states:
            for state in self._queried_states[self._sim.cur_step_global]:
                self.add_grad_from_state(state)

    # ------------------------------------------------------------------------------------
    # ---------------------------------- io & control ------------------------------------
    # ------------------------------------------------------------------------------------

    @assert_active
    def _set_particles_target_state(self, key, name, element_shape, dtype, tensor, envs_idx=None, *, unsafe=False):
        if self.sim.requires_grad and self.sim.cur_t > 0.0:
            gs.logger.warning(
                f"Manually setting particle '{name}'. This is not recommended because it breaks gradient flow."
            )
        envs_idx = self._scene._sanitize_envs_idx(envs_idx, unsafe=unsafe)
        self._tgt[key] = self._sanitize_particles_tensor(element_shape, dtype, tensor, envs_idx=envs_idx)

    def set_position(self, value, envs_idx=None, *, unsafe=False):
        """
        Set the position of all the particles individually, or the center of masswrt the initial configuration of the
        particles as a whole.

        Parameters
        ----------
        value : array_like, shape ([M,] [n_particles,] 3)
            Tensor of particle positions.
        envs_idx : None | int | array_like, shape (M,), optional
            The indices of the environments to set. If None, all environments will be considered. Defaults to None.
        """
        # Determine whether the position of all the particles has been specified, or only the center of mass
        poss = to_gs_tensor(value, dtype=gs.tc_float)
        if poss.ndim == 1 or (poss.ndim == 2 and poss.shape[0] != self._n_particles):
            poss = self._init_particles_offset + poss.unsqueeze(-2)

        self._set_particles_target_state("pos", "position", (3,), gs.tc_float, poss, envs_idx, unsafe=unsafe)

    @gs.assert_built
    def set_particles_pos(self, poss, particles_idx_local=None, envs_idx=None, *, unsafe=False):
        """
        Set the position of some particles.

        Parameters
        ----------
        poss: torch.Tensor, shape (M, N, 3)
            Target position of each particle.
        particles_idx_local : torch.Tensor, shape (M, N)
            Index of the particles relative to this entity. If None, all particles will be considered. Defaults to None.
        envs_idx : torch.Tensor, shape (M,)
            The indices of the environments to set. If None, all environments will be considered. Defaults to None.
        """
        raise NotImplementedError

    @gs.assert_built
    def _set_particles_pos_grad(self, poss_grad):
        """
        Set gradients for particle positions at a given substep.

        Parameters
        ----------
        poss_grad : torch.Tensor, shape (M, n_particles, 3)
            The gradients for particle positions.
        """
        raise NotImplementedError

    @gs.assert_built
    def get_particles_pos(self, envs_idx=None, *, unsafe=False):
        """
        Retrieve current particle positions from the solver.

        Parameters
        ----------
        envs_idx : None | int | array_like, shape (M,), optional
            The indices of the environments to set. If None, all environments will be considered. Defaults to None.

        Returns
        -------
        poss : torch.Tensor, shape (M, n_particles, 3)
            Tensor of particle positions.
        """
        raise NotImplementedError

    def set_velocity(self, vels, envs_idx=None, *, unsafe=False):
        """
        Set the velocity of all the particles individually.

        Parameters
        ----------
        vels : array_like, shape ([M,] [n_particles,] 3)
            Tensor of particle velocities.
        envs_idx : None | array_like, optional
            The indices of the environments to set. If None, all environments will be considered. Defaults to None.
        """
        self._set_particles_target_state("vel", "velocity", (3,), gs.tc_float, vels, envs_idx, unsafe=unsafe)

    @gs.assert_built
    def set_particles_vel(self, vels, particles_idx_local=None, envs_idx=None, *, unsafe=False):
        """
        Set the velocity of some particles.

        Parameters
        ----------
        vels: torch.Tensor, shape (M, N, 3)
            Target velocity of each particle.
        particles_idx_local : torch.Tensor, shape (M, N)
            Index of the particles relative to this entity. If None, all particles will be considered. Defaults to None.
        envs_idx : torch.Tensor, shape (M,)
            The indices of the environments to set. If None, all environments will be considered. Defaults to None.
        """
        raise NotImplementedError

    @gs.assert_built
    def _set_particles_vel_grad(self, vels_grad):
        """
        Set gradients for particle velocities at a specific frame.

        Parameters
        ----------
        vels_grad : torch.Tensor, shape (M, n_particles, 3)
            The gradients for particle velocities.
        """
        raise NotImplementedError

    @gs.assert_built
    def get_particles_vel(self, envs_idx=None, *, unsafe=False):
        """
        Retrieve current particle velocities from the solver.

        Parameters
        ----------
        envs_idx : None | int | array_like, shape (M,), optional
            The indices of the environments to set. If None, all environments will be considered. Defaults to None.

        Returns
        -------
        poss : torch.Tensor
            Tensor of particle velocities, shape (M, n_particles, 3).
        """
        raise NotImplementedError

    def set_active(self, actives, envs_idx=None, *, unsafe=False):
        """
        Set the activeness state of all the particles individually.

        Parameters
        ----------
        actives : int, array_like, shape ([M,] [n_particles,])
            Tensor of particle activeness boolean flags.
        envs_idx : None | int | array_like, shape (M,), optional
            The indices of the environments to set. If None, all environments will be considered. Defaults to None.
        """
        self._set_particles_target_state("act", "activeness", (), gs.tc_bool, actives, envs_idx, unsafe=unsafe)
        self.active = bool((self._tgt["act"] == gs.ACTIVE).any())

    def activate(self):
        """
        Activate all particles of the entity in simulation, making them eligible for updates.
        """
        gs.logger.info(f"{self.__class__.__name__} <{self._uid}> activated.")
        self.set_active(gs.ACTIVE)

    def deactivate(self):
        """
        Deactivate all particles of the entity in simulation, stopping them from receiving for updates.
        """
        gs.logger.info(f"{self.__class__.__name__} <{self._uid}> deactivated.")
        self.set_active(gs.INACTIVE)

    @gs.assert_built
    def set_particles_active(self, actives, particles_idx_local=None, envs_idx=None, *, unsafe=False):
        """
        Set the velocity of some particles.

        Parameters
        ----------
        actives: torch.Tensor, shape (M, N, 3)
            Activeness boolean flags for each particle.
        particles_idx_local : torch.Tensor, shape (M, N)
            Index of the particles relative to this entity. If None, all particles will be considered. Defaults to None.
        envs_idx : torch.Tensor, shape (M,)
            The indices of the environments to set. If None, all environments will be considered. Defaults to None.
        """
        raise NotImplementedError

    def get_particles_active(self, envs_idx=None, *, unsafe=False):
        """
        Retrieve current particle activeness boolean flags from the solver.

        Parameters
        ----------
        envs_idx : None | int | array_like, shape (M,), optional
            The indices of the environments to set. If None, all environments will be considered. Defaults to None.

        Returns
        -------
        poss : torch.Tensor, shape (M, n_particles, 3)
            Tensor of particle activeness boolean flags.
        """
        raise NotImplementedError

    def get_mass(self, envs_idx=None, *, unsafe=False):
        """
        Return the total mass of the entity.

        Parameters
        ----------
        envs_idx : None | int | array_like, shape (M,), optional
            The indices of the environments to set. If None, all environments will be considered. Defaults to None.

        Returns
        -------
        mass : torch.Tensor, shape (M,)
            The computed total mass.
        """
        envs_idx = self._scene._sanitize_envs_idx(envs_idx, unsafe=unsafe)
        mass = torch.empty((len(envs_idx),), dtype=gs.tc_float, device=gs.device)
        self.solver._kernel_get_mass(mass, envs_idx)
        return mass

    # ------------------------------------------------------------------------------------
    # -------------------------------------- utils ---------------------------------------
    # ------------------------------------------------------------------------------------

    @gs.assert_built
    def find_closest_particle(self, pos, envs_idx=None, *, unsafe=False):
        """
        Find the index of the particle closest to a given position.

        Parameters
        ----------
        pos : array_like, shape ([M,] 3)
            The target position to compare against.
        envs_idx : None | int | array_like, shape (M,), optional
            The indices of the environments to set. If None, all environments will be considered. Defaults to None.

        Returns
        -------
        closest_idx : torch.Tensor, shape (M,)
            The index of the closest particle.
        """
        pos = to_gs_tensor(pos, dtype=gs.tc_float)
        if pos.ndim == 1:
            pos = pos.reshape((1, 3)).expand((self._scene._B, 3))
        if pos.shape != (self._scene._B, 3):
            gs.raise_exception(f"Invalid tensor shape {pos.shape} (expected {pos}).")
        _pos = pos.contiguous()
        if _pos is not pos:
            gs.logger.debug(ALLOCATE_TENSOR_WARNING)

        cur_particles = self.get_particles_pos(envs_idx)
        distances = torch.linalg.norm(cur_particles - pos.unsqueeze(1), dim=-1)
        closest_idx = torch.argmin(distances, dim=-1).to(dtype=gs.tc_int)
        if self._scene.n_envs == 0:
            closest_idx = closest_idx.squeeze(0)
        return closest_idx

    # ------------------------------------------------------------------------------------
    # ----------------------------------- properties -------------------------------------
    # ------------------------------------------------------------------------------------

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
