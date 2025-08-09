import numpy as np
import gstaichi as ti
import torch

import genesis as gs
from genesis.engine.states.entities import MPMEntityState
from genesis.utils.misc import to_gs_tensor

from .particle_entity import ParticleEntity


@ti.data_oriented
class MPMEntity(ParticleEntity):
    """
    MPM-based particle entity.

    Parameters
    ----------
    scene : Scene
        Scene object this entity belongs to.
    solver : Solver
        The solver responsible for simulating this entity.
    material : Material
        Material used to determine physical behavior (e.g., Snow, Sand).
    morph : Morph
        Shape description used for particle sampling.
    surface : Surface
        Surface or texture representation.
    particle_size : float
        Particle size for discretization.
    idx : int
        Unique index of the entity.
    particle_start : int
        Starting particle index.
    vvert_start : int
        Start index for visual vertices (unused if no skinning).
    vface_start : int
        Start index for visual faces (unused if no skinning).
    """

    def __init__(
        self, scene, solver, material, morph, surface, particle_size, idx, particle_start, vvert_start, vface_start
    ):
        if isinstance(material, (gs.materials.MPM.Liquid, gs.materials.MPM.Sand, gs.materials.MPM.Snow)):
            need_skinning = False
        else:
            need_skinning = True

        super().__init__(
            scene,
            solver,
            material,
            morph,
            surface,
            particle_size,
            idx,
            particle_start,
            vvert_start,
            vface_start,
            need_skinning=need_skinning,
        )

    def init_tgt_keys(self):
        """
        Initialize target keys used for buffer-based state tracking.

        Sets up the list of keys for target states, including velocity, position, activeness, and actuation.
        """
        self._tgt_keys = ["vel", "pos", "act", "actu"]

    def _add_to_solver_(self):
        self._solver._kernel_add_particles(
            self._sim.cur_substep_local,
            self.active,
            self._particle_start,
            self._n_particles,
            self._material.idx,
            self._material._default_Jp,
            self._material.rho,
            self._particles,
        )

    def set_pos(self, f, pos):
        """
        Set particle positions at a specific frame.

        Parameters
        ----------
        f : int
            The current substep index.
        pos : gs.Tensor
            A tensor of shape (n_envs, n_particles, 3) representing particle positions.
        """
        self.solver._kernel_set_particles_pos(f, self._particle_start, self._n_particles, pos)

    def set_pos_grad(self, f, pos_grad):
        """
        Set gradients for particle positions at a specific frame.

        Parameters
        ----------
        f : int
            The current substep index.
        pos_grad : gs.Tensor
            A tensor of shape (n_particles, 3) containing gradients for particle positions.
        """
        self.solver._kernel_set_particles_pos_grad(
            f,
            self._particle_start,
            self._n_particles,
            pos_grad,
        )

    def set_vel(self, f, vel):
        """
        Set particle velocities at a specific frame.

        Parameters
        ----------
        f : int
            The current substep index.
        vel : gs.Tensor
            A tensor of shape (n_particles, 3) representing particle velocities.
        """
        self.solver._kernel_set_particles_vel(
            f,
            self._particle_start,
            self._n_particles,
            vel,
        )

    def set_vel_grad(self, f, vel_grad):
        """
        Set gradients for particle velocities at a specific frame.

        Parameters
        ----------
        f : int
            The current substep index.
        vel_grad : gs.Tensor
            A tensor of shape (n_particles, 3) containing gradients for particle velocities.
        """
        self.solver._kernel_set_particles_vel_grad(
            f,
            self._particle_start,
            self._n_particles,
            vel_grad,
        )

    def set_actu(self, f, actu):
        """
        Set particle actuation values at a specific frame.

        Parameters
        ----------
        f : int
            The current substep index.
        actu : gs.Tensor
            A tensor of shape (n_particles,) or (n_groups,) or (B, n_groups) representing actuation values.
        """
        self.solver._kernel_set_particles_actu(
            f,
            self._particle_start,
            self._n_particles,
            self.material.n_groups,
            actu,
        )

    def set_actu_grad(self, f, actu_grad):
        """
        Set gradients for particle actuation values.

        Parameters
        ----------
        f : int
            The current substep index.
        actu_grad : gs.Tensor
            A tensor containing gradients for actuation inputs.
        """
        self.solver._kernel_set_particles_actu_grad(
            f,
            self._particle_start,
            self._n_particles,
            actu_grad,
        )

    def set_muscle_group(self, muscle_group):
        """
        Set the muscle group index for each particle.

        Parameters
        ----------
        muscle_group : gs.Tensor
            A tensor of shape (n_particles,) with integer group IDs.
        """
        self.solver._kernel_set_muscle_group(
            self._particle_start,
            self._n_particles,
            muscle_group,
        )

    def get_muscle_group(self):
        """
        Retrieve the muscle group index for each particle.

        Returns
        -------
        muscle_group : gs.Tensor
            A tensor of shape (n_particles,) containing the muscle group ID of each particle.
        """
        muscle_group = gs.zeros((self._n_particles,), dtype=int, requires_grad=False, scene=self._scene)
        self.solver._kernel_get_muscle_group(
            self._particle_start,
            self._n_particles,
            muscle_group,
        )

        return muscle_group

    def set_muscle_direction(self, muscle_direction):
        """
        Set the muscle fiber direction for each particle.

        Parameters
        ----------
        muscle_direction : gs.Tensor
            A tensor of shape (n_particles, 3) with unit vectors representing muscle directions.
        """
        self.solver._kernel_set_muscle_direction(
            self._particle_start,
            self._n_particles,
            muscle_direction,
        )

    def set_active(self, f, active):
        """
        Set the activeness state of all particles.

        Parameters
        ----------
        f : int
            The current substep index.
        active : int
            Value indicating whether particles are active (gs.ACTIVE) or inactive (gs.INACTIVE).
        """
        self.solver._kernel_set_particles_active(
            f,
            self._particle_start,
            self._n_particles,
            active,
        )

    def set_active_arr(self, f, active):
        """
        Set per-particle activeness using an array.

        Parameters
        ----------
        f : int
            The current substep index.
        active : gs.Tensor
            A tensor of shape (n_particles,) with activeness values.
        """
        self.solver._kernel_set_particles_active_arr(
            f,
            self._particle_start,
            self._n_particles,
            active,
        )

    def set_actuation(self, actu):
        """
        Set actuation values for muscle groups.

        Parameters
        ----------
        actu : torch.Tensor
            A tensor with shape matching the number of groups or the batch size and number of groups.
            Supported shapes: (), (n_groups,), (n_particles,), (B, n_groups), (B, n_particles), (B, self.n_particles, n_groups).
        """
        self._assert_active()

        actu = to_gs_tensor(actu)

        n_groups = getattr(self.material, "n_groups", 1)

        is_valid = False
        if actu.ndim == 0:
            self._tgt["actu"] = actu.tile((self._sim._B, self.n_particles, n_groups))
            is_valid = True
        elif actu.ndim == 1:
            if actu.shape == (n_groups,):
                self._tgt["actu"] = actu.reshape((1, 1, -1)).tile((self._sim._B, self.n_particles, 1))
                is_valid = True
            elif actu.shape == (n_particles,):
                self._tgt["actu"] = actu.reshape((1, -1, 1)).tile((self._sim._B, 1, n_groups))
                is_valid = True
        elif actu.ndim == 2:
            if actu.shape == (self._sim._B, n_groups):
                self._tgt["actu"] = actu.unsqueeze(1).tile((1, self.n_particles, 1))
                is_valid = True
            if actu.shape == (self._sim._B, n_particles):
                self._tgt["actu"] = actu.unsqueeze(2).tile((1, 1, n_groups))
                is_valid = True
        elif actu.ndim == 3:
            if actu.shape == (self._sim._B, self.n_particles, n_groups):
                self._tgt["actu"] = actu
                is_valid = True
        if not is_valid:
            gs.raise_exception("Tensor shape not supported.")

    def set_muscle(self, muscle_group=None, muscle_direction=None):
        """
        Set both the muscle group indices and direction vectors.

        Parameters
        ----------
        muscle_group : torch.Tensor, optional
            A tensor of shape (n_particles,) with group indices.
        muscle_direction : torch.Tensor, optional
            A tensor of shape (n_particles, 3) with unit vectors.
        """
        self._assert_active()

        if muscle_group is not None:
            n_groups = getattr(self.material, "n_groups", 1)
            max_group_id = muscle_group.max().item()

            muscle_group = to_gs_tensor(muscle_group)

            assert muscle_group.shape == (self.n_particles,)
            assert isinstance(max_group_id, int) and max_group_id < n_groups

            self.set_muscle_group(muscle_group)

        if muscle_direction is not None:
            muscle_direction = to_gs_tensor(muscle_direction)
            assert muscle_direction.shape == (self.n_particles, 3)
            assert torch.allclose(muscle_direction.norm(dim=-1), torch.Tensor([1.0]).to(muscle_direction))

            self.set_muscle_direction(muscle_direction)

    def set_free(self, free):
        """
        Set particles as free or constrained.

        Parameters
        ----------
        free : gs.Tensor
            A tensor of shape (n_particles,) indicating if each particle is free (1) or fixed (0).
        """
        self._assert_active()

        self.solver._kernel_set_free(
            self._particle_start,
            self._n_particles,
            free,
        )

    def get_free(self):
        """
        Get free/fixed status for all particles.

        Returns
        -------
        free : gs.Tensor
            A tensor of shape (n_particles,) indicating free (1) or fixed (0) status.
        """
        self._assert_active()

        free = gs.zeros((self._n_particles,), dtype=gs.tc_bool, requires_grad=False, scene=self._scene)
        self.solver._kernel_get_free(
            self._particle_start,
            self._n_particles,
            free,
        )

        return free

    @ti.kernel
    def clear_grad(self, f: ti.i32):
        """
        Clear all gradients for particle properties at the given substep.

        Parameters
        ----------
        f : int
            The current substep index.
        """
        for i_p, i_b in ti.ndrange(self.n_particles, self._sim._B):
            i_global = i_p + self._particle_start
            self._solver.particles.grad[f, i_global, i_b].pos = 0
            self._solver.particles.grad[f, i_global, i_b].vel = 0
            self._solver.particles.grad[f, i_global, i_b].C = 0
            self._solver.particles.grad[f, i_global, i_b].F = 0
            self._solver.particles.grad[f, i_global, i_b].F_tmp = 0
            self._solver.particles.grad[f, i_global, i_b].Jp = 0
            self._solver.particles.grad[f, i_global, i_b].U = 0
            self._solver.particles.grad[f, i_global, i_b].V = 0
            self._solver.particles.grad[f, i_global, i_b].S = 0
            self._solver.particles.grad[f, i_global, i_b].actu = 0

    def process_input(self, in_backward=False):
        """
        Process buffered target inputs and set them into the solver.

        Parameters
        ----------
        in_backward : bool, optional
            Whether to load target values from the backward pass buffer.
        """
        if in_backward:
            # use negative index because buffer length might not be full
            index = self._sim.cur_step_local - self._sim._steps_local
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
        Process gradients for buffered inputs and backpropagate using custom kernels.
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

    @ti.kernel
    def get_frame(
        self,
        f: ti.i32,
        pos: ti.types.ndarray(),  # shape [B, n_particles, 3]
        vel: ti.types.ndarray(),  # shape [B, n_particles, 3]
        C: ti.types.ndarray(),  # shape [B, n_particles, 3, 3]
        F: ti.types.ndarray(),  # shape [B, n_particles, 3, 3]
        Jp: ti.types.ndarray(),  # shape [B, n_particles]
        active: ti.types.ndarray(),  # shape [B, n_particles]
    ):
        """
        Extract the state of particles at the given frame.

        Parameters
        ----------
        f : int
            Frame index to query.
        pos : ndarray
            Particle positions, shape (B, n_particles, 3).
        vel : ndarray
            Particle velocities, shape (B, n_particles, 3).
        C : ndarray
            Affine matrix C, shape (B, n_particles, 3, 3).
        F : ndarray
            Deformation gradient F, shape (B, n_particles, 3, 3).
        Jp : ndarray
            Volume ratio, shape (B, n_particles).
        active : ndarray
            Particle activeness state, shape (B, n_particles).
        """
        for i_p, i_b in ti.ndrange(self.n_particles, self._sim._B):
            i_global = i_p + self._particle_start
            # Copy pos, vel
            for j in ti.static(range(3)):
                pos[i_b, i_p, j] = self._solver.particles[f, i_global, i_b].pos[j]
                vel[i_b, i_p, j] = self._solver.particles[f, i_global, i_b].vel[j]
                # Copy C, F
                for k in ti.static(range(3)):
                    C[i_b, i_p, j, k] = self._solver.particles[f, i_global, i_b].C[j, k]
                    F[i_b, i_p, j, k] = self._solver.particles[f, i_global, i_b].F[j, k]
            # Copy Jp, active
            Jp[i_b, i_p] = self._solver.particles[f, i_global, i_b].Jp
            active[i_b, i_p] = self._solver.particles_ng[f, i_global, i_b].active

    @ti.kernel
    def set_frame_add_grad_pos(self, f: ti.i32, pos_grad: ti.types.ndarray()):
        """
        Accumulate gradients to particle positions for a frame.

        Parameters
        ----------
        f : int
            Frame index.
        pos_grad : ndarray
            Gradient of particle positions, shape (B, n_particles, 3).
        """
        for i_p, i_b in ti.ndrange(self.n_particles, self._sim._B):
            i_global = i_p + self._particle_start
            for j in ti.static(range(3)):
                self._solver.particles.grad[f, i_global, i_b].pos[j] += pos_grad[i_b, i_p, j]

    @ti.kernel
    def set_frame_add_grad_vel(self, f: ti.i32, vel_grad: ti.types.ndarray()):
        """
        Accumulate gradients to particle velocities for a frame.

        Parameters
        ----------
        f : int
            Frame index.
        vel_grad : ndarray
            Gradient of particle velocities, shape (B, n_particles, 3).
        """
        for i_p, i_b in ti.ndrange(self.n_particles, self._sim._B):
            i_global = i_p + self._particle_start
            for j in ti.static(range(3)):
                self._solver.particles.grad[f, i_global, i_b].vel[j] += vel_grad[i_b, i_p, j]

    @ti.kernel
    def set_frame_add_grad_C(self, f: ti.i32, C_grad: ti.types.ndarray()):
        """
        Accumulate gradients to affine matrices C for a frame.

        Parameters
        ----------
        f : int
            Frame index.
        C_grad : ndarray
            Gradient of C matrices, shape (B, n_particles, 3, 3).
        """
        for i_p, i_b in ti.ndrange(self.n_particles, self._sim._B):
            i_global = i_p + self._particle_start
            for j in ti.static(range(3)):
                for k in ti.static(range(3)):
                    self._solver.particles.grad[f, i_global, i_b].C[j, k] += C_grad[i_b, i_p, j, k]

    @ti.kernel
    def set_frame_add_grad_F(self, f: ti.i32, F_grad: ti.types.ndarray()):
        """
        Accumulate gradients to deformation gradients F for a frame.

        Parameters
        ----------
        f : int
            Frame index.
        F_grad : ndarray
            Gradient of F matrices, shape (B, n_particles, 3, 3).
        """
        for i_p, i_b in ti.ndrange(self.n_particles, self._sim._B):
            i_global = i_p + self._particle_start
            for j in ti.static(range(3)):
                for k in ti.static(range(3)):
                    self._solver.particles.grad[f, i_global, i_b].F[j, k] += F_grad[i_b, i_p, j, k]

    @ti.kernel
    def set_frame_add_grad_Jp(self, f: ti.i32, Jp_grad: ti.types.ndarray()):
        """
        Accumulate gradients to plastic volume ratios Jp for a frame.

        Parameters
        ----------
        f : int
            Frame index.
        Jp_grad : ndarray
            Gradient of Jp values, shape (B, n_particles).
        """
        for i_p, i_b in ti.ndrange(self.n_particles, self._sim._B):
            i_global = i_p + self._particle_start
            self._solver.particles.grad[f, i_global, i_b].Jp += Jp_grad[i_b, i_p]

    def add_grad_from_state(self, state):
        """
        Accumulate gradients from a recorded state back into the solver.

        Parameters
        ----------
        state : MPMEntityState
            The state object containing gradients for physical quantities.
        """
        if state.pos.grad is not None:
            state.pos.assert_contiguous()
            self.set_frame_add_grad_pos(self._sim.cur_substep_local, state.pos.grad)

        if state.vel.grad is not None:
            state.vel.assert_contiguous()
            self.set_frame_add_grad_vel(self._sim.cur_substep_local, state.vel.grad)

        if state.C.grad is not None:
            state.C.assert_contiguous()
            self.set_frame_add_grad_C(self._sim.cur_substep_local, state.C.grad)

        if state.F.grad is not None:
            state.F.assert_contiguous()
            self.set_frame_add_grad_F(self._sim.cur_substep_local, state.F.grad)

        if state.Jp.grad is not None:
            state.Jp.assert_contiguous()
            self.set_frame_add_grad_Jp(self._sim.cur_substep_local, state.Jp.grad)

    @gs.assert_built
    def get_particles(self):
        """
        Retrieve current particle positions from the solver.

        Returns
        -------
        pos : np.ndarray
            Array of particle positions, shape (B, n_particles, 3).
        """
        pos = np.empty((self._sim._B, self.n_particles, 3), dtype=gs.np_float)
        self._kernel_get_particles(self._sim.cur_substep_local, pos)
        return pos

    @ti.kernel
    def _kernel_get_particles(self, f: ti.i32, pos: ti.types.ndarray()):
        for i_p_, i_b in ti.ndrange(self.n_particles, self._sim._B):
            i_p = i_p_ + self._particle_start
            for j in ti.static(range(3)):
                pos[i_b, i_p_, j] = self._solver.particles[f, i_p, i_b].pos[j]

    @gs.assert_built
    def get_state(self):
        """
        Get the current physical state of the particle entity.

        Returns
        -------
        state : MPMEntityState
            The current state of all physical properties of the entity.
        """
        state = MPMEntityState(self, self._sim.cur_step_global)
        self.get_frame(  # TODO: merge with self._solver.get_state?!
            f=self._sim.cur_substep_local,
            pos=state.pos,
            vel=state.vel,
            C=state.C,
            F=state.F,
            Jp=state.Jp,
            active=state.active,
        )

        # we store all queried states to track gradient flow
        self._queried_states.append(state)

        return state

    @ti.kernel
    def _kernel_get_mass(self, mass: ti.types.ndarray()):
        total_mass = 0.0
        for i_p in range(self.n_particles):
            i_global = i_p + self._particle_start
            total_mass += self._solver.particles_info[i_global].mass / self._solver._p_vol_scale
        mass[0] = total_mass
