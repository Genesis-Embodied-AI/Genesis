import numpy as np
import taichi as ti
import torch

import genesis as gs
from genesis.engine.states.entities import MPMEntityState
from genesis.utils.misc import to_gs_tensor

from .particle_entity import ParticleEntity


@ti.data_oriented
class MPMEntity(ParticleEntity):
    """
    MPM-based particle entity.
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
        self.solver._kernel_set_particles_pos(
            f,
            self._particle_start,
            self._n_particles,
            pos,
        )

    def set_pos_grad(self, f, pos_grad):
        self.solver._kernel_set_particles_pos_grad(
            f,
            self._particle_start,
            self._n_particles,
            pos_grad,
        )

    def set_vel(self, f, vel):
        self.solver._kernel_set_particles_vel(
            f,
            self._particle_start,
            self._n_particles,
            vel,
        )

    def set_vel_grad(self, f, vel_grad):
        self.solver._kernel_set_particles_vel_grad(
            f,
            self._particle_start,
            self._n_particles,
            vel_grad,
        )

    def set_actu(self, f, actu):
        self.solver._kernel_set_particles_actu(
            f,
            self._particle_start,
            self._n_particles,
            self.material.n_groups,
            actu,
        )

    def set_actu_grad(self, f, actu_grad):
        self.solver._kernel_set_particles_actu_grad(
            f,
            self._particle_start,
            self._n_particles,
            actu_grad,
        )

    def set_muscle_group(self, muscle_group):
        self.solver._kernel_set_muscle_group(
            self._particle_start,
            self._n_particles,
            muscle_group,
        )

    def get_muscle_group(self):
        muscle_group = gs.zeros((self._n_particles,), dtype=int, requires_grad=False, scene=self._scene)
        self.solver._kernel_get_muscle_group(
            self._particle_start,
            self._n_particles,
            muscle_group,
        )

        return muscle_group

    def set_muscle_direction(self, muscle_direction):
        self.solver._kernel_set_muscle_direction(
            self._particle_start,
            self._n_particles,
            muscle_direction,
        )

    def set_active(self, f, active):
        self.solver._kernel_set_particles_active(
            f,
            self._particle_start,
            self._n_particles,
            active,
        )

    def set_active_arr(self, f, active):
        self.solver._kernel_set_particles_active_arr(
            f,
            self._particle_start,
            self._n_particles,
            active,
        )

    def set_actuation(self, actu):
        self._assert_active()

        actu = to_gs_tensor(actu)

        n_groups = getattr(self.material, "n_groups", 1)

        if len(actu.shape) == 0:
            assert actu.shape == ()
            self._tgt["actu"] = torch.tile(actu, [self.n_particles, n_groups])

        elif len(actu.shape) == 1:
            if actu.shape[0] == n_groups:
                assert self.n_particles != n_groups  # ambiguous
                actu = actu.tile([self.n_particles, 1])
            else:
                assert actu.shape == (self.n_particles,)
                gs.raise_exception("Cannot set per-particle actuation")
            self._tgt["actu"] = actu

        else:
            gs.raise_exception("Tensor shape not supported.")

    def set_muscle(self, muscle_group=None, muscle_direction=None):
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
        self._assert_active()

        self.solver._kernel_set_free(
            self._particle_start,
            self._n_particles,
            free,
        )

    def get_free(self):
        self._assert_active()

        free = gs.zeros((self._n_particles,), dtype=int, requires_grad=False, scene=self._scene)
        self.solver._kernel_get_free(
            self._particle_start,
            self._n_particles,
            free,
        )

        return free

    @ti.kernel
    def clear_grad(self, f: ti.i32):
        for i in range(self.n_particles):
            i_global = i + self._particle_start
            self._solver.particles.grad[f, i_global].pos = 0
            self._solver.particles.grad[f, i_global].vel = 0
            self._solver.particles.grad[f, i_global].C = 0
            self._solver.particles.grad[f, i_global].F = 0
            self._solver.particles.grad[f, i_global].F_tmp = 0
            self._solver.particles.grad[f, i_global].Jp = 0
            self._solver.particles.grad[f, i_global].U = 0
            self._solver.particles.grad[f, i_global].V = 0
            self._solver.particles.grad[f, i_global].S = 0
            self._solver.particles.grad[f, i_global].actu = 0

    def process_input(self, in_backward=False):
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
        pos: ti.types.ndarray(),
        vel: ti.types.ndarray(),
        C: ti.types.ndarray(),
        F: ti.types.ndarray(),
        Jp: ti.types.ndarray(),
        active: ti.types.ndarray(),
    ):
        for i in range(self.n_particles):
            i_global = i + self._particle_start
            for j in ti.static(range(3)):
                pos[i, j] = self._solver.particles[f, i_global].pos[j]
                vel[i, j] = self._solver.particles[f, i_global].vel[j]
                for k in ti.static(range(3)):
                    C[i, j, k] = self._solver.particles[f, i_global].C[j, k]
                    F[i, j, k] = self._solver.particles[f, i_global].F[j, k]
            Jp[i] = self._solver.particles[f, i_global].Jp
            active[i] = self._solver.particles_ng[f, i_global].active

    @ti.kernel
    def set_frame_add_grad_pos(self, f: ti.i32, pos_grad: ti.types.ndarray()):
        for i in range(self.n_particles):
            i_global = i + self._particle_start
            for j in ti.static(range(3)):
                self._solver.particles.grad[f, i_global].pos[j] += pos_grad[i, j]

    @ti.kernel
    def set_frame_add_grad_vel(self, f: ti.i32, vel_grad: ti.types.ndarray()):
        for i in range(self.n_particles):
            i_global = i + self._particle_start
            for j in ti.static(range(3)):
                self._solver.particles.grad[f, i_global].vel[j] += vel_grad[i, j]

    @ti.kernel
    def set_frame_add_grad_C(self, f: ti.i32, C_grad: ti.types.ndarray()):
        for i in range(self.n_particles):
            i_global = i + self._particle_start
            for j in ti.static(range(3)):
                for k in ti.static(range(3)):
                    self._solver.particles.grad[f, i_global].C[j, k] += C_grad[i, j, k]

    @ti.kernel
    def set_frame_add_grad_F(self, f: ti.i32, F_grad: ti.types.ndarray()):
        for i in range(self.n_particles):
            i_global = i + self._particle_start
            for j in ti.static(range(3)):
                for k in ti.static(range(3)):
                    self._solver.particles.grad[f, i_global].F[j, k] += F_grad[i, j, k]

    @ti.kernel
    def set_frame_add_grad_Jp(self, f: ti.i32, Jp_grad: ti.types.ndarray()):
        for i in range(self.n_particles):
            i_global = i + self._particle_start
            self._solver.particles.grad[f, i_global].Jp += Jp_grad[i]

    def add_grad_from_state(self, state):
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
        pos = np.empty((self.n_particles, 3), dtype=gs.np_float)
        self._kernel_get_particles(self._sim.cur_substep_local, pos)
        return pos

    @ti.kernel
    def _kernel_get_particles(self, f: ti.i32, pos: ti.types.ndarray()):
        for i in range(self.n_particles):
            i_global = i + self._particle_start
            for j in ti.static(range(3)):
                pos[i, j] = self._solver.particles[f, i_global].pos[j]

    @gs.assert_built
    def get_state(self):
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
        for i in range(self.n_particles):
            i_global = i + self._particle_start
            total_mass += self._solver.particles_info[i_global].mass / self._solver._p_vol_scale
        mass[0] = total_mass
