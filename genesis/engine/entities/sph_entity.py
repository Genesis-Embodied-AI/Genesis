import taichi as ti

import genesis as gs
from genesis.engine.states.entities import SPHEntityState

from .particle_entity import ParticleEntity


@ti.data_oriented
class SPHEntity(ParticleEntity):
    """
    SPH-based particle entity.
    """

    def __init__(self, scene, solver, material, morph, surface, particle_size, idx, particle_start):
        super().__init__(
            scene, solver, material, morph, surface, particle_size, idx, particle_start, need_skinning=False
        )

    def init_sampler(self):
        self.sampler = self._material.sampler

        valid = True
        if self.sampler == "regular":
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
                f"Only one of the following samplers is supported: [`regular`, `pbs`, `pbs-sdf_res`]. Got: {self.sampler}."
            )

    def _add_to_solver_(self):
        self._solver._kernel_add_particles(
            self._sim.cur_substep_local,
            self.active,
            self._particle_start,
            self._n_particles,
            self._material.rho,
            self._material.stiffness,
            self._material.exponent,
            self._material.mu,
            self._material.gamma,
            self._particles,
        )

    @gs.assert_built
    def set_pos(self, f, pos):
        self.solver._kernel_set_particles_pos(
            f,
            self._particle_start,
            self._n_particles,
            pos,
        )

    def set_pos_grad(self, f: ti.i32, pos_grad: ti.types.ndarray()):
        pass

    @gs.assert_built
    def set_vel(self, f, vel):
        self.solver._kernel_set_particles_vel(
            f,
            self._particle_start,
            self._n_particles,
            vel,
        )

    def set_vel_grad(self, f: ti.i32, vel_grad: ti.types.ndarray()):
        pass

    @gs.assert_built
    def set_active(self, f, active):
        self.solver._kernel_set_particles_active(
            f,
            self._particle_start,
            self._n_particles,
            active,
        )

    def clear_grad(self, f: ti.i32):
        pass

    @ti.kernel
    def get_frame(
        self,
        f: ti.i32,
        pos: ti.types.ndarray(),
        vel: ti.types.ndarray(),
    ):
        for i in range(self.n_particles):
            i_global = i + self._particle_start
            for j in ti.static(range(3)):
                pos[i, j] = self.solver.particles[i_global].pos[j]
                vel[i, j] = self.solver.particles[i_global].vel[j]

    def add_grad_from_state(self, state):
        pass

    @ti.kernel
    def _kernel_get_particles(self, particles: ti.types.ndarray()):
        for i in range(self.n_particles):
            for j in ti.static(range(3)):
                particles[i, j] = self.solver.particles[i + self._particle_start].pos[j]

    @gs.assert_built
    def get_state(self):
        state = SPHEntityState(self, self.sim.cur_step_global)
        self.get_frame(self.sim.cur_substep_local, state.pos, state.vel)

        # we store all queried states to track gradient flow
        self._queried_states.append(state)

        return state

    @ti.kernel
    def _kernel_get_mass(self, mass: ti.types.ndarray()):
        total_mass = 0.0
        for i in range(self.n_particles):
            i_global = i + self._particle_start
            total_mass += self._solver.particles[i_global].m
        mass[0] = total_mass
