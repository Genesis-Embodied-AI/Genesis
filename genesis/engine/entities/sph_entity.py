import gstaichi as ti

import genesis as gs
from genesis.engine.states.entities import SPHEntityState

from .particle_entity import ParticleEntity


@ti.data_oriented
class SPHEntity(ParticleEntity):
    """
    SPH-based particle entity.

    Parameters
    ----------
    scene : Scene
        The simulation scene.
    solver : Solver
        The solver handling the simulation logic.
    material : Material
        Material properties (e.g., density, stiffness).
    morph : Morph
        Morphological configuration.
    surface : Surface
        Surface constraints or geometry.
    particle_size : float
        The size of each particle.
    idx : int
        Index of this entity in the scene.
    particle_start : int
        Start index for the particles belonging to this entity.
    """

    def __init__(self, scene, solver, material, morph, surface, particle_size, idx, particle_start):
        super().__init__(
            scene, solver, material, morph, surface, particle_size, idx, particle_start, need_skinning=False
        )

    def init_sampler(self):
        """
        Initialize the particle sampler based on the material's sampling method.

        Raises
        ------
        GenesisException
            If the sampler is not one of the supported types: 'regular', 'pbs', or 'pbs-sdf_res'.
        """
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
        """
        Set particle positions for the specified frame.

        Parameters
        ----------
        f : int
            Frame index.
        pos : ndarray
            Array of particle positions of shape (n_envs, n_particles, 3).
        """
        self.solver._kernel_set_particles_pos(f, self._particle_start, self._n_particles, pos)

    def set_pos_grad(self, f: ti.i32, pos_grad: ti.types.ndarray()):
        """
        Set gradient of particle positions.

        Parameters
        ----------
        f : int
            Frame index.
        pos_grad : ndarray
            Gradient array for positions.
        """
        pass

    @gs.assert_built
    def set_vel(self, f, vel):
        """
        Set particle velocities for the specified frame.

        Parameters
        ----------
        f : int
            Frame index.
        vel : ndarray
            Array of particle velocities of shape (n_envs, n_particles, 3).
        """
        self.solver._kernel_set_particles_vel(
            f,
            self._particle_start,
            self._n_particles,
            vel,
        )

    def set_vel_grad(self, f: ti.i32, vel_grad: ti.types.ndarray()):
        """
        Set gradient of particle velocities.

        Parameters
        ----------
        f : int
            Frame index.
        vel_grad : ndarray
            Gradient array for velocities.
        """
        pass

    @gs.assert_built
    def set_active(self, f, active):
        """
        Set the active status of particles for a given frame.

        Parameters
        ----------
        f : int
            Frame index.
        active : ndarray
            Boolean array indicating whether each particle is active.
        """
        self.solver._kernel_set_particles_active(
            f,
            self._particle_start,
            self._n_particles,
            active,
        )

    def clear_grad(self, f: ti.i32):
        """
        Placeholder to clear gradients for the specified frame (not yet implemented).

        Parameters
        ----------
        f : int
            Frame index.
        """
        pass

    @ti.kernel
    def get_frame(
        self,
        f: ti.i32,
        pos: ti.types.ndarray(),
        vel: ti.types.ndarray(),
    ):
        """
        Retrieve particle positions and velocities for the given frame.

        Parameters
        ----------
        f : int
            Frame index.
        pos : ndarray
            Output array for positions (n_envs, n_particles, 3).
        vel : ndarray
            Output array for velocities (n_envs, n_particles, 3).
        """
        for i_p, i_b in ti.ndrange(self.n_particles, self._sim._B):
            i_global = i_p + self._particle_start
            for j in ti.static(range(3)):
                pos[i_b, i_p, j] = self.solver.particles[i_global, i_b].pos[j]
                vel[i_b, i_p, j] = self.solver.particles[i_global, i_b].vel[j]

    def add_grad_from_state(self, state):
        """
        Apply gradients from a given state.

        Parameters
        ----------
        state : SPHEntityState
            The state from which to compute gradients.
        """
        pass

    @ti.kernel
    def _kernel_get_particles(self, particles: ti.types.ndarray()):
        for i_p, i_b in ti.ndrange(self.n_particles, self._sim._B):
            for j in ti.static(range(3)):
                particles[i_b, i_p, j] = self.solver.particles[i_p + self._particle_start, i_b].pos[j]

    @gs.assert_built
    def get_state(self):
        """
        Get the current state of the SPHEntity including positions, velocities, .

        Returns
        -------
        state : SPHEntityState
            The current particle state for the entity.
        """
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
            total_mass += self._solver.particles[i_global, 0].m
        mass[0] = total_mass
