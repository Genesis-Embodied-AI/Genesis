import taichi as ti
import genesis as gs
from .sph_entity import SPHEntity


@ti.data_oriented
class MultiphaseSPHEntity(SPHEntity):
    """
    SPH-based particle entity with multiphase support.
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
        phase,
    ):
        """
        Initialize a MultiphaseSPHEntity.
        """
        self.phase = phase  # Store phase information
        super().__init__(
            scene, solver, material, morph, surface, particle_size, idx, particle_start
        )

    def _add_to_solver_(self):
        """
        Adds particles to the solver with phase information.
        """
        self._solver._kernel_add_particles(
            self._sim.cur_substep_local,
            self.active,
            self._particle_start,
            self._n_particles,
            self._material["rho"],
            self._material["stiffness"],
            self._material["exponent"],
            self._material["mu"],
            self._material["gamma"],
            self._material["tau_y"],
            self._material["K"],
            self._material["n"],
            self.phase,  # Pass phase information
            self._particles,
        )

    @gs.assert_built
    def set_phase(self, f, phase):
        """
        Sets the phase of the entity's particles.

        Parameters:
            f: Frame index or identifier.
            phase: Integer identifier for the new phase.
        """
        self.phase = phase
        self.solver._kernel_set_particle_phase(
            f,
            self._particle_start,
            self._n_particles,
            phase,
        )
