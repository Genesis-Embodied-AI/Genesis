import numpy as np
import taichi as ti

import genesis as gs
import genesis.utils.geom as gu
import genesis.utils.particle as pu
from genesis.repr_base import RBC


@ti.data_oriented
class Emitter(RBC):
    def __init__(self, max_particles):
        self._uid = gs.UID()
        self._entity = None

        self._max_particles = max_particles

        self._acc_droplet_len = 0.0  # accumulated droplet length to be emitted

        gs.logger.info(
            f"Creating ~<{self._repr_type()}>~. id: ~~~<{self._uid}>~~~, max_particles: ~<{max_particles}>~."
        )

    def set_entity(self, entity):
        self._entity = entity
        self._sim = entity.sim
        self._solver = entity.solver
        self._next_particle = 0
        gs.logger.info(f"~<{self._repr_briefer()}>~ created using ~<{entity._repr_briefer()}.")

    def reset(self):
        self._next_particle = 0

    def emit(
        self,
        droplet_shape,
        droplet_size,
        droplet_length=None,
        pos=(0.5, 0.5, 1.0),
        direction=(0, 0, -1),
        theta=0.0,
        speed=1.0,
        p_size=None,
    ):
        assert self._entity is not None

        if droplet_shape in ["circle", "sphere", "square"]:
            assert isinstance(droplet_size, (int, float))
        elif droplet_shape == "rectangle":
            assert isinstance(droplet_size, (tuple, list)) and len(droplet_size) == 2
        else:
            gs.raise_exception(f"Unsupported nozzle shape: {droplet_shape}.")

        if np.linalg.norm(direction) < gs.EPS:
            gs.raise_exception("Zero-length direction.")
        else:
            direction = gu.normalize(direction)

        p_size = self._solver.particle_size if p_size is None else p_size

        pos = np.array(pos)
        if droplet_length is None:
            # Use the speed to determine the length of the droplet in the emitting direction
            droplet_length = speed * self._solver.substep_dt * self._sim.substeps + self._acc_droplet_len
            if droplet_length < p_size:  # too short, so we should not emit
                self._acc_droplet_len = droplet_length
                droplet_length = 0.0
            else:
                self._acc_droplet_len = 0.0

        if droplet_length > 0.0:
            if droplet_shape == "circle":
                positions = pu.cylinder_to_particles(
                    p_size=p_size,
                    radius=droplet_size / 2,
                    height=droplet_length,
                    sampler=self._entity.sampler,
                )
            elif droplet_shape == "sphere":  # sphere droplet ignores droplet_length
                positions = pu.sphere_to_particles(
                    p_size=p_size,
                    radius=droplet_size / 2,
                    sampler=self._entity.sampler,
                )
            elif droplet_shape == "square":
                positions = pu.box_to_particles(
                    p_size=p_size,
                    size=np.array([droplet_size, droplet_size, droplet_length]),
                    sampler=self._entity.sampler,
                )
            elif droplet_shape == "rectangle":
                positions = pu.box_to_particles(
                    p_size=p_size,
                    size=np.array([droplet_size[0], droplet_size[1], droplet_length]),
                    sampler=self._entity.sampler,
                )
            else:
                gs.raise_exception()

            positions = gu.transform_by_T(
                positions, gu.trans_R_to_T(pos, gu.z_to_R(direction) @ gu.axis_angle_to_R(np.array([0, 0, 1]), theta))
            ).astype(gs.np_float)

            if not self._solver.boundary.is_inside(positions):
                gs.raise_exception("Emitted particles are outside the boundary.")

            n_particles = len(positions)

            vels = np.tile(direction * speed, (n_particles, 1)).astype(gs.np_float)

            if n_particles > self._entity.n_particles:
                gs.logger.warning(
                    f"Number of particles to emit ({n_particles}) at the current step is larger than the maximum number of particles ({self._entity.n_particles})."
                )

            self._solver._kernel_set_particles_pos(
                self._sim.cur_substep_local,
                self._entity.particle_start + self._next_particle,
                n_particles,
                positions,
            )
            self._solver._kernel_set_particles_vel(
                self._sim.cur_substep_local,
                self._entity.particle_start + self._next_particle,
                n_particles,
                vels,
            )
            self._solver._kernel_set_particles_active(
                self._sim.cur_substep_local,
                self._entity.particle_start + self._next_particle,
                n_particles,
                gs.ACTIVE,
            )

            self._next_particle += n_particles

            # recycle particles
            if self._next_particle + n_particles > self._entity.n_particles:
                self._next_particle = 0

            gs.logger.debug(f"Emitted {n_particles} particles. Next particle index: {self._next_particle}.")

        else:
            gs.logger.debug("Droplet length is too short for current step. Skipping to next step.")

    def emit_omni(self, source_radius=0.1, pos=(0.5, 0.5, 1.0), speed=1.0, particle_size=None):
        """
        Use a sphere-shaped source to emit particles in all directions.

        Parameters:
        ----------
        source_radius: float, optional
            The radius of the sphere source. Particles will be emitted from a shell with inner radius using 0.8 * source_radius and outer radius using source_radius.
        pos: array_like, shape=(3,)
            The center of the sphere source.
        speed: float
            The speed of the emitted particles.
        particle_size: float | None
            The size (diameter) of the emitted particles. The actual number of particles emitted is determined by the volume of the sphere source and the size of the particles. If None, the solver's particle size is used. Note that this particle size only affects computation for number of particles emitted, not the actual size of the particles in simulation and rendering.
        """
        assert self._entity is not None

        pos = np.array(pos)

        if particle_size is None:
            particle_size = self._solver.particle_size

        positions_ = pu.shell_to_particles(
            p_size=particle_size,
            outer_radius=source_radius,
            inner_radius=source_radius * 0.4,
            sampler=self._entity.sampler,
        )

        positions = gu.transform_by_T(positions_, gu.trans_to_T(pos)).astype(gs.np_float)

        if not self._solver.boundary.is_inside(positions):
            gs.raise_exception("Emitted particles are outside the boundary.")

        n_particles = len(positions)
        dists = np.linalg.norm(positions_, axis=1, keepdims=True)
        positions[np.where(dists < gs.EPS)[0]] = np.array([gs.EPS, gs.EPS, gs.EPS])
        vels = (positions_ / dists * speed).astype(gs.np_float)

        if n_particles > self._entity.n_particles:
            gs.logger.warning(
                f"Number of particles to emit ({n_particles}) at the current step is larger than the maximum number of particles ({self._entity.n_particles})."
            )

        self._solver._kernel_set_particles_pos(
            self._sim.cur_substep_local,
            self._entity.particle_start + self._next_particle,
            n_particles,
            positions,
        )
        self._solver._kernel_set_particles_vel(
            self._sim.cur_substep_local,
            self._entity.particle_start + self._next_particle,
            n_particles,
            vels,
        )
        self._solver._kernel_set_particles_active(
            self._sim.cur_substep_local,
            self._entity.particle_start + self._next_particle,
            n_particles,
            gs.ACTIVE,
        )

        self._next_particle += n_particles

        # recycle particles
        if self._next_particle + n_particles > self._entity.n_particles:
            self._next_particle = 0

        gs.logger.debug(f"Emitted {n_particles} particles. Next particle index: {self._next_particle}.")

    @property
    def uid(self):
        return self._uid

    @property
    def entity(self):
        return self._entity

    @property
    def max_particles(self):
        return self._max_particles

    @property
    def solver(self):
        return self._solver

    @property
    def next_particle(self):
        return self._next_particle
