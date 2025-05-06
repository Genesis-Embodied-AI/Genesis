import numpy as np
import taichi as ti
from scipy.spatial import KDTree

import genesis as gs
import genesis.utils.geom as gu
import genesis.utils.mesh as mu
from genesis.engine.entities.base_entity import Entity
from genesis.engine.entities.particle_entity import ParticleEntity
import trimesh


@ti.data_oriented
class PBDTetEntity(ParticleEntity):
    """
    PBD entity represented by tetrahedral elements.

    Parameters
    ----------
    scene : Scene
        The simulation scene this entity is part of.
    solver : Solver
        The PBD solver instance managing this entity.
    material : Material
        Material model defining physical properties such as density and compliance.
    morph : Morph
        Morph object specifying shape and initial transform (position and rotation).
    surface : Surface
        Surface or texture representation.
    particle_size : float
        Target size for particle spacing.
    idx : int
        Unique index of this entity within the scene.
    particle_start : int
        Starting index of this entity's particles in the global particle buffer.
    edge_start : int
        Starting index of this entity's edges in the global edge buffer.
    vvert_start : int
        Starting index of this entity's visual vertices.
    vface_start : int
        Starting index of this entity's visual faces.
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
        edge_start,
        vvert_start,
        vface_start,
    ):

        super().__init__(
            scene, solver, material, morph, surface, particle_size, idx, particle_start, vvert_start, vface_start
        )

        self._edge_start = edge_start

    def sample(self):
        """
        Sample and preprocess the mesh for the PBD tetrahedral entity.

        Applies transformation from the morph, stores mesh vertices and faces,
        and performs remeshing based on the particle size.
        """
        # We don't use ParticleEntity.sample() because we need to maintain the remeshed self._mesh as well
        self._vmesh.apply_transform(gu.trans_quat_to_T(np.array(self._morph.pos), np.array(self._morph.quat)))
        self._vverts = np.array(self._vmesh.verts)
        self._vfaces = np.array(self._vmesh.faces)

        self._mesh = self._vmesh.copy()
        self._mesh.remesh(edge_len_abs=self.particle_size, fix=isinstance(self, PBD3DEntity))

    def _add_to_solver_(self):
        self._kernel_add_particles_edges_to_solver(
            f=self._scene.sim.cur_substep_local,
            particles=self._particles.astype(gs.np_float),
            edges=self._edges.astype(gs.np_int),
            edges_len_rest=self._edges_len_rest.astype(gs.np_float),
            mat_type=self._mat_type,
            active=True,
        )

    @ti.kernel
    def _kernel_add_particles_edges_to_solver(
        self,
        f: ti.i32,
        particles: ti.types.ndarray(),
        edges: ti.types.ndarray(),
        edges_len_rest: ti.types.ndarray(),
        mat_type: ti.i32,
        active: ti.i32,
    ):
        for i_p_ in range(self.n_particles):
            i_p = i_p_ + self._particle_start
            for j in ti.static(range(3)):
                self.solver.particles_info[i_p].pos_rest[j] = particles[i_p_, j]
            self.solver.particles_info[i_p].mat_type = mat_type
            self.solver.particles_info[i_p].mass = self._particle_mass
            self.solver.particles_info[i_p].mu_s = self.material.static_friction
            self.solver.particles_info[i_p].mu_k = self.material.kinetic_friction

        for i_p_, i_b in ti.ndrange(self.n_particles, self._sim._B):
            i_p = i_p_ + self._particle_start
            for j in ti.static(range(3)):
                self.solver.particles[i_p, i_b].pos[j] = particles[i_p_, j]
            self.solver.particles[i_p, i_b].vel = ti.Vector.zero(gs.ti_float, 3)
            self.solver.particles[i_p, i_b].dpos = ti.Vector.zero(gs.ti_float, 3)
            self.solver.particles[i_p, i_b].free = True

            self.solver.particles_ng[i_p, i_b].active = active

        for i_e_ in range(self.n_edges):
            i_e = i_e_ + self._edge_start
            self.solver.edges_info[i_e].stretch_compliance = self.material.stretch_compliance
            self.solver.edges_info[i_e].stretch_relaxation = self.material.stretch_relaxation
            self.solver.edges_info[i_e].len_rest = edges_len_rest[i_e_]
            self.solver.edges_info[i_e].v1 = self._particle_start + edges[i_e_, 0]
            self.solver.edges_info[i_e].v2 = self._particle_start + edges[i_e_, 1]

    def process_input(self, in_backward=False):
        """
        Push position, velocity, and activation target states into the simulator.

        Parameters
        ----------
        in_backward : bool, default=False
            Whether the simulation is in the backward (gradient) pass.
        """
        # TODO: implement this
        pass

    # ------------------------------------------------------------------------------------
    # ---------------------------------- io & control ------------------------------------
    # ------------------------------------------------------------------------------------

    @ti.kernel
    def _kernel_get_particles(self, particles: ti.types.ndarray()):
        for i_p, i_b in ti.ndrange(self.n_particles, self._sim._B):
            for j in ti.static(range(3)):
                particles[i_b, i_p, j] = self.solver.particles[i_p + self._particle_start, i_b].pos[j]

    @gs.assert_built
    def find_closest_particle(self, pos, b=0):
        """
        Find the index of the particle closest to a given position.

        Parameters
        ----------
        pos : array-like
            The target position to compare against.
        b : int, optional
            The environment index, by default 0.

        Returns
        -------
        closest_idx : int
            The index of the closest particle.
        """
        cur_particles = self.get_particles()[b]
        distances = np.linalg.norm(cur_particles - np.array(pos), axis=1)
        closest_idx = np.argmin(distances)
        return closest_idx

    @gs.assert_built
    def fix_particle(self, particle_idx, i_b):
        """
        Fix a particle's position in the simulation.

        Parameters
        ----------
        particle_idx : int
            Index of the particle relative to this entity.
        i_b : int
            Environment index.

        Returns
        -------
        None
        """
        self.solver.fix_particle(particle_idx + self._particle_start, i_b)

    @gs.assert_built
    def set_particle_position(self, particle_idx, pos):
        """
        Set the position of a specific particle.

        Parameters
        ----------
        particle_idx : int
            Index of the particle relative to this entity.
        pos : array-like
            Target position to assign.

        Returns
        -------
        None
        """
        self.solver.set_particle_position(particle_idx + self._particle_start, pos)

    @gs.assert_built
    def set_particle_velocity(self, particle_idx, vel):
        """
        Set the velocity of a specific particle.

        Parameters
        ----------
        particle_idx : int
            Index of the particle relative to this entity.
        vel : array-like
            Target velocity to assign.

        Returns
        -------
        None
        """
        self.solver.set_particle_velocity(particle_idx + self._particle_start, vel)

    @gs.assert_built
    def release_particle(self, particle_idx):
        """
        Release a fixed particle, allowing it to move freely.

        Parameters
        ----------
        particle_idx : int
            Index of the particle relative to this entity.

        Returns
        -------
        None
        """
        self.solver.release_particle(particle_idx + self._particle_start)

    # ------------------------------------------------------------------------------------
    # ----------------------------------- properties -------------------------------------
    # ------------------------------------------------------------------------------------

    @property
    def mesh(self):
        """Mesh."""
        return self._mesh

    @property
    def edges(self):
        """Edge array of the mesh."""
        return self._edges

    @property
    def n_edges(self):
        """Number of edges in the mesh."""
        return len(self._edges)


@ti.data_oriented
class PBD2DEntity(PBDTetEntity):
    """
    PBD entity represented by a 2D mesh.

    Parameters
    ----------
    scene : Scene
        The simulation scene this entity is part of.
    solver : Solver
        The PBD solver instance managing this entity.
    material : Material
        Material model defining physical properties such as density and compliance.
    morph : Morph
        Morph object specifying shape and initial transform (position and rotation).
    surface : Surface
        Surface or texture representation.
    particle_size : float
        Target size for particle spacing.
    idx : int
        Unique index of this entity within the scene.
    particle_start : int
        Starting index of this entity's particles in the global particle buffer.
    edge_start : int
        Starting index of this entity's edges in the global edge buffer.
    inner_edge_start: int
        Starting index of this entity's inner edges in the global buffer.
    vvert_start : int
        Starting index of this entity's visual vertices.
    vface_start : int
        Starting index of this entity's visual faces.
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
        edge_start,
        inner_edge_start,
        vvert_start,
        vface_start,
    ):
        super().__init__(
            scene,
            solver,
            material,
            morph,
            surface,
            particle_size,
            idx,
            particle_start,
            edge_start,
            vvert_start,
            vface_start,
        )

        self._inner_edge_start = inner_edge_start
        self._mat_type = self.solver.MATS.CLOTH

    def sample(self):
        """Sample and preprocess the 2D mesh for the PBD cloth-like entity."""
        super().sample()

        if self._vmesh.area < 1e-6:
            gs.raise_exception("Input mesh has zero surface area.")
        self._mass = self._vmesh.area * self.material.rho

        self._particles = np.array(self._mesh.verts)
        self._edges = np.array(self._mesh.get_unique_edges())

        self._particle_mass = self._mass / len(self._particles)

        # Inner edges are two diagonal edges of each quadrilateral formed by adjacent face pairs
        adjacency, inner_edges = trimesh.graph.face_adjacency(mesh=self._mesh.trimesh, return_edges=True)
        v3 = np.sum(self._mesh.faces[adjacency[:, 0]], axis=1) - inner_edges[:, 0] - inner_edges[:, 1]
        v4 = np.sum(self._mesh.faces[adjacency[:, 1]], axis=1) - inner_edges[:, 0] - inner_edges[:, 1]
        self._inner_edges = np.stack([inner_edges[:, 0], inner_edges[:, 1], v3, v4], axis=1)

        self._edges_len_rest = np.linalg.norm(
            self._particles[self._edges[:, 0]] - self._particles[self._edges[:, 1]], axis=1
        )
        self._inner_edges_len_rest = np.linalg.norm(
            self._particles[self._inner_edges[:, 2]] - self._particles[self._inner_edges[:, 3]], axis=1
        )
        self._n_particles = len(self._particles)

    def _add_to_solver_(self):
        super()._add_to_solver_()

        self._kernel_add_particles_air_resistance_to_solver(
            f=self._scene.sim.cur_substep_local,
        )

        self._kernel_add_inner_edges_to_solver(
            f=self._scene.sim.cur_substep_local,
            inner_edges=self._inner_edges.astype(gs.np_int),
            inner_edges_len_rest=self._inner_edges_len_rest.astype(gs.np_float),
        )

    @ti.kernel
    def _kernel_add_particles_air_resistance_to_solver(
        self,
        f: ti.i32,
    ):
        for i_p_ in range(self.n_particles):
            i_p = i_p_ + self._particle_start
            self.solver.particles_info[i_p].air_resistance = self.material.air_resistance

    @ti.kernel
    def _kernel_add_inner_edges_to_solver(
        self,
        f: ti.i32,
        inner_edges: ti.types.ndarray(),
        inner_edges_len_rest: ti.types.ndarray(),
    ):
        for i_ie_ in range(self.n_inner_edges):
            i_ie = i_ie_ + self._inner_edge_start
            self.solver.inner_edges_info[i_ie].bending_compliance = self.material.bending_compliance
            self.solver.inner_edges_info[i_ie].bending_relaxation = self.material.bending_relaxation
            self.solver.inner_edges_info[i_ie].len_rest = inner_edges_len_rest[i_ie_]
            self.solver.inner_edges_info[i_ie].v1 = self._particle_start + inner_edges[i_ie_, 0]
            self.solver.inner_edges_info[i_ie].v2 = self._particle_start + inner_edges[i_ie_, 1]
            self.solver.inner_edges_info[i_ie].v3 = self._particle_start + inner_edges[i_ie_, 2]
            self.solver.inner_edges_info[i_ie].v4 = self._particle_start + inner_edges[i_ie_, 3]

    @property
    def n_inner_edges(self):
        """The number of inner edges in the 2D mesh."""
        return len(self._inner_edges)


@ti.data_oriented
class PBD3DEntity(PBDTetEntity):
    """
    PBD entity represented by a 3D mesh.

    Parameters
    ----------
    scene : Scene
        The simulation scene this entity is part of.
    solver : Solver
        The PBD solver instance managing this entity.
    material : Material
        Material model defining physical properties such as density and compliance.
    morph : Morph
        Morph object specifying shape and initial transform (position and rotation).
    surface : Surface
        Surface or texture representation.
    particle_size : float
        Target size for particle spacing.
    idx : int
        Unique index of this entity within the scene.
    particle_start : int
        Starting index of this entity's particles in the global particle buffer.
    edge_start : int
        Starting index of this entity's edges in the global edge buffer.
    elem_start: int
        Starting index of this entity's element in the global buffer.
    vvert_start : int
        Starting index of this entity's visual vertices.
    vface_start : int
        Starting index of this entity's visual faces.
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
        edge_start,
        elem_start,
        vvert_start,
        vface_start,
    ):
        super().__init__(
            scene,
            solver,
            material,
            morph,
            surface,
            particle_size,
            idx,
            particle_start,
            edge_start,
            vvert_start,
            vface_start,
        )

        self._elem_start = elem_start

        self._mat_type = self.solver.MATS.ELASTIC

    def sample(self):
        super().sample()

        if self._vmesh.volume < 1e-6:
            gs.raise_exception("Input mesh has zero volume.")
        self._mass = self._vmesh.volume * self.material.rho

        tet_cfg = mu.generate_tetgen_config_from_morph(self.morph)
        self._particles, self._elems = self._mesh.tetrahedralize(tet_cfg)
        self._edges = np.array(
            list(
                set(
                    tuple(sorted([self._elems[i, j], self._elems[i, k]]))
                    for i in range(self._elems.shape[0])
                    for j in range(4)
                    for k in range(j + 1, 4)
                )
            )
        )
        self._particle_mass = self._mass / len(self._particles)

        self._edges_len_rest = np.linalg.norm(
            self._particles[self._edges[:, 0]] - self._particles[self._edges[:, 1]], axis=1
        )
        self._elems_vol_rest = (
            np.linalg.det(self._particles[self._elems[:, 1:]] - self._particles[self._elems[:, 0:1]]) / 6.0
        )
        self._n_particles = len(self._particles)

    def _add_to_solver_(self):
        super()._add_to_solver_()

        self._kernel_add_elems_to_solver(
            elems=self._elems.astype(gs.np_int),
            elems_vol_rest=self._elems_vol_rest.astype(gs.np_float),
        )

    @ti.kernel
    def _kernel_add_elems_to_solver(
        self,
        elems: ti.types.ndarray(),
        elems_vol_rest: ti.types.ndarray(),
    ):
        for i_el_ in range(self.n_elems):
            i_el = i_el_ + self._elem_start
            self.solver.elems_info[i_el].volume_compliance = self.material.volume_compliance
            self.solver.elems_info[i_el].volume_relaxation = self.material.volume_relaxation
            self.solver.elems_info[i_el].vol_rest = elems_vol_rest[i_el_]
            self.solver.elems_info[i_el].v1 = self._particle_start + elems[i_el_, 0]
            self.solver.elems_info[i_el].v2 = self._particle_start + elems[i_el_, 1]
            self.solver.elems_info[i_el].v3 = self._particle_start + elems[i_el_, 2]
            self.solver.elems_info[i_el].v4 = self._particle_start + elems[i_el_, 3]

    @property
    def n_elems(self):
        """The number of tetrahedral elements in the mesh."""
        return len(self._elems)

    @property
    def elem_start(self):
        """The starting index of the elements in the global solver."""
        return self._elem_start

    @property
    def elem_end(self):
        """The ending index of the elements in the global solver."""
        return self._elem_start + self.n_elems


@ti.data_oriented
class PBDParticleEntity(ParticleEntity):
    """
    PBD entity represented solely by particles.

    Parameters
    ----------
    scene : Scene
        The simulation scene this entity is part of.
    solver : Solver
        The PBD solver instance managing this entity.
    material : Material
        Material model defining physical properties such as density and compliance.
    morph : Morph
        Morph object specifying shape and initial transform (position and rotation).
    surface : Surface
        Surface or texture representation.
    particle_size : float
        Target size for particle spacing.
    idx : int
        Unique index of this entity within the scene.
    particle_start : int
        Starting index of this entity's particles in the global particle buffer.
    """

    def __init__(self, scene, solver, material, morph, surface, particle_size, idx, particle_start):
        super().__init__(
            scene, solver, material, morph, surface, particle_size, idx, particle_start, need_skinning=False
        )

    def process_input(self, in_backward=False):
        """
        Push position, velocity, and activation target states into the simulator.

        Parameters
        ----------
        in_backward : bool, default=False
            Whether the simulation is in the backward (gradient) pass.
        """
        # TODO: implement this
        pass

    def _add_to_solver_(self):
        self._kernel_add_particles_to_solver(
            f=self._sim.cur_substep_local,
            particles=self._particles,
            rho=self._material.rho,
            mat_type=self.solver.MATS.LIQUID,
            active=self.active,
        )

    @ti.kernel
    def _kernel_add_particles_to_solver(
        self,
        f: ti.i32,
        particles: ti.types.ndarray(),
        rho: ti.float32,
        mat_type: ti.i32,
        active: ti.i32,
    ):
        for i_p_ in range(self._n_particles):
            i_p = i_p_ + self._particle_start
            for j in ti.static(range(3)):
                self.solver.particles_info[i_p].pos_rest[j] = particles[i_p_, j]

            self.solver.particles_info[i_p].mat_type = mat_type
            self.solver.particles_info[i_p].mass = rho
            self.solver.particles_info[i_p].rho_rest = rho

            self.solver.particles_info[i_p].density_relaxation = self.material.density_relaxation
            self.solver.particles_info[i_p].viscosity_relaxation = self.material.viscosity_relaxation

        for i_p_, i_b in ti.ndrange(self.n_particles, self._sim._B):
            i_p = i_p_ + self._particle_start
            for j in ti.static(range(3)):
                self.solver.particles[i_p, i_b].pos[j] = particles[i_p_, j]
            self.solver.particles[i_p, i_b].vel = ti.Vector.zero(gs.ti_float, 3)
            self.solver.particles[i_p, i_b].dpos = ti.Vector.zero(gs.ti_float, 3)
            self.solver.particles[i_p, i_b].free = True

            self.solver.particles_ng[i_p, i_b].active = active

    @property
    def n_fluid_particles(self):
        """The number of fluid particles."""
        return self.n_particles


@ti.data_oriented
class PBDFreeParticleEntity(ParticleEntity):
    """
    PBD-based entity represented by non-physics particles

    Parameters
    ----------
    scene : Scene
        The simulation scene this entity is part of.
    solver : Solver
        The PBD solver instance managing this entity.
    material : Material
        Material model defining physical properties such as density and compliance.
    morph : Morph
        Morph object specifying shape and initial transform (position and rotation).
    surface : Surface
        Surface or texture representation.
    particle_size : float
        Target size for particle spacing.
    idx : int
        Unique index of this entity within the scene.
    particle_start : int
        Starting index of this entity's particles in the global particle buffer.
    """

    def __init__(self, scene, solver, material, morph, surface, particle_size, idx, particle_start):
        super().__init__(
            scene, solver, material, morph, surface, particle_size, idx, particle_start, need_skinning=False
        )

    def process_input(self, in_backward=False):
        """
        Push position, velocity, and activation target states into the simulator.

        Parameters
        ----------
        in_backward : bool, default=False
            Whether the simulation is in the backward (gradient) pass.
        """
        # TODO: implement this
        pass

    def _add_to_solver_(self):
        self._kernel_add_particles_to_solver(
            f=self._sim.cur_substep_local,
            particles=self._particles,
            rho=self._material.rho,
            mat_type=self.solver.MATS.PARTICLE,
            active=self.active,
        )

    @ti.kernel
    def _kernel_add_particles_to_solver(
        self,
        f: ti.i32,
        particles: ti.types.ndarray(),
        rho: ti.float32,
        mat_type: ti.i32,
        active: ti.i32,
    ):
        for i_p_ in range(self.n_particles):
            i_p = i_p_ + self._particle_start
            for j in ti.static(range(3)):
                self.solver.particles_info[i_p].pos_rest[j] = particles[i_p_, j]

            self.solver.particles_info[i_p].mat_type = mat_type
            self.solver.particles_info[i_p].mass = rho
            self.solver.particles_info[i_p].rho_rest = rho

        for i_p_, i_b in ti.ndrange(self.n_particles, self._sim._B):
            i_p = i_p_ + self._particle_start
            for j in ti.static(range(3)):
                self.solver.particles[i_p, i_b].pos[j] = particles[i_p_, j]
            self.solver.particles[i_p, i_b].vel = ti.Vector.zero(gs.ti_float, 3)
            self.solver.particles[i_p, i_b].dpos = ti.Vector.zero(gs.ti_float, 3)
            self.solver.particles[i_p, i_b].free = True

            self.solver.particles_ng[i_p, i_b].active = active
