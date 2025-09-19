import gstaichi as ti
import numpy as np
import torch
import trimesh

import genesis as gs
import genesis.utils.geom as gu
import genesis.utils.mesh as mu
from genesis.engine.entities.base_entity import Entity
from genesis.engine.entities.particle_entity import ParticleEntity


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

    def _add_particles_to_solver(self):
        self._kernel_add_particles_edges_to_solver(
            f=self._scene.sim.cur_substep_local,
            particles=self._particles,
            edges=self._edges,
            edges_len_rest=self._edges_len_rest,
            material_type=self._material_type,
            active=True,
        )

    @ti.kernel
    def _kernel_add_particles_edges_to_solver(
        self,
        f: ti.i32,
        particles: ti.types.ndarray(),
        edges: ti.types.ndarray(),
        edges_len_rest: ti.types.ndarray(),
        material_type: ti.i32,
        active: ti.i32,
    ):
        for i_p_ in range(self.n_particles):
            i_p = i_p_ + self._particle_start
            for i in ti.static(range(3)):
                self.solver.particles_info[i_p].pos_rest[i] = particles[i_p_, i]
            self.solver.particles_info[i_p].material_type = material_type
            self.solver.particles_info[i_p].mass = self._particle_mass
            self.solver.particles_info[i_p].mu_s = self.material.static_friction
            self.solver.particles_info[i_p].mu_k = self.material.kinetic_friction

        for i_p_, i_b in ti.ndrange(self.n_particles, self._sim._B):
            i_p = i_p_ + self._particle_start
            for i in ti.static(range(3)):
                self.solver.particles[i_p, i_b].pos[i] = particles[i_p_, i]
            self.solver.particles[i_p, i_b].vel = ti.Vector.zero(gs.ti_float, 3)
            self.solver.particles[i_p, i_b].dpos = ti.Vector.zero(gs.ti_float, 3)
            self.solver.particles[i_p, i_b].free = True

            self.solver.particles_ng[i_p, i_b].active = ti.cast(active, gs.ti_bool)

        for i_e_ in range(self.n_edges):
            i_e = i_e_ + self._edge_start
            self.solver.edges_info[i_e].stretch_compliance = self.material.stretch_compliance
            self.solver.edges_info[i_e].stretch_relaxation = self.material.stretch_relaxation
            self.solver.edges_info[i_e].len_rest = edges_len_rest[i_e_]
            self.solver.edges_info[i_e].v1 = self._particle_start + edges[i_e_, 0]
            self.solver.edges_info[i_e].v2 = self._particle_start + edges[i_e_, 1]

    def sample(self):
        """
        Sample and preprocess the mesh for the PBD tetrahedral entity.

        Applies transformation from the morph, stores mesh vertices and faces,
        and performs remeshing based on the particle size.
        """
        # We don't use ParticleEntity.sample() because we need to maintain the remeshed self._mesh as well
        pos = np.asarray(self._morph.pos, dtype=gs.np_float)
        quat = np.asarray(self._morph.quat, dtype=gs.np_float)
        self._vmesh.apply_transform(gu.trans_quat_to_T(pos, quat))
        self._vverts = np.asarray(self._vmesh.verts, dtype=gs.np_float)
        self._vfaces = np.asarray(self._vmesh.faces, dtype=gs.np_float)

        self._mesh = self._vmesh.copy()
        self._mesh.remesh(edge_len_abs=self.particle_size, fix=isinstance(self, PBD3DEntity))

    def _reset_grad(self):
        pass

    def add_grad_from_state(self, state):
        pass

    # ------------------------------------------------------------------------------------
    # ---------------------------------- io & control ------------------------------------
    # ------------------------------------------------------------------------------------

    @gs.assert_built
    def _set_particles_pos(self, poss, particles_idx_local=None, envs_idx=None, *, unsafe=False):
        envs_idx = self._scene._sanitize_envs_idx(envs_idx, unsafe=unsafe)
        particles_idx_local = self._sanitize_particles_idx_local(particles_idx_local, envs_idx, unsafe=unsafe)
        poss = self._sanitize_particles_tensor((3,), gs.tc_float, poss, particles_idx_local, envs_idx)
        particles_idx = particles_idx_local + self._particle_start
        self.solver._kernel_set_particles_pos(particles_idx, envs_idx, poss)
        self.solver._kernel_fix_particles(particles_idx, envs_idx)

    def get_particles_pos(self, envs_idx=None, *, unsafe=False):
        envs_idx = self._scene._sanitize_envs_idx(envs_idx, unsafe=unsafe)
        poss = torch.empty((len(envs_idx), self.n_particles, 3), dtype=gs.tc_float, device=gs.device)
        self.solver._kernel_get_particles_pos(self._particle_start, self.n_particles, envs_idx, poss)
        if self._scene.n_envs == 0:
            poss = poss.squeeze(0)
        return poss

    @gs.assert_built
    def _set_particles_vel(self, vels, particles_idx_local=None, envs_idx=None, *, unsafe=False):
        envs_idx = self._scene._sanitize_envs_idx(envs_idx, unsafe=unsafe)
        particles_idx_local = self._sanitize_particles_idx_local(particles_idx_local, envs_idx, unsafe=unsafe)
        vels = self._sanitize_particles_tensor((3,), gs.tc_float, vels, particles_idx_local, envs_idx)
        particles_idx = particles_idx_local + self._particle_start
        self.solver._kernel_set_particles_vel(particles_idx, envs_idx, vels)
        self.solver._kernel_fix_particles(particles_idx, envs_idx)

    def get_particles_vel(self, envs_idx=None, *, unsafe=False):
        envs_idx = self._scene._sanitize_envs_idx(envs_idx, unsafe=unsafe)
        vels = torch.empty((len(envs_idx), self.n_particles, 3), dtype=gs.tc_float, device=gs.device)
        self.solver._kernel_get_particles_vel(self._particle_start, self.n_particles, envs_idx, vels)
        if self._scene.n_envs == 0:
            vels = vels.squeeze(0)
        return vels

    @gs.assert_built
    def _set_particles_active(self, actives, particles_idx_local=None, envs_idx=None, *, unsafe=False):
        envs_idx = self._scene._sanitize_envs_idx(envs_idx, unsafe=unsafe)
        particles_idx_local = self._sanitize_particles_idx_local(particles_idx_local, envs_idx, unsafe=unsafe)
        actives = self._sanitize_particles_tensor((3,), gs.tc_float, actives, particles_idx_local, envs_idx)
        self.solver._kernel_set_particles_active(particles_idx_local + self._particle_start, envs_idx, actives)

    def get_particles_active(self, envs_idx=None, *, unsafe=False):
        envs_idx = self._scene._sanitize_envs_idx(envs_idx, unsafe=unsafe)
        actives = torch.empty((len(envs_idx), self.n_particles), dtype=gs.tc_float, device=gs.device)
        self.solver._kernel_get_particles_active(self._particle_start, self.n_particles, envs_idx, actives)
        if self._scene.n_envs == 0:
            actives = actives.squeeze(0)
        return actives

    @gs.assert_built
    def fix_particles(self, particles_idx_local, envs_idx=None, *, unsafe=False):
        """
        Fix the position of some particles in the simulation.

        Parameters
        ----------
        particles_idx_local : int | array_like, shape (N,)
            Index of the particles relative to this entity.
        envs_idx : None | int | array_like, shape (M,), optional
            The indices of the environments to set. If None, all environments will be set. Defaults to None.
        """
        envs_idx = self._scene._sanitize_envs_idx(envs_idx, unsafe=unsafe)
        particles_idx_local = self._sanitize_particles_idx_local(particles_idx_local, envs_idx, unsafe=unsafe)
        self.solver._kernel_fix_particles(particles_idx_local + self._particle_start, envs_idx)

    @gs.assert_built
    def release_particle(self, particles_idx_local, envs_idx=None, *, unsafe=False):
        """
        Release some of the fixed particles, allowing them to move freely again.

        Parameters
        ----------
        particles_idx_local : int | array_like, shape (N,)
            Index of the particles relative to this entity.
        envs_idx : None | int | array_like, shape (M,), optional
            The indices of the environments to set. If None, all environments will be set. Defaults to None.
        """
        envs_idx = self._scene._sanitize_envs_idx(envs_idx, unsafe=unsafe)
        particles_idx_local = self._sanitize_particles_idx_local(particles_idx_local, envs_idx, unsafe=unsafe)
        self.solver._kernel_release_particle(particles_idx_local + self._particle_start, envs_idx)

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
        self._material_type = self.solver.MATERIAL.CLOTH

    def sample(self):
        """Sample and preprocess the 2D mesh for the PBD cloth-like entity."""
        super().sample()

        if self._vmesh.area < 1e-6:
            gs.raise_exception("Input mesh has zero surface area.")
        self._mass = self._vmesh.area * self.material.rho

        self._particles = np.asarray(self._mesh.verts, dtype=gs.np_float)
        self._edges = np.asarray(self._mesh.get_unique_edges(), dtype=gs.np_int)

        self._particle_mass = self._mass / len(self._particles)

        # Inner edges are two diagonal edges of each quadrilateral formed by adjacent face pairs
        adjacency, inner_edges = trimesh.graph.face_adjacency(mesh=self._mesh.trimesh, return_edges=True)
        v3 = np.sum(self._mesh.faces[adjacency[:, 0]], axis=1) - inner_edges[:, 0] - inner_edges[:, 1]
        v4 = np.sum(self._mesh.faces[adjacency[:, 1]], axis=1) - inner_edges[:, 0] - inner_edges[:, 1]
        self._inner_edges = np.stack([inner_edges[:, 0], inner_edges[:, 1], v3, v4], axis=1, dtype=gs.np_int)

        self._edges_len_rest = np.linalg.norm(
            self._particles[self._edges[:, 0]] - self._particles[self._edges[:, 1]], axis=1
        )
        self._inner_edges_len_rest = np.linalg.norm(
            self._particles[self._inner_edges[:, 2]] - self._particles[self._inner_edges[:, 3]], axis=1
        )
        self._n_particles = len(self._particles)

    def _add_particles_to_solver(self):
        super()._add_particles_to_solver()

        self._kernel_add_particles_air_resistance_to_solver(
            f=self._scene.sim.cur_substep_local,
        )

        self._kernel_add_inner_edges_to_solver(
            f=self._scene.sim.cur_substep_local,
            inner_edges=self._inner_edges,
            inner_edges_len_rest=self._inner_edges_len_rest,
        )

    @ti.kernel
    def _kernel_add_particles_air_resistance_to_solver(self, f: ti.i32):
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

        self._material_type = self.solver.MATERIAL.ELASTIC

    def sample(self):
        super().sample()

        if self._vmesh.volume < 1e-6:
            gs.raise_exception("Input mesh has zero volume.")
        self._mass = self._vmesh.volume * self.material.rho

        tet_cfg = mu.generate_tetgen_config_from_morph(self.morph)
        particles, elems = self._mesh.tetrahedralize(tet_cfg)
        self._particles = particles.astype(gs.np_float, copy=False)
        self._elems = elems.astype(gs.np_int, copy=False)
        self._edges = np.array(
            list(
                set(
                    tuple(sorted((self._elems[i, j], self._elems[i, k])))
                    for i in range(len(self._elems))
                    for j in range(4)
                    for k in range(j + 1, 4)
                )
            ),
            dtype=gs.np_int,
        )
        self._particle_mass = self._mass / len(self._particles)

        self._edges_len_rest = np.linalg.norm(
            self._particles[self._edges[:, 0]] - self._particles[self._edges[:, 1]], axis=1
        )
        self._elems_vol_rest = (
            np.linalg.det(self._particles[self._elems[:, 1:]] - self._particles[self._elems[:, :1]]) / 6.0
        )
        self._n_particles = len(self._particles)

    def _add_particles_to_solver(self):
        super()._add_particles_to_solver()
        self._kernel_add_elems_to_solver(elems=self._elems, elems_vol_rest=self._elems_vol_rest)

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

    def _add_particles_to_solver(self):
        self._kernel_add_particles_to_solver(
            f=self._sim.cur_substep_local,
            particles=self._particles,
            rho=self._material.rho,
            material_type=self.solver.MATERIAL.LIQUID,
            active=self.active,
        )

    @ti.kernel
    def _kernel_add_particles_to_solver(
        self,
        f: ti.i32,
        particles: ti.types.ndarray(),
        rho: ti.float32,
        material_type: ti.i32,
        active: ti.i32,
    ):
        for i_p_ in range(self._n_particles):
            i_p = i_p_ + self._particle_start
            for j in ti.static(range(3)):
                self.solver.particles_info[i_p].pos_rest[j] = particles[i_p_, j]

            self.solver.particles_info[i_p].material_type = material_type
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

    def _add_particles_to_solver(self):
        self._kernel_add_particles_to_solver(
            f=self._sim.cur_substep_local,
            particles=self._particles,
            rho=self._material.rho,
            material_type=self.solver.MATERIAL.PARTICLE,
            active=self.active,
        )

    @ti.kernel
    def _kernel_add_particles_to_solver(
        self,
        f: ti.i32,
        particles: ti.types.ndarray(),
        rho: ti.float32,
        material_type: ti.i32,
        active: ti.i32,
    ):
        for i_p_ in range(self.n_particles):
            i_p = i_p_ + self._particle_start
            for j in ti.static(range(3)):
                self.solver.particles_info[i_p].pos_rest[j] = particles[i_p_, j]

            self.solver.particles_info[i_p].material_type = material_type
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
