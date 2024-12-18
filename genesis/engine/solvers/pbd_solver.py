import math

import numpy as np
import taichi as ti

import genesis as gs
import genesis.utils.geom as gu
from genesis.engine.boundaries import CubeBoundary
from genesis.engine.entities import (
    PBD2DEntity,
    PBD3DEntity,
    PBDFreeParticleEntity,
    PBDParticleEntity,
)
from genesis.engine.states.solvers import PBDSolverState
from genesis.utils.geom import SpatialHasher

from .base_solver import Solver


@ti.data_oriented
class PBDSolver(Solver):
    # ------------------------------------------------------------------------------------
    # --------------------------------- Initialization -----------------------------------
    # ------------------------------------------------------------------------------------

    class MATS(gs.IntEnum):
        CLOTH = 0
        ELASTIC = 1
        LIQUID = 2
        PARTICLE = 3  # non-physcis particles

    def __init__(self, scene, sim, options):
        super().__init__(scene, sim, options)

        # options
        self._upper_bound = np.array(options.upper_bound)
        self._lower_bound = np.array(options.lower_bound)
        self._particle_size = options.particle_size
        self._max_stretch_solver_iterations = options.max_stretch_solver_iterations
        self._max_bending_solver_iterations = options.max_bending_solver_iterations
        self._max_volume_solver_iterations = options.max_volume_solver_iterations
        self._max_density_solver_iterations = options.max_density_solver_iterations
        self._max_viscosity_solver_iterations = options.max_viscosity_solver_iterations

        self._n_vvert_supports = self.scene.vis_options.n_support_neighbors

        # -Neighbours_Setting-
        self.dist_scale = self.particle_radius / 0.4  # @Zhenjia: double check this
        self.h = 1.0
        self.h_2 = self.h**2
        self.h_6 = self.h**6
        self.h_9 = self.h**9

        # -POLY6_KERNEL-
        self.poly6_Coe = 315.0 / (64 * math.pi)

        # -SPIKY_KERNEL-
        self.spiky_Coe = -45.0 / math.pi

        # -LAMBDAS-
        self.lambda_epsilon = 100.0

        # -S_CORR-
        self.S_Corr_delta_q = 0.3
        self.S_Corr_k = 0.0001

        # -Gradient Approx. delta difference-
        self.g_del = 0.01

        self.vorti_epsilon = 0.01

        # spatial hasher
        self.sh = SpatialHasher(
            cell_size=options.hash_grid_cell_size,
            grid_res=options._hash_grid_res,
        )

        # boundary
        self.setup_boundary()

    def setup_boundary(self):
        self.boundary = CubeBoundary(
            lower=self._lower_bound,
            upper=self._upper_bound,
        )

    def init_vvert_fields(self):
        struct_vvert_info = ti.types.struct(
            support_idxs=ti.types.vector(self._n_vvert_supports, gs.ti_int),
            support_weights=ti.types.vector(self._n_vvert_supports, gs.ti_float),
        )
        self.vverts_info = struct_vvert_info.field(shape=max(1, self._n_vverts), layout=ti.Layout.SOA)

        struct_vvert_state_render = ti.types.struct(
            pos=gs.ti_vec3,
            active=gs.ti_int,
        )
        self.vverts_render = struct_vvert_state_render.field(shape=max(1, self._n_vverts), layout=ti.Layout.SOA)

    def init_particle_fields(self):
        # particles information (static)
        struct_particle_info = ti.types.struct(
            mass=gs.ti_float,
            pos_rest=gs.ti_vec3,
            rho_rest=gs.ti_float,
            mat_type=gs.ti_int,
            mu_s=gs.ti_float,
            mu_k=gs.ti_float,
            air_resistance=gs.ti_float,
            density_relaxation=gs.ti_float,
            viscosity_relaxation=gs.ti_float,
        )
        # particles state (dynamic)
        struct_particle_state = ti.types.struct(
            free=gs.ti_int,  # if not free, the particle is not affected by internal forces and solely controlled by external user until released
            pos=gs.ti_vec3,  # position
            ipos=gs.ti_vec3,  # initial position
            dpos=gs.ti_vec3,  # delta position
            vel=gs.ti_vec3,  # velocity
            lam=gs.ti_float,
            rho=gs.ti_float,
        )

        # dynamic particle state without gradient
        struct_particle_state_ng = ti.types.struct(
            reordered_idx=gs.ti_int,
            active=gs.ti_int,
        )

        # single frame particle state for rendering
        struct_particle_state_render = ti.types.struct(
            pos=gs.ti_vec3,
            vel=gs.ti_vec3,
            active=gs.ti_int,
        )

        self.particles_info = struct_particle_info.field(shape=self._n_particles, layout=ti.Layout.SOA)
        self.particles_info_reordered = struct_particle_info.field(shape=self._n_particles, layout=ti.Layout.SOA)

        self.particles = struct_particle_state.field(shape=self._n_particles, layout=ti.Layout.SOA)
        self.particles_reordered = struct_particle_state.field(shape=self._n_particles, layout=ti.Layout.SOA)

        self.particles_ng = struct_particle_state_ng.field(shape=self._n_particles, layout=ti.Layout.SOA)
        self.particles_ng_reordered = struct_particle_state_ng.field(shape=self._n_particles, layout=ti.Layout.SOA)

        self.particles_render = struct_particle_state_render.field(shape=self._n_particles, layout=ti.Layout.SOA)

    def init_edge_fields(self):
        # edges information for stretch. edge: (v1, v2)
        struct_edge_info = ti.types.struct(
            len_rest=gs.ti_float,
            stretch_compliance=gs.ti_float,
            stretch_relaxation=gs.ti_float,
            v1=gs.ti_int,
            v2=gs.ti_int,
        )
        self.edges_info = struct_edge_info.field(shape=max(1, self._n_edges), layout=ti.Layout.SOA)

        # inner edges information for bending. edge: (v1, v2), adjacent faces: (v1, v2, v3) and (v1, v2, v4)
        struct_inner_edge_info = ti.types.struct(
            len_rest=gs.ti_float,
            bending_compliance=gs.ti_float,
            bending_relaxation=gs.ti_float,
            v1=gs.ti_int,
            v2=gs.ti_int,
            v3=gs.ti_int,
            v4=gs.ti_int,
        )
        self.inner_edges_info = struct_inner_edge_info.field(shape=max(1, self._n_inner_edges), layout=ti.Layout.SOA)

    def init_elem_fields(self):
        struct_elem_info = ti.types.struct(
            vol_rest=gs.ti_float,
            volume_compliance=gs.ti_float,
            volume_relaxation=gs.ti_float,
            v1=gs.ti_int,
            v2=gs.ti_int,
            v3=gs.ti_int,
            v4=gs.ti_int,
        )
        self.elems_info = struct_elem_info.field(shape=max(1, self._n_elems), layout=ti.Layout.SOA)

    def init_ckpt(self):
        self._ckpt = dict()

    def build(self):
        self._n_particles = self.n_particles
        self._n_fluid_particles = self.n_fluid_particles
        self._n_edges = self.n_edges
        self._n_inner_edges = self.n_inner_edges
        self._n_elems = self.n_elems
        self._n_vverts = self.n_vverts
        self._n_vfaces = self.n_vfaces

        if self.is_active():
            self.sh.build()

            self.init_particle_fields()
            self.init_edge_fields()
            self.init_elem_fields()
            self.init_vvert_fields()

            self.init_ckpt()

            for entity in self._entities:
                entity._add_to_solver()

    # ------------------------------------------------------------------------------------
    # -------------------------------------- misc ----------------------------------------
    # ------------------------------------------------------------------------------------

    def add_entity(self, idx, material, morph, surface):
        if isinstance(material, gs.materials.PBD.Cloth):
            entity = PBD2DEntity(
                scene=self.scene,
                solver=self,
                material=material,
                morph=morph,
                surface=surface,
                particle_size=self._particle_size,
                idx=idx,
                particle_start=self.n_particles,
                edge_start=self.n_edges,
                inner_edge_start=self.n_inner_edges,
                vvert_start=self.n_vverts,
                vface_start=self.n_vfaces,
            )

        elif isinstance(material, gs.materials.PBD.Elastic):
            entity = PBD3DEntity(
                scene=self.scene,
                solver=self,
                material=material,
                morph=morph,
                surface=surface,
                particle_size=self._particle_size,
                idx=idx,
                particle_start=self.n_particles,
                edge_start=self.n_edges,
                elem_start=self.n_elems,
                vvert_start=self.n_vverts,
                vface_start=self.n_vfaces,
            )

        elif isinstance(material, gs.materials.PBD.Liquid):
            entity = PBDParticleEntity(
                scene=self.scene,
                solver=self,
                material=material,
                morph=morph,
                surface=surface,
                particle_size=self._particle_size,
                idx=idx,
                particle_start=self.n_particles,
            )

        elif isinstance(material, gs.materials.PBD.Particle):
            entity = PBDFreeParticleEntity(
                scene=self.scene,
                solver=self,
                material=material,
                morph=morph,
                surface=surface,
                particle_size=self._particle_size,
                idx=idx,
                particle_start=self.n_particles,
            )

        else:
            raise NotImplementedError()

        self._entities.append(entity)

        return entity

    def is_active(self):
        return self._n_particles > 0

    # ------------------------------------------------------------------------------------
    # ------------------------------------- utils ----------------------------------------
    # ------------------------------------------------------------------------------------

    @ti.func
    def poly6(self, dist):
        # dist is a VECTOR
        result = 0.0
        d = dist.norm() / self.dist_scale
        if 0 < d < self.h:
            rhs = (self.h_2 - d * d) * (self.h_2 - d * d) * (self.h_2 - d * d)
            result = self.poly6_Coe * rhs / self.h_9
        return result

    @ti.func
    def poly6_scalar(self, dist):
        # dist is a SCALAR
        result = 0.0
        d = dist
        if 0 < d < self.h:
            rhs = (self.h_2 - d * d) * (self.h_2 - d * d) * (self.h_2 - d * d)
            result = self.poly6_Coe * rhs / self.h_9
        return result

    @ti.func
    def spiky(self, dist):
        # dist is a VECTOR
        result = ti.Vector.zero(gs.ti_float, 3)
        d = dist.norm() / self.dist_scale
        if 0 < d < self.h:
            m = (self.h - d) * (self.h - d)
            result = (self.spiky_Coe * m / (self.h_6 * d)) * dist / self.dist_scale
        return result

    @ti.func
    def S_Corr(self, dist):
        upper = self.poly6(dist)
        lower = self.poly6_scalar(self.S_Corr_delta_q)
        m = upper / lower
        return -1.0 * self.S_Corr_k * m * m * m * m

    # ------------------------------------------------------------------------------------
    # ----------------------------------- simulation -------------------------------------
    # ------------------------------------------------------------------------------------
    @ti.kernel
    def _kernel_store_initial_pos(self, f: ti.i32):
        for i in range(self._n_particles):
            self.particles[i].ipos = self.particles[i].pos

    @ti.kernel
    def _kernel_reorder_particles(self, f: ti.i32):
        self.sh.compute_reordered_idx(
            self._n_particles, self.particles.pos, self.particles_ng.active, self.particles_ng.reordered_idx
        )

        # copy to reordered
        self.particles_ng_reordered.active.fill(0)
        for i in range(self._n_particles):
            if self.particles_ng[i].active:
                reordered_idx = self.particles_ng[i].reordered_idx

                self.particles_reordered[reordered_idx] = self.particles[i]
                self.particles_info_reordered[reordered_idx] = self.particles_info[i]
                self.particles_ng_reordered[reordered_idx].active = self.particles_ng[i].active

    @ti.kernel
    def _kernel_apply_external_force(self, f: ti.i32, t: ti.f32):
        for i in range(self._n_particles):
            if self.particles[i].free:
                # gravity
                self.particles[i].vel = self.particles[i].vel + self._gravity[None] * self._substep_dt

                # external force fields
                acc = ti.Vector.zero(gs.ti_float, 3)
                for i_ff in ti.static(range(len(self._ffs))):
                    acc += self._ffs[i_ff].get_acc(self.particles[i].pos, self.particles[i].vel, t, i)
                self.particles[i].vel = self.particles[i].vel + acc * self._substep_dt

                if self.particles_info[i].mat_type == self.MATS.CLOTH:
                    f_air_resistance = (
                        self.particles_info[i].air_resistance * self.particles[i].vel.norm() * self.particles[i].vel
                    )
                    self.particles[i].vel = (
                        self.particles[i].vel - f_air_resistance / self.particles_info[i].mass * self._substep_dt
                    )
            self.particles[i].pos = self.particles[i].pos + self.particles[i].vel * self._substep_dt

    @ti.kernel
    def _kernel_solve_stretch(self, f: ti.i32):
        for _ in ti.static(range(self._max_stretch_solver_iterations)):
            for i_e in range(self._n_edges):
                v1 = self.edges_info[i_e].v1
                v2 = self.edges_info[i_e].v2

                w1 = 1.0 / self.particles_info[v1].mass * self.particles[v1].free
                w2 = 1.0 / self.particles_info[v2].mass * self.particles[v2].free
                n = self.particles[v1].pos - self.particles[v2].pos
                C = n.norm() - self.edges_info[i_e].len_rest
                alpha = self.edges_info[i_e].stretch_compliance / (self._substep_dt**2)
                dp = -C / (w1 + w2 + alpha) * n / n.norm(gs.EPS) * self.edges_info[i_e].stretch_relaxation
                self.particles[v1].dpos += dp * w1
                self.particles[v2].dpos -= dp * w2

            for i in range(self._n_particles):
                if self.particles[i].free and self.particles_info[i].mat_type != self.MATS.PARTICLE:
                    self.particles[i].pos = self.particles[i].pos + self.particles[i].dpos
                    self.particles[i].dpos.fill(0)

    # @ti.kernel
    # def _kernel_solve_bending(self, f: ti.i32):
    #     for i_ie in range(self._n_inner_edges):
    #         v1 = self.inner_edges_info[i_ie].v3
    #         v2 = self.inner_edges_info[i_ie].v4

    #         w1 = 1.0 / self.particles_info[v1].mass * self.particles[v1].free
    #         w2 = 1.0 / self.particles_info[v2].mass * self.particles[v2].free
    #         n = self.particles[v1].pos - self.particles[v2].pos
    #         C = n.norm() - self.inner_edges_info[i_ie].len_rest
    #         alpha = self.inner_edges_info[i_ie].bending_compliance / (self._substep_dt ** 2)
    #         dp = -C / (w1 + w2 + alpha) * n / n.norm(gs.EPS) * self.inner_edges_info[i_ie].bending_relaxation
    #         self.particles[v1].dpos += dp * w1
    #         self.particles[v2].dpos -= dp * w2

    #     for i in range(self._n_particles):
    #         if self.particles[i].free:
    #             self.particles[i].pos = self.particles[i].pos + self.particles[i].dpos
    #             self.particles[i].dpos.fill(0)

    @ti.kernel
    def _kernel_solve_bending(self, f: ti.i32):
        for _ in ti.static(range(self._max_bending_solver_iterations)):
            for i_ie in range(self._n_inner_edges):  # 140 - 142
                v1 = self.inner_edges_info[i_ie].v1
                v2 = self.inner_edges_info[i_ie].v2
                v3 = self.inner_edges_info[i_ie].v3
                v4 = self.inner_edges_info[i_ie].v4

                w1 = 1.0 / self.particles_info[v1].mass * self.particles[v1].free
                w2 = 1.0 / self.particles_info[v2].mass * self.particles[v2].free
                w3 = 1.0 / self.particles_info[v3].mass * self.particles[v3].free
                w4 = 1.0 / self.particles_info[v4].mass * self.particles[v4].free

                if w1 + w2 + w3 + w4 > 0.0:
                    # https://matthias-research.github.io/pages/publications/posBasedDyn.pdf
                    # Appendix A: Bending Constraint Projection
                    p2 = self.particles[v2].pos - self.particles[v1].pos
                    p3 = self.particles[v3].pos - self.particles[v1].pos
                    p4 = self.particles[v4].pos - self.particles[v1].pos
                    l23 = p2.cross(p3).norm()
                    l24 = p2.cross(p4).norm()
                    n1 = p2.cross(p3) / l23
                    n2 = p2.cross(p4) / l24
                    d = ti.math.clamp(n1.dot(n2), -1.0, 1.0)

                    q3 = (p2.cross(n2) + n1.cross(p2) * d) / l23  # eq. (25)
                    q4 = (p2.cross(n1) + n2.cross(p2) * d) / l24  # eq. (26)
                    q2 = -(p3.cross(n2) + n1.cross(p3) * d) / l23 - (p4.cross(n1) + n2.cross(p4) * d) / l24  # eq. (27)
                    q1 = -q2 - q3 - q4
                    # eq. (29)
                    sum_wq = w1 * q1.norm_sqr() + w2 * q2.norm_sqr() + w3 * q3.norm_sqr() + w4 * q4.norm_sqr()
                    constraint = ti.acos(d) - ti.acos(-1.0)

                    # XPBD
                    alpha = self.inner_edges_info[i_ie].bending_compliance / (self._substep_dt**2)
                    constraint = (
                        -ti.sqrt(1 - d**2)
                        * constraint
                        / (sum_wq + alpha)
                        * self.inner_edges_info[i_ie].bending_relaxation
                    )

                    self.particles[v1].dpos += w1 * constraint * q1
                    self.particles[v2].dpos += w2 * constraint * q2
                    self.particles[v3].dpos += w3 * constraint * q3
                    self.particles[v4].dpos += w4 * constraint * q4

            for i in range(self._n_particles):
                if self.particles[i].free and self.particles_info[i].mat_type != self.MATS.PARTICLE:
                    self.particles[i].pos = self.particles[i].pos + self.particles[i].dpos
                    self.particles[i].dpos.fill(0)

    @ti.kernel
    def _kernel_solve_volume(self, f: ti.i32):
        for _ in ti.static(range(self._max_volume_solver_iterations)):
            for i_el in range(self._n_elems):
                v1 = self.elems_info[i_el].v1
                v2 = self.elems_info[i_el].v2
                v3 = self.elems_info[i_el].v3
                v4 = self.elems_info[i_el].v4

                p1 = self.particles[v1].pos
                p2 = self.particles[v2].pos
                p3 = self.particles[v3].pos
                p4 = self.particles[v4].pos

                grad1 = (p4 - p2).cross(p3 - p2) / 6.0
                grad2 = (p3 - p1).cross(p4 - p1) / 6.0
                grad3 = (p4 - p1).cross(p2 - p1) / 6.0
                grad4 = (p2 - p1).cross(p3 - p1) / 6.0

                w1 = 1.0 / self.particles_info[v1].mass * grad1.norm_sqr() * self.particles[v1].free
                w2 = 1.0 / self.particles_info[v2].mass * grad2.norm_sqr() * self.particles[v2].free
                w3 = 1.0 / self.particles_info[v3].mass * grad3.norm_sqr() * self.particles[v3].free
                w4 = 1.0 / self.particles_info[v4].mass * grad4.norm_sqr() * self.particles[v4].free

                if w1 + w2 + w3 + w4 > 0.0:
                    vol = gu.ti_tet_vol(p1, p2, p3, p4)
                    C = vol - self.elems_info[i_el].vol_rest
                    alpha = self.elems_info[i_el].volume_compliance / (self._substep_dt**2)
                    s = -C / (w1 + w2 + w3 + w4 + alpha) * self.elems_info[i_el].volume_relaxation

                    self.particles[v1].dpos += s * w1 * grad1
                    self.particles[v2].dpos += s * w2 * grad2
                    self.particles[v3].dpos += s * w3 * grad3
                    self.particles[v4].dpos += s * w4 * grad4

            for i in range(self._n_particles):
                if self.particles[i].free and self.particles_info[i].mat_type != self.MATS.PARTICLE:
                    self.particles[i].pos = self.particles[i].pos + self.particles[i].dpos
                    self.particles[i].dpos.fill(0)

    @ti.func
    def _func_solve_collision(self, i, j):
        """j -> i"""

        cur_dist = (self.particles_reordered[i].pos - self.particles_reordered[j].pos).norm(gs.EPS)
        rest_dist = (self.particles_info_reordered[i].pos_rest - self.particles_info_reordered[j].pos_rest).norm(gs.EPS)
        target_dist = self._particle_size  # target particle distance is 2 * particle radius, i.e. particle_size
        if cur_dist < target_dist and rest_dist > target_dist:
            wi = 1.0 / self.particles_info_reordered[i].mass * self.particles_reordered[i].free
            wj = 1.0 / self.particles_info_reordered[j].mass * self.particles_reordered[j].free
            n = (self.particles_reordered[i].pos - self.particles_reordered[j].pos) / cur_dist

            ### resolve collision ###
            self.particles_reordered[i].dpos += wi / (wi + wj) * (target_dist - cur_dist) * n

            ### apply friction ###
            # https://mmacklin.com/uppfrta_preprint.pdf
            # equation (23)
            dv = (self.particles_reordered[i].pos - self.particles_reordered[i].ipos) - (
                self.particles_reordered[j].pos - self.particles_reordered[j].ipos
            )
            dpos = -(dv - n * n.dot(dv))
            # equation (24)
            d = target_dist - cur_dist
            mu_s = ti.max(self.particles_info_reordered[i].mu_s, self.particles_info_reordered[j].mu_s)
            mu_k = ti.max(self.particles_info_reordered[i].mu_k, self.particles_info_reordered[j].mu_k)
            if dpos.norm() < mu_s * d:
                self.particles_reordered[i].dpos += wi / (wi + wj) * dpos
            else:
                self.particles_reordered[i].dpos += wi / (wi + wj) * dpos * ti.min(1.0, mu_k * d / dpos.norm(gs.EPS))

    @ti.kernel
    def _kernel_solve_collision(self, f: ti.i32):
        for i in range(self._n_particles):
            if self.particles_info_reordered[i].mat_type != self.MATS.PARTICLE:
                base = self.sh.pos_to_grid(self.particles_reordered[i].pos)
                for offset in ti.grouped(ti.ndrange((-1, 2), (-1, 2), (-1, 2))):
                    slot_idx = self.sh.grid_to_slot(base + offset)
                    for j in range(
                        self.sh.slot_start[slot_idx], self.sh.slot_size[slot_idx] + self.sh.slot_start[slot_idx]
                    ):
                        if i != j and not (
                            self.particles_info_reordered[i].mat_type == self.MATS.LIQUID
                            and self.particles_info_reordered[j].mat_type == self.MATS.LIQUID
                        ):
                            self._func_solve_collision(i, j)

        for i in range(self._n_particles):
            if self.particles_reordered[i].free and self.particles_info_reordered[i].mat_type != self.MATS.PARTICLE:
                self.particles_reordered[i].pos = self.particles_reordered[i].pos + self.particles_reordered[i].dpos
                self.particles_reordered[i].dpos.fill(0)

    @ti.kernel
    def _kernel_solve_boundary_collision(self, f: ti.i32):
        for i in range(self._n_particles):
            # boundary is enforced regardless of whether free
            pos_new, vel_new = self.boundary.impose_pos_vel(self.particles[i].pos, self.particles[i].vel)
            self.particles[i].pos = pos_new
            self.particles[i].vel = vel_new

    @ti.kernel
    def _kernel_solve_density(self, f: ti.i32):
        for _ in ti.static(range(self._max_density_solver_iterations)):
            # ---Calculate lambdas---
            for i in range(self._n_particles):
                if self.particles_info_reordered[i].mat_type == self.MATS.LIQUID:
                    pos_i = self.particles_reordered[i].pos
                    base = self.sh.pos_to_grid(pos_i)
                    lower_sum = 0.0
                    rho = 0.0
                    spiky_i = ti.Vector.zero(gs.ti_float, 3)
                    for offset in ti.grouped(ti.ndrange((-1, 2), (-1, 2), (-1, 2))):
                        slot_idx = self.sh.grid_to_slot(base + offset)
                        for j in range(
                            self.sh.slot_start[slot_idx], self.sh.slot_size[slot_idx] + self.sh.slot_start[slot_idx]
                        ):

                            pos_j = self.particles_reordered[j].pos
                            # ---Poly6---
                            rho += self.poly6(pos_i - pos_j) * self.particles_info_reordered[j].mass
                            # ---Spiky---
                            s = self.spiky(pos_i - pos_j) / self.particles_info_reordered[i].rho_rest
                            spiky_i += s
                            lower_sum += s.dot(s)
                    constraint = (rho / self.particles_info_reordered[i].rho_rest) - 1.0
                    lower_sum += spiky_i.dot(spiky_i)
                    self.particles_reordered[i].lam = -1.0 * (constraint / (lower_sum + self.lambda_epsilon))

            # ---Calculate delta pos---
            for i in range(self._n_particles):
                if self.particles_info_reordered[i].mat_type == self.MATS.LIQUID:
                    pos_i = self.particles_reordered[i].pos
                    base = self.sh.pos_to_grid(pos_i)
                    for offset in ti.grouped(ti.ndrange((-1, 2), (-1, 2), (-1, 2))):
                        slot_idx = self.sh.grid_to_slot(base + offset)
                        for j in range(
                            self.sh.slot_start[slot_idx], self.sh.slot_size[slot_idx] + self.sh.slot_start[slot_idx]
                        ):
                            if i != j:
                                pos_j = self.particles_reordered[j].pos
                                # ---S_Corr---
                                scorr = self.S_Corr(pos_i - pos_j)
                                left = self.particles_reordered[i].lam + self.particles_reordered[j].lam + scorr
                                right = self.spiky(pos_i - pos_j)
                                self.particles_reordered[i].dpos = (
                                    self.particles_reordered[i].dpos
                                    + left
                                    * right
                                    / self.particles_info_reordered[i].rho_rest
                                    * self.dist_scale
                                    * self.particles_info_reordered[i].density_relaxation
                                )

            for i in range(self._n_particles):
                if self.particles_info_reordered[i].mat_type == self.MATS.LIQUID and self.particles_reordered[i].free:
                    self.particles_reordered[i].pos = self.particles_reordered[i].pos + self.particles_reordered[i].dpos
                    self.particles_reordered[i].dpos.fill(0)

    @ti.kernel
    def _kernel_solve_viscosity(self, f: ti.i32):
        for _ in ti.static(range(self._max_viscosity_solver_iterations)):
            for i in range(self._n_particles):
                if self.particles_info_reordered[i].mat_type == self.MATS.LIQUID:
                    pos_i = self.particles_reordered[i].pos
                    base = self.sh.pos_to_grid(pos_i)
                    xsph_sum = ti.Vector.zero(gs.ti_float, 3)
                    omega_sum = ti.Vector.zero(gs.ti_float, 3)
                    # -For Gradient Approx.-
                    dx_sum = ti.Vector.zero(gs.ti_float, 3)
                    dy_sum = ti.Vector.zero(gs.ti_float, 3)
                    dz_sum = ti.Vector.zero(gs.ti_float, 3)
                    n_dx_sum = ti.Vector.zero(gs.ti_float, 3)
                    n_dy_sum = ti.Vector.zero(gs.ti_float, 3)
                    n_dz_sum = ti.Vector.zero(gs.ti_float, 3)
                    dx = ti.Vector([self.g_del, 0.0, 0.0], dt=gs.ti_float)
                    dy = ti.Vector([0.0, self.g_del, 0.0], dt=gs.ti_float)
                    dz = ti.Vector([0.0, 0.0, self.g_del], dt=gs.ti_float)

                    for offset in ti.grouped(ti.ndrange((-1, 2), (-1, 2), (-1, 2))):
                        slot_idx = self.sh.grid_to_slot(base + offset)
                        for j in range(
                            self.sh.slot_start[slot_idx], self.sh.slot_size[slot_idx] + self.sh.slot_start[slot_idx]
                        ):

                            pos_j = self.particles_reordered[j].pos
                            v_ij = (self.particles_reordered[j].pos - self.particles_reordered[j].ipos) - (
                                self.particles_reordered[i].pos - self.particles_reordered[i].ipos
                            )

                            dist = pos_i - pos_j
                            # ---Vorticity---
                            omega_sum += v_ij.cross(self.spiky(dist))
                            # -Gradient Approx.-
                            dx_sum += v_ij.cross(self.spiky(dist + dx))
                            dy_sum += v_ij.cross(self.spiky(dist + dy))
                            dz_sum += v_ij.cross(self.spiky(dist + dz))
                            n_dx_sum += v_ij.cross(self.spiky(dist - dx))
                            n_dy_sum += v_ij.cross(self.spiky(dist - dy))
                            n_dz_sum += v_ij.cross(self.spiky(dist - dz))
                            # ---Viscosity---
                            poly = self.poly6(dist)
                            xsph_sum += poly * v_ij

                    # # ---Vorticity---
                    # n_x = (dx_sum.norm() - n_dx_sum.norm()) / (2 * self.g_del)
                    # n_y = (dy_sum.norm() - n_dy_sum.norm()) / (2 * self.g_del)
                    # n_z = (dz_sum.norm() - n_dz_sum.norm()) / (2 * self.g_del)
                    # n = ti.Vector([n_x, n_y, n_z])
                    # big_n = n.normalized()
                    # if not omega_sum.norm() == 0.0:
                    #     vorticity[p] = vorti_epsilon * big_n.cross(omega_sum)

                    # ---Viscosity---
                    self.particles_reordered[i].dpos = (
                        self.particles_reordered[i].dpos
                        + xsph_sum * self.particles_info_reordered[i].viscosity_relaxation
                    )

            for i in range(self._n_particles):
                if self.particles_info_reordered[i].mat_type == self.MATS.LIQUID and self.particles_reordered[i].free:
                    self.particles_reordered[i].pos = self.particles_reordered[i].pos + self.particles_reordered[i].dpos
                    self.particles_reordered[i].dpos.fill(0)

    @ti.kernel
    def _kernel_compute_velocity(self, f: ti.i32):
        for i in range(self._n_particles):
            self.particles_reordered[i].vel = (
                self.particles_reordered[i].pos - self.particles_reordered[i].ipos
            ) / self._substep_dt

    @ti.kernel
    def _kernel_copy_from_reordered(self, f: ti.i32):
        for i in range(self._n_particles):
            if self.particles_ng[i].active:
                self.particles[i] = self.particles_reordered[self.particles_ng[i].reordered_idx]

    # ------------------------------------------------------------------------------------
    # ------------------------------------ stepping --------------------------------------
    # ------------------------------------------------------------------------------------

    def process_input(self, in_backward=False):
        for entity in self._entities:
            entity.process_input(in_backward=in_backward)

    def process_input_grad(self):
        pass

    def substep_pre_coupling(self, f):
        if self.is_active():
            self._kernel_store_initial_pos(f)
            self._kernel_apply_external_force(f, self._sim.cur_t)

            # topology constraints (doesn't require spatial hashing)
            if self._n_edges > 0:
                self._kernel_solve_stretch(f)

            if self._n_inner_edges > 0:
                self._kernel_solve_bending(f)

            if self._n_elems > 0:
                self._kernel_solve_volume(f)

            # perform spatial hashing
            self._kernel_reorder_particles(f)

            # spatial constraints
            if self._n_particles > 0:
                self._kernel_solve_density(f)
                self._kernel_solve_viscosity(f)

            self._kernel_solve_collision(f)

            # compute effective velocity
            self._kernel_compute_velocity(f)

    def substep_pre_coupling_grad(self, f):
        pass

    def substep_post_coupling(self, f):
        if self.is_active():
            self._kernel_copy_from_reordered(f)

            # boundary collision
            self._kernel_solve_boundary_collision(f)

    def substep_post_coupling_grad(self, f):
        pass

    def reset_grad(self):
        pass

    # ------------------------------------------------------------------------------------
    # ------------------------------------ gradient --------------------------------------
    # ------------------------------------------------------------------------------------

    def collect_output_grads(self):
        """
        Collect gradients from downstream queried states.
        """
        pass

    def add_grad_from_state(self, state):
        pass

    def save_ckpt(self, ckpt_name):
        pass

    def load_ckpt(self, ckpt_name):
        pass

    # ------------------------------------------------------------------------------------
    # --------------------------------------- io -----------------------------------------
    # ------------------------------------------------------------------------------------

    @ti.kernel
    def _kernel_set_particles_pos(
        self,
        f: ti.i32,
        particle_start: ti.i32,
        n_particles: ti.i32,
        pos: ti.types.ndarray(),
    ):
        for i in range(n_particles):
            i_global = i + particle_start
            for k in ti.static(range(3)):
                self.particles[i_global].pos[k] = pos[i, k]

            # we reset vel when directly setting pos
            self.particles[i_global].vel = ti.Vector.zero(gs.ti_float, 3)

    @ti.kernel
    def _kernel_set_particles_vel(
        self,
        f: ti.i32,
        particle_start: ti.i32,
        n_particles: ti.i32,
        vel: ti.types.ndarray(),
    ):
        for i in range(n_particles):
            i_global = i + particle_start
            for k in ti.static(range(3)):
                self.particles[i_global].vel[k] = vel[i, k]

    @ti.kernel
    def _kernel_set_particles_active(
        self,
        f: ti.i32,
        particle_start: ti.i32,
        n_particles: ti.i32,
        active: ti.i32,
    ):
        for i in range(n_particles):
            i_global = i + particle_start
            self.particles_ng[i_global].active = active

    @ti.kernel
    def _kernel_get_frame(self, f: ti.i32, pos: ti.types.ndarray(), vel: ti.types.ndarray(), free: ti.types.ndarray()):
        for i in range(self._n_particles):
            for j in ti.static(range(3)):
                pos[i, j] = self.particles[i].pos[j]
                vel[i, j] = self.particles[i].vel[j]
            free[i] = self.particles[i].free

    @ti.kernel
    def _kernel_set_frame(self, f: ti.i32, pos: ti.types.ndarray(), vel: ti.types.ndarray(), free: ti.types.ndarray()):
        for i in range(self._n_particles):
            for j in ti.static(range(3)):
                self.particles[i].pos[j] = pos[i, j]
                self.particles[i].vel[j] = vel[i, j]
            self.particles[i].free = free[i]

    def get_state(self, f):
        if self.is_active():
            state = PBDSolverState(self.scene)
            self._kernel_get_frame(f, state.pos, state.vel, state.free)
        else:
            state = None
        return state

    def set_state(self, f, state):
        if self.is_active():
            self._kernel_set_frame(f, state.pos, state.vel, state.free)

    @ti.kernel
    def _kernel_update_render_fields(self, f: ti.i32):
        for i in range(self._n_particles):
            if self.particles_ng[i].active:
                self.particles_render[i].pos = self.particles[i].pos
                self.particles_render[i].vel = self.particles[i].vel
            else:
                self.particles_render[i].pos = gu.ti_nowhere()
            self.particles_render[i].active = self.particles_ng[i].active

        for i in range(self._n_vverts):
            vvert_pos = ti.Vector.zero(gs.ti_float, 3)
            for j in range(self._n_vvert_supports):
                vvert_pos += (
                    self.particles[self.vverts_info.support_idxs[i][j]].pos * self.vverts_info.support_weights[i][j]
                )
            self.vverts_render[i].pos = vvert_pos
            self.vverts_render[i].active = self.particles_render[self.vverts_info.support_idxs[i][0]].active

    def update_render_fields(self):
        self._kernel_update_render_fields(self.sim.cur_substep_local)

    @gs.assert_built
    def fix_particle(self, particle_idx):
        self._kernel_fix_particle(particle_idx)

    @ti.kernel
    def _kernel_fix_particle(self, particle_idx: ti.i32):
        self.particles[particle_idx].free = 0

    @gs.assert_built
    def set_particle_position(self, particle_idx, pos):
        self._kernel_set_particle_position(particle_idx, pos)

    @ti.kernel
    def _kernel_set_particle_position(self, particle_idx: ti.i32, pos: ti.types.ndarray()):
        for i in range(3):
            self.particles[particle_idx].pos[i] = pos[i]
        self.particles[particle_idx].free = 0

    @gs.assert_built
    def set_particle_velocity(self, particle_idx, vel):
        self._kernel_set_particle_velocity(particle_idx, vel)

    @ti.kernel
    def _kernel_set_particle_velocity(self, particle_idx: ti.i32, vel: ti.types.ndarray()):
        for i in range(3):
            self.particles[particle_idx].vel[i] = vel[i]
        self.particles[particle_idx].free = 0

    @gs.assert_built
    def release_particle(self, particle_idx):
        self._kernel_release_particle(particle_idx)

    @ti.kernel
    def _kernel_release_particle(self, particle_idx: ti.i32):
        self.particles[particle_idx].free = 1

    # ------------------------------------------------------------------------------------
    # ----------------------------------- properties -------------------------------------
    # ------------------------------------------------------------------------------------
    @property
    def n_particles(self):
        if self.is_built:
            return self._n_particles
        else:
            return sum([entity.n_particles for entity in self._entities])

    @property
    def n_fluid_particles(self):
        if self.is_built:
            return self._n_fluid_particles
        else:
            return sum(
                [entity.n_fluid_particles if hasattr(entity, "n_fluid_particles") else 0 for entity in self._entities]
            )

    @property
    def n_edges(self):
        if self.is_built:
            return self._n_edges
        else:
            return sum([entity.n_edges if hasattr(entity, "n_edges") else 0 for entity in self._entities])

    @property
    def n_inner_edges(self):
        if self.is_built:
            return self._n_inner_edges
        else:
            return sum([entity.n_inner_edges if hasattr(entity, "n_inner_edges") else 0 for entity in self._entities])

    @property
    def n_elems(self):
        if self.is_built:
            return self._n_elems
        else:
            return sum([entity.n_elems if hasattr(entity, "n_elems") else 0 for entity in self._entities])

    @property
    def n_vverts(self):
        if self.is_built:
            return self._n_vverts
        else:
            return sum([entity.n_vverts if hasattr(entity, "n_vverts") else 0 for entity in self._entities])

    @property
    def n_vfaces(self):
        if self.is_built:
            return self._n_vfaces
        else:
            return sum([entity.n_vfaces if hasattr(entity, "n_vfaces") else 0 for entity in self._entities])

    @property
    def particle_size(self):
        return self._particle_size

    @property
    def particle_radius(self):
        return self._particle_size / 2.0

    @property
    def hash_grid_res(self):
        return self.sh.grid_res

    @property
    def hash_grid_cell_size(self):
        return self.sh.cell_size

    @property
    def upper_bound(self):
        return self._upper_bound

    @property
    def lower_bound(self):
        return self._lower_bound
