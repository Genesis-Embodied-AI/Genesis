import numpy as np
import taichi as ti
import torch

import genesis as gs
import genesis.utils.geom as gu
from genesis.engine.boundaries import CubeBoundary
from genesis.engine.entities import MPMEntity
from genesis.engine.states.solvers import MPMSolverState

from .base_solver import Solver


@ti.data_oriented
class MPMSolver(Solver):
    # ------------------------------------------------------------------------------------
    # --------------------------------- Initialization -----------------------------------
    # ------------------------------------------------------------------------------------

    def __init__(self, scene, sim, options):
        super().__init__(scene, sim, options)

        # options
        self._grid_density = options.grid_density
        self._particle_size = options.particle_size
        self._upper_bound = np.array(options.upper_bound)
        self._lower_bound = np.array(options.lower_bound)
        self._leaf_block_size = options.leaf_block_size
        self._use_sparse_grid = options.use_sparse_grid
        self._enable_CPIC = options.enable_CPIC

        self._n_vvert_supports = self.scene.vis_options.n_support_neighbors

        # NOTE: the magnitude of `_p_vol` doesn't affect MPM simulation itself, but it's used to compute particle mass, the scale of which matters in coupling.
        # `_p_vol_scale`` is used to avoid potential numerical instability, as the actual `_p_vol` is very small.
        # We need to account for this scale when handling coupling.
        self._p_vol_real = float(self._particle_size**3)
        self._p_vol_scale = float(1e3)
        self._p_vol = self._p_vol_real * self._p_vol_scale

        # other derived parameters
        self._dx = float(1.0 / self._grid_density)
        self._inv_dx = float(self._grid_density)
        self._lower_bound_cell = np.round(self._grid_density * self._lower_bound).astype(gs.np_int)
        self._upper_bound_cell = np.round(self._grid_density * self._upper_bound).astype(gs.np_int)
        self._grid_res = self._upper_bound_cell - self._lower_bound_cell + 1  # +1 to include both corner
        self._grid_offset = ti.Vector(self._lower_bound_cell)
        if self._use_sparse_grid:
            self._grid_res = (np.ceil(self._grid_res / self._leaf_block_size) * self._leaf_block_size).astype(gs.np_int)

            if sim.requires_grad:
                gs.raise_exception("Sparse grid is not supported in differentiable mode.")

        # materials
        self._mats = list()
        self._mats_idx = list()
        self._mats_update_F_S_Jp = list()
        self._mats_update_stress = list()

        # boundary
        self.setup_boundary()

    def setup_boundary(self):
        # safety padding
        self.boundary_padding = 3 * self._dx
        self.boundary = CubeBoundary(
            lower=self._lower_bound + self.boundary_padding,
            upper=self._upper_bound - self.boundary_padding,
        )

    def init_particle_fields(self):
        # dynamic particle state
        struct_particle_state = ti.types.struct(
            pos=gs.ti_vec3,  # position
            vel=gs.ti_vec3,  # velocity
            C=gs.ti_mat3,  # affine velocity field
            F=gs.ti_mat3,  # deformation gradient
            F_tmp=gs.ti_mat3,  # temp deformation gradient
            U=gs.ti_mat3,  # SVD
            V=gs.ti_mat3,  # SVD
            S=gs.ti_mat3,  # SVD
            actu=gs.ti_float,  # actuation
            Jp=gs.ti_float,  # volume ratio
        )

        # dynamic particle state without gradient
        struct_particle_state_ng = ti.types.struct(
            active=gs.ti_int,
        )

        # static particle info
        struct_particle_info = ti.types.struct(
            mat_idx=gs.ti_int,
            mass=gs.ti_float,
            default_Jp=gs.ti_float,
            free=gs.ti_int,
            # for muscle
            muscle_group=gs.ti_int,
            muscle_direction=gs.ti_vec3,
        )

        # single frame particle state for rendering
        struct_particle_state_render = ti.types.struct(
            pos=gs.ti_vec3,
            vel=gs.ti_vec3,
            active=gs.ti_int,
        )

        # construct fields
        self.particles = struct_particle_state.field(
            shape=(self._sim.substeps_local + 1, self._n_particles),
            needs_grad=True,
            layout=ti.Layout.SOA,
        )
        self.particles_ng = struct_particle_state_ng.field(
            shape=(self._sim.substeps_local + 1, self._n_particles),
            needs_grad=False,
            layout=ti.Layout.SOA,
        )
        self.particles_info = struct_particle_info.field(
            shape=self._n_particles, needs_grad=False, layout=ti.Layout.SOA
        )
        self.particles_render = struct_particle_state_render.field(
            shape=self._n_particles, needs_grad=False, layout=ti.Layout.SOA
        )

    def init_grid_fields(self):
        grid_cell_state = ti.types.struct(
            vel_in=gs.ti_vec3,  # input momentum/velocity
            mass=gs.ti_float,  # mass
            vel_out=gs.ti_vec3,  # output momentum/velocity
        )

        if self._use_sparse_grid:
            # temporal block -> coarse block -> fine block
            self.grid_block_0 = ti.root.dense(ti.axes(0), self._sim.substeps_local + 1)
            self.grid_block_1 = self.grid_block_0.pointer(ti.axes(1, 2, 3), self._grid_res // self._leaf_block_size)
            self.grid_block_2 = self.grid_block_1.dense(ti.axes(1, 2, 3), self._leaf_block_size)

            self.grid = grid_cell_state.field(needs_grad=True)
            self.grid_block_2.place(self.grid, self.grid.grad)

            self.deactivate_grid_block()

        else:
            self.grid = grid_cell_state.field(
                shape=(self._sim.substeps_local + 1, *self._grid_res), needs_grad=True, layout=ti.Layout.SOA
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

    def deactivate_grid_block(self):
        self.grid_block_1.deactivate_all()

    def init_ckpt(self):
        self._ckpt = dict()

    def reset_grad(self):
        self.particles.grad.fill(0)
        self.grid.grad.fill(0)

        for entity in self._entities:
            entity.reset_grad()

    def build(self):
        # particles and entities
        self._n_particles = self.n_particles
        self._n_vverts = self.n_vverts
        self._n_vfaces = self.n_vfaces

        self._coupler = self.sim._coupler

        if self.is_active():
            if self._enable_CPIC:
                gs.logger.warning(
                    "Kernel compilation takes longer when running MPM solver in CPIC mode. Please be patient."
                )
                if self._sim.requires_grad:
                    gs.raise_exception(
                        "CPIC is not supported in differentiable mode yet. Submit a feature request if you need it."
                    )

            self.init_particle_fields()
            self.init_grid_fields()
            self.init_vvert_fields()
            self.init_ckpt()

            for entity in self._entities:
                entity._add_to_solver()

            # reference: https://github.com/taichi-dev/taichi_elements/blob/d19678869a28b09a32ef415b162e35dc929b792d/engine/mpm_solver.py#L84
            suggested_dt = 2e-2 * self._dx
            if self.substep_dt > suggested_dt:
                gs.logger.warning(
                    f"Current `substep_dt` ({self.substep_dt:.6g}) is greater than suggested_dt ({suggested_dt:.6g}, calculated based on `grid_density`). Simulation might be unstable."
                )

    def add_entity(self, idx, material, morph, surface):
        self.add_material(material)

        # create entity
        entity = MPMEntity(
            scene=self._scene,
            solver=self,
            material=material,
            morph=morph,
            surface=surface,
            particle_size=self._particle_size,
            idx=idx,
            particle_start=self.n_particles,
            vvert_start=self.n_vverts,
            vface_start=self.n_vfaces,
        )
        self._entities.append(entity)

        return entity

    def add_material(self, material):
        # add material's update methods if not matching any existing material
        exist = False
        for mat in self._mats:
            if material == mat:
                material._idx = mat._idx
                exist = True
                break
        self._mats.append(material)
        if not exist:
            material._idx = len(self._mats_idx)
            self._mats_idx.append(material._idx)
            self._mats_update_F_S_Jp.append(material.update_F_S_Jp)
            self._mats_update_stress.append(material.update_stress)

    def is_active(self):
        return self.n_particles > 0

    # ------------------------------------------------------------------------------------
    # ----------------------------------- simulation -------------------------------------
    # ------------------------------------------------------------------------------------

    @ti.kernel
    def compute_F_tmp(self, f: ti.i32):
        for i in range(self._n_particles):
            if self.particles_ng[f, i].active:
                self.particles[f, i].F_tmp = (
                    ti.Matrix.identity(gs.ti_float, 3) + self.substep_dt * self.particles[f, i].C
                ) @ self.particles[f, i].F

    @ti.kernel
    def svd(self, f: ti.i32):
        for i in range(self._n_particles):
            if self.particles_ng[f, i].active:
                self.particles[f, i].U, self.particles[f, i].S, self.particles[f, i].V = ti.svd(
                    self.particles[f, i].F_tmp, gs.ti_float
                )

    @ti.kernel
    def svd_grad(self, f: ti.i32):
        for i in range(self._n_particles):
            if self.particles_ng[f, i].active:
                self.particles.grad[f, i].F_tmp += self.backward_svd(
                    self.particles.grad[f, i].U,
                    self.particles.grad[f, i].S,
                    self.particles.grad[f, i].V,
                    self.particles[f, i].U,
                    self.particles[f, i].S,
                    self.particles[f, i].V,
                )

    @ti.func
    def backward_svd(self, grad_U, grad_S, grad_V, U, S, V):
        # https://github.com/pytorch/pytorch/blob/ab0a04dc9c8b84d4a03412f1c21a6c4a2cefd36c/tools/autograd/templates/Functions.cpp
        vt = V.transpose()
        ut = U.transpose()
        S_term = U @ grad_S @ vt

        s = ti.Vector.zero(gs.ti_float, 3)
        s = ti.Vector([S[0, 0], S[1, 1], S[2, 2]]) ** 2
        F = ti.Matrix.zero(gs.ti_float, 3, 3)
        for i, j in ti.static(ti.ndrange(3, 3)):
            if i == j:
                F[i, j] = 0
            else:
                F[i, j] = 1.0 / self.clamp(s[j] - s[i])
        u_term = U @ ((F * (ut @ grad_U - grad_U.transpose() @ U)) @ S) @ vt
        v_term = U @ (S @ ((F * (vt @ grad_V - grad_V.transpose() @ V)) @ vt))
        return u_term + v_term + S_term

    @ti.func
    def clamp(self, a):
        if a >= 0:
            a = ti.max(a, 1e-6)
        else:
            a = ti.min(a, -1e-6)
        return a

    @ti.func
    def stencil_range(self):
        return ti.ndrange(3, 3, 3)

    @ti.kernel
    def p2g(self, f: ti.i32):
        for i in range(self._n_particles):
            if self.particles_ng[f, i].active:
                # A. update F (deformation gradient), S (Sigma from SVD(F), essentially represents volume) and Jp (volume compression ratio) based on material type
                J = self.particles[f, i].S.determinant()
                F_new = ti.Matrix.zero(gs.ti_float, 3, 3)
                S_new = ti.Matrix.zero(gs.ti_float, 3, 3)
                Jp_new = gs.ti_float(1.0)
                for mat_idx in ti.static(self._mats_idx):
                    if self.particles_info[i].mat_idx == mat_idx:
                        F_new, S_new, Jp_new = self._mats_update_F_S_Jp[mat_idx](
                            J=J,
                            F_tmp=self.particles[f, i].F_tmp,
                            U=self.particles[f, i].U,
                            S=self.particles[f, i].S,
                            V=self.particles[f, i].V,
                            Jp=self.particles[f, i].Jp,
                        )
                self.particles[f + 1, i].F = F_new
                self.particles[f + 1, i].Jp = Jp_new

                # B. compute stress
                # NOTE:
                # 1. Here we pass in both F_tmp and the updated F_new because in the official taichi example, F_new is used for stress computation. However, although this works for both elastic and elasto-plastic materials, it is mathematically incorrect for liquid material with non-zero viscosity (mu). In the latter case, stress computation needs to be based on the F_tmp (deformation gradient before resetting to identity).
                # 2. Jp is only used by Snow material, and it uses Jp from the previous frame, not the updated one.
                stress = ti.Matrix.zero(gs.ti_float, 3, 3)
                for mat_idx in ti.static(self._mats_idx):
                    if self.particles_info[i].mat_idx == mat_idx:
                        stress = self._mats_update_stress[mat_idx](
                            U=self.particles[f, i].U,
                            S=S_new,
                            V=self.particles[f, i].V,
                            F_tmp=self.particles[f, i].F_tmp,
                            F_new=F_new,
                            J=J,
                            Jp=self.particles[f, i].Jp,
                            actu=self.particles[f, i].actu,
                            m_dir=self.particles_info[i].muscle_direction,
                        )
                stress = (-self.substep_dt * self._p_vol * 4 * self._inv_dx * self._inv_dx) * stress
                affine = stress + self.particles_info[i].mass * self.particles[f, i].C

                # C. project onto grid
                base = ti.floor(self.particles[f, i].pos * self._inv_dx - 0.5).cast(gs.ti_int)
                fx = self.particles[f, i].pos * self._inv_dx - base.cast(gs.ti_float)
                w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
                for offset in ti.static(ti.grouped(self.stencil_range())):
                    dpos = (offset.cast(gs.ti_float) - fx) * self._dx
                    weight = ti.cast(1.0, gs.ti_float)
                    for d in ti.static(range(3)):
                        weight *= w[offset[d]][d]

                    if ti.static(self._enable_CPIC):
                        # check if particle and cell center are at different side of any thin object
                        cell_pos = (base + offset) * self._dx

                        sep_geom_idx = -1
                        for i_g in range(self.sim.rigid_solver.n_geoms):
                            if self.sim.rigid_solver.geoms_info[i_g].needs_coup:
                                sdf_normal_particle = self._coupler.mpm_rigid_normal[i, i_g]
                                sdf_normal_cell = self.sim.rigid_solver.sdf.sdf_normal_world(cell_pos, i_g, 0)
                                if sdf_normal_particle.dot(sdf_normal_cell) < 0:  # separated by geom i_g
                                    sep_geom_idx = i_g
                                    break
                        self._coupler.cpic_flag[i, offset[0], offset[1], offset[2]] = sep_geom_idx
                        if sep_geom_idx == -1:
                            self.grid[f, base - self._grid_offset + offset].vel_in += weight * (
                                self.particles_info[i].mass * self.particles[f, i].vel + affine @ dpos
                            )
                            self.grid[f, base - self._grid_offset + offset].mass += weight * self.particles_info[i].mass
                    else:
                        self.grid[f, base - self._grid_offset + offset].vel_in += weight * (
                            self.particles_info[i].mass * self.particles[f, i].vel + affine @ dpos
                        )
                        self.grid[f, base - self._grid_offset + offset].mass += weight * self.particles_info[i].mass

                    if self.particles_info[i].free == 0:  # non-free particles behave as boundary conditions
                        self.grid[f, base - self._grid_offset + offset].vel_in = ti.Vector.zero(gs.ti_float, 3)

    @ti.kernel
    def g2p(self, f: ti.i32):
        for i in range(self._n_particles):
            if self.particles_ng[f, i].active:
                base = ti.floor(self.particles[f, i].pos * self._inv_dx - 0.5).cast(gs.ti_int)
                fx = self.particles[f, i].pos * self._inv_dx - base.cast(gs.ti_float)
                w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0) ** 2, 0.5 * (fx - 0.5) ** 2]
                new_vel = ti.Vector.zero(gs.ti_float, 3)
                new_C = ti.Matrix.zero(gs.ti_float, 3, 3)
                for offset in ti.static(ti.grouped(self.stencil_range())):
                    dpos = offset.cast(gs.ti_float) - fx
                    grid_vel = self.grid[f, base - self._grid_offset + offset].vel_out
                    weight = ti.cast(1.0, gs.ti_float)
                    for d in ti.static(range(3)):
                        weight *= w[offset[d]][d]

                    if ti.static(self._enable_CPIC):
                        sep_geom_idx = self._coupler.cpic_flag[i, offset[0], offset[1], offset[2]]
                        if sep_geom_idx != -1:
                            grid_vel = self.sim.coupler._func_collide_in_rigid_geom(
                                self.particles[f, i].pos,
                                self.particles[f, i].vel,
                                self.particles_info[i].mass * weight / self._p_vol_scale,
                                self._coupler.mpm_rigid_normal[i, sep_geom_idx],
                                1.0,
                                sep_geom_idx,
                                0,
                            )

                    new_vel += weight * grid_vel
                    new_C += 4 * self._inv_dx * weight * grid_vel.outer_product(dpos)

                # compute actual new_pos with new_vel
                new_pos = self.particles[f, i].pos + self.substep_dt * new_vel

                # impose boundary for safety, in case simulation explodes and tries to access illegal cell address
                new_pos, new_vel = self.boundary.impose_pos_vel(new_pos, new_vel)

                # advect to next frame
                self.particles[f + 1, i].vel = new_vel
                self.particles[f + 1, i].C = new_C
                self.particles[f + 1, i].pos = new_pos

            else:
                self.particles[f + 1, i].vel = self.particles[f, i].vel
                self.particles[f + 1, i].pos = self.particles[f, i].pos
                self.particles[f + 1, i].C = self.particles[f, i].C
                self.particles[f + 1, i].F = self.particles[f, i].F
                self.particles[f + 1, i].Jp = self.particles[f, i].Jp

            self.particles_ng[f + 1, i].active = self.particles_ng[f, i].active

    # ------------------------------------------------------------------------------------
    # ------------------------------------ stepping --------------------------------------
    # ------------------------------------------------------------------------------------

    def process_input(self, in_backward=False):
        for entity in self._entities:
            entity.process_input(in_backward=in_backward)

    def process_input_grad(self):
        for entity in self._entities[::-1]:
            entity.process_input_grad()

    def substep_pre_coupling(self, f):
        if not self._use_sparse_grid:
            self.reset_grid_and_grad(f)
        self.compute_F_tmp(f)
        self.svd(f)
        self.p2g(f)

    def substep_pre_coupling_grad(self, f):
        self.p2g.grad(f)
        self.svd_grad(f)
        self.compute_F_tmp.grad(f)

    def substep_post_coupling(self, f):
        self.g2p(f)
        if self._use_sparse_grid:
            # trick: without ti.sync it can be slow
            ti.sync()

    def substep_post_coupling_grad(self, f):
        self.g2p.grad(f)

    @ti.kernel
    def copy_frame(self, source: ti.i32, target: ti.i32):
        for i in range(self._n_particles):
            self.particles[target, i].pos = self.particles[source, i].pos
            self.particles[target, i].vel = self.particles[source, i].vel
            self.particles[target, i].F = self.particles[source, i].F
            self.particles[target, i].C = self.particles[source, i].C
            self.particles[target, i].Jp = self.particles[source, i].Jp
            self.particles_ng[target, i].active = self.particles_ng[source, i].active

    @ti.kernel
    def copy_grad(self, source: ti.i32, target: ti.i32):
        for i in range(self._n_particles):
            self.particles.grad[target, i].pos = self.particles.grad[source, i].pos
            self.particles.grad[target, i].vel = self.particles.grad[source, i].vel
            self.particles.grad[target, i].F = self.particles.grad[source, i].F
            self.particles.grad[target, i].C = self.particles.grad[source, i].C
            self.particles.grad[target, i].Jp = self.particles.grad[source, i].Jp
            self.particles_ng[target, i].active = self.particles_ng[source, i].active

    @ti.kernel
    def reset_grid_and_grad(self, f: ti.i32):
        for I in ti.grouped(ti.ndrange(*self._grid_res)):
            self.grid[f, I].vel_in = 0
            self.grid[f, I].mass = 0
            self.grid[f, I].vel_out = 0
            self.grid.grad[f, I].vel_in = 0
            self.grid.grad[f, I].mass = 0
            self.grid.grad[f, I].vel_out = 0

    @ti.kernel
    def reset_grad_till_frame(self, f: ti.i32):
        for i, j in ti.ndrange(f, self._n_particles):
            self.particles.grad[i, j].pos = 0
            self.particles.grad[i, j].vel = 0
            self.particles.grad[i, j].C = 0
            self.particles.grad[i, j].F = 0
            self.particles.grad[i, j].F_tmp = 0
            self.particles.grad[i, j].Jp = 0
            self.particles.grad[i, j].U = 0
            self.particles.grad[i, j].V = 0
            self.particles.grad[i, j].S = 0

    # ------------------------------------------------------------------------------------
    # ------------------------------------ gradient --------------------------------------
    # ------------------------------------------------------------------------------------

    def collect_output_grads(self):
        """
        Collect gradients from downstream queried states.
        """
        for entity in self._entities:
            entity.collect_output_grads()

    def add_grad_from_state(self, state):
        if self.is_active():
            if state.pos.grad is not None:
                state.pos.assert_contiguous()
                self.add_grad_from_pos(self._sim.cur_substep_local, state.pos.grad)

            if state.vel.grad is not None:
                state.vel.assert_contiguous()
                self.add_grad_from_vel(self._sim.cur_substep_local, state.vel.grad)

            if state.C.grad is not None:
                state.C.assert_contiguous()
                self.add_grad_from_C(self._sim.cur_substep_local, state.C.grad)

            if state.F.grad is not None:
                state.F.assert_contiguous()
                self.add_grad_from_F(self._sim.cur_substep_local, state.F.grad)

            if state.Jp.grad is not None:
                state.Jp.assert_contiguous()
                self.add_grad_from_Jp(self._sim.cur_substep_local, state.Jp.grad)

    @ti.kernel
    def add_grad_from_pos(self, f: ti.i32, pos_grad: ti.types.ndarray()):
        for i in range(self._n_particles):
            for j in ti.static(range(3)):
                self.particles.grad[f, i].pos[j] += pos_grad[i, j]

    @ti.kernel
    def add_grad_from_vel(self, f: ti.i32, vel_grad: ti.types.ndarray()):
        for i in range(self._n_particles):
            for j in ti.static(range(3)):
                self.particles.grad[f, i].vel[j] += vel_grad[i, j]

    @ti.kernel
    def add_grad_from_C(self, f: ti.i32, C_grad: ti.types.ndarray()):
        for i in range(self._n_particles):
            for j in ti.static(range(3)):
                for k in ti.static(range(3)):
                    self.particles.grad[f, i].C[j, k] += C_grad[i, j, k]

    @ti.kernel
    def add_grad_from_F(self, f: ti.i32, F_grad: ti.types.ndarray()):
        for i in range(self._n_particles):
            for j in ti.static(range(3)):
                for k in ti.static(range(3)):
                    self.particles.grad[f, i].F[j, k] += F_grad[i, j, k]

    @ti.kernel
    def add_grad_from_Jp(self, f: ti.i32, Jp_grad: ti.types.ndarray()):
        for i in range(self._n_particles):
            self.particles.grad[f, i].Jp += Jp_grad[i]

    def save_ckpt(self, ckpt_name):
        if self._sim.requires_grad:
            if ckpt_name not in self._ckpt:
                self._ckpt[ckpt_name] = dict()
                self._ckpt[ckpt_name]["pos"] = torch.zeros((self._n_particles, 3), dtype=gs.tc_float)
                self._ckpt[ckpt_name]["vel"] = torch.zeros((self._n_particles, 3), dtype=gs.tc_float)
                self._ckpt[ckpt_name]["C"] = torch.zeros((self._n_particles, 3, 3), dtype=gs.tc_float)
                self._ckpt[ckpt_name]["F"] = torch.zeros((self._n_particles, 3, 3), dtype=gs.tc_float)
                self._ckpt[ckpt_name]["Jp"] = torch.zeros((self._n_particles,), dtype=gs.tc_float)
                self._ckpt[ckpt_name]["active"] = torch.zeros((self._n_particles,), dtype=torch.int32)

            self._kernel_get_state(
                0,
                self._ckpt[ckpt_name]["pos"],
                self._ckpt[ckpt_name]["vel"],
                self._ckpt[ckpt_name]["C"],
                self._ckpt[ckpt_name]["F"],
                self._ckpt[ckpt_name]["Jp"],
                self._ckpt[ckpt_name]["active"],
            )

            for entity in self._entities:
                entity.save_ckpt(ckpt_name)

        # restart from frame 0 in memory
        if self._use_sparse_grid:
            self.deactivate_grid_block()
        self.copy_frame(self._sim.substeps_local, 0)

    def load_ckpt(self, ckpt_name):
        if self._use_sparse_grid:
            self.deactivate_grid_block()
        self.copy_frame(0, self._sim.substeps_local)
        self.copy_grad(0, self._sim.substeps_local)

        if self._sim.requires_grad:
            self.reset_grad_till_frame(self._sim.substeps_local)

            self._kernel_set_state(
                0,
                self._ckpt[ckpt_name]["pos"],
                self._ckpt[ckpt_name]["vel"],
                self._ckpt[ckpt_name]["C"],
                self._ckpt[ckpt_name]["F"],
                self._ckpt[ckpt_name]["Jp"],
                self._ckpt[ckpt_name]["active"],
            )

            for entity in self._entities:
                entity.load_ckpt(ckpt_name=ckpt_name)

    # ------------------------------------------------------------------------------------
    # --------------------------------------- io -----------------------------------------
    # ------------------------------------------------------------------------------------

    @ti.kernel
    def _kernel_add_particles(
        self,
        f: ti.i32,
        active: ti.i32,
        particle_start: ti.i32,
        n_particles: ti.i32,
        mat_idx: ti.i32,
        mat_default_Jp: ti.f32,
        mat_rho: ti.f32,
        pos: ti.types.ndarray(),
    ):
        for i in range(n_particles):
            i_global = i + particle_start
            for j in ti.static(range(3)):
                self.particles[f, i_global].pos[j] = pos[i, j]
            self.particles[f, i_global].vel = ti.Vector.zero(gs.ti_float, 3)
            self.particles[f, i_global].F = ti.Matrix.identity(gs.ti_float, 3)
            self.particles[f, i_global].C = ti.Matrix.zero(gs.ti_float, 3, 3)
            self.particles[f, i_global].Jp = mat_default_Jp
            self.particles[f, i_global].actu = gs.ti_float(0.0)

            self.particles_ng[f, i_global].active = active

            self.particles_info[i_global].mat_idx = mat_idx
            self.particles_info[i_global].default_Jp = mat_default_Jp
            self.particles_info[i_global].mass = self._p_vol * mat_rho
            self.particles_info[i_global].free = 1
            self.particles_info[i_global].muscle_group = 0
            self.particles_info[i_global].muscle_direction = ti.Vector([0.0, 0.0, 1.0], dt=gs.ti_float)

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
                self.particles[f, i_global].pos[k] = pos[i, k]

            # we restore these whenever directly setting positions
            self.particles[f, i_global].vel = ti.Vector.zero(gs.ti_float, 3)
            self.particles[f, i_global].F = ti.Matrix.identity(gs.ti_float, 3)
            self.particles[f, i_global].C = ti.Matrix.zero(gs.ti_float, 3, 3)
            self.particles[f, i_global].Jp = self.particles_info[i_global].default_Jp

    @ti.kernel
    def _kernel_set_particles_pos_grad(
        self,
        f: ti.i32,
        particle_start: ti.i32,
        n_particles: ti.i32,
        pos_grad: ti.types.ndarray(),
    ):
        for i in range(n_particles):
            i_global = i + particle_start
            for k in ti.static(range(3)):
                pos_grad[i, k] = self.particles.grad[f, i_global].pos[k]

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
                self.particles[f, i_global].vel[k] = vel[i, k]

    @ti.kernel
    def _kernel_set_particles_vel_grad(
        self,
        f: ti.i32,
        particle_start: ti.i32,
        n_particles: ti.i32,
        vel_grad: ti.types.ndarray(),
    ):
        for i in range(n_particles):
            i_global = i + particle_start
            for k in ti.static(range(3)):
                vel_grad[i, k] = self.particles.grad[f, i_global].vel[k]

    @ti.kernel
    def _kernel_set_particles_actu(
        self,
        f: ti.i32,
        particle_start: ti.i32,
        n_particles: ti.i32,
        n_groups: ti.i32,
        actu: ti.types.ndarray(),
    ):
        for i in range(n_particles):
            i_global = i + particle_start
            for j in range(n_groups):
                if self.particles_info[i_global].muscle_group == j:
                    self.particles[f, i_global].actu = actu[i, j]

    @ti.kernel
    def _kernel_set_particles_actu_grad(
        self,
        f: ti.i32,
        particle_start: ti.i32,
        n_particles: ti.i32,
        actu_grad: ti.types.ndarray(),
    ):
        for i in range(n_particles):
            i_global = i + particle_start
            actu_grad[i] = self.particles.grad[f, i_global].actu

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
            self.particles_ng[f, i_global].active = active

    @ti.kernel
    def _kernel_set_particles_active_arr(
        self,
        f: ti.i32,
        particle_start: ti.i32,
        n_particles: ti.i32,
        active: ti.types.ndarray(),
    ):
        for i in range(n_particles):
            i_global = i + particle_start
            self.particles_ng[f, i_global].active = active[i]

    @ti.kernel
    def _kernel_get_particles_active_arr(
        self,
        f: ti.i32,
        particle_start: ti.i32,
        n_particles: ti.i32,
        active: ti.types.ndarray(),
    ):
        for i in range(n_particles):
            i_global = i + particle_start
            active[i] = self.particles_ng[f, i_global].active

    @ti.kernel
    def _kernel_set_muscle_group(
        self,
        particle_start: ti.i32,
        n_particles: ti.i32,
        muscle_group: ti.types.ndarray(),
    ):
        for i in range(n_particles):
            i_global = i + particle_start
            self.particles_info[i_global].muscle_group = muscle_group[i]

    @ti.kernel
    def _kernel_get_muscle_group(
        self,
        particle_start: ti.i32,
        n_particles: ti.i32,
        muscle_group: ti.types.ndarray(),
    ):
        for i in range(n_particles):
            i_global = i + particle_start
            muscle_group[i] = self.particles_info[i_global].muscle_group

    @ti.kernel
    def _kernel_set_muscle_direction(
        self,
        particle_start: ti.i32,
        n_particles: ti.i32,
        muscle_direction: ti.types.ndarray(),
    ):
        for i in range(n_particles):
            i_global = i + particle_start
            for j in ti.static(range(3)):
                self.particles_info[i_global].muscle_direction[j] = muscle_direction[i, j]

    @ti.kernel
    def _kernel_set_free(
        self,
        particle_start: ti.i32,
        n_particles: ti.i32,
        free: ti.types.ndarray(),
    ):
        for i in range(n_particles):
            i_global = i + particle_start
            self.particles_info[i_global].free = free[i]

    @ti.kernel
    def _kernel_get_free(
        self,
        particle_start: ti.i32,
        n_particles: ti.i32,
        free: ti.types.ndarray(),
    ):
        for i in range(n_particles):
            i_global = i + particle_start
            free[i] = self.particles_info[i_global].free

    @ti.kernel
    def _kernel_get_state(
        self,
        f: ti.i32,
        pos: ti.types.ndarray(),
        vel: ti.types.ndarray(),
        C: ti.types.ndarray(),
        F: ti.types.ndarray(),
        Jp: ti.types.ndarray(),
        active: ti.types.ndarray(),
    ):
        for i in range(self._n_particles):
            for j in ti.static(range(3)):
                pos[i, j] = self.particles[f, i].pos[j]
                vel[i, j] = self.particles[f, i].vel[j]
                for k in ti.static(range(3)):
                    C[i, j, k] = self.particles[f, i].C[j, k]
                    F[i, j, k] = self.particles[f, i].F[j, k]
            Jp[i] = self.particles[f, i].Jp
            active[i] = self.particles_ng[f, i].active

    @ti.kernel
    def _kernel_set_state(
        self,
        f: ti.i32,
        pos: ti.types.ndarray(),
        vel: ti.types.ndarray(),
        C: ti.types.ndarray(),
        F: ti.types.ndarray(),
        Jp: ti.types.ndarray(),
        active: ti.types.ndarray(),
    ):
        for i in range(self._n_particles):
            for j in ti.static(range(3)):
                self.particles[f, i].pos[j] = pos[i, j]
                self.particles[f, i].vel[j] = vel[i, j]
                for k in ti.static(range(3)):
                    self.particles[f, i].C[j, k] = C[i, j, k]
                    self.particles[f, i].F[j, k] = F[i, j, k]
            self.particles[f, i].Jp = Jp[i]
            self.particles_ng[f, i].active = active[i]

    def get_state(self, f):
        if self.is_active():
            state = MPMSolverState(self._scene)
            self._kernel_get_state(f, state.pos, state.vel, state.C, state.F, state.Jp, state.active)
        else:
            state = None
        return state

    def set_state(self, f, state):
        if self.is_active():
            self._kernel_set_state(f, state.pos, state.vel, state.C, state.F, state.Jp, state.active)

    @ti.kernel
    def _kernel_update_render_fields(self, f: ti.i32):
        for i in range(self._n_particles):
            if self.particles_ng[f, i].active:
                self.particles_render[i].pos = self.particles[f, i].pos
                self.particles_render[i].vel = self.particles[f, i].vel
            else:
                self.particles_render[i].pos = gu.ti_nowhere()
            self.particles_render[i].active = self.particles_ng[f, i].active

        for i in range(self._n_vverts):
            vvert_pos = ti.Vector.zero(gs.ti_float, 3)
            for j in range(self._n_vvert_supports):
                vvert_pos += (
                    self.particles[f, self.vverts_info.support_idxs[i][j]].pos * self.vverts_info.support_weights[i][j]
                )
            self.vverts_render[i].pos = vvert_pos
            self.vverts_render[i].active = self.particles_render[self.vverts_info.support_idxs[i][0]].active

    def update_render_fields(self):
        self._kernel_update_render_fields(self.sim.cur_substep_local)

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
    def n_vverts(self):
        if self.is_built:
            return self._n_vverts
        else:
            return sum([entity.n_vverts for entity in self._entities])

    @property
    def n_vfaces(self):
        if self.is_built:
            return self._n_vfaces
        else:
            return sum([entity.n_vfaces for entity in self._entities])

    @property
    def grid_density(self):
        return self._grid_density

    @property
    def particle_size(self):
        return self._particle_size

    @property
    def particle_radius(self):
        return self._particle_size / 2.0

    @property
    def upper_bound(self):
        return self._upper_bound

    @property
    def lower_bound(self):
        return self._lower_bound

    @property
    def leaf_block_size(self):
        return self._leaf_block_size

    @property
    def use_sparse_grid(self):
        return self._use_sparse_grid

    @property
    def dx(self):
        return self._dx

    @property
    def inv_dx(self):
        return self._inv_dx

    @property
    def p_vol_real(self):
        return self._p_vol_real

    @property
    def p_vol(self):
        return self._p_vol

    @property
    def p_vol_scale(self):
        return self._p_vol_scale

    @property
    def is_built(self):
        return self._scene._is_built

    @property
    def lower_bound_cell(self):
        return self._lower_bound_cell

    @property
    def upper_bound_cell(self):
        return self._upper_bound_cell

    @property
    def grid_res(self):
        return self._grid_res

    @property
    def grid_offset(self):
        return self._grid_offset

    @property
    def enable_CPIC(self):
        return self._enable_CPIC
