import numpy as np
import taichi as ti

import genesis as gs
import genesis.utils.geom as gu
from genesis.engine.boundaries import CubeBoundary
from genesis.engine.entities import SPHEntity
from genesis.engine.states.solvers import SPHSolverState

from .base_solver import Solver


@ti.data_oriented
class SPHSolver(Solver):
    # ------------------------------------------------------------------------------------
    # --------------------------------- Initialization -----------------------------------
    # ------------------------------------------------------------------------------------

    def __init__(self, scene, sim, options):
        super().__init__(scene, sim, options)

        # options
        self._particle_size = options.particle_size
        self._support_radius = options._support_radius
        self._pressure_solver = options.pressure_solver

        # DFSPH parameters
        self._df_max_error_div = options.max_divergence_error
        self._df_max_error_den = options.max_density_error_percent
        self._df_max_div_iters = options.max_divergence_solver_iterations
        self._df_max_den_iters = options.max_density_solver_iterations
        self._df_eps = 1e-5

        self._upper_bound = np.array(options.upper_bound)
        self._lower_bound = np.array(options.lower_bound)

        self._p_vol = 0.8 * self._particle_size**3  # 0.8 is an empirical value

        # spatial hasher
        self.sh = gu.SpatialHasher(
            cell_size=options.hash_grid_cell_size,
            grid_res=options._hash_grid_res,
        )
        # boundary
        self.setup_boundary()

    def setup_boundary(self):
        self.boundary = CubeBoundary(
            lower=self._lower_bound,
            upper=self._upper_bound,
            # restitution=0.5,
        )

    def init_particle_fields(self):
        # dynamic particle state
        struct_particle_state = ti.types.struct(
            pos=gs.ti_vec3,  # position
            vel=gs.ti_vec3,  # velocity
            acc=gs.ti_vec3,  # acceleration
            rho=gs.ti_float,  # density
            p=gs.ti_float,  # pressure
            dfsph_factor=gs.ti_float,  # DFSPH use: Factor for Divergence and density solver
            drho=gs.ti_float,  # density deritivate
        )

        # dynamic particle state without gradient
        struct_particle_state_ng = ti.types.struct(
            reordered_idx=gs.ti_int,
            active=gs.ti_int,
        )

        # static particle info
        struct_particle_info = ti.types.struct(
            rho=gs.ti_float,  # rest density
            mass=gs.ti_float,  # mass
            stiffness=gs.ti_float,
            exponent=gs.ti_float,
            mu=gs.ti_float,  # viscosity
            gamma=gs.ti_float,  # surface tension
        )

        # single frame particle state for rendering
        struct_particle_state_render = ti.types.struct(
            pos=gs.ti_vec3,
            vel=gs.ti_vec3,
            active=gs.ti_int,
        )

        # construct fields
        self.particles = struct_particle_state.field(shape=(self._n_particles,), needs_grad=False, layout=ti.Layout.SOA)
        self.particles_ng = struct_particle_state_ng.field(
            shape=(self._n_particles,), needs_grad=False, layout=ti.Layout.SOA
        )
        self.particles_info = struct_particle_info.field(
            shape=(self._n_particles,), needs_grad=False, layout=ti.Layout.SOA
        )

        self.particles_reordered = struct_particle_state.field(
            shape=(self._n_particles,), needs_grad=False, layout=ti.Layout.SOA
        )
        self.particles_ng_reordered = struct_particle_state_ng.field(
            shape=(self._n_particles,), needs_grad=False, layout=ti.Layout.SOA
        )
        self.particles_info_reordered = struct_particle_info.field(
            shape=(self._n_particles,), needs_grad=False, layout=ti.Layout.SOA
        )

        self.particles_render = struct_particle_state_render.field(
            shape=self._n_particles, needs_grad=False, layout=ti.Layout.SOA
        )

    def init_ckpt(self):
        self._ckpt = dict()

    def reset_grad(self):
        pass

    def build(self):
        # particles and entities
        self._n_particles = self.n_particles

        self._coupler = self.sim._coupler

        if self.is_active():
            self.sh.build()
            self.init_particle_fields()
            self.init_ckpt()

            for entity in self.entities:
                entity._add_to_solver()

            # TODO: @Mingrui: this is a temporary hack. Need to support per-particle density
            self._density0 = self.particles_info[0].rho

    # ------------------------------------------------------------------------------------
    # -------------------------------------- misc ----------------------------------------
    # ------------------------------------------------------------------------------------

    def add_entity(self, idx, material, morph, surface):
        entity = SPHEntity(
            scene=self.scene,
            solver=self,
            material=material,
            morph=morph,
            surface=surface,
            particle_size=self._particle_size,
            idx=idx,
            particle_start=self.n_particles,
        )

        self.entities.append(entity)
        return entity

    def is_active(self):
        return self.n_particles > 0

    # ------------------------------------------------------------------------------------
    # ----------------------------------- simulation -------------------------------------
    # ------------------------------------------------------------------------------------

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

        if ti.static(self._coupler._rigid_sph):
            for i, i_g in ti.ndrange(self._n_particles, self._coupler.rigid_solver.n_geoms):
                if self.particles_ng[i].active:
                    self._coupler.sph_rigid_normal_reordered[self.particles_ng[i].reordered_idx, i_g] = (
                        self._coupler.sph_rigid_normal[i, i_g]
                    )

    @ti.kernel
    def _kernel_copy_from_reordered(self, f: ti.i32):
        for i in range(self._n_particles):
            if self.particles_ng[i].active:
                # only need to copy back dynamic state, i.e. self.particles
                self.particles[i] = self.particles_reordered[self.particles_ng[i].reordered_idx]

        if ti.static(self._coupler._rigid_sph):
            for i, i_g in ti.ndrange(self._n_particles, self._coupler.rigid_solver.n_geoms):
                if self.particles_ng[i].active:
                    self._coupler.sph_rigid_normal[i, i_g] = self._coupler.sph_rigid_normal_reordered[
                        self.particles_ng[i].reordered_idx, i_g
                    ]

    @ti.func
    def _task_compute_rho(self, i, j, ret: ti.template()):
        ret += self._p_vol * self.cubic_kernel(
            (self.particles_reordered[i].pos - self.particles_reordered[j].pos).norm()
        )

    @ti.kernel
    def _kernel_compute_rho(self, f: ti.i32):
        for i in range(self._n_particles):
            if self.particles_ng_reordered[i].active:
                self.particles_reordered[i].rho = self._p_vol * self.cubic_kernel(0.0)
                den = 0.0
                self.sh.for_all_neighbors(
                    i,
                    self.particles_reordered.pos,
                    self._support_radius,
                    den,
                    self._task_compute_rho,
                )
                self.particles_reordered[i].rho += den
                self.particles_reordered[i].rho *= self.particles_info_reordered[i].rho

    @ti.func
    def _task_compute_non_pressure_forces(self, i, j, ret: ti.template()):
        ############## Surface Tension ###############
        d_ij = self.particles_reordered[i].pos - self.particles_reordered[j].pos

        if d_ij.norm() > self._particle_size:
            ret -= (
                self.particles_info_reordered[i].gamma
                / self.particles_info_reordered[i].mass
                * self.particles_info_reordered[j].mass
                * d_ij
                * self.cubic_kernel(d_ij.norm())
            )
        else:
            ret -= (
                self.particles_info_reordered[i].gamma
                / self.particles_info_reordered[i].mass
                * self.particles_info_reordered[j].mass
                * d_ij
                * self.cubic_kernel(self._particle_size)
            )

        ############### Viscosity Force ###############
        # Compute the viscosity force contribution
        v_ij = (self.particles_reordered[i].vel - self.particles_reordered[j].vel).dot(d_ij)

        d = 2 * (3 + 2)
        f_v = (
            d
            * self.particles_info_reordered[i].mu
            * (self.particles_info_reordered[j].mass / self.particles_reordered[j].rho)
            * v_ij
            / (d_ij.norm() ** 2 + 0.01 * self._support_radius**2)
            * self.cubic_kernel_derivative(d_ij)
        )
        ret += f_v

    @ti.kernel
    def _kernel_compute_non_pressure_forces(self, f: ti.i32, t: ti.f32):
        for i in range(self._n_particles):
            if self.particles_ng_reordered[i].active:
                acc = self._gravity[None]
                self.sh.for_all_neighbors(
                    i,
                    self.particles_reordered.pos,
                    self._support_radius,
                    acc,
                    self._task_compute_non_pressure_forces,
                )

                # external force fields
                for i_ff in ti.static(range(len(self._ffs))):
                    acc += self._ffs[i_ff].get_acc(self.particles_reordered[i].pos, self.particles_reordered[i].vel, t)
                self.particles_reordered[i].acc = acc

    @ti.func
    def _task_compute_pressure_forces(self, i, j, ret: ti.template()):
        dp_i = self.particles_reordered[i].p / self.particles_reordered[i].rho ** 2
        rho_j = (
            self.particles_reordered[j].rho
            * self.particles_info_reordered[j].rho
            / self.particles_info_reordered[j].rho
        )
        dp_j = self.particles_reordered[j].p / rho_j**2

        # Compute the pressure force contribution, Symmetric Formula
        ret += (
            -self.particles_info_reordered[j].rho
            * self._p_vol
            * (dp_i + dp_j)
            * self.cubic_kernel_derivative(self.particles_reordered[i].pos - self.particles_reordered[j].pos)
        )

    @ti.kernel
    def _kernel_compute_pressure_forces(self, f: ti.i32):
        for i in range(self._n_particles):
            if self.particles_ng_reordered[i].active:
                self.particles_reordered[i].rho = ti.max(
                    self.particles_reordered[i].rho, self.particles_info_reordered[i].rho
                )
                self.particles_reordered[i].p = self.particles_info_reordered[i].stiffness * (
                    ti.pow(
                        self.particles_reordered[i].rho / self.particles_info_reordered[i].rho,
                        self.particles_info_reordered[i].exponent,
                    )
                    - 1.0
                )

        for i in range(self._n_particles):
            if self.particles_ng_reordered[i].active:
                acc = ti.Vector.zero(gs.ti_float, 3)
                self.sh.for_all_neighbors(
                    i,
                    self.particles_reordered.pos,
                    self._support_radius,
                    acc,
                    self._task_compute_pressure_forces,
                )
                self.particles_reordered[i].acc += acc

    @ti.kernel
    def _kernel_advect_velocity(self, f: ti.i32):
        for i in range(self._n_particles):
            if self.particles_ng_reordered[i].active:
                self.particles_reordered[i].vel = (
                    self.particles_reordered[i].vel + self._substep_dt * self.particles_reordered[i].acc
                )

    @ti.kernel
    def _kernel_advect_position(self, f: ti.i32):
        for i in range(self._n_particles):
            if self.particles_ng_reordered[i].active:
                pos, new_vel = self.particles_reordered[i].pos, self.particles_reordered[i].vel

                # advect
                new_pos = pos + self._substep_dt * new_vel

                # impose boundary
                new_pos, new_vel = self.boundary.impose_pos_vel(new_pos, new_vel)
                self.particles_reordered[i].vel = new_vel
                self.particles_reordered[i].pos = new_pos

    # ------------------------------------------------------------------------------------
    # ------------------------------------- DFSPH ----------------------------------------
    # ------------------------------------------------------------------------------------
    @ti.func
    def _task_compute_DFSPH_factor(self, i, j, ret: ti.template()):
        # Fluid neighbors
        grad_j = -self._p_vol * self.cubic_kernel_derivative(
            self.particles_reordered[i].pos - self.particles_reordered[j].pos
        )
        ret[3] += grad_j.norm_sqr()  # sum_grad_p_k
        for ii in ti.static(range(3)):  # grad_p_i
            ret[ii] -= grad_j[ii]

    @ti.kernel
    def _kernel_compute_DFSPH_factor(self, f: ti.i32):
        for i in range(self._n_particles):
            if self.particles_ng_reordered[i].active:
                sum_grad_p_k = 0.0
                grad_p_i = ti.Vector.zero(gs.ti_float, 3)

                # `ret` concatenates `grad_p_i` and `sum_grad_p_k`
                ret = ti.Vector.zero(gs.ti_float, 4)

                self.sh.for_all_neighbors(
                    i, self.particles_reordered.pos, self._support_radius, ret, self._task_compute_DFSPH_factor
                )

                sum_grad_p_k = ret[3]
                for ii in ti.static(range(3)):
                    grad_p_i[ii] = ret[ii]
                sum_grad_p_k += grad_p_i.norm_sqr()

                # Compute pressure stiffness denominator
                factor = 0.0
                if sum_grad_p_k > 1e-6:
                    factor = -1.0 / sum_grad_p_k
                else:
                    factor = 0.0
                self.particles_reordered[i].dfsph_factor = factor

    @ti.func
    def _task_compute_density_time_derivative(self, i, j, ret: ti.template()):
        v_i = self.particles_reordered[i].vel
        v_j = self.particles_reordered[j].vel

        x_i = self.particles_reordered[i].pos
        x_j = self.particles_reordered[j].pos

        # Fluid neighbors
        ret.drho += self._p_vol * (v_i - v_j).dot(self.cubic_kernel_derivative(x_i - x_j))
        ret.num_neighbors += 1

    @ti.kernel
    def _kernel_compute_density_time_derivative(self):
        for i in range(self._n_particles):
            if self.particles_ng_reordered[i].active:
                ret = ti.Struct(drho=0.0, num_neighbors=0)
                self.sh.for_all_neighbors(
                    i,
                    self.particles_reordered.pos,
                    self._support_radius,
                    ret,
                    self._task_compute_density_time_derivative,
                )

                # only correct positive divergence
                drho = ti.max(ret.drho, 0.0)
                num_neighbors = ret.num_neighbors

                # Do not perform divergence solve when paritlce deficiency happens
                if num_neighbors < 20:
                    drho = 0.0

                self.particles_reordered[i].drho = drho

    @ti.func
    def _task_divergence_solver_iteration(self, i, j, ret: ti.template()):
        # Fluid neighbors
        b_j = self.particles_reordered[j].drho
        k_j = b_j * self.particles_reordered[j].dfsph_factor
        k_sum = (
            self._density0 / self._density0 * ret.k_i + k_j
        )  # TODO: make the neighbor density different for multiphase fluid
        if ti.abs(k_sum) > self._df_eps:
            grad_p_j = -self._p_vol * self.cubic_kernel_derivative(
                self.particles_reordered.pos[i] - self.particles_reordered.pos[j]
            )
            ret.dv -= (
                k_sum * grad_p_j
            )  # ki, kj already contain inverse density, i.e., density canceled if not mutiphase flow

    @ti.kernel
    def _kernel_divergence_solver_iteration(self):
        # Perform Jacobi iteration
        for i in range(self._n_particles):
            if self.particles_ng_reordered[i].active:
                # evaluate rhs
                b_i = self.particles_reordered[i].drho
                k_i = b_i * self.particles_reordered[i].dfsph_factor
                ret = ti.Struct(dv=ti.Vector.zero(gs.ti_float, 3), k_i=k_i)
                # TODO: if warm start
                # get_kappa_V += k_i
                self.sh.for_all_neighbors(
                    i, self.particles_reordered.pos, self._support_radius, ret, self._task_divergence_solver_iteration
                )
                self.particles_reordered.vel[i] = self.particles_reordered.vel[i] + ret.dv

    @ti.kernel
    def _kernel_compute_density_error(self, offset: float) -> float:
        density_error = 0.0
        for i in range(self._n_particles):
            if self.particles_ng_reordered[i].active:
                density_error += self._density0 * self.particles_reordered[i].drho - offset
        return density_error

    def _divergence_solver_iteration(self):
        self._kernel_divergence_solver_iteration()
        self._kernel_compute_density_time_derivative()
        density_err = self._kernel_compute_density_error(0.0)
        return density_err / self._n_particles

    def _divergence_solve(self, f: ti.i32):
        # TODO: warm start
        # Compute velocity of density change
        self._kernel_compute_density_time_derivative()
        inv_dt = 1 / self._substep_dt
        # self._kernel_multiply_time_step(self.ps.dfsph_factor, inv_dt)

        iteration = 0

        # Start solver
        avg_density_err = 0.0

        while iteration < self._df_max_div_iters:

            avg_density_err = self._divergence_solver_iteration()
            # Max allowed density fluctuation
            # The SI unit for divergence is s^-1, use max density error divided by time step size
            eta = inv_dt * self._df_max_error_div * 0.01 * self._density0
            # print("eta ", eta)
            if avg_density_err <= eta:
                break
            iteration += 1

        gs.logger.debug(f"DFSPH - iteration V: {iteration} Avg divergence err: {avg_density_err / self._density0:.4f}")

        # Multiply by h, the time step size has to be removed
        # to make the stiffness value independent
        # of the time step size

        # TODO: if warm start
        # also remove for kappa v

        # self._kernel_multiply_time_step(self.ps.dfsph_factor, self.dt[None])

    @ti.kernel
    def _kernel_predict_velocity(self, f: ti.i32):
        # compute new velocities only considering non-pressure forces
        for i in range(self._n_particles):
            if self.particles_ng_reordered[i].active:
                self.particles_reordered[i].vel += self._substep_dt * self.particles_reordered[i].acc

    @ti.func
    def _task_compute_density_star(self, i, j, ret: ti.template()):
        v_i = self.particles_reordered[i].vel
        v_j = self.particles_reordered[j].vel
        x_i = self.particles_reordered[i].pos
        x_j = self.particles_reordered[j].pos
        ret += self._p_vol * (v_i - v_j).dot(self.cubic_kernel_derivative(x_i - x_j))

    @ti.kernel
    def _kernel_compute_density_star(self):
        for i in range(self._n_particles):
            if self.particles_ng_reordered[i].active:
                delta = 0.0
                self.sh.for_all_neighbors(
                    i,
                    self.particles_reordered.pos,
                    self._support_radius,
                    delta,
                    self._task_compute_density_star,
                )
                drho = self.particles_reordered[i].rho / self._density0 + self._substep_dt * delta
                self.particles_reordered[i].drho = ti.max(drho, 1.0)  # - 1.0

    @ti.func
    def density_solve_iteration_task(self, i, j, ret: ti.template()):
        # Fluid neighbors
        b_j = self.particles_reordered[j].drho - 1.0
        k_j = b_j * self.particles_reordered[j].dfsph_factor
        k_sum = (
            self._density0 / self._density0 * ret.k_i + k_j
        )  # TODO: make the neighbor density0 different for multiphase fluid
        if ti.abs(k_sum) > self._df_eps:
            grad_p_j = -self._p_vol * self.cubic_kernel_derivative(
                self.particles_reordered[i].pos - self.particles_reordered[j].pos
            )
            # Directly update velocities instead of storing pressure accelerations
            ret.dv -= (
                self._substep_dt * k_sum * grad_p_j
            )  # ki, kj already contain inverse density, i.e., density canceled if not mutiphase flow

    @ti.kernel
    def _kernel_density_solve_iteration(self):
        # Compute pressure forces
        for i in range(self._n_particles):
            if self.particles_ng_reordered[i].active:
                # Evaluate rhs
                b_i = self.particles_reordered[i].drho - 1.0
                k_i = b_i * self.particles_reordered[i].dfsph_factor

                ret = ti.Struct(dv=ti.Vector.zero(gs.ti_float, 3), k_i=k_i)

                # TODO: if warmstart
                # get kappa V
                self.sh.for_all_neighbors(
                    i, self.particles_reordered.pos, self._support_radius, ret, self.density_solve_iteration_task
                )
                self.particles_reordered[i].vel = self.particles_reordered[i].vel + ret.dv

    def _density_solve_iteration(self):
        self._kernel_density_solve_iteration()
        self._kernel_compute_density_star()
        density_err = self._kernel_compute_density_error(self._density0)
        return density_err / self._n_particles

    @ti.kernel
    def _kernel_multiply_time_step(self, field: ti.template(), time_step: float):
        for i in range(self._n_particles):
            if self.particles_ng_reordered[i].active:
                field[i] *= time_step

    def _density_solve(self, f: ti.i32):
        inv_dt2 = 1 / (self._substep_dt * self._substep_dt)

        # TODO: warm start

        # Compute density star
        self._kernel_compute_density_star()

        self._kernel_multiply_time_step(self.particles_reordered.dfsph_factor, inv_dt2)

        iteration = 0

        # Start solver
        avg_density_err = 0.0

        while iteration < self._df_max_den_iters:
            avg_density_err = self._density_solve_iteration()
            # Max allowed density fluctuation
            eta = self._df_max_error_den * 0.01 * self._density0
            if avg_density_err <= eta:
                break
            iteration += 1

        gs.logger.debug(f"DFSPH - iterations: {iteration} Avg density err: {avg_density_err:.4f} kg/m^3")

        # Multiply by h, the time step size has to be removed
        # to make the stiffness value independent
        # of the time step size

        # TODO: if warm start
        # also remove for kappa v

    # ------------------------------------------------------------------------------------
    # ------------------------------------- utils ----------------------------------------
    # ------------------------------------------------------------------------------------

    @ti.func
    def cubic_kernel(self, r_norm):
        res = ti.cast(0.0, gs.ti_float)
        h = self._support_radius
        # value of cubic spline smoothing kernel
        k = 1.0
        k = 8 / np.pi
        k /= h**3
        q = r_norm / h
        if q <= 1.0:
            if q <= 0.5:
                q2 = q * q
                q3 = q2 * q
                res = k * (6.0 * q3 - 6.0 * q2 + 1)
            else:
                res = k * 2 * ti.pow(1 - q, 3.0)
        return res

    @ti.func
    def cubic_kernel_derivative(self, r):
        h = self._support_radius
        # derivative of cubic spline smoothing kernel
        k = 1.0
        k = 8 / np.pi
        k = 6.0 * k / h**3
        r_norm = r.norm()
        q = r_norm / h
        res = ti.Vector.zero(gs.ti_float, 3)
        if r_norm > 1e-5 and q <= 1.0:
            grad_q = r / (r_norm * h)
            if q <= 0.5:
                res = k * q * (3.0 * q - 2.0) * grad_q
            else:
                factor = 1.0 - q
                res = k * (-factor * factor) * grad_q
        return res

    # ------------------------------------------------------------------------------------
    # ------------------------------------ stepping --------------------------------------
    # ------------------------------------------------------------------------------------

    def process_input(self, in_backward=False):
        for entity in self.entities:
            entity.process_input(in_backward=in_backward)

    def process_input_grad(self):
        for entity in self.entities[::-1]:
            entity.process_input_grad()

    def substep_pre_coupling(self, f):
        if self.is_active():
            self._kernel_reorder_particles(f)
            if self._pressure_solver == "WCSPH":
                self._kernel_compute_rho(f)
                self._kernel_compute_non_pressure_forces(f, self._sim.cur_t)
                self._kernel_compute_pressure_forces(f)
                self._kernel_advect_velocity(f)
            elif self._pressure_solver == "DFSPH":
                self._kernel_compute_rho(f)
                self._kernel_compute_DFSPH_factor(f)
                self._divergence_solve(f)
                self._kernel_advect_velocity(f)
                self._kernel_compute_non_pressure_forces(f, self._sim.cur_t)
                self._kernel_predict_velocity(f)
                self._density_solve(f)

    def substep_pre_coupling_grad(self, f):
        pass

    def substep_post_coupling(self, f):
        if self.is_active():
            self._kernel_advect_position(f)
            self._kernel_copy_from_reordered(f)

    def substep_post_coupling_grad(self, f):
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
    def _kernel_add_particles(
        self,
        f: ti.i32,
        active: ti.i32,
        particle_start: ti.i32,
        n_particles: ti.i32,
        mat_rho: ti.f32,
        mat_stiffness: ti.f32,
        mat_exponent: ti.f32,
        mat_mu: ti.f32,
        mat_gamma: ti.f32,
        pos: ti.types.ndarray(),
    ):
        for i in range(n_particles):
            i_global = i + particle_start
            for j in ti.static(range(3)):
                self.particles[i_global].pos[j] = pos[i, j]
            self.particles[i_global].vel = ti.Vector.zero(gs.ti_float, 3)
            self.particles[i_global].p = 0

            self.particles_ng[i_global].active = active

            self.particles_info[i_global].rho = mat_rho
            self.particles_info[i_global].stiffness = mat_stiffness
            self.particles_info[i_global].exponent = mat_exponent
            self.particles_info[i_global].mu = mat_mu
            self.particles_info[i_global].gamma = mat_gamma
            self.particles_info[i_global].mass = self._p_vol * mat_rho

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

            # we reset vel and acc when directly setting pos
            self.particles[i_global].vel = ti.Vector.zero(gs.ti_float, 3)
            self.particles[i_global].acc = ti.Vector.zero(gs.ti_float, 3)

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

            # we reset acc when directly setting vel
            self.particles[i_global].acc = ti.Vector.zero(gs.ti_float, 3)

    @ti.kernel
    def get_frame(
        self,
        f: ti.i32,
        pos: ti.types.ndarray(),
        vel: ti.types.ndarray(),
        active: ti.types.ndarray(),
    ):
        for i in range(self._n_particles):
            for j in ti.static(range(3)):
                pos[i, j] = self.particles[i].pos[j]
                vel[i, j] = self.particles[i].vel[j]
            active[i] = self.particles_ng[i].active

    @ti.kernel
    def set_frame(
        self,
        f: ti.i32,
        pos: ti.types.ndarray(),
        vel: ti.types.ndarray(),
        active: ti.types.ndarray(),
    ):
        for i in range(self._n_particles):
            for j in ti.static(range(3)):
                self.particles[i].pos[j] = pos[i, j]
                self.particles[i].vel[j] = vel[i, j]
            self.particles_ng[i].active = active[i]

    def set_state(self, f, state):
        if self.is_active():
            self.set_frame(f, state.pos, state.vel, state.active)

    def get_state(self, f):
        if self.is_active():
            state = SPHSolverState(self.scene)
            self.get_frame(f, state.pos, state.vel, state.active)
        else:
            state = None
        return state

    @ti.kernel
    def _kernel_update_render_fields(self, f: ti.i32):
        for i in range(self._n_particles):
            if self.particles_ng[i].active:
                self.particles_render[i].pos = self.particles[i].pos
                self.particles_render[i].vel = self.particles[i].vel
            else:
                self.particles_render[i].pos = gu.ti_nowhere()
            self.particles_render[i].active = self.particles_ng[i].active

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
    def p_vol(self):
        return self._p_vol

    @property
    def particle_size(self):
        return self._particle_size

    @property
    def particle_radius(self):
        return self._particle_size / 2.0

    @property
    def support_radius(self):
        return self._support_radius

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
