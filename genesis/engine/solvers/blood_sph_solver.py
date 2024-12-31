import taichi as ti

import genesis as gs
from genesis.engine.entities import MultiphaseSPHEntity
from .sph_solver import SPHSolver


@ti.data_oriented
class BloodSPHSolver(SPHSolver):
    """
    Specialized SPH Solver for Blood Simulation with Multiphase Support.
    """

    def __init__(self, scene, sim, options):
        super().__init__(scene, sim, options)

        # Additional parameters for multiphase simulation
        self._max_phases = 5  # Plasma, RBCs, WBCs, Platelets, Fibrin
        self._adhesion_strength = 0.1  # Tunable parameter for inter-phase adhesion

        # Initialize biochemical parameters
        self.activation_threshold = 10.0  # threshold for platelet activation
        self.activation_rate = 0.1  # Rate at which biochemical concentration increases
        self.diffusion_coefficient = 0.01  # Diffusion rate for biochemical species
        self.concentration_threshold = 1.0  # Threshold for phase transition

        # Clot phase properties (phase=4)
        self.clot_tau_y = 50.0  # Increased yield stress for clots
        self.clot_K = 2000.0
        self.clot_n = 0.8

    def init_particle_fields(self):
        # Extend the base particle structures to include phase and biochemical properties
        struct_particle_state = ti.types.struct(
            pos=gs.ti_vec3,  # position
            vel=gs.ti_vec3,  # velocity
            acc=gs.ti_vec3,  # acceleration
            rho=gs.ti_float,  # density
            p=gs.ti_float,  # pressure
            dfsph_factor=gs.ti_float,  # DFSPH use: Factor for Divergence and density solver
            drho=gs.ti_float,  # density derivative
            velocity_gradient=ti.types.matrix(
                3, 3, gs.ti_float
            ),  # Velocity gradient tensor
            shear_rate=gs.ti_float,  # Shear rate magnitude
        )

        struct_particle_state_ng = ti.types.struct(
            reordered_idx=gs.ti_int,
            active=gs.ti_int,
        )

        struct_particle_info = ti.types.struct(
            rho=gs.ti_float,  # rest density
            mass=gs.ti_float,  # mass
            stiffness=gs.ti_float,  # pressure stiffness
            exponent=gs.ti_float,  # pressure exponent
            mu=gs.ti_float,  # viscosity
            gamma=gs.ti_float,  # surface tension
            tau_y=gs.ti_float,  # yield stress
            K=gs.ti_float,  # consistency index
            n=gs.ti_float,  # flow behavior index
            phase=gs.ti_int,  # phase identifier
            biochemical_state=gs.ti_int,  # biochemical state (e.g., 0: inactive, 1: activated)
            concentration=gs.ti_float,  # concentration of biochemical species
        )

        struct_particle_state_render = ti.types.struct(
            pos=gs.ti_vec3,
            vel=gs.ti_vec3,
            active=gs.ti_int,
        )

        # Construct fields
        self.particles = struct_particle_state.field(
            shape=(self._n_particles,), needs_grad=False, layout=ti.Layout.SOA
        )
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

    def add_entity(self, idx, material, morph, surface, phase=0):
        """
        Adds a MultiphaseSPHEntity to the solver.

        Parameters:
            idx: Entity index.
            material: Material properties dictionary.
            morph: Morphology parameters.
            surface: Surface parameters.
            phase: Integer identifier for the phase/type of the entity.

        Returns:
            The created MultiphaseSPHEntity instance.
        """
        entity = MultiphaseSPHEntity(
            scene=self.scene,
            solver=self,
            material=material,
            morph=morph,
            surface=surface,
            particle_size=self._particle_size,
            idx=idx,
            particle_start=self.n_particles,
            phase=phase,
        )

        self.entities.append(entity)
        self.n_particles += entity.n_particles  # Update total particle count
        return entity

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
        # Additional material properties for multiphase simulation
        mat_tau_y: ti.f32 = 0.0,
        mat_K: ti.f32 = 0.0,
        mat_n: ti.f32 = 0.0,
        phase: ti.i32 = 0,
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
            self.particles_info[i_global].tau_y = mat_tau_y
            self.particles_info[i_global].K = mat_K
            self.particles_info[i_global].n = mat_n
            self.particles_info[i_global].phase = phase  # Assign phase
            self.particles_info[
                i_global
            ].biochemical_state = 0  # Initialize biochemical state
            self.particles_info[
                i_global
            ].concentration = 0.0  # Initialize concentration

    @ti.kernel
    def _kernel_set_particle_phase(
        self,
        f: ti.i32,
        particle_start: ti.i32,
        n_particles: ti.i32,
        phase: ti.i32,
    ):
        for i in range(n_particles):
            i_global = i + particle_start
            self.particles_info[i_global].phase = phase

    @ti.func
    def _task_compute_non_pressure_forces_multiphase(self, i, j, ret: ti.template):
        phase_i = self.particles_info_reordered[i].phase
        phase_j = self.particles_info_reordered[j].phase

        # Compute distance and direction
        d_ij = self.particles_reordered[i].pos - self.particles_reordered[j].pos
        r_norm = d_ij.norm()
        if r_norm > self._particle_size:
            e_ij = d_ij / r_norm
        else:
            e_ij = ti.Vector.zero(gs.ti_float, 3)

        # Surface Tension
        if r_norm > self._particle_size:
            ret -= (
                self.particles_info_reordered[i].gamma
                / self.particles_info_reordered[i].mass
                * self.particles_info_reordered[j].mass
                * d_ij
                * self.cubic_kernel(r_norm)
            )
        else:
            ret -= (
                self.particles_info_reordered[i].gamma
                / self.particles_info_reordered[i].mass
                * self.particles_info_reordered[j].mass
                * d_ij
                * self.cubic_kernel(self._particle_size)
            )

        # Viscosity Force based on HB Model
        shear_rate = self.particles_reordered[i].shear_rate

        tau_y = self.particles_info_reordered[i].tau_y
        K = self.particles_info_reordered[i].K
        n = self.particles_info_reordered[i].n

        eta = self.compute_viscosity(shear_rate, tau_y, K, n)

        f_v = (
            4
            * eta
            * self.particles_info_reordered[j].mass
            * self.particles_info_reordered[j].rho
            * self.cubic_kernel_derivative(d_ij)
        )
        ret += f_v

        # Inter-Phase Interaction Forces
        if phase_i != phase_j:
            # Example: Adhesion forces between plasma and RBCs
            adhesion_strength = self._adhesion_strength  # Tunable parameter
            ret += adhesion_strength * e_ij * self.cubic_kernel_derivative(d_ij)

        return ret

    @ti.kernel
    def _kernel_compute_non_pressure_forces_multiphase(self, f: ti.i32, t: ti.f32):
        for i in range(self._n_particles):
            if self.particles_ng_reordered[i].active:
                acc = ti.Vector.zero(gs.ti_float, 3)
                self.sh.for_all_neighbors(
                    i,
                    self.particles_reordered.pos,
                    self._support_radius,
                    acc,
                    self._task_compute_non_pressure_forces_multiphase,
                )

                # External force fields
                for i_ff in ti.static(range(len(self._ffs))):
                    acc += self._ffs[i_ff].get_acc(
                        self.particles_reordered[i].pos,
                        self.particles_reordered[i].vel,
                        t,
                    )

                self.particles_reordered[i].acc = acc

    @ti.func
    def compute_viscosity(self, shear_rate, tau_y, K, n):
        if shear_rate > 1e-5:
            shear_stress = tau_y + K * (shear_rate**n)
            eta = shear_stress / shear_rate
        else:
            eta = 1e6  # Represents solid-like behavior
        return eta

    @ti.func
    def _task_compute_velocity_gradient(self, i, j, grad: ti.template()):
        # Compute the distance vector between particles
        d_ij = self.particles_reordered[i].pos - self.particles_reordered[j].pos
        r_norm = d_ij.norm(gs.EPS)
        if r_norm > gs.EPS:
            e_ij = d_ij / r_norm
            # Compute the gradient contribution
            grad += (
                self.particles_reordered[j].vel.outer_product(e_ij)
            ) * self.cubic_kernel_derivative(d_ij).dot(e_ij)

    @ti.kernel
    def _kernel_compute_velocity_gradient_and_shear_rate(self):
        for i in range(self._n_particles):
            if self.particles_ng_reordered[i].active:
                # Initialize the velocity gradient tensor to zero
                grad = ti.Matrix.zero(gs.ti_float, 3, 3)
                self.sh.for_all_neighbors(
                    i,
                    self.particles_reordered.pos,
                    self._support_radius,
                    grad,
                    self._task_compute_velocity_gradient,
                )
                # Assign the computed velocity gradient tensor
                self.particles_reordered[i].velocity_gradient = grad
                # Compute the shear rate magnitude as the Frobenius norm of the velocity gradient tensor
                self.particles_reordered[i].shear_rate = grad.norm()

    @ti.kernel
    def _kernel_biochemical_reactions(self, t: ti.f32):
        for i in range(self._n_particles):
            if self.particles_ng_reordered[i].active:
                # Check if the particle is in plasma phase and shear rate exceeds the threshold
                if (
                    self.particles_info_reordered[i].phase == 0
                    and self.particles_reordered[i].shear_rate
                    > self.activation_threshold
                    and self.particles_info_reordered[i].biochemical_state == 0
                ):
                    # Activate the particle
                    self.particles_info_reordered[i].biochemical_state = 1  # Activated
                    # Increase concentration based on activation rate
                    self.particles_info_reordered[i].concentration += (
                        self.activation_rate * self._substep_dt
                    )

    @ti.func
    def _task_diffusion(self, i, j, diff: ti.template()):
        # Compute the distance vector and its norm
        d_ij = self.particles_reordered[i].pos - self.particles_reordered[j].pos
        r_norm = d_ij.norm(gs.EPS)
        # Compute the concentration difference
        concentration_diff = (
            self.particles_info_reordered[j].concentration
            - self.particles_info_reordered[i].concentration
        )
        # Diffusion contribution
        diff += (
            self.diffusion_coefficient * concentration_diff * self.cubic_kernel(r_norm)
        )

    @ti.kernel
    def _kernel_biochemical_diffusion(self, t: ti.f32):
        for i in range(self._n_particles):
            if self.particles_ng_reordered[i].active:
                concentration_diffusion = 0.0
                self.sh.for_all_neighbors(
                    i,
                    self.particles_reordered.pos,
                    self._support_radius,
                    concentration_diffusion,
                    self._task_diffusion,
                )
                # Update concentration based on diffusion
                self.particles_info_reordered[i].concentration += (
                    self.diffusion_coefficient
                    * concentration_diffusion
                    * self._substep_dt
                )

    @ti.kernel
    def _kernel_phase_transition_biochemical(self):
        for i in range(self._n_particles):
            if (
                self.particles_ng_reordered[i].active
                and self.particles_info_reordered[i].phase == 0
            ):
                if (
                    self.particles_info_reordered[i].biochemical_state == 1
                    and self.particles_info_reordered[i].concentration
                    > self.concentration_threshold
                ):
                    # Transition to clot phase (phase=4)
                    self.particles_info_reordered[i].phase = 4  # Clot phase
                    # Update material properties for clot
                    self.particles_info_reordered[i].tau_y = self.clot_tau_y
                    self.particles_info_reordered[i].K = self.clot_K
                    self.particles_info_reordered[i].n = self.clot_n
                    # Reset biochemical state and concentration
                    self.particles_info_reordered[i].biochemical_state = 0
                    self.particles_info_reordered[i].concentration = 0.0

    def substep_pre_coupling(self, f):
        if self.is_active():
            self._kernel_reorder_particles(f)
            if self._pressure_solver == "WCSPH":
                self._kernel_compute_rho(f)
                self._kernel_compute_non_pressure_forces_multiphase(f, self._sim.cur_t)
                self._kernel_compute_pressure_forces(f)
                self._kernel_compute_velocity_gradient_and_shear_rate()
                self._kernel_biochemical_reactions(self._sim.cur_t)
                self._kernel_biochemical_diffusion(self._sim.cur_t)
                self._kernel_phase_transition_biochemical()
                # TODO: How do we handle FSI?
                self._kernel_advect_velocity(f)
            elif self._pressure_solver == "DFSPH":
                self._kernel_compute_rho(f)
                self._kernel_compute_DFSPH_factor(f)
                self._divergence_solve(f)
                self._kernel_advect_velocity(f)
                self._kernel_compute_non_pressure_forces_multiphase(f, self._sim.cur_t)
                self._kernel_predict_velocity(f)
                self._density_solve(f)
