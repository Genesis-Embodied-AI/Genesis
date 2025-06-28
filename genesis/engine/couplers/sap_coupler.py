from typing import TYPE_CHECKING
import numpy as np
import taichi as ti

import genesis as gs
from genesis.options.solvers import SAPCouplerOptions
from genesis.repr_base import RBC
from genesis.engine.bvh import AABB, LBVH

if TYPE_CHECKING:
    from genesis.engine.simulator import Simulator


@ti.func
def tet_barycentric(p, tet_vertices):
    """
    Compute the barycentric coordinates of point p with respect to the tetrahedron defined by tet_vertices.
    tet_vertices is a matrix of shape (3, 4) where each column is a vertex of the tetrahedron.
    """
    v0 = tet_vertices[:, 0]
    v1 = tet_vertices[:, 1]
    v2 = tet_vertices[:, 2]
    v3 = tet_vertices[:, 3]

    # Compute the vectors from the vertices to the point p
    v1_p = p - v1
    v2_p = p - v2
    v3_p = p - v3

    # Compute the volumes of the tetrahedra formed by the point and the vertices
    vol_tet = ti.math.dot(v1 - v0, ti.math.cross(v2 - v0, v3 - v0))

    # Compute the barycentric coordinates
    b0 = -ti.math.dot(v1_p, ti.math.cross(v2 - v1, v3 - v1)) / vol_tet
    b1 = ti.math.dot(v2_p, ti.math.cross(v3 - v2, v0 - v2)) / vol_tet
    b2 = -ti.math.dot(v3_p, ti.math.cross(v0 - v3, v1 - v3)) / vol_tet
    b3 = 1.0 - b0 - b1 - b2

    return ti.Vector([b0, b1, b2, b3])


@ti.data_oriented
class SAPCoupler(RBC):
    """
    This class handles all the coupling between different solvers using the
    Semi-Analytic Primal (SAP) contact solver used in Drake.

    Note
    ----
    Paper reference: https://arxiv.org/abs/2110.10107
    Drake reference: https://drake.mit.edu/release_notes/v1.5.0.html
    """

    # ------------------------------------------------------------------------------------
    # --------------------------------- Initialization -----------------------------------
    # ------------------------------------------------------------------------------------

    def __init__(
        self,
        simulator: "Simulator",
        options: "SAPCouplerOptions",
    ) -> None:
        self.sim = simulator
        self.options = options
        self.rigid_solver = self.sim.rigid_solver
        self.fem_solver = self.sim.fem_solver
        self._n_sap_iterations = options.n_sap_iterations
        self._sap_threshold = options.sap_threshold
        self._n_pcg_iterations = options.n_pcg_iterations
        self._pcg_threshold = options.pcg_threshold
        self._n_linesearch_iterations = options.n_linesearch_iterations
        self._linesearch_c = options.linesearch_c
        self._linesearch_tau = options.linesearch_tau

    # ------------------------------------------------------------------------------------
    # --------------------------------- Initialization -----------------------------------
    # ------------------------------------------------------------------------------------

    def build(self) -> None:
        self._B = self.sim._B
        self._rigid_fem = self.rigid_solver.is_active() and self.fem_solver.is_active() and self.options.rigid_fem

        if self.fem_solver.is_active():
            if self.fem_solver._use_implicit_solver is False:
                gs.raise_exception(
                    "SAPCoupler requires FEM to use implicit solver. "
                    "Please set `use_implicit_solver=True` in FEM options."
                )
            self.init_fem_fields()

        self.init_sap_fields()
        self.init_pcg_fields()
        self.init_linesearch_fields()

    def reset(self):
        pass

    def init_fem_fields(self):
        fem_solver = self.fem_solver
        self.fem_pressure = ti.field(gs.ti_float, shape=(fem_solver.n_vertices))
        fem_pressure_np = np.concatenate([fem_entity.pressure_field_np for fem_entity in fem_solver.entities])
        self.fem_pressure.from_numpy(fem_pressure_np)
        self.fem_pressure_gradient = ti.field(gs.ti_vec3, shape=(fem_solver._B, fem_solver.n_elements))
        self.sap_contact_info_type = ti.types.struct(
            k=gs.ti_float,  # contact stiffness
            phi0=gs.ti_float,  # initial signed distance
            fn0=gs.ti_float,  # initial normal force magnitude
            taud=gs.ti_float,  # dissipation time scale
            Rn=gs.ti_float,  # Regularitaion for normal
            Rt=gs.ti_float,  # Regularitaion for tangential
            vn_hat=gs.ti_float,  # Stablization for normal velocity
            mu=gs.ti_float,  # friction coefficient
            mu_hat=gs.ti_float,  # friction coefficient regularized
            mu_factor=gs.ti_float,  # friction coefficient factor, 1/(1+mu_tilde**2)
            energy=gs.ti_float,  # energy
            gamma=gs.ti_vec3,  # dual variable for normal
            G=gs.ti_mat3,  # Hessian matrix
        )
        self.fem_floor_contact_pair_type = ti.types.struct(
            active=gs.ti_int,  # whether the contact pair is active
            batch_idx=gs.ti_int,  # batch index
            geom_idx=gs.ti_int,  # index of the FEM element
            intersection_code=gs.ti_int,  # intersection code for the element
            distance=gs.ti_vec4,  # distance vector for the element
            barycentric=gs.ti_vec4,  # barycentric coordinates of the contact point
            sap_info=self.sap_contact_info_type,  # contact info
        )
        self.max_fem_floor_contact_pairs = fem_solver.n_surfaces * fem_solver._B
        self.n_fem_floor_contact_pairs = ti.field(gs.ti_int, shape=())
        self.fem_floor_contact_pairs = self.fem_floor_contact_pair_type.field(shape=(self.max_fem_floor_contact_pairs,))

        self.fem_self_contact_pair_candidate_type = ti.types.struct(
            batch_idx=gs.ti_int,  # batch index
            normal=gs.ti_vec3,  # contact plane normal
            x=gs.ti_vec3,  # a point on the contact plane
            geom_idx0=gs.ti_int,  # index of the FEM element0
            intersection_code0=gs.ti_int,  # intersection code for element0
            distance0=gs.ti_vec4,  # distance vector for element0
            geom_idx1=gs.ti_int,  # index of the FEM element1
        )
        self.fem_self_contact_pair_type = ti.types.struct(
            batch_idx=gs.ti_int,  # batch index
            normal=gs.ti_vec3,  # contact plane normal
            tangent0=gs.ti_vec3,  # contact plane tangent0
            tangent1=gs.ti_vec3,  # contact plane tangent1
            geom_idx0=gs.ti_int,  # index of the FEM element0
            geom_idx1=gs.ti_int,  # index of the FEM element1
            barycentric0=gs.ti_vec4,  # barycentric coordinates of the contact point in tet 0
            barycentric1=gs.ti_vec4,  # barycentric coordinates of the contact point in tet 1
            sap_info=self.sap_contact_info_type,  # contact info
        )
        self.max_fem_self_contact_pair_candidates = fem_solver.n_surfaces * fem_solver._B * 8
        self.max_fem_self_contact_pairs = fem_solver.n_surfaces * fem_solver._B
        self.n_fem_self_contact_pair_candidates = ti.field(gs.ti_int, shape=())
        self.n_fem_self_contact_pairs = ti.field(gs.ti_int, shape=())
        self.fem_self_contact_pair_candidates = self.fem_self_contact_pair_candidate_type.field(
            shape=(self.max_fem_self_contact_pair_candidates,)
        )
        self.fem_self_contact_pairs = self.fem_self_contact_pair_type.field(shape=(self.max_fem_self_contact_pairs,))
        # TODO change to surface element only instead of all elements
        self.fem_aabb = AABB(fem_solver._B, fem_solver.n_elements)
        self.fem_bvh = LBVH(self.fem_aabb, max_n_query_result_per_aabb=32)
        self.fem_bvh.register_fem_tet_filter(fem_solver)

        # Lookup table for marching tetrahedra edges
        kMarchingTetsEdgeTable_np = np.array(
            [
                [-1, -1, -1, -1],
                [0, 3, 2, -1],
                [0, 1, 4, -1],
                [4, 3, 2, 1],
                [1, 2, 5, -1],
                [0, 3, 5, 1],
                [0, 2, 5, 4],
                [3, 5, 4, -1],
                [3, 4, 5, -1],
                [4, 5, 2, 0],
                [1, 5, 3, 0],
                [1, 5, 2, -1],
                [1, 2, 3, 4],
                [0, 4, 1, -1],
                [0, 2, 3, -1],
                [-1, -1, -1, -1],
            ]
        )
        self.kMarchingTetsEdgeTable = ti.field(gs.ti_ivec4, shape=kMarchingTetsEdgeTable_np.shape[0])
        self.kMarchingTetsEdgeTable.from_numpy(kMarchingTetsEdgeTable_np)

        kTetEdges_np = np.array([[0, 1], [1, 2], [2, 0], [0, 3], [1, 3], [2, 3]])
        self.kTetEdges = ti.field(gs.ti_ivec2, shape=kTetEdges_np.shape[0])
        self.kTetEdges.from_numpy(kTetEdges_np)

    def init_sap_fields(self):
        self.batch_active = ti.field(
            dtype=ti.u1,
            shape=self.sim._B,
            needs_grad=False,
        )
        self.v = ti.field(gs.ti_vec3, shape=(self.fem_solver._B, self.fem_solver.n_vertices))
        self.v_diff = ti.field(gs.ti_vec3, shape=(self.fem_solver._B, self.fem_solver.n_vertices))
        self.gradient = ti.field(gs.ti_vec3, shape=(self.fem_solver._B, self.fem_solver.n_vertices))

    def init_pcg_fields(self):
        self.batch_pcg_active = ti.field(
            dtype=ti.u1,
            shape=self.sim._B,
            needs_grad=False,
        )

        pcg_state = ti.types.struct(
            rTr=gs.ti_float,
            rTz=gs.ti_float,
            rTr_new=gs.ti_float,
            rTz_new=gs.ti_float,
            pTAp=gs.ti_float,
            alpha=gs.ti_float,
            beta=gs.ti_float,
        )

        self.pcg_state = pcg_state.field(
            shape=self.sim._B,
            needs_grad=False,
            layout=ti.Layout.SOA,
        )

        pcg_state_v = ti.types.struct(
            diag3x3=gs.ti_mat3,  # diagonal 3-by-3 block of the hessian
            prec=gs.ti_mat3,  # preconditioner
            x=gs.ti_vec3,  # solution vector
            r=gs.ti_vec3,  # residual vector
            z=gs.ti_vec3,  # preconditioned residual vector
            p=gs.ti_vec3,  # search direction vector
            Ap=gs.ti_vec3,  # matrix-vector product
        )

        self.pcg_state_v = pcg_state_v.field(
            shape=(self.sim._B, self.fem_solver.n_vertices),
            needs_grad=False,
            layout=ti.Layout.SOA,
        )

    def init_linesearch_fields(self):
        self.batch_linesearch_active = ti.field(
            dtype=ti.u1,
            shape=self.sim._B,
            needs_grad=False,
        )

        linesearch_state = ti.types.struct(
            prev_energy=gs.ti_float,
            energy=gs.ti_float,
            step_size=gs.ti_float,
            m=gs.ti_float,
        )

        self.linesearch_state = linesearch_state.field(
            shape=self.sim._B,
            needs_grad=False,
            layout=ti.Layout.SOA,
        )

        linesearch_state_v = ti.types.struct(
            x_prev=gs.ti_vec3,  # solution vector
        )

        self.linesearch_state_v = linesearch_state_v.field(
            shape=(self.sim._B, self.fem_solver.n_vertices),
            needs_grad=False,
            layout=ti.Layout.SOA,
        )

    # ------------------------------------------------------------------------------------
    # -------------------------------------- Main ----------------------------------------
    # ------------------------------------------------------------------------------------

    def preprocess(self, f):
        pass

    def couple(self, f):
        self.has_contact = False
        if self.fem_solver.is_active():
            self.fem_compute_pressure_gradient(f)

            self.fem_floor_detection(f)
            if self.n_fem_floor_contact_pairs[None] > self.max_fem_floor_contact_pairs:
                raise ValueError(
                    f"Number of floor contact pairs {self.n_fem_floor_contact_pairs[None]} "
                    f"exceeds max_fem_floor_contact_pairs {self.max_fem_floor_contact_pairs}"
                )
            self.has_fem_floor_contact = self.n_fem_floor_contact_pairs[None] > 0
            self.has_contact = self.has_contact or self.has_fem_floor_contact

            self.fem_self_detection(f)
            self.has_fem_self_contact = self.n_fem_self_contact_pairs[None] > 0
            self.has_contact = self.has_contact or self.has_fem_self_contact

        if self.has_contact:
            self.sap_solve(f)
            self.update_vel(f)

    def couple_grad(self, f):
        gs.raise_exception("couple_grad is not available for HydroelasticCoupler. Please use Coupler instead.")

    @ti.kernel
    def update_vel(self, f: ti.i32):
        fem_solver = self.fem_solver
        for i_b, i_v in ti.ndrange(fem_solver._B, fem_solver.n_vertices):
            self.fem_solver.elements_v[f + 1, i_v, i_b].vel = self.v[i_b, i_v]

    @ti.kernel
    def fem_compute_pressure_gradient(self, f: ti.i32):
        fem_solver = self.fem_solver
        for i_b, i_e in ti.ndrange(fem_solver._B, fem_solver.n_elements):
            grad = ti.static(self.fem_pressure_gradient)
            grad[i_b, i_e].fill(0.0)

            for i in ti.static(range(4)):
                i_v0 = fem_solver.elements_i[i_e].el2v[i]
                i_v1 = fem_solver.elements_i[i_e].el2v[(i + 1) % 4]
                i_v2 = fem_solver.elements_i[i_e].el2v[(i + 2) % 4]
                i_V3 = fem_solver.elements_i[i_e].el2v[(i + 3) % 4]
                pos_v0 = fem_solver.elements_v[f + 1, i_v0, i_b].pos
                pos_v1 = fem_solver.elements_v[f + 1, i_v1, i_b].pos
                pos_v2 = fem_solver.elements_v[f + 1, i_v2, i_b].pos
                pos_v3 = fem_solver.elements_v[f + 1, i_V3, i_b].pos

                e10 = pos_v0 - pos_v1
                e12 = pos_v2 - pos_v1
                e13 = pos_v3 - pos_v1

                area_vector = e12.cross(e13)  # area vector of the triangle formed by v1, v2, v3
                signed_volume = area_vector.dot(e10)  # signed volume of the tetrahedron formed by v0, v1, v2, v3
                if ti.abs(signed_volume) > gs.EPS:
                    grad_i = area_vector / signed_volume
                    grad[i_b, i_e] += grad_i * self.fem_pressure[i_v0]

    # ------------------------------------------------------------------------------------
    # ------------------------------------- Solve ----------------------------------------
    # ------------------------------------------------------------------------------------

    def sap_solve(self, f):
        self.init_sap_solve(f)
        for i in range(self._n_sap_iterations):
            # init gradient and preconditioner
            self.compute_non_contact_gradient_diag(f, i)

            # compute contact hessian and gradient
            self.compute_contact_gradient_hessian_diag_prec()

            # solve for the vertex velocity
            self.pcg_solve()

            # line search
            self.linesearch(f)
            # TODO add convergence check

    def init_sap_solve(self, f: ti.i32):
        self.init_v(f)
        self.batch_active.fill(1)
        if self.has_fem_floor_contact:
            self.compute_fem_floor_regularization()
        if self.has_fem_self_contact:
            self.compute_fem_self_regularization()

    @ti.kernel
    def init_v(self, f: ti.i32):
        fem_solver = self.fem_solver
        for i_b, i_v in ti.ndrange(fem_solver._B, fem_solver.n_vertices):
            self.v[i_b, i_v] = fem_solver.elements_v[f + 1, i_v, i_b].vel

    def compute_non_contact_gradient_diag(self, f: ti.i32, iter: int):
        self.init_non_contact_gradient_diag(f)
        # No need to do this for iter=0 because v=v* and A(v-v*) = 0
        if iter > 0:
            self.compute_inertia_elastic_gradient()

    @ti.kernel
    def init_non_contact_gradient_diag(self, f: ti.i32):
        fem_solver = self.fem_solver
        dt2 = fem_solver._substep_dt**2
        for i_b, i_v in ti.ndrange(fem_solver._B, fem_solver.n_vertices):
            self.gradient[i_b, i_v].fill(0.0)
            # was using position now using velocity, need to multiply dt^2
            self.pcg_state_v[i_b, i_v].diag3x3 = fem_solver.pcg_state_v[i_b, i_v].diag3x3 * dt2
            self.v_diff[i_b, i_v] = self.v[i_b, i_v] - fem_solver.elements_v[f + 1, i_v, i_b].vel

    @ti.kernel
    def compute_inertia_elastic_gradient(self):
        fem_solver = self.fem_solver
        dt2 = fem_solver._substep_dt**2
        damping_alpha_dt = fem_solver._damping_alpha * fem_solver._substep_dt
        damping_alpha_factor = damping_alpha_dt + 1.0
        damping_beta_over_dt = fem_solver._damping_beta / fem_solver._substep_dt
        damping_beta_factor = damping_beta_over_dt + 1.0

        for i_b, i_v in ti.ndrange(fem_solver._B, fem_solver.n_vertices):
            self.gradient[i_b, i_v] = (
                fem_solver.elements_v_info[i_v].mass_over_dt2 * self.v_diff[i_b, i_v] * dt2 * damping_alpha_factor
            )

        for i_b, i_e in ti.ndrange(fem_solver._B, fem_solver.n_elements):
            V_dt2 = fem_solver.elements_i[i_e].V * dt2
            B = fem_solver.elements_i[i_e].B
            s = -B[0, :] - B[1, :] - B[2, :]  # s is the negative sum of B rows
            p9 = ti.Vector([0.0] * 9, dt=gs.ti_float)
            i_v0, i_v1, i_v2, i_v3 = fem_solver.elements_i[i_e].el2v

            for i in ti.static(range(3)):
                p9[i * 3 : i * 3 + 3] = (
                    B[0, i] * self.v_diff[i_b, i_v0]
                    + B[1, i] * self.v_diff[i_b, i_v1]
                    + B[2, i] * self.v_diff[i_b, i_v2]
                    + s[i] * self.v_diff[i_b, i_v3]
                )

            new_p9 = ti.Vector([0.0] * 9, dt=gs.ti_float)

            for i in ti.static(range(3)):
                new_p9[i * 3 : i * 3 + 3] = (
                    fem_solver.elements_el_hessian[i_b, i, 0, i_e] @ p9[0:3]
                    + fem_solver.elements_el_hessian[i_b, i, 1, i_e] @ p9[3:6]
                    + fem_solver.elements_el_hessian[i_b, i, 2, i_e] @ p9[6:9]
                )

            # atomic
            self.gradient[i_b, i_v0] += (
                (B[0, 0] * new_p9[0:3] + B[0, 1] * new_p9[3:6] + B[0, 2] * new_p9[6:9]) * V_dt2 * damping_beta_factor
            )
            self.gradient[i_b, i_v1] += (
                (B[1, 0] * new_p9[0:3] + B[1, 1] * new_p9[3:6] + B[1, 2] * new_p9[6:9]) * V_dt2 * damping_beta_factor
            )
            self.gradient[i_b, i_v2] += (
                (B[2, 0] * new_p9[0:3] + B[2, 1] * new_p9[3:6] + B[2, 2] * new_p9[6:9]) * V_dt2 * damping_beta_factor
            )
            self.gradient[i_b, i_v3] += (
                (s[0] * new_p9[0:3] + s[1] * new_p9[3:6] + s[2] * new_p9[6:9]) * V_dt2 * damping_beta_factor
            )

    def compute_contact_gradient_hessian_diag_prec(self):
        if self.has_fem_floor_contact:
            self.compute_fem_floor_gradient_hessian_diag()
        if self.has_fem_self_contact:
            self.compute_fem_self_gradient_hessian_diag()
        self.compute_preconditioner()

    @ti.kernel
    def compute_preconditioner(self):
        fem_solver = self.fem_solver
        for i_b, i_v in ti.ndrange(fem_solver._B, fem_solver.n_vertices):
            if not self.batch_active[i_b]:
                continue
            self.pcg_state_v[i_b, i_v].prec = self.pcg_state_v[i_b, i_v].diag3x3.inverse()

    def compute_Ap(self):
        self.compute_inertia_elastic_Ap()
        # Contact
        if self.has_fem_floor_contact:
            self.compute_fem_floor_Ap()
        if self.has_fem_self_contact:
            self.compute_fem_self_Ap()

    @ti.kernel
    def compute_inertia_elastic_Ap(self):
        fem_solver = self.fem_solver
        dt2 = fem_solver._substep_dt**2
        damping_alpha_dt = fem_solver._damping_alpha * fem_solver._substep_dt
        damping_alpha_factor = damping_alpha_dt + 1.0
        damping_beta_over_dt = fem_solver._damping_beta / fem_solver._substep_dt
        damping_beta_factor = damping_beta_over_dt + 1.0

        # Inerita
        for i_b, i_v in ti.ndrange(fem_solver._B, fem_solver.n_vertices):
            if not self.batch_pcg_active[i_b]:
                continue
            self.pcg_state_v[i_b, i_v].Ap = (
                fem_solver.elements_v_info[i_v].mass_over_dt2
                * self.pcg_state_v[i_b, i_v].p
                * dt2
                * damping_alpha_factor
            )

        # Elasticity
        for i_b, i_e in ti.ndrange(fem_solver._B, fem_solver.n_elements):
            if not self.batch_pcg_active[i_b]:
                continue
            V_dt2 = fem_solver.elements_i[i_e].V * dt2
            B = fem_solver.elements_i[i_e].B
            s = -B[0, :] - B[1, :] - B[2, :]  # s is the negative sum of B rows
            p9 = ti.Vector([0.0] * 9, dt=gs.ti_float)
            i_v0, i_v1, i_v2, i_v3 = fem_solver.elements_i[i_e].el2v

            for i in ti.static(range(3)):
                p9[i * 3 : i * 3 + 3] = (
                    B[0, i] * self.pcg_state_v[i_b, i_v0].p
                    + B[1, i] * self.pcg_state_v[i_b, i_v1].p
                    + B[2, i] * self.pcg_state_v[i_b, i_v2].p
                    + s[i] * self.pcg_state_v[i_b, i_v3].p
                )

            new_p9 = ti.Vector([0.0] * 9, dt=gs.ti_float)

            for i in ti.static(range(3)):
                new_p9[i * 3 : i * 3 + 3] = (
                    fem_solver.elements_el_hessian[i_b, i, 0, i_e] @ p9[0:3]
                    + fem_solver.elements_el_hessian[i_b, i, 1, i_e] @ p9[3:6]
                    + fem_solver.elements_el_hessian[i_b, i, 2, i_e] @ p9[6:9]
                )

            # atomic
            self.pcg_state_v[i_b, i_v0].Ap += (
                (B[0, 0] * new_p9[0:3] + B[0, 1] * new_p9[3:6] + B[0, 2] * new_p9[6:9]) * V_dt2 * damping_beta_factor
            )
            self.pcg_state_v[i_b, i_v1].Ap += (
                (B[1, 0] * new_p9[0:3] + B[1, 1] * new_p9[3:6] + B[1, 2] * new_p9[6:9]) * V_dt2 * damping_beta_factor
            )
            self.pcg_state_v[i_b, i_v2].Ap += (
                (B[2, 0] * new_p9[0:3] + B[2, 1] * new_p9[3:6] + B[2, 2] * new_p9[6:9]) * V_dt2 * damping_beta_factor
            )
            self.pcg_state_v[i_b, i_v3].Ap += (
                (s[0] * new_p9[0:3] + s[1] * new_p9[3:6] + s[2] * new_p9[6:9]) * V_dt2 * damping_beta_factor
            )

    @ti.kernel
    def init_pcg_solve(self):
        fem_solver = self.fem_solver
        for i_b in ti.ndrange(self._B):
            self.batch_pcg_active[i_b] = self.batch_active[i_b]
            if not self.batch_pcg_active[i_b]:
                continue
            self.pcg_state[i_b].rTr = 0.0
            self.pcg_state[i_b].rTz = 0.0
        for i_b, i_v in ti.ndrange(self._B, fem_solver.n_vertices):
            if not self.batch_pcg_active[i_b]:
                continue
            self.pcg_state_v[i_b, i_v].x = 0
            self.pcg_state_v[i_b, i_v].r = -self.gradient[i_b, i_v]
            self.pcg_state_v[i_b, i_v].z = self.pcg_state_v[i_b, i_v].prec @ self.pcg_state_v[i_b, i_v].r
            self.pcg_state_v[i_b, i_v].p = self.pcg_state_v[i_b, i_v].z
            ti.atomic_add(self.pcg_state[i_b].rTr, self.pcg_state_v[i_b, i_v].r.dot(self.pcg_state_v[i_b, i_v].r))
            ti.atomic_add(self.pcg_state[i_b].rTz, self.pcg_state_v[i_b, i_v].r.dot(self.pcg_state_v[i_b, i_v].z))
        for i_b in ti.ndrange(self._B):
            if not self.batch_pcg_active[i_b]:
                continue
            self.batch_pcg_active[i_b] = self.pcg_state[i_b].rTr > self._pcg_threshold

    def one_pcg_iter(self):
        self.compute_Ap()
        self._kernel_one_pcg_iter()

    @ti.kernel
    def _kernel_one_pcg_iter(self):
        fem_solver = self.fem_solver
        # compute pTAp
        for i_b in ti.ndrange(self._B):
            if not self.batch_pcg_active[i_b]:
                continue
            self.pcg_state[i_b].pTAp = 0.0
        for i_b, i_v in ti.ndrange(self._B, fem_solver.n_vertices):
            if not self.batch_pcg_active[i_b]:
                continue
            ti.atomic_add(self.pcg_state[i_b].pTAp, self.pcg_state_v[i_b, i_v].p.dot(self.pcg_state_v[i_b, i_v].Ap))

        # compute alpha and update x, r, z, rTr, rTz
        for i_b in ti.ndrange(self._B):
            if not self.batch_pcg_active[i_b]:
                continue
            self.pcg_state[i_b].alpha = self.pcg_state[i_b].rTz / self.pcg_state[i_b].pTAp
            self.pcg_state[i_b].rTr_new = 0.0
            self.pcg_state[i_b].rTz_new = 0.0
        for i_b, i_v in ti.ndrange(self._B, fem_solver.n_vertices):
            if not self.batch_pcg_active[i_b]:
                continue
            self.pcg_state_v[i_b, i_v].x += self.pcg_state[i_b].alpha * self.pcg_state_v[i_b, i_v].p
            self.pcg_state_v[i_b, i_v].r -= self.pcg_state[i_b].alpha * self.pcg_state_v[i_b, i_v].Ap
            self.pcg_state_v[i_b, i_v].z = self.pcg_state_v[i_b, i_v].prec @ self.pcg_state_v[i_b, i_v].r
            ti.atomic_add(self.pcg_state[i_b].rTr_new, self.pcg_state_v[i_b, i_v].r.dot(self.pcg_state_v[i_b, i_v].r))
            ti.atomic_add(self.pcg_state[i_b].rTz_new, self.pcg_state_v[i_b, i_v].r.dot(self.pcg_state_v[i_b, i_v].z))

        # check convergence
        for i_b in ti.ndrange(self._B):
            if not self.batch_pcg_active[i_b]:
                continue
            self.batch_pcg_active[i_b] = self.pcg_state[i_b].rTr_new > self._pcg_threshold

        # update beta, rTr, rTz
        for i_b in ti.ndrange(self._B):
            if not self.batch_pcg_active[i_b]:
                continue
            self.pcg_state[i_b].beta = self.pcg_state[i_b].rTr_new / self.pcg_state[i_b].rTr
            self.pcg_state[i_b].rTr = self.pcg_state[i_b].rTr_new
            self.pcg_state[i_b].rTz = self.pcg_state[i_b].rTz_new

        # update p
        for i_b, i_v in ti.ndrange(self._B, fem_solver.n_vertices):
            if not self.batch_pcg_active[i_b]:
                continue
            self.pcg_state_v[i_b, i_v].p = (
                self.pcg_state_v[i_b, i_v].z + self.pcg_state[i_b].beta * self.pcg_state_v[i_b, i_v].p
            )

    def pcg_solve(self):
        self.init_pcg_solve()
        for i in range(self._n_pcg_iterations):
            self.one_pcg_iter()

    def compute_total_energy(self, f: ti.i32, energy):
        self.compute_inertia_elastic_energy(f, energy)
        # Contact
        if self.has_fem_floor_contact:
            self.compute_fem_floor_energy(energy)
        if self.has_fem_self_contact:
            self.compute_fem_self_energy(energy)

    @ti.kernel
    def compute_inertia_elastic_energy(self, f: ti.i32, energy: ti.template()):
        fem_solver = self.fem_solver
        dt2 = fem_solver._substep_dt**2
        damping_alpha_dt = fem_solver._damping_alpha * fem_solver._substep_dt
        damping_alpha_factor = damping_alpha_dt + 1.0
        damping_beta_over_dt = fem_solver._damping_beta / fem_solver._substep_dt
        damping_beta_factor = damping_beta_over_dt + 1.0

        for i_b in ti.ndrange(self._B):
            if not self.batch_linesearch_active[i_b]:
                continue
            energy[i_b] = 0.0

        # Inertia
        for i_b, i_v in ti.ndrange(self._B, fem_solver.n_vertices):
            if not self.batch_linesearch_active[i_b]:
                continue
            self.v_diff[i_b, i_v] = self.v[i_b, i_v] - fem_solver.elements_v[f + 1, i_v, i_b].vel
            energy[i_b] += (
                0.5
                * fem_solver.elements_v_info[i_v].mass_over_dt2
                * self.v_diff[i_b, i_v].dot(self.v_diff[i_b, i_v])
                * dt2
                * damping_alpha_factor
            )

        # Elastic
        for i_b, i_e in ti.ndrange(self._B, fem_solver.n_elements):
            if not self.batch_linesearch_active[i_b]:
                continue

            V_dt2 = fem_solver.elements_i[i_e].V * dt2
            B = fem_solver.elements_i[i_e].B
            s = -B[0, :] - B[1, :] - B[2, :]  # s is the negative sum of B rows
            p9 = ti.Vector.zero(gs.ti_float, 9)
            i_v0, i_v1, i_v2, i_v3 = fem_solver.elements_i[i_e].el2v

            for i in ti.static(range(3)):
                p9[i * 3 : i * 3 + 3] = (
                    B[0, i] * self.v_diff[i_b, i_v0]
                    + B[1, i] * self.v_diff[i_b, i_v1]
                    + B[2, i] * self.v_diff[i_b, i_v2]
                    + s[i] * self.v_diff[i_b, i_v3]
                )

            H9_p9 = ti.Vector.zero(gs.ti_float, 9)

            for i in ti.static(range(3)):
                H9_p9[i * 3 : i * 3 + 3] = (
                    fem_solver.elements_el_hessian[i_b, i, 0, i_e] @ p9[0:3]
                    + fem_solver.elements_el_hessian[i_b, i, 1, i_e] @ p9[3:6]
                    + fem_solver.elements_el_hessian[i_b, i, 2, i_e] @ p9[6:9]
                )

            energy[i_b] += 0.5 * V_dt2 * p9.dot(H9_p9) * damping_beta_factor

    def init_linesearch(self, f: ti.i32):
        self._kernel_init_linesearch()
        self.compute_total_energy(f, self.linesearch_state.prev_energy)

    @ti.kernel
    def _kernel_init_linesearch(self):
        fem_solver = self.fem_solver
        for i_b in ti.ndrange(self._B):
            self.batch_linesearch_active[i_b] = self.batch_active[i_b]
            if not self.batch_linesearch_active[i_b]:
                continue
            self.linesearch_state[i_b].step_size = 1.0 / ti.static(self._linesearch_tau)
            self.linesearch_state[i_b].m = 0.0

        # x_prev, m
        for i_b, i_v in ti.ndrange(self._B, fem_solver.n_vertices):
            if not self.batch_linesearch_active[i_b]:
                continue
            self.linesearch_state_v[i_b, i_v].x_prev = self.v[i_b, i_v]
            self.linesearch_state[i_b].m += self.pcg_state_v[i_b, i_v].x.dot(self.gradient[i_b, i_v])

    def one_linesearch_iter(self, f: ti.i32):
        self.update_velocity()
        self.compute_total_energy(f, self.linesearch_state.energy)
        self.check_linesearch_convergence()

    @ti.kernel
    def update_velocity(self):
        fem_solver = self.fem_solver

        # update vel
        for i_b, i_v in ti.ndrange(self._B, fem_solver.n_vertices):
            if not self.batch_linesearch_active[i_b]:
                continue
            self.v[i_b, i_v] = (
                self.linesearch_state_v[i_b, i_v].x_prev
                + self.linesearch_state[i_b].step_size * self.pcg_state_v[i_b, i_v].x
            )

    @ti.kernel
    def check_linesearch_convergence(self):
        # check condition
        for i_b in ti.ndrange(self._B):
            if not self.batch_linesearch_active[i_b]:
                continue
            self.batch_linesearch_active[i_b] = (
                self.linesearch_state[i_b].energy
                > self.linesearch_state[i_b].prev_energy
                + self._linesearch_c * self.linesearch_state[i_b].step_size * self.linesearch_state[i_b].m
            )
            if not self.batch_linesearch_active[i_b]:
                continue
            self.linesearch_state[i_b].step_size *= self._linesearch_tau

    def linesearch(self, f: ti.i32):
        """
        Note
        ------
        https://en.wikipedia.org/wiki/Backtracking_line_search#Algorithm
        """
        self.init_linesearch(f)
        for i in range(self._n_linesearch_iterations):
            self.one_linesearch_iter(f)

    # ------------------------------------------------------------------------------------
    # --------------------------------- Contact Common -----------------------------------
    # ------------------------------------------------------------------------------------

    @ti.func
    def compute_contact_gamma_G(self, sap_info, i_p, vc):
        y = ti.Vector([0.0, 0.0, sap_info[i_p].vn_hat]) - vc
        y[0] /= sap_info[i_p].Rt
        y[1] /= sap_info[i_p].Rt
        y[2] /= sap_info[i_p].Rn
        yr = y[:2].norm(gs.EPS)
        yn = y[2]

        t_hat = y[:2] / yr
        contact_mode = self.compute_contact_mode(sap_info[i_p].mu, sap_info[i_p].mu_hat, yr, yn)
        sap_info[i_p].gamma.fill(0.0)
        sap_info[i_p].G.fill(0.0)
        if contact_mode == 0:  # Sticking
            sap_info[i_p].gamma = y
            sap_info[i_p].G[0, 0] = 1.0 / sap_info[i_p].Rt
            sap_info[i_p].G[1, 1] = 1.0 / sap_info[i_p].Rt
            sap_info[i_p].G[2, 2] = 1.0 / sap_info[i_p].Rn
        elif contact_mode == 1:  # Sliding
            gn = (yn + sap_info[i_p].mu_hat * yr) * sap_info[i_p].mu_factor
            gt = sap_info[i_p].mu * gn * t_hat
            sap_info[i_p].gamma = ti.Vector([gt[0], gt[1], gn])
            P = t_hat.outer_product(t_hat)
            Pperp = ti.Matrix.identity(gs.ti_float, 2) - P
            dgt_dyt = sap_info[i_p].mu * (gn / yr * Pperp + sap_info[i_p].mu_hat * sap_info[i_p].mu_factor * P)
            dgt_dyn = sap_info[i_p].mu * sap_info[i_p].mu_factor * t_hat
            dgn_dyt = sap_info[i_p].mu_hat * sap_info[i_p].mu_factor * t_hat
            dgn_dyn = sap_info[i_p].mu_factor

            sap_info[i_p].G[:2, :2] = dgt_dyt
            sap_info[i_p].G[:2, 2] = dgt_dyn
            sap_info[i_p].G[2, :2] = dgn_dyt
            sap_info[i_p].G[2, 2] = dgn_dyn

            sap_info[i_p].G[:, :2] *= 1.0 / sap_info[i_p].Rt
            sap_info[i_p].G[:, 2] *= 1.0 / sap_info[i_p].Rn

        else:  # No contact
            pass

    @ti.func
    def compute_contact_energy(self, sap_info, i_p, vc):
        y = ti.Vector([0.0, 0.0, sap_info[i_p].vn_hat]) - vc
        y[0] /= sap_info[i_p].Rt
        y[1] /= sap_info[i_p].Rt
        y[2] /= sap_info[i_p].Rn
        yr = y[:2].norm(gs.EPS)
        yn = y[2]

        t_hat = y[:2] / yr
        contact_mode = self.compute_contact_mode(sap_info[i_p].mu, sap_info[i_p].mu_hat, yr, yn)
        sap_info[i_p].gamma.fill(0.0)
        if contact_mode == 0:  # Sticking
            sap_info[i_p].gamma = y
        elif contact_mode == 1:  # Sliding
            gn = (yn + sap_info[i_p].mu_hat * yr) * sap_info[i_p].mu_factor
            gt = sap_info[i_p].mu * gn * t_hat
            sap_info[i_p].gamma = ti.Vector([gt[0], gt[1], gn])

        else:  # No contact
            pass

        R_gamma = sap_info[i_p].gamma
        R_gamma[0] *= sap_info[i_p].Rt
        R_gamma[1] *= sap_info[i_p].Rt
        R_gamma[2] *= sap_info[i_p].Rn
        sap_info[i_p].energy = 0.5 * sap_info[i_p].gamma.dot(R_gamma)

    @ti.func
    def compute_contact_mode(self, mu, mu_hat, yr, yn):
        """
        Compute the contact mode based on the friction coefficients and the relative velocities.
        Returns:
            0: Sticking
            1: Sliding
            2: No contact
        """
        result = 2  # No contact
        if yr <= mu * yn:
            result = 0  # Sticking
        elif -mu_hat * yr < yn and yn < yr / mu:
            result = 1  # Sliding
        return result

    @ti.func
    def compute_contact_regularization(self, sap_info, i_p, w_rms, mu, time_step):
        beta = ti.static(1.0)
        beta_factor = ti.static(beta**2 / (4.0 * ti.math.pi**2))
        k = sap_info[i_p].k
        taud = sap_info[i_p].taud
        Rn = max(beta_factor * w_rms, 1.0 / (time_step * k * (time_step + taud)))
        sigma = ti.static(1.0e-3)
        Rt = sigma * w_rms
        vn_hat = -sap_info[i_p].phi0 / (time_step + taud)
        sap_info[i_p].Rn = Rn
        sap_info[i_p].Rt = Rt
        sap_info[i_p].vn_hat = vn_hat
        sap_info[i_p].mu = mu
        sap_info[i_p].mu_hat = sap_info[i_p].mu * Rt / Rn
        sap_info[i_p].mu_factor = 1.0 / (1.0 + sap_info[i_p].mu * sap_info[i_p].mu_hat)

    # ------------------------------------------------------------------------------------
    # -------------------------------------- AABB ----------------------------------------
    # ------------------------------------------------------------------------------------

    @ti.kernel
    def coupute_fem_aabb(self, f: ti.i32):
        fem_solver = self.fem_solver
        aabbs = ti.static(self.fem_aabb.aabbs)
        for i_b, i_e in ti.ndrange(fem_solver._B, fem_solver.n_elements):
            aabbs[i_b, i_e].min.fill(np.inf)
            aabbs[i_b, i_e].max.fill(-np.inf)
            i_v = fem_solver.elements_i[i_e].el2v

            for i in ti.static(range(4)):
                pos_v = fem_solver.elements_v[f, i_v[i], i_b].pos
                aabbs[i_b, i_e].min = ti.min(aabbs[i_b, i_e].min, pos_v)
                aabbs[i_b, i_e].max = ti.max(aabbs[i_b, i_e].max, pos_v)

    # ------------------------------------------------------------------------------------
    # ---------------------------------- FEM vs Floor ------------------------------------
    # ------------------------------------------------------------------------------------

    @ti.kernel
    def fem_floor_detection(self, f: ti.i32):
        fem_solver = self.fem_solver
        pairs = ti.static(self.fem_floor_contact_pairs)
        sap_info = ti.static(pairs.sap_info)
        # Compute contact pairs
        self.n_fem_floor_contact_pairs[None] = 0
        # TODO Check surface element only instead of all elements
        for i_b, i_e in ti.ndrange(fem_solver._B, fem_solver.n_elements):
            intersection_code = ti.int32(0)
            distance = ti.Vector([0.0, 0.0, 0.0, 0.0])
            for i in ti.static(range(4)):
                i_v = fem_solver.elements_i[i_e].el2v[i]
                pos_v = fem_solver.elements_v[f + 1, i_v, i_b].pos
                distance[i] = pos_v.z - fem_solver.floor_height
                if distance[i] > 0:
                    intersection_code |= 1 << i

            # check if the element intersect with the floor
            if intersection_code != 0 and intersection_code != 15:
                pair_idx = ti.atomic_add(self.n_fem_floor_contact_pairs[None], 1)
                if pair_idx < self.max_fem_floor_contact_pairs:
                    pairs[pair_idx].batch_idx = i_b
                    pairs[pair_idx].geom_idx = i_e
                    pairs[pair_idx].intersection_code = intersection_code
                    pairs[pair_idx].distance = distance

        # Compute data for each contact pair
        for i_p in range(self.n_fem_floor_contact_pairs[None]):
            pair = pairs[i_p]
            pairs[i_p].active = 1  # mark the contact pair as active
            i_b = pair.batch_idx
            i_e = pair.geom_idx
            intersection_code = pair.intersection_code
            distance = pair.distance
            intersected_edges = ti.static(self.kMarchingTetsEdgeTable)[intersection_code]
            tet_vertices = ti.Matrix.zero(gs.ti_float, 3, 4)  # 4 vertices
            tet_pressures = ti.Vector.zero(gs.ti_float, 4)  # pressures at the vertices

            for i in ti.static(range(4)):
                i_v = fem_solver.elements_i[i_e].el2v[i]
                tet_vertices[:, i] = fem_solver.elements_v[f + 1, i_v, i_b].pos
                tet_pressures[i] = self.fem_pressure[i_v]

            polygon_vertices = ti.Matrix.zero(gs.ti_float, 3, 4)  # 3 or 4 vertices
            total_area = gs.EPS  # avoid division by zero
            total_area_weighted_centroid = ti.Vector([0.0, 0.0, 0.0])
            for i in range(4):
                if intersected_edges[i] >= 0:
                    edge = ti.static(self.kTetEdges)[intersected_edges[i]]
                    pos_v0 = tet_vertices[:, edge[0]]
                    pos_v1 = tet_vertices[:, edge[1]]
                    d_v0 = distance[edge[0]]
                    d_v1 = distance[edge[1]]
                    t = d_v0 / (d_v0 - d_v1)
                    polygon_vertices[:, i] = pos_v0 + t * (pos_v1 - pos_v0)

                    # Compute tirangle area and centroid
                    if i >= 2:
                        e1 = polygon_vertices[:, i - 1] - polygon_vertices[:, 0]
                        e2 = polygon_vertices[:, i] - polygon_vertices[:, 0]
                        area = 0.5 * e1.cross(e2).norm()
                        total_area += area
                        total_area_weighted_centroid += (
                            area * (polygon_vertices[:, 0] + polygon_vertices[:, i - 1] + polygon_vertices[:, i]) / 3.0
                        )

            centroid = total_area_weighted_centroid / total_area

            # Compute barycentric coordinates
            barycentric = tet_barycentric(centroid, tet_vertices)
            pressure = (
                barycentric[0] * tet_pressures[0]
                + barycentric[1] * tet_pressures[1]
                + barycentric[2] * tet_pressures[2]
                + barycentric[3] * tet_pressures[3]
            )
            pairs[i_p].barycentric = barycentric

            C = ti.static(1.0e8)
            deformable_g = C
            rigid_g = self.fem_pressure_gradient[i_b, i_e].z
            # TODO A better way to handle corner cases where pressure and pressure gradient are ill defined
            if total_area < gs.EPS or rigid_g < gs.EPS:
                pairs[i_p].active = 0
                continue
            g = 1.0 / (1.0 / deformable_g + 1.0 / rigid_g)  # harmonic average
            rigid_k = total_area * g
            rigid_phi0 = -pressure / g
            rigid_fn0 = total_area * pressure
            # TODO custom dissipation
            sap_info[i_p].k = rigid_k  # contact stiffness
            sap_info[i_p].phi0 = rigid_phi0
            sap_info[i_p].fn0 = rigid_fn0
            sap_info[i_p].taud = 0.1  # Drake uses 100ms as default

    @ti.kernel
    def compute_fem_floor_regularization(self):
        pairs = ti.static(self.fem_floor_contact_pairs)
        sap_info = ti.static(pairs.sap_info)
        time_step = self.sim._substep_dt
        dt2_inv = 1.0 / (time_step**2)
        fem_solver = self.fem_solver

        for i_p in range(self.n_fem_floor_contact_pairs[None]):
            if pairs[i_p].active == 0:
                continue
            i_b = pairs[i_p].batch_idx
            i_e = pairs[i_p].geom_idx
            W = ti.Matrix.zero(gs.ti_float, 3, 3)
            # W = sum (JA^-1J^T)
            # With floor, J is Identity times the barycentric coordinates
            for i in ti.static(range(4)):
                i_v = fem_solver.elements_i[i_e].el2v[i]
                W += pairs[i_p].barycentric[i] ** 2 * fem_solver.pcg_state_v[i_b, i_v].prec
            w_rms = W.norm() / 3.0 * dt2_inv
            self.compute_contact_regularization(sap_info, i_p, w_rms, fem_solver.elements_i[i_e].friction_mu, time_step)

    @ti.kernel
    def compute_fem_floor_gradient_hessian_diag(self):
        pairs = ti.static(self.fem_floor_contact_pairs)
        sap_info = ti.static(pairs.sap_info)
        fem_solver = self.fem_solver
        for i_p in range(self.n_fem_floor_contact_pairs[None]):
            if pairs[i_p].active == 0:
                continue
            i_b = pairs[i_p].batch_idx
            i_e = pairs[i_p].geom_idx
            vc = ti.Vector([0.0, 0.0, 0.0])
            # With floor, the contact frame is the same as the world frame
            for i in ti.static(range(4)):
                i_v = fem_solver.elements_i[i_e].el2v[i]
                vc += pairs[i_p].barycentric[i] * self.v[i_b, i_v]
            self.compute_contact_gamma_G(sap_info, i_p, vc)
            for i in ti.static(range(4)):
                i_v = fem_solver.elements_i[i_e].el2v[i]
                self.gradient[i_b, i_v] -= pairs[i_p].barycentric[i] * sap_info[i_p].gamma
                self.pcg_state_v[i_b, i_v].diag3x3 += pairs[i_p].barycentric[i] ** 2 * sap_info[i_p].G

    @ti.kernel
    def compute_fem_floor_energy(self, energy: ti.template()):
        pairs = ti.static(self.fem_floor_contact_pairs)
        sap_info = ti.static(pairs.sap_info)
        fem_solver = self.fem_solver

        for i_p in range(self.n_fem_floor_contact_pairs[None]):
            if pairs[i_p].active == 0:
                continue
            i_b = pairs[i_p].batch_idx
            if not self.batch_linesearch_active[i_b]:
                continue
            i_e = pairs[i_p].geom_idx
            vc = ti.Vector([0.0, 0.0, 0.0])
            # With floor, the contact frame is the same as the world frame
            for i in ti.static(range(4)):
                i_v = fem_solver.elements_i[i_e].el2v[i]
                vc += pairs[i_p].barycentric[i] * self.v[i_b, i_v]
            self.compute_contact_energy(sap_info, i_p, vc)
            energy[i_b] += sap_info[i_p].energy

    @ti.kernel
    def compute_fem_floor_Ap(self):
        for i_p in range(self.n_fem_floor_contact_pairs[None]):
            fem_solver = self.fem_solver
            pairs = ti.static(self.fem_floor_contact_pairs)
            sap_info = ti.static(pairs.sap_info)
            if pairs[i_p].active == 0:
                continue
            i_b = pairs[i_p].batch_idx
            i_e = pairs[i_p].geom_idx

            x = ti.Vector.zero(gs.ti_float, 3)
            for i in ti.static(range(4)):
                i_v = fem_solver.elements_i[i_e].el2v[i]
                x += pairs[i_p].barycentric[i] * self.pcg_state_v[i_b, i_v].p

            x = sap_info[i_p].G @ x

            for i in ti.static(range(4)):
                i_v = fem_solver.elements_i[i_e].el2v[i]
                self.pcg_state_v[i_b, i_v].Ap += pairs[i_p].barycentric[i] * x

    # ------------------------------------------------------------------------------------
    # ----------------------------------- FEM vs FEM -------------------------------------
    # ------------------------------------------------------------------------------------

    def fem_self_detection(self, f: ti.i32):
        self.coupute_fem_aabb(f + 1)
        self.fem_bvh.build()
        self.fem_bvh.query(self.fem_aabb.aabbs)
        if self.fem_bvh.query_result_count[None] > self.fem_bvh.max_n_query_results:
            raise ValueError(
                f"Query result count {self.fem_bvh.query_result_count[None]} "
                f"exceeds max_n_query_results {self.fem_bvh.max_n_query_results}"
            )
        self.compute_fem_self_pair_candidates(f + 1)
        if self.n_fem_self_contact_pairs[None] > self.max_fem_self_contact_pair_candidates:
            raise ValueError(
                f"Number of self contact pair candidates {self.n_fem_self_contact_pair_candidates[None]} "
                f"exceeds max_fem_self_contact_pair_candidates {self.max_fem_self_contact_pair_candidates}"
            )
        self.compute_fem_self_pairs(f + 1)
        if self.n_fem_self_contact_pairs[None] > self.max_fem_self_contact_pairs:
            raise ValueError(
                f"Number of self contact pairs {self.n_fem_self_contact_pairs[None]} "
                f"exceeds max_fem_self_contact_pairs {self.max_fem_self_contact_pairs}"
            )

    @ti.kernel
    def compute_fem_self_pair_candidates(self, f: ti.i32):
        fem_solver = self.fem_solver
        candidates = ti.static(self.fem_self_contact_pair_candidates)
        self.n_fem_self_contact_pair_candidates[None] = 0
        for i_r in ti.ndrange(self.fem_bvh.query_result_count[None]):
            i_b, i_a, i_q = self.fem_bvh.query_result[i_r]
            i_v0 = fem_solver.elements_i[i_a].el2v[0]
            i_v1 = fem_solver.elements_i[i_q].el2v[0]
            x0 = fem_solver.elements_v[f, i_v0, i_b].pos
            x1 = fem_solver.elements_v[f, i_v1, i_b].pos
            p0 = self.fem_pressure[i_v0]
            p1 = self.fem_pressure[i_v1]
            g0 = self.fem_pressure_gradient[i_b, i_a]
            g1 = self.fem_pressure_gradient[i_b, i_q]
            g0_norm = g0.norm()
            g1_norm = g1.norm()
            if g0_norm < gs.EPS or g1_norm < gs.EPS:
                continue
            # Calculate the isosurface, i.e. equal pressure plane defined by x and normal
            # Solve for p0 + g0.dot(x - x0) = p1 + g1.dot(x - x1)
            normal = g0 - g1
            magnitude = normal.norm()
            if magnitude < gs.EPS:
                continue
            normal /= magnitude
            b = p1 - p0 - g1.dot(x1) + g0.dot(x0)
            x = b / magnitude * normal
            # Check that the normal is pointing along g0 and against g1, some allowance as used in Drake
            threshold = ti.static(np.cos(np.pi * 5.0 / 8.0))
            if normal.dot(g0) < threshold * g0_norm or normal.dot(g1) > -threshold * g1_norm:
                continue

            intersection_code0 = ti.int32(0)
            distance0 = ti.Vector([0.0, 0.0, 0.0, 0.0])
            intersection_code1 = ti.int32(0)
            distance1 = ti.Vector([0.0, 0.0, 0.0, 0.0])
            for i in ti.static(range(4)):
                i_v = fem_solver.elements_i[i_a].el2v[i]
                pos_v = fem_solver.elements_v[f, i_v, i_b].pos
                distance0[i] = (pos_v - x).dot(normal)  # signed distance
                if distance0[i] > 0:
                    intersection_code0 |= 1 << i
            for i in ti.static(range(4)):
                i_v = fem_solver.elements_i[i_q].el2v[i]
                pos_v = fem_solver.elements_v[f, i_v, i_b].pos
                distance1[i] = (pos_v - x).dot(normal)
                if distance1[i] > 0:
                    intersection_code1 |= 1 << i
            # Fast check for whether both tets intersect with the plane
            if (
                intersection_code0 == 0
                or intersection_code1 == 0
                or intersection_code0 == 15
                or intersection_code1 == 15
            ):
                continue
            candidate_idx = ti.atomic_add(self.n_fem_self_contact_pair_candidates[None], 1)
            if candidate_idx < self.max_fem_self_contact_pair_candidates:
                candidates[candidate_idx].batch_idx = i_b
                candidates[candidate_idx].normal = normal
                candidates[candidate_idx].x = x
                candidates[candidate_idx].geom_idx0 = i_a
                candidates[candidate_idx].intersection_code0 = intersection_code0
                candidates[candidate_idx].distance0 = distance0
                candidates[candidate_idx].geom_idx1 = i_q

    @ti.kernel
    def compute_fem_self_pairs(self, f: ti.i32):
        """
        Computes the FEM self contact pairs and their properties.
        Intersection code reference:
        https://github.com/RobotLocomotion/drake/blob/8c3a249184ed09f0faab3c678536d66d732809ce/geometry/proximity/field_intersection.cc#L87
        """
        fem_solver = self.fem_solver
        candidates = ti.static(self.fem_self_contact_pair_candidates)
        pairs = ti.static(self.fem_self_contact_pairs)
        sap_info = ti.static(pairs.sap_info)
        normal_signs = ti.Vector([1.0, -1.0, 1.0, -1.0])  # make normal point outward
        self.n_fem_self_contact_pairs[None] = 0
        for i_c in range(self.n_fem_self_contact_pair_candidates[None]):
            i_b = candidates[i_c].batch_idx
            i_e0 = candidates[i_c].geom_idx0
            i_e1 = candidates[i_c].geom_idx1
            intersection_code0 = candidates[i_c].intersection_code0
            distance0 = candidates[i_c].distance0
            intersected_edges0 = ti.static(self.kMarchingTetsEdgeTable)[intersection_code0]
            tet_vertices0 = ti.Matrix.zero(gs.ti_float, 3, 4)  # 4 vertices of tet 0
            tet_pressures0 = ti.Vector.zero(gs.ti_float, 4)  # pressures at the vertices of tet 0
            tet_vertices1 = ti.Matrix.zero(gs.ti_float, 3, 4)  # 4 vertices of tet 1

            for i in ti.static(range(4)):
                i_v = fem_solver.elements_i[i_e0].el2v[i]
                tet_vertices0[:, i] = fem_solver.elements_v[f, i_v, i_b].pos
                tet_pressures0[i] = self.fem_pressure[i_v]

            for i in ti.static(range(4)):
                i_v = fem_solver.elements_i[i_e1].el2v[i]
                tet_vertices1[:, i] = fem_solver.elements_v[f, i_v, i_b].pos

            polygon_vertices = ti.Matrix.zero(gs.ti_float, 3, 8)  # maximum 8 vertices
            polygon_n_vertices = 0
            clipped_vertices = ti.Matrix.zero(gs.ti_float, 3, 8)  # maximum 8 vertices
            clipped_n_vertices = 0
            for i in range(4):
                if intersected_edges0[i] >= 0:
                    edge = ti.static(self.kTetEdges)[intersected_edges0[i]]
                    pos_v0 = tet_vertices0[:, edge[0]]
                    pos_v1 = tet_vertices0[:, edge[1]]
                    d_v0 = distance0[edge[0]]
                    d_v1 = distance0[edge[1]]
                    t = d_v0 / (d_v0 - d_v1)
                    polygon_vertices[:, polygon_n_vertices] = pos_v0 + t * (pos_v1 - pos_v0)
                    polygon_n_vertices += 1
            # Intersects the polygon with the four halfspaces of the four triangles
            # of the tetrahedral element1.
            for face in range(4):
                clipped_n_vertices = 0
                x = tet_vertices1[:, (face + 1) % 4]
                normal = (tet_vertices1[:, (face + 2) % 4] - x).cross(
                    tet_vertices1[:, (face + 3) % 4] - x
                ) * normal_signs[face]
                normal /= normal.norm()

                distances = ti.Vector.zero(gs.ti_float, 8)
                for i in range(polygon_n_vertices):
                    distances[i] = (polygon_vertices[:, i] - x).dot(normal)

                for i in range(polygon_n_vertices):
                    j = (i + 1) % polygon_n_vertices
                    if distances[i] <= 0.0:
                        clipped_vertices[:, clipped_n_vertices] = polygon_vertices[:, i]
                        clipped_n_vertices += 1
                        if distances[j] > 0.0:
                            wa = distances[j] / (distances[j] - distances[i])
                            wb = 1.0 - wa
                            clipped_vertices[:, clipped_n_vertices] = (
                                wa * polygon_vertices[:, i] + wb * polygon_vertices[:, j]
                            )
                            clipped_n_vertices += 1
                    elif distances[j] <= 0.0:
                        wa = distances[j] / (distances[j] - distances[i])
                        wb = 1.0 - wa
                        clipped_vertices[:, clipped_n_vertices] = (
                            wa * polygon_vertices[:, i] + wb * polygon_vertices[:, j]
                        )
                        clipped_n_vertices += 1
                polygon_n_vertices = clipped_n_vertices
                polygon_vertices = clipped_vertices

                if polygon_n_vertices < 3:
                    # If the polygon has less than 3 vertices, it is not a valid contact
                    break

            if polygon_n_vertices < 3:
                continue

            # compute centroid and area of the polygon
            total_area = gs.EPS  # avoid division by zero
            total_area_weighted_centroid = ti.Vector([0.0, 0.0, 0.0])
            for i in range(2, polygon_n_vertices):
                e1 = polygon_vertices[:, i - 1] - polygon_vertices[:, 0]
                e2 = polygon_vertices[:, i] - polygon_vertices[:, 0]
                area = 0.5 * e1.cross(e2).norm()
                total_area += area
                total_area_weighted_centroid += (
                    area * (polygon_vertices[:, 0] + polygon_vertices[:, i - 1] + polygon_vertices[:, i]) / 3.0
                )

            if total_area < gs.EPS:
                continue
            centroid = total_area_weighted_centroid / total_area
            barycentric0 = tet_barycentric(centroid, tet_vertices0)
            barycentric1 = tet_barycentric(centroid, tet_vertices1)
            tangent0 = polygon_vertices[:, 0] - centroid
            tangent0 /= tangent0.norm()
            tangent1 = candidates[i_c].normal.cross(tangent0)
            i_p = ti.atomic_add(self.n_fem_self_contact_pairs[None], 1)
            if i_p < self.max_fem_self_contact_pairs:
                pairs[i_p].batch_idx = i_b
                pairs[i_p].normal = candidates[i_c].normal
                pairs[i_p].tangent0 = tangent0
                pairs[i_p].tangent1 = tangent1
                pairs[i_p].geom_idx0 = i_e0
                pairs[i_p].geom_idx1 = i_e1
                pairs[i_p].barycentric0 = barycentric0
                pairs[i_p].barycentric1 = barycentric1
                pressure = (
                    barycentric0[0] * tet_pressures0[0]
                    + barycentric0[1] * tet_pressures0[1]
                    + barycentric0[2] * tet_pressures0[2]
                    + barycentric0[3] * tet_pressures0[3]
                )

                deformable_g = ti.static(1.0e8)
                deformable_k = total_area * deformable_g
                deformable_phi0 = (
                    -pressure / deformable_g * 2
                )  # This is a very approximated value, different from Drake
                deformable_fn0 = -deformable_k * deformable_phi0
                # TODO custom dissipation
                sap_info[i_p].k = deformable_k  # contact stiffness
                sap_info[i_p].phi0 = deformable_phi0
                sap_info[i_p].fn0 = deformable_fn0
                sap_info[i_p].taud = 0.1  # Drake uses 100ms as default

    @ti.kernel
    def compute_fem_self_regularization(self):
        pairs = ti.static(self.fem_self_contact_pairs)
        sap_info = ti.static(pairs.sap_info)
        time_step = self.sim._substep_dt
        dt2_inv = 1.0 / (time_step**2)
        fem_solver = self.fem_solver

        for i_p in range(self.n_fem_self_contact_pairs[None]):
            i_b = pairs[i_p].batch_idx
            i_e0 = pairs[i_p].geom_idx0
            i_e1 = pairs[i_p].geom_idx1
            W = ti.Matrix.zero(gs.ti_float, 3, 3)
            world = ti.Matrix.cols([pairs[i_p].tangent0, pairs[i_p].tangent1, pairs[i_p].normal])
            # W = sum (JA^-1J^T)
            for i in ti.static(range(4)):
                i_v = fem_solver.elements_i[i_e0].el2v[i]
                W += pairs[i_p].barycentric0[i] ** 2 * fem_solver.pcg_state_v[i_b, i_v].prec
            for i in ti.static(range(4)):
                i_v = fem_solver.elements_i[i_e1].el2v[i]
                W += pairs[i_p].barycentric1[i] ** 2 * fem_solver.pcg_state_v[i_b, i_v].prec
            W = world.transpose() @ W @ world
            w_rms = W.norm() / 3.0 * dt2_inv
            mu = ti.sqrt(fem_solver.elements_i[i_e0].friction_mu * fem_solver.elements_i[i_e1].friction_mu)
            self.compute_contact_regularization(sap_info, i_p, w_rms, mu, time_step)

    @ti.kernel
    def compute_fem_self_gradient_hessian_diag(self):
        pairs = ti.static(self.fem_self_contact_pairs)
        sap_info = ti.static(pairs.sap_info)
        fem_solver = self.fem_solver
        for i_p in range(self.n_fem_self_contact_pairs[None]):
            i_b = pairs[i_p].batch_idx
            i_e0 = pairs[i_p].geom_idx0
            i_e1 = pairs[i_p].geom_idx1
            # contact velocity
            vc = ti.Vector([0.0, 0.0, 0.0])
            for i in ti.static(range(4)):
                i_v = fem_solver.elements_i[i_e0].el2v[i]
                vc += pairs[i_p].barycentric0[i] * self.v[i_b, i_v]
            for i in ti.static(range(4)):
                i_v = fem_solver.elements_i[i_e1].el2v[i]
                vc -= pairs[i_p].barycentric1[i] * self.v[i_b, i_v]
            # project to contact frame
            vc = ti.Vector([vc.dot(pairs[i_p].tangent0), vc.dot(pairs[i_p].tangent1), vc.dot(pairs[i_p].normal)])

            self.compute_contact_gamma_G(sap_info, i_p, vc)

            # project back to world frame
            world = ti.Matrix.cols([pairs[i_p].tangent0, pairs[i_p].tangent1, pairs[i_p].normal])
            sap_info[i_p].gamma = world @ sap_info[i_p].gamma
            sap_info[i_p].G = world @ sap_info[i_p].G @ world.transpose()

            # apply to vertices
            for i in ti.static(range(4)):
                i_v = fem_solver.elements_i[i_e0].el2v[i]
                self.gradient[i_b, i_v] -= pairs[i_p].barycentric0[i] * sap_info[i_p].gamma
                self.pcg_state_v[i_b, i_v].diag3x3 += pairs[i_p].barycentric0[i] ** 2 * sap_info[i_p].G
            for i in ti.static(range(4)):
                i_v = fem_solver.elements_i[i_e1].el2v[i]
                self.gradient[i_b, i_v] += pairs[i_p].barycentric1[i] * sap_info[i_p].gamma
                self.pcg_state_v[i_b, i_v].diag3x3 += pairs[i_p].barycentric1[i] ** 2 * sap_info[i_p].G

    @ti.kernel
    def compute_fem_self_energy(self, energy: ti.template()):
        pairs = ti.static(self.fem_self_contact_pairs)
        sap_info = ti.static(pairs.sap_info)
        fem_solver = self.fem_solver

        for i_p in range(self.n_fem_self_contact_pairs[None]):
            i_b = pairs[i_p].batch_idx
            if not self.batch_linesearch_active[i_b]:
                continue
            i_e0 = pairs[i_p].geom_idx0
            i_e1 = pairs[i_p].geom_idx1
            # contact velocity
            vc = ti.Vector([0.0, 0.0, 0.0])
            for i in ti.static(range(4)):
                i_v = fem_solver.elements_i[i_e0].el2v[i]
                vc += pairs[i_p].barycentric0[i] * self.v[i_b, i_v]
            for i in ti.static(range(4)):
                i_v = fem_solver.elements_i[i_e1].el2v[i]
                vc -= pairs[i_p].barycentric1[i] * self.v[i_b, i_v]
            # project to contact frame
            vc = ti.Vector([vc.dot(pairs[i_p].tangent0), vc.dot(pairs[i_p].tangent1), vc.dot(pairs[i_p].normal)])
            self.compute_contact_energy(sap_info, i_p, vc)
            energy[i_b] += sap_info[i_p].energy

    @ti.kernel
    def compute_fem_self_Ap(self):
        fem_solver = self.fem_solver
        pairs = ti.static(self.fem_self_contact_pairs)
        sap_info = ti.static(pairs.sap_info)
        for i_p in range(self.n_fem_self_contact_pairs[None]):
            i_b = pairs[i_p].batch_idx
            i_e0 = pairs[i_p].geom_idx0
            i_e1 = pairs[i_p].geom_idx1

            x = ti.Vector.zero(gs.ti_float, 3)
            for i in ti.static(range(4)):
                i_v = fem_solver.elements_i[i_e0].el2v[i]
                x += pairs[i_p].barycentric0[i] * self.pcg_state_v[i_b, i_v].p
            for i in ti.static(range(4)):
                i_v = fem_solver.elements_i[i_e1].el2v[i]
                x -= pairs[i_p].barycentric1[i] * self.pcg_state_v[i_b, i_v].p

            x = sap_info[i_p].G @ x

            for i in ti.static(range(4)):
                i_v = fem_solver.elements_i[i_e0].el2v[i]
                self.pcg_state_v[i_b, i_v].Ap += pairs[i_p].barycentric0[i] * x
            for i in ti.static(range(4)):
                i_v = fem_solver.elements_i[i_e1].el2v[i]
                self.pcg_state_v[i_b, i_v].Ap -= pairs[i_p].barycentric1[i] * x

    # ------------------------------------------------------------------------------------
    # ----------------------------------- Properties -------------------------------------
    # ------------------------------------------------------------------------------------
    @property
    def active_solvers(self):
        """All the active solvers managed by the scene's simulator."""
        return self.sim.active_solvers
