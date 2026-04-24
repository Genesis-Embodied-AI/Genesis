import dataclasses
import math
from enum import IntEnum

import quadrants as qd
import numpy as np
import torch

import genesis as gs

if not gs._initialized:
    gs.raise_exception("Genesis hasn't been initialized. Did you call `gs.init()`?")


_TENSOR_BACKEND = qd.Backend.NDARRAY if gs.use_ndarray else qd.Backend.FIELD


class _AutoInitMeta(type):
    """Metaclass that generates __init__ from annotations, like a mutable dataclass."""

    def __new__(cls, name, bases, namespace):
        names = tuple(namespace.get("__annotations__", {}).keys())
        defaults = {k: namespace[k] for k in names if k in namespace}

        def __init__(self, *args, **kwargs):
            assigned = defaults.copy()
            if len(args) > len(names):
                raise TypeError(f"{name}() takes {len(names)} positional arguments but {len(args)} were given")
            for key, value in zip(names, args):
                assigned[key] = value
            for key, value in kwargs.items():
                if key not in names:
                    raise TypeError(f"{name}() got unexpected keyword argument '{key}'")
                if key in names[: len(args)]:
                    raise TypeError(f"{name}() got multiple values for argument '{key}'")
                assigned[key] = value
            for key in names:
                if key not in assigned:
                    raise TypeError(f"{name}() missing required argument: '{key}'")
            for key, value in assigned.items():
                setattr(self, key, value)

        namespace["__init__"] = __init__
        return super().__new__(cls, name, bases, namespace)


PLACEHOLDER = qd.tensor(gs.qd_float, (), backend=_TENSOR_BACKEND)


def maybe_shape(shape, is_on):
    return shape if is_on else ()


def V_SCALAR_FROM(dtype, value):
    data = qd.tensor(dtype, (), backend=_TENSOR_BACKEND)
    data.fill(value)
    return data


# =========================================== ErrorCode ===========================================


class ErrorCode(IntEnum):
    SUCCESS = 0b000000000000000000000000000000000
    OVERFLOW_CANDIDATE_CONTACTS = 0b00000000000000000000000000000001
    OVERFLOW_COLLISION_PAIRS = 0b00000000000000000000000000000010
    OVERFLOW_HIBERNATION_ISLANDS = 0b00000000000000000000000000000100
    INVALID_FORCE_NAN = 0b00000000000000000000000000001000
    INVALID_ACC_NAN = 0b00000000000000000000000000010000


# =========================================== RigidGlobalInfo ===========================================


@dataclasses.dataclass(frozen=True)
class StructRigidGlobalInfo:
    # *_bw: Cache for backward pass
    n_awake_dofs: qd.Tensor
    awake_dofs: qd.Tensor
    n_awake_entities: qd.Tensor
    awake_entities: qd.Tensor
    n_awake_links: qd.Tensor
    awake_links: qd.Tensor
    qpos0: qd.Tensor
    qpos: qd.Tensor
    qpos_next: qd.Tensor
    links_T: qd.Tensor
    envs_offset: qd.Tensor
    geoms_init_AABB: qd.Tensor
    mass_mat: qd.Tensor
    mass_mat_L: qd.Tensor
    mass_mat_L_bw: qd.Tensor
    mass_mat_D_inv: qd.Tensor
    mass_mat_mask: qd.Tensor
    meaninertia: qd.Tensor
    mass_parent_mask: qd.Tensor
    gravity: qd.Tensor
    # Runtime constants
    substep_dt: qd.Tensor
    iterations: qd.Tensor
    tolerance: qd.Tensor
    ls_iterations: qd.Tensor
    ls_tolerance: qd.Tensor
    noslip_iterations: qd.Tensor
    noslip_tolerance: qd.Tensor
    n_equalities: qd.Tensor
    n_candidate_equalities: qd.Tensor
    hibernation_thresh_acc: qd.Tensor
    hibernation_thresh_vel: qd.Tensor
    EPS: qd.Tensor


def get_rigid_global_info(solver, kinematic_only):
    _B = solver._B

    mass_mat_shape = (solver.n_dofs_, solver.n_dofs_, _B)
    if math.prod(mass_mat_shape) > np.iinfo(np.int32).max:
        gs.raise_exception(
            f"Mass matrix shape (n_dofs={solver.n_dofs_}, n_dofs={solver.n_dofs_}, n_envs={_B}) is too large."
        )
    requires_grad = solver._requires_grad
    mass_mat_shape_bw = maybe_shape((2, *mass_mat_shape), requires_grad)
    if math.prod(mass_mat_shape_bw) > np.iinfo(np.int32).max:
        gs.raise_exception(
            f"Mass matrix buffer shape (2, n_dofs={solver.n_dofs_}, n_dofs={solver.n_dofs_}, n_envs={_B}) is too large."
        )

    # FIXME: Add a better split between kinematic and Genesis
    if kinematic_only:
        return StructRigidGlobalInfo(
            envs_offset=qd.Vector.tensor(3, gs.qd_float, (_B,), backend=_TENSOR_BACKEND),
            gravity=qd.Vector.tensor(3, gs.qd_float, (), backend=_TENSOR_BACKEND),
            meaninertia=qd.tensor(gs.qd_float, (), backend=_TENSOR_BACKEND),
            n_awake_dofs=qd.tensor(gs.qd_int, (_B,), backend=_TENSOR_BACKEND),
            n_awake_entities=qd.tensor(gs.qd_int, (_B,), backend=_TENSOR_BACKEND),
            n_awake_links=qd.tensor(gs.qd_int, (_B,), backend=_TENSOR_BACKEND),
            awake_dofs=qd.tensor(gs.qd_int, (solver.n_dofs_, _B), backend=_TENSOR_BACKEND),
            awake_entities=qd.tensor(gs.qd_int, (solver.n_entities_, _B), backend=_TENSOR_BACKEND),
            awake_links=qd.tensor(gs.qd_int, (solver.n_links_, _B), backend=_TENSOR_BACKEND),
            qpos0=qd.tensor(gs.qd_float, (solver.n_qs_, _B), backend=_TENSOR_BACKEND),
            qpos=qd.tensor(gs.qd_float, (solver.n_qs_, _B), backend=_TENSOR_BACKEND),
            qpos_next=qd.tensor(gs.qd_float, (solver.n_qs_, _B), backend=_TENSOR_BACKEND),
            links_T=qd.Matrix.tensor(4, 4, gs.qd_float, (solver.n_links_,), backend=_TENSOR_BACKEND),
            geoms_init_AABB=qd.Vector.tensor(3, gs.qd_float, (), backend=_TENSOR_BACKEND),
            mass_mat=qd.tensor(gs.qd_float, (), backend=_TENSOR_BACKEND),
            mass_mat_L=qd.tensor(gs.qd_float, (), backend=_TENSOR_BACKEND),
            mass_mat_L_bw=qd.tensor(gs.qd_float, (), backend=_TENSOR_BACKEND),
            mass_mat_D_inv=qd.tensor(gs.qd_float, (), backend=_TENSOR_BACKEND),
            mass_mat_mask=qd.tensor(gs.qd_bool, (), backend=_TENSOR_BACKEND),
            mass_parent_mask=qd.tensor(gs.qd_float, (), backend=_TENSOR_BACKEND),
            substep_dt=V_SCALAR_FROM(dtype=gs.qd_float, value=0.0),
            iterations=V_SCALAR_FROM(dtype=gs.qd_int, value=0),
            tolerance=V_SCALAR_FROM(dtype=gs.qd_float, value=0.0),
            ls_iterations=V_SCALAR_FROM(dtype=gs.qd_int, value=0),
            ls_tolerance=V_SCALAR_FROM(dtype=gs.qd_float, value=0.0),
            noslip_iterations=V_SCALAR_FROM(dtype=gs.qd_int, value=0),
            noslip_tolerance=V_SCALAR_FROM(dtype=gs.qd_float, value=0.0),
            n_equalities=V_SCALAR_FROM(dtype=gs.qd_int, value=0),
            n_candidate_equalities=V_SCALAR_FROM(dtype=gs.qd_int, value=0),
            hibernation_thresh_acc=V_SCALAR_FROM(dtype=gs.qd_float, value=0.0),
            hibernation_thresh_vel=V_SCALAR_FROM(dtype=gs.qd_float, value=0.0),
            EPS=V_SCALAR_FROM(dtype=gs.qd_float, value=gs.EPS),
        )

    return StructRigidGlobalInfo(
        envs_offset=qd.Vector.tensor(3, gs.qd_float, (_B,), backend=_TENSOR_BACKEND),
        gravity=qd.Vector.tensor(3, gs.qd_float, (_B,), backend=_TENSOR_BACKEND),
        meaninertia=qd.tensor(gs.qd_float, (_B,), backend=_TENSOR_BACKEND),
        n_awake_dofs=qd.tensor(gs.qd_int, (_B,), backend=_TENSOR_BACKEND),
        n_awake_entities=qd.tensor(gs.qd_int, (_B,), backend=_TENSOR_BACKEND),
        n_awake_links=qd.tensor(gs.qd_int, (_B,), backend=_TENSOR_BACKEND),
        awake_dofs=qd.tensor(gs.qd_int, (solver.n_dofs_, _B), backend=_TENSOR_BACKEND),
        awake_entities=qd.tensor(gs.qd_int, (solver.n_entities_, _B), backend=_TENSOR_BACKEND),
        awake_links=qd.tensor(gs.qd_int, (solver.n_links_, _B), backend=_TENSOR_BACKEND),
        qpos0=qd.tensor(gs.qd_float, (solver.n_qs_, _B), backend=_TENSOR_BACKEND),
        qpos=qd.tensor(gs.qd_float, (solver.n_qs_, _B), backend=_TENSOR_BACKEND, needs_grad=requires_grad),
        qpos_next=qd.tensor(gs.qd_float, (solver.n_qs_, _B), backend=_TENSOR_BACKEND, needs_grad=requires_grad),
        links_T=qd.Matrix.tensor(4, 4, gs.qd_float, (solver.n_links_,), backend=_TENSOR_BACKEND),
        geoms_init_AABB=qd.Vector.tensor(3, gs.qd_float, (solver.n_geoms_, 8), backend=_TENSOR_BACKEND),
        mass_mat=qd.tensor(gs.qd_float, mass_mat_shape, backend=_TENSOR_BACKEND, needs_grad=requires_grad),
        mass_mat_L=qd.tensor(gs.qd_float, mass_mat_shape, backend=_TENSOR_BACKEND, needs_grad=requires_grad),
        mass_mat_L_bw=qd.tensor(gs.qd_float, mass_mat_shape_bw, backend=_TENSOR_BACKEND, needs_grad=requires_grad),
        mass_mat_D_inv=qd.tensor(gs.qd_float, (solver.n_dofs_, _B), backend=_TENSOR_BACKEND, needs_grad=requires_grad),
        mass_mat_mask=qd.tensor(gs.qd_bool, (solver.n_entities_, _B), backend=_TENSOR_BACKEND),
        mass_parent_mask=qd.tensor(gs.qd_float, (solver.n_dofs_, solver.n_dofs_), backend=_TENSOR_BACKEND),
        substep_dt=V_SCALAR_FROM(dtype=gs.qd_float, value=solver._substep_dt),
        iterations=V_SCALAR_FROM(dtype=gs.qd_int, value=solver._options.iterations),
        tolerance=V_SCALAR_FROM(dtype=gs.qd_float, value=solver._options.tolerance),
        ls_iterations=V_SCALAR_FROM(dtype=gs.qd_int, value=solver._options.ls_iterations),
        ls_tolerance=V_SCALAR_FROM(dtype=gs.qd_float, value=solver._options.ls_tolerance),
        noslip_iterations=V_SCALAR_FROM(dtype=gs.qd_int, value=solver._options.noslip_iterations),
        noslip_tolerance=V_SCALAR_FROM(dtype=gs.qd_float, value=solver._options.noslip_tolerance),
        n_equalities=V_SCALAR_FROM(dtype=gs.qd_int, value=solver._n_equalities),
        n_candidate_equalities=V_SCALAR_FROM(dtype=gs.qd_int, value=solver.n_candidate_equalities_),
        hibernation_thresh_acc=V_SCALAR_FROM(dtype=gs.qd_float, value=solver._hibernation_thresh_acc),
        hibernation_thresh_vel=V_SCALAR_FROM(dtype=gs.qd_float, value=solver._hibernation_thresh_vel),
        EPS=V_SCALAR_FROM(dtype=gs.qd_float, value=gs.EPS),
    )


# =========================================== Constraint ===========================================


@dataclasses.dataclass(frozen=True)
class StructConstraintState:
    is_warmstart: qd.Tensor
    n_constraints: qd.Tensor
    qd_n_equalities: qd.Tensor
    jac: qd.Tensor
    diag: qd.Tensor
    aref: qd.Tensor
    jac_relevant_dofs: qd.Tensor
    jac_n_relevant_dofs: qd.Tensor
    n_constraints_equality: qd.Tensor
    n_constraints_frictionloss: qd.Tensor
    improved: qd.Tensor
    Jaref: qd.Tensor
    Ma: qd.Tensor
    Ma_ws: qd.Tensor
    grad: qd.Tensor
    Mgrad: qd.Tensor
    MinvJT: qd.Tensor
    search: qd.Tensor
    efc_D: qd.Tensor
    efc_frictionloss: qd.Tensor
    efc_force: qd.Tensor
    efc_b: qd.Tensor
    efc_AR: qd.Tensor
    active: qd.Tensor
    prev_active: qd.Tensor
    qfrc_constraint: qd.Tensor
    qacc: qd.Tensor
    qacc_ws: qd.Tensor
    qacc_prev: qd.Tensor
    cost_ws: qd.Tensor
    gauss: qd.Tensor
    cost: qd.Tensor
    prev_cost: qd.Tensor
    gtol: qd.Tensor
    mv: qd.Tensor
    jv: qd.Tensor
    quad_gauss: qd.Tensor
    ls_alpha: qd.Tensor
    ls_p0_cost: qd.Tensor
    ls_alpha_newton: qd.Tensor
    ls_gtol: qd.Tensor
    eq_sum: qd.Tensor
    ls_it: qd.Tensor
    ls_result: qd.Tensor
    # Optional CG fields
    cg_prev_grad: qd.Tensor
    cg_prev_Mgrad: qd.Tensor
    cg_beta: qd.Tensor
    cg_pg_dot_pMg: qd.Tensor
    # Optional Newton fields
    # Hessian matrix of the optimization problem as a dense 2D tensor.
    # Note that only the lower triangular part is updated for efficiency because this matrix is symmetric by definition.
    # As a result, the values of the strictly upper triangular part is undefined.
    # In practice, this variable is re-purposed to store the Cholesky factor L st H = L @ L.T to spare memory resources.
    # TODO: Optimize storage to only allocate memory half of the Hessian matrix to sparse memory resources.
    nt_H: qd.Tensor
    nt_vec: qd.Tensor
    # Compacted list of constraints whose active state changed, used by incremental Cholesky update
    # to reduce GPU thread divergence by iterating only over constraints that need processing.
    incr_changed_idx: qd.Tensor
    incr_n_changed: qd.Tensor
    # Backward gradients
    dL_dqacc: qd.Tensor
    dL_dM: qd.Tensor
    dL_djac: qd.Tensor
    dL_daref: qd.Tensor
    dL_defc_D: qd.Tensor
    dL_dforce: qd.Tensor
    # Backward buffers for linear system solver
    bw_u: qd.Tensor
    bw_r: qd.Tensor
    bw_p: qd.Tensor
    bw_Ap: qd.Tensor
    bw_Ju: qd.Tensor
    bw_y: qd.Tensor
    bw_w: qd.Tensor
    # Timers for profiling
    timers: qd.Tensor
    # Per-env flag: 0 = use incremental Hessian+Cholesky, 1 = use full tiled rebuild
    use_full_hessian: qd.Tensor
    # Solver loop iteration counter (0-indexed, increments each iteration in the graph loop)
    solver_iter_counter: qd.Tensor
    # Always ndarray (not field): graph_do_while requires the same physical ndarray on every call.
    graph_counter: qd.types.ndarray()
    early_exit_flag: qd.Tensor


def get_constraint_state(constraint_solver, solver):
    _B = solver._B
    len_constraints_ = constraint_solver.len_constraints_

    jac_shape = (len_constraints_, solver.n_dofs_, _B)
    efc_AR_shape = maybe_shape((len_constraints_, len_constraints_, _B), solver._options.noslip_iterations > 0)
    efc_b_shape = maybe_shape((len_constraints_, _B), solver._options.noslip_iterations > 0)
    jac_relevant_dofs_shape = maybe_shape(jac_shape, constraint_solver.sparse_solve)
    jac_n_relevant_dofs_shape = maybe_shape((len_constraints_, _B), constraint_solver.sparse_solve)

    if math.prod(jac_shape) > np.iinfo(np.int32).max:
        gs.raise_exception(
            f"Jacobian shape (n_constraints={len_constraints_}, n_dofs={solver.n_dofs_}, n_envs={_B}) is too large."
        )
    if math.prod(efc_AR_shape) > np.iinfo(np.int32).max:
        gs.logger.warning(
            f"efc_AR shape (n_constraints={len_constraints_}, n_constraints={solver.n_dofs_}, n_envs={_B}) is too "
            "large. Consider manually setting a smaller 'max_collision_pairs' in RigidOptions to reduce the size of "
            "reserved memory. "
        )

    # /!\ Changing allocation order of these tensors may reduce runtime speed by >10%  /!\
    return StructConstraintState(
        n_constraints=qd.tensor(gs.qd_int, (_B,), backend=_TENSOR_BACKEND),
        qd_n_equalities=qd.tensor(gs.qd_int, (_B,), backend=_TENSOR_BACKEND),
        n_constraints_equality=qd.tensor(gs.qd_int, (_B,), backend=_TENSOR_BACKEND),
        n_constraints_frictionloss=qd.tensor(gs.qd_int, (_B,), backend=_TENSOR_BACKEND),
        is_warmstart=qd.tensor(gs.qd_bool, (_B,), backend=_TENSOR_BACKEND),
        improved=qd.tensor(gs.qd_bool, (_B,), backend=_TENSOR_BACKEND),
        cost_ws=qd.tensor(gs.qd_float, (_B,), backend=_TENSOR_BACKEND),
        gauss=qd.tensor(gs.qd_float, (_B,), backend=_TENSOR_BACKEND),
        cost=qd.tensor(gs.qd_float, (_B,), backend=_TENSOR_BACKEND),
        prev_cost=qd.tensor(gs.qd_float, (_B,), backend=_TENSOR_BACKEND),
        gtol=qd.tensor(gs.qd_float, (_B,), backend=_TENSOR_BACKEND),
        ls_it=qd.tensor(gs.qd_int, (_B,), backend=_TENSOR_BACKEND),
        ls_result=qd.tensor(gs.qd_int, (_B,), backend=_TENSOR_BACKEND),
        cg_beta=qd.tensor(gs.qd_float, (_B,), backend=_TENSOR_BACKEND),
        cg_pg_dot_pMg=qd.tensor(gs.qd_float, (_B,), backend=_TENSOR_BACKEND),
        quad_gauss=qd.tensor(gs.qd_float, (3, _B), backend=_TENSOR_BACKEND),
        ls_alpha=qd.tensor(gs.qd_float, (_B,), backend=_TENSOR_BACKEND),
        ls_p0_cost=qd.tensor(gs.qd_float, (_B,), backend=_TENSOR_BACKEND),
        ls_alpha_newton=qd.tensor(gs.qd_float, (_B,), backend=_TENSOR_BACKEND),
        ls_gtol=qd.tensor(gs.qd_float, (_B,), backend=_TENSOR_BACKEND),
        eq_sum=qd.tensor(gs.qd_float, (3, _B), backend=_TENSOR_BACKEND),
        Ma=qd.tensor(gs.qd_float, (solver.n_dofs_, _B), backend=_TENSOR_BACKEND),
        Ma_ws=qd.tensor(gs.qd_float, (solver.n_dofs_, _B), backend=_TENSOR_BACKEND),
        grad=qd.tensor(gs.qd_float, (solver.n_dofs_, _B), backend=_TENSOR_BACKEND),
        Mgrad=qd.tensor(gs.qd_float, (solver.n_dofs_, _B), backend=_TENSOR_BACKEND),
        MinvJT=qd.tensor(gs.qd_float, maybe_shape(jac_shape, solver._options.noslip_iterations > 0), backend=_TENSOR_BACKEND),
        search=qd.tensor(gs.qd_float, (solver.n_dofs_, _B), backend=_TENSOR_BACKEND),
        qfrc_constraint=qd.tensor(gs.qd_float, (solver.n_dofs_, _B), backend=_TENSOR_BACKEND),
        qacc=qd.tensor(gs.qd_float, (solver.n_dofs_, _B), backend=_TENSOR_BACKEND),
        qacc_ws=qd.tensor(gs.qd_float, (solver.n_dofs_, _B), backend=_TENSOR_BACKEND),
        qacc_prev=qd.tensor(gs.qd_float, (solver.n_dofs_, _B), backend=_TENSOR_BACKEND),
        mv=qd.tensor(gs.qd_float, (solver.n_dofs_, _B), backend=_TENSOR_BACKEND),
        cg_prev_grad=qd.tensor(gs.qd_float, (solver.n_dofs_, _B), backend=_TENSOR_BACKEND),
        cg_prev_Mgrad=qd.tensor(gs.qd_float, (solver.n_dofs_, _B), backend=_TENSOR_BACKEND),
        nt_vec=qd.tensor(gs.qd_float, (solver.n_dofs_, _B), backend=_TENSOR_BACKEND),
        nt_H=qd.tensor(gs.qd_float, (_B, solver.n_dofs_, solver.n_dofs_), backend=_TENSOR_BACKEND),
        incr_changed_idx=qd.tensor(gs.qd_int, (len_constraints_, _B), backend=_TENSOR_BACKEND),
        incr_n_changed=qd.tensor(gs.qd_int, (_B,), backend=_TENSOR_BACKEND),
        efc_b=qd.tensor(gs.qd_float, efc_b_shape, backend=_TENSOR_BACKEND),
        efc_AR=qd.tensor(gs.qd_float, efc_AR_shape, backend=_TENSOR_BACKEND),
        # Tier-1 constraint state: allocated as qd.Tensor wrappers
        # (Phase-1 migration; see perso_hugh/doc/genesis_tensor_migration.md).
        active=qd.tensor(gs.qd_bool, shape=(len_constraints_, _B), backend=_TENSOR_BACKEND),
        prev_active=qd.tensor(gs.qd_bool, (len_constraints_, _B), backend=_TENSOR_BACKEND),
        diag=qd.tensor(gs.qd_float, shape=(len_constraints_, _B), backend=_TENSOR_BACKEND),
        aref=qd.tensor(gs.qd_float, (len_constraints_, _B), backend=_TENSOR_BACKEND),
        Jaref=qd.tensor(gs.qd_float, shape=(len_constraints_, _B), backend=_TENSOR_BACKEND),
        efc_frictionloss=qd.tensor(gs.qd_float, shape=(len_constraints_, _B), backend=_TENSOR_BACKEND),
        efc_force=qd.tensor(gs.qd_float, (len_constraints_, _B), backend=_TENSOR_BACKEND),
        efc_D=qd.tensor(gs.qd_float, shape=(len_constraints_, _B), backend=_TENSOR_BACKEND),
        jv=qd.tensor(gs.qd_float, shape=(len_constraints_, _B), backend=_TENSOR_BACKEND),
        jac=qd.tensor(gs.qd_float, jac_shape, backend=_TENSOR_BACKEND),
        jac_relevant_dofs=qd.tensor(gs.qd_int, jac_relevant_dofs_shape, backend=_TENSOR_BACKEND),
        jac_n_relevant_dofs=qd.tensor(gs.qd_int, jac_n_relevant_dofs_shape, backend=_TENSOR_BACKEND),
        # Backward gradients
        dL_dqacc=qd.tensor(gs.qd_float, maybe_shape((solver.n_dofs_, _B), solver._requires_grad), backend=_TENSOR_BACKEND),
        dL_dM=qd.tensor(gs.qd_float, maybe_shape((solver.n_dofs_, solver.n_dofs_, _B), solver._requires_grad), backend=_TENSOR_BACKEND),
        dL_djac=qd.tensor(gs.qd_float, maybe_shape((len_constraints_, solver.n_dofs_, _B), solver._requires_grad), backend=_TENSOR_BACKEND),
        dL_daref=qd.tensor(gs.qd_float, maybe_shape((len_constraints_, _B), solver._requires_grad), backend=_TENSOR_BACKEND),
        dL_defc_D=qd.tensor(gs.qd_float, maybe_shape((len_constraints_, _B), solver._requires_grad), backend=_TENSOR_BACKEND),
        dL_dforce=qd.tensor(gs.qd_float, maybe_shape((solver.n_dofs_, _B), solver._requires_grad), backend=_TENSOR_BACKEND),
        bw_u=qd.tensor(gs.qd_float, maybe_shape((solver.n_dofs_, _B), solver._requires_grad), backend=_TENSOR_BACKEND),
        bw_r=qd.tensor(gs.qd_float, maybe_shape((solver.n_dofs_, _B), solver._requires_grad), backend=_TENSOR_BACKEND),
        bw_p=qd.tensor(gs.qd_float, maybe_shape((solver.n_dofs_, _B), solver._requires_grad), backend=_TENSOR_BACKEND),
        bw_Ap=qd.tensor(gs.qd_float, maybe_shape((solver.n_dofs_, _B), solver._requires_grad), backend=_TENSOR_BACKEND),
        bw_Ju=qd.tensor(gs.qd_float, maybe_shape((len_constraints_, _B), solver._requires_grad), backend=_TENSOR_BACKEND),
        bw_y=qd.tensor(gs.qd_float, maybe_shape((len_constraints_, _B), solver._requires_grad), backend=_TENSOR_BACKEND),
        bw_w=qd.tensor(gs.qd_float, maybe_shape((len_constraints_, _B), solver._requires_grad), backend=_TENSOR_BACKEND),
        # Timers
        timers=qd.tensor(qd.i64 if gs.backend != gs.metal else qd.i32, (10, _B), backend=_TENSOR_BACKEND),
        use_full_hessian=qd.tensor(qd.i32, (_B,), backend=_TENSOR_BACKEND),
        solver_iter_counter=qd.tensor(qd.i32, (), backend=_TENSOR_BACKEND),
        graph_counter=qd.ndarray(qd.i32, shape=()),
        early_exit_flag=qd.tensor(qd.i32, (), backend=_TENSOR_BACKEND),
    )


# =========================================== Collider ===========================================


@dataclasses.dataclass(frozen=True)
class StructContactData:
    geom_a: qd.Tensor
    geom_b: qd.Tensor
    penetration: qd.Tensor
    normal: qd.Tensor
    pos: qd.Tensor
    friction: qd.Tensor
    sol_params: qd.Tensor
    force: qd.Tensor
    link_a: qd.Tensor
    link_b: qd.Tensor
    pair_idx: qd.Tensor


def get_contact_data(solver, max_contact_pairs, requires_grad):
    _B = solver._B
    max_contact_pairs_ = max(max_contact_pairs, 1)

    return StructContactData(
        geom_a=qd.tensor(gs.qd_int, (max_contact_pairs_, _B), backend=_TENSOR_BACKEND),
        geom_b=qd.tensor(gs.qd_int, (max_contact_pairs_, _B), backend=_TENSOR_BACKEND),
        normal=qd.tensor(gs.qd_vec3, (max_contact_pairs_, _B), backend=_TENSOR_BACKEND, needs_grad=requires_grad),
        pos=qd.tensor(gs.qd_vec3, (max_contact_pairs_, _B), backend=_TENSOR_BACKEND, needs_grad=requires_grad),
        penetration=qd.tensor(gs.qd_float, (max_contact_pairs_, _B), backend=_TENSOR_BACKEND, needs_grad=requires_grad),
        friction=qd.tensor(gs.qd_float, (max_contact_pairs_, _B), backend=_TENSOR_BACKEND),
        sol_params=qd.Vector.tensor(7, gs.qd_float, (max_contact_pairs_, _B), backend=_TENSOR_BACKEND),
        force=qd.tensor(gs.qd_vec3, (max_contact_pairs_, _B), backend=_TENSOR_BACKEND),
        link_a=qd.tensor(gs.qd_int, (max_contact_pairs_, _B), backend=_TENSOR_BACKEND),
        link_b=qd.tensor(gs.qd_int, (max_contact_pairs_, _B), backend=_TENSOR_BACKEND),
        pair_idx=qd.tensor(gs.qd_int, (max_contact_pairs_, _B), backend=_TENSOR_BACKEND),
    )


@dataclasses.dataclass(frozen=True)
class StructDiffContactInput:
    ### Non-differentiable input data
    # Geom id of the two geometries
    geom_a: qd.Tensor
    geom_b: qd.Tensor
    # Local positions of the 3 vertices from the two geometries that define the face on the Minkowski difference
    local_pos1_a: qd.Tensor
    local_pos1_b: qd.Tensor
    local_pos1_c: qd.Tensor
    local_pos2_a: qd.Tensor
    local_pos2_b: qd.Tensor
    local_pos2_c: qd.Tensor
    # Local positions of the 1 vertex from the two geometries that define the support point for the face above
    w_local_pos1: qd.Tensor
    w_local_pos2: qd.Tensor
    # Reference id of the contact point, which is needed for the backward pass
    ref_id: qd.Tensor
    # Flag whether the contact data can be computed in numerically stable way in both the forward and backward passes
    valid: qd.Tensor
    ### Differentiable input data
    # Reference penetration depth, which is needed for computing the weight of the contact point
    ref_penetration: qd.Tensor


def get_diff_contact_input(_B, max_contacts_per_pair, is_active, requires_grad=False):
    shape = maybe_shape((_B, max_contacts_per_pair), is_active and requires_grad)
    return StructDiffContactInput(
        geom_a=qd.tensor(gs.qd_int, shape, backend=_TENSOR_BACKEND),
        geom_b=qd.tensor(gs.qd_int, shape, backend=_TENSOR_BACKEND),
        local_pos1_a=qd.Vector.tensor(3, gs.qd_float, shape, backend=_TENSOR_BACKEND),
        local_pos1_b=qd.Vector.tensor(3, gs.qd_float, shape, backend=_TENSOR_BACKEND),
        local_pos1_c=qd.Vector.tensor(3, gs.qd_float, shape, backend=_TENSOR_BACKEND),
        local_pos2_a=qd.Vector.tensor(3, gs.qd_float, shape, backend=_TENSOR_BACKEND),
        local_pos2_b=qd.Vector.tensor(3, gs.qd_float, shape, backend=_TENSOR_BACKEND),
        local_pos2_c=qd.Vector.tensor(3, gs.qd_float, shape, backend=_TENSOR_BACKEND),
        w_local_pos1=qd.Vector.tensor(3, gs.qd_float, shape, backend=_TENSOR_BACKEND),
        w_local_pos2=qd.Vector.tensor(3, gs.qd_float, shape, backend=_TENSOR_BACKEND),
        ref_id=qd.tensor(gs.qd_int, shape, backend=_TENSOR_BACKEND),
        valid=qd.tensor(gs.qd_int, shape, backend=_TENSOR_BACKEND),
        ref_penetration=qd.tensor(gs.qd_float, shape, backend=_TENSOR_BACKEND, needs_grad=True),
    )


@dataclasses.dataclass(frozen=True)
class StructSortBuffer:
    value: qd.Tensor
    i_g: qd.Tensor
    is_max: qd.Tensor


def get_sort_buffer(solver):
    _B = solver._B

    return StructSortBuffer(
        value=qd.tensor(gs.qd_float, (2 * solver.n_geoms_, _B), backend=_TENSOR_BACKEND),
        i_g=qd.tensor(gs.qd_int, (2 * solver.n_geoms_, _B), backend=_TENSOR_BACKEND),
        is_max=qd.tensor(gs.qd_bool, (2 * solver.n_geoms_, _B), backend=_TENSOR_BACKEND),
    )


@dataclasses.dataclass(frozen=True)
class StructContactCache:
    normal: qd.Tensor


def get_contact_cache(solver, n_possible_pairs):
    _B = solver._B
    return StructContactCache(
        normal=qd.Vector.tensor(3, gs.qd_float, (n_possible_pairs, _B), backend=_TENSOR_BACKEND),
    )


@dataclasses.dataclass(frozen=True)
class StructAggList:
    curr: qd.Tensor
    n: qd.Tensor
    start: qd.Tensor


def get_agg_list(solver):
    _B = solver._B
    n_entities = max(solver.n_entities, 1)

    return StructAggList(
        curr=qd.tensor(gs.qd_int, (n_entities, _B), backend=_TENSOR_BACKEND),
        n=qd.tensor(gs.qd_int, (n_entities, _B), backend=_TENSOR_BACKEND),
        start=qd.tensor(gs.qd_int, (n_entities, _B), backend=_TENSOR_BACKEND),
    )


@dataclasses.dataclass(frozen=True)
class StructContactIslandState:
    ci_edges: qd.Tensor
    edge_id: qd.Tensor
    constraint_list: qd.Tensor
    constraint_id: qd.Tensor
    entity_edge: StructAggList
    island_col: StructAggList
    island_hibernated: qd.Tensor
    island_entity: StructAggList
    entity_id: qd.Tensor
    n_edges: qd.Tensor
    n_islands: qd.Tensor
    n_stack: qd.Tensor
    entity_island: qd.Tensor
    stack: qd.Tensor
    entity_idx_to_next_entity_idx_in_hibernated_island: qd.Tensor


def get_contact_island_state(solver, collider):
    _B = solver._B
    max_contact_pairs = max(collider._collider_info.max_contact_pairs[None], 1)
    n_entities = max(solver.n_entities, 1)

    # When hibernation is enabled, the island construction adds edges for hibernated entity chains
    # in addition to contact edges. The chain construction is cyclic (last entity links back to first),
    # so worst case: each entity contributes one hibernation edge, totaling n_entities hibernation edges.
    max_hibernation_edges = n_entities if solver._use_hibernation else 0
    max_edges = max_contact_pairs + max_hibernation_edges

    return StructContactIslandState(
        ci_edges=qd.tensor(gs.qd_int, (max_edges, 2, _B), backend=_TENSOR_BACKEND),
        edge_id=qd.tensor(gs.qd_int, (max_edges * 2, _B), backend=_TENSOR_BACKEND),
        constraint_list=qd.tensor(gs.qd_int, (max_contact_pairs, _B), backend=_TENSOR_BACKEND),
        constraint_id=qd.tensor(gs.qd_int, (max_contact_pairs * 2, _B), backend=_TENSOR_BACKEND),
        entity_edge=get_agg_list(solver),
        island_col=get_agg_list(solver),
        island_hibernated=qd.tensor(gs.qd_int, (n_entities, _B), backend=_TENSOR_BACKEND),
        island_entity=get_agg_list(solver),
        entity_id=qd.tensor(gs.qd_int, (n_entities, _B), backend=_TENSOR_BACKEND),
        n_edges=qd.tensor(gs.qd_int, (_B,), backend=_TENSOR_BACKEND),
        n_islands=qd.tensor(gs.qd_int, (_B,), backend=_TENSOR_BACKEND),
        n_stack=qd.tensor(gs.qd_int, (_B,), backend=_TENSOR_BACKEND),
        entity_island=qd.tensor(gs.qd_int, (n_entities, _B), backend=_TENSOR_BACKEND),
        stack=qd.tensor(gs.qd_int, (n_entities, _B), backend=_TENSOR_BACKEND),
        entity_idx_to_next_entity_idx_in_hibernated_island=qd.tensor(gs.qd_int, (n_entities, _B), backend=_TENSOR_BACKEND),
    )


@dataclasses.dataclass(frozen=True)
class StructNarrowphaseWorkQueues:
    mpr_i_b: qd.Tensor
    mpr_i_ga: qd.Tensor
    mpr_i_gb: qd.Tensor
    mpr_i_pair: qd.Tensor
    mpr_contact_pos_0: qd.Tensor
    mpr_normal_0: qd.Tensor
    mpr_penetration_0: qd.Tensor
    gjk_i_b: qd.Tensor
    gjk_i_ga: qd.Tensor
    gjk_i_gb: qd.Tensor
    gjk_i_pair: qd.Tensor
    gjk_contact_pos_0: qd.Tensor
    gjk_normal_0: qd.Tensor
    gjk_penetration_0: qd.Tensor
    mpr_queue_size: qd.Tensor
    gjk_queue_size: qd.Tensor
    gjk_queue_size_k2: qd.Tensor
    mpr_work_counter: qd.Tensor
    gjk_work_counter: qd.Tensor


def get_narrowphase_work_queues(max_entries):
    return StructNarrowphaseWorkQueues(
        mpr_i_b=qd.tensor(gs.qd_int, (max_entries,), backend=_TENSOR_BACKEND),
        mpr_i_ga=qd.tensor(gs.qd_int, (max_entries,), backend=_TENSOR_BACKEND),
        mpr_i_gb=qd.tensor(gs.qd_int, (max_entries,), backend=_TENSOR_BACKEND),
        mpr_i_pair=qd.tensor(gs.qd_int, (max_entries,), backend=_TENSOR_BACKEND),
        mpr_contact_pos_0=qd.Vector.tensor(3, gs.qd_float, (max_entries,), backend=_TENSOR_BACKEND),
        mpr_normal_0=qd.Vector.tensor(3, gs.qd_float, (max_entries,), backend=_TENSOR_BACKEND),
        mpr_penetration_0=qd.tensor(gs.qd_float, (max_entries,), backend=_TENSOR_BACKEND),
        gjk_i_b=qd.tensor(gs.qd_int, (max_entries,), backend=_TENSOR_BACKEND),
        gjk_i_ga=qd.tensor(gs.qd_int, (max_entries,), backend=_TENSOR_BACKEND),
        gjk_i_gb=qd.tensor(gs.qd_int, (max_entries,), backend=_TENSOR_BACKEND),
        gjk_i_pair=qd.tensor(gs.qd_int, (max_entries,), backend=_TENSOR_BACKEND),
        gjk_contact_pos_0=qd.Vector.tensor(3, gs.qd_float, (max_entries,), backend=_TENSOR_BACKEND),
        gjk_normal_0=qd.Vector.tensor(3, gs.qd_float, (max_entries,), backend=_TENSOR_BACKEND),
        gjk_penetration_0=qd.tensor(gs.qd_float, (max_entries,), backend=_TENSOR_BACKEND),
        mpr_queue_size=qd.tensor(gs.qd_int, (1,), backend=_TENSOR_BACKEND),
        gjk_queue_size=qd.tensor(gs.qd_int, (1,), backend=_TENSOR_BACKEND),
        gjk_queue_size_k2=qd.tensor(gs.qd_int, (1,), backend=_TENSOR_BACKEND),
        mpr_work_counter=qd.tensor(gs.qd_int, (1,), backend=_TENSOR_BACKEND),
        gjk_work_counter=qd.tensor(gs.qd_int, (1,), backend=_TENSOR_BACKEND),
    )


@dataclasses.dataclass(frozen=True)
class StructColliderState:
    sort_buffer: StructSortBuffer
    contact_data: StructContactData
    active_buffer: qd.Tensor
    n_broad_pairs: qd.Tensor
    broad_collision_pairs: qd.Tensor
    active_buffer_awake: qd.Tensor
    active_buffer_hib: qd.Tensor
    box_depth: qd.Tensor
    box_points: qd.Tensor
    box_pts: qd.Tensor
    box_lines: qd.Tensor
    box_linesu: qd.Tensor
    box_axi: qd.Tensor
    box_ppts2: qd.Tensor
    box_pu: qd.Tensor
    xyz_max_min: qd.Tensor
    prism: qd.Tensor
    n_contacts: qd.Tensor
    n_contacts_hibernated: qd.Tensor
    first_time: qd.Tensor
    contact_cache: StructContactCache
    # Input data for differentiable contact detection used in the backward pass
    diff_contact_input: StructDiffContactInput
    narrowphase_work_queues: StructNarrowphaseWorkQueues
    contact_sort_key: qd.Tensor
    contact_sort_idx: qd.Tensor


def get_collider_state(
    solver,
    static_rigid_sim_config,
    n_possible_pairs,
    max_collision_pairs_broad_k,
    collider_info,
    collider_static_config,
):
    _B = solver._B
    n_geoms = solver.n_geoms_
    max_collision_pairs = min(solver.max_collision_pairs, n_possible_pairs)
    max_collision_pairs_broad = max_collision_pairs * max_collision_pairs_broad_k
    max_contact_pairs = max_collision_pairs * collider_static_config.n_contacts_per_pair
    requires_grad = static_rigid_sim_config.requires_grad

    box_depth_shape = maybe_shape(
        (collider_static_config.n_contacts_per_pair, _B), static_rigid_sim_config.box_box_detection
    )
    box_points_shape = maybe_shape(
        (collider_static_config.n_contacts_per_pair, _B), static_rigid_sim_config.box_box_detection
    )
    box_pts_shape = maybe_shape((6, _B), static_rigid_sim_config.box_box_detection)
    box_lines_shape = maybe_shape((4, _B), static_rigid_sim_config.box_box_detection)
    box_linesu_shape = maybe_shape((4, _B), static_rigid_sim_config.box_box_detection)
    box_axi_shape = maybe_shape((3, _B), static_rigid_sim_config.box_box_detection)
    box_ppts2_shape = maybe_shape((4, 2, _B), static_rigid_sim_config.box_box_detection)
    box_pu_shape = maybe_shape((4, _B), static_rigid_sim_config.box_box_detection)

    return StructColliderState(
        sort_buffer=get_sort_buffer(solver),
        active_buffer=qd.tensor(gs.qd_int, (n_geoms, _B), backend=_TENSOR_BACKEND),
        n_broad_pairs=qd.tensor(gs.qd_int, (_B,), backend=_TENSOR_BACKEND),
        active_buffer_awake=qd.tensor(gs.qd_int, (n_geoms, _B), backend=_TENSOR_BACKEND),
        active_buffer_hib=qd.tensor(gs.qd_int, (n_geoms, _B), backend=_TENSOR_BACKEND),
        box_depth=qd.tensor(gs.qd_float, box_depth_shape, backend=_TENSOR_BACKEND),
        box_points=qd.Vector.tensor(3, gs.qd_float, box_points_shape, backend=_TENSOR_BACKEND),
        box_pts=qd.Vector.tensor(3, gs.qd_float, box_pts_shape, backend=_TENSOR_BACKEND),
        box_lines=qd.Vector.tensor(6, gs.qd_float, box_lines_shape, backend=_TENSOR_BACKEND),
        box_linesu=qd.Vector.tensor(6, gs.qd_float, box_linesu_shape, backend=_TENSOR_BACKEND),
        box_axi=qd.Vector.tensor(3, gs.qd_float, box_axi_shape, backend=_TENSOR_BACKEND),
        box_ppts2=qd.tensor(gs.qd_float, box_ppts2_shape, backend=_TENSOR_BACKEND),
        box_pu=qd.Vector.tensor(3, gs.qd_float, box_pu_shape, backend=_TENSOR_BACKEND),
        xyz_max_min=qd.tensor(gs.qd_float, (6, _B), backend=_TENSOR_BACKEND),
        prism=qd.Vector.tensor(3, gs.qd_float, (6, _B), backend=_TENSOR_BACKEND),
        n_contacts=qd.tensor(gs.qd_int, (_B,), backend=_TENSOR_BACKEND),
        n_contacts_hibernated=qd.tensor(gs.qd_int, (_B,), backend=_TENSOR_BACKEND),
        first_time=qd.tensor(gs.qd_bool, (_B,), backend=_TENSOR_BACKEND),
        contact_cache=get_contact_cache(solver, n_possible_pairs),
        broad_collision_pairs=qd.Vector.tensor(2, gs.qd_int, (max(max_collision_pairs_broad, 1), _B), backend=_TENSOR_BACKEND),
        contact_data=get_contact_data(solver, max_contact_pairs, requires_grad),
        diff_contact_input=get_diff_contact_input(_B, max(max_contact_pairs, 1), True, requires_grad),
        narrowphase_work_queues=get_narrowphase_work_queues(
            max(max_collision_pairs_broad * _B, 1) if collider_static_config.has_non_box_plane_convex_convex else 1
        ),
        contact_sort_key=qd.tensor(gs.qd_float, (max(max_contact_pairs, 1), _B), backend=_TENSOR_BACKEND),
        contact_sort_idx=qd.tensor(gs.qd_int, (max(max_contact_pairs, 1), _B), backend=_TENSOR_BACKEND),
    )


@dataclasses.dataclass(frozen=True)
class StructColliderInfo:
    vert_neighbors: qd.Tensor
    vert_neighbor_start: qd.Tensor
    vert_n_neighbors: qd.Tensor
    # (i_ga, i_gb) -> dense pair index, or -1 if invalid. Used by SAP broadphase, narrowphase, and contact cache.
    collision_pair_idx: qd.Tensor
    max_possible_pairs: qd.Tensor
    max_collision_pairs: qd.Tensor
    max_contact_pairs: qd.Tensor
    max_collision_pairs_broad: qd.Tensor
    # Compact list of valid collision pairs. Used by all-vs-all broadphase to dispatch valid pairs to GPU threads.
    n_valid_pairs: qd.Tensor
    valid_collision_pairs: qd.Tensor
    # Terrain fields
    terrain_hf: qd.Tensor
    terrain_rc: qd.Tensor
    terrain_scale: qd.Tensor
    terrain_xyz_maxmin: qd.Tensor
    # multi contact perturbation and tolerance
    mc_perturbation: qd.Tensor
    mc_tolerance: qd.Tensor
    mpr_to_gjk_overlap_ratio: qd.Tensor
    # differentiable contact tolerance
    diff_pos_tolerance: qd.Tensor
    diff_normal_tolerance: qd.Tensor


def get_collider_info(solver, n_vert_neighbors, n_valid_pairs, collider_static_config, **kwargs):
    for geom in solver.geoms:
        if geom.type == gs.GEOM_TYPE.TERRAIN:
            terrain_hf_shape = geom.entity.terrain_hf.shape
            break
    else:
        terrain_hf_shape = 1

    return StructColliderInfo(
        vert_neighbors=qd.tensor(gs.qd_int, (max(n_vert_neighbors, 1),), backend=_TENSOR_BACKEND),
        vert_neighbor_start=qd.tensor(gs.qd_int, (solver.n_verts_,), backend=_TENSOR_BACKEND),
        vert_n_neighbors=qd.tensor(gs.qd_int, (solver.n_verts_,), backend=_TENSOR_BACKEND),
        collision_pair_idx=qd.tensor(gs.qd_int, (solver.n_geoms_, solver.n_geoms_), backend=_TENSOR_BACKEND),
        max_possible_pairs=qd.tensor(gs.qd_int, (), backend=_TENSOR_BACKEND),
        max_collision_pairs=qd.tensor(gs.qd_int, (), backend=_TENSOR_BACKEND),
        max_contact_pairs=qd.tensor(gs.qd_int, (), backend=_TENSOR_BACKEND),
        max_collision_pairs_broad=qd.tensor(gs.qd_int, (), backend=_TENSOR_BACKEND),
        n_valid_pairs=V_SCALAR_FROM(dtype=gs.qd_int, value=n_valid_pairs),
        valid_collision_pairs=qd.tensor(gs.qd_ivec2, (max(n_valid_pairs, 1),), backend=_TENSOR_BACKEND),
        terrain_hf=qd.tensor(gs.qd_float, terrain_hf_shape, backend=_TENSOR_BACKEND),
        terrain_rc=qd.tensor(gs.qd_int, (2,), backend=_TENSOR_BACKEND),
        terrain_scale=qd.tensor(gs.qd_float, (2,), backend=_TENSOR_BACKEND),
        terrain_xyz_maxmin=qd.tensor(gs.qd_float, (6,), backend=_TENSOR_BACKEND),
        mc_perturbation=V_SCALAR_FROM(dtype=gs.qd_float, value=kwargs["mc_perturbation"]),
        mc_tolerance=V_SCALAR_FROM(dtype=gs.qd_float, value=kwargs["mc_tolerance"]),
        mpr_to_gjk_overlap_ratio=V_SCALAR_FROM(dtype=gs.qd_float, value=kwargs["mpr_to_gjk_overlap_ratio"]),
        diff_pos_tolerance=V_SCALAR_FROM(dtype=gs.qd_float, value=kwargs["diff_pos_tolerance"]),
        diff_normal_tolerance=V_SCALAR_FROM(dtype=gs.qd_float, value=kwargs["diff_normal_tolerance"]),
    )


@qd.data_oriented
class StructColliderStaticConfig(metaclass=_AutoInitMeta):
    has_terrain: bool
    # True when the scene has convex-convex collision pairs not handled by
    # func_narrow_phase_convex_specializations (box-box, plane-box). Computed once
    # at scene build time by iterating all geom pairs in collider._init_static_config().
    # On GPU, the split narrowphase path runs (contact0 + multicontact + sort).
    # On CPU, falls back to the monolithic func_narrow_phase_convex_vs_convex.
    has_non_box_plane_convex_convex: bool
    has_convex_specialization: bool
    has_nonconvex_nonterrain: bool
    # maximum number of contact pairs per collision pair
    n_contacts_per_pair: int
    # ccd algorithm
    ccd_algorithm: int


# =========================================== MPR ===========================================


@dataclasses.dataclass(frozen=True)
class StructMPRSimplexSupport:
    v1: qd.Tensor
    v2: qd.Tensor
    v: qd.Tensor


def get_mpr_simplex_support(B_):
    return StructMPRSimplexSupport(
        v1=qd.Vector.tensor(3, gs.qd_float, (4, B_), backend=_TENSOR_BACKEND),
        v2=qd.Vector.tensor(3, gs.qd_float, (4, B_), backend=_TENSOR_BACKEND),
        v=qd.Vector.tensor(3, gs.qd_float, (4, B_), backend=_TENSOR_BACKEND),
    )


@dataclasses.dataclass(frozen=True)
class StructMPRState:
    simplex_support: StructMPRSimplexSupport
    simplex_size: qd.Tensor


def get_mpr_state(B_):
    return StructMPRState(
        simplex_support=get_mpr_simplex_support(B_),
        simplex_size=qd.tensor(gs.qd_int, (B_,), backend=_TENSOR_BACKEND),
    )


@dataclasses.dataclass(frozen=True)
class StructMPRInfo:
    CCD_EPS: qd.Tensor
    CCD_TOLERANCE: qd.Tensor
    CCD_ITERATIONS: qd.Tensor


def get_mpr_info(**kwargs):
    return StructMPRInfo(
        CCD_EPS=V_SCALAR_FROM(dtype=gs.qd_float, value=kwargs["CCD_EPS"]),
        CCD_TOLERANCE=V_SCALAR_FROM(dtype=gs.qd_float, value=kwargs["CCD_TOLERANCE"]),
        CCD_ITERATIONS=V_SCALAR_FROM(dtype=gs.qd_float, value=kwargs["CCD_ITERATIONS"]),
    )


# =========================================== GJK ===========================================


@dataclasses.dataclass(frozen=True)
class StructMDVertex:
    # Vertex of the Minkowski difference
    obj1: qd.Tensor
    obj2: qd.Tensor
    local_obj1: qd.Tensor
    local_obj2: qd.Tensor
    id1: qd.Tensor
    id2: qd.Tensor
    mink: qd.Tensor


def get_gjk_simplex_vertex(_B, is_active):
    shape = maybe_shape((_B, 4), is_active)
    return StructMDVertex(
        obj1=qd.Vector.tensor(3, gs.qd_float, shape, backend=_TENSOR_BACKEND),
        obj2=qd.Vector.tensor(3, gs.qd_float, shape, backend=_TENSOR_BACKEND),
        local_obj1=qd.Vector.tensor(3, gs.qd_float, shape, backend=_TENSOR_BACKEND),
        local_obj2=qd.Vector.tensor(3, gs.qd_float, shape, backend=_TENSOR_BACKEND),
        id1=qd.tensor(gs.qd_int, shape, backend=_TENSOR_BACKEND),
        id2=qd.tensor(gs.qd_int, shape, backend=_TENSOR_BACKEND),
        mink=qd.Vector.tensor(3, gs.qd_float, shape, backend=_TENSOR_BACKEND),
    )


def get_epa_polytope_vertex(_B, gjk_info, is_active):
    max_num_polytope_verts = 5 + gjk_info.epa_max_iterations[None]
    shape = maybe_shape((_B, max_num_polytope_verts), is_active)
    return StructMDVertex(
        obj1=qd.Vector.tensor(3, gs.qd_float, shape, backend=_TENSOR_BACKEND),
        obj2=qd.Vector.tensor(3, gs.qd_float, shape, backend=_TENSOR_BACKEND),
        local_obj1=qd.Vector.tensor(3, gs.qd_float, shape, backend=_TENSOR_BACKEND),
        local_obj2=qd.Vector.tensor(3, gs.qd_float, shape, backend=_TENSOR_BACKEND),
        id1=qd.tensor(gs.qd_int, shape, backend=_TENSOR_BACKEND),
        id2=qd.tensor(gs.qd_int, shape, backend=_TENSOR_BACKEND),
        mink=qd.Vector.tensor(3, gs.qd_float, shape, backend=_TENSOR_BACKEND),
    )


@dataclasses.dataclass(frozen=True)
class StructGJKSimplex:
    nverts: qd.Tensor
    dist: qd.Tensor


def get_gjk_simplex(_B, is_active):
    shape = maybe_shape((_B,), is_active)
    return StructGJKSimplex(
        nverts=qd.tensor(gs.qd_int, shape, backend=_TENSOR_BACKEND),
        dist=qd.tensor(gs.qd_float, shape, backend=_TENSOR_BACKEND),
    )


@dataclasses.dataclass(frozen=True)
class StructGJKSimplexBuffer:
    normal: qd.Tensor
    sdist: qd.Tensor


def get_gjk_simplex_buffer(_B, is_active):
    shape = maybe_shape((_B, 4), is_active)
    return StructGJKSimplexBuffer(
        normal=qd.Vector.tensor(3, gs.qd_float, shape, backend=_TENSOR_BACKEND),
        sdist=qd.tensor(gs.qd_float, shape, backend=_TENSOR_BACKEND),
    )


@dataclasses.dataclass(frozen=True)
class StructEPAPolytope:
    nverts: qd.Tensor
    nfaces: qd.Tensor
    nfaces_map: qd.Tensor
    horizon_nedges: qd.Tensor
    horizon_w: qd.Tensor


def get_epa_polytope(_B, is_active):
    shape = maybe_shape((_B,), is_active)
    return StructEPAPolytope(
        nverts=qd.tensor(gs.qd_int, shape, backend=_TENSOR_BACKEND),
        nfaces=qd.tensor(gs.qd_int, shape, backend=_TENSOR_BACKEND),
        nfaces_map=qd.tensor(gs.qd_int, shape, backend=_TENSOR_BACKEND),
        horizon_nedges=qd.tensor(gs.qd_int, shape, backend=_TENSOR_BACKEND),
        horizon_w=qd.Vector.tensor(3, gs.qd_float, shape, backend=_TENSOR_BACKEND),
    )


@dataclasses.dataclass(frozen=True)
class StructEPAPolytopeFace:
    verts_idx: qd.Tensor
    adj_idx: qd.Tensor
    normal: qd.Tensor
    dist2: qd.Tensor
    map_idx: qd.Tensor
    visited: qd.Tensor


def get_epa_polytope_face(_B, polytope_max_faces, is_active):
    shape = maybe_shape((_B, polytope_max_faces), is_active)
    return StructEPAPolytopeFace(
        verts_idx=qd.Vector.tensor(3, gs.qd_int, shape, backend=_TENSOR_BACKEND),
        adj_idx=qd.Vector.tensor(3, gs.qd_int, shape, backend=_TENSOR_BACKEND),
        normal=qd.Vector.tensor(3, gs.qd_float, shape, backend=_TENSOR_BACKEND),
        dist2=qd.tensor(gs.qd_float, shape, backend=_TENSOR_BACKEND),
        map_idx=qd.tensor(gs.qd_int, shape, backend=_TENSOR_BACKEND),
        visited=qd.tensor(gs.qd_int, shape, backend=_TENSOR_BACKEND),
    )


@dataclasses.dataclass(frozen=True)
class StructEPAPolytopeHorizonData:
    face_idx: qd.Tensor
    edge_idx: qd.Tensor


def get_epa_polytope_horizon_data(_B, polytope_max_horizons, is_active):
    shape = maybe_shape((_B, polytope_max_horizons), is_active)
    return StructEPAPolytopeHorizonData(
        face_idx=qd.tensor(gs.qd_int, shape, backend=_TENSOR_BACKEND),
        edge_idx=qd.tensor(gs.qd_int, shape, backend=_TENSOR_BACKEND),
    )


@dataclasses.dataclass(frozen=True)
class StructContactFace:
    vert1: qd.Tensor
    vert2: qd.Tensor
    endverts: qd.Tensor
    normal1: qd.Tensor
    normal2: qd.Tensor
    id1: qd.Tensor
    id2: qd.Tensor


def get_contact_face(_B, max_contact_polygon_verts, is_active):
    shape = maybe_shape((_B, max_contact_polygon_verts), is_active)
    return StructContactFace(
        vert1=qd.Vector.tensor(3, gs.qd_float, shape, backend=_TENSOR_BACKEND),
        vert2=qd.Vector.tensor(3, gs.qd_float, shape, backend=_TENSOR_BACKEND),
        endverts=qd.Vector.tensor(3, gs.qd_float, shape, backend=_TENSOR_BACKEND),
        normal1=qd.Vector.tensor(3, gs.qd_float, shape, backend=_TENSOR_BACKEND),
        normal2=qd.Vector.tensor(3, gs.qd_float, shape, backend=_TENSOR_BACKEND),
        id1=qd.tensor(gs.qd_int, shape, backend=_TENSOR_BACKEND),
        id2=qd.tensor(gs.qd_int, shape, backend=_TENSOR_BACKEND),
    )


@dataclasses.dataclass(frozen=True)
class StructContactNormal:
    endverts: qd.Tensor
    normal: qd.Tensor
    id: qd.Tensor


def get_contact_normal(_B, max_contact_polygon_verts, is_active):
    shape = maybe_shape((_B, max_contact_polygon_verts), is_active)
    return StructContactNormal(
        endverts=qd.Vector.tensor(3, gs.qd_float, shape, backend=_TENSOR_BACKEND),
        normal=qd.Vector.tensor(3, gs.qd_float, shape, backend=_TENSOR_BACKEND),
        id=qd.tensor(gs.qd_int, shape, backend=_TENSOR_BACKEND),
    )


@dataclasses.dataclass(frozen=True)
class StructContactHalfspace:
    normal: qd.Tensor
    dist: qd.Tensor


def get_contact_halfspace(_B, max_contact_polygon_verts, is_active):
    shape = maybe_shape((_B, max_contact_polygon_verts), is_active)
    return StructContactHalfspace(
        normal=qd.Vector.tensor(3, gs.qd_float, shape, backend=_TENSOR_BACKEND),
        dist=qd.tensor(gs.qd_float, shape, backend=_TENSOR_BACKEND),
    )


@dataclasses.dataclass(frozen=True)
class StructWitness:
    point_obj1: qd.Tensor
    point_obj2: qd.Tensor


def get_witness(_B, max_contacts_per_pair, is_active):
    shape = maybe_shape((_B, max_contacts_per_pair), is_active)
    return StructWitness(
        point_obj1=qd.Vector.tensor(3, gs.qd_float, shape, backend=_TENSOR_BACKEND),
        point_obj2=qd.Vector.tensor(3, gs.qd_float, shape, backend=_TENSOR_BACKEND),
    )


@dataclasses.dataclass(frozen=True)
class StructGJKState:
    support_mesh_prev_vertex_id: qd.Tensor
    simplex_vertex: StructMDVertex
    simplex_buffer: StructGJKSimplexBuffer
    simplex: StructGJKSimplex
    simplex_vertex_intersect: StructMDVertex
    simplex_buffer_intersect: StructGJKSimplexBuffer
    nsimplex: qd.Tensor
    last_searched_simplex_vertex_id: qd.Tensor
    polytope: StructEPAPolytope
    polytope_verts: StructMDVertex
    polytope_faces: StructEPAPolytopeFace
    polytope_faces_map: qd.Tensor
    polytope_horizon_data: StructEPAPolytopeHorizonData
    polytope_horizon_stack: StructEPAPolytopeHorizonData
    contact_faces: StructContactFace
    contact_normals: StructContactNormal
    contact_halfspaces: StructContactHalfspace
    contact_clipped_polygons: qd.Tensor
    multi_contact_flag: qd.Tensor
    witness: StructWitness
    n_witness: qd.Tensor
    n_contacts: qd.Tensor
    contact_pos: qd.Tensor
    normal: qd.Tensor
    is_col: qd.Tensor
    penetration: qd.Tensor
    distance: qd.Tensor
    # Differentiable contact detection
    diff_contact_input: StructDiffContactInput
    n_diff_contact_input: qd.Tensor
    diff_penetration: qd.Tensor


def get_gjk_state(_B, static_rigid_sim_config, gjk_info, is_active, requires_grad=False):
    enable_mujoco_compatibility = static_rigid_sim_config.enable_mujoco_compatibility
    polytope_max_faces = gjk_info.polytope_max_faces[None]
    max_contacts_per_pair = gjk_info.max_contacts_per_pair[None]
    max_contact_polygon_verts = gjk_info.max_contact_polygon_verts[None]

    # FIXME: Define GJKState and MujocoCompatGJKState that derives from the former but defines additional attributes
    return StructGJKState(
        # GJK simplex
        support_mesh_prev_vertex_id=qd.tensor(gs.qd_int, (_B, 2), backend=_TENSOR_BACKEND),
        simplex_vertex=get_gjk_simplex_vertex(_B, is_active),
        simplex_buffer=get_gjk_simplex_buffer(_B, is_active),
        simplex=get_gjk_simplex(_B, is_active),
        last_searched_simplex_vertex_id=qd.tensor(gs.qd_int, (_B,), backend=_TENSOR_BACKEND),
        simplex_vertex_intersect=get_gjk_simplex_vertex(_B, is_active),
        simplex_buffer_intersect=get_gjk_simplex_buffer(_B, is_active),
        nsimplex=qd.tensor(gs.qd_int, (_B,), backend=_TENSOR_BACKEND),
        # EPA polytope
        polytope=get_epa_polytope(_B, is_active),
        polytope_verts=get_epa_polytope_vertex(_B, gjk_info, is_active),
        polytope_faces=get_epa_polytope_face(_B, polytope_max_faces, is_active),
        polytope_faces_map=qd.tensor(gs.qd_int, (_B, polytope_max_faces), backend=_TENSOR_BACKEND),
        polytope_horizon_data=get_epa_polytope_horizon_data(_B, 6 + gjk_info.epa_max_iterations[None], is_active),
        polytope_horizon_stack=get_epa_polytope_horizon_data(_B, polytope_max_faces * 3, is_active),
        # Multi-contact detection (MuJoCo compatibility)
        contact_faces=get_contact_face(_B, max_contact_polygon_verts, is_active),
        contact_normals=get_contact_normal(_B, max_contact_polygon_verts, is_active),
        contact_halfspaces=get_contact_halfspace(_B, max_contact_polygon_verts, is_active),
        contact_clipped_polygons=qd.Vector.tensor(3, gs.qd_float, (_B, 2, max_contact_polygon_verts), backend=_TENSOR_BACKEND),
        multi_contact_flag=qd.tensor(gs.qd_bool, (_B,), backend=_TENSOR_BACKEND),
        # Final results
        witness=get_witness(_B, max_contacts_per_pair, is_active),
        n_witness=qd.tensor(gs.qd_int, (_B,), backend=_TENSOR_BACKEND),
        n_contacts=qd.tensor(gs.qd_int, (_B,), backend=_TENSOR_BACKEND),
        contact_pos=qd.Vector.tensor(3, gs.qd_float, (_B, max_contacts_per_pair), backend=_TENSOR_BACKEND),
        normal=qd.Vector.tensor(3, gs.qd_float, (_B, max_contacts_per_pair), backend=_TENSOR_BACKEND),
        is_col=qd.tensor(gs.qd_bool, (_B,), backend=_TENSOR_BACKEND),
        penetration=qd.tensor(gs.qd_float, (_B,), backend=_TENSOR_BACKEND),
        distance=qd.tensor(gs.qd_float, (_B,), backend=_TENSOR_BACKEND),
        diff_contact_input=get_diff_contact_input(_B, max(max_contacts_per_pair, 1), is_active, requires_grad),
        n_diff_contact_input=qd.tensor(gs.qd_int, (_B,), backend=_TENSOR_BACKEND),
        diff_penetration=qd.tensor(gs.qd_float, maybe_shape((_B, max_contacts_per_pair), requires_grad), backend=_TENSOR_BACKEND),
    )


def get_gjk_state_contact_only(_B):
    """Minimal GJK state for contact detection only (no EPA, no multi-contact).

    Used by kernel 1 to run func_gjk as a boolean overlap test. All EPA polytope,
    multi-contact, and differentiable fields are allocated at dummy size (1,) since
    func_gjk never accesses them.
    """
    _dummy_B = 1

    return StructGJKState(
        support_mesh_prev_vertex_id=qd.tensor(gs.qd_int, (_B, 2), backend=_TENSOR_BACKEND),
        simplex_vertex=get_gjk_simplex_vertex(_B, is_active=True),
        simplex_buffer=get_gjk_simplex_buffer(_B, is_active=True),
        simplex=get_gjk_simplex(_B, is_active=True),
        last_searched_simplex_vertex_id=qd.tensor(gs.qd_int, (_B,), backend=_TENSOR_BACKEND),
        simplex_vertex_intersect=get_gjk_simplex_vertex(_B, is_active=True),
        simplex_buffer_intersect=get_gjk_simplex_buffer(_B, is_active=True),
        nsimplex=qd.tensor(gs.qd_int, (_B,), backend=_TENSOR_BACKEND),
        # EPA — dummy allocations, never accessed by func_gjk
        polytope=get_epa_polytope(_dummy_B, is_active=True),
        polytope_verts=StructMDVertex(
            obj1=qd.Vector.tensor(3, gs.qd_float, (1, 1), backend=_TENSOR_BACKEND),
            obj2=qd.Vector.tensor(3, gs.qd_float, (1, 1), backend=_TENSOR_BACKEND),
            local_obj1=qd.Vector.tensor(3, gs.qd_float, (1, 1), backend=_TENSOR_BACKEND),
            local_obj2=qd.Vector.tensor(3, gs.qd_float, (1, 1), backend=_TENSOR_BACKEND),
            id1=qd.tensor(gs.qd_int, (1, 1), backend=_TENSOR_BACKEND),
            id2=qd.tensor(gs.qd_int, (1, 1), backend=_TENSOR_BACKEND),
            mink=qd.Vector.tensor(3, gs.qd_float, (1, 1), backend=_TENSOR_BACKEND),
        ),
        polytope_faces=get_epa_polytope_face(_dummy_B, 1, is_active=True),
        polytope_faces_map=qd.tensor(gs.qd_int, (1, 1), backend=_TENSOR_BACKEND),
        polytope_horizon_data=get_epa_polytope_horizon_data(_dummy_B, 1, is_active=True),
        polytope_horizon_stack=get_epa_polytope_horizon_data(_dummy_B, 1, is_active=True),
        # Multi-contact — dummy
        contact_faces=get_contact_face(_dummy_B, 1, is_active=True),
        contact_normals=get_contact_normal(_dummy_B, 1, is_active=True),
        contact_halfspaces=get_contact_halfspace(_dummy_B, 1, is_active=True),
        contact_clipped_polygons=qd.Vector.tensor(3, gs.qd_float, (1, 2, 1), backend=_TENSOR_BACKEND),
        multi_contact_flag=qd.tensor(gs.qd_bool, (_B,), backend=_TENSOR_BACKEND),
        # Results — full _B for fields func_gjk writes; dummy for EPA-only fields
        witness=get_witness(_B, 1, is_active=True),
        n_witness=qd.tensor(gs.qd_int, (_B,), backend=_TENSOR_BACKEND),
        n_contacts=qd.tensor(gs.qd_int, (1,), backend=_TENSOR_BACKEND),
        contact_pos=qd.Vector.tensor(3, gs.qd_float, (1, 1), backend=_TENSOR_BACKEND),
        normal=qd.Vector.tensor(3, gs.qd_float, (1, 1), backend=_TENSOR_BACKEND),
        is_col=qd.tensor(gs.qd_bool, (1,), backend=_TENSOR_BACKEND),
        penetration=qd.tensor(gs.qd_float, (1,), backend=_TENSOR_BACKEND),
        distance=qd.tensor(gs.qd_float, (_B,), backend=_TENSOR_BACKEND),
        diff_contact_input=get_diff_contact_input(_dummy_B, 1, is_active=False),
        n_diff_contact_input=qd.tensor(gs.qd_int, (1,), backend=_TENSOR_BACKEND),
        diff_penetration=qd.tensor(gs.qd_float, (), backend=_TENSOR_BACKEND),
    )


@dataclasses.dataclass(frozen=True)
class StructGJKInfo:
    max_contacts_per_pair: qd.Tensor
    max_contact_polygon_verts: qd.Tensor
    # Maximum number of iterations for GJK and EPA algorithms
    gjk_max_iterations: qd.Tensor
    epa_max_iterations: qd.Tensor
    FLOAT_MIN: qd.Tensor
    FLOAT_MIN_SQ: qd.Tensor
    FLOAT_MAX: qd.Tensor
    FLOAT_MAX_SQ: qd.Tensor
    # Tolerance for stopping GJK and EPA algorithms when they converge (only for non-discrete geometries).
    tolerance: qd.Tensor
    # If the distance between two objects is smaller than this value, we consider them colliding.
    collision_eps: qd.Tensor
    # In safe GJK, we do not allow degenerate simplex to happen, because it becomes the main reason of EPA errors.
    # To prevent degeneracy, we throw away the simplex that has smaller degeneracy measure (e.g. colinearity,
    # coplanarity) than this threshold.
    simplex_max_degeneracy_sq: qd.Tensor
    polytope_max_faces: qd.Tensor
    # Threshold for reprojection error when we compute the witness points from the polytope. In computing the
    # witness points, we project the origin onto the polytope faces and compute the barycentric coordinates of the
    # projected point. To confirm the projection is valid, we compute the projected point using the barycentric
    # coordinates and compare it with the original projected point. If the difference is larger than this threshold,
    # we consider the projection invalid, because it means numerical errors are too large.
    # We check both relative and absolute errors: the relative error catches numerically degenerate faces,
    # while the absolute error prevents false rejections on smooth geometries (e.g. spheres) where
    # polytope faces become extremely small near convergence, amplifying the relative error.
    polytope_max_rel_reprojection_error: qd.Tensor
    polytope_max_abs_reprojection_error: qd.Tensor
    # Tolerance for normal alignment between (face-face) or (edge-face). The normals should align within this
    # tolerance to be considered as a valid parallel contact.
    contact_face_tol: qd.Tensor
    contact_edge_tol: qd.Tensor
    # Epsilon values for differentiable contact. [eps_boundary] denotes the maximum distance between the face
    # and the support point in the direction of the face normal. If this distance is 0, the face is on the
    # boundary of the Minkowski difference. For [eps_distance], the distance between the origin and the face
    # should not exceed this eps value plus the default EPA depth. For [eps_affine], the affine coordinates
    # of the origin's projection onto the face should not violate [0, 1] range by this eps value.
    # FIXME: Adjust these values based on the case study.
    diff_contact_eps_boundary: qd.Tensor
    diff_contact_eps_distance: qd.Tensor
    diff_contact_eps_affine: qd.Tensor
    # The minimum norm of the normal to be considered as a valid normal in the differentiable formulation.
    diff_contact_min_normal_norm: qd.Tensor
    # The minimum penetration depth to be considered as a valid contact in the differentiable formulation.
    # The contact with penetration depth smaller than this value is ignored in the differentiable formulation.
    # This should be large enough to be safe from numerical errors, because in the backward pass, the computed
    # penetration depth could be different from the forward pass due to the numerical errors. If this value is
    # too small, the non-zero penetration depth could be falsely computed to 0 in the backward pass and thus
    # produce nan values for the contact normal.
    diff_contact_min_penetration: qd.Tensor


def get_gjk_info(**kwargs):
    return StructGJKInfo(
        max_contacts_per_pair=V_SCALAR_FROM(dtype=gs.qd_int, value=kwargs["max_contacts_per_pair"]),
        max_contact_polygon_verts=V_SCALAR_FROM(dtype=gs.qd_int, value=kwargs["max_contact_polygon_verts"]),
        gjk_max_iterations=V_SCALAR_FROM(dtype=gs.qd_int, value=kwargs["gjk_max_iterations"]),
        epa_max_iterations=V_SCALAR_FROM(dtype=gs.qd_int, value=kwargs["epa_max_iterations"]),
        FLOAT_MIN=V_SCALAR_FROM(dtype=gs.qd_float, value=kwargs["FLOAT_MIN"]),
        FLOAT_MIN_SQ=V_SCALAR_FROM(dtype=gs.qd_float, value=kwargs["FLOAT_MIN"] ** 2),
        FLOAT_MAX=V_SCALAR_FROM(dtype=gs.qd_float, value=kwargs["FLOAT_MAX"]),
        FLOAT_MAX_SQ=V_SCALAR_FROM(dtype=gs.qd_float, value=kwargs["FLOAT_MAX"] ** 2),
        tolerance=V_SCALAR_FROM(dtype=gs.qd_float, value=kwargs["tolerance"]),
        collision_eps=V_SCALAR_FROM(dtype=gs.qd_float, value=kwargs["collision_eps"]),
        simplex_max_degeneracy_sq=V_SCALAR_FROM(dtype=gs.qd_float, value=kwargs["simplex_max_degeneracy_sq"]),
        polytope_max_faces=V_SCALAR_FROM(dtype=gs.qd_int, value=kwargs["polytope_max_faces"]),
        polytope_max_rel_reprojection_error=V_SCALAR_FROM(
            dtype=gs.qd_float, value=kwargs["polytope_max_rel_reprojection_error"]
        ),
        polytope_max_abs_reprojection_error=V_SCALAR_FROM(
            dtype=gs.qd_float, value=kwargs["polytope_max_abs_reprojection_error"]
        ),
        contact_face_tol=V_SCALAR_FROM(dtype=gs.qd_float, value=kwargs["contact_face_tol"]),
        contact_edge_tol=V_SCALAR_FROM(dtype=gs.qd_float, value=kwargs["contact_edge_tol"]),
        diff_contact_eps_boundary=V_SCALAR_FROM(dtype=gs.qd_float, value=kwargs["diff_contact_eps_boundary"]),
        diff_contact_eps_distance=V_SCALAR_FROM(dtype=gs.qd_float, value=kwargs["diff_contact_eps_distance"]),
        diff_contact_eps_affine=V_SCALAR_FROM(dtype=gs.qd_float, value=kwargs["diff_contact_eps_affine"]),
        diff_contact_min_normal_norm=V_SCALAR_FROM(dtype=gs.qd_float, value=kwargs["diff_contact_min_normal_norm"]),
        diff_contact_min_penetration=V_SCALAR_FROM(dtype=gs.qd_float, value=kwargs["diff_contact_min_penetration"]),
    )


@qd.data_oriented
class StructGJKStaticConfig(metaclass=_AutoInitMeta):
    enable_mujoco_multi_contact: bool


# =========================================== SupportField ===========================================


@dataclasses.dataclass(frozen=True)
class StructSupportFieldInfo:
    support_cell_start: qd.Tensor
    support_v: qd.Tensor
    support_vid: qd.Tensor
    support_res: qd.Tensor


def get_support_field_info(n_geoms, n_support_cells, support_res):
    return StructSupportFieldInfo(
        support_cell_start=qd.tensor(gs.qd_int, (max(n_geoms, 1),), backend=_TENSOR_BACKEND),
        support_v=qd.Vector.tensor(3, gs.qd_float, (max(n_support_cells, 1),), backend=_TENSOR_BACKEND),
        support_vid=qd.tensor(gs.qd_int, (max(n_support_cells, 1),), backend=_TENSOR_BACKEND),
        support_res=V_SCALAR_FROM(dtype=gs.qd_int, value=support_res),
    )


# =========================================== SDF ===========================================


@dataclasses.dataclass(frozen=True)
class StructSDFGeomInfo:
    T_mesh_to_sdf: qd.Tensor
    sdf_res: qd.Tensor
    sdf_max: qd.Tensor
    sdf_cell_size: qd.Tensor
    sdf_cell_start: qd.Tensor


def get_sdf_geom_info(n_geoms):
    return StructSDFGeomInfo(
        T_mesh_to_sdf=qd.Matrix.tensor(4, 4, gs.qd_float, (n_geoms,), backend=_TENSOR_BACKEND),
        sdf_res=qd.Vector.tensor(3, gs.qd_int, (n_geoms,), backend=_TENSOR_BACKEND),
        sdf_max=qd.tensor(gs.qd_float, (n_geoms,), backend=_TENSOR_BACKEND),
        sdf_cell_size=qd.tensor(gs.qd_float, (n_geoms,), backend=_TENSOR_BACKEND),
        sdf_cell_start=qd.tensor(gs.qd_int, (n_geoms,), backend=_TENSOR_BACKEND),
    )


@dataclasses.dataclass(frozen=True)
class StructSDFInfo:
    geoms_info: StructSDFGeomInfo
    geoms_sdf_start: qd.Tensor
    geoms_sdf_val: qd.Tensor
    geoms_sdf_grad: qd.Tensor
    geoms_sdf_closest_vert: qd.Tensor


def get_sdf_info(n_geoms, n_cells):
    if math.prod((n_cells, 3)) > np.iinfo(np.int32).max:
        gs.raise_exception(
            f"SDF Gradient shape (n_cells={n_cells}, 3) is too large. Consider manually setting larger "
            "'sdf_cell_size' in 'gs.materials.Rigid' options."
        )

    return StructSDFInfo(
        geoms_info=get_sdf_geom_info(max(n_geoms, 1)),
        geoms_sdf_start=qd.tensor(gs.qd_int, (max(n_geoms, 1),), backend=_TENSOR_BACKEND),
        geoms_sdf_val=qd.tensor(gs.qd_float, (max(n_cells, 1),), backend=_TENSOR_BACKEND),
        geoms_sdf_grad=qd.Vector.tensor(3, gs.qd_float, (max(n_cells, 1),), backend=_TENSOR_BACKEND),
        geoms_sdf_closest_vert=qd.tensor(gs.qd_int, (max(n_cells, 1),), backend=_TENSOR_BACKEND),
    )


# =========================================== DofsInfo and DofsState ===========================================


@dataclasses.dataclass(frozen=True)
class StructDofsInfo:
    entity_idx: qd.Tensor
    stiffness: qd.Tensor
    invweight: qd.Tensor
    armature: qd.Tensor
    damping: qd.Tensor
    frictionloss: qd.Tensor
    motion_ang: qd.Tensor
    motion_vel: qd.Tensor
    limit: qd.Tensor
    act_gain: qd.Tensor
    act_bias: qd.Tensor
    force_range: qd.Tensor


def get_dofs_info(solver):
    shape = (solver.n_dofs_, solver._B) if solver._options.batch_dofs_info else (solver.n_dofs_,)

    return StructDofsInfo(
        entity_idx=qd.tensor(gs.qd_int, shape, backend=_TENSOR_BACKEND),
        stiffness=qd.tensor(gs.qd_float, shape, backend=_TENSOR_BACKEND),
        invweight=qd.tensor(gs.qd_float, shape, backend=_TENSOR_BACKEND),
        armature=qd.tensor(gs.qd_float, shape, backend=_TENSOR_BACKEND),
        damping=qd.tensor(gs.qd_float, shape, backend=_TENSOR_BACKEND),
        frictionloss=qd.tensor(gs.qd_float, shape, backend=_TENSOR_BACKEND),
        motion_ang=qd.tensor(gs.qd_vec3, shape, backend=_TENSOR_BACKEND),
        motion_vel=qd.tensor(gs.qd_vec3, shape, backend=_TENSOR_BACKEND),
        limit=qd.tensor(gs.qd_vec2, shape, backend=_TENSOR_BACKEND),
        act_gain=qd.tensor(gs.qd_float, shape, backend=_TENSOR_BACKEND),
        act_bias=qd.tensor(gs.qd_vec3, shape, backend=_TENSOR_BACKEND),
        force_range=qd.tensor(gs.qd_vec2, shape, backend=_TENSOR_BACKEND),
    )


@dataclasses.dataclass(frozen=True)
class StructDofsState:
    # *_bw: Cache to avoid overwriting for backward pass
    force: qd.Tensor
    qf_bias: qd.Tensor
    qf_passive: qd.Tensor
    qf_actuator: qd.Tensor
    qf_applied: qd.Tensor
    act_length: qd.Tensor
    pos: qd.Tensor
    vel: qd.Tensor
    vel_prev: qd.Tensor
    vel_next: qd.Tensor
    acc: qd.Tensor
    acc_bw: qd.Tensor
    acc_smooth: qd.Tensor
    acc_smooth_bw: qd.Tensor
    qf_smooth: qd.Tensor
    qf_constraint: qd.Tensor
    cdof_ang: qd.Tensor
    cdof_vel: qd.Tensor
    cdofvel_ang: qd.Tensor
    cdofvel_vel: qd.Tensor
    cdofd_ang: qd.Tensor
    cdofd_vel: qd.Tensor
    f_vel: qd.Tensor
    f_ang: qd.Tensor
    ctrl_force: qd.Tensor
    ctrl_pos: qd.Tensor
    ctrl_vel: qd.Tensor
    ctrl_mode: qd.Tensor
    hibernated: qd.Tensor


def get_dofs_state(solver):
    shape = (solver.n_dofs_, solver._B)
    requires_grad = solver._requires_grad
    shape_bw = maybe_shape((2, *shape), requires_grad)

    return StructDofsState(
        force=qd.tensor(gs.qd_float, shape, backend=_TENSOR_BACKEND, needs_grad=requires_grad),
        qf_bias=qd.tensor(gs.qd_float, shape, backend=_TENSOR_BACKEND, needs_grad=requires_grad),
        qf_passive=qd.tensor(gs.qd_float, shape, backend=_TENSOR_BACKEND, needs_grad=requires_grad),
        qf_actuator=qd.tensor(gs.qd_float, shape, backend=_TENSOR_BACKEND, needs_grad=requires_grad),
        qf_applied=qd.tensor(gs.qd_float, shape, backend=_TENSOR_BACKEND, needs_grad=requires_grad),
        act_length=qd.tensor(gs.qd_float, shape, backend=_TENSOR_BACKEND, needs_grad=requires_grad),
        pos=qd.tensor(gs.qd_float, shape, backend=_TENSOR_BACKEND, needs_grad=requires_grad),
        vel=qd.tensor(gs.qd_float, shape, backend=_TENSOR_BACKEND, needs_grad=requires_grad),
        vel_prev=qd.tensor(gs.qd_float, shape, backend=_TENSOR_BACKEND, needs_grad=requires_grad),
        vel_next=qd.tensor(gs.qd_float, shape, backend=_TENSOR_BACKEND, needs_grad=requires_grad),
        acc=qd.tensor(gs.qd_float, shape, backend=_TENSOR_BACKEND, needs_grad=requires_grad),
        acc_bw=qd.tensor(gs.qd_float, shape_bw, backend=_TENSOR_BACKEND, needs_grad=requires_grad),
        acc_smooth=qd.tensor(gs.qd_float, shape, backend=_TENSOR_BACKEND, needs_grad=requires_grad),
        acc_smooth_bw=qd.tensor(gs.qd_float, shape_bw, backend=_TENSOR_BACKEND, needs_grad=requires_grad),
        qf_smooth=qd.tensor(gs.qd_float, shape, backend=_TENSOR_BACKEND, needs_grad=requires_grad),
        qf_constraint=qd.tensor(gs.qd_float, shape, backend=_TENSOR_BACKEND, needs_grad=requires_grad),
        cdof_ang=qd.tensor(gs.qd_vec3, shape, backend=_TENSOR_BACKEND, needs_grad=requires_grad),
        cdof_vel=qd.tensor(gs.qd_vec3, shape, backend=_TENSOR_BACKEND, needs_grad=requires_grad),
        cdofvel_ang=qd.tensor(gs.qd_vec3, shape, backend=_TENSOR_BACKEND, needs_grad=requires_grad),
        cdofvel_vel=qd.tensor(gs.qd_vec3, shape, backend=_TENSOR_BACKEND, needs_grad=requires_grad),
        cdofd_ang=qd.tensor(gs.qd_vec3, shape, backend=_TENSOR_BACKEND, needs_grad=requires_grad),
        cdofd_vel=qd.tensor(gs.qd_vec3, shape, backend=_TENSOR_BACKEND, needs_grad=requires_grad),
        f_vel=qd.tensor(gs.qd_vec3, shape, backend=_TENSOR_BACKEND, needs_grad=requires_grad),
        f_ang=qd.tensor(gs.qd_vec3, shape, backend=_TENSOR_BACKEND, needs_grad=requires_grad),
        ctrl_force=qd.tensor(gs.qd_float, shape, backend=_TENSOR_BACKEND, needs_grad=requires_grad),
        ctrl_pos=qd.tensor(gs.qd_float, shape, backend=_TENSOR_BACKEND, needs_grad=requires_grad),
        ctrl_vel=qd.tensor(gs.qd_float, shape, backend=_TENSOR_BACKEND, needs_grad=requires_grad),
        ctrl_mode=qd.tensor(gs.qd_int, shape, backend=_TENSOR_BACKEND),
        hibernated=qd.tensor(gs.qd_int, shape, backend=_TENSOR_BACKEND),
    )


# =========================================== LinksState and LinksInfo ===========================================


@dataclasses.dataclass(frozen=True)
class StructLinksState:
    # *_bw: Cache to avoid overwriting for backward pass
    cinr_inertial: qd.Tensor
    cinr_pos: qd.Tensor
    cinr_quat: qd.Tensor
    cinr_mass: qd.Tensor
    crb_inertial: qd.Tensor
    crb_pos: qd.Tensor
    crb_quat: qd.Tensor
    crb_mass: qd.Tensor
    cdd_vel: qd.Tensor
    cdd_ang: qd.Tensor
    pos: qd.Tensor
    quat: qd.Tensor
    pos_bw: qd.Tensor
    quat_bw: qd.Tensor
    i_pos: qd.Tensor
    i_pos_bw: qd.Tensor
    i_quat: qd.Tensor
    j_pos: qd.Tensor
    j_quat: qd.Tensor
    j_pos_bw: qd.Tensor
    j_quat_bw: qd.Tensor
    j_vel: qd.Tensor
    j_ang: qd.Tensor
    cd_ang: qd.Tensor
    cd_vel: qd.Tensor
    cd_ang_bw: qd.Tensor
    cd_vel_bw: qd.Tensor
    mass_sum: qd.Tensor
    root_COM: qd.Tensor  # COM of the kinematic tree
    root_COM_bw: qd.Tensor
    mass_shift: qd.Tensor
    i_pos_shift: qd.Tensor
    cacc_ang: qd.Tensor
    cacc_lin: qd.Tensor
    cfrc_ang: qd.Tensor
    cfrc_vel: qd.Tensor
    cfrc_applied_ang: qd.Tensor
    cfrc_applied_vel: qd.Tensor
    cfrc_coupling_ang: qd.Tensor
    cfrc_coupling_vel: qd.Tensor
    contact_force: qd.Tensor
    hibernated: qd.Tensor


def get_links_state(solver):
    max_n_joints_per_link = solver._static_rigid_sim_config.max_n_joints_per_link
    shape = (solver.n_links_, solver._B)
    requires_grad = solver._requires_grad
    shape_bw = (solver.n_links_, max(max_n_joints_per_link + 1, 1), solver._B)

    return StructLinksState(
        cinr_inertial=qd.tensor(gs.qd_mat3, shape, backend=_TENSOR_BACKEND, needs_grad=requires_grad),
        cinr_pos=qd.tensor(gs.qd_vec3, shape, backend=_TENSOR_BACKEND, needs_grad=requires_grad),
        cinr_quat=qd.tensor(gs.qd_vec4, shape, backend=_TENSOR_BACKEND, needs_grad=requires_grad),
        cinr_mass=qd.tensor(gs.qd_float, shape, backend=_TENSOR_BACKEND, needs_grad=requires_grad),
        crb_inertial=qd.tensor(gs.qd_mat3, shape, backend=_TENSOR_BACKEND, needs_grad=requires_grad),
        crb_pos=qd.tensor(gs.qd_vec3, shape, backend=_TENSOR_BACKEND, needs_grad=requires_grad),
        crb_quat=qd.tensor(gs.qd_vec4, shape, backend=_TENSOR_BACKEND, needs_grad=requires_grad),
        crb_mass=qd.tensor(gs.qd_float, shape, backend=_TENSOR_BACKEND, needs_grad=requires_grad),
        cdd_vel=qd.tensor(gs.qd_vec3, shape, backend=_TENSOR_BACKEND, needs_grad=requires_grad),
        cdd_ang=qd.tensor(gs.qd_vec3, shape, backend=_TENSOR_BACKEND, needs_grad=requires_grad),
        pos=qd.tensor(gs.qd_vec3, shape, backend=_TENSOR_BACKEND, needs_grad=requires_grad),
        quat=qd.tensor(gs.qd_vec4, shape, backend=_TENSOR_BACKEND, needs_grad=requires_grad),
        pos_bw=qd.tensor(gs.qd_vec3, shape_bw, backend=_TENSOR_BACKEND, needs_grad=requires_grad),
        quat_bw=qd.tensor(gs.qd_vec4, shape_bw, backend=_TENSOR_BACKEND, needs_grad=requires_grad),
        i_pos=qd.tensor(gs.qd_vec3, shape, backend=_TENSOR_BACKEND, needs_grad=requires_grad),
        i_pos_bw=qd.tensor(gs.qd_vec3, shape, backend=_TENSOR_BACKEND, needs_grad=requires_grad),
        i_quat=qd.tensor(gs.qd_vec4, shape, backend=_TENSOR_BACKEND, needs_grad=requires_grad),
        j_pos=qd.tensor(gs.qd_vec3, shape, backend=_TENSOR_BACKEND, needs_grad=requires_grad),
        j_quat=qd.tensor(gs.qd_vec4, shape, backend=_TENSOR_BACKEND, needs_grad=requires_grad),
        j_pos_bw=qd.tensor(gs.qd_vec3, shape_bw, backend=_TENSOR_BACKEND, needs_grad=requires_grad),
        j_quat_bw=qd.tensor(gs.qd_vec4, shape_bw, backend=_TENSOR_BACKEND, needs_grad=requires_grad),
        j_vel=qd.tensor(gs.qd_vec3, shape, backend=_TENSOR_BACKEND, needs_grad=requires_grad),
        j_ang=qd.tensor(gs.qd_vec3, shape, backend=_TENSOR_BACKEND, needs_grad=requires_grad),
        cd_ang=qd.tensor(gs.qd_vec3, shape, backend=_TENSOR_BACKEND, needs_grad=requires_grad),
        cd_vel=qd.tensor(gs.qd_vec3, shape, backend=_TENSOR_BACKEND, needs_grad=requires_grad),
        cd_ang_bw=qd.tensor(gs.qd_vec3, shape_bw, backend=_TENSOR_BACKEND, needs_grad=requires_grad),
        cd_vel_bw=qd.tensor(gs.qd_vec3, shape_bw, backend=_TENSOR_BACKEND, needs_grad=requires_grad),
        mass_sum=qd.tensor(gs.qd_float, shape, backend=_TENSOR_BACKEND, needs_grad=requires_grad),
        root_COM=qd.tensor(gs.qd_vec3, shape, backend=_TENSOR_BACKEND, needs_grad=requires_grad),
        root_COM_bw=qd.tensor(gs.qd_vec3, shape, backend=_TENSOR_BACKEND, needs_grad=requires_grad),
        mass_shift=qd.tensor(gs.qd_float, shape, backend=_TENSOR_BACKEND, needs_grad=requires_grad),
        i_pos_shift=qd.tensor(gs.qd_vec3, shape, backend=_TENSOR_BACKEND, needs_grad=requires_grad),
        cacc_ang=qd.tensor(gs.qd_vec3, shape, backend=_TENSOR_BACKEND, needs_grad=requires_grad),
        cacc_lin=qd.tensor(gs.qd_vec3, shape, backend=_TENSOR_BACKEND, needs_grad=requires_grad),
        cfrc_ang=qd.tensor(gs.qd_vec3, shape, backend=_TENSOR_BACKEND, needs_grad=requires_grad),
        cfrc_vel=qd.tensor(gs.qd_vec3, shape, backend=_TENSOR_BACKEND, needs_grad=requires_grad),
        cfrc_applied_ang=qd.tensor(gs.qd_vec3, shape, backend=_TENSOR_BACKEND, needs_grad=requires_grad),
        cfrc_applied_vel=qd.tensor(gs.qd_vec3, shape, backend=_TENSOR_BACKEND, needs_grad=requires_grad),
        cfrc_coupling_ang=qd.tensor(gs.qd_vec3, shape, backend=_TENSOR_BACKEND, needs_grad=requires_grad),
        cfrc_coupling_vel=qd.tensor(gs.qd_vec3, shape, backend=_TENSOR_BACKEND, needs_grad=requires_grad),
        contact_force=qd.tensor(gs.qd_vec3, shape, backend=_TENSOR_BACKEND, needs_grad=requires_grad),
        hibernated=qd.tensor(gs.qd_int, shape, backend=_TENSOR_BACKEND),
    )


@dataclasses.dataclass(frozen=True)
class StructLinksInfo:
    parent_idx: qd.Tensor
    root_idx: qd.Tensor
    q_start: qd.Tensor
    dof_start: qd.Tensor
    joint_start: qd.Tensor
    q_end: qd.Tensor
    dof_end: qd.Tensor
    joint_end: qd.Tensor
    n_dofs: qd.Tensor
    pos: qd.Tensor
    quat: qd.Tensor
    invweight: qd.Tensor
    is_fixed: qd.Tensor
    inertial_pos: qd.Tensor
    inertial_quat: qd.Tensor
    inertial_i: qd.Tensor
    inertial_mass: qd.Tensor
    entity_idx: qd.Tensor
    # Heterogeneous simulation support: per-link geom/vgeom index ranges
    geom_start: qd.Tensor
    geom_end: qd.Tensor
    vgeom_start: qd.Tensor
    vgeom_end: qd.Tensor


def get_links_info(solver):
    links_info_shape = (solver.n_links_, solver._B) if solver._options.batch_links_info else solver.n_links_

    return StructLinksInfo(
        parent_idx=qd.tensor(gs.qd_int, links_info_shape, backend=_TENSOR_BACKEND),
        root_idx=qd.tensor(gs.qd_int, links_info_shape, backend=_TENSOR_BACKEND),
        q_start=qd.tensor(gs.qd_int, links_info_shape, backend=_TENSOR_BACKEND),
        dof_start=qd.tensor(gs.qd_int, links_info_shape, backend=_TENSOR_BACKEND),
        joint_start=qd.tensor(gs.qd_int, links_info_shape, backend=_TENSOR_BACKEND),
        q_end=qd.tensor(gs.qd_int, links_info_shape, backend=_TENSOR_BACKEND),
        dof_end=qd.tensor(gs.qd_int, links_info_shape, backend=_TENSOR_BACKEND),
        joint_end=qd.tensor(gs.qd_int, links_info_shape, backend=_TENSOR_BACKEND),
        n_dofs=qd.tensor(gs.qd_int, links_info_shape, backend=_TENSOR_BACKEND),
        pos=qd.tensor(gs.qd_vec3, links_info_shape, backend=_TENSOR_BACKEND),
        quat=qd.tensor(gs.qd_vec4, links_info_shape, backend=_TENSOR_BACKEND),
        invweight=qd.tensor(gs.qd_vec2, links_info_shape, backend=_TENSOR_BACKEND),
        is_fixed=qd.tensor(gs.qd_bool, links_info_shape, backend=_TENSOR_BACKEND),
        inertial_pos=qd.tensor(gs.qd_vec3, links_info_shape, backend=_TENSOR_BACKEND),
        inertial_quat=qd.tensor(gs.qd_vec4, links_info_shape, backend=_TENSOR_BACKEND),
        inertial_i=qd.tensor(gs.qd_mat3, links_info_shape, backend=_TENSOR_BACKEND),
        inertial_mass=qd.tensor(gs.qd_float, links_info_shape, backend=_TENSOR_BACKEND),
        entity_idx=qd.tensor(gs.qd_int, links_info_shape, backend=_TENSOR_BACKEND),
        # Heterogeneous simulation support: per-link geom/vgeom index ranges
        geom_start=qd.tensor(gs.qd_int, links_info_shape, backend=_TENSOR_BACKEND),
        geom_end=qd.tensor(gs.qd_int, links_info_shape, backend=_TENSOR_BACKEND),
        vgeom_start=qd.tensor(gs.qd_int, links_info_shape, backend=_TENSOR_BACKEND),
        vgeom_end=qd.tensor(gs.qd_int, links_info_shape, backend=_TENSOR_BACKEND),
    )


# =========================================== JointsInfo and JointsState ===========================================


@dataclasses.dataclass(frozen=True)
class StructJointsInfo:
    type: qd.Tensor
    sol_params: qd.Tensor
    q_start: qd.Tensor
    dof_start: qd.Tensor
    q_end: qd.Tensor
    dof_end: qd.Tensor
    n_dofs: qd.Tensor
    pos: qd.Tensor


def get_joints_info(solver):
    shape = (solver.n_joints_, solver._B) if solver._options.batch_joints_info else (solver.n_joints_,)

    return StructJointsInfo(
        type=qd.tensor(gs.qd_int, shape, backend=_TENSOR_BACKEND),
        sol_params=qd.tensor(gs.qd_vec7, shape, backend=_TENSOR_BACKEND),
        q_start=qd.tensor(gs.qd_int, shape, backend=_TENSOR_BACKEND),
        dof_start=qd.tensor(gs.qd_int, shape, backend=_TENSOR_BACKEND),
        q_end=qd.tensor(gs.qd_int, shape, backend=_TENSOR_BACKEND),
        dof_end=qd.tensor(gs.qd_int, shape, backend=_TENSOR_BACKEND),
        n_dofs=qd.tensor(gs.qd_int, shape, backend=_TENSOR_BACKEND),
        pos=qd.tensor(gs.qd_vec3, shape, backend=_TENSOR_BACKEND),
    )


@dataclasses.dataclass(frozen=True)
class StructJointsState:
    xanchor: qd.Tensor
    xaxis: qd.Tensor


def get_joints_state(solver):
    shape = (solver.n_joints_, solver._B)
    requires_grad = solver._requires_grad

    return StructJointsState(
        xanchor=qd.tensor(gs.qd_vec3, shape, backend=_TENSOR_BACKEND, needs_grad=requires_grad),
        xaxis=qd.tensor(gs.qd_vec3, shape, backend=_TENSOR_BACKEND, needs_grad=requires_grad),
    )


# =========================================== GeomsInfo and GeomsState ===========================================


@dataclasses.dataclass(frozen=True)
class StructGeomsInfo:
    pos: qd.Tensor
    center: qd.Tensor
    quat: qd.Tensor
    data: qd.Tensor
    link_idx: qd.Tensor
    type: qd.Tensor
    friction: qd.Tensor
    sol_params: qd.Tensor
    vert_num: qd.Tensor
    vert_start: qd.Tensor
    vert_end: qd.Tensor
    verts_state_start: qd.Tensor
    verts_state_end: qd.Tensor
    face_num: qd.Tensor
    face_start: qd.Tensor
    face_end: qd.Tensor
    edge_num: qd.Tensor
    edge_start: qd.Tensor
    edge_end: qd.Tensor
    is_convex: qd.Tensor
    contype: qd.Tensor
    conaffinity: qd.Tensor
    is_fixed: qd.Tensor
    is_decomposed: qd.Tensor
    needs_coup: qd.Tensor
    coup_friction: qd.Tensor
    coup_softness: qd.Tensor
    coup_restitution: qd.Tensor


def get_geoms_info(solver):
    shape = (solver.n_geoms_,)

    return StructGeomsInfo(
        pos=qd.tensor(gs.qd_vec3, shape, backend=_TENSOR_BACKEND),
        center=qd.tensor(gs.qd_vec3, shape, backend=_TENSOR_BACKEND),
        quat=qd.tensor(gs.qd_vec4, shape, backend=_TENSOR_BACKEND),
        data=qd.tensor(gs.qd_vec7, shape, backend=_TENSOR_BACKEND),
        link_idx=qd.tensor(gs.qd_int, shape, backend=_TENSOR_BACKEND),
        type=qd.tensor(gs.qd_int, shape, backend=_TENSOR_BACKEND),
        friction=qd.tensor(gs.qd_float, shape, backend=_TENSOR_BACKEND),
        sol_params=qd.tensor(gs.qd_vec7, shape, backend=_TENSOR_BACKEND),
        vert_num=qd.tensor(gs.qd_int, shape, backend=_TENSOR_BACKEND),
        vert_start=qd.tensor(gs.qd_int, shape, backend=_TENSOR_BACKEND),
        vert_end=qd.tensor(gs.qd_int, shape, backend=_TENSOR_BACKEND),
        verts_state_start=qd.tensor(gs.qd_int, shape, backend=_TENSOR_BACKEND),
        verts_state_end=qd.tensor(gs.qd_int, shape, backend=_TENSOR_BACKEND),
        face_num=qd.tensor(gs.qd_int, shape, backend=_TENSOR_BACKEND),
        face_start=qd.tensor(gs.qd_int, shape, backend=_TENSOR_BACKEND),
        face_end=qd.tensor(gs.qd_int, shape, backend=_TENSOR_BACKEND),
        edge_num=qd.tensor(gs.qd_int, shape, backend=_TENSOR_BACKEND),
        edge_start=qd.tensor(gs.qd_int, shape, backend=_TENSOR_BACKEND),
        edge_end=qd.tensor(gs.qd_int, shape, backend=_TENSOR_BACKEND),
        is_convex=qd.tensor(gs.qd_bool, shape, backend=_TENSOR_BACKEND),
        contype=qd.tensor(gs.qd_int, shape, backend=_TENSOR_BACKEND),
        conaffinity=qd.tensor(gs.qd_int, shape, backend=_TENSOR_BACKEND),
        is_fixed=qd.tensor(gs.qd_bool, shape, backend=_TENSOR_BACKEND),
        is_decomposed=qd.tensor(gs.qd_bool, shape, backend=_TENSOR_BACKEND),
        needs_coup=qd.tensor(gs.qd_int, shape, backend=_TENSOR_BACKEND),
        coup_friction=qd.tensor(gs.qd_float, shape, backend=_TENSOR_BACKEND),
        coup_softness=qd.tensor(gs.qd_float, shape, backend=_TENSOR_BACKEND),
        coup_restitution=qd.tensor(gs.qd_float, shape, backend=_TENSOR_BACKEND),
    )


@dataclasses.dataclass(frozen=True)
class StructGeomsState:
    pos: qd.Tensor
    quat: qd.Tensor
    aabb_min: qd.Tensor
    aabb_max: qd.Tensor
    verts_updated: qd.Tensor
    min_buffer_idx: qd.Tensor
    max_buffer_idx: qd.Tensor
    hibernated: qd.Tensor
    friction_ratio: qd.Tensor


def get_geoms_state(solver):
    shape = (solver.n_geoms_, solver._B)
    requires_grad = solver._static_rigid_sim_config.requires_grad

    return StructGeomsState(
        pos=qd.tensor(gs.qd_vec3, shape, backend=_TENSOR_BACKEND, needs_grad=requires_grad),
        quat=qd.tensor(gs.qd_vec4, shape, backend=_TENSOR_BACKEND, needs_grad=requires_grad),
        aabb_min=qd.tensor(gs.qd_vec3, shape, backend=_TENSOR_BACKEND),
        aabb_max=qd.tensor(gs.qd_vec3, shape, backend=_TENSOR_BACKEND),
        verts_updated=qd.tensor(gs.qd_bool, shape, backend=_TENSOR_BACKEND),
        min_buffer_idx=qd.tensor(gs.qd_int, shape, backend=_TENSOR_BACKEND),
        max_buffer_idx=qd.tensor(gs.qd_int, shape, backend=_TENSOR_BACKEND),
        hibernated=qd.tensor(gs.qd_int, shape, backend=_TENSOR_BACKEND),
        friction_ratio=qd.tensor(gs.qd_float, shape, backend=_TENSOR_BACKEND),
    )


# =========================================== VertsInfo ===========================================


@dataclasses.dataclass(frozen=True)
class StructVertsInfo:
    init_pos: qd.Tensor
    init_normal: qd.Tensor
    geom_idx: qd.Tensor
    init_center_pos: qd.Tensor
    verts_state_idx: qd.Tensor
    is_fixed: qd.Tensor


def get_verts_info(solver):
    shape = (solver.n_verts_,)

    return StructVertsInfo(
        init_pos=qd.tensor(gs.qd_vec3, shape, backend=_TENSOR_BACKEND),
        init_normal=qd.tensor(gs.qd_vec3, shape, backend=_TENSOR_BACKEND),
        geom_idx=qd.tensor(gs.qd_int, shape, backend=_TENSOR_BACKEND),
        init_center_pos=qd.tensor(gs.qd_vec3, shape, backend=_TENSOR_BACKEND),
        verts_state_idx=qd.tensor(gs.qd_int, shape, backend=_TENSOR_BACKEND),
        is_fixed=qd.tensor(gs.qd_bool, shape, backend=_TENSOR_BACKEND),
    )


# =========================================== FacesInfo ===========================================


@dataclasses.dataclass(frozen=True)
class StructFacesInfo:
    verts_idx: qd.Tensor
    geom_idx: qd.Tensor


def get_faces_info(solver):
    shape = (solver.n_faces_,)

    return StructFacesInfo(
        verts_idx=qd.tensor(gs.qd_ivec3, shape, backend=_TENSOR_BACKEND),
        geom_idx=qd.tensor(gs.qd_int, shape, backend=_TENSOR_BACKEND),
    )


# =========================================== EdgesInfo ===========================================


@dataclasses.dataclass(frozen=True)
class StructEdgesInfo:
    v0: qd.Tensor
    v1: qd.Tensor
    length: qd.Tensor


def get_edges_info(solver):
    shape = (solver.n_edges_,)

    return StructEdgesInfo(
        v0=qd.tensor(gs.qd_int, shape, backend=_TENSOR_BACKEND),
        v1=qd.tensor(gs.qd_int, shape, backend=_TENSOR_BACKEND),
        length=qd.tensor(gs.qd_float, shape, backend=_TENSOR_BACKEND),
    )


# =========================================== VertsState ===========================================


@dataclasses.dataclass(frozen=True)
class StructVertsState:
    pos: qd.Tensor


def get_free_verts_state(solver):
    return StructVertsState(
        pos=qd.tensor(gs.qd_vec3, (solver.n_free_verts_, solver._B), backend=_TENSOR_BACKEND),
    )


def get_fixed_verts_state(solver):
    return StructVertsState(
        pos=qd.tensor(gs.qd_vec3, (solver.n_fixed_verts_,), backend=_TENSOR_BACKEND),
    )


# =========================================== VvertsInfo ===========================================


@dataclasses.dataclass(frozen=True)
class StructVvertsInfo:
    init_pos: qd.Tensor
    init_vnormal: qd.Tensor
    vgeom_idx: qd.Tensor


def get_vverts_info(solver):
    shape = (solver.n_vverts_,)

    return StructVvertsInfo(
        init_pos=qd.tensor(gs.qd_vec3, shape, backend=_TENSOR_BACKEND),
        init_vnormal=qd.tensor(gs.qd_vec3, shape, backend=_TENSOR_BACKEND),
        vgeom_idx=qd.tensor(gs.qd_int, shape, backend=_TENSOR_BACKEND),
    )


# =========================================== VfacesInfo ===========================================


@dataclasses.dataclass(frozen=True)
class StructVfacesInfo:
    vverts_idx: qd.Tensor
    vgeom_idx: qd.Tensor


def get_vfaces_info(solver):
    shape = (solver.n_vfaces_,)

    return StructVfacesInfo(
        vverts_idx=qd.tensor(gs.qd_ivec3, shape, backend=_TENSOR_BACKEND),
        vgeom_idx=qd.tensor(gs.qd_int, shape, backend=_TENSOR_BACKEND),
    )


# =========================================== VgeomsInfo ===========================================


@dataclasses.dataclass(frozen=True)
class StructVgeomsInfo:
    pos: qd.Tensor
    quat: qd.Tensor
    link_idx: qd.Tensor
    vvert_num: qd.Tensor
    vvert_start: qd.Tensor
    vvert_end: qd.Tensor
    vface_num: qd.Tensor
    vface_start: qd.Tensor
    vface_end: qd.Tensor
    color: qd.Tensor


def get_vgeoms_info(solver):
    shape = (solver.n_vgeoms_,)

    return StructVgeomsInfo(
        pos=qd.tensor(gs.qd_vec3, shape, backend=_TENSOR_BACKEND),
        quat=qd.tensor(gs.qd_vec4, shape, backend=_TENSOR_BACKEND),
        link_idx=qd.tensor(gs.qd_int, shape, backend=_TENSOR_BACKEND),
        vvert_num=qd.tensor(gs.qd_int, shape, backend=_TENSOR_BACKEND),
        vvert_start=qd.tensor(gs.qd_int, shape, backend=_TENSOR_BACKEND),
        vvert_end=qd.tensor(gs.qd_int, shape, backend=_TENSOR_BACKEND),
        vface_num=qd.tensor(gs.qd_int, shape, backend=_TENSOR_BACKEND),
        vface_start=qd.tensor(gs.qd_int, shape, backend=_TENSOR_BACKEND),
        vface_end=qd.tensor(gs.qd_int, shape, backend=_TENSOR_BACKEND),
        color=qd.tensor(gs.qd_vec4, shape, backend=_TENSOR_BACKEND),
    )


# =========================================== VGeomsState ===========================================


@dataclasses.dataclass(frozen=True)
class StructVgeomsState:
    pos: qd.Tensor
    quat: qd.Tensor


def get_vgeoms_state(solver):
    shape = (solver.n_vgeoms_, solver._B)

    return StructVgeomsState(
        pos=qd.tensor(gs.qd_vec3, shape, backend=_TENSOR_BACKEND),
        quat=qd.tensor(gs.qd_vec4, shape, backend=_TENSOR_BACKEND),
    )


# =========================================== EqualitiesInfo ===========================================


@dataclasses.dataclass(frozen=True)
class StructEqualitiesInfo:
    eq_obj1id: qd.Tensor
    eq_obj2id: qd.Tensor
    eq_data: qd.Tensor
    eq_type: qd.Tensor
    sol_params: qd.Tensor


def get_equalities_info(solver):
    shape = (solver.n_candidate_equalities_, solver._B)

    return StructEqualitiesInfo(
        eq_obj1id=qd.tensor(gs.qd_int, shape, backend=_TENSOR_BACKEND),
        eq_obj2id=qd.tensor(gs.qd_int, shape, backend=_TENSOR_BACKEND),
        eq_data=qd.tensor(gs.qd_vec11, shape, backend=_TENSOR_BACKEND),
        eq_type=qd.tensor(gs.qd_int, shape, backend=_TENSOR_BACKEND),
        sol_params=qd.tensor(gs.qd_vec7, shape, backend=_TENSOR_BACKEND),
    )


# =========================================== EntitiesInfo ===========================================


@dataclasses.dataclass(frozen=True)
class StructEntitiesInfo:
    dof_start: qd.Tensor
    dof_end: qd.Tensor
    n_dofs: qd.Tensor
    link_start: qd.Tensor
    link_end: qd.Tensor
    n_links: qd.Tensor
    geom_start: qd.Tensor
    geom_end: qd.Tensor
    n_geoms: qd.Tensor
    gravity_compensation: qd.Tensor
    is_local_collision_mask: qd.Tensor


def get_entities_info(solver):
    shape = (solver.n_entities_,)

    return StructEntitiesInfo(
        dof_start=qd.tensor(gs.qd_int, shape, backend=_TENSOR_BACKEND),
        dof_end=qd.tensor(gs.qd_int, shape, backend=_TENSOR_BACKEND),
        n_dofs=qd.tensor(gs.qd_int, shape, backend=_TENSOR_BACKEND),
        link_start=qd.tensor(gs.qd_int, shape, backend=_TENSOR_BACKEND),
        link_end=qd.tensor(gs.qd_int, shape, backend=_TENSOR_BACKEND),
        n_links=qd.tensor(gs.qd_int, shape, backend=_TENSOR_BACKEND),
        geom_start=qd.tensor(gs.qd_int, shape, backend=_TENSOR_BACKEND),
        geom_end=qd.tensor(gs.qd_int, shape, backend=_TENSOR_BACKEND),
        n_geoms=qd.tensor(gs.qd_int, shape, backend=_TENSOR_BACKEND),
        gravity_compensation=qd.tensor(gs.qd_float, shape, backend=_TENSOR_BACKEND),
        is_local_collision_mask=qd.tensor(gs.qd_bool, shape, backend=_TENSOR_BACKEND),
    )


# =========================================== EntitiesState ===========================================


@dataclasses.dataclass(frozen=True)
class StructEntitiesState:
    hibernated: qd.Tensor


def get_entities_state(solver):
    return StructEntitiesState(
        hibernated=qd.tensor(gs.qd_int, (solver.n_entities_, solver._B), backend=_TENSOR_BACKEND),
    )


# =========================================== RigidAdjointCache ===========================================
@dataclasses.dataclass(frozen=True)
class StructRigidAdjointCache:
    # This cache stores intermediate values during rigid body simulation to use Quadrants's AD. Quadrants's AD requires
    # us not to overwrite the values that have been read during the forward pass, so we need to store the intemediate
    # values in this cache to avoid overwriting them. Specifically, after we compute next frame's qpos, dofs_vel, and
    # dofs_acc, we need to store them in this cache because we overwrite the values in the next frame. See how
    # [kernel_save_adjoint_cache] is used in [rigid_solver.py] to store the values in this cache.
    qpos: qd.Tensor
    dofs_vel: qd.Tensor
    dofs_acc: qd.Tensor


def get_rigid_adjoint_cache(solver):
    substeps_local = solver._sim.substeps_local
    requires_grad = solver._requires_grad

    return StructRigidAdjointCache(
        qpos=qd.tensor(gs.qd_float, (substeps_local + 1, solver.n_qs_, solver._B), backend=_TENSOR_BACKEND, needs_grad=requires_grad),
        dofs_vel=qd.tensor(gs.qd_float, (substeps_local + 1, solver.n_dofs_, solver._B), backend=_TENSOR_BACKEND, needs_grad=requires_grad),
        dofs_acc=qd.tensor(gs.qd_float, (substeps_local + 1, solver.n_dofs_, solver._B), backend=_TENSOR_BACKEND, needs_grad=requires_grad),
    )


# =================================== StructRigidSimStaticConfig ===================================


@qd.data_oriented
class StructRigidSimStaticConfig(metaclass=_AutoInitMeta):
    backend: int
    para_level: int
    enable_collision: bool
    use_hibernation: bool
    batch_links_info: bool
    batch_dofs_info: bool
    batch_joints_info: bool
    enable_heterogeneous: bool
    enable_mujoco_compatibility: bool
    enable_multi_contact: bool
    enable_joint_limit: bool
    box_box_detection: bool
    sparse_solve: bool
    integrator: int
    solver_type: int
    requires_grad: bool
    prefer_decomposed_solver: int = -1  # -1 = None (auto), 0 = False, 1 = True
    parallel_init: bool = False  # parallelize init over (constraints, envs) when GPU is not saturated by envs alone
    broadphase_traversal: int = 0
    enable_tiled_cholesky_mass_matrix: bool = False
    enable_tiled_cholesky_hessian: bool = False
    tiled_n_dofs_per_entity: int = -1
    tiled_n_dofs: int = -1
    max_n_links_per_entity: int = -1
    max_n_joints_per_link: int = -1
    max_n_dofs_per_joint: int = -1
    max_n_qs_per_link: int = -1
    max_n_dofs_per_entity: int = -1
    max_n_dofs_per_link: int = -1
    max_n_geoms_per_entity: int = -1
    n_entities: int = -1
    n_links: int = -1
    n_geoms: int = -1


# =========================================== DataManager ===========================================


@qd.data_oriented
class DataManager:
    def __init__(self, solver, kinematic_only):
        self.rigid_global_info = get_rigid_global_info(solver, kinematic_only)

        self.dofs_info = get_dofs_info(solver)
        self.dofs_state = get_dofs_state(solver)
        self.links_info = get_links_info(solver)
        self.links_state = get_links_state(solver)
        self.joints_info = get_joints_info(solver)
        self.joints_state = get_joints_state(solver)

        self.entities_info = get_entities_info(solver)
        self.entities_state = get_entities_state(solver)

        self.vverts_info = get_vverts_info(solver)
        self.vfaces_info = get_vfaces_info(solver)

        self.vgeoms_info = get_vgeoms_info(solver)
        self.vgeoms_state = get_vgeoms_state(solver)

        if not kinematic_only:
            self.geoms_info = get_geoms_info(solver)
            self.geoms_state = get_geoms_state(solver)

            self.verts_info = get_verts_info(solver)
            self.faces_info = get_faces_info(solver)
            self.edges_info = get_edges_info(solver)

            self.free_verts_state = get_free_verts_state(solver)
            self.fixed_verts_state = get_fixed_verts_state(solver)

            self.equalities_info = get_equalities_info(solver)

        if solver._static_rigid_sim_config.requires_grad:
            # Data structures required for backward pass
            self.dofs_state_adjoint_cache = get_dofs_state(solver)
            self.links_state_adjoint_cache = get_links_state(solver)
            self.joints_state_adjoint_cache = get_joints_state(solver)
            self.geoms_state_adjoint_cache = get_geoms_state(solver)

        self.rigid_adjoint_cache = get_rigid_adjoint_cache(solver)
        self.errno = qd.tensor(gs.qd_int, (solver._B,), backend=_TENSOR_BACKEND)


# =========================================== ViewerRaycastResult ===========================================


@dataclasses.dataclass(frozen=True)
class StructViewerRaycastResult:
    distance: qd.Tensor
    geom_idx: qd.Tensor
    hit_point: qd.Tensor
    normal: qd.Tensor
    env_idx: qd.Tensor


def get_viewer_raycast_result():
    return StructViewerRaycastResult(
        distance=qd.tensor(gs.qd_float, (), backend=_TENSOR_BACKEND),
        geom_idx=qd.tensor(gs.qd_int, (), backend=_TENSOR_BACKEND),
        hit_point=qd.Vector.tensor(3, gs.qd_float, (), backend=_TENSOR_BACKEND),
        normal=qd.Vector.tensor(3, gs.qd_float, (), backend=_TENSOR_BACKEND),
        env_idx=qd.tensor(gs.qd_int, (), backend=_TENSOR_BACKEND),
    )


DofsState = StructDofsState if gs.use_ndarray else qd.template()
DofsInfo = StructDofsInfo if gs.use_ndarray else qd.template()
GeomsState = StructGeomsState if gs.use_ndarray else qd.template()
GeomsInfo = StructGeomsInfo if gs.use_ndarray else qd.template()
GeomsInitAABB = qd.Tensor
LinksState = StructLinksState if gs.use_ndarray else qd.template()
LinksInfo = StructLinksInfo if gs.use_ndarray else qd.template()
JointsInfo = StructJointsInfo if gs.use_ndarray else qd.template()
JointsState = StructJointsState if gs.use_ndarray else qd.template()
VertsState = StructVertsState if gs.use_ndarray else qd.template()
VertsInfo = StructVertsInfo if gs.use_ndarray else qd.template()
EdgesInfo = StructEdgesInfo if gs.use_ndarray else qd.template()
FacesInfo = StructFacesInfo if gs.use_ndarray else qd.template()
VVertsInfo = StructVvertsInfo if gs.use_ndarray else qd.template()
VFacesInfo = StructVfacesInfo if gs.use_ndarray else qd.template()
VGeomsInfo = StructVgeomsInfo if gs.use_ndarray else qd.template()
VGeomsState = StructVgeomsState if gs.use_ndarray else qd.template()
EntitiesState = StructEntitiesState if gs.use_ndarray else qd.template()
EntitiesInfo = StructEntitiesInfo if gs.use_ndarray else qd.template()
EqualitiesInfo = StructEqualitiesInfo if gs.use_ndarray else qd.template()
RigidGlobalInfo = StructRigidGlobalInfo if gs.use_ndarray else qd.template()
ColliderState = StructColliderState if gs.use_ndarray else qd.template()
ColliderInfo = StructColliderInfo if gs.use_ndarray else qd.template()
MPRState = StructMPRState if gs.use_ndarray else qd.template()
MPRInfo = StructMPRInfo if gs.use_ndarray else qd.template()
SupportFieldInfo = StructSupportFieldInfo if gs.use_ndarray else qd.template()
ConstraintState = StructConstraintState if gs.use_ndarray else qd.template()
GJKState = StructGJKState if gs.use_ndarray else qd.template()
GJKInfo = StructGJKInfo if gs.use_ndarray else qd.template()
SDFInfo = StructSDFInfo if gs.use_ndarray else qd.template()
ContactIslandState = StructContactIslandState if gs.use_ndarray else qd.template()
DiffContactInput = StructDiffContactInput if gs.use_ndarray else qd.template()
RigidAdjointCache = StructRigidAdjointCache if gs.use_ndarray else qd.template()
RaycastResult = StructViewerRaycastResult if gs.use_ndarray else qd.template()
