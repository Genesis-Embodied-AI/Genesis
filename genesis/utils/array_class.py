import dataclasses
import math
from enum import IntEnum

import quadrants as qd
from typing_extensions import dataclass_transform  # Made it into standard lib from Python 3.12
import numpy as np
import torch

import genesis as gs


def _tensor_backend():
    return qd.Backend.NDARRAY if gs.use_ndarray else qd.Backend.FIELD


def V(*args, **kwargs):
    return qd.tensor(*args, backend=_tensor_backend(), **kwargs)


def V_VEC(*args, **kwargs):
    return qd.Vector.tensor(*args, backend=_tensor_backend(), **kwargs)


def V_MAT(*args, **kwargs):
    return qd.Matrix.tensor(*args, backend=_tensor_backend(), **kwargs)


def maybe_shape(shape, is_on):
    return shape if is_on else ()


@dataclass_transform(eq_default=True, kw_only_default=False, frozen_default=True)
class AutoInitMeta(type):
    """Metaclass that generates __init__ from annotations, like a mutable dataclass."""

    def __new__(cls, name, bases, namespace):
        names = tuple(namespace["__annotations__"].keys())
        defaults = {k: namespace[k] for k in names if k in namespace}

        def __init__(self, *args, **kwargs):
            # Initialize assigned arguments from defaults
            assigned = defaults.copy()

            # Assign positional arguments
            if len(args) > len(names):
                raise TypeError(f"{name}() takes {len(names)} positional arguments but {len(args)} were given")
            for key, value in zip(names, args):
                assigned[key] = value

            # Assign keyword arguments
            for key, value in kwargs.items():
                if key not in names:
                    raise TypeError(f"{name}() got unexpected keyword argument '{key}'")
                if key in names[: len(args)]:
                    raise TypeError(f"{name}() got multiple values for argument '{key}'")
                assigned[key] = value

            # Check for missing arguments
            for key in names:
                if key not in assigned:
                    raise TypeError(f"{name}() missing required argument: '{key}'")

            # Set attributes
            for key, value in assigned.items():
                setattr(self, key, value)

        namespace["__init__"] = __init__

        return super().__new__(cls, name, bases, namespace)


def V_SCALAR_FROM(dtype, value):
    data = V(dtype=dtype, shape=())
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


@dataclasses.dataclass(eq=True, kw_only=False, frozen=True)
class RigidGlobalInfo:
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
        return RigidGlobalInfo(
            envs_offset=V_VEC(3, dtype=gs.qd_float, shape=(_B,)),
            gravity=V_VEC(3, dtype=gs.qd_float, shape=()),
            meaninertia=V(dtype=gs.qd_float, shape=()),
            n_awake_dofs=V(dtype=gs.qd_int, shape=(_B,)),
            n_awake_entities=V(dtype=gs.qd_int, shape=(_B,)),
            n_awake_links=V(dtype=gs.qd_int, shape=(_B,)),
            awake_dofs=V(dtype=gs.qd_int, shape=(solver.n_dofs_, _B)),
            awake_entities=V(dtype=gs.qd_int, shape=(solver.n_entities_, _B)),
            awake_links=V(dtype=gs.qd_int, shape=(solver.n_links_, _B)),
            qpos0=V(dtype=gs.qd_float, shape=(solver.n_qs_, _B)),
            qpos=V(dtype=gs.qd_float, shape=(solver.n_qs_, _B)),
            qpos_next=V(dtype=gs.qd_float, shape=(solver.n_qs_, _B)),
            links_T=V_MAT(n=4, m=4, dtype=gs.qd_float, shape=(solver.n_links_,)),
            geoms_init_AABB=V_VEC(3, dtype=gs.qd_float, shape=()),
            mass_mat=V(dtype=gs.qd_float, shape=()),
            mass_mat_L=V(dtype=gs.qd_float, shape=()),
            mass_mat_L_bw=V(dtype=gs.qd_float, shape=()),
            mass_mat_D_inv=V(dtype=gs.qd_float, shape=()),
            mass_mat_mask=V(dtype=gs.qd_bool, shape=()),
            mass_parent_mask=V(dtype=gs.qd_float, shape=()),
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

    return RigidGlobalInfo(
        envs_offset=V_VEC(3, dtype=gs.qd_float, shape=(_B,)),
        gravity=V_VEC(3, dtype=gs.qd_float, shape=(_B,)),
        meaninertia=V(dtype=gs.qd_float, shape=(_B,)),
        n_awake_dofs=V(dtype=gs.qd_int, shape=(_B,)),
        n_awake_entities=V(dtype=gs.qd_int, shape=(_B,)),
        n_awake_links=V(dtype=gs.qd_int, shape=(_B,)),
        awake_dofs=V(dtype=gs.qd_int, shape=(solver.n_dofs_, _B)),
        awake_entities=V(dtype=gs.qd_int, shape=(solver.n_entities_, _B)),
        awake_links=V(dtype=gs.qd_int, shape=(solver.n_links_, _B)),
        qpos0=V(dtype=gs.qd_float, shape=(solver.n_qs_, _B)),
        qpos=V(dtype=gs.qd_float, shape=(solver.n_qs_, _B), needs_grad=requires_grad),
        qpos_next=V(dtype=gs.qd_float, shape=(solver.n_qs_, _B), needs_grad=requires_grad),
        links_T=V_MAT(n=4, m=4, dtype=gs.qd_float, shape=(solver.n_links_,)),
        geoms_init_AABB=V_VEC(3, dtype=gs.qd_float, shape=(solver.n_geoms_, 8)),
        mass_mat=V(dtype=gs.qd_float, shape=mass_mat_shape, needs_grad=requires_grad),
        mass_mat_L=V(dtype=gs.qd_float, shape=mass_mat_shape, needs_grad=requires_grad),
        mass_mat_L_bw=V(dtype=gs.qd_float, shape=mass_mat_shape_bw, needs_grad=requires_grad),
        mass_mat_D_inv=V(dtype=gs.qd_float, shape=(solver.n_dofs_, _B), needs_grad=requires_grad),
        mass_mat_mask=V(dtype=gs.qd_bool, shape=(solver.n_entities_, _B)),
        mass_parent_mask=V(dtype=gs.qd_float, shape=(solver.n_dofs_, solver.n_dofs_)),
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


@dataclasses.dataclass(eq=True, kw_only=False, frozen=True)
class ConstraintState:
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
    return ConstraintState(
        n_constraints=V(dtype=gs.qd_int, shape=(_B,)),
        qd_n_equalities=V(dtype=gs.qd_int, shape=(_B,)),
        n_constraints_equality=V(dtype=gs.qd_int, shape=(_B,)),
        n_constraints_frictionloss=V(dtype=gs.qd_int, shape=(_B,)),
        is_warmstart=V(dtype=gs.qd_bool, shape=(_B,)),
        improved=V(dtype=gs.qd_bool, shape=(_B,)),
        cost_ws=V(dtype=gs.qd_float, shape=(_B,)),
        gauss=V(dtype=gs.qd_float, shape=(_B,)),
        cost=V(dtype=gs.qd_float, shape=(_B,)),
        prev_cost=V(dtype=gs.qd_float, shape=(_B,)),
        gtol=V(dtype=gs.qd_float, shape=(_B,)),
        ls_it=V(dtype=gs.qd_int, shape=(_B,)),
        ls_result=V(dtype=gs.qd_int, shape=(_B,)),
        cg_beta=V(dtype=gs.qd_float, shape=(_B,)),
        cg_pg_dot_pMg=V(dtype=gs.qd_float, shape=(_B,)),
        quad_gauss=V(dtype=gs.qd_float, shape=(3, _B)),
        ls_alpha=V(dtype=gs.qd_float, shape=(_B,)),
        ls_p0_cost=V(dtype=gs.qd_float, shape=(_B,)),
        ls_alpha_newton=V(dtype=gs.qd_float, shape=(_B,)),
        ls_gtol=V(dtype=gs.qd_float, shape=(_B,)),
        eq_sum=V(dtype=gs.qd_float, shape=(3, _B)),
        Ma=V(dtype=gs.qd_float, shape=(solver.n_dofs_, _B)),
        Ma_ws=V(dtype=gs.qd_float, shape=(solver.n_dofs_, _B)),
        grad=V(dtype=gs.qd_float, shape=(solver.n_dofs_, _B)),
        Mgrad=V(dtype=gs.qd_float, shape=(solver.n_dofs_, _B)),
        MinvJT=V(dtype=gs.qd_float, shape=maybe_shape(jac_shape, solver._options.noslip_iterations > 0)),
        search=V(dtype=gs.qd_float, shape=(solver.n_dofs_, _B)),
        qfrc_constraint=V(dtype=gs.qd_float, shape=(solver.n_dofs_, _B)),
        qacc=V(dtype=gs.qd_float, shape=(solver.n_dofs_, _B)),
        qacc_ws=V(dtype=gs.qd_float, shape=(solver.n_dofs_, _B)),
        qacc_prev=V(dtype=gs.qd_float, shape=(solver.n_dofs_, _B)),
        mv=V(dtype=gs.qd_float, shape=(solver.n_dofs_, _B)),
        cg_prev_grad=V(dtype=gs.qd_float, shape=(solver.n_dofs_, _B)),
        cg_prev_Mgrad=V(dtype=gs.qd_float, shape=(solver.n_dofs_, _B)),
        nt_vec=V(dtype=gs.qd_float, shape=(solver.n_dofs_, _B)),
        nt_H=V(dtype=gs.qd_float, shape=(_B, solver.n_dofs_, solver.n_dofs_)),
        incr_changed_idx=V(dtype=gs.qd_int, shape=(len_constraints_, _B)),
        incr_n_changed=V(dtype=gs.qd_int, shape=(_B,)),
        efc_b=V(dtype=gs.qd_float, shape=efc_b_shape),
        efc_AR=V(dtype=gs.qd_float, shape=efc_AR_shape),
        # Tier-1 constraint state: allocated as qd.Tensor wrappers
        # (Phase-1 migration; see perso_hugh/doc/genesis_tensor_migration.md).
        active=V(dtype=gs.qd_bool, shape=(len_constraints_, _B)),
        prev_active=V(dtype=gs.qd_bool, shape=(len_constraints_, _B)),
        diag=V(dtype=gs.qd_float, shape=(len_constraints_, _B)),
        aref=V(dtype=gs.qd_float, shape=(len_constraints_, _B)),
        Jaref=V(dtype=gs.qd_float, shape=(len_constraints_, _B)),
        efc_frictionloss=V(dtype=gs.qd_float, shape=(len_constraints_, _B)),
        efc_force=V(dtype=gs.qd_float, shape=(len_constraints_, _B)),
        efc_D=V(dtype=gs.qd_float, shape=(len_constraints_, _B)),
        jv=V(dtype=gs.qd_float, shape=(len_constraints_, _B)),
        jac=V(dtype=gs.qd_float, shape=jac_shape),
        jac_relevant_dofs=V(dtype=gs.qd_int, shape=jac_relevant_dofs_shape),
        jac_n_relevant_dofs=V(dtype=gs.qd_int, shape=jac_n_relevant_dofs_shape),
        # Backward gradients
        dL_dqacc=V(dtype=gs.qd_float, shape=maybe_shape((solver.n_dofs_, _B), solver._requires_grad)),
        dL_dM=V(dtype=gs.qd_float, shape=maybe_shape((solver.n_dofs_, solver.n_dofs_, _B), solver._requires_grad)),
        dL_djac=V(dtype=gs.qd_float, shape=maybe_shape((len_constraints_, solver.n_dofs_, _B), solver._requires_grad)),
        dL_daref=V(dtype=gs.qd_float, shape=maybe_shape((len_constraints_, _B), solver._requires_grad)),
        dL_defc_D=V(dtype=gs.qd_float, shape=maybe_shape((len_constraints_, _B), solver._requires_grad)),
        dL_dforce=V(dtype=gs.qd_float, shape=maybe_shape((solver.n_dofs_, _B), solver._requires_grad)),
        bw_u=V(dtype=gs.qd_float, shape=maybe_shape((solver.n_dofs_, _B), solver._requires_grad)),
        bw_r=V(dtype=gs.qd_float, shape=maybe_shape((solver.n_dofs_, _B), solver._requires_grad)),
        bw_p=V(dtype=gs.qd_float, shape=maybe_shape((solver.n_dofs_, _B), solver._requires_grad)),
        bw_Ap=V(dtype=gs.qd_float, shape=maybe_shape((solver.n_dofs_, _B), solver._requires_grad)),
        bw_Ju=V(dtype=gs.qd_float, shape=maybe_shape((len_constraints_, _B), solver._requires_grad)),
        bw_y=V(dtype=gs.qd_float, shape=maybe_shape((len_constraints_, _B), solver._requires_grad)),
        bw_w=V(dtype=gs.qd_float, shape=maybe_shape((len_constraints_, _B), solver._requires_grad)),
        # Timers
        timers=V(dtype=qd.i64 if gs.backend != gs.metal else qd.i32, shape=(10, _B)),
        use_full_hessian=V(dtype=qd.i32, shape=(_B,)),
        solver_iter_counter=V(dtype=qd.i32, shape=()),
        graph_counter=qd.ndarray(qd.i32, shape=()),
        early_exit_flag=V(dtype=qd.i32, shape=()),
    )


# =========================================== Collider ===========================================


@dataclasses.dataclass(eq=True, kw_only=False, frozen=True)
class ContactData:
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

    return ContactData(
        geom_a=V(dtype=gs.qd_int, shape=(max_contact_pairs_, _B)),
        geom_b=V(dtype=gs.qd_int, shape=(max_contact_pairs_, _B)),
        normal=V(dtype=gs.qd_vec3, shape=(max_contact_pairs_, _B), needs_grad=requires_grad),
        pos=V(dtype=gs.qd_vec3, shape=(max_contact_pairs_, _B), needs_grad=requires_grad),
        penetration=V(dtype=gs.qd_float, shape=(max_contact_pairs_, _B), needs_grad=requires_grad),
        friction=V(dtype=gs.qd_float, shape=(max_contact_pairs_, _B)),
        sol_params=V_VEC(7, dtype=gs.qd_float, shape=(max_contact_pairs_, _B)),
        force=V(dtype=gs.qd_vec3, shape=(max_contact_pairs_, _B)),
        link_a=V(dtype=gs.qd_int, shape=(max_contact_pairs_, _B)),
        link_b=V(dtype=gs.qd_int, shape=(max_contact_pairs_, _B)),
        pair_idx=V(dtype=gs.qd_int, shape=(max_contact_pairs_, _B)),
    )


@dataclasses.dataclass(eq=True, kw_only=False, frozen=True)
class DiffContactInput:
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
    return DiffContactInput(
        geom_a=V(dtype=gs.qd_int, shape=shape),
        geom_b=V(dtype=gs.qd_int, shape=shape),
        local_pos1_a=V_VEC(3, dtype=gs.qd_float, shape=shape),
        local_pos1_b=V_VEC(3, dtype=gs.qd_float, shape=shape),
        local_pos1_c=V_VEC(3, dtype=gs.qd_float, shape=shape),
        local_pos2_a=V_VEC(3, dtype=gs.qd_float, shape=shape),
        local_pos2_b=V_VEC(3, dtype=gs.qd_float, shape=shape),
        local_pos2_c=V_VEC(3, dtype=gs.qd_float, shape=shape),
        w_local_pos1=V_VEC(3, dtype=gs.qd_float, shape=shape),
        w_local_pos2=V_VEC(3, dtype=gs.qd_float, shape=shape),
        ref_id=V(dtype=gs.qd_int, shape=shape),
        valid=V(dtype=gs.qd_int, shape=shape),
        ref_penetration=V(dtype=gs.qd_float, shape=shape, needs_grad=True),
    )


@dataclasses.dataclass(eq=True, kw_only=False, frozen=True)
class SortBuffer:
    value: qd.Tensor
    i_g: qd.Tensor
    is_max: qd.Tensor


def get_sort_buffer(solver):
    _B = solver._B

    return SortBuffer(
        value=V(dtype=gs.qd_float, shape=(2 * solver.n_geoms_, _B)),
        i_g=V(dtype=gs.qd_int, shape=(2 * solver.n_geoms_, _B)),
        is_max=V(dtype=gs.qd_bool, shape=(2 * solver.n_geoms_, _B)),
    )


@dataclasses.dataclass(eq=True, kw_only=False, frozen=True)
class ContactCache:
    normal: qd.Tensor


def get_contact_cache(solver, n_possible_pairs):
    _B = solver._B
    return ContactCache(
        normal=V_VEC(3, dtype=gs.qd_float, shape=(n_possible_pairs, _B)),
    )


@dataclasses.dataclass(eq=True, kw_only=False, frozen=True)
class AggList:
    curr: qd.Tensor
    n: qd.Tensor
    start: qd.Tensor


def get_agg_list(solver):
    _B = solver._B
    n_entities = max(solver.n_entities, 1)

    return AggList(
        curr=V(dtype=gs.qd_int, shape=(n_entities, _B)),
        n=V(dtype=gs.qd_int, shape=(n_entities, _B)),
        start=V(dtype=gs.qd_int, shape=(n_entities, _B)),
    )


@dataclasses.dataclass(eq=True, kw_only=False, frozen=True)
class ContactIslandState:
    ci_edges: qd.Tensor
    edge_id: qd.Tensor
    constraint_list: qd.Tensor
    constraint_id: qd.Tensor
    entity_edge: AggList
    island_col: AggList
    island_hibernated: qd.Tensor
    island_entity: AggList
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

    return ContactIslandState(
        ci_edges=V(dtype=gs.qd_int, shape=(max_edges, 2, _B)),
        edge_id=V(dtype=gs.qd_int, shape=(max_edges * 2, _B)),
        constraint_list=V(dtype=gs.qd_int, shape=(max_contact_pairs, _B)),
        constraint_id=V(dtype=gs.qd_int, shape=(max_contact_pairs * 2, _B)),
        entity_edge=get_agg_list(solver),
        island_col=get_agg_list(solver),
        island_hibernated=V(dtype=gs.qd_int, shape=(n_entities, _B)),
        island_entity=get_agg_list(solver),
        entity_id=V(dtype=gs.qd_int, shape=(n_entities, _B)),
        n_edges=V(dtype=gs.qd_int, shape=(_B,)),
        n_islands=V(dtype=gs.qd_int, shape=(_B,)),
        n_stack=V(dtype=gs.qd_int, shape=(_B,)),
        entity_island=V(dtype=gs.qd_int, shape=(n_entities, _B)),
        stack=V(dtype=gs.qd_int, shape=(n_entities, _B)),
        entity_idx_to_next_entity_idx_in_hibernated_island=V(dtype=gs.qd_int, shape=(n_entities, _B)),
    )


@dataclasses.dataclass(eq=True, kw_only=False, frozen=True)
class NarrowphaseWorkQueues:
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
    return NarrowphaseWorkQueues(
        mpr_i_b=V(dtype=gs.qd_int, shape=(max_entries,)),
        mpr_i_ga=V(dtype=gs.qd_int, shape=(max_entries,)),
        mpr_i_gb=V(dtype=gs.qd_int, shape=(max_entries,)),
        mpr_i_pair=V(dtype=gs.qd_int, shape=(max_entries,)),
        mpr_contact_pos_0=V_VEC(3, dtype=gs.qd_float, shape=(max_entries,)),
        mpr_normal_0=V_VEC(3, dtype=gs.qd_float, shape=(max_entries,)),
        mpr_penetration_0=V(dtype=gs.qd_float, shape=(max_entries,)),
        gjk_i_b=V(dtype=gs.qd_int, shape=(max_entries,)),
        gjk_i_ga=V(dtype=gs.qd_int, shape=(max_entries,)),
        gjk_i_gb=V(dtype=gs.qd_int, shape=(max_entries,)),
        gjk_i_pair=V(dtype=gs.qd_int, shape=(max_entries,)),
        gjk_contact_pos_0=V_VEC(3, dtype=gs.qd_float, shape=(max_entries,)),
        gjk_normal_0=V_VEC(3, dtype=gs.qd_float, shape=(max_entries,)),
        gjk_penetration_0=V(dtype=gs.qd_float, shape=(max_entries,)),
        mpr_queue_size=V(dtype=gs.qd_int, shape=(1,)),
        gjk_queue_size=V(dtype=gs.qd_int, shape=(1,)),
        gjk_queue_size_k2=V(dtype=gs.qd_int, shape=(1,)),
        mpr_work_counter=V(dtype=gs.qd_int, shape=(1,)),
        gjk_work_counter=V(dtype=gs.qd_int, shape=(1,)),
    )


@dataclasses.dataclass(eq=True, kw_only=False, frozen=True)
class ColliderState:
    sort_buffer: SortBuffer
    contact_data: ContactData
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
    contact_cache: ContactCache
    # Input data for differentiable contact detection used in the backward pass
    diff_contact_input: DiffContactInput
    narrowphase_work_queues: NarrowphaseWorkQueues
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

    return ColliderState(
        sort_buffer=get_sort_buffer(solver),
        active_buffer=V(dtype=gs.qd_int, shape=(n_geoms, _B)),
        n_broad_pairs=V(dtype=gs.qd_int, shape=(_B,)),
        active_buffer_awake=V(dtype=gs.qd_int, shape=(n_geoms, _B)),
        active_buffer_hib=V(dtype=gs.qd_int, shape=(n_geoms, _B)),
        box_depth=V(dtype=gs.qd_float, shape=box_depth_shape),
        box_points=V_VEC(3, dtype=gs.qd_float, shape=box_points_shape),
        box_pts=V_VEC(3, dtype=gs.qd_float, shape=box_pts_shape),
        box_lines=V_VEC(6, dtype=gs.qd_float, shape=box_lines_shape),
        box_linesu=V_VEC(6, dtype=gs.qd_float, shape=box_linesu_shape),
        box_axi=V_VEC(3, dtype=gs.qd_float, shape=box_axi_shape),
        box_ppts2=V(dtype=gs.qd_float, shape=box_ppts2_shape),
        box_pu=V_VEC(3, dtype=gs.qd_float, shape=box_pu_shape),
        xyz_max_min=V(dtype=gs.qd_float, shape=(6, _B)),
        prism=V_VEC(3, dtype=gs.qd_float, shape=(6, _B)),
        n_contacts=V(dtype=gs.qd_int, shape=(_B,)),
        n_contacts_hibernated=V(dtype=gs.qd_int, shape=(_B,)),
        first_time=V(dtype=gs.qd_bool, shape=(_B,)),
        contact_cache=get_contact_cache(solver, n_possible_pairs),
        broad_collision_pairs=V_VEC(2, dtype=gs.qd_int, shape=(max(max_collision_pairs_broad, 1), _B)),
        contact_data=get_contact_data(solver, max_contact_pairs, requires_grad),
        diff_contact_input=get_diff_contact_input(_B, max(max_contact_pairs, 1), True, requires_grad),
        narrowphase_work_queues=get_narrowphase_work_queues(
            max(max_collision_pairs_broad * _B, 1) if collider_static_config.has_non_box_plane_convex_convex else 1
        ),
        contact_sort_key=V(dtype=gs.qd_float, shape=(max(max_contact_pairs, 1), _B)),
        contact_sort_idx=V(dtype=gs.qd_int, shape=(max(max_contact_pairs, 1), _B)),
    )


@dataclasses.dataclass(eq=True, kw_only=False, frozen=True)
class ColliderInfo:
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

    return ColliderInfo(
        vert_neighbors=V(dtype=gs.qd_int, shape=(max(n_vert_neighbors, 1),)),
        vert_neighbor_start=V(dtype=gs.qd_int, shape=(solver.n_verts_,)),
        vert_n_neighbors=V(dtype=gs.qd_int, shape=(solver.n_verts_,)),
        collision_pair_idx=V(dtype=gs.qd_int, shape=(solver.n_geoms_, solver.n_geoms_)),
        max_possible_pairs=V(dtype=gs.qd_int, shape=()),
        max_collision_pairs=V(dtype=gs.qd_int, shape=()),
        max_contact_pairs=V(dtype=gs.qd_int, shape=()),
        max_collision_pairs_broad=V(dtype=gs.qd_int, shape=()),
        n_valid_pairs=V_SCALAR_FROM(dtype=gs.qd_int, value=n_valid_pairs),
        valid_collision_pairs=V(dtype=gs.qd_ivec2, shape=(max(n_valid_pairs, 1),)),
        terrain_hf=V(dtype=gs.qd_float, shape=terrain_hf_shape),
        terrain_rc=V(dtype=gs.qd_int, shape=(2,)),
        terrain_scale=V(dtype=gs.qd_float, shape=(2,)),
        terrain_xyz_maxmin=V(dtype=gs.qd_float, shape=(6,)),
        mc_perturbation=V_SCALAR_FROM(dtype=gs.qd_float, value=kwargs["mc_perturbation"]),
        mc_tolerance=V_SCALAR_FROM(dtype=gs.qd_float, value=kwargs["mc_tolerance"]),
        mpr_to_gjk_overlap_ratio=V_SCALAR_FROM(dtype=gs.qd_float, value=kwargs["mpr_to_gjk_overlap_ratio"]),
        diff_pos_tolerance=V_SCALAR_FROM(dtype=gs.qd_float, value=kwargs["diff_pos_tolerance"]),
        diff_normal_tolerance=V_SCALAR_FROM(dtype=gs.qd_float, value=kwargs["diff_normal_tolerance"]),
    )


@qd.data_oriented
class ColliderStaticConfig(metaclass=AutoInitMeta):
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


@dataclasses.dataclass(eq=True, kw_only=False, frozen=True)
class MPRSimplexSupport:
    v1: qd.Tensor
    v2: qd.Tensor
    v: qd.Tensor


def get_mpr_simplex_support(B_):
    return MPRSimplexSupport(
        v1=V_VEC(3, dtype=gs.qd_float, shape=(4, B_)),
        v2=V_VEC(3, dtype=gs.qd_float, shape=(4, B_)),
        v=V_VEC(3, dtype=gs.qd_float, shape=(4, B_)),
    )


@dataclasses.dataclass(eq=True, kw_only=False, frozen=True)
class MPRState:
    simplex_support: MPRSimplexSupport
    simplex_size: qd.Tensor


def get_mpr_state(B_):
    return MPRState(
        simplex_support=get_mpr_simplex_support(B_),
        simplex_size=V(dtype=gs.qd_int, shape=(B_,)),
    )


@dataclasses.dataclass(eq=True, kw_only=False, frozen=True)
class MPRInfo:
    CCD_EPS: qd.Tensor
    CCD_TOLERANCE: qd.Tensor
    CCD_ITERATIONS: qd.Tensor


def get_mpr_info(**kwargs):
    return MPRInfo(
        CCD_EPS=V_SCALAR_FROM(dtype=gs.qd_float, value=kwargs["CCD_EPS"]),
        CCD_TOLERANCE=V_SCALAR_FROM(dtype=gs.qd_float, value=kwargs["CCD_TOLERANCE"]),
        CCD_ITERATIONS=V_SCALAR_FROM(dtype=gs.qd_float, value=kwargs["CCD_ITERATIONS"]),
    )


# =========================================== GJK ===========================================


@dataclasses.dataclass(eq=True, kw_only=False, frozen=True)
class MDVertex:
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
    return MDVertex(
        obj1=V_VEC(3, dtype=gs.qd_float, shape=shape),
        obj2=V_VEC(3, dtype=gs.qd_float, shape=shape),
        local_obj1=V_VEC(3, dtype=gs.qd_float, shape=shape),
        local_obj2=V_VEC(3, dtype=gs.qd_float, shape=shape),
        id1=V(dtype=gs.qd_int, shape=shape),
        id2=V(dtype=gs.qd_int, shape=shape),
        mink=V_VEC(3, dtype=gs.qd_float, shape=shape),
    )


def get_epa_polytope_vertex(_B, gjk_info, is_active):
    max_num_polytope_verts = 5 + gjk_info.epa_max_iterations[None]
    shape = maybe_shape((_B, max_num_polytope_verts), is_active)
    return MDVertex(
        obj1=V_VEC(3, dtype=gs.qd_float, shape=shape),
        obj2=V_VEC(3, dtype=gs.qd_float, shape=shape),
        local_obj1=V_VEC(3, dtype=gs.qd_float, shape=shape),
        local_obj2=V_VEC(3, dtype=gs.qd_float, shape=shape),
        id1=V(dtype=gs.qd_int, shape=shape),
        id2=V(dtype=gs.qd_int, shape=shape),
        mink=V_VEC(3, dtype=gs.qd_float, shape=shape),
    )


@dataclasses.dataclass(eq=True, kw_only=False, frozen=True)
class GJKSimplex:
    nverts: qd.Tensor
    dist: qd.Tensor


def get_gjk_simplex(_B, is_active):
    shape = maybe_shape((_B,), is_active)
    return GJKSimplex(
        nverts=V(dtype=gs.qd_int, shape=shape),
        dist=V(dtype=gs.qd_float, shape=shape),
    )


@dataclasses.dataclass(eq=True, kw_only=False, frozen=True)
class GJKSimplexBuffer:
    normal: qd.Tensor
    sdist: qd.Tensor


def get_gjk_simplex_buffer(_B, is_active):
    shape = maybe_shape((_B, 4), is_active)
    return GJKSimplexBuffer(
        normal=V_VEC(3, dtype=gs.qd_float, shape=shape),
        sdist=V(dtype=gs.qd_float, shape=shape),
    )


@dataclasses.dataclass(eq=True, kw_only=False, frozen=True)
class EPAPolytope:
    nverts: qd.Tensor
    nfaces: qd.Tensor
    nfaces_map: qd.Tensor
    horizon_nedges: qd.Tensor
    horizon_w: qd.Tensor


def get_epa_polytope(_B, is_active):
    shape = maybe_shape((_B,), is_active)
    return EPAPolytope(
        nverts=V(dtype=gs.qd_int, shape=shape),
        nfaces=V(dtype=gs.qd_int, shape=shape),
        nfaces_map=V(dtype=gs.qd_int, shape=shape),
        horizon_nedges=V(dtype=gs.qd_int, shape=shape),
        horizon_w=V_VEC(3, dtype=gs.qd_float, shape=shape),
    )


@dataclasses.dataclass(eq=True, kw_only=False, frozen=True)
class EPAPolytopeFace:
    verts_idx: qd.Tensor
    adj_idx: qd.Tensor
    normal: qd.Tensor
    dist2: qd.Tensor
    map_idx: qd.Tensor
    visited: qd.Tensor


def get_epa_polytope_face(_B, polytope_max_faces, is_active):
    shape = maybe_shape((_B, polytope_max_faces), is_active)
    return EPAPolytopeFace(
        verts_idx=V_VEC(3, dtype=gs.qd_int, shape=shape),
        adj_idx=V_VEC(3, dtype=gs.qd_int, shape=shape),
        normal=V_VEC(3, dtype=gs.qd_float, shape=shape),
        dist2=V(dtype=gs.qd_float, shape=shape),
        map_idx=V(dtype=gs.qd_int, shape=shape),
        visited=V(dtype=gs.qd_int, shape=shape),
    )


@dataclasses.dataclass(eq=True, kw_only=False, frozen=True)
class EPAPolytopeHorizonData:
    face_idx: qd.Tensor
    edge_idx: qd.Tensor


def get_epa_polytope_horizon_data(_B, polytope_max_horizons, is_active):
    shape = maybe_shape((_B, polytope_max_horizons), is_active)
    return EPAPolytopeHorizonData(
        face_idx=V(dtype=gs.qd_int, shape=shape),
        edge_idx=V(dtype=gs.qd_int, shape=shape),
    )


@dataclasses.dataclass(eq=True, kw_only=False, frozen=True)
class ContactFace:
    vert1: qd.Tensor
    vert2: qd.Tensor
    endverts: qd.Tensor
    normal1: qd.Tensor
    normal2: qd.Tensor
    id1: qd.Tensor
    id2: qd.Tensor


def get_contact_face(_B, max_contact_polygon_verts, is_active):
    shape = maybe_shape((_B, max_contact_polygon_verts), is_active)
    return ContactFace(
        vert1=V_VEC(3, dtype=gs.qd_float, shape=shape),
        vert2=V_VEC(3, dtype=gs.qd_float, shape=shape),
        endverts=V_VEC(3, dtype=gs.qd_float, shape=shape),
        normal1=V_VEC(3, dtype=gs.qd_float, shape=shape),
        normal2=V_VEC(3, dtype=gs.qd_float, shape=shape),
        id1=V(dtype=gs.qd_int, shape=shape),
        id2=V(dtype=gs.qd_int, shape=shape),
    )


@dataclasses.dataclass(eq=True, kw_only=False, frozen=True)
class ContactNormal:
    endverts: qd.Tensor
    normal: qd.Tensor
    id: qd.Tensor


def get_contact_normal(_B, max_contact_polygon_verts, is_active):
    shape = maybe_shape((_B, max_contact_polygon_verts), is_active)
    return ContactNormal(
        endverts=V_VEC(3, dtype=gs.qd_float, shape=shape),
        normal=V_VEC(3, dtype=gs.qd_float, shape=shape),
        id=V(dtype=gs.qd_int, shape=shape),
    )


@dataclasses.dataclass(eq=True, kw_only=False, frozen=True)
class ContactHalfspace:
    normal: qd.Tensor
    dist: qd.Tensor


def get_contact_halfspace(_B, max_contact_polygon_verts, is_active):
    shape = maybe_shape((_B, max_contact_polygon_verts), is_active)
    return ContactHalfspace(
        normal=V_VEC(3, dtype=gs.qd_float, shape=shape),
        dist=V(dtype=gs.qd_float, shape=shape),
    )


@dataclasses.dataclass(eq=True, kw_only=False, frozen=True)
class Witness:
    point_obj1: qd.Tensor
    point_obj2: qd.Tensor


def get_witness(_B, max_contacts_per_pair, is_active):
    shape = maybe_shape((_B, max_contacts_per_pair), is_active)
    return Witness(
        point_obj1=V_VEC(3, dtype=gs.qd_float, shape=shape),
        point_obj2=V_VEC(3, dtype=gs.qd_float, shape=shape),
    )


@dataclasses.dataclass(eq=True, kw_only=False, frozen=True)
class GJKState:
    support_mesh_prev_vertex_id: qd.Tensor
    simplex_vertex: MDVertex
    simplex_buffer: GJKSimplexBuffer
    simplex: GJKSimplex
    simplex_vertex_intersect: MDVertex
    simplex_buffer_intersect: GJKSimplexBuffer
    nsimplex: qd.Tensor
    last_searched_simplex_vertex_id: qd.Tensor
    polytope: EPAPolytope
    polytope_verts: MDVertex
    polytope_faces: EPAPolytopeFace
    polytope_faces_map: qd.Tensor
    polytope_horizon_data: EPAPolytopeHorizonData
    polytope_horizon_stack: EPAPolytopeHorizonData
    contact_faces: ContactFace
    contact_normals: ContactNormal
    contact_halfspaces: ContactHalfspace
    contact_clipped_polygons: qd.Tensor
    multi_contact_flag: qd.Tensor
    witness: Witness
    n_witness: qd.Tensor
    n_contacts: qd.Tensor
    contact_pos: qd.Tensor
    normal: qd.Tensor
    is_col: qd.Tensor
    penetration: qd.Tensor
    distance: qd.Tensor
    # Differentiable contact detection
    diff_contact_input: DiffContactInput
    n_diff_contact_input: qd.Tensor
    diff_penetration: qd.Tensor


def get_gjk_state(_B, static_rigid_sim_config, gjk_info, is_active, requires_grad=False):
    enable_mujoco_compatibility = static_rigid_sim_config.enable_mujoco_compatibility
    polytope_max_faces = gjk_info.polytope_max_faces[None]
    max_contacts_per_pair = gjk_info.max_contacts_per_pair[None]
    max_contact_polygon_verts = gjk_info.max_contact_polygon_verts[None]

    # FIXME: Define GJKState and MujocoCompatGJKState that derives from the former but defines additional attributes
    return GJKState(
        # GJK simplex
        support_mesh_prev_vertex_id=V(dtype=gs.qd_int, shape=(_B, 2)),
        simplex_vertex=get_gjk_simplex_vertex(_B, is_active),
        simplex_buffer=get_gjk_simplex_buffer(_B, is_active),
        simplex=get_gjk_simplex(_B, is_active),
        last_searched_simplex_vertex_id=V(dtype=gs.qd_int, shape=(_B,)),
        simplex_vertex_intersect=get_gjk_simplex_vertex(_B, is_active),
        simplex_buffer_intersect=get_gjk_simplex_buffer(_B, is_active),
        nsimplex=V(dtype=gs.qd_int, shape=(_B,)),
        # EPA polytope
        polytope=get_epa_polytope(_B, is_active),
        polytope_verts=get_epa_polytope_vertex(_B, gjk_info, is_active),
        polytope_faces=get_epa_polytope_face(_B, polytope_max_faces, is_active),
        polytope_faces_map=V(dtype=gs.qd_int, shape=(_B, polytope_max_faces)),
        polytope_horizon_data=get_epa_polytope_horizon_data(_B, 6 + gjk_info.epa_max_iterations[None], is_active),
        polytope_horizon_stack=get_epa_polytope_horizon_data(_B, polytope_max_faces * 3, is_active),
        # Multi-contact detection (MuJoCo compatibility)
        contact_faces=get_contact_face(_B, max_contact_polygon_verts, is_active),
        contact_normals=get_contact_normal(_B, max_contact_polygon_verts, is_active),
        contact_halfspaces=get_contact_halfspace(_B, max_contact_polygon_verts, is_active),
        contact_clipped_polygons=V_VEC(3, dtype=gs.qd_float, shape=(_B, 2, max_contact_polygon_verts)),
        multi_contact_flag=V(dtype=gs.qd_bool, shape=(_B,)),
        # Final results
        witness=get_witness(_B, max_contacts_per_pair, is_active),
        n_witness=V(dtype=gs.qd_int, shape=(_B,)),
        n_contacts=V(dtype=gs.qd_int, shape=(_B,)),
        contact_pos=V_VEC(3, dtype=gs.qd_float, shape=(_B, max_contacts_per_pair)),
        normal=V_VEC(3, dtype=gs.qd_float, shape=(_B, max_contacts_per_pair)),
        is_col=V(dtype=gs.qd_bool, shape=(_B,)),
        penetration=V(dtype=gs.qd_float, shape=(_B,)),
        distance=V(dtype=gs.qd_float, shape=(_B,)),
        diff_contact_input=get_diff_contact_input(_B, max(max_contacts_per_pair, 1), is_active, requires_grad),
        n_diff_contact_input=V(dtype=gs.qd_int, shape=(_B,)),
        diff_penetration=V(dtype=gs.qd_float, shape=maybe_shape((_B, max_contacts_per_pair), requires_grad)),
    )


def get_gjk_state_contact_only(_B):
    """Minimal GJK state for contact detection only (no EPA, no multi-contact).

    Used by kernel 1 to run func_gjk as a boolean overlap test. All EPA polytope,
    multi-contact, and differentiable fields are allocated at dummy size (1,) since
    func_gjk never accesses them.
    """
    _dummy_B = 1

    return GJKState(
        support_mesh_prev_vertex_id=V(dtype=gs.qd_int, shape=(_B, 2)),
        simplex_vertex=get_gjk_simplex_vertex(_B, is_active=True),
        simplex_buffer=get_gjk_simplex_buffer(_B, is_active=True),
        simplex=get_gjk_simplex(_B, is_active=True),
        last_searched_simplex_vertex_id=V(dtype=gs.qd_int, shape=(_B,)),
        simplex_vertex_intersect=get_gjk_simplex_vertex(_B, is_active=True),
        simplex_buffer_intersect=get_gjk_simplex_buffer(_B, is_active=True),
        nsimplex=V(dtype=gs.qd_int, shape=(_B,)),
        # EPA — dummy allocations, never accessed by func_gjk
        polytope=get_epa_polytope(_dummy_B, is_active=True),
        polytope_verts=MDVertex(
            obj1=V_VEC(3, dtype=gs.qd_float, shape=(1, 1)),
            obj2=V_VEC(3, dtype=gs.qd_float, shape=(1, 1)),
            local_obj1=V_VEC(3, dtype=gs.qd_float, shape=(1, 1)),
            local_obj2=V_VEC(3, dtype=gs.qd_float, shape=(1, 1)),
            id1=V(dtype=gs.qd_int, shape=(1, 1)),
            id2=V(dtype=gs.qd_int, shape=(1, 1)),
            mink=V_VEC(3, dtype=gs.qd_float, shape=(1, 1)),
        ),
        polytope_faces=get_epa_polytope_face(_dummy_B, 1, is_active=True),
        polytope_faces_map=V(dtype=gs.qd_int, shape=(1, 1)),
        polytope_horizon_data=get_epa_polytope_horizon_data(_dummy_B, 1, is_active=True),
        polytope_horizon_stack=get_epa_polytope_horizon_data(_dummy_B, 1, is_active=True),
        # Multi-contact — dummy
        contact_faces=get_contact_face(_dummy_B, 1, is_active=True),
        contact_normals=get_contact_normal(_dummy_B, 1, is_active=True),
        contact_halfspaces=get_contact_halfspace(_dummy_B, 1, is_active=True),
        contact_clipped_polygons=V_VEC(3, dtype=gs.qd_float, shape=(1, 2, 1)),
        multi_contact_flag=V(dtype=gs.qd_bool, shape=(_B,)),
        # Results — full _B for fields func_gjk writes; dummy for EPA-only fields
        witness=get_witness(_B, 1, is_active=True),
        n_witness=V(dtype=gs.qd_int, shape=(_B,)),
        n_contacts=V(dtype=gs.qd_int, shape=(1,)),
        contact_pos=V_VEC(3, dtype=gs.qd_float, shape=(1, 1)),
        normal=V_VEC(3, dtype=gs.qd_float, shape=(1, 1)),
        is_col=V(dtype=gs.qd_bool, shape=(1,)),
        penetration=V(dtype=gs.qd_float, shape=(1,)),
        distance=V(dtype=gs.qd_float, shape=(_B,)),
        diff_contact_input=get_diff_contact_input(_dummy_B, 1, is_active=False),
        n_diff_contact_input=V(dtype=gs.qd_int, shape=(1,)),
        diff_penetration=V(dtype=gs.qd_float, shape=()),
    )


@dataclasses.dataclass(eq=True, kw_only=False, frozen=True)
class GJKInfo:
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
    return GJKInfo(
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
class GJKStaticConfig(metaclass=AutoInitMeta):
    # This is disabled by default, because it is often less stable than the other multi-contact detection algorithm.
    # However, we keep the code here for compatibility with MuJoCo and for possible future use.
    enable_mujoco_multi_contact: bool


# =========================================== SupportField ===========================================


@dataclasses.dataclass(eq=True, kw_only=False, frozen=True)
class SupportFieldInfo:
    support_cell_start: qd.Tensor
    support_v: qd.Tensor
    support_vid: qd.Tensor
    support_res: qd.Tensor


def get_support_field_info(n_geoms, n_support_cells, support_res):
    return SupportFieldInfo(
        support_cell_start=V(dtype=gs.qd_int, shape=(max(n_geoms, 1),)),
        support_v=V_VEC(3, dtype=gs.qd_float, shape=(max(n_support_cells, 1),)),
        support_vid=V(dtype=gs.qd_int, shape=(max(n_support_cells, 1),)),
        support_res=V_SCALAR_FROM(dtype=gs.qd_int, value=support_res),
    )


# =========================================== SDF ===========================================


@dataclasses.dataclass(eq=True, kw_only=False, frozen=True)
class SDFGeomInfo:
    T_mesh_to_sdf: qd.Tensor
    sdf_res: qd.Tensor
    sdf_max: qd.Tensor
    sdf_cell_size: qd.Tensor
    sdf_cell_start: qd.Tensor


def get_sdf_geom_info(n_geoms):
    return SDFGeomInfo(
        T_mesh_to_sdf=V_MAT(n=4, m=4, dtype=gs.qd_float, shape=(n_geoms,)),
        sdf_res=V_VEC(3, dtype=gs.qd_int, shape=(n_geoms,)),
        sdf_max=V(dtype=gs.qd_float, shape=(n_geoms,)),
        sdf_cell_size=V(dtype=gs.qd_float, shape=(n_geoms,)),
        sdf_cell_start=V(dtype=gs.qd_int, shape=(n_geoms,)),
    )


@dataclasses.dataclass(eq=True, kw_only=False, frozen=True)
class SDFInfo:
    geoms_info: SDFGeomInfo
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

    return SDFInfo(
        geoms_info=get_sdf_geom_info(max(n_geoms, 1)),
        geoms_sdf_start=V(dtype=gs.qd_int, shape=(max(n_geoms, 1),)),
        geoms_sdf_val=V(dtype=gs.qd_float, shape=(max(n_cells, 1),)),
        geoms_sdf_grad=V_VEC(3, dtype=gs.qd_float, shape=(max(n_cells, 1),)),
        geoms_sdf_closest_vert=V(dtype=gs.qd_int, shape=(max(n_cells, 1),)),
    )


# =========================================== DofsInfo and DofsState ===========================================


@dataclasses.dataclass(eq=True, kw_only=False, frozen=True)
class DofsInfo:
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

    return DofsInfo(
        entity_idx=V(dtype=gs.qd_int, shape=shape),
        stiffness=V(dtype=gs.qd_float, shape=shape),
        invweight=V(dtype=gs.qd_float, shape=shape),
        armature=V(dtype=gs.qd_float, shape=shape),
        damping=V(dtype=gs.qd_float, shape=shape),
        frictionloss=V(dtype=gs.qd_float, shape=shape),
        motion_ang=V(dtype=gs.qd_vec3, shape=shape),
        motion_vel=V(dtype=gs.qd_vec3, shape=shape),
        limit=V(dtype=gs.qd_vec2, shape=shape),
        act_gain=V(dtype=gs.qd_float, shape=shape),
        act_bias=V(dtype=gs.qd_vec3, shape=shape),
        force_range=V(dtype=gs.qd_vec2, shape=shape),
    )


@dataclasses.dataclass(eq=True, kw_only=False, frozen=True)
class DofsState:
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

    return DofsState(
        force=V(dtype=gs.qd_float, shape=shape, needs_grad=requires_grad),
        qf_bias=V(dtype=gs.qd_float, shape=shape, needs_grad=requires_grad),
        qf_passive=V(dtype=gs.qd_float, shape=shape, needs_grad=requires_grad),
        qf_actuator=V(dtype=gs.qd_float, shape=shape, needs_grad=requires_grad),
        qf_applied=V(dtype=gs.qd_float, shape=shape, needs_grad=requires_grad),
        act_length=V(dtype=gs.qd_float, shape=shape, needs_grad=requires_grad),
        pos=V(dtype=gs.qd_float, shape=shape, needs_grad=requires_grad),
        vel=V(dtype=gs.qd_float, shape=shape, needs_grad=requires_grad),
        vel_prev=V(dtype=gs.qd_float, shape=shape, needs_grad=requires_grad),
        vel_next=V(dtype=gs.qd_float, shape=shape, needs_grad=requires_grad),
        acc=V(dtype=gs.qd_float, shape=shape, needs_grad=requires_grad),
        acc_bw=V(dtype=gs.qd_float, shape=shape_bw, needs_grad=requires_grad),
        acc_smooth=V(dtype=gs.qd_float, shape=shape, needs_grad=requires_grad),
        acc_smooth_bw=V(dtype=gs.qd_float, shape=shape_bw, needs_grad=requires_grad),
        qf_smooth=V(dtype=gs.qd_float, shape=shape, needs_grad=requires_grad),
        qf_constraint=V(dtype=gs.qd_float, shape=shape, needs_grad=requires_grad),
        cdof_ang=V(dtype=gs.qd_vec3, shape=shape, needs_grad=requires_grad),
        cdof_vel=V(dtype=gs.qd_vec3, shape=shape, needs_grad=requires_grad),
        cdofvel_ang=V(dtype=gs.qd_vec3, shape=shape, needs_grad=requires_grad),
        cdofvel_vel=V(dtype=gs.qd_vec3, shape=shape, needs_grad=requires_grad),
        cdofd_ang=V(dtype=gs.qd_vec3, shape=shape, needs_grad=requires_grad),
        cdofd_vel=V(dtype=gs.qd_vec3, shape=shape, needs_grad=requires_grad),
        f_vel=V(dtype=gs.qd_vec3, shape=shape, needs_grad=requires_grad),
        f_ang=V(dtype=gs.qd_vec3, shape=shape, needs_grad=requires_grad),
        ctrl_force=V(dtype=gs.qd_float, shape=shape, needs_grad=requires_grad),
        ctrl_pos=V(dtype=gs.qd_float, shape=shape, needs_grad=requires_grad),
        ctrl_vel=V(dtype=gs.qd_float, shape=shape, needs_grad=requires_grad),
        ctrl_mode=V(dtype=gs.qd_int, shape=shape),
        hibernated=V(dtype=gs.qd_int, shape=shape),
    )


# =========================================== LinksState and LinksInfo ===========================================


@dataclasses.dataclass(eq=True, kw_only=False, frozen=True)
class LinksState:
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

    return LinksState(
        cinr_inertial=V(dtype=gs.qd_mat3, shape=shape, needs_grad=requires_grad),
        cinr_pos=V(dtype=gs.qd_vec3, shape=shape, needs_grad=requires_grad),
        cinr_quat=V(dtype=gs.qd_vec4, shape=shape, needs_grad=requires_grad),
        cinr_mass=V(dtype=gs.qd_float, shape=shape, needs_grad=requires_grad),
        crb_inertial=V(dtype=gs.qd_mat3, shape=shape, needs_grad=requires_grad),
        crb_pos=V(dtype=gs.qd_vec3, shape=shape, needs_grad=requires_grad),
        crb_quat=V(dtype=gs.qd_vec4, shape=shape, needs_grad=requires_grad),
        crb_mass=V(dtype=gs.qd_float, shape=shape, needs_grad=requires_grad),
        cdd_vel=V(dtype=gs.qd_vec3, shape=shape, needs_grad=requires_grad),
        cdd_ang=V(dtype=gs.qd_vec3, shape=shape, needs_grad=requires_grad),
        pos=V(dtype=gs.qd_vec3, shape=shape, needs_grad=requires_grad),
        quat=V(dtype=gs.qd_vec4, shape=shape, needs_grad=requires_grad),
        pos_bw=V(dtype=gs.qd_vec3, shape=shape_bw, needs_grad=requires_grad),
        quat_bw=V(dtype=gs.qd_vec4, shape=shape_bw, needs_grad=requires_grad),
        i_pos=V(dtype=gs.qd_vec3, shape=shape, needs_grad=requires_grad),
        i_pos_bw=V(dtype=gs.qd_vec3, shape=shape, needs_grad=requires_grad),
        i_quat=V(dtype=gs.qd_vec4, shape=shape, needs_grad=requires_grad),
        j_pos=V(dtype=gs.qd_vec3, shape=shape, needs_grad=requires_grad),
        j_quat=V(dtype=gs.qd_vec4, shape=shape, needs_grad=requires_grad),
        j_pos_bw=V(dtype=gs.qd_vec3, shape=shape_bw, needs_grad=requires_grad),
        j_quat_bw=V(dtype=gs.qd_vec4, shape=shape_bw, needs_grad=requires_grad),
        j_vel=V(dtype=gs.qd_vec3, shape=shape, needs_grad=requires_grad),
        j_ang=V(dtype=gs.qd_vec3, shape=shape, needs_grad=requires_grad),
        cd_ang=V(dtype=gs.qd_vec3, shape=shape, needs_grad=requires_grad),
        cd_vel=V(dtype=gs.qd_vec3, shape=shape, needs_grad=requires_grad),
        cd_ang_bw=V(dtype=gs.qd_vec3, shape=shape_bw, needs_grad=requires_grad),
        cd_vel_bw=V(dtype=gs.qd_vec3, shape=shape_bw, needs_grad=requires_grad),
        mass_sum=V(dtype=gs.qd_float, shape=shape, needs_grad=requires_grad),
        root_COM=V(dtype=gs.qd_vec3, shape=shape, needs_grad=requires_grad),
        root_COM_bw=V(dtype=gs.qd_vec3, shape=shape, needs_grad=requires_grad),
        mass_shift=V(dtype=gs.qd_float, shape=shape, needs_grad=requires_grad),
        i_pos_shift=V(dtype=gs.qd_vec3, shape=shape, needs_grad=requires_grad),
        cacc_ang=V(dtype=gs.qd_vec3, shape=shape, needs_grad=requires_grad),
        cacc_lin=V(dtype=gs.qd_vec3, shape=shape, needs_grad=requires_grad),
        cfrc_ang=V(dtype=gs.qd_vec3, shape=shape, needs_grad=requires_grad),
        cfrc_vel=V(dtype=gs.qd_vec3, shape=shape, needs_grad=requires_grad),
        cfrc_applied_ang=V(dtype=gs.qd_vec3, shape=shape, needs_grad=requires_grad),
        cfrc_applied_vel=V(dtype=gs.qd_vec3, shape=shape, needs_grad=requires_grad),
        cfrc_coupling_ang=V(dtype=gs.qd_vec3, shape=shape, needs_grad=requires_grad),
        cfrc_coupling_vel=V(dtype=gs.qd_vec3, shape=shape, needs_grad=requires_grad),
        contact_force=V(dtype=gs.qd_vec3, shape=shape, needs_grad=requires_grad),
        hibernated=V(dtype=gs.qd_int, shape=shape),
    )


@dataclasses.dataclass(eq=True, kw_only=False, frozen=True)
class LinksInfo:
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

    return LinksInfo(
        parent_idx=V(dtype=gs.qd_int, shape=links_info_shape),
        root_idx=V(dtype=gs.qd_int, shape=links_info_shape),
        q_start=V(dtype=gs.qd_int, shape=links_info_shape),
        dof_start=V(dtype=gs.qd_int, shape=links_info_shape),
        joint_start=V(dtype=gs.qd_int, shape=links_info_shape),
        q_end=V(dtype=gs.qd_int, shape=links_info_shape),
        dof_end=V(dtype=gs.qd_int, shape=links_info_shape),
        joint_end=V(dtype=gs.qd_int, shape=links_info_shape),
        n_dofs=V(dtype=gs.qd_int, shape=links_info_shape),
        pos=V(dtype=gs.qd_vec3, shape=links_info_shape),
        quat=V(dtype=gs.qd_vec4, shape=links_info_shape),
        invweight=V(dtype=gs.qd_vec2, shape=links_info_shape),
        is_fixed=V(dtype=gs.qd_bool, shape=links_info_shape),
        inertial_pos=V(dtype=gs.qd_vec3, shape=links_info_shape),
        inertial_quat=V(dtype=gs.qd_vec4, shape=links_info_shape),
        inertial_i=V(dtype=gs.qd_mat3, shape=links_info_shape),
        inertial_mass=V(dtype=gs.qd_float, shape=links_info_shape),
        entity_idx=V(dtype=gs.qd_int, shape=links_info_shape),
        # Heterogeneous simulation support: per-link geom/vgeom index ranges
        geom_start=V(dtype=gs.qd_int, shape=links_info_shape),
        geom_end=V(dtype=gs.qd_int, shape=links_info_shape),
        vgeom_start=V(dtype=gs.qd_int, shape=links_info_shape),
        vgeom_end=V(dtype=gs.qd_int, shape=links_info_shape),
    )


# =========================================== JointsInfo and JointsState ===========================================


@dataclasses.dataclass(eq=True, kw_only=False, frozen=True)
class JointsInfo:
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

    return JointsInfo(
        type=V(dtype=gs.qd_int, shape=shape),
        sol_params=V(dtype=gs.qd_vec7, shape=shape),
        q_start=V(dtype=gs.qd_int, shape=shape),
        dof_start=V(dtype=gs.qd_int, shape=shape),
        q_end=V(dtype=gs.qd_int, shape=shape),
        dof_end=V(dtype=gs.qd_int, shape=shape),
        n_dofs=V(dtype=gs.qd_int, shape=shape),
        pos=V(dtype=gs.qd_vec3, shape=shape),
    )


@dataclasses.dataclass(eq=True, kw_only=False, frozen=True)
class JointsState:
    xanchor: qd.Tensor
    xaxis: qd.Tensor


def get_joints_state(solver):
    shape = (solver.n_joints_, solver._B)
    requires_grad = solver._requires_grad

    return JointsState(
        xanchor=V(dtype=gs.qd_vec3, shape=shape, needs_grad=requires_grad),
        xaxis=V(dtype=gs.qd_vec3, shape=shape, needs_grad=requires_grad),
    )


# =========================================== GeomsInfo and GeomsState ===========================================


@dataclasses.dataclass(eq=True, kw_only=False, frozen=True)
class GeomsInfo:
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

    return GeomsInfo(
        pos=V(dtype=gs.qd_vec3, shape=shape),
        center=V(dtype=gs.qd_vec3, shape=shape),
        quat=V(dtype=gs.qd_vec4, shape=shape),
        data=V(dtype=gs.qd_vec7, shape=shape),
        link_idx=V(dtype=gs.qd_int, shape=shape),
        type=V(dtype=gs.qd_int, shape=shape),
        friction=V(dtype=gs.qd_float, shape=shape),
        sol_params=V(dtype=gs.qd_vec7, shape=shape),
        vert_num=V(dtype=gs.qd_int, shape=shape),
        vert_start=V(dtype=gs.qd_int, shape=shape),
        vert_end=V(dtype=gs.qd_int, shape=shape),
        verts_state_start=V(dtype=gs.qd_int, shape=shape),
        verts_state_end=V(dtype=gs.qd_int, shape=shape),
        face_num=V(dtype=gs.qd_int, shape=shape),
        face_start=V(dtype=gs.qd_int, shape=shape),
        face_end=V(dtype=gs.qd_int, shape=shape),
        edge_num=V(dtype=gs.qd_int, shape=shape),
        edge_start=V(dtype=gs.qd_int, shape=shape),
        edge_end=V(dtype=gs.qd_int, shape=shape),
        is_convex=V(dtype=gs.qd_bool, shape=shape),
        contype=V(dtype=gs.qd_int, shape=shape),
        conaffinity=V(dtype=gs.qd_int, shape=shape),
        is_fixed=V(dtype=gs.qd_bool, shape=shape),
        is_decomposed=V(dtype=gs.qd_bool, shape=shape),
        needs_coup=V(dtype=gs.qd_int, shape=shape),
        coup_friction=V(dtype=gs.qd_float, shape=shape),
        coup_softness=V(dtype=gs.qd_float, shape=shape),
        coup_restitution=V(dtype=gs.qd_float, shape=shape),
    )


@dataclasses.dataclass(eq=True, kw_only=False, frozen=True)
class GeomsState:
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

    return GeomsState(
        pos=V(dtype=gs.qd_vec3, shape=shape, needs_grad=requires_grad),
        quat=V(dtype=gs.qd_vec4, shape=shape, needs_grad=requires_grad),
        aabb_min=V(dtype=gs.qd_vec3, shape=shape),
        aabb_max=V(dtype=gs.qd_vec3, shape=shape),
        verts_updated=V(dtype=gs.qd_bool, shape=shape),
        min_buffer_idx=V(dtype=gs.qd_int, shape=shape),
        max_buffer_idx=V(dtype=gs.qd_int, shape=shape),
        hibernated=V(dtype=gs.qd_int, shape=shape),
        friction_ratio=V(dtype=gs.qd_float, shape=shape),
    )


# =========================================== VertsInfo ===========================================


@dataclasses.dataclass(eq=True, kw_only=False, frozen=True)
class VertsInfo:
    init_pos: qd.Tensor
    init_normal: qd.Tensor
    geom_idx: qd.Tensor
    init_center_pos: qd.Tensor
    verts_state_idx: qd.Tensor
    is_fixed: qd.Tensor


def get_verts_info(solver):
    shape = (solver.n_verts_,)

    return VertsInfo(
        init_pos=V(dtype=gs.qd_vec3, shape=shape),
        init_normal=V(dtype=gs.qd_vec3, shape=shape),
        geom_idx=V(dtype=gs.qd_int, shape=shape),
        init_center_pos=V(dtype=gs.qd_vec3, shape=shape),
        verts_state_idx=V(dtype=gs.qd_int, shape=shape),
        is_fixed=V(dtype=gs.qd_bool, shape=shape),
    )


# =========================================== FacesInfo ===========================================


@dataclasses.dataclass(eq=True, kw_only=False, frozen=True)
class FacesInfo:
    verts_idx: qd.Tensor
    geom_idx: qd.Tensor


def get_faces_info(solver):
    shape = (solver.n_faces_,)

    return FacesInfo(
        verts_idx=V(dtype=gs.qd_ivec3, shape=shape),
        geom_idx=V(dtype=gs.qd_int, shape=shape),
    )


# =========================================== EdgesInfo ===========================================


@dataclasses.dataclass(eq=True, kw_only=False, frozen=True)
class EdgesInfo:
    v0: qd.Tensor
    v1: qd.Tensor
    length: qd.Tensor


def get_edges_info(solver):
    shape = (solver.n_edges_,)

    return EdgesInfo(
        v0=V(dtype=gs.qd_int, shape=shape),
        v1=V(dtype=gs.qd_int, shape=shape),
        length=V(dtype=gs.qd_float, shape=shape),
    )


# =========================================== VertsState ===========================================


@dataclasses.dataclass(eq=True, kw_only=False, frozen=True)
class VertsState:
    pos: qd.Tensor


def get_free_verts_state(solver):
    return VertsState(
        pos=V(dtype=gs.qd_vec3, shape=(solver.n_free_verts_, solver._B)),
    )


def get_fixed_verts_state(solver):
    return VertsState(
        pos=V(dtype=gs.qd_vec3, shape=(solver.n_fixed_verts_,)),
    )


# =========================================== VvertsInfo ===========================================


@dataclasses.dataclass(eq=True, kw_only=False, frozen=True)
class VVertsInfo:
    init_pos: qd.Tensor
    init_vnormal: qd.Tensor
    vgeom_idx: qd.Tensor


def get_vverts_info(solver):
    shape = (solver.n_vverts_,)

    return VVertsInfo(
        init_pos=V(dtype=gs.qd_vec3, shape=shape),
        init_vnormal=V(dtype=gs.qd_vec3, shape=shape),
        vgeom_idx=V(dtype=gs.qd_int, shape=shape),
    )


# =========================================== VfacesInfo ===========================================


@dataclasses.dataclass(eq=True, kw_only=False, frozen=True)
class VFacesInfo:
    vverts_idx: qd.Tensor
    vgeom_idx: qd.Tensor


def get_vfaces_info(solver):
    shape = (solver.n_vfaces_,)

    return VFacesInfo(
        vverts_idx=V(dtype=gs.qd_ivec3, shape=shape),
        vgeom_idx=V(dtype=gs.qd_int, shape=shape),
    )


# =========================================== VgeomsInfo ===========================================


@dataclasses.dataclass(eq=True, kw_only=False, frozen=True)
class VGeomsInfo:
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

    return VGeomsInfo(
        pos=V(dtype=gs.qd_vec3, shape=shape),
        quat=V(dtype=gs.qd_vec4, shape=shape),
        link_idx=V(dtype=gs.qd_int, shape=shape),
        vvert_num=V(dtype=gs.qd_int, shape=shape),
        vvert_start=V(dtype=gs.qd_int, shape=shape),
        vvert_end=V(dtype=gs.qd_int, shape=shape),
        vface_num=V(dtype=gs.qd_int, shape=shape),
        vface_start=V(dtype=gs.qd_int, shape=shape),
        vface_end=V(dtype=gs.qd_int, shape=shape),
        color=V(dtype=gs.qd_vec4, shape=shape),
    )


# =========================================== VGeomsState ===========================================


@dataclasses.dataclass(eq=True, kw_only=False, frozen=True)
class VGeomsState:
    pos: qd.Tensor
    quat: qd.Tensor


def get_vgeoms_state(solver):
    shape = (solver.n_vgeoms_, solver._B)

    return VGeomsState(
        pos=V(dtype=gs.qd_vec3, shape=shape),
        quat=V(dtype=gs.qd_vec4, shape=shape),
    )


# =========================================== EqualitiesInfo ===========================================


@dataclasses.dataclass(eq=True, kw_only=False, frozen=True)
class EqualitiesInfo:
    eq_obj1id: qd.Tensor
    eq_obj2id: qd.Tensor
    eq_data: qd.Tensor
    eq_type: qd.Tensor
    sol_params: qd.Tensor


def get_equalities_info(solver):
    shape = (solver.n_candidate_equalities_, solver._B)

    return EqualitiesInfo(
        eq_obj1id=V(dtype=gs.qd_int, shape=shape),
        eq_obj2id=V(dtype=gs.qd_int, shape=shape),
        eq_data=V(dtype=gs.qd_vec11, shape=shape),
        eq_type=V(dtype=gs.qd_int, shape=shape),
        sol_params=V(dtype=gs.qd_vec7, shape=shape),
    )


# =========================================== EntitiesInfo ===========================================


@dataclasses.dataclass(eq=True, kw_only=False, frozen=True)
class EntitiesInfo:
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

    return EntitiesInfo(
        dof_start=V(dtype=gs.qd_int, shape=shape),
        dof_end=V(dtype=gs.qd_int, shape=shape),
        n_dofs=V(dtype=gs.qd_int, shape=shape),
        link_start=V(dtype=gs.qd_int, shape=shape),
        link_end=V(dtype=gs.qd_int, shape=shape),
        n_links=V(dtype=gs.qd_int, shape=shape),
        geom_start=V(dtype=gs.qd_int, shape=shape),
        geom_end=V(dtype=gs.qd_int, shape=shape),
        n_geoms=V(dtype=gs.qd_int, shape=shape),
        gravity_compensation=V(dtype=gs.qd_float, shape=shape),
        is_local_collision_mask=V(dtype=gs.qd_bool, shape=shape),
    )


# =========================================== EntitiesState ===========================================


@dataclasses.dataclass(eq=True, kw_only=False, frozen=True)
class EntitiesState:
    hibernated: qd.Tensor


def get_entities_state(solver):
    return EntitiesState(
        hibernated=V(dtype=gs.qd_int, shape=(solver.n_entities_, solver._B)),
    )


# =========================================== RigidAdjointCache ===========================================
@dataclasses.dataclass(eq=True, kw_only=False, frozen=True)
class RigidAdjointCache:
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

    return RigidAdjointCache(
        qpos=V(dtype=gs.qd_float, shape=(substeps_local + 1, solver.n_qs_, solver._B), needs_grad=requires_grad),
        dofs_vel=V(dtype=gs.qd_float, shape=(substeps_local + 1, solver.n_dofs_, solver._B), needs_grad=requires_grad),
        dofs_acc=V(dtype=gs.qd_float, shape=(substeps_local + 1, solver.n_dofs_, solver._B), needs_grad=requires_grad),
    )


# =================================== RigidSimStaticConfig ===================================


@qd.data_oriented
class RigidSimStaticConfig(metaclass=AutoInitMeta):
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
        self.errno = V(dtype=gs.qd_int, shape=(solver._B,))


# =========================================== RaycastResult ===========================================


@dataclasses.dataclass(eq=True, kw_only=False, frozen=True)
class RaycastResult:
    distance: qd.Tensor
    geom_idx: qd.Tensor
    hit_point: qd.Tensor
    normal: qd.Tensor


def get_raycast_result(n_envs: int):
    return RaycastResult(
        distance=V(dtype=gs.qd_float, shape=(n_envs,)),
        geom_idx=V(dtype=gs.qd_int, shape=(n_envs,)),
        hit_point=V_VEC(3, dtype=gs.qd_float, shape=(n_envs,)),
        normal=V_VEC(3, dtype=gs.qd_float, shape=(n_envs,)),
    )


GeomsInitAABB = qd.Tensor
