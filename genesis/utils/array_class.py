import dataclasses
import math
from enum import IntEnum
from functools import partial

import quadrants as qd
import numpy as np
from typing_extensions import dataclass_transform  # Made it into standard lib from Python 3.12

import genesis as gs

if not gs._initialized:
    gs.raise_exception("Genesis hasn't been initialized. Did you call `gs.init()`?")


V_ANNOTATION = qd.types.ndarray() if gs.use_ndarray else qd.template
V = qd.ndarray if gs.use_ndarray else qd.field
V_VEC = qd.Vector.ndarray if gs.use_ndarray else qd.Vector.field
V_MAT = qd.Matrix.ndarray if gs.use_ndarray else qd.Matrix.field

DATA_ORIENTED = partial(dataclasses.dataclass, frozen=True) if gs.use_ndarray else qd.data_oriented
PLACEHOLDER = V(dtype=gs.qd_float, shape=())


def maybe_shape(shape, is_on):
    return shape if is_on else ()


@dataclass_transform(eq_default=True, order_default=True, kw_only_default=False, frozen_default=True)
class AutoInitMeta(type):
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


BASE_METACLASS = type if gs.use_ndarray else AutoInitMeta


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
    INVALID_IPC_NAN = 0b00000000000000000000000000100000


# =========================================== RigidGlobalInfo ===========================================


@DATA_ORIENTED
class StructRigidGlobalInfo(metaclass=BASE_METACLASS):
    # *_bw: Cache for backward pass
    n_awake_dofs: V_ANNOTATION
    awake_dofs: V_ANNOTATION
    n_awake_entities: V_ANNOTATION
    awake_entities: V_ANNOTATION
    n_awake_links: V_ANNOTATION
    awake_links: V_ANNOTATION
    qpos0: V_ANNOTATION
    qpos: V_ANNOTATION
    qpos_next: V_ANNOTATION
    links_T: V_ANNOTATION
    envs_offset: V_ANNOTATION
    geoms_init_AABB: V_ANNOTATION
    mass_mat: V_ANNOTATION
    mass_mat_L: V_ANNOTATION
    mass_mat_L_bw: V_ANNOTATION
    mass_mat_D_inv: V_ANNOTATION
    mass_mat_mask: V_ANNOTATION
    meaninertia: V_ANNOTATION
    mass_parent_mask: V_ANNOTATION
    gravity: V_ANNOTATION
    # Runtime constants
    substep_dt: V_ANNOTATION
    iterations: V_ANNOTATION
    tolerance: V_ANNOTATION
    ls_iterations: V_ANNOTATION
    ls_tolerance: V_ANNOTATION
    noslip_iterations: V_ANNOTATION
    noslip_tolerance: V_ANNOTATION
    n_equalities: V_ANNOTATION
    n_candidate_equalities: V_ANNOTATION
    hibernation_thresh_acc: V_ANNOTATION
    hibernation_thresh_vel: V_ANNOTATION
    EPS: V_ANNOTATION


def get_rigid_global_info(solver):
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

    return StructRigidGlobalInfo(
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


@DATA_ORIENTED
class StructConstraintState(metaclass=BASE_METACLASS):
    is_warmstart: V_ANNOTATION
    n_constraints: V_ANNOTATION
    qd_n_equalities: V_ANNOTATION
    jac: V_ANNOTATION
    diag: V_ANNOTATION
    aref: V_ANNOTATION
    jac_relevant_dofs: V_ANNOTATION
    jac_n_relevant_dofs: V_ANNOTATION
    n_constraints_equality: V_ANNOTATION
    n_constraints_frictionloss: V_ANNOTATION
    improved: V_ANNOTATION
    Jaref: V_ANNOTATION
    Ma: V_ANNOTATION
    Ma_ws: V_ANNOTATION
    grad: V_ANNOTATION
    Mgrad: V_ANNOTATION
    search: V_ANNOTATION
    efc_D: V_ANNOTATION
    efc_frictionloss: V_ANNOTATION
    efc_force: V_ANNOTATION
    efc_b: V_ANNOTATION
    efc_AR: V_ANNOTATION
    active: V_ANNOTATION
    prev_active: V_ANNOTATION
    qfrc_constraint: V_ANNOTATION
    qacc: V_ANNOTATION
    qacc_ws: V_ANNOTATION
    qacc_prev: V_ANNOTATION
    cost_ws: V_ANNOTATION
    gauss: V_ANNOTATION
    cost: V_ANNOTATION
    prev_cost: V_ANNOTATION
    gtol: V_ANNOTATION
    mv: V_ANNOTATION
    jv: V_ANNOTATION
    quad_gauss: V_ANNOTATION
    candidates: V_ANNOTATION
    eq_sum: V_ANNOTATION
    ls_it: V_ANNOTATION
    ls_result: V_ANNOTATION
    # Optional CG fields
    cg_prev_grad: V_ANNOTATION
    cg_prev_Mgrad: V_ANNOTATION
    cg_beta: V_ANNOTATION
    cg_pg_dot_pMg: V_ANNOTATION
    # Optional Newton fields
    # Hessian matrix of the optimization problem as a dense 2D tensor.
    # Note that only the lower triangular part is updated for efficiency because this matrix is symmetric by definition.
    # As a result, the values of the strictly upper triangular part is undefined.
    # In practice, this variable is re-purposed to store the Cholesky factor L st H = L @ L.T to spare memory resources.
    # TODO: Optimize storage to only allocate memory half of the Hessian matrix to sparse memory resources.
    nt_H: V_ANNOTATION
    nt_vec: V_ANNOTATION
    # Compacted list of constraints whose active state changed, used by incremental Cholesky update
    # to reduce GPU thread divergence by iterating only over constraints that need processing.
    incr_changed_idx: V_ANNOTATION
    incr_n_changed: V_ANNOTATION
    # Backward gradients
    dL_dqacc: V_ANNOTATION
    dL_dM: V_ANNOTATION
    dL_djac: V_ANNOTATION
    dL_daref: V_ANNOTATION
    dL_defc_D: V_ANNOTATION
    dL_dforce: V_ANNOTATION
    # Backward buffers for linear system solver
    bw_u: V_ANNOTATION
    bw_r: V_ANNOTATION
    bw_p: V_ANNOTATION
    bw_Ap: V_ANNOTATION
    bw_Ju: V_ANNOTATION
    bw_y: V_ANNOTATION
    bw_w: V_ANNOTATION
    # Timers for profiling
    timers: V_ANNOTATION


def get_constraint_state(constraint_solver, solver):
    _B = solver._B
    len_constraints_ = constraint_solver.len_constraints_

    jac_shape = (len_constraints_, solver.n_dofs_, _B)
    efc_AR_shape = maybe_shape((len_constraints_, len_constraints_, _B), solver._options.noslip_iterations > 0)
    efc_b_shape = maybe_shape((len_constraints_, _B), solver._options.noslip_iterations > 0)
    jac_relevant_dofs_shape = maybe_shape((len_constraints_, solver.n_dofs_, _B), constraint_solver.sparse_solve)
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
        candidates=V(dtype=gs.qd_float, shape=(12, _B)),
        eq_sum=V(dtype=gs.qd_float, shape=(3, _B)),
        Ma=V(dtype=gs.qd_float, shape=(solver.n_dofs_, _B)),
        Ma_ws=V(dtype=gs.qd_float, shape=(solver.n_dofs_, _B)),
        grad=V(dtype=gs.qd_float, shape=(solver.n_dofs_, _B)),
        Mgrad=V(dtype=gs.qd_float, shape=(solver.n_dofs_, _B)),
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
    )


# =========================================== Collider ===========================================


@DATA_ORIENTED
class StructContactData(metaclass=BASE_METACLASS):
    geom_a: V_ANNOTATION
    geom_b: V_ANNOTATION
    penetration: V_ANNOTATION
    normal: V_ANNOTATION
    pos: V_ANNOTATION
    friction: V_ANNOTATION
    sol_params: V_ANNOTATION
    force: V_ANNOTATION
    link_a: V_ANNOTATION
    link_b: V_ANNOTATION


def get_contact_data(solver, max_contact_pairs, requires_grad):
    _B = solver._B
    max_contact_pairs_ = max(max_contact_pairs, 1)

    return StructContactData(
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
    )


@DATA_ORIENTED
class StructDiffContactInput(metaclass=BASE_METACLASS):
    ### Non-differentiable input data
    # Geom id of the two geometries
    geom_a: V_ANNOTATION
    geom_b: V_ANNOTATION
    # Local positions of the 3 vertices from the two geometries that define the face on the Minkowski difference
    local_pos1_a: V_ANNOTATION
    local_pos1_b: V_ANNOTATION
    local_pos1_c: V_ANNOTATION
    local_pos2_a: V_ANNOTATION
    local_pos2_b: V_ANNOTATION
    local_pos2_c: V_ANNOTATION
    # Local positions of the 1 vertex from the two geometries that define the support point for the face above
    w_local_pos1: V_ANNOTATION
    w_local_pos2: V_ANNOTATION
    # Reference id of the contact point, which is needed for the backward pass
    ref_id: V_ANNOTATION
    # Flag whether the contact data can be computed in numerically stable way in both the forward and backward passes
    valid: V_ANNOTATION
    ### Differentiable input data
    # Reference penetration depth, which is needed for computing the weight of the contact point
    ref_penetration: V_ANNOTATION


def get_diff_contact_input(solver, max_contacts_per_pair, is_active):
    _B = solver._B
    shape = maybe_shape((_B, max_contacts_per_pair), is_active and solver._requires_grad)
    return StructDiffContactInput(
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


@DATA_ORIENTED
class StructSortBuffer(metaclass=BASE_METACLASS):
    value: V_ANNOTATION
    i_g: V_ANNOTATION
    is_max: V_ANNOTATION


def get_sort_buffer(solver):
    _B = solver._B

    return StructSortBuffer(
        value=V(dtype=gs.qd_float, shape=(2 * solver.n_geoms_, _B)),
        i_g=V(dtype=gs.qd_int, shape=(2 * solver.n_geoms_, _B)),
        is_max=V(dtype=gs.qd_bool, shape=(2 * solver.n_geoms_, _B)),
    )


@DATA_ORIENTED
class StructContactCache(metaclass=BASE_METACLASS):
    normal: V_ANNOTATION


def get_contact_cache(solver, n_possible_pairs):
    _B = solver._B
    return StructContactCache(
        normal=V_VEC(3, dtype=gs.qd_float, shape=(n_possible_pairs, _B)),
    )


@DATA_ORIENTED
class StructAggList(metaclass=BASE_METACLASS):
    curr: V_ANNOTATION
    n: V_ANNOTATION
    start: V_ANNOTATION


def get_agg_list(solver):
    _B = solver._B
    n_entities = max(solver.n_entities, 1)

    return StructAggList(
        curr=V(dtype=gs.qd_int, shape=(n_entities, _B)),
        n=V(dtype=gs.qd_int, shape=(n_entities, _B)),
        start=V(dtype=gs.qd_int, shape=(n_entities, _B)),
    )


@DATA_ORIENTED
class StructContactIslandState(metaclass=BASE_METACLASS):
    ci_edges: V_ANNOTATION
    edge_id: V_ANNOTATION
    constraint_list: V_ANNOTATION
    constraint_id: V_ANNOTATION
    entity_edge: StructAggList
    island_col: StructAggList
    island_hibernated: V_ANNOTATION
    island_entity: StructAggList
    entity_id: V_ANNOTATION
    n_edges: V_ANNOTATION
    n_islands: V_ANNOTATION
    n_stack: V_ANNOTATION
    entity_island: V_ANNOTATION
    stack: V_ANNOTATION
    entity_idx_to_next_entity_idx_in_hibernated_island: V_ANNOTATION


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


@DATA_ORIENTED
class StructColliderState(metaclass=BASE_METACLASS):
    sort_buffer: StructSortBuffer
    contact_data: StructContactData
    active_buffer: V_ANNOTATION
    n_broad_pairs: V_ANNOTATION
    broad_collision_pairs: V_ANNOTATION
    active_buffer_awake: V_ANNOTATION
    active_buffer_hib: V_ANNOTATION
    box_depth: V_ANNOTATION
    box_points: V_ANNOTATION
    box_pts: V_ANNOTATION
    box_lines: V_ANNOTATION
    box_linesu: V_ANNOTATION
    box_axi: V_ANNOTATION
    box_ppts2: V_ANNOTATION
    box_pu: V_ANNOTATION
    xyz_max_min: V_ANNOTATION
    prism: V_ANNOTATION
    n_contacts: V_ANNOTATION
    n_contacts_hibernated: V_ANNOTATION
    first_time: V_ANNOTATION
    contact_cache: StructContactCache
    # Input data for differentiable contact detection used in the backward pass
    diff_contact_input: StructDiffContactInput


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
        diff_contact_input=get_diff_contact_input(solver, max(max_contact_pairs, 1), is_active=True),
    )


@DATA_ORIENTED
class StructColliderInfo(metaclass=BASE_METACLASS):
    vert_neighbors: V_ANNOTATION
    vert_neighbor_start: V_ANNOTATION
    vert_n_neighbors: V_ANNOTATION
    collision_pair_idx: V_ANNOTATION
    max_possible_pairs: V_ANNOTATION
    max_collision_pairs: V_ANNOTATION
    max_contact_pairs: V_ANNOTATION
    max_collision_pairs_broad: V_ANNOTATION
    # Terrain fields
    terrain_hf: V_ANNOTATION
    terrain_rc: V_ANNOTATION
    terrain_scale: V_ANNOTATION
    terrain_xyz_maxmin: V_ANNOTATION
    # multi contact perturbation and tolerance
    mc_perturbation: V_ANNOTATION
    mc_tolerance: V_ANNOTATION
    mpr_to_gjk_overlap_ratio: V_ANNOTATION
    # differentiable contact tolerance
    diff_pos_tolerance: V_ANNOTATION
    diff_normal_tolerance: V_ANNOTATION


def get_collider_info(solver, n_vert_neighbors, collider_static_config, **kwargs):
    for geom in solver.geoms:
        if geom.type == gs.GEOM_TYPE.TERRAIN:
            terrain_hf_shape = geom.entity.terrain_hf.shape
            break
    else:
        terrain_hf_shape = 1

    return StructColliderInfo(
        vert_neighbors=V(dtype=gs.qd_int, shape=(max(n_vert_neighbors, 1),)),
        vert_neighbor_start=V(dtype=gs.qd_int, shape=(solver.n_verts_,)),
        vert_n_neighbors=V(dtype=gs.qd_int, shape=(solver.n_verts_,)),
        collision_pair_idx=V(dtype=gs.qd_int, shape=(solver.n_geoms_, solver.n_geoms_)),
        max_possible_pairs=V(dtype=gs.qd_int, shape=()),
        max_collision_pairs=V(dtype=gs.qd_int, shape=()),
        max_contact_pairs=V(dtype=gs.qd_int, shape=()),
        max_collision_pairs_broad=V(dtype=gs.qd_int, shape=()),
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
class StructColliderStaticConfig(metaclass=AutoInitMeta):
    has_terrain: bool
    has_convex_convex: bool
    has_convex_specialization: bool
    has_nonconvex_nonterrain: bool
    # maximum number of contact pairs per collision pair
    n_contacts_per_pair: int
    # ccd algorithm
    ccd_algorithm: int


# =========================================== MPR ===========================================


@DATA_ORIENTED
class StructMPRSimplexSupport(metaclass=BASE_METACLASS):
    v1: V_ANNOTATION
    v2: V_ANNOTATION
    v: V_ANNOTATION


def get_mpr_simplex_support(B_):
    return StructMPRSimplexSupport(
        v1=V_VEC(3, dtype=gs.qd_float, shape=(4, B_)),
        v2=V_VEC(3, dtype=gs.qd_float, shape=(4, B_)),
        v=V_VEC(3, dtype=gs.qd_float, shape=(4, B_)),
    )


@DATA_ORIENTED
class StructMPRState(metaclass=BASE_METACLASS):
    simplex_support: StructMPRSimplexSupport
    simplex_size: V_ANNOTATION


def get_mpr_state(B_):
    return StructMPRState(
        simplex_support=get_mpr_simplex_support(B_),
        simplex_size=V(dtype=gs.qd_int, shape=(B_,)),
    )


@DATA_ORIENTED
class StructMPRInfo(metaclass=BASE_METACLASS):
    CCD_EPS: V_ANNOTATION
    CCD_TOLERANCE: V_ANNOTATION
    CCD_ITERATIONS: V_ANNOTATION


def get_mpr_info(**kwargs):
    return StructMPRInfo(
        CCD_EPS=V_SCALAR_FROM(dtype=gs.qd_float, value=kwargs["CCD_EPS"]),
        CCD_TOLERANCE=V_SCALAR_FROM(dtype=gs.qd_float, value=kwargs["CCD_TOLERANCE"]),
        CCD_ITERATIONS=V_SCALAR_FROM(dtype=gs.qd_float, value=kwargs["CCD_ITERATIONS"]),
    )


# =========================================== GJK ===========================================


@DATA_ORIENTED
class StructMDVertex(metaclass=BASE_METACLASS):
    # Vertex of the Minkowski difference
    obj1: V_ANNOTATION
    obj2: V_ANNOTATION
    local_obj1: V_ANNOTATION
    local_obj2: V_ANNOTATION
    id1: V_ANNOTATION
    id2: V_ANNOTATION
    mink: V_ANNOTATION


def get_gjk_simplex_vertex(solver, is_active):
    _B = solver._B
    shape = maybe_shape((_B, 4), is_active)
    return StructMDVertex(
        obj1=V_VEC(3, dtype=gs.qd_float, shape=shape),
        obj2=V_VEC(3, dtype=gs.qd_float, shape=shape),
        local_obj1=V_VEC(3, dtype=gs.qd_float, shape=shape),
        local_obj2=V_VEC(3, dtype=gs.qd_float, shape=shape),
        id1=V(dtype=gs.qd_int, shape=shape),
        id2=V(dtype=gs.qd_int, shape=shape),
        mink=V_VEC(3, dtype=gs.qd_float, shape=shape),
    )


def get_epa_polytope_vertex(solver, gjk_info, is_active):
    _B = solver._B
    max_num_polytope_verts = 5 + gjk_info.epa_max_iterations[None]
    shape = maybe_shape((_B, max_num_polytope_verts), is_active)
    return StructMDVertex(
        obj1=V_VEC(3, dtype=gs.qd_float, shape=shape),
        obj2=V_VEC(3, dtype=gs.qd_float, shape=shape),
        local_obj1=V_VEC(3, dtype=gs.qd_float, shape=shape),
        local_obj2=V_VEC(3, dtype=gs.qd_float, shape=shape),
        id1=V(dtype=gs.qd_int, shape=shape),
        id2=V(dtype=gs.qd_int, shape=shape),
        mink=V_VEC(3, dtype=gs.qd_float, shape=shape),
    )


@DATA_ORIENTED
class StructGJKSimplex(metaclass=BASE_METACLASS):
    nverts: V_ANNOTATION
    dist: V_ANNOTATION


def get_gjk_simplex(solver, is_active):
    _B = solver._B
    shape = maybe_shape((_B,), is_active)
    return StructGJKSimplex(
        nverts=V(dtype=gs.qd_int, shape=shape),
        dist=V(dtype=gs.qd_float, shape=shape),
    )


@DATA_ORIENTED
class StructGJKSimplexBuffer(metaclass=BASE_METACLASS):
    normal: V_ANNOTATION
    sdist: V_ANNOTATION


def get_gjk_simplex_buffer(solver, is_active):
    _B = solver._B
    shape = maybe_shape((_B, 4), is_active)
    return StructGJKSimplexBuffer(
        normal=V_VEC(3, dtype=gs.qd_float, shape=shape),
        sdist=V(dtype=gs.qd_float, shape=shape),
    )


@DATA_ORIENTED
class StructEPAPolytope(metaclass=BASE_METACLASS):
    nverts: V_ANNOTATION
    nfaces: V_ANNOTATION
    nfaces_map: V_ANNOTATION
    horizon_nedges: V_ANNOTATION
    horizon_w: V_ANNOTATION


def get_epa_polytope(solver, is_active):
    _B = solver._B
    shape = maybe_shape((_B,), is_active)
    return StructEPAPolytope(
        nverts=V(dtype=gs.qd_int, shape=shape),
        nfaces=V(dtype=gs.qd_int, shape=shape),
        nfaces_map=V(dtype=gs.qd_int, shape=shape),
        horizon_nedges=V(dtype=gs.qd_int, shape=shape),
        horizon_w=V_VEC(3, dtype=gs.qd_float, shape=shape),
    )


@DATA_ORIENTED
class StructEPAPolytopeFace(metaclass=BASE_METACLASS):
    verts_idx: V_ANNOTATION
    adj_idx: V_ANNOTATION
    normal: V_ANNOTATION
    dist2: V_ANNOTATION
    map_idx: V_ANNOTATION
    visited: V_ANNOTATION


def get_epa_polytope_face(solver, polytope_max_faces, is_active):
    _B = solver._B
    shape = maybe_shape((_B, polytope_max_faces), is_active)
    return StructEPAPolytopeFace(
        verts_idx=V_VEC(3, dtype=gs.qd_int, shape=shape),
        adj_idx=V_VEC(3, dtype=gs.qd_int, shape=shape),
        normal=V_VEC(3, dtype=gs.qd_float, shape=shape),
        dist2=V(dtype=gs.qd_float, shape=shape),
        map_idx=V(dtype=gs.qd_int, shape=shape),
        visited=V(dtype=gs.qd_int, shape=shape),
    )


@DATA_ORIENTED
class StructEPAPolytopeHorizonData(metaclass=BASE_METACLASS):
    face_idx: V_ANNOTATION
    edge_idx: V_ANNOTATION


def get_epa_polytope_horizon_data(solver, polytope_max_horizons, is_active):
    _B = solver._B
    shape = maybe_shape((_B, polytope_max_horizons), is_active)
    return StructEPAPolytopeHorizonData(
        face_idx=V(dtype=gs.qd_int, shape=shape),
        edge_idx=V(dtype=gs.qd_int, shape=shape),
    )


@DATA_ORIENTED
class StructContactFace(metaclass=BASE_METACLASS):
    vert1: V_ANNOTATION
    vert2: V_ANNOTATION
    endverts: V_ANNOTATION
    normal1: V_ANNOTATION
    normal2: V_ANNOTATION
    id1: V_ANNOTATION
    id2: V_ANNOTATION


def get_contact_face(solver, max_contact_polygon_verts, is_active):
    _B = solver._B
    shape = maybe_shape((_B, max_contact_polygon_verts), is_active)
    return StructContactFace(
        vert1=V_VEC(3, dtype=gs.qd_float, shape=shape),
        vert2=V_VEC(3, dtype=gs.qd_float, shape=shape),
        endverts=V_VEC(3, dtype=gs.qd_float, shape=shape),
        normal1=V_VEC(3, dtype=gs.qd_float, shape=shape),
        normal2=V_VEC(3, dtype=gs.qd_float, shape=shape),
        id1=V(dtype=gs.qd_int, shape=shape),
        id2=V(dtype=gs.qd_int, shape=shape),
    )


@DATA_ORIENTED
class StructContactNormal(metaclass=BASE_METACLASS):
    endverts: V_ANNOTATION
    normal: V_ANNOTATION
    id: V_ANNOTATION


def get_contact_normal(solver, max_contact_polygon_verts, is_active):
    _B = solver._B
    shape = maybe_shape((_B, max_contact_polygon_verts), is_active)
    return StructContactNormal(
        endverts=V_VEC(3, dtype=gs.qd_float, shape=shape),
        normal=V_VEC(3, dtype=gs.qd_float, shape=shape),
        id=V(dtype=gs.qd_int, shape=shape),
    )


@DATA_ORIENTED
class StructContactHalfspace(metaclass=BASE_METACLASS):
    normal: V_ANNOTATION
    dist: V_ANNOTATION


def get_contact_halfspace(solver, max_contact_polygon_verts, is_active):
    _B = solver._B
    shape = maybe_shape((_B, max_contact_polygon_verts), is_active)
    return StructContactHalfspace(
        normal=V_VEC(3, dtype=gs.qd_float, shape=shape),
        dist=V(dtype=gs.qd_float, shape=shape),
    )


@DATA_ORIENTED
class StructWitness(metaclass=BASE_METACLASS):
    point_obj1: V_ANNOTATION
    point_obj2: V_ANNOTATION


def get_witness(solver, max_contacts_per_pair, is_active):
    _B = solver._B
    shape = maybe_shape((_B, max_contacts_per_pair), is_active)
    return StructWitness(
        point_obj1=V_VEC(3, dtype=gs.qd_float, shape=shape),
        point_obj2=V_VEC(3, dtype=gs.qd_float, shape=shape),
    )


@DATA_ORIENTED
class StructGJKState(metaclass=BASE_METACLASS):
    support_mesh_prev_vertex_id: V_ANNOTATION
    simplex_vertex: StructMDVertex
    simplex_buffer: StructGJKSimplexBuffer
    simplex: StructGJKSimplex
    simplex_vertex_intersect: StructMDVertex
    simplex_buffer_intersect: StructGJKSimplexBuffer
    nsimplex: V_ANNOTATION
    last_searched_simplex_vertex_id: V_ANNOTATION
    polytope: StructEPAPolytope
    polytope_verts: StructMDVertex
    polytope_faces: StructEPAPolytopeFace
    polytope_faces_map: V_ANNOTATION
    polytope_horizon_data: StructEPAPolytopeHorizonData
    polytope_horizon_stack: StructEPAPolytopeHorizonData
    contact_faces: StructContactFace
    contact_normals: StructContactNormal
    contact_halfspaces: StructContactHalfspace
    contact_clipped_polygons: V_ANNOTATION
    multi_contact_flag: V_ANNOTATION
    witness: StructWitness
    n_witness: V_ANNOTATION
    n_contacts: V_ANNOTATION
    contact_pos: V_ANNOTATION
    normal: V_ANNOTATION
    is_col: V_ANNOTATION
    penetration: V_ANNOTATION
    distance: V_ANNOTATION
    # Differentiable contact detection
    diff_contact_input: StructDiffContactInput
    n_diff_contact_input: V_ANNOTATION
    diff_penetration: V_ANNOTATION


def get_gjk_state(solver, static_rigid_sim_config, gjk_info, is_active):
    _B = solver._B
    enable_mujoco_compatibility = static_rigid_sim_config.enable_mujoco_compatibility
    polytope_max_faces = gjk_info.polytope_max_faces[None]
    max_contacts_per_pair = gjk_info.max_contacts_per_pair[None]
    max_contact_polygon_verts = gjk_info.max_contact_polygon_verts[None]
    requires_grad = solver._static_rigid_sim_config.requires_grad

    # FIXME: Define GJKState and MujocoCompatGJKState that derives from the former but defines additional attributes
    return StructGJKState(
        # GJK simplex
        support_mesh_prev_vertex_id=V(dtype=gs.qd_int, shape=(_B, 2)),
        simplex_vertex=get_gjk_simplex_vertex(solver, is_active),
        simplex_buffer=get_gjk_simplex_buffer(solver, is_active),
        simplex=get_gjk_simplex(solver, is_active),
        last_searched_simplex_vertex_id=V(dtype=gs.qd_int, shape=(_B,)),
        simplex_vertex_intersect=get_gjk_simplex_vertex(solver, is_active),
        simplex_buffer_intersect=get_gjk_simplex_buffer(solver, is_active),
        nsimplex=V(dtype=gs.qd_int, shape=(_B,)),
        # EPA polytope
        polytope=get_epa_polytope(solver, is_active),
        polytope_verts=get_epa_polytope_vertex(solver, gjk_info, is_active),
        polytope_faces=get_epa_polytope_face(solver, polytope_max_faces, is_active),
        polytope_faces_map=V(dtype=gs.qd_int, shape=(_B, polytope_max_faces)),
        polytope_horizon_data=get_epa_polytope_horizon_data(solver, 6 + gjk_info.epa_max_iterations[None], is_active),
        polytope_horizon_stack=get_epa_polytope_horizon_data(solver, polytope_max_faces * 3, is_active),
        # Multi-contact detection (MuJoCo compatibility)
        contact_faces=get_contact_face(solver, max_contact_polygon_verts, is_active),
        contact_normals=get_contact_normal(solver, max_contact_polygon_verts, is_active),
        contact_halfspaces=get_contact_halfspace(solver, max_contact_polygon_verts, is_active),
        contact_clipped_polygons=V_VEC(3, dtype=gs.qd_float, shape=(_B, 2, max_contact_polygon_verts)),
        multi_contact_flag=V(dtype=gs.qd_bool, shape=(_B,)),
        # Final results
        witness=get_witness(solver, max_contacts_per_pair, is_active),
        n_witness=V(dtype=gs.qd_int, shape=(_B,)),
        n_contacts=V(dtype=gs.qd_int, shape=(_B,)),
        contact_pos=V_VEC(3, dtype=gs.qd_float, shape=(_B, max_contacts_per_pair)),
        normal=V_VEC(3, dtype=gs.qd_float, shape=(_B, max_contacts_per_pair)),
        is_col=V(dtype=gs.qd_bool, shape=(_B,)),
        penetration=V(dtype=gs.qd_float, shape=(_B,)),
        distance=V(dtype=gs.qd_float, shape=(_B,)),
        diff_contact_input=get_diff_contact_input(solver, max(max_contacts_per_pair, 1), is_active),
        n_diff_contact_input=V(dtype=gs.qd_int, shape=(_B,)),
        diff_penetration=V(dtype=gs.qd_float, shape=maybe_shape((_B, max_contacts_per_pair), requires_grad)),
    )


@DATA_ORIENTED
class StructGJKInfo(metaclass=BASE_METACLASS):
    max_contacts_per_pair: V_ANNOTATION
    max_contact_polygon_verts: V_ANNOTATION
    # Maximum number of iterations for GJK and EPA algorithms
    gjk_max_iterations: V_ANNOTATION
    epa_max_iterations: V_ANNOTATION
    FLOAT_MIN: V_ANNOTATION
    FLOAT_MIN_SQ: V_ANNOTATION
    FLOAT_MAX: V_ANNOTATION
    FLOAT_MAX_SQ: V_ANNOTATION
    # Tolerance for stopping GJK and EPA algorithms when they converge (only for non-discrete geometries).
    tolerance: V_ANNOTATION
    # If the distance between two objects is smaller than this value, we consider them colliding.
    collision_eps: V_ANNOTATION
    # In safe GJK, we do not allow degenerate simplex to happen, because it becomes the main reason of EPA errors.
    # To prevent degeneracy, we throw away the simplex that has smaller degeneracy measure (e.g. colinearity,
    # coplanarity) than this threshold.
    simplex_max_degeneracy_sq: V_ANNOTATION
    polytope_max_faces: V_ANNOTATION
    # Threshold for reprojection error when we compute the witness points from the polytope. In computing the
    # witness points, we project the origin onto the polytope faces and compute the barycentric coordinates of the
    # projected point. To confirm the projection is valid, we compute the projected point using the barycentric
    # coordinates and compare it with the original projected point. If the difference is larger than this threshold,
    # we consider the projection invalid, because it means numerical errors are too large.
    polytope_max_reprojection_error: V_ANNOTATION
    # Tolerance for normal alignment between (face-face) or (edge-face). The normals should align within this
    # tolerance to be considered as a valid parallel contact.
    contact_face_tol: V_ANNOTATION
    contact_edge_tol: V_ANNOTATION
    # Epsilon values for differentiable contact. [eps_boundary] denotes the maximum distance between the face
    # and the support point in the direction of the face normal. If this distance is 0, the face is on the
    # boundary of the Minkowski difference. For [eps_distance], the distance between the origin and the face
    # should not exceed this eps value plus the default EPA depth. For [eps_affine], the affine coordinates
    # of the origin's projection onto the face should not violate [0, 1] range by this eps value.
    # FIXME: Adjust these values based on the case study.
    diff_contact_eps_boundary: V_ANNOTATION
    diff_contact_eps_distance: V_ANNOTATION
    diff_contact_eps_affine: V_ANNOTATION
    # The minimum norm of the normal to be considered as a valid normal in the differentiable formulation.
    diff_contact_min_normal_norm: V_ANNOTATION
    # The minimum penetration depth to be considered as a valid contact in the differentiable formulation.
    # The contact with penetration depth smaller than this value is ignored in the differentiable formulation.
    # This should be large enough to be safe from numerical errors, because in the backward pass, the computed
    # penetration depth could be different from the forward pass due to the numerical errors. If this value is
    # too small, the non-zero penetration depth could be falsely computed to 0 in the backward pass and thus
    # produce nan values for the contact normal.
    diff_contact_min_penetration: V_ANNOTATION


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
        polytope_max_reprojection_error=V_SCALAR_FROM(
            dtype=gs.qd_float, value=kwargs["polytope_max_reprojection_error"]
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
class StructGJKStaticConfig(metaclass=AutoInitMeta):
    # This is disabled by default, because it is often less stable than the other multi-contact detection algorithm.
    # However, we keep the code here for compatibility with MuJoCo and for possible future use.
    enable_mujoco_multi_contact: bool


# =========================================== SupportField ===========================================


@DATA_ORIENTED
class StructSupportFieldInfo(metaclass=BASE_METACLASS):
    support_cell_start: V_ANNOTATION
    support_v: V_ANNOTATION
    support_vid: V_ANNOTATION
    support_res: V_ANNOTATION


def get_support_field_info(n_geoms, n_support_cells, support_res):
    return StructSupportFieldInfo(
        support_cell_start=V(dtype=gs.qd_int, shape=(max(n_geoms, 1),)),
        support_v=V_VEC(3, dtype=gs.qd_float, shape=(max(n_support_cells, 1),)),
        support_vid=V(dtype=gs.qd_int, shape=(max(n_support_cells, 1),)),
        support_res=V_SCALAR_FROM(dtype=gs.qd_int, value=support_res),
    )


# =========================================== SDF ===========================================


@DATA_ORIENTED
class StructSDFGeomInfo(metaclass=BASE_METACLASS):
    T_mesh_to_sdf: V_ANNOTATION
    sdf_res: V_ANNOTATION
    sdf_max: V_ANNOTATION
    sdf_cell_size: V_ANNOTATION
    sdf_cell_start: V_ANNOTATION


def get_sdf_geom_info(n_geoms):
    return StructSDFGeomInfo(
        T_mesh_to_sdf=V_MAT(n=4, m=4, dtype=gs.qd_float, shape=(n_geoms,)),
        sdf_res=V_VEC(3, dtype=gs.qd_int, shape=(n_geoms,)),
        sdf_max=V(dtype=gs.qd_float, shape=(n_geoms,)),
        sdf_cell_size=V(dtype=gs.qd_float, shape=(n_geoms,)),
        sdf_cell_start=V(dtype=gs.qd_int, shape=(n_geoms,)),
    )


@DATA_ORIENTED
class StructSDFInfo(metaclass=BASE_METACLASS):
    geoms_info: StructSDFGeomInfo
    geoms_sdf_start: V_ANNOTATION
    geoms_sdf_val: V_ANNOTATION
    geoms_sdf_grad: V_ANNOTATION
    geoms_sdf_closest_vert: V_ANNOTATION


def get_sdf_info(n_geoms, n_cells):
    if math.prod((n_cells, 3)) > np.iinfo(np.int32).max:
        gs.raise_exception(
            f"SDF Gradient shape (n_cells={n_cells}, 3) is too large. Consider manually setting larger "
            "'sdf_cell_size' in 'gs.materials.Rigid' options."
        )

    return StructSDFInfo(
        geoms_info=get_sdf_geom_info(max(n_geoms, 1)),
        geoms_sdf_start=V(dtype=gs.qd_int, shape=(max(n_geoms, 1),)),
        geoms_sdf_val=V(dtype=gs.qd_float, shape=(max(n_cells, 1),)),
        geoms_sdf_grad=V_VEC(3, dtype=gs.qd_float, shape=(max(n_cells, 1),)),
        geoms_sdf_closest_vert=V(dtype=gs.qd_int, shape=(max(n_cells, 1),)),
    )


# =========================================== DofsInfo and DofsState ===========================================


@DATA_ORIENTED
class StructDofsInfo(metaclass=BASE_METACLASS):
    entity_idx: V_ANNOTATION
    stiffness: V_ANNOTATION
    invweight: V_ANNOTATION
    armature: V_ANNOTATION
    damping: V_ANNOTATION
    frictionloss: V_ANNOTATION
    motion_ang: V_ANNOTATION
    motion_vel: V_ANNOTATION
    limit: V_ANNOTATION
    kp: V_ANNOTATION
    kv: V_ANNOTATION
    force_range: V_ANNOTATION


def get_dofs_info(solver):
    shape = (solver.n_dofs_, solver._B) if solver._options.batch_dofs_info else (solver.n_dofs_,)

    return StructDofsInfo(
        entity_idx=V(dtype=gs.qd_int, shape=shape),
        stiffness=V(dtype=gs.qd_float, shape=shape),
        invweight=V(dtype=gs.qd_float, shape=shape),
        armature=V(dtype=gs.qd_float, shape=shape),
        damping=V(dtype=gs.qd_float, shape=shape),
        frictionloss=V(dtype=gs.qd_float, shape=shape),
        motion_ang=V(dtype=gs.qd_vec3, shape=shape),
        motion_vel=V(dtype=gs.qd_vec3, shape=shape),
        limit=V(dtype=gs.qd_vec2, shape=shape),
        kp=V(dtype=gs.qd_float, shape=shape),
        kv=V(dtype=gs.qd_float, shape=shape),
        force_range=V(dtype=gs.qd_vec2, shape=shape),
    )


@DATA_ORIENTED
class StructDofsState(metaclass=BASE_METACLASS):
    # *_bw: Cache to avoid overwriting for backward pass
    force: V_ANNOTATION
    qf_bias: V_ANNOTATION
    qf_passive: V_ANNOTATION
    qf_actuator: V_ANNOTATION
    qf_applied: V_ANNOTATION
    act_length: V_ANNOTATION
    pos: V_ANNOTATION
    vel: V_ANNOTATION
    vel_prev: V_ANNOTATION
    vel_next: V_ANNOTATION
    acc: V_ANNOTATION
    acc_bw: V_ANNOTATION
    acc_smooth: V_ANNOTATION
    acc_smooth_bw: V_ANNOTATION
    qf_smooth: V_ANNOTATION
    qf_constraint: V_ANNOTATION
    cdof_ang: V_ANNOTATION
    cdof_vel: V_ANNOTATION
    cdofvel_ang: V_ANNOTATION
    cdofvel_vel: V_ANNOTATION
    cdofd_ang: V_ANNOTATION
    cdofd_vel: V_ANNOTATION
    f_vel: V_ANNOTATION
    f_ang: V_ANNOTATION
    ctrl_force: V_ANNOTATION
    ctrl_pos: V_ANNOTATION
    ctrl_vel: V_ANNOTATION
    ctrl_mode: V_ANNOTATION
    hibernated: V_ANNOTATION


def get_dofs_state(solver):
    shape = (solver.n_dofs_, solver._B)
    requires_grad = solver._requires_grad
    shape_bw = maybe_shape((2, *shape), requires_grad)

    return StructDofsState(
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


@DATA_ORIENTED
class StructLinksState(metaclass=BASE_METACLASS):
    # *_bw: Cache to avoid overwriting for backward pass
    cinr_inertial: V_ANNOTATION
    cinr_pos: V_ANNOTATION
    cinr_quat: V_ANNOTATION
    cinr_mass: V_ANNOTATION
    crb_inertial: V_ANNOTATION
    crb_pos: V_ANNOTATION
    crb_quat: V_ANNOTATION
    crb_mass: V_ANNOTATION
    cdd_vel: V_ANNOTATION
    cdd_ang: V_ANNOTATION
    pos: V_ANNOTATION
    quat: V_ANNOTATION
    pos_bw: V_ANNOTATION
    quat_bw: V_ANNOTATION
    i_pos: V_ANNOTATION
    i_pos_bw: V_ANNOTATION
    i_quat: V_ANNOTATION
    j_pos: V_ANNOTATION
    j_quat: V_ANNOTATION
    j_pos_bw: V_ANNOTATION
    j_quat_bw: V_ANNOTATION
    j_vel: V_ANNOTATION
    j_ang: V_ANNOTATION
    cd_ang: V_ANNOTATION
    cd_vel: V_ANNOTATION
    cd_ang_bw: V_ANNOTATION
    cd_vel_bw: V_ANNOTATION
    mass_sum: V_ANNOTATION
    root_COM: V_ANNOTATION  # COM of the kinematic tree
    root_COM_bw: V_ANNOTATION
    mass_shift: V_ANNOTATION
    i_pos_shift: V_ANNOTATION
    cacc_ang: V_ANNOTATION
    cacc_lin: V_ANNOTATION
    cfrc_ang: V_ANNOTATION
    cfrc_vel: V_ANNOTATION
    cfrc_applied_ang: V_ANNOTATION
    cfrc_applied_vel: V_ANNOTATION
    cfrc_coupling_ang: V_ANNOTATION
    cfrc_coupling_vel: V_ANNOTATION
    contact_force: V_ANNOTATION
    hibernated: V_ANNOTATION


def get_links_state(solver):
    max_n_joints_per_link = solver._static_rigid_sim_config.max_n_joints_per_link
    shape = (solver.n_links_, solver._B)
    requires_grad = solver._requires_grad
    shape_bw = (solver.n_links_, max(max_n_joints_per_link + 1, 1), solver._B)

    return StructLinksState(
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


@DATA_ORIENTED
class StructLinksInfo(metaclass=BASE_METACLASS):
    parent_idx: V_ANNOTATION
    root_idx: V_ANNOTATION
    q_start: V_ANNOTATION
    dof_start: V_ANNOTATION
    joint_start: V_ANNOTATION
    q_end: V_ANNOTATION
    dof_end: V_ANNOTATION
    joint_end: V_ANNOTATION
    n_dofs: V_ANNOTATION
    pos: V_ANNOTATION
    quat: V_ANNOTATION
    invweight: V_ANNOTATION
    is_fixed: V_ANNOTATION
    inertial_pos: V_ANNOTATION
    inertial_quat: V_ANNOTATION
    inertial_i: V_ANNOTATION
    inertial_mass: V_ANNOTATION
    entity_idx: V_ANNOTATION
    # Heterogeneous simulation support: per-link geom/vgeom index ranges
    geom_start: V_ANNOTATION
    geom_end: V_ANNOTATION
    vgeom_start: V_ANNOTATION
    vgeom_end: V_ANNOTATION


def get_links_info(solver):
    links_info_shape = (solver.n_links_, solver._B) if solver._options.batch_links_info else solver.n_links_

    return StructLinksInfo(
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


@DATA_ORIENTED
class StructJointsInfo(metaclass=BASE_METACLASS):
    type: V_ANNOTATION
    sol_params: V_ANNOTATION
    q_start: V_ANNOTATION
    dof_start: V_ANNOTATION
    q_end: V_ANNOTATION
    dof_end: V_ANNOTATION
    n_dofs: V_ANNOTATION
    pos: V_ANNOTATION


def get_joints_info(solver):
    shape = (solver.n_joints_, solver._B) if solver._options.batch_joints_info else (solver.n_joints_,)

    return StructJointsInfo(
        type=V(dtype=gs.qd_int, shape=shape),
        sol_params=V(dtype=gs.qd_vec7, shape=shape),
        q_start=V(dtype=gs.qd_int, shape=shape),
        dof_start=V(dtype=gs.qd_int, shape=shape),
        q_end=V(dtype=gs.qd_int, shape=shape),
        dof_end=V(dtype=gs.qd_int, shape=shape),
        n_dofs=V(dtype=gs.qd_int, shape=shape),
        pos=V(dtype=gs.qd_vec3, shape=shape),
    )


@DATA_ORIENTED
class StructJointsState(metaclass=BASE_METACLASS):
    xanchor: V_ANNOTATION
    xaxis: V_ANNOTATION


def get_joints_state(solver):
    shape = (solver.n_joints_, solver._B)
    requires_grad = solver._requires_grad

    return StructJointsState(
        xanchor=V(dtype=gs.qd_vec3, shape=shape, needs_grad=requires_grad),
        xaxis=V(dtype=gs.qd_vec3, shape=shape, needs_grad=requires_grad),
    )


# =========================================== GeomsInfo and GeomsState ===========================================


@DATA_ORIENTED
class StructGeomsInfo(metaclass=BASE_METACLASS):
    pos: V_ANNOTATION
    center: V_ANNOTATION
    quat: V_ANNOTATION
    data: V_ANNOTATION
    link_idx: V_ANNOTATION
    type: V_ANNOTATION
    friction: V_ANNOTATION
    sol_params: V_ANNOTATION
    vert_num: V_ANNOTATION
    vert_start: V_ANNOTATION
    vert_end: V_ANNOTATION
    verts_state_start: V_ANNOTATION
    verts_state_end: V_ANNOTATION
    face_num: V_ANNOTATION
    face_start: V_ANNOTATION
    face_end: V_ANNOTATION
    edge_num: V_ANNOTATION
    edge_start: V_ANNOTATION
    edge_end: V_ANNOTATION
    is_convex: V_ANNOTATION
    contype: V_ANNOTATION
    conaffinity: V_ANNOTATION
    is_fixed: V_ANNOTATION
    is_decomposed: V_ANNOTATION
    needs_coup: V_ANNOTATION
    coup_friction: V_ANNOTATION
    coup_softness: V_ANNOTATION
    coup_restitution: V_ANNOTATION


def get_geoms_info(solver):
    shape = (solver.n_geoms_,)

    return StructGeomsInfo(
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


@DATA_ORIENTED
class StructGeomsState(metaclass=BASE_METACLASS):
    pos: V_ANNOTATION
    quat: V_ANNOTATION
    aabb_min: V_ANNOTATION
    aabb_max: V_ANNOTATION
    verts_updated: V_ANNOTATION
    min_buffer_idx: V_ANNOTATION
    max_buffer_idx: V_ANNOTATION
    hibernated: V_ANNOTATION
    friction_ratio: V_ANNOTATION


def get_geoms_state(solver):
    shape = (solver.n_geoms_, solver._B)
    requires_grad = solver._static_rigid_sim_config.requires_grad

    return StructGeomsState(
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


@DATA_ORIENTED
class StructVertsInfo(metaclass=BASE_METACLASS):
    init_pos: V_ANNOTATION
    init_normal: V_ANNOTATION
    geom_idx: V_ANNOTATION
    init_center_pos: V_ANNOTATION
    verts_state_idx: V_ANNOTATION
    is_fixed: V_ANNOTATION


def get_verts_info(solver):
    shape = (solver.n_verts_,)

    return StructVertsInfo(
        init_pos=V(dtype=gs.qd_vec3, shape=shape),
        init_normal=V(dtype=gs.qd_vec3, shape=shape),
        geom_idx=V(dtype=gs.qd_int, shape=shape),
        init_center_pos=V(dtype=gs.qd_vec3, shape=shape),
        verts_state_idx=V(dtype=gs.qd_int, shape=shape),
        is_fixed=V(dtype=gs.qd_bool, shape=shape),
    )


# =========================================== FacesInfo ===========================================


@DATA_ORIENTED
class StructFacesInfo(metaclass=BASE_METACLASS):
    verts_idx: V_ANNOTATION
    geom_idx: V_ANNOTATION


def get_faces_info(solver):
    shape = (solver.n_faces_,)

    return StructFacesInfo(
        verts_idx=V(dtype=gs.qd_ivec3, shape=shape),
        geom_idx=V(dtype=gs.qd_int, shape=shape),
    )


# =========================================== EdgesInfo ===========================================


@DATA_ORIENTED
class StructEdgesInfo(metaclass=BASE_METACLASS):
    v0: V_ANNOTATION
    v1: V_ANNOTATION
    length: V_ANNOTATION


def get_edges_info(solver):
    shape = (solver.n_edges_,)

    return StructEdgesInfo(
        v0=V(dtype=gs.qd_int, shape=shape),
        v1=V(dtype=gs.qd_int, shape=shape),
        length=V(dtype=gs.qd_float, shape=shape),
    )


# =========================================== VertsState ===========================================


@DATA_ORIENTED
class StructVertsState(metaclass=BASE_METACLASS):
    pos: V_ANNOTATION


def get_free_verts_state(solver):
    return StructVertsState(
        pos=V(dtype=gs.qd_vec3, shape=(solver.n_free_verts_, solver._B)),
    )


def get_fixed_verts_state(solver):
    return StructVertsState(
        pos=V(dtype=gs.qd_vec3, shape=(solver.n_fixed_verts_,)),
    )


# =========================================== VvertsInfo ===========================================


@DATA_ORIENTED
class StructVvertsInfo(metaclass=BASE_METACLASS):
    init_pos: V_ANNOTATION
    init_vnormal: V_ANNOTATION
    vgeom_idx: V_ANNOTATION


def get_vverts_info(solver):
    shape = (solver.n_vverts_,)

    return StructVvertsInfo(
        init_pos=V(dtype=gs.qd_vec3, shape=shape),
        init_vnormal=V(dtype=gs.qd_vec3, shape=shape),
        vgeom_idx=V(dtype=gs.qd_int, shape=shape),
    )


# =========================================== VfacesInfo ===========================================


@DATA_ORIENTED
class StructVfacesInfo(metaclass=BASE_METACLASS):
    vverts_idx: V_ANNOTATION
    vgeom_idx: V_ANNOTATION


def get_vfaces_info(solver):
    shape = (solver.n_vfaces_,)

    return StructVfacesInfo(
        vverts_idx=V(dtype=gs.qd_ivec3, shape=shape),
        vgeom_idx=V(dtype=gs.qd_int, shape=shape),
    )


# =========================================== VgeomsInfo ===========================================


@DATA_ORIENTED
class StructVgeomsInfo(metaclass=BASE_METACLASS):
    pos: V_ANNOTATION
    quat: V_ANNOTATION
    link_idx: V_ANNOTATION
    vvert_num: V_ANNOTATION
    vvert_start: V_ANNOTATION
    vvert_end: V_ANNOTATION
    vface_num: V_ANNOTATION
    vface_start: V_ANNOTATION
    vface_end: V_ANNOTATION
    color: V_ANNOTATION


def get_vgeoms_info(solver):
    shape = (solver.n_vgeoms_,)

    return StructVgeomsInfo(
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


@DATA_ORIENTED
class StructVgeomsState(metaclass=BASE_METACLASS):
    pos: V_ANNOTATION
    quat: V_ANNOTATION


def get_vgeoms_state(solver):
    shape = (solver.n_vgeoms_, solver._B)

    return StructVgeomsState(
        pos=V(dtype=gs.qd_vec3, shape=shape),
        quat=V(dtype=gs.qd_vec4, shape=shape),
    )


# =========================================== EqualitiesInfo ===========================================


@DATA_ORIENTED
class StructEqualitiesInfo(metaclass=BASE_METACLASS):
    eq_obj1id: V_ANNOTATION
    eq_obj2id: V_ANNOTATION
    eq_data: V_ANNOTATION
    eq_type: V_ANNOTATION
    sol_params: V_ANNOTATION


def get_equalities_info(solver):
    shape = (solver.n_candidate_equalities_, solver._B)

    return StructEqualitiesInfo(
        eq_obj1id=V(dtype=gs.qd_int, shape=shape),
        eq_obj2id=V(dtype=gs.qd_int, shape=shape),
        eq_data=V(dtype=gs.qd_vec11, shape=shape),
        eq_type=V(dtype=gs.qd_int, shape=shape),
        sol_params=V(dtype=gs.qd_vec7, shape=shape),
    )


# =========================================== EntitiesInfo ===========================================


@DATA_ORIENTED
class StructEntitiesInfo(metaclass=BASE_METACLASS):
    dof_start: V_ANNOTATION
    dof_end: V_ANNOTATION
    n_dofs: V_ANNOTATION
    link_start: V_ANNOTATION
    link_end: V_ANNOTATION
    n_links: V_ANNOTATION
    geom_start: V_ANNOTATION
    geom_end: V_ANNOTATION
    n_geoms: V_ANNOTATION
    gravity_compensation: V_ANNOTATION
    is_local_collision_mask: V_ANNOTATION


def get_entities_info(solver):
    shape = (solver.n_entities_,)

    return StructEntitiesInfo(
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


@DATA_ORIENTED
class StructEntitiesState(metaclass=BASE_METACLASS):
    hibernated: V_ANNOTATION


def get_entities_state(solver):
    return StructEntitiesState(
        hibernated=V(dtype=gs.qd_int, shape=(solver.n_entities_, solver._B)),
    )


# =========================================== RigidAdjointCache ===========================================
@DATA_ORIENTED
class StructRigidAdjointCache(metaclass=BASE_METACLASS):
    # This cache stores intermediate values during rigid body simulation to use Quadrants's AD. Quadrants's AD requires
    # us not to overwrite the values that have been read during the forward pass, so we need to store the intemediate
    # values in this cache to avoid overwriting them. Specifically, after we compute next frame's qpos, dofs_vel, and
    # dofs_acc, we need to store them in this cache because we overwrite the values in the next frame. See how
    # [kernel_save_adjoint_cache] is used in [rigid_solver.py] to store the values in this cache.
    qpos: V_ANNOTATION
    dofs_vel: V_ANNOTATION
    dofs_acc: V_ANNOTATION


def get_rigid_adjoint_cache(solver):
    substeps_local = solver._sim.substeps_local
    requires_grad = solver._requires_grad

    return StructRigidAdjointCache(
        qpos=V(dtype=gs.qd_float, shape=(substeps_local + 1, solver.n_qs_, solver._B), needs_grad=requires_grad),
        dofs_vel=V(dtype=gs.qd_float, shape=(substeps_local + 1, solver.n_dofs_, solver._B), needs_grad=requires_grad),
        dofs_acc=V(dtype=gs.qd_float, shape=(substeps_local + 1, solver.n_dofs_, solver._B), needs_grad=requires_grad),
    )


# =================================== StructRigidSimStaticConfig ===================================


@qd.data_oriented
class StructRigidSimStaticConfig(metaclass=AutoInitMeta):
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
    def __init__(self, solver):
        self.rigid_global_info = get_rigid_global_info(solver)

        self.dofs_info = get_dofs_info(solver)
        self.dofs_state = get_dofs_state(solver)
        self.links_info = get_links_info(solver)
        self.links_state = get_links_state(solver)
        self.joints_info = get_joints_info(solver)
        self.joints_state = get_joints_state(solver)
        self.geoms_info = get_geoms_info(solver)
        self.geoms_state = get_geoms_state(solver)

        self.verts_info = get_verts_info(solver)
        self.faces_info = get_faces_info(solver)
        self.edges_info = get_edges_info(solver)

        self.free_verts_state = get_free_verts_state(solver)
        self.fixed_verts_state = get_fixed_verts_state(solver)

        self.vverts_info = get_vverts_info(solver)
        self.vfaces_info = get_vfaces_info(solver)

        self.vgeoms_info = get_vgeoms_info(solver)
        self.vgeoms_state = get_vgeoms_state(solver)

        self.equalities_info = get_equalities_info(solver)

        self.entities_info = get_entities_info(solver)
        self.entities_state = get_entities_state(solver)

        if solver._static_rigid_sim_config.requires_grad:
            # Data structures required for backward pass
            self.dofs_state_adjoint_cache = get_dofs_state(solver)
            self.links_state_adjoint_cache = get_links_state(solver)
            self.joints_state_adjoint_cache = get_joints_state(solver)
            self.geoms_state_adjoint_cache = get_geoms_state(solver)

        self.rigid_adjoint_cache = get_rigid_adjoint_cache(solver)
        self.errno = V(dtype=gs.qd_int, shape=(solver._B,))


# =========================================== ViewerRaycastResult ===========================================


@DATA_ORIENTED
class StructViewerRaycastResult(metaclass=BASE_METACLASS):
    distance: V_ANNOTATION
    geom_idx: V_ANNOTATION
    hit_point: V_ANNOTATION
    normal: V_ANNOTATION
    env_idx: V_ANNOTATION


def get_viewer_raycast_result():
    return StructViewerRaycastResult(
        distance=V(dtype=gs.qd_float, shape=()),
        geom_idx=V(dtype=gs.qd_int, shape=()),
        hit_point=V_VEC(3, dtype=gs.qd_float, shape=()),
        normal=V_VEC(3, dtype=gs.qd_float, shape=()),
        env_idx=V(dtype=gs.qd_int, shape=()),
    )


DofsState = StructDofsState if gs.use_ndarray else qd.template()
DofsInfo = StructDofsInfo if gs.use_ndarray else qd.template()
GeomsState = StructGeomsState if gs.use_ndarray else qd.template()
GeomsInfo = StructGeomsInfo if gs.use_ndarray else qd.template()
GeomsInitAABB = V_ANNOTATION
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
