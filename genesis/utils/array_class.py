import dataclasses
import math
from enum import IntEnum
from functools import partial
from typing import TYPE_CHECKING

import quadrants as qd
import numpy as np
import torch
from typing_extensions import dataclass_transform  # Made it into standard lib from Python 3.12

import genesis as gs

if not gs._initialized:
    gs.raise_exception("Genesis hasn't been initialized. Did you call `gs.init()`?")


if TYPE_CHECKING:
    _STRUCT_FIELD_ANNOTATION = qd.Tensor | qd.Field | qd.Ndarray
    DATA_ORIENTED = dataclasses.dataclass
else:
    _STRUCT_FIELD_ANNOTATION = qd.types.ndarray() if gs.use_ndarray else qd.template
    DATA_ORIENTED = partial(dataclasses.dataclass, frozen=True) if gs.use_ndarray else qd.data_oriented

_TENSOR_BACKEND = qd.Backend.NDARRAY if gs.use_ndarray else qd.Backend.FIELD


PLACEHOLDER = qd.tensor(gs.qd_float, (), backend=_TENSOR_BACKEND)


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


@DATA_ORIENTED
class StructRigidGlobalInfo(metaclass=BASE_METACLASS):
    # *_bw: Cache for backward pass
    n_awake_dofs: _STRUCT_FIELD_ANNOTATION
    awake_dofs: _STRUCT_FIELD_ANNOTATION
    n_awake_entities: _STRUCT_FIELD_ANNOTATION
    awake_entities: _STRUCT_FIELD_ANNOTATION
    n_awake_links: _STRUCT_FIELD_ANNOTATION
    awake_links: _STRUCT_FIELD_ANNOTATION
    qpos0: _STRUCT_FIELD_ANNOTATION
    qpos: _STRUCT_FIELD_ANNOTATION
    qpos_next: _STRUCT_FIELD_ANNOTATION
    links_T: _STRUCT_FIELD_ANNOTATION
    envs_offset: _STRUCT_FIELD_ANNOTATION
    geoms_init_AABB: _STRUCT_FIELD_ANNOTATION
    mass_mat: _STRUCT_FIELD_ANNOTATION
    mass_mat_L: _STRUCT_FIELD_ANNOTATION
    mass_mat_L_bw: _STRUCT_FIELD_ANNOTATION
    mass_mat_D_inv: _STRUCT_FIELD_ANNOTATION
    mass_mat_mask: _STRUCT_FIELD_ANNOTATION
    meaninertia: _STRUCT_FIELD_ANNOTATION
    mass_parent_mask: _STRUCT_FIELD_ANNOTATION
    gravity: _STRUCT_FIELD_ANNOTATION
    # Runtime constants
    substep_dt: _STRUCT_FIELD_ANNOTATION
    iterations: _STRUCT_FIELD_ANNOTATION
    tolerance: _STRUCT_FIELD_ANNOTATION
    ls_iterations: _STRUCT_FIELD_ANNOTATION
    ls_tolerance: _STRUCT_FIELD_ANNOTATION
    noslip_iterations: _STRUCT_FIELD_ANNOTATION
    noslip_tolerance: _STRUCT_FIELD_ANNOTATION
    n_equalities: _STRUCT_FIELD_ANNOTATION
    n_candidate_equalities: _STRUCT_FIELD_ANNOTATION
    hibernation_thresh_acc: _STRUCT_FIELD_ANNOTATION
    hibernation_thresh_vel: _STRUCT_FIELD_ANNOTATION
    EPS: _STRUCT_FIELD_ANNOTATION


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


@DATA_ORIENTED
class StructConstraintState(metaclass=BASE_METACLASS):
    is_warmstart: _STRUCT_FIELD_ANNOTATION
    n_constraints: _STRUCT_FIELD_ANNOTATION
    qd_n_equalities: _STRUCT_FIELD_ANNOTATION
    jac: _STRUCT_FIELD_ANNOTATION
    diag: _STRUCT_FIELD_ANNOTATION
    aref: _STRUCT_FIELD_ANNOTATION
    jac_relevant_dofs: _STRUCT_FIELD_ANNOTATION
    jac_n_relevant_dofs: _STRUCT_FIELD_ANNOTATION
    n_constraints_equality: _STRUCT_FIELD_ANNOTATION
    n_constraints_frictionloss: _STRUCT_FIELD_ANNOTATION
    improved: _STRUCT_FIELD_ANNOTATION
    Jaref: _STRUCT_FIELD_ANNOTATION
    Ma: _STRUCT_FIELD_ANNOTATION
    Ma_ws: _STRUCT_FIELD_ANNOTATION
    grad: _STRUCT_FIELD_ANNOTATION
    Mgrad: _STRUCT_FIELD_ANNOTATION
    MinvJT: _STRUCT_FIELD_ANNOTATION
    search: _STRUCT_FIELD_ANNOTATION
    efc_D: _STRUCT_FIELD_ANNOTATION
    efc_frictionloss: _STRUCT_FIELD_ANNOTATION
    efc_force: _STRUCT_FIELD_ANNOTATION
    efc_b: _STRUCT_FIELD_ANNOTATION
    efc_AR: _STRUCT_FIELD_ANNOTATION
    active: _STRUCT_FIELD_ANNOTATION
    prev_active: _STRUCT_FIELD_ANNOTATION
    qfrc_constraint: _STRUCT_FIELD_ANNOTATION
    qacc: _STRUCT_FIELD_ANNOTATION
    qacc_ws: _STRUCT_FIELD_ANNOTATION
    qacc_prev: _STRUCT_FIELD_ANNOTATION
    cost_ws: _STRUCT_FIELD_ANNOTATION
    gauss: _STRUCT_FIELD_ANNOTATION
    cost: _STRUCT_FIELD_ANNOTATION
    prev_cost: _STRUCT_FIELD_ANNOTATION
    gtol: _STRUCT_FIELD_ANNOTATION
    mv: _STRUCT_FIELD_ANNOTATION
    jv: _STRUCT_FIELD_ANNOTATION
    quad_gauss: _STRUCT_FIELD_ANNOTATION
    ls_alpha: _STRUCT_FIELD_ANNOTATION
    ls_p0_cost: _STRUCT_FIELD_ANNOTATION
    ls_alpha_newton: _STRUCT_FIELD_ANNOTATION
    ls_gtol: _STRUCT_FIELD_ANNOTATION
    eq_sum: _STRUCT_FIELD_ANNOTATION
    ls_it: _STRUCT_FIELD_ANNOTATION
    ls_result: _STRUCT_FIELD_ANNOTATION
    # Optional CG fields
    cg_prev_grad: _STRUCT_FIELD_ANNOTATION
    cg_prev_Mgrad: _STRUCT_FIELD_ANNOTATION
    cg_beta: _STRUCT_FIELD_ANNOTATION
    cg_pg_dot_pMg: _STRUCT_FIELD_ANNOTATION
    # Optional Newton fields
    # Hessian matrix of the optimization problem as a dense 2D tensor.
    # Note that only the lower triangular part is updated for efficiency because this matrix is symmetric by definition.
    # As a result, the values of the strictly upper triangular part is undefined.
    # In practice, this variable is re-purposed to store the Cholesky factor L st H = L @ L.T to spare memory resources.
    # TODO: Optimize storage to only allocate memory half of the Hessian matrix to sparse memory resources.
    nt_H: _STRUCT_FIELD_ANNOTATION
    nt_vec: _STRUCT_FIELD_ANNOTATION
    # Compacted list of constraints whose active state changed, used by incremental Cholesky update
    # to reduce GPU thread divergence by iterating only over constraints that need processing.
    incr_changed_idx: _STRUCT_FIELD_ANNOTATION
    incr_n_changed: _STRUCT_FIELD_ANNOTATION
    # Backward gradients
    dL_dqacc: _STRUCT_FIELD_ANNOTATION
    dL_dM: _STRUCT_FIELD_ANNOTATION
    dL_djac: _STRUCT_FIELD_ANNOTATION
    dL_daref: _STRUCT_FIELD_ANNOTATION
    dL_defc_D: _STRUCT_FIELD_ANNOTATION
    dL_dforce: _STRUCT_FIELD_ANNOTATION
    # Backward buffers for linear system solver
    bw_u: _STRUCT_FIELD_ANNOTATION
    bw_r: _STRUCT_FIELD_ANNOTATION
    bw_p: _STRUCT_FIELD_ANNOTATION
    bw_Ap: _STRUCT_FIELD_ANNOTATION
    bw_Ju: _STRUCT_FIELD_ANNOTATION
    bw_y: _STRUCT_FIELD_ANNOTATION
    bw_w: _STRUCT_FIELD_ANNOTATION
    # Timers for profiling
    timers: _STRUCT_FIELD_ANNOTATION
    # Per-env flag: 0 = use incremental Hessian+Cholesky, 1 = use full tiled rebuild
    use_full_hessian: _STRUCT_FIELD_ANNOTATION
    # Solver loop iteration counter (0-indexed, increments each iteration in the graph loop)
    solver_iter_counter: _STRUCT_FIELD_ANNOTATION
    # Always ndarray (not field): graph_do_while requires the same physical ndarray on every call.
    graph_counter: qd.types.ndarray()
    early_exit_flag: _STRUCT_FIELD_ANNOTATION


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


@DATA_ORIENTED
class StructContactData(metaclass=BASE_METACLASS):
    geom_a: _STRUCT_FIELD_ANNOTATION
    geom_b: _STRUCT_FIELD_ANNOTATION
    penetration: _STRUCT_FIELD_ANNOTATION
    normal: _STRUCT_FIELD_ANNOTATION
    pos: _STRUCT_FIELD_ANNOTATION
    friction: _STRUCT_FIELD_ANNOTATION
    sol_params: _STRUCT_FIELD_ANNOTATION
    force: _STRUCT_FIELD_ANNOTATION
    link_a: _STRUCT_FIELD_ANNOTATION
    link_b: _STRUCT_FIELD_ANNOTATION
    pair_idx: _STRUCT_FIELD_ANNOTATION


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


@DATA_ORIENTED
class StructDiffContactInput(metaclass=BASE_METACLASS):
    ### Non-differentiable input data
    # Geom id of the two geometries
    geom_a: _STRUCT_FIELD_ANNOTATION
    geom_b: _STRUCT_FIELD_ANNOTATION
    # Local positions of the 3 vertices from the two geometries that define the face on the Minkowski difference
    local_pos1_a: _STRUCT_FIELD_ANNOTATION
    local_pos1_b: _STRUCT_FIELD_ANNOTATION
    local_pos1_c: _STRUCT_FIELD_ANNOTATION
    local_pos2_a: _STRUCT_FIELD_ANNOTATION
    local_pos2_b: _STRUCT_FIELD_ANNOTATION
    local_pos2_c: _STRUCT_FIELD_ANNOTATION
    # Local positions of the 1 vertex from the two geometries that define the support point for the face above
    w_local_pos1: _STRUCT_FIELD_ANNOTATION
    w_local_pos2: _STRUCT_FIELD_ANNOTATION
    # Reference id of the contact point, which is needed for the backward pass
    ref_id: _STRUCT_FIELD_ANNOTATION
    # Flag whether the contact data can be computed in numerically stable way in both the forward and backward passes
    valid: _STRUCT_FIELD_ANNOTATION
    ### Differentiable input data
    # Reference penetration depth, which is needed for computing the weight of the contact point
    ref_penetration: _STRUCT_FIELD_ANNOTATION


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


@DATA_ORIENTED
class StructSortBuffer(metaclass=BASE_METACLASS):
    value: _STRUCT_FIELD_ANNOTATION
    i_g: _STRUCT_FIELD_ANNOTATION
    is_max: _STRUCT_FIELD_ANNOTATION


def get_sort_buffer(solver):
    _B = solver._B

    return StructSortBuffer(
        value=qd.tensor(gs.qd_float, (2 * solver.n_geoms_, _B), backend=_TENSOR_BACKEND),
        i_g=qd.tensor(gs.qd_int, (2 * solver.n_geoms_, _B), backend=_TENSOR_BACKEND),
        is_max=qd.tensor(gs.qd_bool, (2 * solver.n_geoms_, _B), backend=_TENSOR_BACKEND),
    )


@DATA_ORIENTED
class StructContactCache(metaclass=BASE_METACLASS):
    normal: _STRUCT_FIELD_ANNOTATION


def get_contact_cache(solver, n_possible_pairs):
    _B = solver._B
    return StructContactCache(
        normal=qd.Vector.tensor(3, gs.qd_float, (n_possible_pairs, _B), backend=_TENSOR_BACKEND),
    )


@DATA_ORIENTED
class StructAggList(metaclass=BASE_METACLASS):
    curr: _STRUCT_FIELD_ANNOTATION
    n: _STRUCT_FIELD_ANNOTATION
    start: _STRUCT_FIELD_ANNOTATION


def get_agg_list(solver):
    _B = solver._B
    n_entities = max(solver.n_entities, 1)

    return StructAggList(
        curr=qd.tensor(gs.qd_int, (n_entities, _B), backend=_TENSOR_BACKEND),
        n=qd.tensor(gs.qd_int, (n_entities, _B), backend=_TENSOR_BACKEND),
        start=qd.tensor(gs.qd_int, (n_entities, _B), backend=_TENSOR_BACKEND),
    )


@DATA_ORIENTED
class StructContactIslandState(metaclass=BASE_METACLASS):
    ci_edges: _STRUCT_FIELD_ANNOTATION
    edge_id: _STRUCT_FIELD_ANNOTATION
    constraint_list: _STRUCT_FIELD_ANNOTATION
    constraint_id: _STRUCT_FIELD_ANNOTATION
    entity_edge: StructAggList
    island_col: StructAggList
    island_hibernated: _STRUCT_FIELD_ANNOTATION
    island_entity: StructAggList
    entity_id: _STRUCT_FIELD_ANNOTATION
    n_edges: _STRUCT_FIELD_ANNOTATION
    n_islands: _STRUCT_FIELD_ANNOTATION
    n_stack: _STRUCT_FIELD_ANNOTATION
    entity_island: _STRUCT_FIELD_ANNOTATION
    stack: _STRUCT_FIELD_ANNOTATION
    entity_idx_to_next_entity_idx_in_hibernated_island: _STRUCT_FIELD_ANNOTATION


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


@DATA_ORIENTED
class StructNarrowphaseWorkQueues(metaclass=BASE_METACLASS):
    mpr_i_b: _STRUCT_FIELD_ANNOTATION
    mpr_i_ga: _STRUCT_FIELD_ANNOTATION
    mpr_i_gb: _STRUCT_FIELD_ANNOTATION
    mpr_i_pair: _STRUCT_FIELD_ANNOTATION
    mpr_contact_pos_0: _STRUCT_FIELD_ANNOTATION
    mpr_normal_0: _STRUCT_FIELD_ANNOTATION
    mpr_penetration_0: _STRUCT_FIELD_ANNOTATION
    gjk_i_b: _STRUCT_FIELD_ANNOTATION
    gjk_i_ga: _STRUCT_FIELD_ANNOTATION
    gjk_i_gb: _STRUCT_FIELD_ANNOTATION
    gjk_i_pair: _STRUCT_FIELD_ANNOTATION
    gjk_contact_pos_0: _STRUCT_FIELD_ANNOTATION
    gjk_normal_0: _STRUCT_FIELD_ANNOTATION
    gjk_penetration_0: _STRUCT_FIELD_ANNOTATION
    mpr_queue_size: _STRUCT_FIELD_ANNOTATION
    gjk_queue_size: _STRUCT_FIELD_ANNOTATION
    gjk_queue_size_k2: _STRUCT_FIELD_ANNOTATION
    mpr_work_counter: _STRUCT_FIELD_ANNOTATION
    gjk_work_counter: _STRUCT_FIELD_ANNOTATION


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


@DATA_ORIENTED
class StructColliderState(metaclass=BASE_METACLASS):
    sort_buffer: StructSortBuffer
    contact_data: StructContactData
    active_buffer: _STRUCT_FIELD_ANNOTATION
    n_broad_pairs: _STRUCT_FIELD_ANNOTATION
    broad_collision_pairs: _STRUCT_FIELD_ANNOTATION
    active_buffer_awake: _STRUCT_FIELD_ANNOTATION
    active_buffer_hib: _STRUCT_FIELD_ANNOTATION
    box_depth: _STRUCT_FIELD_ANNOTATION
    box_points: _STRUCT_FIELD_ANNOTATION
    box_pts: _STRUCT_FIELD_ANNOTATION
    box_lines: _STRUCT_FIELD_ANNOTATION
    box_linesu: _STRUCT_FIELD_ANNOTATION
    box_axi: _STRUCT_FIELD_ANNOTATION
    box_ppts2: _STRUCT_FIELD_ANNOTATION
    box_pu: _STRUCT_FIELD_ANNOTATION
    xyz_max_min: _STRUCT_FIELD_ANNOTATION
    prism: _STRUCT_FIELD_ANNOTATION
    n_contacts: _STRUCT_FIELD_ANNOTATION
    n_contacts_hibernated: _STRUCT_FIELD_ANNOTATION
    first_time: _STRUCT_FIELD_ANNOTATION
    contact_cache: StructContactCache
    # Input data for differentiable contact detection used in the backward pass
    diff_contact_input: StructDiffContactInput
    narrowphase_work_queues: StructNarrowphaseWorkQueues
    contact_sort_key: _STRUCT_FIELD_ANNOTATION
    contact_sort_idx: _STRUCT_FIELD_ANNOTATION


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


@DATA_ORIENTED
class StructColliderInfo(metaclass=BASE_METACLASS):
    vert_neighbors: _STRUCT_FIELD_ANNOTATION
    vert_neighbor_start: _STRUCT_FIELD_ANNOTATION
    vert_n_neighbors: _STRUCT_FIELD_ANNOTATION
    # (i_ga, i_gb) -> dense pair index, or -1 if invalid. Used by SAP broadphase, narrowphase, and contact cache.
    collision_pair_idx: _STRUCT_FIELD_ANNOTATION
    max_possible_pairs: _STRUCT_FIELD_ANNOTATION
    max_collision_pairs: _STRUCT_FIELD_ANNOTATION
    max_contact_pairs: _STRUCT_FIELD_ANNOTATION
    max_collision_pairs_broad: _STRUCT_FIELD_ANNOTATION
    # Compact list of valid collision pairs. Used by all-vs-all broadphase to dispatch valid pairs to GPU threads.
    n_valid_pairs: _STRUCT_FIELD_ANNOTATION
    valid_collision_pairs: _STRUCT_FIELD_ANNOTATION
    # Terrain fields
    terrain_hf: _STRUCT_FIELD_ANNOTATION
    terrain_rc: _STRUCT_FIELD_ANNOTATION
    terrain_scale: _STRUCT_FIELD_ANNOTATION
    terrain_xyz_maxmin: _STRUCT_FIELD_ANNOTATION
    # multi contact perturbation and tolerance
    mc_perturbation: _STRUCT_FIELD_ANNOTATION
    mc_tolerance: _STRUCT_FIELD_ANNOTATION
    mpr_to_gjk_overlap_ratio: _STRUCT_FIELD_ANNOTATION
    # differentiable contact tolerance
    diff_pos_tolerance: _STRUCT_FIELD_ANNOTATION
    diff_normal_tolerance: _STRUCT_FIELD_ANNOTATION


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
class StructColliderStaticConfig(metaclass=AutoInitMeta):
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


@DATA_ORIENTED
class StructMPRSimplexSupport(metaclass=BASE_METACLASS):
    v1: _STRUCT_FIELD_ANNOTATION
    v2: _STRUCT_FIELD_ANNOTATION
    v: _STRUCT_FIELD_ANNOTATION


def get_mpr_simplex_support(B_):
    return StructMPRSimplexSupport(
        v1=qd.Vector.tensor(3, gs.qd_float, (4, B_), backend=_TENSOR_BACKEND),
        v2=qd.Vector.tensor(3, gs.qd_float, (4, B_), backend=_TENSOR_BACKEND),
        v=qd.Vector.tensor(3, gs.qd_float, (4, B_), backend=_TENSOR_BACKEND),
    )


@DATA_ORIENTED
class StructMPRState(metaclass=BASE_METACLASS):
    simplex_support: StructMPRSimplexSupport
    simplex_size: _STRUCT_FIELD_ANNOTATION


def get_mpr_state(B_):
    return StructMPRState(
        simplex_support=get_mpr_simplex_support(B_),
        simplex_size=qd.tensor(gs.qd_int, (B_,), backend=_TENSOR_BACKEND),
    )


@DATA_ORIENTED
class StructMPRInfo(metaclass=BASE_METACLASS):
    CCD_EPS: _STRUCT_FIELD_ANNOTATION
    CCD_TOLERANCE: _STRUCT_FIELD_ANNOTATION
    CCD_ITERATIONS: _STRUCT_FIELD_ANNOTATION


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
    obj1: _STRUCT_FIELD_ANNOTATION
    obj2: _STRUCT_FIELD_ANNOTATION
    local_obj1: _STRUCT_FIELD_ANNOTATION
    local_obj2: _STRUCT_FIELD_ANNOTATION
    id1: _STRUCT_FIELD_ANNOTATION
    id2: _STRUCT_FIELD_ANNOTATION
    mink: _STRUCT_FIELD_ANNOTATION


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


@DATA_ORIENTED
class StructGJKSimplex(metaclass=BASE_METACLASS):
    nverts: _STRUCT_FIELD_ANNOTATION
    dist: _STRUCT_FIELD_ANNOTATION


def get_gjk_simplex(_B, is_active):
    shape = maybe_shape((_B,), is_active)
    return StructGJKSimplex(
        nverts=qd.tensor(gs.qd_int, shape, backend=_TENSOR_BACKEND),
        dist=qd.tensor(gs.qd_float, shape, backend=_TENSOR_BACKEND),
    )


@DATA_ORIENTED
class StructGJKSimplexBuffer(metaclass=BASE_METACLASS):
    normal: _STRUCT_FIELD_ANNOTATION
    sdist: _STRUCT_FIELD_ANNOTATION


def get_gjk_simplex_buffer(_B, is_active):
    shape = maybe_shape((_B, 4), is_active)
    return StructGJKSimplexBuffer(
        normal=qd.Vector.tensor(3, gs.qd_float, shape, backend=_TENSOR_BACKEND),
        sdist=qd.tensor(gs.qd_float, shape, backend=_TENSOR_BACKEND),
    )


@DATA_ORIENTED
class StructEPAPolytope(metaclass=BASE_METACLASS):
    nverts: _STRUCT_FIELD_ANNOTATION
    nfaces: _STRUCT_FIELD_ANNOTATION
    nfaces_map: _STRUCT_FIELD_ANNOTATION
    horizon_nedges: _STRUCT_FIELD_ANNOTATION
    horizon_w: _STRUCT_FIELD_ANNOTATION


def get_epa_polytope(_B, is_active):
    shape = maybe_shape((_B,), is_active)
    return StructEPAPolytope(
        nverts=qd.tensor(gs.qd_int, shape, backend=_TENSOR_BACKEND),
        nfaces=qd.tensor(gs.qd_int, shape, backend=_TENSOR_BACKEND),
        nfaces_map=qd.tensor(gs.qd_int, shape, backend=_TENSOR_BACKEND),
        horizon_nedges=qd.tensor(gs.qd_int, shape, backend=_TENSOR_BACKEND),
        horizon_w=qd.Vector.tensor(3, gs.qd_float, shape, backend=_TENSOR_BACKEND),
    )


@DATA_ORIENTED
class StructEPAPolytopeFace(metaclass=BASE_METACLASS):
    verts_idx: _STRUCT_FIELD_ANNOTATION
    adj_idx: _STRUCT_FIELD_ANNOTATION
    normal: _STRUCT_FIELD_ANNOTATION
    dist2: _STRUCT_FIELD_ANNOTATION
    map_idx: _STRUCT_FIELD_ANNOTATION
    visited: _STRUCT_FIELD_ANNOTATION


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


@DATA_ORIENTED
class StructEPAPolytopeHorizonData(metaclass=BASE_METACLASS):
    face_idx: _STRUCT_FIELD_ANNOTATION
    edge_idx: _STRUCT_FIELD_ANNOTATION


def get_epa_polytope_horizon_data(_B, polytope_max_horizons, is_active):
    shape = maybe_shape((_B, polytope_max_horizons), is_active)
    return StructEPAPolytopeHorizonData(
        face_idx=qd.tensor(gs.qd_int, shape, backend=_TENSOR_BACKEND),
        edge_idx=qd.tensor(gs.qd_int, shape, backend=_TENSOR_BACKEND),
    )


@DATA_ORIENTED
class StructContactFace(metaclass=BASE_METACLASS):
    vert1: _STRUCT_FIELD_ANNOTATION
    vert2: _STRUCT_FIELD_ANNOTATION
    endverts: _STRUCT_FIELD_ANNOTATION
    normal1: _STRUCT_FIELD_ANNOTATION
    normal2: _STRUCT_FIELD_ANNOTATION
    id1: _STRUCT_FIELD_ANNOTATION
    id2: _STRUCT_FIELD_ANNOTATION


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


@DATA_ORIENTED
class StructContactNormal(metaclass=BASE_METACLASS):
    endverts: _STRUCT_FIELD_ANNOTATION
    normal: _STRUCT_FIELD_ANNOTATION
    id: _STRUCT_FIELD_ANNOTATION


def get_contact_normal(_B, max_contact_polygon_verts, is_active):
    shape = maybe_shape((_B, max_contact_polygon_verts), is_active)
    return StructContactNormal(
        endverts=qd.Vector.tensor(3, gs.qd_float, shape, backend=_TENSOR_BACKEND),
        normal=qd.Vector.tensor(3, gs.qd_float, shape, backend=_TENSOR_BACKEND),
        id=qd.tensor(gs.qd_int, shape, backend=_TENSOR_BACKEND),
    )


@DATA_ORIENTED
class StructContactHalfspace(metaclass=BASE_METACLASS):
    normal: _STRUCT_FIELD_ANNOTATION
    dist: _STRUCT_FIELD_ANNOTATION


def get_contact_halfspace(_B, max_contact_polygon_verts, is_active):
    shape = maybe_shape((_B, max_contact_polygon_verts), is_active)
    return StructContactHalfspace(
        normal=qd.Vector.tensor(3, gs.qd_float, shape, backend=_TENSOR_BACKEND),
        dist=qd.tensor(gs.qd_float, shape, backend=_TENSOR_BACKEND),
    )


@DATA_ORIENTED
class StructWitness(metaclass=BASE_METACLASS):
    point_obj1: _STRUCT_FIELD_ANNOTATION
    point_obj2: _STRUCT_FIELD_ANNOTATION


def get_witness(_B, max_contacts_per_pair, is_active):
    shape = maybe_shape((_B, max_contacts_per_pair), is_active)
    return StructWitness(
        point_obj1=qd.Vector.tensor(3, gs.qd_float, shape, backend=_TENSOR_BACKEND),
        point_obj2=qd.Vector.tensor(3, gs.qd_float, shape, backend=_TENSOR_BACKEND),
    )


@DATA_ORIENTED
class StructGJKState(metaclass=BASE_METACLASS):
    support_mesh_prev_vertex_id: _STRUCT_FIELD_ANNOTATION
    simplex_vertex: StructMDVertex
    simplex_buffer: StructGJKSimplexBuffer
    simplex: StructGJKSimplex
    simplex_vertex_intersect: StructMDVertex
    simplex_buffer_intersect: StructGJKSimplexBuffer
    nsimplex: _STRUCT_FIELD_ANNOTATION
    last_searched_simplex_vertex_id: _STRUCT_FIELD_ANNOTATION
    polytope: StructEPAPolytope
    polytope_verts: StructMDVertex
    polytope_faces: StructEPAPolytopeFace
    polytope_faces_map: _STRUCT_FIELD_ANNOTATION
    polytope_horizon_data: StructEPAPolytopeHorizonData
    polytope_horizon_stack: StructEPAPolytopeHorizonData
    contact_faces: StructContactFace
    contact_normals: StructContactNormal
    contact_halfspaces: StructContactHalfspace
    contact_clipped_polygons: _STRUCT_FIELD_ANNOTATION
    multi_contact_flag: _STRUCT_FIELD_ANNOTATION
    witness: StructWitness
    n_witness: _STRUCT_FIELD_ANNOTATION
    n_contacts: _STRUCT_FIELD_ANNOTATION
    contact_pos: _STRUCT_FIELD_ANNOTATION
    normal: _STRUCT_FIELD_ANNOTATION
    is_col: _STRUCT_FIELD_ANNOTATION
    penetration: _STRUCT_FIELD_ANNOTATION
    distance: _STRUCT_FIELD_ANNOTATION
    # Differentiable contact detection
    diff_contact_input: StructDiffContactInput
    n_diff_contact_input: _STRUCT_FIELD_ANNOTATION
    diff_penetration: _STRUCT_FIELD_ANNOTATION


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


@DATA_ORIENTED
class StructGJKInfo(metaclass=BASE_METACLASS):
    max_contacts_per_pair: _STRUCT_FIELD_ANNOTATION
    max_contact_polygon_verts: _STRUCT_FIELD_ANNOTATION
    # Maximum number of iterations for GJK and EPA algorithms
    gjk_max_iterations: _STRUCT_FIELD_ANNOTATION
    epa_max_iterations: _STRUCT_FIELD_ANNOTATION
    FLOAT_MIN: _STRUCT_FIELD_ANNOTATION
    FLOAT_MIN_SQ: _STRUCT_FIELD_ANNOTATION
    FLOAT_MAX: _STRUCT_FIELD_ANNOTATION
    FLOAT_MAX_SQ: _STRUCT_FIELD_ANNOTATION
    # Tolerance for stopping GJK and EPA algorithms when they converge (only for non-discrete geometries).
    tolerance: _STRUCT_FIELD_ANNOTATION
    # If the distance between two objects is smaller than this value, we consider them colliding.
    collision_eps: _STRUCT_FIELD_ANNOTATION
    # In safe GJK, we do not allow degenerate simplex to happen, because it becomes the main reason of EPA errors.
    # To prevent degeneracy, we throw away the simplex that has smaller degeneracy measure (e.g. colinearity,
    # coplanarity) than this threshold.
    simplex_max_degeneracy_sq: _STRUCT_FIELD_ANNOTATION
    polytope_max_faces: _STRUCT_FIELD_ANNOTATION
    # Threshold for reprojection error when we compute the witness points from the polytope. In computing the
    # witness points, we project the origin onto the polytope faces and compute the barycentric coordinates of the
    # projected point. To confirm the projection is valid, we compute the projected point using the barycentric
    # coordinates and compare it with the original projected point. If the difference is larger than this threshold,
    # we consider the projection invalid, because it means numerical errors are too large.
    # We check both relative and absolute errors: the relative error catches numerically degenerate faces,
    # while the absolute error prevents false rejections on smooth geometries (e.g. spheres) where
    # polytope faces become extremely small near convergence, amplifying the relative error.
    polytope_max_rel_reprojection_error: _STRUCT_FIELD_ANNOTATION
    polytope_max_abs_reprojection_error: _STRUCT_FIELD_ANNOTATION
    # Tolerance for normal alignment between (face-face) or (edge-face). The normals should align within this
    # tolerance to be considered as a valid parallel contact.
    contact_face_tol: _STRUCT_FIELD_ANNOTATION
    contact_edge_tol: _STRUCT_FIELD_ANNOTATION
    # Epsilon values for differentiable contact. [eps_boundary] denotes the maximum distance between the face
    # and the support point in the direction of the face normal. If this distance is 0, the face is on the
    # boundary of the Minkowski difference. For [eps_distance], the distance between the origin and the face
    # should not exceed this eps value plus the default EPA depth. For [eps_affine], the affine coordinates
    # of the origin's projection onto the face should not violate [0, 1] range by this eps value.
    # FIXME: Adjust these values based on the case study.
    diff_contact_eps_boundary: _STRUCT_FIELD_ANNOTATION
    diff_contact_eps_distance: _STRUCT_FIELD_ANNOTATION
    diff_contact_eps_affine: _STRUCT_FIELD_ANNOTATION
    # The minimum norm of the normal to be considered as a valid normal in the differentiable formulation.
    diff_contact_min_normal_norm: _STRUCT_FIELD_ANNOTATION
    # The minimum penetration depth to be considered as a valid contact in the differentiable formulation.
    # The contact with penetration depth smaller than this value is ignored in the differentiable formulation.
    # This should be large enough to be safe from numerical errors, because in the backward pass, the computed
    # penetration depth could be different from the forward pass due to the numerical errors. If this value is
    # too small, the non-zero penetration depth could be falsely computed to 0 in the backward pass and thus
    # produce nan values for the contact normal.
    diff_contact_min_penetration: _STRUCT_FIELD_ANNOTATION


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
class StructGJKStaticConfig(metaclass=AutoInitMeta):
    # This is disabled by default, because it is often less stable than the other multi-contact detection algorithm.
    # However, we keep the code here for compatibility with MuJoCo and for possible future use.
    enable_mujoco_multi_contact: bool


# =========================================== SupportField ===========================================


@DATA_ORIENTED
class StructSupportFieldInfo(metaclass=BASE_METACLASS):
    support_cell_start: _STRUCT_FIELD_ANNOTATION
    support_v: _STRUCT_FIELD_ANNOTATION
    support_vid: _STRUCT_FIELD_ANNOTATION
    support_res: _STRUCT_FIELD_ANNOTATION


def get_support_field_info(n_geoms, n_support_cells, support_res):
    return StructSupportFieldInfo(
        support_cell_start=qd.tensor(gs.qd_int, (max(n_geoms, 1),), backend=_TENSOR_BACKEND),
        support_v=qd.Vector.tensor(3, gs.qd_float, (max(n_support_cells, 1),), backend=_TENSOR_BACKEND),
        support_vid=qd.tensor(gs.qd_int, (max(n_support_cells, 1),), backend=_TENSOR_BACKEND),
        support_res=V_SCALAR_FROM(dtype=gs.qd_int, value=support_res),
    )


# =========================================== SDF ===========================================


@DATA_ORIENTED
class StructSDFGeomInfo(metaclass=BASE_METACLASS):
    T_mesh_to_sdf: _STRUCT_FIELD_ANNOTATION
    sdf_res: _STRUCT_FIELD_ANNOTATION
    sdf_max: _STRUCT_FIELD_ANNOTATION
    sdf_cell_size: _STRUCT_FIELD_ANNOTATION
    sdf_cell_start: _STRUCT_FIELD_ANNOTATION


def get_sdf_geom_info(n_geoms):
    return StructSDFGeomInfo(
        T_mesh_to_sdf=qd.Matrix.tensor(4, 4, gs.qd_float, (n_geoms,), backend=_TENSOR_BACKEND),
        sdf_res=qd.Vector.tensor(3, gs.qd_int, (n_geoms,), backend=_TENSOR_BACKEND),
        sdf_max=qd.tensor(gs.qd_float, (n_geoms,), backend=_TENSOR_BACKEND),
        sdf_cell_size=qd.tensor(gs.qd_float, (n_geoms,), backend=_TENSOR_BACKEND),
        sdf_cell_start=qd.tensor(gs.qd_int, (n_geoms,), backend=_TENSOR_BACKEND),
    )


@DATA_ORIENTED
class StructSDFInfo(metaclass=BASE_METACLASS):
    geoms_info: StructSDFGeomInfo
    geoms_sdf_start: _STRUCT_FIELD_ANNOTATION
    geoms_sdf_val: _STRUCT_FIELD_ANNOTATION
    geoms_sdf_grad: _STRUCT_FIELD_ANNOTATION
    geoms_sdf_closest_vert: _STRUCT_FIELD_ANNOTATION


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


@DATA_ORIENTED
class StructDofsInfo(metaclass=BASE_METACLASS):
    entity_idx: _STRUCT_FIELD_ANNOTATION
    stiffness: _STRUCT_FIELD_ANNOTATION
    invweight: _STRUCT_FIELD_ANNOTATION
    armature: _STRUCT_FIELD_ANNOTATION
    damping: _STRUCT_FIELD_ANNOTATION
    frictionloss: _STRUCT_FIELD_ANNOTATION
    motion_ang: _STRUCT_FIELD_ANNOTATION
    motion_vel: _STRUCT_FIELD_ANNOTATION
    limit: _STRUCT_FIELD_ANNOTATION
    act_gain: _STRUCT_FIELD_ANNOTATION
    act_bias: _STRUCT_FIELD_ANNOTATION
    force_range: _STRUCT_FIELD_ANNOTATION


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


@DATA_ORIENTED
class StructDofsState(metaclass=BASE_METACLASS):
    # *_bw: Cache to avoid overwriting for backward pass
    force: _STRUCT_FIELD_ANNOTATION
    qf_bias: _STRUCT_FIELD_ANNOTATION
    qf_passive: _STRUCT_FIELD_ANNOTATION
    qf_actuator: _STRUCT_FIELD_ANNOTATION
    qf_applied: _STRUCT_FIELD_ANNOTATION
    act_length: _STRUCT_FIELD_ANNOTATION
    pos: _STRUCT_FIELD_ANNOTATION
    vel: _STRUCT_FIELD_ANNOTATION
    vel_prev: _STRUCT_FIELD_ANNOTATION
    vel_next: _STRUCT_FIELD_ANNOTATION
    acc: _STRUCT_FIELD_ANNOTATION
    acc_bw: _STRUCT_FIELD_ANNOTATION
    acc_smooth: _STRUCT_FIELD_ANNOTATION
    acc_smooth_bw: _STRUCT_FIELD_ANNOTATION
    qf_smooth: _STRUCT_FIELD_ANNOTATION
    qf_constraint: _STRUCT_FIELD_ANNOTATION
    cdof_ang: _STRUCT_FIELD_ANNOTATION
    cdof_vel: _STRUCT_FIELD_ANNOTATION
    cdofvel_ang: _STRUCT_FIELD_ANNOTATION
    cdofvel_vel: _STRUCT_FIELD_ANNOTATION
    cdofd_ang: _STRUCT_FIELD_ANNOTATION
    cdofd_vel: _STRUCT_FIELD_ANNOTATION
    f_vel: _STRUCT_FIELD_ANNOTATION
    f_ang: _STRUCT_FIELD_ANNOTATION
    ctrl_force: _STRUCT_FIELD_ANNOTATION
    ctrl_pos: _STRUCT_FIELD_ANNOTATION
    ctrl_vel: _STRUCT_FIELD_ANNOTATION
    ctrl_mode: _STRUCT_FIELD_ANNOTATION
    hibernated: _STRUCT_FIELD_ANNOTATION


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


@DATA_ORIENTED
class StructLinksState(metaclass=BASE_METACLASS):
    # *_bw: Cache to avoid overwriting for backward pass
    cinr_inertial: _STRUCT_FIELD_ANNOTATION
    cinr_pos: _STRUCT_FIELD_ANNOTATION
    cinr_quat: _STRUCT_FIELD_ANNOTATION
    cinr_mass: _STRUCT_FIELD_ANNOTATION
    crb_inertial: _STRUCT_FIELD_ANNOTATION
    crb_pos: _STRUCT_FIELD_ANNOTATION
    crb_quat: _STRUCT_FIELD_ANNOTATION
    crb_mass: _STRUCT_FIELD_ANNOTATION
    cdd_vel: _STRUCT_FIELD_ANNOTATION
    cdd_ang: _STRUCT_FIELD_ANNOTATION
    pos: _STRUCT_FIELD_ANNOTATION
    quat: _STRUCT_FIELD_ANNOTATION
    pos_bw: _STRUCT_FIELD_ANNOTATION
    quat_bw: _STRUCT_FIELD_ANNOTATION
    i_pos: _STRUCT_FIELD_ANNOTATION
    i_pos_bw: _STRUCT_FIELD_ANNOTATION
    i_quat: _STRUCT_FIELD_ANNOTATION
    j_pos: _STRUCT_FIELD_ANNOTATION
    j_quat: _STRUCT_FIELD_ANNOTATION
    j_pos_bw: _STRUCT_FIELD_ANNOTATION
    j_quat_bw: _STRUCT_FIELD_ANNOTATION
    j_vel: _STRUCT_FIELD_ANNOTATION
    j_ang: _STRUCT_FIELD_ANNOTATION
    cd_ang: _STRUCT_FIELD_ANNOTATION
    cd_vel: _STRUCT_FIELD_ANNOTATION
    cd_ang_bw: _STRUCT_FIELD_ANNOTATION
    cd_vel_bw: _STRUCT_FIELD_ANNOTATION
    mass_sum: _STRUCT_FIELD_ANNOTATION
    root_COM: qd.Tensor  # COM of the kinematic tree
    root_COM_bw: _STRUCT_FIELD_ANNOTATION
    mass_shift: _STRUCT_FIELD_ANNOTATION
    i_pos_shift: _STRUCT_FIELD_ANNOTATION
    cacc_ang: _STRUCT_FIELD_ANNOTATION
    cacc_lin: _STRUCT_FIELD_ANNOTATION
    cfrc_ang: _STRUCT_FIELD_ANNOTATION
    cfrc_vel: _STRUCT_FIELD_ANNOTATION
    cfrc_applied_ang: _STRUCT_FIELD_ANNOTATION
    cfrc_applied_vel: _STRUCT_FIELD_ANNOTATION
    cfrc_coupling_ang: _STRUCT_FIELD_ANNOTATION
    cfrc_coupling_vel: _STRUCT_FIELD_ANNOTATION
    contact_force: _STRUCT_FIELD_ANNOTATION
    hibernated: _STRUCT_FIELD_ANNOTATION


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


@DATA_ORIENTED
class StructLinksInfo(metaclass=BASE_METACLASS):
    parent_idx: _STRUCT_FIELD_ANNOTATION
    root_idx: _STRUCT_FIELD_ANNOTATION
    q_start: _STRUCT_FIELD_ANNOTATION
    dof_start: _STRUCT_FIELD_ANNOTATION
    joint_start: _STRUCT_FIELD_ANNOTATION
    q_end: _STRUCT_FIELD_ANNOTATION
    dof_end: _STRUCT_FIELD_ANNOTATION
    joint_end: _STRUCT_FIELD_ANNOTATION
    n_dofs: _STRUCT_FIELD_ANNOTATION
    pos: _STRUCT_FIELD_ANNOTATION
    quat: _STRUCT_FIELD_ANNOTATION
    invweight: _STRUCT_FIELD_ANNOTATION
    is_fixed: _STRUCT_FIELD_ANNOTATION
    inertial_pos: _STRUCT_FIELD_ANNOTATION
    inertial_quat: _STRUCT_FIELD_ANNOTATION
    inertial_i: _STRUCT_FIELD_ANNOTATION
    inertial_mass: _STRUCT_FIELD_ANNOTATION
    entity_idx: _STRUCT_FIELD_ANNOTATION
    # Heterogeneous simulation support: per-link geom/vgeom index ranges
    geom_start: _STRUCT_FIELD_ANNOTATION
    geom_end: _STRUCT_FIELD_ANNOTATION
    vgeom_start: _STRUCT_FIELD_ANNOTATION
    vgeom_end: _STRUCT_FIELD_ANNOTATION


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


@DATA_ORIENTED
class StructJointsInfo(metaclass=BASE_METACLASS):
    type: _STRUCT_FIELD_ANNOTATION
    sol_params: _STRUCT_FIELD_ANNOTATION
    q_start: _STRUCT_FIELD_ANNOTATION
    dof_start: _STRUCT_FIELD_ANNOTATION
    q_end: _STRUCT_FIELD_ANNOTATION
    dof_end: _STRUCT_FIELD_ANNOTATION
    n_dofs: _STRUCT_FIELD_ANNOTATION
    pos: _STRUCT_FIELD_ANNOTATION


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


@DATA_ORIENTED
class StructJointsState(metaclass=BASE_METACLASS):
    xanchor: _STRUCT_FIELD_ANNOTATION
    xaxis: _STRUCT_FIELD_ANNOTATION


def get_joints_state(solver):
    shape = (solver.n_joints_, solver._B)
    requires_grad = solver._requires_grad

    return StructJointsState(
        xanchor=qd.tensor(gs.qd_vec3, shape, backend=_TENSOR_BACKEND, needs_grad=requires_grad),
        xaxis=qd.tensor(gs.qd_vec3, shape, backend=_TENSOR_BACKEND, needs_grad=requires_grad),
    )


# =========================================== GeomsInfo and GeomsState ===========================================


@DATA_ORIENTED
class StructGeomsInfo(metaclass=BASE_METACLASS):
    pos: _STRUCT_FIELD_ANNOTATION
    center: _STRUCT_FIELD_ANNOTATION
    quat: _STRUCT_FIELD_ANNOTATION
    data: _STRUCT_FIELD_ANNOTATION
    link_idx: _STRUCT_FIELD_ANNOTATION
    type: _STRUCT_FIELD_ANNOTATION
    friction: _STRUCT_FIELD_ANNOTATION
    sol_params: _STRUCT_FIELD_ANNOTATION
    vert_num: _STRUCT_FIELD_ANNOTATION
    vert_start: _STRUCT_FIELD_ANNOTATION
    vert_end: _STRUCT_FIELD_ANNOTATION
    verts_state_start: _STRUCT_FIELD_ANNOTATION
    verts_state_end: _STRUCT_FIELD_ANNOTATION
    face_num: _STRUCT_FIELD_ANNOTATION
    face_start: _STRUCT_FIELD_ANNOTATION
    face_end: _STRUCT_FIELD_ANNOTATION
    edge_num: _STRUCT_FIELD_ANNOTATION
    edge_start: _STRUCT_FIELD_ANNOTATION
    edge_end: _STRUCT_FIELD_ANNOTATION
    is_convex: _STRUCT_FIELD_ANNOTATION
    contype: _STRUCT_FIELD_ANNOTATION
    conaffinity: _STRUCT_FIELD_ANNOTATION
    is_fixed: _STRUCT_FIELD_ANNOTATION
    is_decomposed: _STRUCT_FIELD_ANNOTATION
    needs_coup: _STRUCT_FIELD_ANNOTATION
    coup_friction: _STRUCT_FIELD_ANNOTATION
    coup_softness: _STRUCT_FIELD_ANNOTATION
    coup_restitution: _STRUCT_FIELD_ANNOTATION


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


@DATA_ORIENTED
class StructGeomsState(metaclass=BASE_METACLASS):
    pos: _STRUCT_FIELD_ANNOTATION
    quat: _STRUCT_FIELD_ANNOTATION
    aabb_min: _STRUCT_FIELD_ANNOTATION
    aabb_max: _STRUCT_FIELD_ANNOTATION
    verts_updated: _STRUCT_FIELD_ANNOTATION
    min_buffer_idx: _STRUCT_FIELD_ANNOTATION
    max_buffer_idx: _STRUCT_FIELD_ANNOTATION
    hibernated: _STRUCT_FIELD_ANNOTATION
    friction_ratio: _STRUCT_FIELD_ANNOTATION


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


@DATA_ORIENTED
class StructVertsInfo(metaclass=BASE_METACLASS):
    init_pos: _STRUCT_FIELD_ANNOTATION
    init_normal: _STRUCT_FIELD_ANNOTATION
    geom_idx: _STRUCT_FIELD_ANNOTATION
    init_center_pos: _STRUCT_FIELD_ANNOTATION
    verts_state_idx: _STRUCT_FIELD_ANNOTATION
    is_fixed: _STRUCT_FIELD_ANNOTATION


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


@DATA_ORIENTED
class StructFacesInfo(metaclass=BASE_METACLASS):
    verts_idx: _STRUCT_FIELD_ANNOTATION
    geom_idx: _STRUCT_FIELD_ANNOTATION


def get_faces_info(solver):
    shape = (solver.n_faces_,)

    return StructFacesInfo(
        verts_idx=qd.tensor(gs.qd_ivec3, shape, backend=_TENSOR_BACKEND),
        geom_idx=qd.tensor(gs.qd_int, shape, backend=_TENSOR_BACKEND),
    )


# =========================================== EdgesInfo ===========================================


@DATA_ORIENTED
class StructEdgesInfo(metaclass=BASE_METACLASS):
    v0: _STRUCT_FIELD_ANNOTATION
    v1: _STRUCT_FIELD_ANNOTATION
    length: _STRUCT_FIELD_ANNOTATION


def get_edges_info(solver):
    shape = (solver.n_edges_,)

    return StructEdgesInfo(
        v0=qd.tensor(gs.qd_int, shape, backend=_TENSOR_BACKEND),
        v1=qd.tensor(gs.qd_int, shape, backend=_TENSOR_BACKEND),
        length=qd.tensor(gs.qd_float, shape, backend=_TENSOR_BACKEND),
    )


# =========================================== VertsState ===========================================


@DATA_ORIENTED
class StructVertsState(metaclass=BASE_METACLASS):
    pos: _STRUCT_FIELD_ANNOTATION


def get_free_verts_state(solver):
    return StructVertsState(
        pos=qd.tensor(gs.qd_vec3, (solver.n_free_verts_, solver._B), backend=_TENSOR_BACKEND),
    )


def get_fixed_verts_state(solver):
    return StructVertsState(
        pos=qd.tensor(gs.qd_vec3, (solver.n_fixed_verts_,), backend=_TENSOR_BACKEND),
    )


# =========================================== VvertsInfo ===========================================


@DATA_ORIENTED
class StructVvertsInfo(metaclass=BASE_METACLASS):
    init_pos: _STRUCT_FIELD_ANNOTATION
    init_vnormal: _STRUCT_FIELD_ANNOTATION
    vgeom_idx: _STRUCT_FIELD_ANNOTATION


def get_vverts_info(solver):
    shape = (solver.n_vverts_,)

    return StructVvertsInfo(
        init_pos=qd.tensor(gs.qd_vec3, shape, backend=_TENSOR_BACKEND),
        init_vnormal=qd.tensor(gs.qd_vec3, shape, backend=_TENSOR_BACKEND),
        vgeom_idx=qd.tensor(gs.qd_int, shape, backend=_TENSOR_BACKEND),
    )


# =========================================== VfacesInfo ===========================================


@DATA_ORIENTED
class StructVfacesInfo(metaclass=BASE_METACLASS):
    vverts_idx: _STRUCT_FIELD_ANNOTATION
    vgeom_idx: _STRUCT_FIELD_ANNOTATION


def get_vfaces_info(solver):
    shape = (solver.n_vfaces_,)

    return StructVfacesInfo(
        vverts_idx=qd.tensor(gs.qd_ivec3, shape, backend=_TENSOR_BACKEND),
        vgeom_idx=qd.tensor(gs.qd_int, shape, backend=_TENSOR_BACKEND),
    )


# =========================================== VgeomsInfo ===========================================


@DATA_ORIENTED
class StructVgeomsInfo(metaclass=BASE_METACLASS):
    pos: _STRUCT_FIELD_ANNOTATION
    quat: _STRUCT_FIELD_ANNOTATION
    link_idx: _STRUCT_FIELD_ANNOTATION
    vvert_num: _STRUCT_FIELD_ANNOTATION
    vvert_start: _STRUCT_FIELD_ANNOTATION
    vvert_end: _STRUCT_FIELD_ANNOTATION
    vface_num: _STRUCT_FIELD_ANNOTATION
    vface_start: _STRUCT_FIELD_ANNOTATION
    vface_end: _STRUCT_FIELD_ANNOTATION
    color: _STRUCT_FIELD_ANNOTATION


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


@DATA_ORIENTED
class StructVgeomsState(metaclass=BASE_METACLASS):
    pos: _STRUCT_FIELD_ANNOTATION
    quat: _STRUCT_FIELD_ANNOTATION


def get_vgeoms_state(solver):
    shape = (solver.n_vgeoms_, solver._B)

    return StructVgeomsState(
        pos=qd.tensor(gs.qd_vec3, shape, backend=_TENSOR_BACKEND),
        quat=qd.tensor(gs.qd_vec4, shape, backend=_TENSOR_BACKEND),
    )


# =========================================== EqualitiesInfo ===========================================


@DATA_ORIENTED
class StructEqualitiesInfo(metaclass=BASE_METACLASS):
    eq_obj1id: _STRUCT_FIELD_ANNOTATION
    eq_obj2id: _STRUCT_FIELD_ANNOTATION
    eq_data: _STRUCT_FIELD_ANNOTATION
    eq_type: _STRUCT_FIELD_ANNOTATION
    sol_params: _STRUCT_FIELD_ANNOTATION


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


@DATA_ORIENTED
class StructEntitiesInfo(metaclass=BASE_METACLASS):
    dof_start: _STRUCT_FIELD_ANNOTATION
    dof_end: _STRUCT_FIELD_ANNOTATION
    n_dofs: _STRUCT_FIELD_ANNOTATION
    link_start: _STRUCT_FIELD_ANNOTATION
    link_end: _STRUCT_FIELD_ANNOTATION
    n_links: _STRUCT_FIELD_ANNOTATION
    geom_start: _STRUCT_FIELD_ANNOTATION
    geom_end: _STRUCT_FIELD_ANNOTATION
    n_geoms: _STRUCT_FIELD_ANNOTATION
    gravity_compensation: _STRUCT_FIELD_ANNOTATION
    is_local_collision_mask: _STRUCT_FIELD_ANNOTATION


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


@DATA_ORIENTED
class StructEntitiesState(metaclass=BASE_METACLASS):
    hibernated: _STRUCT_FIELD_ANNOTATION


def get_entities_state(solver):
    return StructEntitiesState(
        hibernated=qd.tensor(gs.qd_int, (solver.n_entities_, solver._B), backend=_TENSOR_BACKEND),
    )


# =========================================== RigidAdjointCache ===========================================
@DATA_ORIENTED
class StructRigidAdjointCache(metaclass=BASE_METACLASS):
    # This cache stores intermediate values during rigid body simulation to use Quadrants's AD. Quadrants's AD requires
    # us not to overwrite the values that have been read during the forward pass, so we need to store the intemediate
    # values in this cache to avoid overwriting them. Specifically, after we compute next frame's qpos, dofs_vel, and
    # dofs_acc, we need to store them in this cache because we overwrite the values in the next frame. See how
    # [kernel_save_adjoint_cache] is used in [rigid_solver.py] to store the values in this cache.
    qpos: _STRUCT_FIELD_ANNOTATION
    dofs_vel: _STRUCT_FIELD_ANNOTATION
    dofs_acc: _STRUCT_FIELD_ANNOTATION


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


@DATA_ORIENTED
class StructViewerRaycastResult(metaclass=BASE_METACLASS):
    distance: _STRUCT_FIELD_ANNOTATION
    geom_idx: _STRUCT_FIELD_ANNOTATION
    hit_point: _STRUCT_FIELD_ANNOTATION
    normal: _STRUCT_FIELD_ANNOTATION
    env_idx: _STRUCT_FIELD_ANNOTATION


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
