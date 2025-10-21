import math
import dataclasses
from typing import NamedTuple

import gstaichi as ti
import numpy as np

import genesis as gs


if not gs._initialized:
    gs.raise_exception("Genesis hasn't been initialized. Did you call `gs.init()`?")


# FIXME: NamedTuple in Python < 3.11 does not support annotation types that are not callable
V_ANNOTATION = ti.types.ndarray() if gs.use_ndarray else ti.template
V = ti.ndarray if gs.use_ndarray else ti.field
V_VEC = ti.Vector.ndarray if gs.use_ndarray else ti.Vector.field
V_MAT = ti.Matrix.ndarray if gs.use_ndarray else ti.Matrix.field

DATA_ORIENTED = dataclasses.dataclass if gs.use_ndarray else ti.data_oriented
BASE_CLASS = object if gs.use_ndarray else NamedTuple

# =========================================== RigidGlobalInfo ===========================================


@DATA_ORIENTED
class StructRigidGlobalInfo(BASE_CLASS):
    n_awake_dofs: V_ANNOTATION
    awake_dofs: V_ANNOTATION
    n_awake_entities: V_ANNOTATION
    awake_entities: V_ANNOTATION
    n_awake_links: V_ANNOTATION
    awake_links: V_ANNOTATION
    qpos0: V_ANNOTATION
    qpos: V_ANNOTATION
    links_T: V_ANNOTATION
    envs_offset: V_ANNOTATION
    geoms_init_AABB: V_ANNOTATION
    mass_mat: V_ANNOTATION
    mass_mat_L: V_ANNOTATION
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
    n_equalities_candidate: V_ANNOTATION
    hibernation_thresh_acc: V_ANNOTATION
    hibernation_thresh_vel: V_ANNOTATION


def get_rigid_global_info(solver):
    _B = solver._B

    substep_dt = V(dtype=gs.ti_float, shape=(_B,))
    substep_dt.fill(solver._substep_dt)
    iterations = V(dtype=gs.ti_int, shape=())
    iterations.fill(solver._options.iterations)
    tolerance = V(dtype=gs.ti_float, shape=())
    tolerance.fill(solver._options.tolerance)
    ls_iterations = V(dtype=gs.ti_int, shape=())
    ls_iterations.fill(solver._options.ls_iterations)
    ls_tolerance = V(dtype=gs.ti_float, shape=())
    ls_tolerance.fill(solver._options.ls_tolerance)

    noslip_iterations = V(dtype=gs.ti_int, shape=())
    noslip_iterations.fill(solver._options.noslip_iterations)
    noslip_tolerance = V(dtype=gs.ti_float, shape=())
    noslip_tolerance.fill(solver._options.noslip_tolerance)
    n_equalities = V(dtype=gs.ti_int, shape=())
    n_equalities.fill(solver._n_equalities)
    n_equalities_candidate = V(dtype=gs.ti_int, shape=())
    n_equalities_candidate.fill(solver.n_equalities_candidate)
    hibernation_thresh_acc = V(dtype=gs.ti_float, shape=())
    hibernation_thresh_acc.fill(solver._hibernation_thresh_acc)
    hibernation_thresh_vel = V(dtype=gs.ti_float, shape=())
    hibernation_thresh_vel.fill(solver._hibernation_thresh_vel)

    return StructRigidGlobalInfo(
        n_awake_dofs=V(dtype=gs.ti_int, shape=(_B,)),
        awake_dofs=V(dtype=gs.ti_int, shape=(solver.n_dofs_, _B)),
        n_awake_entities=V(dtype=gs.ti_int, shape=(_B,)),
        awake_entities=V(dtype=gs.ti_int, shape=(solver.n_entities_, _B)),
        n_awake_links=V(dtype=gs.ti_int, shape=(_B,)),
        awake_links=V(dtype=gs.ti_int, shape=(solver.n_links_, _B)),
        qpos0=V(dtype=gs.ti_float, shape=(solver.n_qs_, _B)),
        qpos=V(dtype=gs.ti_float, shape=(solver.n_qs_, _B)),
        links_T=V_MAT(n=4, m=4, dtype=gs.ti_float, shape=(solver.n_links_,)),
        envs_offset=V_VEC(3, dtype=gs.ti_float, shape=(_B,)),
        geoms_init_AABB=V_VEC(3, dtype=gs.ti_float, shape=(solver.n_geoms_, 8)),
        mass_mat=V(dtype=gs.ti_float, shape=(solver.n_dofs_, solver.n_dofs_, _B)),
        mass_mat_L=V(dtype=gs.ti_float, shape=(solver.n_dofs_, solver.n_dofs_, _B)),
        mass_mat_D_inv=V(dtype=gs.ti_float, shape=(solver.n_dofs_, _B)),
        mass_mat_mask=V(dtype=gs.ti_bool, shape=(solver.n_entities_, _B)),
        meaninertia=V(dtype=gs.ti_float, shape=(_B,)),
        mass_parent_mask=V(dtype=gs.ti_float, shape=(solver.n_dofs_, solver.n_dofs_)),
        gravity=V_VEC(3, dtype=gs.ti_float, shape=(_B,)),
        substep_dt=substep_dt,
        iterations=iterations,
        tolerance=tolerance,
        ls_iterations=ls_iterations,
        ls_tolerance=ls_tolerance,
        noslip_iterations=noslip_iterations,
        noslip_tolerance=noslip_tolerance,
        n_equalities=n_equalities,
        n_equalities_candidate=n_equalities_candidate,
        hibernation_thresh_acc=hibernation_thresh_acc,
        hibernation_thresh_vel=hibernation_thresh_vel,
    )


# =========================================== Constraint ===========================================


@DATA_ORIENTED
class StructConstraintState(BASE_CLASS):
    n_constraints: V_ANNOTATION
    ti_n_equalities: V_ANNOTATION
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
    quad: V_ANNOTATION
    candidates: V_ANNOTATION
    ls_it: V_ANNOTATION
    ls_result: V_ANNOTATION
    # Optional CG fields
    cg_prev_grad: V_ANNOTATION
    cg_prev_Mgrad: V_ANNOTATION
    cg_beta: V_ANNOTATION
    cg_pg_dot_pMg: V_ANNOTATION
    # Optional Newton fields
    nt_H: V_ANNOTATION
    nt_vec: V_ANNOTATION


def get_constraint_state(constraint_solver, solver):
    _B = solver._B
    len_constraints_ = constraint_solver.len_constraints_

    jac_shape = (len_constraints_, solver.n_dofs_, _B)
    if math.prod(jac_shape) > np.iinfo(np.int32).max:
        gs.raise_exception(
            f"Jacobian shape {jac_shape} is too large for int32. Consider reducing the number of constraints or the "
            "number of degrees of freedom."
        )

    if solver._options.noslip_iterations > 0:
        if len_constraints_**2 * _B > 2e9:
            gs.logger.warning(
                f"efc_AR shape {len_constraints_}x{len_constraints_}x{_B} is very large. Consider manually set a "
                "smaller 'max_collision_pairs' in RigidOptions to reduce the size of reserved memory. "
            )
        efc_AR_shape = (len_constraints_, len_constraints_, _B)
        efc_b_shape = (len_constraints_, _B)
    else:
        efc_AR_shape = (1,)
        efc_b_shape = (1,)

    return StructConstraintState(
        n_constraints=V(dtype=gs.ti_int, shape=(_B,)),
        ti_n_equalities=V(gs.ti_int, shape=(_B,)),
        jac=V(dtype=gs.ti_float, shape=(len_constraints_, solver.n_dofs_, _B)),
        diag=V(dtype=gs.ti_float, shape=(len_constraints_, _B)),
        aref=V(dtype=gs.ti_float, shape=(len_constraints_, _B)),
        jac_relevant_dofs=V(gs.ti_int, shape=(len_constraints_, solver.n_dofs_, _B)),
        jac_n_relevant_dofs=V(gs.ti_int, shape=(len_constraints_, _B)),
        n_constraints_equality=V(gs.ti_int, shape=(_B,)),
        n_constraints_frictionloss=V(gs.ti_int, shape=(_B,)),
        improved=V(gs.ti_int, shape=(_B,)),
        Jaref=V(dtype=gs.ti_float, shape=(len_constraints_, _B)),
        Ma=V(dtype=gs.ti_float, shape=(solver.n_dofs_, _B)),
        Ma_ws=V(dtype=gs.ti_float, shape=(solver.n_dofs_, _B)),
        grad=V(dtype=gs.ti_float, shape=(solver.n_dofs_, _B)),
        Mgrad=V(dtype=gs.ti_float, shape=(solver.n_dofs_, _B)),
        search=V(dtype=gs.ti_float, shape=(solver.n_dofs_, _B)),
        efc_D=V(dtype=gs.ti_float, shape=(len_constraints_, _B)),
        efc_frictionloss=V(dtype=gs.ti_float, shape=(len_constraints_, _B)),
        efc_force=V(dtype=gs.ti_float, shape=(len_constraints_, _B)),
        efc_b=V(dtype=gs.ti_float, shape=efc_b_shape),
        efc_AR=V(dtype=gs.ti_float, shape=efc_AR_shape),
        active=V(dtype=gs.ti_bool, shape=(len_constraints_, _B)),
        prev_active=V(dtype=gs.ti_bool, shape=(len_constraints_, _B)),
        qfrc_constraint=V(dtype=gs.ti_float, shape=(solver.n_dofs_, _B)),
        qacc=V(dtype=gs.ti_float, shape=(solver.n_dofs_, _B)),
        qacc_ws=V(dtype=gs.ti_float, shape=(solver.n_dofs_, _B)),
        qacc_prev=V(dtype=gs.ti_float, shape=(solver.n_dofs_, _B)),
        cost_ws=V(gs.ti_float, shape=(_B,)),
        gauss=V(gs.ti_float, shape=(_B,)),
        cost=V(gs.ti_float, shape=(_B,)),
        prev_cost=V(gs.ti_float, shape=(_B,)),
        gtol=V(gs.ti_float, shape=(_B,)),
        mv=V(dtype=gs.ti_float, shape=(solver.n_dofs_, _B)),
        jv=V(dtype=gs.ti_float, shape=(len_constraints_, _B)),
        quad_gauss=V(dtype=gs.ti_float, shape=(3, _B)),
        quad=V(dtype=gs.ti_float, shape=(len_constraints_, 3, _B)),
        candidates=V(dtype=gs.ti_float, shape=(12, _B)),
        ls_it=V(gs.ti_float, shape=(_B,)),
        ls_result=V(gs.ti_int, shape=(_B,)),
        cg_prev_grad=V(dtype=gs.ti_float, shape=(solver.n_dofs_, _B)),
        cg_prev_Mgrad=V(dtype=gs.ti_float, shape=(solver.n_dofs_, _B)),
        cg_beta=V(gs.ti_float, shape=(_B,)),
        cg_pg_dot_pMg=V(gs.ti_float, shape=(_B,)),
        nt_H=V(dtype=gs.ti_float, shape=(solver.n_dofs_, solver.n_dofs_, _B)),
        nt_vec=V(dtype=gs.ti_float, shape=(solver.n_dofs_, _B)),
    )


# =========================================== Collider ===========================================


@DATA_ORIENTED
class StructContactData(BASE_CLASS):
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
        geom_a=V(dtype=gs.ti_int, shape=(max_contact_pairs_, _B)),
        geom_b=V(dtype=gs.ti_int, shape=(max_contact_pairs_, _B)),
        normal=V(dtype=gs.ti_vec3, shape=(max_contact_pairs_, _B), needs_grad=requires_grad),
        pos=V(dtype=gs.ti_vec3, shape=(max_contact_pairs_, _B), needs_grad=requires_grad),
        penetration=V(dtype=gs.ti_float, shape=(max_contact_pairs_, _B), needs_grad=requires_grad),
        friction=V(dtype=gs.ti_float, shape=(max_contact_pairs_, _B)),
        sol_params=V_VEC(7, dtype=gs.ti_float, shape=(max_contact_pairs_, _B)),
        force=V(dtype=gs.ti_vec3, shape=(max_contact_pairs_, _B)),
        link_a=V(dtype=gs.ti_int, shape=(max_contact_pairs_, _B)),
        link_b=V(dtype=gs.ti_int, shape=(max_contact_pairs_, _B)),
    )


@DATA_ORIENTED
class StructDiffContactInput(BASE_CLASS):
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


def get_diff_contact_input(solver, max_contacts_per_pair):
    _B = solver._B

    return StructDiffContactInput(
        geom_a=V(dtype=gs.ti_int, shape=(_B, max_contacts_per_pair)),
        geom_b=V(dtype=gs.ti_int, shape=(_B, max_contacts_per_pair)),
        local_pos1_a=V_VEC(3, dtype=gs.ti_float, shape=(_B, max_contacts_per_pair)),
        local_pos1_b=V_VEC(3, dtype=gs.ti_float, shape=(_B, max_contacts_per_pair)),
        local_pos1_c=V_VEC(3, dtype=gs.ti_float, shape=(_B, max_contacts_per_pair)),
        local_pos2_a=V_VEC(3, dtype=gs.ti_float, shape=(_B, max_contacts_per_pair)),
        local_pos2_b=V_VEC(3, dtype=gs.ti_float, shape=(_B, max_contacts_per_pair)),
        local_pos2_c=V_VEC(3, dtype=gs.ti_float, shape=(_B, max_contacts_per_pair)),
        w_local_pos1=V_VEC(3, dtype=gs.ti_float, shape=(_B, max_contacts_per_pair)),
        w_local_pos2=V_VEC(3, dtype=gs.ti_float, shape=(_B, max_contacts_per_pair)),
        ref_id=V(dtype=gs.ti_int, shape=(_B, max_contacts_per_pair)),
        valid=V(dtype=gs.ti_int, shape=(_B, max_contacts_per_pair)),
        ref_penetration=V(dtype=gs.ti_float, shape=(_B, max_contacts_per_pair), needs_grad=True),
    )


@DATA_ORIENTED
class StructSortBuffer(BASE_CLASS):
    value: V_ANNOTATION
    i_g: V_ANNOTATION
    is_max: V_ANNOTATION


def get_sort_buffer(solver):
    _B = solver._B

    return StructSortBuffer(
        value=V(dtype=gs.ti_float, shape=(2 * solver.n_geoms_, _B)),
        i_g=V(dtype=gs.ti_int, shape=(2 * solver.n_geoms_, _B)),
        is_max=V(dtype=gs.ti_bool, shape=(2 * solver.n_geoms_, _B)),
    )


@DATA_ORIENTED
class StructContactCache(BASE_CLASS):
    normal: V_ANNOTATION


def get_contact_cache(solver):
    _B = solver._B
    return StructContactCache(
        normal=V_VEC(3, dtype=gs.ti_float, shape=(solver.n_geoms_, solver.n_geoms_, _B)),
    )


@DATA_ORIENTED
class StructAggList(BASE_CLASS):
    curr: V_ANNOTATION
    n: V_ANNOTATION
    start: V_ANNOTATION


def get_agg_list(solver):
    _B = solver._B
    n_entities = solver.n_entities

    return StructAggList(
        curr=V(dtype=gs.ti_int, shape=(n_entities, _B)),
        n=V(dtype=gs.ti_int, shape=(n_entities, _B)),
        start=V(dtype=gs.ti_int, shape=(n_entities, _B)),
    )


@DATA_ORIENTED
class StructContactIslandState(BASE_CLASS):
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

    return StructContactIslandState(
        ci_edges=V(dtype=gs.ti_int, shape=(max_contact_pairs, 2, _B)),
        edge_id=V(dtype=gs.ti_int, shape=(max_contact_pairs * 2, _B)),
        constraint_list=V(dtype=gs.ti_int, shape=(max_contact_pairs, _B)),
        constraint_id=V(dtype=gs.ti_int, shape=(max_contact_pairs * 2, _B)),
        entity_edge=get_agg_list(solver),
        island_col=get_agg_list(solver),
        island_hibernated=V(dtype=gs.ti_int, shape=(solver.n_entities, _B)),
        island_entity=get_agg_list(solver),
        entity_id=V(dtype=gs.ti_int, shape=(solver.n_entities, _B)),
        n_edges=V(dtype=gs.ti_int, shape=(_B,)),
        n_islands=V(dtype=gs.ti_int, shape=(_B,)),
        n_stack=V(dtype=gs.ti_int, shape=(_B,)),
        entity_island=V(dtype=gs.ti_int, shape=(solver.n_entities, _B)),
        stack=V(dtype=gs.ti_int, shape=(solver.n_entities, _B)),
        entity_idx_to_next_entity_idx_in_hibernated_island=V(dtype=gs.ti_int, shape=(solver.n_entities, _B)),
    )


@DATA_ORIENTED
class StructColliderState(BASE_CLASS):
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

    return StructColliderState(
        sort_buffer=get_sort_buffer(solver),
        contact_data=get_contact_data(solver, max_contact_pairs, requires_grad),
        active_buffer=V(dtype=gs.ti_int, shape=(n_geoms, _B)),
        n_broad_pairs=V(dtype=gs.ti_int, shape=(_B,)),
        broad_collision_pairs=V_VEC(2, dtype=gs.ti_int, shape=(max(max_collision_pairs_broad, 1), _B)),
        active_buffer_awake=V(dtype=gs.ti_int, shape=(n_geoms, _B)),
        active_buffer_hib=V(dtype=gs.ti_int, shape=(n_geoms, _B)),
        box_depth=V(dtype=gs.ti_float, shape=(collider_info.box_MAXCONPAIR[None], _B)),
        box_points=V_VEC(3, dtype=gs.ti_float, shape=(collider_info.box_MAXCONPAIR[None], _B)),
        box_pts=V_VEC(3, dtype=gs.ti_float, shape=(6, _B)),
        box_lines=V_VEC(6, dtype=gs.ti_float, shape=(4, _B)),
        box_linesu=V_VEC(6, dtype=gs.ti_float, shape=(4, _B)),
        box_axi=V_VEC(3, dtype=gs.ti_float, shape=(3, _B)),
        box_ppts2=V(dtype=gs.ti_float, shape=(4, 2, _B)),
        box_pu=V_VEC(3, dtype=gs.ti_float, shape=(4, _B)),
        xyz_max_min=V(dtype=gs.ti_float, shape=(6, _B)),
        prism=V_VEC(3, dtype=gs.ti_float, shape=(6, _B)),
        n_contacts=V(dtype=gs.ti_int, shape=(_B,)),
        n_contacts_hibernated=V(dtype=gs.ti_int, shape=(_B,)),
        first_time=V(dtype=gs.ti_int, shape=(_B,)),
        contact_cache=get_contact_cache(solver),
        diff_contact_input=get_diff_contact_input(solver, max(max_contact_pairs, 1) if requires_grad else 1),
    )


@DATA_ORIENTED
class StructColliderInfo(BASE_CLASS):
    vert_neighbors: V_ANNOTATION
    vert_neighbor_start: V_ANNOTATION
    vert_n_neighbors: V_ANNOTATION
    collision_pair_validity: V_ANNOTATION
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
    mpr_to_sdf_overlap_ratio: V_ANNOTATION
    # maximum number of contact points for box-box collision detection
    box_MAXCONPAIR: V_ANNOTATION
    # differentiable contact tolerance
    diff_pos_tolerance: V_ANNOTATION
    diff_normal_tolerance: V_ANNOTATION


def get_collider_info(
    solver,
    n_vert_neighbors,
    collider_static_config,
    **kwargs,
):
    for geom in solver.geoms:
        if geom.type == gs.GEOM_TYPE.TERRAIN:
            terrain_hf_shape = geom.entity.terrain_hf.shape
            break
    else:
        terrain_hf_shape = 1

    mc_perturbation = V(dtype=gs.ti_float, shape=())
    mc_perturbation.fill(kwargs["mc_perturbation"])
    mc_tolerance = V(dtype=gs.ti_float, shape=())
    mc_tolerance.fill(kwargs["mc_tolerance"])
    mpr_to_sdf_overlap_ratio = V(dtype=gs.ti_float, shape=())
    mpr_to_sdf_overlap_ratio.fill(kwargs["mpr_to_sdf_overlap_ratio"])
    box_MAXCONPAIR = V(dtype=gs.ti_int, shape=())
    box_MAXCONPAIR.fill(kwargs["box_MAXCONPAIR"])
    diff_pos_tolerance = V(dtype=gs.ti_float, shape=())
    diff_pos_tolerance.fill(kwargs["diff_pos_tolerance"])
    diff_normal_tolerance = V(dtype=gs.ti_float, shape=())
    diff_normal_tolerance.fill(kwargs["diff_normal_tolerance"])

    return StructColliderInfo(
        vert_neighbors=V(dtype=gs.ti_int, shape=(max(n_vert_neighbors, 1),)),
        vert_neighbor_start=V(dtype=gs.ti_int, shape=solver.n_verts_),
        vert_n_neighbors=V(dtype=gs.ti_int, shape=solver.n_verts_),
        collision_pair_validity=V(dtype=gs.ti_int, shape=(solver.n_geoms_, solver.n_geoms_)),
        max_possible_pairs=V(dtype=gs.ti_int, shape=()),
        max_collision_pairs=V(dtype=gs.ti_int, shape=()),
        max_contact_pairs=V(dtype=gs.ti_int, shape=()),
        max_collision_pairs_broad=V(dtype=gs.ti_int, shape=()),
        terrain_hf=V(dtype=gs.ti_float, shape=terrain_hf_shape),
        terrain_rc=V(dtype=gs.ti_int, shape=2),
        terrain_scale=V(dtype=gs.ti_float, shape=2),
        terrain_xyz_maxmin=V(dtype=gs.ti_float, shape=6),
        mc_perturbation=mc_perturbation,
        mc_tolerance=mc_tolerance,
        mpr_to_sdf_overlap_ratio=mpr_to_sdf_overlap_ratio,
        box_MAXCONPAIR=box_MAXCONPAIR,
        diff_pos_tolerance=diff_pos_tolerance,
        diff_normal_tolerance=diff_normal_tolerance,
    )


# FIXME: Fast cache does not support 'NamedTuple' for now.
# See PR: https://github.com/Genesis-Embodied-AI/gstaichi/pull/248.
@ti.data_oriented
class StructColliderStaticConfig:  # (NamedTuple):
    has_nonconvex_nonterrain: bool
    has_terrain: bool
    # maximum number of contact pairs per collision pair
    n_contacts_per_pair: int
    # ccd algorithm
    ccd_algorithm: int

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


# =========================================== MPR ===========================================


@DATA_ORIENTED
class StructMPRSimplexSupport(BASE_CLASS):
    v1: V_ANNOTATION
    v2: V_ANNOTATION
    v: V_ANNOTATION


def get_mpr_simplex_support(B_):
    return StructMPRSimplexSupport(
        v1=V_VEC(3, dtype=gs.ti_float, shape=(4, B_)),
        v2=V_VEC(3, dtype=gs.ti_float, shape=(4, B_)),
        v=V_VEC(3, dtype=gs.ti_float, shape=(4, B_)),
    )


@DATA_ORIENTED
class StructMPRState(BASE_CLASS):
    simplex_support: StructMPRSimplexSupport
    simplex_size: V_ANNOTATION


def get_mpr_state(B_):
    return StructMPRState(
        simplex_support=get_mpr_simplex_support(B_),
        simplex_size=V(dtype=gs.ti_int, shape=(B_,)),
    )


# =========================================== GJK ===========================================


@DATA_ORIENTED
class StructMDVertex(BASE_CLASS):
    # Vertex of the Minkowski difference
    obj1: V_ANNOTATION
    obj2: V_ANNOTATION
    local_obj1: V_ANNOTATION
    local_obj2: V_ANNOTATION
    id1: V_ANNOTATION
    id2: V_ANNOTATION
    mink: V_ANNOTATION


def get_gjk_simplex_vertex(solver):
    _B = solver._B

    return StructMDVertex(
        obj1=V_VEC(3, dtype=gs.ti_float, shape=(_B, 4)),
        obj2=V_VEC(3, dtype=gs.ti_float, shape=(_B, 4)),
        local_obj1=V_VEC(3, dtype=gs.ti_float, shape=(_B, 4)),
        local_obj2=V_VEC(3, dtype=gs.ti_float, shape=(_B, 4)),
        id1=V(dtype=gs.ti_int, shape=(_B, 4)),
        id2=V(dtype=gs.ti_int, shape=(_B, 4)),
        mink=V_VEC(3, dtype=gs.ti_float, shape=(_B, 4)),
    )


def get_epa_polytope_vertex(solver, gjk_static_config):
    _B = solver._B
    max_num_polytope_verts = 5 + gjk_static_config.epa_max_iterations

    return StructMDVertex(
        obj1=V_VEC(3, dtype=gs.ti_float, shape=(_B, max_num_polytope_verts)),
        obj2=V_VEC(3, dtype=gs.ti_float, shape=(_B, max_num_polytope_verts)),
        local_obj1=V_VEC(3, dtype=gs.ti_float, shape=(_B, max_num_polytope_verts)),
        local_obj2=V_VEC(3, dtype=gs.ti_float, shape=(_B, max_num_polytope_verts)),
        id1=V(dtype=gs.ti_int, shape=(_B, max_num_polytope_verts)),
        id2=V(dtype=gs.ti_int, shape=(_B, max_num_polytope_verts)),
        mink=V_VEC(3, dtype=gs.ti_float, shape=(_B, max_num_polytope_verts)),
    )


@DATA_ORIENTED
class StructGJKSimplex(BASE_CLASS):
    nverts: V_ANNOTATION
    dist: V_ANNOTATION


def get_gjk_simplex(solver):
    _B = solver._B

    return StructGJKSimplex(
        nverts=V(dtype=gs.ti_int, shape=(_B,)),
        dist=V(dtype=gs.ti_float, shape=(_B,)),
    )


@DATA_ORIENTED
class StructGJKSimplexBuffer(BASE_CLASS):
    normal: V_ANNOTATION
    sdist: V_ANNOTATION


def get_gjk_simplex_buffer(solver):
    _B = solver._B

    return StructGJKSimplexBuffer(
        normal=V_VEC(3, dtype=gs.ti_float, shape=(_B, 4)),
        sdist=V(dtype=gs.ti_float, shape=(_B, 4)),
    )


@DATA_ORIENTED
class StructEPAPolytope(BASE_CLASS):
    nverts: V_ANNOTATION
    nfaces: V_ANNOTATION
    nfaces_map: V_ANNOTATION
    horizon_nedges: V_ANNOTATION
    horizon_w: V_ANNOTATION


def get_epa_polytope(solver):
    _B = solver._B

    return StructEPAPolytope(
        nverts=V(dtype=gs.ti_int, shape=(_B,)),
        nfaces=V(dtype=gs.ti_int, shape=(_B,)),
        nfaces_map=V(dtype=gs.ti_int, shape=(_B,)),
        horizon_nedges=V(dtype=gs.ti_int, shape=(_B,)),
        horizon_w=V_VEC(3, dtype=gs.ti_float, shape=(_B,)),
    )


@DATA_ORIENTED
class StructEPAPolytopeFace(BASE_CLASS):
    verts_idx: V_ANNOTATION
    adj_idx: V_ANNOTATION
    normal: V_ANNOTATION
    dist2: V_ANNOTATION
    map_idx: V_ANNOTATION
    visited: V_ANNOTATION


def get_epa_polytope_face(solver, polytope_max_faces):
    _B = solver._B

    return StructEPAPolytopeFace(
        verts_idx=V_VEC(3, dtype=gs.ti_int, shape=(_B, polytope_max_faces)),
        adj_idx=V_VEC(3, dtype=gs.ti_int, shape=(_B, polytope_max_faces)),
        normal=V_VEC(3, dtype=gs.ti_float, shape=(_B, polytope_max_faces)),
        dist2=V(dtype=gs.ti_float, shape=(_B, polytope_max_faces)),
        map_idx=V(dtype=gs.ti_int, shape=(_B, polytope_max_faces)),
        visited=V(dtype=gs.ti_int, shape=(_B, polytope_max_faces)),
    )


@DATA_ORIENTED
class StructEPAPolytopeHorizonData(BASE_CLASS):
    face_idx: V_ANNOTATION
    edge_idx: V_ANNOTATION


def get_epa_polytope_horizon_data(solver, polytope_max_horizons):
    _B = solver._B

    return StructEPAPolytopeHorizonData(
        face_idx=V(dtype=gs.ti_int, shape=(_B, polytope_max_horizons)),
        edge_idx=V(dtype=gs.ti_int, shape=(_B, polytope_max_horizons)),
    )


@DATA_ORIENTED
class StructContactFace(BASE_CLASS):
    vert1: V_ANNOTATION
    vert2: V_ANNOTATION
    endverts: V_ANNOTATION
    normal1: V_ANNOTATION
    normal2: V_ANNOTATION
    id1: V_ANNOTATION
    id2: V_ANNOTATION


def get_contact_face(solver, max_contact_polygon_verts):
    _B = solver._B

    return StructContactFace(
        vert1=V_VEC(3, dtype=gs.ti_float, shape=(_B, max_contact_polygon_verts)),
        vert2=V_VEC(3, dtype=gs.ti_float, shape=(_B, max_contact_polygon_verts)),
        endverts=V_VEC(3, dtype=gs.ti_float, shape=(_B, max_contact_polygon_verts)),
        normal1=V_VEC(3, dtype=gs.ti_float, shape=(_B, max_contact_polygon_verts)),
        normal2=V_VEC(3, dtype=gs.ti_float, shape=(_B, max_contact_polygon_verts)),
        id1=V(dtype=gs.ti_int, shape=(_B, max_contact_polygon_verts)),
        id2=V(dtype=gs.ti_int, shape=(_B, max_contact_polygon_verts)),
    )


@DATA_ORIENTED
class StructContactNormal(BASE_CLASS):
    endverts: V_ANNOTATION
    normal: V_ANNOTATION
    id: V_ANNOTATION


def get_contact_normal(solver, max_contact_polygon_verts):
    _B = solver._B

    return StructContactNormal(
        endverts=V_VEC(3, dtype=gs.ti_float, shape=(_B, max_contact_polygon_verts)),
        normal=V_VEC(3, dtype=gs.ti_float, shape=(_B, max_contact_polygon_verts)),
        id=V(dtype=gs.ti_int, shape=(_B, max_contact_polygon_verts)),
    )


@DATA_ORIENTED
class StructContactHalfspace(BASE_CLASS):
    normal: V_ANNOTATION
    dist: V_ANNOTATION


def get_contact_halfspace(solver, max_contact_polygon_verts):
    _B = solver._B

    return StructContactHalfspace(
        normal=V_VEC(3, dtype=gs.ti_float, shape=(_B, max_contact_polygon_verts)),
        dist=V(dtype=gs.ti_float, shape=(_B, max_contact_polygon_verts)),
    )


@DATA_ORIENTED
class StructWitness(BASE_CLASS):
    point_obj1: V_ANNOTATION
    point_obj2: V_ANNOTATION


def get_witness(solver, max_contacts_per_pair):
    _B = solver._B

    return StructWitness(
        point_obj1=V_VEC(3, dtype=gs.ti_float, shape=(_B, max_contacts_per_pair)),
        point_obj2=V_VEC(3, dtype=gs.ti_float, shape=(_B, max_contacts_per_pair)),
    )


@DATA_ORIENTED
class StructGJKState(BASE_CLASS):
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


def get_gjk_state(solver, static_rigid_sim_config, gjk_static_config):
    _B = solver._B
    enable_mujoco_compatibility = static_rigid_sim_config.enable_mujoco_compatibility
    polytope_max_faces = gjk_static_config.polytope_max_faces
    max_contacts_per_pair = gjk_static_config.max_contacts_per_pair
    max_contact_polygon_verts = gjk_static_config.max_contact_polygon_verts
    requires_grad = solver._static_rigid_sim_config.requires_grad

    # FIXME: Define GJKState and MujocoCompatGJKState that derives from the former but defines additional attributes
    return StructGJKState(
        # GJK simplex
        support_mesh_prev_vertex_id=V(dtype=gs.ti_int, shape=(_B, 2)),
        simplex_vertex=get_gjk_simplex_vertex(solver),
        simplex_buffer=get_gjk_simplex_buffer(solver),
        simplex=get_gjk_simplex(solver),
        last_searched_simplex_vertex_id=V(dtype=gs.ti_int, shape=(_B,)),
        simplex_vertex_intersect=get_gjk_simplex_vertex(solver),
        simplex_buffer_intersect=get_gjk_simplex_buffer(solver),
        nsimplex=V(dtype=gs.ti_int, shape=(_B,)),
        # EPA polytope
        polytope=get_epa_polytope(solver),
        polytope_verts=get_epa_polytope_vertex(solver, gjk_static_config),
        polytope_faces=get_epa_polytope_face(solver, polytope_max_faces),
        polytope_faces_map=V(dtype=gs.ti_int, shape=(_B, polytope_max_faces)),
        polytope_horizon_data=get_epa_polytope_horizon_data(solver, 6 + gjk_static_config.epa_max_iterations),
        polytope_horizon_stack=get_epa_polytope_horizon_data(solver, polytope_max_faces * 3),
        # Multi-contact detection (MuJoCo compatibility)
        contact_faces=get_contact_face(solver, max_contact_polygon_verts),
        contact_normals=get_contact_normal(solver, max_contact_polygon_verts),
        contact_halfspaces=get_contact_halfspace(solver, max_contact_polygon_verts),
        contact_clipped_polygons=V_VEC(3, dtype=gs.ti_float, shape=(_B, 2, max_contact_polygon_verts)),
        multi_contact_flag=V(dtype=gs.ti_bool, shape=(_B,)),
        # Final results
        witness=get_witness(solver, max_contacts_per_pair),
        n_witness=V(dtype=gs.ti_int, shape=(_B,)),
        n_contacts=V(dtype=gs.ti_int, shape=(_B,)),
        contact_pos=V_VEC(3, dtype=gs.ti_float, shape=(_B, max_contacts_per_pair)),
        normal=V_VEC(3, dtype=gs.ti_float, shape=(_B, max_contacts_per_pair)),
        is_col=V(dtype=gs.ti_bool, shape=(_B,)),
        penetration=V(dtype=gs.ti_float, shape=(_B,)),
        distance=V(dtype=gs.ti_float, shape=(_B,)),
        diff_contact_input=get_diff_contact_input(solver, max(max_contacts_per_pair, 1) if requires_grad else 1),
        n_diff_contact_input=V(dtype=gs.ti_int, shape=(_B,)),
        diff_penetration=V(dtype=gs.ti_float, shape=(_B, max_contacts_per_pair)),
    )


# =========================================== SupportField ===========================================


@DATA_ORIENTED
class StructSupportFieldInfo(BASE_CLASS):
    support_cell_start: V_ANNOTATION
    support_v: V_ANNOTATION
    support_vid: V_ANNOTATION


def get_support_field_info(n_geoms, n_support_cells):
    return StructSupportFieldInfo(
        support_cell_start=V(dtype=gs.ti_int, shape=(max(n_geoms, 1),)),
        support_v=V_VEC(3, dtype=gs.ti_float, shape=(max(n_support_cells, 1),)),
        support_vid=V(dtype=gs.ti_int, shape=(max(n_support_cells, 1),)),
    )


# =========================================== SDF ===========================================


@DATA_ORIENTED
class StructSDFGeomInfo(BASE_CLASS):
    T_mesh_to_sdf: V_ANNOTATION
    sdf_res: V_ANNOTATION
    sdf_max: V_ANNOTATION
    sdf_cell_size: V_ANNOTATION
    sdf_cell_start: V_ANNOTATION


def get_sdf_geom_info(n_geoms):
    return StructSDFGeomInfo(
        T_mesh_to_sdf=V_MAT(n=4, m=4, dtype=gs.ti_float, shape=(n_geoms,)),
        sdf_res=V_VEC(3, dtype=gs.ti_int, shape=(n_geoms,)),
        sdf_max=V(dtype=gs.ti_float, shape=(n_geoms,)),
        sdf_cell_size=V(dtype=gs.ti_float, shape=(n_geoms,)),
        sdf_cell_start=V(dtype=gs.ti_int, shape=(n_geoms,)),
    )


@DATA_ORIENTED
class StructSDFInfo(BASE_CLASS):
    geoms_info: StructSDFGeomInfo
    geoms_sdf_start: V_ANNOTATION
    geoms_sdf_val: V_ANNOTATION
    geoms_sdf_grad: V_ANNOTATION
    geoms_sdf_closest_vert: V_ANNOTATION


def get_sdf_info(n_geoms, n_cells):
    return StructSDFInfo(
        geoms_info=get_sdf_geom_info(max(n_geoms, 1)),
        geoms_sdf_start=V(dtype=gs.ti_int, shape=(max(n_geoms, 1),)),
        geoms_sdf_val=V(dtype=gs.ti_float, shape=(max(n_cells, 1),)),
        geoms_sdf_grad=V_VEC(3, dtype=gs.ti_float, shape=(max(n_cells, 1),)),
        geoms_sdf_closest_vert=V(dtype=gs.ti_int, shape=(max(n_cells, 1),)),
    )


# =========================================== DofsInfo and DofsState ===========================================


@DATA_ORIENTED
class StructDofsInfo(BASE_CLASS):
    stiffness: V_ANNOTATION
    invweight: V_ANNOTATION
    armature: V_ANNOTATION
    damping: V_ANNOTATION
    frictionloss: V_ANNOTATION
    motion_ang: V_ANNOTATION
    motion_vel: V_ANNOTATION
    limit: V_ANNOTATION
    dof_start: V_ANNOTATION
    kp: V_ANNOTATION
    kv: V_ANNOTATION
    force_range: V_ANNOTATION


def get_dofs_info(solver):
    shape = (solver.n_dofs_, solver._B) if solver._options.batch_dofs_info else (solver.n_dofs_,)

    return StructDofsInfo(
        stiffness=V(dtype=gs.ti_float, shape=shape),
        invweight=V(dtype=gs.ti_float, shape=shape),
        armature=V(dtype=gs.ti_float, shape=shape),
        damping=V(dtype=gs.ti_float, shape=shape),
        frictionloss=V(dtype=gs.ti_float, shape=shape),
        motion_ang=V(dtype=gs.ti_vec3, shape=shape),
        motion_vel=V(dtype=gs.ti_vec3, shape=shape),
        limit=V(dtype=gs.ti_vec2, shape=shape),
        dof_start=V(dtype=gs.ti_int, shape=shape),
        kp=V(dtype=gs.ti_float, shape=shape),
        kv=V(dtype=gs.ti_float, shape=shape),
        force_range=V(dtype=gs.ti_vec2, shape=shape),
    )


@DATA_ORIENTED
class StructDofsState(BASE_CLASS):
    force: V_ANNOTATION
    qf_bias: V_ANNOTATION
    qf_passive: V_ANNOTATION
    qf_actuator: V_ANNOTATION
    qf_applied: V_ANNOTATION
    act_length: V_ANNOTATION
    pos: V_ANNOTATION
    vel: V_ANNOTATION
    vel_prev: V_ANNOTATION
    acc: V_ANNOTATION
    acc_smooth: V_ANNOTATION
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

    return StructDofsState(
        force=V(dtype=gs.ti_float, shape=shape),
        qf_bias=V(dtype=gs.ti_float, shape=shape),
        qf_passive=V(dtype=gs.ti_float, shape=shape),
        qf_actuator=V(dtype=gs.ti_float, shape=shape),
        qf_applied=V(dtype=gs.ti_float, shape=shape),
        act_length=V(dtype=gs.ti_float, shape=shape),
        pos=V(dtype=gs.ti_float, shape=shape),
        vel=V(dtype=gs.ti_float, shape=shape),
        vel_prev=V(dtype=gs.ti_float, shape=shape),
        acc=V(dtype=gs.ti_float, shape=shape),
        acc_smooth=V(dtype=gs.ti_float, shape=shape),
        qf_smooth=V(dtype=gs.ti_float, shape=shape),
        qf_constraint=V(dtype=gs.ti_float, shape=shape),
        cdof_ang=V(dtype=gs.ti_vec3, shape=shape),
        cdof_vel=V(dtype=gs.ti_vec3, shape=shape),
        cdofvel_ang=V(dtype=gs.ti_vec3, shape=shape),
        cdofvel_vel=V(dtype=gs.ti_vec3, shape=shape),
        cdofd_ang=V(dtype=gs.ti_vec3, shape=shape),
        cdofd_vel=V(dtype=gs.ti_vec3, shape=shape),
        f_vel=V(dtype=gs.ti_vec3, shape=shape),
        f_ang=V(dtype=gs.ti_vec3, shape=shape),
        ctrl_force=V(dtype=gs.ti_float, shape=shape),
        ctrl_pos=V(dtype=gs.ti_float, shape=shape),
        ctrl_vel=V(dtype=gs.ti_float, shape=shape),
        ctrl_mode=V(dtype=gs.ti_int, shape=shape),
        hibernated=V(dtype=gs.ti_int, shape=shape),
    )


# =========================================== LinksState and LinksInfo ===========================================


@DATA_ORIENTED
class StructLinksState(BASE_CLASS):
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
    i_pos: V_ANNOTATION
    i_quat: V_ANNOTATION
    j_pos: V_ANNOTATION
    j_quat: V_ANNOTATION
    j_vel: V_ANNOTATION
    j_ang: V_ANNOTATION
    cd_ang: V_ANNOTATION
    cd_vel: V_ANNOTATION
    mass_sum: V_ANNOTATION
    root_COM: V_ANNOTATION  # COM of the kinematic tree
    mass_shift: V_ANNOTATION
    i_pos_shift: V_ANNOTATION
    cacc_ang: V_ANNOTATION
    cacc_lin: V_ANNOTATION
    cfrc_ang: V_ANNOTATION
    cfrc_vel: V_ANNOTATION
    cfrc_applied_ang: V_ANNOTATION
    cfrc_applied_vel: V_ANNOTATION
    contact_force: V_ANNOTATION
    hibernated: V_ANNOTATION


def get_links_state(solver):
    shape = (solver.n_links_, solver._B)

    return StructLinksState(
        cinr_inertial=V(dtype=gs.ti_mat3, shape=shape),
        cinr_pos=V(dtype=gs.ti_vec3, shape=shape),
        cinr_quat=V(dtype=gs.ti_vec4, shape=shape),
        cinr_mass=V(dtype=gs.ti_float, shape=shape),
        crb_inertial=V(dtype=gs.ti_mat3, shape=shape),
        crb_pos=V(dtype=gs.ti_vec3, shape=shape),
        crb_quat=V(dtype=gs.ti_vec4, shape=shape),
        crb_mass=V(dtype=gs.ti_float, shape=shape),
        cdd_vel=V(dtype=gs.ti_vec3, shape=shape),
        cdd_ang=V(dtype=gs.ti_vec3, shape=shape),
        pos=V(dtype=gs.ti_vec3, shape=shape),
        quat=V(dtype=gs.ti_vec4, shape=shape),
        i_pos=V(dtype=gs.ti_vec3, shape=shape),
        i_quat=V(dtype=gs.ti_vec4, shape=shape),
        j_pos=V(dtype=gs.ti_vec3, shape=shape),
        j_quat=V(dtype=gs.ti_vec4, shape=shape),
        j_vel=V(dtype=gs.ti_vec3, shape=shape),
        j_ang=V(dtype=gs.ti_vec3, shape=shape),
        cd_ang=V(dtype=gs.ti_vec3, shape=shape),
        cd_vel=V(dtype=gs.ti_vec3, shape=shape),
        mass_sum=V(dtype=gs.ti_float, shape=shape),
        root_COM=V(dtype=gs.ti_vec3, shape=shape),
        mass_shift=V(dtype=gs.ti_float, shape=shape),
        i_pos_shift=V(dtype=gs.ti_vec3, shape=shape),
        cacc_ang=V(dtype=gs.ti_vec3, shape=shape),
        cacc_lin=V(dtype=gs.ti_vec3, shape=shape),
        cfrc_ang=V(dtype=gs.ti_vec3, shape=shape),
        cfrc_vel=V(dtype=gs.ti_vec3, shape=shape),
        cfrc_applied_ang=V(dtype=gs.ti_vec3, shape=shape),
        cfrc_applied_vel=V(dtype=gs.ti_vec3, shape=shape),
        contact_force=V(dtype=gs.ti_vec3, shape=shape),
        hibernated=V(dtype=gs.ti_int, shape=shape),
    )


@DATA_ORIENTED
class StructLinksInfo(BASE_CLASS):
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


def get_links_info(solver):
    links_info_shape = (solver.n_links_, solver._B) if solver._options.batch_links_info else solver.n_links_

    return StructLinksInfo(
        parent_idx=V(dtype=gs.ti_int, shape=links_info_shape),
        root_idx=V(dtype=gs.ti_int, shape=links_info_shape),
        q_start=V(dtype=gs.ti_int, shape=links_info_shape),
        dof_start=V(dtype=gs.ti_int, shape=links_info_shape),
        joint_start=V(dtype=gs.ti_int, shape=links_info_shape),
        q_end=V(dtype=gs.ti_int, shape=links_info_shape),
        dof_end=V(dtype=gs.ti_int, shape=links_info_shape),
        joint_end=V(dtype=gs.ti_int, shape=links_info_shape),
        n_dofs=V(dtype=gs.ti_int, shape=links_info_shape),
        pos=V(dtype=gs.ti_vec3, shape=links_info_shape),
        quat=V(dtype=gs.ti_vec4, shape=links_info_shape),
        invweight=V(dtype=gs.ti_vec2, shape=links_info_shape),
        is_fixed=V(dtype=gs.ti_bool, shape=links_info_shape),
        inertial_pos=V(dtype=gs.ti_vec3, shape=links_info_shape),
        inertial_quat=V(dtype=gs.ti_vec4, shape=links_info_shape),
        inertial_i=V(dtype=gs.ti_mat3, shape=links_info_shape),
        inertial_mass=V(dtype=gs.ti_float, shape=links_info_shape),
        entity_idx=V(dtype=gs.ti_int, shape=links_info_shape),
    )


# =========================================== JointsInfo and JointsState ===========================================


@DATA_ORIENTED
class StructJointsInfo(BASE_CLASS):
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
        type=V(dtype=gs.ti_int, shape=shape),
        sol_params=V(dtype=gs.ti_vec7, shape=shape),
        q_start=V(dtype=gs.ti_int, shape=shape),
        dof_start=V(dtype=gs.ti_int, shape=shape),
        q_end=V(dtype=gs.ti_int, shape=shape),
        dof_end=V(dtype=gs.ti_int, shape=shape),
        n_dofs=V(dtype=gs.ti_int, shape=shape),
        pos=V(dtype=gs.ti_vec3, shape=shape),
    )


@DATA_ORIENTED
class StructJointsState(BASE_CLASS):
    xanchor: V_ANNOTATION
    xaxis: V_ANNOTATION


def get_joints_state(solver):
    shape = (solver.n_joints_, solver._B)

    return StructJointsState(
        xanchor=V(dtype=gs.ti_vec3, shape=shape),
        xaxis=V(dtype=gs.ti_vec3, shape=shape),
    )


# =========================================== GeomsInfo and GeomsState ===========================================


@DATA_ORIENTED
class StructGeomsInfo(BASE_CLASS):
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
        pos=V(dtype=gs.ti_vec3, shape=shape),
        center=V(dtype=gs.ti_vec3, shape=shape),
        quat=V(dtype=gs.ti_vec4, shape=shape),
        data=V(dtype=gs.ti_vec7, shape=shape),
        link_idx=V(dtype=gs.ti_int, shape=shape),
        type=V(dtype=gs.ti_int, shape=shape),
        friction=V(dtype=gs.ti_float, shape=shape),
        sol_params=V(dtype=gs.ti_vec7, shape=shape),
        vert_num=V(dtype=gs.ti_int, shape=shape),
        vert_start=V(dtype=gs.ti_int, shape=shape),
        vert_end=V(dtype=gs.ti_int, shape=shape),
        verts_state_start=V(dtype=gs.ti_int, shape=shape),
        verts_state_end=V(dtype=gs.ti_int, shape=shape),
        face_num=V(dtype=gs.ti_int, shape=shape),
        face_start=V(dtype=gs.ti_int, shape=shape),
        face_end=V(dtype=gs.ti_int, shape=shape),
        edge_num=V(dtype=gs.ti_int, shape=shape),
        edge_start=V(dtype=gs.ti_int, shape=shape),
        edge_end=V(dtype=gs.ti_int, shape=shape),
        is_convex=V(dtype=gs.ti_bool, shape=shape),
        contype=V(dtype=gs.ti_int, shape=shape),
        conaffinity=V(dtype=gs.ti_int, shape=shape),
        is_fixed=V(dtype=gs.ti_bool, shape=shape),
        is_decomposed=V(dtype=gs.ti_bool, shape=shape),
        needs_coup=V(dtype=gs.ti_int, shape=shape),
        coup_friction=V(dtype=gs.ti_float, shape=shape),
        coup_softness=V(dtype=gs.ti_float, shape=shape),
        coup_restitution=V(dtype=gs.ti_float, shape=shape),
    )


@DATA_ORIENTED
class StructGeomsState(BASE_CLASS):
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
        pos=V(dtype=gs.ti_vec3, shape=shape, needs_grad=requires_grad),
        quat=V(dtype=gs.ti_vec4, shape=shape, needs_grad=requires_grad),
        aabb_min=V(dtype=gs.ti_vec3, shape=shape),
        aabb_max=V(dtype=gs.ti_vec3, shape=shape),
        verts_updated=V(dtype=gs.ti_bool, shape=shape),
        min_buffer_idx=V(dtype=gs.ti_int, shape=shape),
        max_buffer_idx=V(dtype=gs.ti_int, shape=shape),
        hibernated=V(dtype=gs.ti_int, shape=shape),
        friction_ratio=V(dtype=gs.ti_float, shape=shape),
    )


# =========================================== VertsInfo ===========================================


@DATA_ORIENTED
class StructVertsInfo(BASE_CLASS):
    init_pos: V_ANNOTATION
    init_normal: V_ANNOTATION
    geom_idx: V_ANNOTATION
    init_center_pos: V_ANNOTATION
    verts_state_idx: V_ANNOTATION
    is_fixed: V_ANNOTATION


def get_verts_info(solver):
    shape = (solver.n_verts_,)

    return StructVertsInfo(
        init_pos=V(dtype=gs.ti_vec3, shape=shape),
        init_normal=V(dtype=gs.ti_vec3, shape=shape),
        geom_idx=V(dtype=gs.ti_int, shape=shape),
        init_center_pos=V(dtype=gs.ti_vec3, shape=shape),
        verts_state_idx=V(dtype=gs.ti_int, shape=shape),
        is_fixed=V(dtype=gs.ti_bool, shape=shape),
    )


# =========================================== FacesInfo ===========================================


@DATA_ORIENTED
class StructFacesInfo(BASE_CLASS):
    verts_idx: V_ANNOTATION
    geom_idx: V_ANNOTATION


def get_faces_info(solver):
    shape = (solver.n_faces_,)

    return StructFacesInfo(
        verts_idx=V(dtype=gs.ti_ivec3, shape=shape),
        geom_idx=V(dtype=gs.ti_int, shape=shape),
    )


# =========================================== EdgesInfo ===========================================


@DATA_ORIENTED
class StructEdgesInfo(BASE_CLASS):
    v0: V_ANNOTATION
    v1: V_ANNOTATION
    length: V_ANNOTATION


def get_edges_info(solver):
    shape = (solver.n_edges_,)

    return StructEdgesInfo(
        v0=V(dtype=gs.ti_int, shape=shape),
        v1=V(dtype=gs.ti_int, shape=shape),
        length=V(dtype=gs.ti_float, shape=shape),
    )


# =========================================== VertsState ===========================================


@DATA_ORIENTED
class StructVertsState(BASE_CLASS):
    pos: V_ANNOTATION


def get_free_verts_state(solver):
    return StructVertsState(
        pos=V(dtype=gs.ti_vec3, shape=(solver.n_free_verts_, solver._B)),
    )


def get_fixed_verts_state(solver):
    return StructVertsState(
        pos=V(dtype=gs.ti_vec3, shape=(solver.n_fixed_verts_,)),
    )


# =========================================== VvertsInfo ===========================================


@DATA_ORIENTED
class StructVvertsInfo(BASE_CLASS):
    init_pos: V_ANNOTATION
    init_vnormal: V_ANNOTATION
    vgeom_idx: V_ANNOTATION


def get_vverts_info(solver):
    shape = (solver.n_vverts_,)

    return StructVvertsInfo(
        init_pos=V(dtype=gs.ti_vec3, shape=shape),
        init_vnormal=V(dtype=gs.ti_vec3, shape=shape),
        vgeom_idx=V(dtype=gs.ti_int, shape=shape),
    )


# =========================================== VfacesInfo ===========================================


@DATA_ORIENTED
class StructVfacesInfo(BASE_CLASS):
    vverts_idx: V_ANNOTATION
    vgeom_idx: V_ANNOTATION


def get_vfaces_info(solver):
    shape = (solver.n_vfaces_,)

    return StructVfacesInfo(
        vverts_idx=V(dtype=gs.ti_ivec3, shape=shape),
        vgeom_idx=V(dtype=gs.ti_int, shape=shape),
    )


# =========================================== VgeomsInfo ===========================================


@DATA_ORIENTED
class StructVgeomsInfo(BASE_CLASS):
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
        pos=V(dtype=gs.ti_vec3, shape=shape),
        quat=V(dtype=gs.ti_vec4, shape=shape),
        link_idx=V(dtype=gs.ti_int, shape=shape),
        vvert_num=V(dtype=gs.ti_int, shape=shape),
        vvert_start=V(dtype=gs.ti_int, shape=shape),
        vvert_end=V(dtype=gs.ti_int, shape=shape),
        vface_num=V(dtype=gs.ti_int, shape=shape),
        vface_start=V(dtype=gs.ti_int, shape=shape),
        vface_end=V(dtype=gs.ti_int, shape=shape),
        color=V(dtype=gs.ti_vec4, shape=shape),
    )


# =========================================== VGeomsState ===========================================


@DATA_ORIENTED
class StructVgeomsState(BASE_CLASS):
    pos: V_ANNOTATION
    quat: V_ANNOTATION


def get_vgeoms_state(solver):
    shape = (solver.n_vgeoms_, solver._B)

    return StructVgeomsState(
        pos=V(dtype=gs.ti_vec3, shape=shape),
        quat=V(dtype=gs.ti_vec4, shape=shape),
    )


# =========================================== EqualitiesInfo ===========================================


@DATA_ORIENTED
class StructEqualitiesInfo(BASE_CLASS):
    eq_obj1id: V_ANNOTATION
    eq_obj2id: V_ANNOTATION
    eq_data: V_ANNOTATION
    eq_type: V_ANNOTATION
    sol_params: V_ANNOTATION


def get_equalities_info(solver):
    shape = (solver.n_equalities_candidate, solver._B)

    return StructEqualitiesInfo(
        eq_obj1id=V(dtype=gs.ti_int, shape=shape),
        eq_obj2id=V(dtype=gs.ti_int, shape=shape),
        eq_data=V(dtype=gs.ti_vec11, shape=shape),
        eq_type=V(dtype=gs.ti_int, shape=shape),
        sol_params=V(dtype=gs.ti_vec7, shape=shape),
    )


# =========================================== EntitiesInfo ===========================================


@DATA_ORIENTED
class StructEntitiesInfo(BASE_CLASS):
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
        dof_start=V(dtype=gs.ti_int, shape=shape),
        dof_end=V(dtype=gs.ti_int, shape=shape),
        n_dofs=V(dtype=gs.ti_int, shape=shape),
        link_start=V(dtype=gs.ti_int, shape=shape),
        link_end=V(dtype=gs.ti_int, shape=shape),
        n_links=V(dtype=gs.ti_int, shape=shape),
        geom_start=V(dtype=gs.ti_int, shape=shape),
        geom_end=V(dtype=gs.ti_int, shape=shape),
        n_geoms=V(dtype=gs.ti_int, shape=shape),
        gravity_compensation=V(dtype=gs.ti_float, shape=shape),
        is_local_collision_mask=V(dtype=gs.ti_bool, shape=shape),
    )


# =========================================== EntitiesState ===========================================


@DATA_ORIENTED
class StructEntitiesState(BASE_CLASS):
    hibernated: V_ANNOTATION


def get_entities_state(solver):
    return StructEntitiesState(
        hibernated=V(dtype=gs.ti_int, shape=(solver.n_entities_, solver._B)),
    )


# =================================== StructRigidSimStaticConfig ===================================


@ti.data_oriented
class StructRigidSimStaticConfig:  # (NamedTuple):
    para_level: int
    requires_grad: bool
    use_hibernation: bool
    batch_links_info: bool
    batch_dofs_info: bool
    batch_joints_info: bool
    enable_mujoco_compatibility: bool
    enable_multi_contact: bool
    enable_collision: bool
    box_box_detection: bool
    sparse_solve: bool
    integrator: int
    solver_type: int

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


# =========================================== DataManager ===========================================


@ti.data_oriented
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


DofsState = StructDofsState if gs.use_ndarray else ti.template()
DofsInfo = StructDofsInfo if gs.use_ndarray else ti.template()
GeomsState = StructGeomsState if gs.use_ndarray else ti.template()
GeomsInfo = StructGeomsInfo if gs.use_ndarray else ti.template()
GeomsInitAABB = V_ANNOTATION
LinksState = StructLinksState if gs.use_ndarray else ti.template()
LinksInfo = StructLinksInfo if gs.use_ndarray else ti.template()
JointsInfo = StructJointsInfo if gs.use_ndarray else ti.template()
JointsState = StructJointsState if gs.use_ndarray else ti.template()
VertsState = StructVertsState if gs.use_ndarray else ti.template()
VertsInfo = StructVertsInfo if gs.use_ndarray else ti.template()
EdgesInfo = StructEdgesInfo if gs.use_ndarray else ti.template()
FacesInfo = StructFacesInfo if gs.use_ndarray else ti.template()
VVertsInfo = StructVvertsInfo if gs.use_ndarray else ti.template()
VFacesInfo = StructVfacesInfo if gs.use_ndarray else ti.template()
VGeomsInfo = StructVgeomsInfo if gs.use_ndarray else ti.template()
VGeomsState = StructVgeomsState if gs.use_ndarray else ti.template()
EntitiesState = StructEntitiesState if gs.use_ndarray else ti.template()
EntitiesInfo = StructEntitiesInfo if gs.use_ndarray else ti.template()
EqualitiesInfo = StructEqualitiesInfo if gs.use_ndarray else ti.template()
RigidGlobalInfo = StructRigidGlobalInfo if gs.use_ndarray else ti.template()
ColliderState = StructColliderState if gs.use_ndarray else ti.template()
ColliderInfo = StructColliderInfo if gs.use_ndarray else ti.template()
MPRState = StructMPRState if gs.use_ndarray else ti.template()
SupportFieldInfo = StructSupportFieldInfo if gs.use_ndarray else ti.template()
ConstraintState = StructConstraintState if gs.use_ndarray else ti.template()
GJKState = StructGJKState if gs.use_ndarray else ti.template()
SDFInfo = StructSDFInfo if gs.use_ndarray else ti.template()
ContactIslandState = StructContactIslandState if gs.use_ndarray else ti.template()
DiffContactInput = StructDiffContactInput if gs.use_ndarray else ti.template()
