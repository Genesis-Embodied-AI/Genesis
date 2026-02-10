"""
Thread-local state functions for narrow-phase collision detection.

This module provides thread-local versions of collision detection functions that use
per-thread copies of geometry state instead of modifying shared global state. This
enables race-free parallelization across collision pairs within the same environment.

The key difference from the original narrowphase.py functions is that geometry perturbations
for multi-contact detection are performed on thread-local copies (stored in registers),
preventing intra-environment race conditions when multiple threads process different
collision pairs involving the same geometry.

## Functions Included

This module only contains functions in the multi-contact perturbation code path:

- `func_convex_convex_contact_local`: Thread-local version of the main convex-convex
  collision detection function that applies geometry perturbations for multi-contact.

## Functions NOT Included (and why)

The following functions from narrowphase.py do not need local versions:

### SDF Collision Functions

- `func_contact_sphere_sdf`
- `func_contact_vertex_sdf`
- `func_contact_edge_sdf`
- `func_contact_convex_convex_sdf`

**Why not localized:**
- Use signed distance fields (SDF) for collision detection
- No geometry perturbations - single contact point only
- Already thread-safe

### Terrain Collision Functions

- `func_contact_mpr_terrain`
- `func_add_prism_vert`

**Why not localized:**
- Terrain is static/global geometry
- No multi-contact perturbations for terrain
- Already thread-safe

### High-Level Dispatcher Functions

- `func_narrow_phase_convex_vs_convex`
- `func_narrow_phase_diff_convex_vs_convex`
- `func_narrow_phase_convex_specializations`
- `func_narrow_phase_any_vs_terrain`
- `func_narrow_phase_nonconvex_vs_nonterrain`

**Why not localized:**
- These are kernel entry points that loop over collision pairs
- They call `func_convex_convex_contact` (or its local version) internally
- The dispatchers themselves don't touch geometry state
- When using the local version, the dispatcher would call
  `func_convex_convex_contact_local` instead of `func_convex_convex_contact`

## Usage

To enable collision-level parallelization, use the local function instead of the
original when parallelizing over (env, collision) pairs:

```python
# Original: Parallelizes over environments only
for i_b, i_pair in ti.ndrange(n_envs, n_possible_pairs):
    func_convex_convex_contact(...)  # May have race conditions

# Thread-safe: Parallelizes over (env, collision) pairs
for i_b, i_col in ti.ndrange(n_envs, max_collisions_per_env):
    func_convex_convex_contact_local(...)  # Thread-safe!
```
"""

import sys

import gstaichi as ti

import genesis as gs
from genesis.engine.solvers.rigid.collider import (
    diff_gjk_local,
    gjk_local,
    mpr_local,
)
from genesis.engine.solvers.rigid.collider.box_contact import func_plane_box_contact
from genesis.engine.solvers.rigid.collider.contact import (
    func_add_contact,
    func_add_diff_contact_input,
    func_compute_tolerance,
    func_contact_orthogonals,
)
from genesis.engine.solvers.rigid.collider.contact_local import func_rotate_frame_local
from genesis.engine.solvers.rigid.collider import narrowphase
from genesis.utils import array_class, sdf_local
import genesis.utils.geom as gu


@ti.func
def func_contact_vertex_sdf_local(
    i_ga,
    i_gb,
    i_b,
    ga_pos: ti.types.vector(3, dtype=gs.ti_float),
    ga_quat: ti.types.vector(4, dtype=gs.ti_float),
    gb_pos: ti.types.vector(3, dtype=gs.ti_float),
    gb_quat: ti.types.vector(4, dtype=gs.ti_float),
    geoms_state: array_class.GeomsState,
    geoms_info: array_class.GeomsInfo,
    verts_info: array_class.VertsInfo,
    rigid_global_info: array_class.RigidGlobalInfo,
    collider_static_config: ti.template(),
    sdf_info: array_class.SDFInfo,
):
    """
    Thread-local version of func_contact_vertex_sdf.

    Detects collision between a vertex-based geometry (geom A) and an SDF-based geometry (geom B),
    using provided thread-local geometry poses instead of reading from geoms_state.

    Args:
        i_ga: Geometry A index
        i_gb: Geometry B index
        i_b: Batch index
        ga_pos: Thread-local geometry A position
        ga_quat: Thread-local geometry A quaternion
        gb_pos: Thread-local geometry B position
        gb_quat: Thread-local geometry B quaternion
        geoms_state: Global geometry state (for AABB only)
        geoms_info: Geometry information
        verts_info: Vertex information
        rigid_global_info: Global rigid body information
        collider_static_config: Collider static configuration
        sdf_info: SDF information

    Returns:
        is_col: Whether collision is detected
        normal: Contact normal
        penetration: Penetration depth
        contact_pos: Contact position
    """
    is_col = False
    penetration = gs.ti_float(0.0)
    normal = ti.Vector.zero(gs.ti_float, 3)
    contact_pos = ti.Vector.zero(gs.ti_float, 3)

    for i_v in range(geoms_info.vert_start[i_ga], geoms_info.vert_end[i_ga]):
        vertex_pos = gu.ti_transform_by_trans_quat(verts_info.init_pos[i_v], ga_pos, ga_quat)
        if narrowphase.func_point_in_geom_aabb(i_gb, i_b, geoms_state, geoms_info, vertex_pos):
            new_penetration = -sdf_local.sdf_func_world_local(
                geoms_info=geoms_info,
                sdf_info=sdf_info,
                pos_world=vertex_pos,
                geom_idx=i_gb,
                geom_pos=gb_pos,
                geom_quat=gb_quat,
            )
            if new_penetration > penetration:
                is_col = True
                contact_pos = vertex_pos
                penetration = new_penetration

    if is_col:
        # Compute contact normal only once, and only in case of contact
        normal = sdf_local.sdf_func_normal_world_local(
            geoms_info=geoms_info,
            rigid_global_info=rigid_global_info,
            collider_static_config=collider_static_config,
            sdf_info=sdf_info,
            pos_world=contact_pos,
            geom_idx=i_gb,
            geom_pos=gb_pos,
            geom_quat=gb_quat,
        )

        # The contact point must be offsetted by half the penetration depth
        contact_pos = contact_pos + 0.5 * penetration * normal

    return is_col, normal, penetration, contact_pos


@ti.func
def func_convex_convex_contact_local(
    i_ga,
    i_gb,
    i_b,
    links_state: array_class.LinksState,
    links_info: array_class.LinksInfo,
    geoms_state: array_class.GeomsState,
    geoms_info: array_class.GeomsInfo,
    geoms_init_AABB: array_class.GeomsInitAABB,
    verts_info: array_class.VertsInfo,
    faces_info: array_class.FacesInfo,
    edges_info: array_class.EdgesInfo,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: ti.template(),
    collider_state: array_class.ColliderState,
    collider_info: array_class.ColliderInfo,
    collider_static_config: ti.template(),
    mpr_state: array_class.MPRState,
    mpr_info: array_class.MPRInfo,
    gjk_state: array_class.GJKState,
    gjk_info: array_class.GJKInfo,
    gjk_static_config: ti.template(),
    sdf_info: array_class.SDFInfo,
    support_field_info: array_class.SupportFieldInfo,
    diff_contact_input: array_class.DiffContactInput,
    errno: array_class.V_ANNOTATION,
):
    """
    Thread-local version of func_convex_convex_contact.

    This function uses thread-local copies of geometry state for multi-contact perturbations,
    preventing race conditions when multiple threads process collisions involving the same
    geometry in the same environment.

    Key approach:
    1. Load original geometry state into thread-local variables
    2. For multi-contact perturbations, update the thread-local copies
    3. Pass thread-local pos/quat directly to _local collision detection functions
    4. These functions never read from or write to geoms_state.pos/quat

    This eliminates all intra-environment race conditions on geometry state.
    """
    if geoms_info.type[i_ga] == gs.GEOM_TYPE.PLANE and geoms_info.type[i_gb] == gs.GEOM_TYPE.BOX:
        # Plane-box collision doesn't use perturbations, so call original function
        if ti.static(sys.platform == "darwin"):
            func_plane_box_contact(
                i_ga=i_ga,
                i_gb=i_gb,
                i_b=i_b,
                geoms_state=geoms_state,
                geoms_info=geoms_info,
                geoms_init_AABB=geoms_init_AABB,
                verts_info=verts_info,
                static_rigid_sim_config=static_rigid_sim_config,
                collider_state=collider_state,
                collider_info=collider_info,
                collider_static_config=collider_static_config,
                errno=errno,
            )
    else:
        EPS = rigid_global_info.EPS[None]

        # Disabling multi-contact for pairs of decomposed geoms would speed up simulation but may cause physical
        # instabilities in the few cases where multiple contact points are actually need. Increasing the tolerance
        # criteria to get rid of redundant contact points seems to be a better option.
        multi_contact = (
            static_rigid_sim_config.enable_multi_contact
            # and not (self._solver.geoms_info[i_ga].is_decomposed and self._solver.geoms_info[i_gb].is_decomposed)
            and geoms_info.type[i_ga] != gs.GEOM_TYPE.SPHERE
            and geoms_info.type[i_ga] != gs.GEOM_TYPE.ELLIPSOID
            and geoms_info.type[i_gb] != gs.GEOM_TYPE.SPHERE
            and geoms_info.type[i_gb] != gs.GEOM_TYPE.ELLIPSOID
        )

        tolerance = func_compute_tolerance(
            i_ga, i_gb, i_b, collider_info.mc_tolerance[None], geoms_info, geoms_init_AABB
        )
        diff_pos_tolerance = func_compute_tolerance(
            i_ga, i_gb, i_b, collider_info.diff_pos_tolerance[None], geoms_info, geoms_init_AABB
        )
        diff_normal_tolerance = collider_info.diff_normal_tolerance[None]

        # Load original geometry state into thread-local variables
        # These are the UNPERTURBED states used as reference point for each independent perturbation
        ga_pos_original = geoms_state.pos[i_ga, i_b]
        ga_quat_original = geoms_state.quat[i_ga, i_b]
        gb_pos_original = geoms_state.pos[i_gb, i_b]
        gb_quat_original = geoms_state.quat[i_gb, i_b]

        # Current (possibly perturbed) state - initialized to original, updated during perturbations
        ga_pos_current = ga_pos_original
        ga_quat_current = ga_quat_original
        gb_pos_current = gb_pos_original
        gb_quat_current = gb_quat_original

        # Pre-allocate buffers
        is_col_0 = False
        penetration_0 = gs.ti_float(0.0)
        normal_0 = ti.Vector.zero(gs.ti_float, 3)
        contact_pos_0 = ti.Vector.zero(gs.ti_float, 3)

        is_col = False
        penetration = gs.ti_float(0.0)
        normal = ti.Vector.zero(gs.ti_float, 3)
        contact_pos = ti.Vector.zero(gs.ti_float, 3)

        n_con = gs.ti_int(0)
        axis_0 = ti.Vector.zero(gs.ti_float, 3)
        axis_1 = ti.Vector.zero(gs.ti_float, 3)
        qrot = ti.Vector.zero(gs.ti_float, 4)

        i_pair = collider_info.collision_pair_idx[(i_gb, i_ga) if i_ga > i_gb else (i_ga, i_gb)]

        for i_detection in range(5):
            prefer_gjk = (
                collider_static_config.ccd_algorithm == narrowphase.CCD_ALGORITHM_CODE.GJK
                or collider_static_config.ccd_algorithm == narrowphase.CCD_ALGORITHM_CODE.MJ_GJK
            )

            # Apply perturbations to thread-local state
            if multi_contact and is_col_0:
                # Perturbation axis must not be aligned with the principal axes of inertia the geometry,
                # otherwise it would be more sensitive to ill-conditionning.
                axis = (2 * (i_detection % 2) - 1) * axis_0 + (1 - 2 * ((i_detection // 2) % 2)) * axis_1
                qrot = gu.ti_rotvec_to_quat(collider_info.mc_perturbation[None] * axis, EPS)

                # Apply perturbation starting from original state
                ga_result = func_rotate_frame_local(
                    pos=ga_pos_original, quat=ga_quat_original, contact_pos=contact_pos_0, qrot=qrot
                )
                ga_pos_current = ga_result.pos
                ga_quat_current = ga_result.quat

                gb_result = func_rotate_frame_local(
                    pos=gb_pos_original, quat=gb_quat_original, contact_pos=contact_pos_0, qrot=gu.ti_inv_quat(qrot)
                )
                gb_pos_current = gb_result.pos
                gb_quat_current = gb_result.quat

            if (multi_contact and is_col_0) or (i_detection == 0):
                if geoms_info.type[i_ga] == gs.GEOM_TYPE.PLANE:
                    # Plane collision detection
                    plane_dir = ti.Vector(
                        [geoms_info.data[i_ga][0], geoms_info.data[i_ga][1], geoms_info.data[i_ga][2]], dt=gs.ti_float
                    )
                    plane_dir = gu.ti_transform_by_quat(plane_dir, ga_quat_current)
                    normal = -plane_dir.normalized()

                    # Use thread-local support driver
                    v1 = mpr_local.support_driver_local(
                        geoms_info=geoms_info,
                        collider_state=collider_state,
                        collider_static_config=collider_static_config,
                        support_field_info=support_field_info,
                        direction=normal,
                        i_g=i_gb,
                        i_b=i_b,
                        pos=gb_pos_current,
                        quat=gb_quat_current,
                    )
                    penetration = normal.dot(v1 - ga_pos_current)
                    contact_pos = v1 - 0.5 * penetration * normal
                    is_col = penetration > 0.0
                else:
                    ### MPR, MJ_MPR
                    if ti.static(
                        collider_static_config.ccd_algorithm
                        in (narrowphase.CCD_ALGORITHM_CODE.MPR, narrowphase.CCD_ALGORITHM_CODE.MJ_MPR)
                    ):
                        # Try using MPR before anything else
                        is_mpr_updated = False
                        normal_ws = collider_state.contact_cache.normal[i_pair, i_b]
                        is_mpr_guess_direction_available = (ti.abs(normal_ws) > EPS).any()

                        for i_mpr in range(2):
                            if i_mpr == 1:
                                # Try without warm-start if no contact was detected using it.
                                # When penetration depth is very shallow, MPR may wrongly classify two geometries as not
                                # in contact while they actually are. This helps to improve contact persistence without
                                # increasing much the overall computational cost since the fallback should not be
                                # triggered very often.
                                if ti.static(not static_rigid_sim_config.enable_mujoco_compatibility):
                                    if (i_detection == 0) and not is_col and is_mpr_guess_direction_available:
                                        normal_ws = ti.Vector.zero(gs.ti_float, 3)
                                        is_mpr_guess_direction_available = False
                                        is_mpr_updated = False

                            if not is_mpr_updated:
                                # Use thread-local MPR function
                                is_col, normal, penetration, contact_pos = mpr_local.func_mpr_contact_local(
                                    geoms_info=geoms_info,
                                    geoms_init_AABB=geoms_init_AABB,
                                    rigid_global_info=rigid_global_info,
                                    static_rigid_sim_config=static_rigid_sim_config,
                                    collider_state=collider_state,
                                    collider_static_config=collider_static_config,
                                    mpr_state=mpr_state,
                                    mpr_info=mpr_info,
                                    support_field_info=support_field_info,
                                    i_ga=i_ga,
                                    i_gb=i_gb,
                                    i_b=i_b,
                                    normal_ws=normal_ws,
                                    pos_a=ga_pos_current,
                                    quat_a=ga_quat_current,
                                    pos_b=gb_pos_current,
                                    quat_b=gb_quat_current,
                                )
                                is_mpr_updated = True

                        # Fallback on GJK if collision is detected by MPR if the initial penetration is already quite
                        # large, and either no collision direction was cached or the geometries have large overlap. This
                        # contact information provided by MPR may be unreliable in these cases.
                        if ti.static(collider_static_config.ccd_algorithm == narrowphase.CCD_ALGORITHM_CODE.MPR):
                            if penetration > tolerance:
                                prefer_gjk = not is_mpr_guess_direction_available or (
                                    collider_info.mc_tolerance[None] * penetration
                                    >= collider_info.mpr_to_gjk_overlap_ratio[None] * tolerance
                                )

                    ### GJK, MJ_GJK
                    if ti.static(collider_static_config.ccd_algorithm != narrowphase.CCD_ALGORITHM_CODE.MJ_MPR):
                        if prefer_gjk:
                            if ti.static(static_rigid_sim_config.requires_grad):
                                # Use thread-local differentiable GJK function directly
                                diff_gjk_local.func_gjk_contact_local(
                                    links_state,
                                    links_info,
                                    geoms_state,
                                    geoms_info,
                                    geoms_init_AABB,
                                    verts_info,
                                    faces_info,
                                    rigid_global_info,
                                    static_rigid_sim_config,
                                    collider_state,
                                    collider_static_config,
                                    gjk_state,
                                    gjk_info,
                                    support_field_info,
                                    diff_contact_input,
                                    i_ga,
                                    i_gb,
                                    i_b,
                                    ga_pos_current,
                                    ga_quat_current,
                                    gb_pos_current,
                                    gb_quat_current,
                                    diff_pos_tolerance,
                                    diff_normal_tolerance,
                                )
                            else:
                                # Use thread-local non-differentiable GJK function
                                gjk_local.func_gjk_contact_local(
                                    geoms_state=geoms_state,
                                    geoms_info=geoms_info,
                                    verts_info=verts_info,
                                    faces_info=faces_info,
                                    rigid_global_info=rigid_global_info,
                                    static_rigid_sim_config=static_rigid_sim_config,
                                    collider_state=collider_state,
                                    collider_static_config=collider_static_config,
                                    gjk_state=gjk_state,
                                    gjk_info=gjk_info,
                                    gjk_static_config=gjk_static_config,
                                    support_field_info=support_field_info,
                                    i_ga=i_ga,
                                    i_gb=i_gb,
                                    i_b=i_b,
                                    pos_a=ga_pos_current,
                                    quat_a=ga_quat_current,
                                    pos_b=gb_pos_current,
                                    quat_b=gb_quat_current,
                                )

                            is_col = gjk_state.is_col[i_b] == 1
                            penetration = gjk_state.penetration[i_b]
                            n_contacts = gjk_state.n_contacts[i_b]

                            if is_col:
                                if ti.static(static_rigid_sim_config.requires_grad):
                                    for i_c in range(n_contacts):
                                        func_add_diff_contact_input(
                                            i_ga,
                                            i_gb,
                                            i_b,
                                            i_c,
                                            gjk_state,
                                            collider_state,
                                            collider_info,
                                        )
                                        func_add_contact(
                                            i_ga,
                                            i_gb,
                                            gjk_state.normal[i_b, i_c],
                                            gjk_state.contact_pos[i_b, i_c],
                                            gjk_state.diff_penetration[i_b, i_c],
                                            i_b,
                                            geoms_state,
                                            geoms_info,
                                            collider_state,
                                            collider_info,
                                            errno,
                                        )
                                    break
                                else:
                                    if gjk_state.multi_contact_flag[i_b]:
                                        # Since we already found multiple contact points, add the discovered contact
                                        # points and stop multi-contact search.
                                        for i_c in range(n_contacts):
                                            # Ignore contact points if the number of contacts exceeds the limit.
                                            if i_c < ti.static(collider_static_config.n_contacts_per_pair):
                                                contact_pos = gjk_state.contact_pos[i_b, i_c]
                                                normal = gjk_state.normal[i_b, i_c]
                                                if ti.static(static_rigid_sim_config.requires_grad):
                                                    penetration = gjk_state.diff_penetration[i_b, i_c]
                                                func_add_contact(
                                                    i_ga,
                                                    i_gb,
                                                    normal,
                                                    contact_pos,
                                                    penetration,
                                                    i_b,
                                                    geoms_state,
                                                    geoms_info,
                                                    collider_state,
                                                    collider_info,
                                                    errno,
                                                )

                                        break
                                    else:
                                        contact_pos = gjk_state.contact_pos[i_b, 0]
                                        normal = gjk_state.normal[i_b, 0]

            # First detection: save results and add contact if found
            if i_detection == 0:
                is_col_0, normal_0, penetration_0, contact_pos_0 = is_col, normal, penetration, contact_pos
                if is_col_0:
                    func_add_contact(
                        i_ga,
                        i_gb,
                        normal_0,
                        contact_pos_0,
                        penetration_0,
                        i_b,
                        geoms_state,
                        geoms_info,
                        collider_state,
                        collider_info,
                        errno,
                    )
                    if multi_contact:
                        # Compute perturbation axes for subsequent detections
                        axis_0, axis_1 = func_contact_orthogonals(
                            i_ga,
                            i_gb,
                            normal,
                            i_b,
                            links_state,
                            links_info,
                            geoms_state,
                            geoms_info,
                            geoms_init_AABB,
                            rigid_global_info,
                            static_rigid_sim_config,
                        )
                        n_con = 1

                    if ti.static(
                        collider_static_config.ccd_algorithm
                        in (narrowphase.CCD_ALGORITHM_CODE.MPR, narrowphase.CCD_ALGORITHM_CODE.GJK)
                    ):
                        collider_state.contact_cache.normal[i_pair, i_b] = normal
                else:
                    # Clear collision normal cache if not in contact
                    collider_state.contact_cache.normal[i_pair, i_b] = ti.Vector.zero(gs.ti_float, 3)

            # Subsequent detections: correct contact position and add if valid
            elif multi_contact and is_col_0 > 0 and is_col > 0:
                if ti.static(
                    collider_static_config.ccd_algorithm
                    in (narrowphase.CCD_ALGORITHM_CODE.MPR, narrowphase.CCD_ALGORITHM_CODE.GJK)
                ):
                    # Project contact points and correct for perturbation
                    contact_point_a = (
                        gu.ti_transform_by_quat(
                            (contact_pos - 0.5 * penetration * normal) - contact_pos_0,
                            gu.ti_inv_quat(qrot),
                        )
                        + contact_pos_0
                    )
                    contact_point_b = (
                        gu.ti_transform_by_quat(
                            (contact_pos + 0.5 * penetration * normal) - contact_pos_0,
                            qrot,
                        )
                        + contact_pos_0
                    )
                    contact_pos = 0.5 * (contact_point_a + contact_point_b)

                    # First-order correction of the normal direction.
                    # The way the contact normal gets twisted by applying perturbation of geometry poses is
                    # unpredictable as it depends on the final portal discovered by MPR. Alternatively, let compute
                    # the mininal rotation that makes the corrected twisted normal as closed as possible to the
                    # original one, up to the scale of the perturbation, then apply first-order Taylor expension of
                    # Rodrigues' rotation formula.
                    twist_rotvec = ti.math.clamp(
                        normal.cross(normal_0),
                        -collider_info.mc_perturbation[None],
                        collider_info.mc_perturbation[None],
                    )
                    normal = normal + twist_rotvec.cross(normal)

                    # Make sure that the penetration is still positive before adding contact point.
                    # Note that adding some negative tolerance improves physical stability by encouraging persistent
                    # contact points and thefore more continuous contact forces, without changing the mean-field
                    # dynamics since zero-penetration contact points should not induce any force.
                    penetration = normal.dot(contact_point_b - contact_point_a)
                if ti.static(collider_static_config.ccd_algorithm == narrowphase.CCD_ALGORITHM_CODE.MJ_GJK):
                    # Only change penetration to the initial one, because the normal vector could change abruptly
                    # under MuJoCo's GJK-EPA.
                    penetration = penetration_0

                # Discard contact point is repeated
                repeated = False
                for i_c in range(n_con):
                    if not repeated:
                        idx_prev = collider_state.n_contacts[i_b] - 1 - i_c
                        prev_contact = collider_state.contact_data.pos[idx_prev, i_b]
                        if (contact_pos - prev_contact).norm() < tolerance:
                            repeated = True

                if not repeated:
                    if penetration > -tolerance:
                        penetration = ti.max(penetration, 0.0)
                        func_add_contact(
                            i_ga,
                            i_gb,
                            normal,
                            contact_pos,
                            penetration,
                            i_b,
                            geoms_state,
                            geoms_info,
                            collider_state,
                            collider_info,
                            errno,
                        )
                        n_con = n_con + 1


@ti.func
def func_contact_edge_sdf_local(
    i_ga,
    i_gb,
    i_b,
    ga_pos: ti.types.vector(3, dtype=gs.ti_float),
    ga_quat: ti.types.vector(4, dtype=gs.ti_float),
    gb_pos: ti.types.vector(3, dtype=gs.ti_float),
    gb_quat: ti.types.vector(4, dtype=gs.ti_float),
    geoms_state: array_class.GeomsState,  # For AABB only
    geoms_info: array_class.GeomsInfo,
    verts_info: array_class.VertsInfo,
    edges_info: array_class.EdgesInfo,
    rigid_global_info: array_class.RigidGlobalInfo,
    collider_static_config: ti.template(),
    sdf_info: array_class.SDFInfo,
):
    """
    Thread-local version: Detect collision between edge-based geometry and SDF-based geometry.

    This function accepts geometry poses as explicit parameters instead of reading from global state,
    enabling thread-safe perturbations for multi-contact detection.
    """
    EPS = rigid_global_info.EPS[None]

    is_col = False
    penetration = gs.ti_float(0.0)
    normal = ti.Vector.zero(gs.ti_float, 3)
    contact_pos = ti.Vector.zero(gs.ti_float, 3)

    ga_sdf_cell_size = sdf_info.geoms_info.sdf_cell_size[i_ga]

    for i_e in range(geoms_info.edge_start[i_ga], geoms_info.edge_end[i_ga]):
        cur_length = edges_info.length[i_e]
        if cur_length > ga_sdf_cell_size:
            i_v0 = edges_info.v0[i_e]
            i_v1 = edges_info.v1[i_e]

            p_0 = gu.ti_transform_by_trans_quat(verts_info.init_pos[i_v0], ga_pos, ga_quat)
            p_1 = gu.ti_transform_by_trans_quat(verts_info.init_pos[i_v1], ga_pos, ga_quat)
            vec_01 = gu.ti_normalize(p_1 - p_0, EPS)

            sdf_grad_0_b = sdf_local.sdf_func_grad_world_local(
                geoms_info=geoms_info,
                rigid_global_info=rigid_global_info,
                collider_static_config=collider_static_config,
                sdf_info=sdf_info,
                pos_world=p_0,
                geom_idx=i_gb,
                geom_pos=gb_pos,
                geom_quat=gb_quat,
            )
            sdf_grad_1_b = sdf_local.sdf_func_grad_world_local(
                geoms_info=geoms_info,
                rigid_global_info=rigid_global_info,
                collider_static_config=collider_static_config,
                sdf_info=sdf_info,
                pos_world=p_1,
                geom_idx=i_gb,
                geom_pos=gb_pos,
                geom_quat=gb_quat,
            )

            # check if the edge on a is facing towards mesh b
            sdf_grad_0_a = sdf_local.sdf_func_grad_world_local(
                geoms_info=geoms_info,
                rigid_global_info=rigid_global_info,
                collider_static_config=collider_static_config,
                sdf_info=sdf_info,
                pos_world=p_0,
                geom_idx=i_ga,
                geom_pos=ga_pos,
                geom_quat=ga_quat,
            )
            sdf_grad_1_a = sdf_local.sdf_func_grad_world_local(
                geoms_info=geoms_info,
                rigid_global_info=rigid_global_info,
                collider_static_config=collider_static_config,
                sdf_info=sdf_info,
                pos_world=p_1,
                geom_idx=i_ga,
                geom_pos=ga_pos,
                geom_quat=ga_quat,
            )
            normal_edge_0 = sdf_grad_0_a - sdf_grad_0_a.dot(vec_01) * vec_01
            normal_edge_1 = sdf_grad_1_a - sdf_grad_1_a.dot(vec_01) * vec_01

            if normal_edge_0.dot(sdf_grad_0_b) < 0 or normal_edge_1.dot(sdf_grad_1_b) < 0:
                # check if closest point is between the two points
                if sdf_grad_0_b.dot(vec_01) < 0 and sdf_grad_1_b.dot(vec_01) > 0:
                    while cur_length > ga_sdf_cell_size:
                        p_mid = 0.5 * (p_0 + p_1)
                        if (
                            sdf_local.sdf_func_grad_world_local(
                                geoms_info=geoms_info,
                                rigid_global_info=rigid_global_info,
                                collider_static_config=collider_static_config,
                                sdf_info=sdf_info,
                                pos_world=p_mid,
                                geom_idx=i_gb,
                                geom_pos=gb_pos,
                                geom_quat=gb_quat,
                            ).dot(vec_01)
                            < 0
                        ):
                            p_0 = p_mid
                        else:
                            p_1 = p_mid
                        cur_length = 0.5 * cur_length

                    p = 0.5 * (p_0 + p_1)
                    new_penetration = -sdf_local.sdf_func_world_local(
                        geoms_info=geoms_info,
                        sdf_info=sdf_info,
                        pos_world=p,
                        geom_idx=i_gb,
                        geom_pos=gb_pos,
                        geom_quat=gb_quat,
                    )

                    if new_penetration > penetration:
                        is_col = True
                        normal = sdf_local.sdf_func_normal_world_local(
                            geoms_info=geoms_info,
                            rigid_global_info=rigid_global_info,
                            collider_static_config=collider_static_config,
                            sdf_info=sdf_info,
                            pos_world=p,
                            geom_idx=i_gb,
                            geom_pos=gb_pos,
                            geom_quat=gb_quat,
                        )
                        contact_pos = p
                        penetration = new_penetration

    # The contact point must be offsetted by half the penetration depth, for consistency with MPR
    contact_pos = contact_pos + 0.5 * penetration * normal

    return is_col, normal, penetration, contact_pos
