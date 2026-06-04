"""
Narrow-phase collision detection functions.

This module contains SDF-based contact detection, convex-convex contact,
terrain detection, box-box contact, and multi-contact search algorithms.
"""

from enum import IntEnum

import quadrants as qd

import genesis as gs
import genesis.utils.array_class as array_class
import genesis.utils.geom as gu
import genesis.utils.sdf as sdf

from . import capsule_contact, diff_gjk, gjk, mpr
from .box_contact import (
    func_box_box_contact,
    func_plane_box_contact,
    func_sphere_box_contact,
)
from .contact import (
    func_add_contact,
    func_add_diff_contact_input,
    func_apply_smooth_refinement,
    func_compute_mj_tolerance,
    func_compute_tolerance,
    func_contact_orthogonals,
    func_rotate_frame,
    func_set_contact,
)
from .utils import func_point_in_geom_aabb


class CCD_ALGORITHM_CODE(IntEnum):
    """Convex collision detection algorithm codes."""

    # Our MPR (with SDF)
    MPR = 0
    # MuJoCo MPR
    MJ_MPR = 1
    # Our GJK
    GJK = 2
    # MuJoCo GJK
    MJ_GJK = 3


@qd.func
def func_contact_sphere_sdf(
    i_ga,
    i_gb,
    i_b,
    geoms_state: array_class.GeomsState,
    geoms_info: array_class.GeomsInfo,
    rigid_global_info: array_class.RigidGlobalInfo,
    collider_static_config: qd.template(),
    sdf_info: array_class.SDFInfo,
):
    is_col = False
    penetration = gs.qd_float(0.0)
    normal = qd.Vector.zero(gs.qd_float, 3)
    contact_pos = qd.Vector.zero(gs.qd_float, 3)

    sphere_center = geoms_state.pos[i_ga, i_b]
    sphere_radius = geoms_info.data[i_ga][0]

    center_to_b_dist = sdf.sdf_func_world(geoms_state, geoms_info, sdf_info, sphere_center, i_gb, i_b)
    if center_to_b_dist < sphere_radius:
        is_col = True
        normal = sdf.sdf_func_normal_world(
            geoms_state, geoms_info, rigid_global_info, collider_static_config, sdf_info, sphere_center, i_gb, i_b
        )
        penetration = sphere_radius - center_to_b_dist
        contact_pos = sphere_center - (sphere_radius - 0.5 * penetration) * normal

    return is_col, normal, penetration, contact_pos


@qd.func
def func_add_polytope_vertex_contacts_sdf(
    i_ga,
    i_gb,
    i_b,
    i_pair,
    ga_pos: qd.types.vector(3),
    ga_quat: qd.types.vector(4),
    gb_pos: qd.types.vector(3),
    gb_quat: qd.types.vector(4),
    tolerance,
    geoms_state: array_class.GeomsState,
    geoms_info: array_class.GeomsInfo,
    geoms_init_AABB: array_class.GeomsInitAABB,
    verts_info: array_class.VertsInfo,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: qd.template(),
    collider_static_config: qd.template(),
    sdf_info: array_class.SDFInfo,
    collider_state: array_class.ColliderState,
    collider_info: array_class.ColliderInfo,
    errno: qd.Tensor,
):
    # Emit up to n_max contacts at the deepest spatially-diverse vertices of A penetrating (or near-touching) B's
    # surface. Pass 1 scans every vertex of A, evaluates B's grid SDF at each, and keeps the n_max deepest in a small
    # buffer; spatial diversity is enforced during insertion so the buffer doesn't collapse onto one feature. Pass 2
    # emits a contact at each kept vertex with a normal sampled from B's SDF gradient (per-vertex when the local
    # gradient is reliable, falling back to A's centre gradient otherwise). The deepest selection is required for
    # fine-tessellated A (spoons, ico-spheres): with O(n_verts) body work, the buffer must capture the actual contact
    # patch and not arbitrary body verts that would starve the solver of normal-force capacity. The margin is sized to
    # B's smallest SDF cell so verts within one cell of the surface still register a contact even when the kernel pen
    # reads <= 0; the synthetic pen tapers smoothly across the band to avoid a discontinuity that would drive a settled
    # body into a limit-cycle oscillation.
    n_max = qd.static(
        collider_static_config.n_contacts_per_nonconvex_pair if static_rigid_sim_config.enable_multi_contact else 1
    )
    EPS = rigid_global_info.EPS[None]
    gb_cell = sdf_info.geoms_info.sdf_cell_size[i_gb]
    margin = qd.min(qd.min(gb_cell[0], gb_cell[1]), gb_cell[2])
    synthetic_pen_max = 1e-4

    # Bounding-sphere-vs-SDF coarse reject at A's centre. Every point of A lies within rbound_a of
    # geoms_info.center[i_ga], so when B's SDF at A's centre exceeds rbound_a no point of A can reach B's surface and
    # the O(n_verts) scan is skipped. rbound_a is the tight sphere around A's AABB centred at geoms_info.center[i_ga]
    # (which is not necessarily the AABB midpoint for decomposed convex pieces). The reject is gated on
    # can_use_sd_reject: only valid when B's SDF query is exact - true for SPHERE and PLANE (analytical) and for grid
    # B with the query point inside the grid; outside the grid the SDF falls back to a proxy that can over-report
    # distance and silently miss a contact. A directional/SAT bound that uses the SDF gradient at A's centre would be
    # tighter but is unsafe on nonconvex B: the centre gradient is a local linearisation, so an A vertex on the
    # opposite side can still reach a different feature of B that the centre points away from.
    center_local = geoms_info.center[i_ga]
    rbound_a_sq = gs.qd_float(0.0)
    for k in qd.static(range(8)):
        delta = geoms_init_AABB[i_ga, k] - center_local
        d_sq = delta.dot(delta)
        if d_sq > rbound_a_sq:
            rbound_a_sq = d_sq
    rbound_a = qd.sqrt(rbound_a_sq)
    center_a_world = gu.qd_transform_by_trans_quat(center_local, ga_pos, ga_quat)
    can_use_sd_reject = geoms_info.type[i_gb] == gs.GEOM_TYPE.SPHERE or geoms_info.type[i_gb] == gs.GEOM_TYPE.PLANE
    if not can_use_sd_reject:
        pos_mesh = gu.qd_inv_transform_by_trans_quat(center_a_world, gb_pos, gb_quat)
        pos_sdf = gu.qd_transform_by_T(pos_mesh, sdf_info.geoms_info.T_mesh_to_sdf[i_gb])
        can_use_sd_reject = not sdf.sdf_func_is_outside_sdf_grid(sdf_info, pos_sdf, i_gb)
    sd_center = sdf.sdf_func_world_local(geoms_info, sdf_info, center_a_world, i_gb, gb_pos, gb_quat)

    n_added = 0
    if (not can_use_sd_reject) or sd_center <= rbound_a:
        # Pass 1: select the n_max deepest spatially-diverse vertices of A by grid SDF pen. The buffer keeps verts at
        # least `diversity_radius` apart in world space: a candidate close to an existing entry replaces that entry
        # only when deeper, otherwise it displaces the weakest entry. The radius starts at the default `tolerance`
        # and is widened by the "needle extent" of A - how much of the long axis sticks out beyond twice the
        # cross-section - divided across n_max slots. A near-cube/sphere/spoon-bowl has zero (clipped) needle extent
        # and stays at `tolerance`, which lets a small curved patch keep its tight cluster of contacts without
        # picking up rim verts whose grad is tilted relative to the surface and would inject torque; a 1:1:16 rod
        # recovers a radius near rbound_a/n_max so the buffer spreads along the long axis instead of collapsing onto
        # the deepest tip and letting the body pivot about a single contact patch.
        ext = geoms_init_AABB[i_ga, 7] - geoms_init_AABB[i_ga, 0]
        ext_max = qd.max(qd.max(ext[0], ext[1]), ext[2])
        ext_sum_other = ext[0] + ext[1] + ext[2] - ext_max
        needle_extent = ext_max - gs.qd_float(2.0) * ext_sum_other
        diversity_radius = qd.max(tolerance, needle_extent * gs.qd_float(0.5 / n_max))
        top_iv = qd.Vector.zero(gs.qd_int, n_max)
        top_pen = qd.Vector.zero(gs.qd_float, n_max)
        for k in range(n_max):
            top_pen[k] = -gs.qd_float(1e30)
            top_iv[k] = -1
        for i_v in range(geoms_info.vert_start[i_ga], geoms_info.vert_end[i_ga]):
            vertex_pos = gu.qd_transform_by_trans_quat(verts_info.init_pos[i_v], ga_pos, ga_quat)
            if func_point_in_geom_aabb(geoms_state, i_gb, i_b, vertex_pos):
                pen_v = -sdf.sdf_func_world_local(geoms_info, sdf_info, vertex_pos, i_gb, gb_pos, gb_quat)
                if pen_v > -margin:
                    close_idx = -1
                    for k in range(n_max):
                        if close_idx < 0 and top_iv[k] >= 0:
                            other_pos = gu.qd_transform_by_trans_quat(verts_info.init_pos[top_iv[k]], ga_pos, ga_quat)
                            if (vertex_pos - other_pos).norm() < diversity_radius:
                                close_idx = k
                    if close_idx >= 0:
                        if pen_v > top_pen[close_idx]:
                            top_pen[close_idx] = pen_v
                            top_iv[close_idx] = i_v
                    else:
                        weakest_idx = 0
                        for k in range(1, n_max):
                            if top_pen[k] < top_pen[weakest_idx]:
                                weakest_idx = k
                        if pen_v > top_pen[weakest_idx]:
                            top_pen[weakest_idx] = pen_v
                            top_iv[weakest_idx] = i_v

        # Pass 2: emit contacts at the selected vertices. Reference normal is the grid SDF gradient sampled at A's
        # center, which gives a stable per-patch normal whenever A is small relative to B's features. Per-vertex
        # grads agreeing with that reference (positive dot product) are kept verbatim - this is what allows an A
        # wedged at a concave L-corner to expose both axis-aligned face normals (floor + wall). A per-vertex grad
        # that opposes the reference indicates the vertex is closer to B's OPPOSITE surface across a thin feature,
        # i.e. A has partially tunneled through; using its raw grad would push A further through, so we fall back
        # to the reference direction.
        grad_center = sdf.sdf_func_grad_world_local(
            geoms_info, rigid_global_info, collider_static_config, sdf_info, center_a_world, i_gb, gb_pos, gb_quat
        )
        normal_center = gu.qd_normalize(grad_center, EPS)
        # When two comparably-sized bodies meet "across" each other (the crossed-thin-rod regime: A's center sits within
        # one A-bounding-radius of B's surface, with both bodies of similar bounding-sphere size), the grid SDF gradient
        # at A's center is poorly conditioned - it lands on B's local radial direction, which is perpendicular to the
        # closing motion. The geometric line from B's origin to A's center is a stronger reference there: it points
        # along the relative-pose offset, which for a head-on closing pair coincides with the closing direction. The
        # per-vertex grad is also locally radial and just as biased, so in this regime we use the closing direction as
        # the FINAL normal rather than letting per-vertex grad override it. The size-ratio gate keeps the existing
        # behavior for one-big-one-small pairs where the SDF grad at A's center is reliable and the closing-direction
        # line is wrong (sphere on a large floor mesh).
        rbound_b_sq = gs.qd_float(0.0)
        b_center_local = geoms_info.center[i_gb]
        for k in qd.static(range(8)):
            delta_b = geoms_init_AABB[i_gb, k] - b_center_local
            d_sq_b = delta_b.dot(delta_b)
            if d_sq_b > rbound_b_sq:
                rbound_b_sq = d_sq_b
        rbound_b = qd.sqrt(rbound_b_sq)
        center_b_world = gu.qd_transform_by_trans_quat(b_center_local, gb_pos, gb_quat)
        use_closing_dir = qd.abs(sd_center) < rbound_a and rbound_a_sq > gs.qd_float(0.25) * rbound_b_sq
        approach_depth_pair = gs.qd_float(0.0)
        # Set when A wraps around B so that B passes through A along the center-to-center axis. There the SDF gradient
        # at A's center is ill-conditioned (it sits inside B, away from any surface), so per-vertex grads are trusted
        # directly rather than filtered against that unreliable reference normal.
        enclosed_axis = False
        if use_closing_dir:
            closing_dir = center_a_world - center_b_world
            if closing_dir.norm() > EPS:
                closing_normal = gu.qd_normalize(closing_dir, EPS)
                # OBB extent of A and B along the closing direction. Using the identity (R . e_i) . d = e_i . (R^T . d),
                # we only need one inverse rotation per body to express closing_dir in that body's local frame; the OBB
                # extent is then a 3-term weighted sum of local half-extents against the abs components of the
                # local-frame direction. The geometric overlap along the closing axis is (h_a + h_b) - |distance between
                # centers along that axis|. This is the depth the bodies have advanced into each other along the
                # direction the constraint will push them apart, and it is much larger than the per-vertex SDF
                # "distance to nearest surface" for crossed thin geoms where most A verts sit on A's outer skin one
                # radial gap away from B's lateral surface.
                half_ext_a = (geoms_init_AABB[i_ga, 7] - geoms_init_AABB[i_ga, 0]) * gs.qd_float(0.5)
                half_ext_b = (geoms_init_AABB[i_gb, 7] - geoms_init_AABB[i_gb, 0]) * gs.qd_float(0.5)
                d_local_a = gu.qd_inv_transform_by_quat(closing_normal, ga_quat)
                d_local_b = gu.qd_inv_transform_by_quat(closing_normal, gb_quat)
                h_a = half_ext_a.dot(qd.abs(d_local_a))
                h_b = half_ext_b.dot(qd.abs(d_local_b))
                # Reject the override when A wraps around B: the center-to-center line is then B's through-axis and the
                # OBB "overlap" is the pass-through extent, not a real interpenetration, so resolving along it would
                # eject A. The pose-robust signature is that A's own center lies in a cavity rather than inside A's
                # material, so query A's SDF at A's center: positive for such a hollow/annular A, negative for the solid
                # A of the genuine crossed-thin-geom regime.
                sd_a_self = sdf.sdf_func_world_local(geoms_info, sdf_info, center_a_world, i_ga, ga_pos, ga_quat)
                if sd_a_self > EPS:
                    use_closing_dir = False
                    enclosed_axis = True
                else:
                    normal_center = closing_normal
                    center_proj = qd.abs((center_a_world - center_b_world).dot(closing_normal))
                    approach_depth_pair = h_a + h_b - center_proj
            else:
                use_closing_dir = False
        for k in range(n_max):
            if top_iv[k] >= 0:
                i_v = top_iv[k]
                vertex_pos = gu.qd_transform_by_trans_quat(verts_info.init_pos[i_v], ga_pos, ga_quat)
                pen_v = top_pen[k]
                grad_v = sdf.sdf_func_grad_world_local(
                    geoms_info,
                    rigid_global_info,
                    collider_static_config,
                    sdf_info,
                    vertex_pos,
                    i_gb,
                    gb_pos,
                    gb_quat,
                )
                # Per-vertex grad magnitude classifies the local grid SDF regime.
                # - Smoothed (|grad| < 0.5): trilinear interpolation has smoothed across a feature. The pen value is
                #   unreliable. Emit a tapered synthetic pen for approach detection.
                # - Edge (0.5 <= |grad| <= 0.9): the vertex sits on a concave seam. The kernel pen is unreliable. Emit
                #   a bounded synthetic pen so the contact registers without dominating the clean face contacts that
                #   wedge the body at the seam.
                # - Clean (|grad| > 0.9): trust the kernel pen.
                grad_norm = grad_v.norm()
                pen_emit = gs.qd_float(0.0)
                contact_pos_v = vertex_pos
                if grad_norm < 0.5:
                    if pen_v > 0.0:
                        pen_emit = qd.min(pen_v, margin)
                    else:
                        pen_emit = synthetic_pen_max * (1.0 + pen_v / margin)
                elif grad_norm > 0.9:
                    if pen_v > 0.0:
                        pen_emit = pen_v
                elif pen_v > 0.0:
                    pen_emit = synthetic_pen_max
                normal_v = normal_center
                if enclosed_axis:
                    # The reference normal (grad at A's center) is unreliable here, so trust each vertex's local grad
                    # when well-conditioned: it points out of B's surface at that vertex, keeping the normals diverse
                    # across the contact patch so the net force balances instead of pushing A off to one side.
                    if grad_norm > 0.5:
                        normal_v = gu.qd_normalize(grad_v, EPS)
                elif not use_closing_dir and grad_v.dot(normal_center) > 0.0:
                    normal_v = gu.qd_normalize(grad_v, EPS)
                # In the closing-direction regime, the SDF "distance to nearest surface" measured at a vertex on A's
                # outer skin is the small radial gap to B's lateral, not the much larger approach depth along the
                # closing axis. Use the geometric approach depth as a floor on pen_emit so the constraint solver sees
                # the actual overlap rather than just the radial gap.
                if use_closing_dir and pen_v > 0.0 and approach_depth_pair > pen_emit:
                    pen_emit = approach_depth_pair
                repeated = False
                for j in range(n_added):
                    idx_prev = collider_state.n_contacts[i_b] - 1 - j
                    if (contact_pos_v - collider_state.contact_data.pos[idx_prev, i_b]).norm() < tolerance:
                        repeated = True
                if not repeated and pen_emit > 0.0:
                    # Snap the contact position onto A's smooth surface when A is a smooth primitive
                    # (SPHERE/ELLIPSOID/CAPSULE). The tessellation vertex sits an O(tessellation chord error) inboard
                    # of the true surface; on a settled static contact that offset becomes a torque arm and drives a
                    # slow tangential drift. The refinement is a no-op for polytope-typed A.
                    contact_pos_v = func_apply_smooth_refinement(
                        i_ga,
                        i_gb,
                        normal_v,
                        pen_emit,
                        contact_pos_v,
                        ga_pos,
                        ga_quat,
                        gb_pos,
                        gb_quat,
                        geoms_info,
                        static_rigid_sim_config,
                    )
                    func_add_contact(
                        i_ga,
                        i_gb,
                        normal_v,
                        contact_pos_v,
                        pen_emit,
                        i_b,
                        i_pair,
                        geoms_state,
                        geoms_info,
                        collider_state,
                        collider_info,
                        errno,
                    )
                    n_added = n_added + 1
    return n_added


@qd.func
def func_contact_vertex_sdf(
    i_ga,
    i_gb,
    i_b,
    ga_pos: qd.types.vector(3),
    ga_quat: qd.types.vector(4),
    gb_pos: qd.types.vector(3),
    gb_quat: qd.types.vector(4),
    geoms_state: array_class.GeomsState,
    geoms_info: array_class.GeomsInfo,
    verts_info: array_class.VertsInfo,
    rigid_global_info: array_class.RigidGlobalInfo,
    collider_static_config: qd.template(),
    sdf_info: array_class.SDFInfo,
):
    is_col = False
    penetration = gs.qd_float(0.0)
    normal = qd.Vector.zero(gs.qd_float, 3)
    contact_pos = qd.Vector.zero(gs.qd_float, 3)

    for i_v in range(geoms_info.vert_start[i_ga], geoms_info.vert_end[i_ga]):
        vertex_pos = gu.qd_transform_by_trans_quat(verts_info.init_pos[i_v], ga_pos, ga_quat)
        if func_point_in_geom_aabb(geoms_state, i_gb, i_b, vertex_pos):
            new_penetration = -sdf.sdf_func_world_local(geoms_info, sdf_info, vertex_pos, i_gb, gb_pos, gb_quat)
            if new_penetration > penetration:
                is_col = True
                contact_pos = vertex_pos
                penetration = new_penetration

    if is_col:
        # Sample B's SDF gradient at A's geometric center when A is a convex geom (primitive or convexified mesh). For
        # any convex shape the centroid of its vertices is by construction inside its hull, so the sampling point sits
        # deep inside A's surface and several SDF cells away from B's zero-isosurface in this post-penetration
        # configuration; the tri-linear interpolation of B's gradient there is dominated by the smooth interior and not
        # by the cell-aligned noise that contaminates the gradient evaluated at the deepest iterated vertex. Without
        # this override the contact normal acquires a tangential component that flips frame to frame as the deepest
        # vertex migrates between neighbouring tessellated points, driving spheres at rest into a ~m/s vertical jitter
        # and a slow lateral drift. We use `geoms_info.center` - already populated and used by MPR for the same
        # purpose - rather than A's frame origin so the property generalises beyond primitives whose modeller happened
        # to centre the pivot.
        center_a = gu.qd_transform_by_trans_quat(geoms_info.center[i_ga], ga_pos, ga_quat)
        normal_sample = center_a if geoms_info.is_convex[i_ga] else contact_pos
        normal = sdf.sdf_func_normal_world_local(
            geoms_info, rigid_global_info, collider_static_config, sdf_info, normal_sample, i_gb, gb_pos, gb_quat
        )

        # Shift contact_pos from the deepest vertex (interior side) by half the penetration along the outward normal
        # so it lands at the midpoint between A's surface and B's surface.
        contact_pos = contact_pos + 0.5 * penetration * normal

    return is_col, normal, penetration, contact_pos


@qd.func
def func_contact_nonconvex_convex_sdf(
    i_ga,
    i_gb,
    i_b,
    ga_pos: qd.types.vector(3),
    ga_quat: qd.types.vector(4),
    gb_pos: qd.types.vector(3),
    gb_quat: qd.types.vector(4),
    geoms_state: array_class.GeomsState,
    geoms_info: array_class.GeomsInfo,
    geoms_init_AABB: array_class.GeomsInitAABB,
    verts_info: array_class.VertsInfo,
    rigid_global_info: array_class.RigidGlobalInfo,
    collider_static_config: qd.template(),
    sdf_info: array_class.SDFInfo,
):
    # Contact between geoms in a mixed-convexity pair where A is the smaller-AABB side (whether convex or not) and B
    # is the other side, used in lieu of the symmetric two-pass dispatch of func_narrow_phase_nonconvex_vs_nonterrain.
    # A's vertex enumeration alone finds the deepest penetration into B - in a mixed pair the smaller geom is always
    # the one penetrating the larger - and skipping the other side's O(n_verts) scan kills the cost that dominates
    # when the larger side is a static mesh with many tens of thousands of vertices. A bounding-sphere-vs-SDF coarse
    # reject at A's centre mirrors MPR's "if no support certifies overlap, exit" pattern: every point of A lies within
    # rbound_a of geoms_info.center[i_ga], so when B's SDF at A's centre exceeds rbound_a no point of A can reach B's
    # zero level set and the entire vertex scan is skipped.
    is_col = False
    penetration = gs.qd_float(0.0)
    normal = qd.Vector.zero(gs.qd_float, 3)
    contact_pos = qd.Vector.zero(gs.qd_float, 3)

    # rbound_a is the radius of the smallest sphere centred at geoms_info.center[i_ga] that contains A's AABB.
    # func_compute_geom_rbound returns half the AABB diagonal, which is correct only when the geom centre coincides
    # with the AABB midpoint - true for primitives but not for arbitrary decomposed convex mesh pieces, whose centroid
    # (used as geoms_info.center by MPR) is offset. Iterating the 8 AABB corners gives the tight per-centre bound and
    # ensures the reject test never discards a pair whose vertices could still be inside B.
    center_local = geoms_info.center[i_ga]
    rbound_a_sq = gs.qd_float(0.0)
    for k in qd.static(range(8)):
        delta = geoms_init_AABB[i_ga, k] - center_local
        d_sq = delta.dot(delta)
        if d_sq > rbound_a_sq:
            rbound_a_sq = d_sq
    rbound_a = qd.sqrt(rbound_a_sq)

    # The reject is only valid when the SDF value at A's centre is the true distance to B's surface. For SPHERE and
    # PLANE B that is the analytical branch of sdf_func_world_local - exact everywhere. For grid-based SDFs the query
    # returns the true trilinear interpolation only when the point falls inside the SDF grid; outside the grid
    # sdf_func_world_local routes to sdf_func_proxy_sdf, which returns ||P - grid_center||_world + sdf_max. That proxy
    # is NOT a one-sided bound on true_sdf: with R_mesh the mesh's bounding radius from grid_center, the gap
    # proxy - true_sdf lies in [sdf_max - R_mesh, sdf_max + R_mesh]. Typical grids use a 20% padding so
    # sdf_max ~ 0.2*R_mesh < R_mesh, and the proxy can overestimate true_sdf by up to ~1.2*R_mesh - large enough to
    # satisfy proxy > rbound_a while true_sdf < rbound_a, silently missing a real contact. Falling through to the
    # vertex scan when outside the grid costs at most one extra func_contact_vertex_sdf pass on the smaller side and
    # keeps the reject correct in all cases.
    center_a_world = gu.qd_transform_by_trans_quat(center_local, ga_pos, ga_quat)
    # SPHERE B has an analytical SDF; grid-based B is only safe when the query point falls inside the grid.
    can_use_sd_reject = geoms_info.type[i_gb] == gs.GEOM_TYPE.SPHERE
    if not can_use_sd_reject:
        pos_mesh = gu.qd_inv_transform_by_trans_quat(center_a_world, gb_pos, gb_quat)
        pos_sdf = gu.qd_transform_by_T(pos_mesh, sdf_info.geoms_info.T_mesh_to_sdf[i_gb])
        can_use_sd_reject = not sdf.sdf_func_is_outside_sdf_grid(sdf_info, pos_sdf, i_gb)
    sd_center = sdf.sdf_func_world_local(geoms_info, sdf_info, center_a_world, i_gb, gb_pos, gb_quat)

    if (not can_use_sd_reject) or sd_center <= rbound_a:
        is_col, normal, penetration, contact_pos = func_contact_vertex_sdf(
            i_ga,
            i_gb,
            i_b,
            ga_pos,
            ga_quat,
            gb_pos,
            gb_quat,
            geoms_state,
            geoms_info,
            verts_info,
            rigid_global_info,
            collider_static_config,
            sdf_info,
        )

    return is_col, normal, penetration, contact_pos


@qd.func
def func_contact_edge_sdf(
    i_ga,
    i_gb,
    i_b,
    ga_pos: qd.types.vector(3),
    ga_quat: qd.types.vector(4),
    gb_pos: qd.types.vector(3),
    gb_quat: qd.types.vector(4),
    geoms_state: array_class.GeomsState,  # For AABB only
    geoms_info: array_class.GeomsInfo,
    verts_info: array_class.VertsInfo,
    edges_info: array_class.EdgesInfo,
    rigid_global_info: array_class.RigidGlobalInfo,
    collider_static_config: qd.template(),
    sdf_info: array_class.SDFInfo,
):
    EPS = rigid_global_info.EPS[None]

    is_col = False
    penetration = gs.qd_float(0.0)
    normal = qd.Vector.zero(gs.qd_float, 3)
    contact_pos = qd.Vector.zero(gs.qd_float, 3)

    # Use the smallest per-axis cell size as the edge-length threshold so we still subdivide edges that are short
    # relative to the finest grid resolution available.
    ga_sdf_cell_size_vec = sdf_info.geoms_info.sdf_cell_size[i_ga]
    ga_sdf_cell_size = qd.min(qd.min(ga_sdf_cell_size_vec[0], ga_sdf_cell_size_vec[1]), ga_sdf_cell_size_vec[2])

    for i_e in range(geoms_info.edge_start[i_ga], geoms_info.edge_end[i_ga]):
        cur_length = edges_info.length[i_e]
        if cur_length > ga_sdf_cell_size:
            i_v0 = edges_info.v0[i_e]
            i_v1 = edges_info.v1[i_e]

            p_0 = gu.qd_transform_by_trans_quat(verts_info.init_pos[i_v0], ga_pos, ga_quat)
            p_1 = gu.qd_transform_by_trans_quat(verts_info.init_pos[i_v1], ga_pos, ga_quat)
            vec_01 = gu.qd_normalize(p_1 - p_0, EPS)

            sdf_grad_0_b = sdf.sdf_func_grad_world_local(
                geoms_info, rigid_global_info, collider_static_config, sdf_info, p_0, i_gb, gb_pos, gb_quat
            )
            sdf_grad_1_b = sdf.sdf_func_grad_world_local(
                geoms_info, rigid_global_info, collider_static_config, sdf_info, p_1, i_gb, gb_pos, gb_quat
            )

            # check if the edge on a is facing towards mesh b
            sdf_grad_0_a = sdf.sdf_func_grad_world_local(
                geoms_info, rigid_global_info, collider_static_config, sdf_info, p_0, i_ga, ga_pos, ga_quat
            )
            sdf_grad_1_a = sdf.sdf_func_grad_world_local(
                geoms_info, rigid_global_info, collider_static_config, sdf_info, p_1, i_ga, ga_pos, ga_quat
            )
            normal_edge_0 = sdf_grad_0_a - sdf_grad_0_a.dot(vec_01) * vec_01
            normal_edge_1 = sdf_grad_1_a - sdf_grad_1_a.dot(vec_01) * vec_01

            if normal_edge_0.dot(sdf_grad_0_b) < 0 or normal_edge_1.dot(sdf_grad_1_b) < 0:
                # check if closest point is between the two points
                if sdf_grad_0_b.dot(vec_01) < 0 and sdf_grad_1_b.dot(vec_01) > 0:
                    while cur_length > ga_sdf_cell_size:
                        p_mid = 0.5 * (p_0 + p_1)
                        if (
                            sdf.sdf_func_grad_world_local(
                                geoms_info,
                                rigid_global_info,
                                collider_static_config,
                                sdf_info,
                                p_mid,
                                i_gb,
                                gb_pos,
                                gb_quat,
                            ).dot(vec_01)
                            < 0
                        ):
                            p_0 = p_mid
                        else:
                            p_1 = p_mid
                        cur_length = 0.5 * cur_length

                    p = 0.5 * (p_0 + p_1)
                    new_penetration = -sdf.sdf_func_world_local(geoms_info, sdf_info, p, i_gb, gb_pos, gb_quat)

                    if new_penetration > penetration:
                        is_col = True
                        normal = sdf.sdf_func_normal_world_local(
                            geoms_info, rigid_global_info, collider_static_config, sdf_info, p, i_gb, gb_pos, gb_quat
                        )
                        contact_pos = p
                        penetration = new_penetration

    # The contact point must be offsetted by half the penetration depth, for consistency with MPR
    contact_pos = contact_pos + 0.5 * penetration * normal

    return is_col, normal, penetration, contact_pos


@qd.func
def func_contact_convex_convex_sdf(
    i_ga,
    i_gb,
    i_b,
    i_va_ws,
    geoms_state: array_class.GeomsState,
    geoms_info: array_class.GeomsInfo,
    verts_info: array_class.VertsInfo,
    collider_info: array_class.ColliderInfo,
    collider_static_config: qd.template(),
    sdf_info: array_class.SDFInfo,
    rigid_global_info: array_class.RigidGlobalInfo,
    enable_edge_detection_fallback: qd.template(),
):
    EPS = rigid_global_info.EPS[None]

    gb_vert_start = geoms_info.vert_start[i_gb]
    ga_pos = geoms_state.pos[i_ga, i_b]
    ga_quat = geoms_state.quat[i_ga, i_b]
    gb_pos = geoms_state.pos[i_gb, i_b]
    gb_quat = geoms_state.quat[i_gb, i_b]

    is_col = False
    penetration = gs.qd_float(0.0)
    normal = qd.Vector.zero(gs.qd_float, 3)
    contact_pos = qd.Vector.zero(gs.qd_float, 3)

    i_va = i_va_ws
    if i_va == -1:
        # start traversing on the vertex graph with a smart initial vertex
        pos_vb = gu.qd_transform_by_trans_quat(verts_info.init_pos[gb_vert_start], gb_pos, gb_quat)
        i_va = sdf.sdf_func_find_closest_vert(geoms_state, geoms_info, sdf_info, pos_vb, i_ga, i_b)
    i_v_closest = i_va
    pos_v_closest = gu.qd_transform_by_trans_quat(verts_info.init_pos[i_v_closest], ga_pos, ga_quat)
    sd_v_closest = sdf.sdf_func_world(geoms_state, geoms_info, sdf_info, pos_v_closest, i_gb, i_b)

    while True:
        for i_neighbor_ in range(
            collider_info.vert_neighbor_start[i_va],
            collider_info.vert_neighbor_start[i_va] + collider_info.vert_n_neighbors[i_va],
        ):
            i_neighbor = collider_info.vert_neighbors[i_neighbor_]
            pos_neighbor = gu.qd_transform_by_trans_quat(verts_info.init_pos[i_neighbor], ga_pos, ga_quat)
            sd_neighbor = sdf.sdf_func_world(geoms_state, geoms_info, sdf_info, pos_neighbor, i_gb, i_b)
            if sd_neighbor < sd_v_closest - 1e-5:  # 1e-5 (0.01mm) to avoid endless loop due to numerical instability
                i_v_closest = i_neighbor
                sd_v_closest = sd_neighbor
                pos_v_closest = pos_neighbor

        if i_v_closest == i_va:  # no better neighbor
            break
        else:
            i_va = i_v_closest

    # i_va is the deepest vertex
    pos_a = pos_v_closest
    if sd_v_closest < 0.0:
        is_col = True
        normal = sdf.sdf_func_normal_world(
            geoms_state, geoms_info, rigid_global_info, collider_static_config, sdf_info, pos_a, i_gb, i_b
        )
        penetration = -sd_v_closest
        contact_pos = pos_a
    elif enable_edge_detection_fallback:  # check edge surrounding it
        for i_neighbor_ in range(
            collider_info.vert_neighbor_start[i_va],
            collider_info.vert_neighbor_start[i_va] + collider_info.vert_n_neighbors[i_va],
        ):
            i_neighbor = collider_info.vert_neighbors[i_neighbor_]

            p_0 = pos_v_closest
            p_1 = gu.qd_transform_by_trans_quat(verts_info.init_pos[i_neighbor], ga_pos, ga_quat)
            vec_01 = gu.qd_normalize(p_1 - p_0, EPS)

            sdf_grad_0_b = sdf.sdf_func_grad_world(
                geoms_state, geoms_info, rigid_global_info, collider_static_config, sdf_info, p_0, i_gb, i_b
            )
            sdf_grad_1_b = sdf.sdf_func_grad_world(
                geoms_state, geoms_info, rigid_global_info, collider_static_config, sdf_info, p_1, i_gb, i_b
            )

            # check if the edge on a is facing towards mesh b (I am not 100% sure about this, subject to removal)
            sdf_grad_0_a = sdf.sdf_func_grad_world(
                geoms_state, geoms_info, rigid_global_info, collider_static_config, sdf_info, p_0, i_ga, i_b
            )
            sdf_grad_1_a = sdf.sdf_func_grad_world(
                geoms_state, geoms_info, rigid_global_info, collider_static_config, sdf_info, p_1, i_ga, i_b
            )
            normal_edge_0 = sdf_grad_0_a - sdf_grad_0_a.dot(vec_01) * vec_01
            normal_edge_1 = sdf_grad_1_a - sdf_grad_1_a.dot(vec_01) * vec_01

            if normal_edge_0.dot(sdf_grad_0_b) < 0 or normal_edge_1.dot(sdf_grad_1_b) < 0:
                # check if closest point is between the two points
                if sdf_grad_0_b.dot(vec_01) < 0 and sdf_grad_1_b.dot(vec_01) > 0:
                    cur_length = (p_1 - p_0).norm()
                    ga_sdf_cell_size_vec = sdf_info.geoms_info.sdf_cell_size[i_ga]
                    ga_sdf_cell_size = qd.min(
                        qd.min(ga_sdf_cell_size_vec[0], ga_sdf_cell_size_vec[1]), ga_sdf_cell_size_vec[2]
                    )
                    while cur_length > ga_sdf_cell_size:
                        p_mid = 0.5 * (p_0 + p_1)
                        side = sdf.sdf_func_grad_world(
                            geoms_state,
                            geoms_info,
                            rigid_global_info,
                            collider_static_config,
                            sdf_info,
                            p_mid,
                            i_gb,
                            i_b,
                        ).dot(vec_01)
                        if side < 0:
                            p_0 = p_mid
                        else:
                            p_1 = p_mid

                        cur_length = 0.5 * cur_length

                    p = 0.5 * (p_0 + p_1)
                    new_penetration = -sdf.sdf_func_world(geoms_state, geoms_info, sdf_info, p, i_gb, i_b)

                    if new_penetration > 0.0:
                        is_col = True
                        normal = sdf.sdf_func_normal_world(
                            geoms_state, geoms_info, rigid_global_info, collider_static_config, sdf_info, p, i_gb, i_b
                        )
                        contact_pos = p
                        penetration = new_penetration
                        break

    # The contact point must be offsetted by half the penetration depth, for consistency with MPR
    contact_pos = contact_pos + 0.5 * penetration * normal

    return is_col, normal, penetration, contact_pos, i_va


@qd.func
def func_contact_mpr_terrain(
    i_ga,
    i_gb,
    i_b,
    links_state: array_class.LinksState,
    links_info: array_class.LinksInfo,
    geoms_state: array_class.GeomsState,
    geoms_info: array_class.GeomsInfo,
    geoms_init_AABB: array_class.GeomsInitAABB,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: qd.template(),
    collider_state: array_class.ColliderState,
    collider_info: array_class.ColliderInfo,
    collider_static_config: qd.template(),
    mpr_state: array_class.MPRState,
    mpr_info: array_class.MPRInfo,
    support_field_info: array_class.SupportFieldInfo,
    errno: qd.Tensor,
):
    ga_pos, ga_quat = geoms_state.pos[i_ga, i_b], geoms_state.quat[i_ga, i_b]
    gb_pos, gb_quat = geoms_state.pos[i_gb, i_b], geoms_state.quat[i_gb, i_b]
    margin = gs.qd_float(0.0)
    EPS = rigid_global_info.EPS[None]

    multi_contact = (
        qd.static(static_rigid_sim_config.enable_multi_contact)
        and geoms_info.type[i_ga] != gs.GEOM_TYPE.SPHERE
        and geoms_info.type[i_ga] != gs.GEOM_TYPE.ELLIPSOID
    )

    is_return = False
    tolerance = func_compute_tolerance(i_ga, i_gb, i_b, collider_info.mc_tolerance[None], geoms_info, geoms_init_AABB)

    if not is_return:
        # Transform to terrain's frame (using local variables, not modifying global state)
        ga_pos_terrain_frame, ga_quat_terrain_frame = gu.qd_transform_pos_quat_by_trans_quat(
            ga_pos - gb_pos,
            ga_quat,
            qd.Vector.zero(gs.qd_float, 3),
            gu.qd_inv_quat(gb_quat),
        )
        gb_pos_terrain_frame = qd.Vector.zero(gs.qd_float, 3)
        gb_quat_terrain_frame = gu.qd_identity_quat()

        for i_axis, i_m in qd.ndrange(3, 2):
            direction = qd.Vector.zero(gs.qd_float, 3)
            if i_m == 0:
                direction[i_axis] = 1.0
            else:
                direction[i_axis] = -1.0
            v1 = mpr.support_driver(
                geoms_info,
                collider_state,
                collider_static_config,
                support_field_info,
                direction,
                i_ga,
                i_b,
                ga_pos_terrain_frame,
                ga_quat_terrain_frame,
            )
            collider_state.xyz_max_min[3 * i_m + i_axis, i_b] = v1[i_axis]

        for i in qd.static(range(3)):
            collider_state.prism[i, i_b][2] = collider_info.terrain_xyz_maxmin[5]

            if (
                collider_info.terrain_xyz_maxmin[i] < collider_state.xyz_max_min[i + 3, i_b] - margin
                or collider_info.terrain_xyz_maxmin[i + 3] > collider_state.xyz_max_min[i, i_b] + margin
            ):
                is_return = True

        if not is_return:
            sh = collider_info.terrain_scale[0]
            r_min = gs.qd_int(qd.floor((collider_state.xyz_max_min[3, i_b] - collider_info.terrain_xyz_maxmin[3]) / sh))
            r_max = gs.qd_int(qd.ceil((collider_state.xyz_max_min[0, i_b] - collider_info.terrain_xyz_maxmin[3]) / sh))
            c_min = gs.qd_int(qd.floor((collider_state.xyz_max_min[4, i_b] - collider_info.terrain_xyz_maxmin[4]) / sh))
            c_max = gs.qd_int(qd.ceil((collider_state.xyz_max_min[1, i_b] - collider_info.terrain_xyz_maxmin[4]) / sh))

            r_min = qd.max(0, r_min)
            c_min = qd.max(0, c_min)
            r_max = qd.min(collider_info.terrain_rc[0] - 1, r_max)
            c_max = qd.min(collider_info.terrain_rc[1] - 1, c_max)

            # Multi-contact perturbation state. The initial detection (i_detection == 0) finds the first face-vs-face
            # contact; subsequent passes rotate geom A by a small angle about an axis orthogonal to that contact's
            # normal, which tips the box face onto a different corner. After undoing the rotation each perturbed contact
            # lands at a different corner of the contact patch, stabilizing a flat box on a triangulated cell.
            # Perturbation is only applied when the initial contact is face-vs-face (snap fired) - cliff-edge contacts
            # are kept as a single MPR contact.
            is_col_0 = False
            face_face_0 = False
            contact_pos_0 = qd.Vector.zero(gs.qd_float, 3)
            normal_0 = qd.Vector.zero(gs.qd_float, 3)
            axis_0 = qd.Vector.zero(gs.qd_float, 3)
            axis_1 = qd.Vector.zero(gs.qd_float, 3)
            qrot = qd.Vector.zero(gs.qd_float, 4)
            n_con = 0
            n_detections = 5 if multi_contact else 1

            ga_pos_tf = ga_pos_terrain_frame
            ga_quat_tf = ga_quat_terrain_frame
            for i_detection in range(n_detections):
                if i_detection > 0 and not face_face_0:
                    break
                if i_detection > 0:
                    axis = (2 * (i_detection % 2) - 1) * axis_0 + (1 - 2 * ((i_detection // 2) % 2)) * axis_1
                    qrot = gu.qd_rotvec_to_quat(collider_info.mc_perturbation[None] * axis, EPS)
                    ga_pos_curr, ga_quat_curr = func_rotate_frame(ga_pos, ga_quat, contact_pos_0, qrot)
                    ga_pos_tf, ga_quat_tf = gu.qd_transform_pos_quat_by_trans_quat(
                        ga_pos_curr - gb_pos,
                        ga_quat_curr,
                        qd.Vector.zero(gs.qd_float, 3),
                        gu.qd_inv_quat(gb_quat),
                    )
                center_a = gu.qd_transform_by_trans_quat(geoms_info.center[i_ga], ga_pos_tf, ga_quat_tf)

                for r in range(r_min, r_max):
                    nvert = 0
                    for c in range(c_min, c_max + 1):
                        for i in range(2):
                            if n_con < qd.static(collider_static_config.n_contacts_per_convex_pair):
                                nvert = nvert + 1
                                func_add_prism_vert(
                                    sh * (r + i) + collider_info.terrain_xyz_maxmin[3],
                                    sh * c + collider_info.terrain_xyz_maxmin[4],
                                    collider_info.terrain_hf[r + i, c] + margin,
                                    i_b,
                                    collider_state,
                                )
                                if nvert > 2 and (
                                    collider_state.prism[3, i_b][2] >= collider_state.xyz_max_min[5, i_b]
                                    or collider_state.prism[4, i_b][2] >= collider_state.xyz_max_min[5, i_b]
                                    or collider_state.prism[5, i_b][2] >= collider_state.xyz_max_min[5, i_b]
                                ):
                                    center_b = qd.Vector.zero(gs.qd_float, 3)
                                    for i_p in qd.static(range(6)):
                                        center_b = center_b + collider_state.prism[i_p, i_b]
                                    center_b = center_b / 6.0

                                    is_col, normal, penetration, contact_pos = mpr.func_mpr_contact_from_centers(
                                        geoms_info,
                                        static_rigid_sim_config,
                                        collider_state,
                                        collider_static_config,
                                        mpr_state,
                                        mpr_info,
                                        support_field_info,
                                        i_ga,
                                        i_gb,
                                        i_b,
                                        center_a,
                                        center_b,
                                        ga_pos_tf,
                                        ga_quat_tf,
                                        gb_pos_terrain_frame,
                                        gb_quat_terrain_frame,
                                    )
                                    if is_col:
                                        snap_fired = False
                                        face_face = False
                                        # Snap normal to the prism's top face normal when MPR's reported normal is
                                        # already close to it. Cell boundaries on a SMOOTH heightfield are
                                        # discretization artefacts, not physical edges, and MPR's polytope-edge radial
                                        # normal there picks up a small position-dependent bias relative to the exact
                                        # face normal. Only snap when the bias is small (dot > 0.95) so that contacts on
                                        # real cliff edges - where MPR's normal is genuinely far from any single cell's
                                        # top face normal - keep MPR's result.
                                        e1 = collider_state.prism[4, i_b] - collider_state.prism[3, i_b]
                                        e2 = collider_state.prism[5, i_b] - collider_state.prism[3, i_b]
                                        top_face_normal = e1.cross(e2).normalized()
                                        if top_face_normal[2] < 0.0:
                                            top_face_normal = -top_face_normal
                                        if top_face_normal.dot(normal) > 0.95:
                                            normal = top_face_normal
                                            snap_fired = True
                                            # Cell is essentially horizontal in the terrain's local frame (within ~8
                                            # deg). Independent of the terrain's world orientation.
                                            face_face = top_face_normal[2] > 0.99

                                        normal = gu.qd_transform_by_quat(normal, gb_quat)
                                        contact_pos = gu.qd_transform_by_quat(contact_pos, gb_quat)
                                        contact_pos = contact_pos + gb_pos

                                        # No perturbation correction: the perturbation magnitude (mc_perturbation,
                                        # default 1e-2 rad) is so small that the perturbed contact_pos sits within a
                                        # millimeter of the unperturbed contact patch. The deduplication tolerance
                                        # downstream picks the unique corner contacts and the constraint solver
                                        # tolerates the residual offset.

                                        contact_pos = func_apply_smooth_refinement(
                                            i_ga,
                                            i_gb,
                                            normal,
                                            penetration,
                                            contact_pos,
                                            geoms_state.pos[i_ga, i_b],
                                            geoms_state.quat[i_ga, i_b],
                                            geoms_state.pos[i_gb, i_b],
                                            geoms_state.quat[i_gb, i_b],
                                            geoms_info,
                                            static_rigid_sim_config,
                                        )

                                        valid = True
                                        i_c = collider_state.n_contacts[i_b]
                                        for j in range(n_con):
                                            if (
                                                contact_pos - collider_state.contact_data.pos[i_c - j - 1, i_b]
                                            ).norm() < tolerance:
                                                valid = False
                                                break
                                        if valid and i_detection > 0 and not snap_fired:
                                            # Perturbed contacts are only kept when they still describe a face-vs-face
                                            # contact on the same horizontal cell face. A perturbed corner that landed
                                            # against a cliff wall (snap did not fire) is a phantom contact.
                                            valid = False

                                        if valid:
                                            i_pair = collider_info.collision_pair_idx[
                                                (i_gb, i_ga) if i_ga > i_gb else (i_ga, i_gb)
                                            ]
                                            func_add_contact(
                                                i_ga,
                                                i_gb,
                                                normal,
                                                contact_pos,
                                                penetration,
                                                i_b,
                                                i_pair,
                                                geoms_state,
                                                geoms_info,
                                                collider_state,
                                                collider_info,
                                                errno,
                                            )
                                            n_con = n_con + 1
                                            if i_detection == 0 and not is_col_0:
                                                is_col_0 = True
                                                # Perturbation only applies on essentially-horizontal cell faces (in the
                                                # terrain's local frame, within ~8 deg of the heightfield +z): on
                                                # steeper slopes the corner contacts generated by perturbation map to
                                                # different parts of neighbouring cells and create constraint-solver
                                                # oscillation.
                                                face_face_0 = face_face
                                                contact_pos_0 = contact_pos
                                                normal_0 = normal
                                                if face_face_0:
                                                    axis_0, axis_1 = func_contact_orthogonals(
                                                        i_ga,
                                                        i_gb,
                                                        normal_0,
                                                        i_b,
                                                        links_state,
                                                        links_info,
                                                        geoms_state,
                                                        geoms_info,
                                                        geoms_init_AABB,
                                                        rigid_global_info,
                                                        static_rigid_sim_config,
                                                    )


@qd.func
def func_add_prism_vert(
    x,
    y,
    z,
    i_b,
    collider_state: array_class.ColliderState,
):
    collider_state.prism[0, i_b] = collider_state.prism[1, i_b]
    collider_state.prism[1, i_b] = collider_state.prism[2, i_b]
    collider_state.prism[3, i_b] = collider_state.prism[4, i_b]
    collider_state.prism[4, i_b] = collider_state.prism[5, i_b]

    collider_state.prism[2, i_b][0] = x
    collider_state.prism[5, i_b][0] = x
    collider_state.prism[2, i_b][1] = y
    collider_state.prism[5, i_b][1] = y
    collider_state.prism[5, i_b][2] = z


@qd.func
def func_recompute_perturbed_contact(
    i_ga,
    i_gb,
    i_scratch,
    normal: qd.types.vector(3),
    penetration,
    contact_pos: qd.types.vector(3),
    normal_0: qd.types.vector(3),
    contact_pos_0: qd.types.vector(3),
    qrot: qd.types.vector(4),
    ga_pos_original: qd.types.vector(3),
    ga_quat_original: qd.types.vector(4),
    gb_pos_original: qd.types.vector(3),
    gb_quat_original: qd.types.vector(4),
    used_gjk,
    geoms_info: array_class.GeomsInfo,
    rigid_global_info: array_class.RigidGlobalInfo,
    collider_info: array_class.ColliderInfo,
    mpr_state: array_class.MPRState,
    gjk_state: array_class.GJKState,
    static_rigid_sim_config: qd.template(),
):
    """
    Recompute a perturbed multi-contact point exactly, by un-rotating the portal the perturbed detection found.

    Multi-contact spreads contact points by detecting collisions on slightly rotated copies of the two geometries.
    The contact normal and penetration must be recovered for the unperturbed configuration. The contact normal is a
    property of the Minkowski difference (both geometries), so neither geom's surface normal alone captures it.
    Instead the MPR portal - the triangle of support-point pairs bounding the contact - is un-rotated back to the
    unperturbed pose (each support point by the inverse of its own geom's perturbation), and the face normal of the
    resulting Minkowski triangle gives the exact contact normal, with the penetration as the portal's distance to the
    Minkowski origin. The position is reconstructed analytically on the smooth side.
    """
    # qrot is applied to geom A and its inverse to geom B; precompute the rotation matrix once (R for qrot, its
    # transpose for the inverse) and reuse it for every un-rotation below instead of re-deriving it per call.
    R = gu.qd_quat_to_R(qrot, rigid_global_info.EPS[None])
    R_inv = R.transpose()
    contact_point_a = R_inv @ ((contact_pos - 0.5 * penetration * normal) - contact_pos_0) + contact_pos_0
    contact_point_b = R @ ((contact_pos + 0.5 * penetration * normal) - contact_pos_0) + contact_pos_0
    contact_pos = 0.5 * (contact_point_a + contact_point_b)

    # The unperturbed contact normal is recovered per detection method, using only the data that method exposes. The
    # multi-contact perturbation is symmetric (geom A by +qrot, geom B by -qrot, over +/- axis pairs), so methods that
    # keep the perturbed normal still yield an unbiased contact set: the per-contact tilts cancel in aggregate (no
    # drift), and the pruning kernel's mean normal averages them back to the true normal (the patch stays coplanar).
    #  - PLANE: the normal is rigid to the plane geom (geom A, rotated by qrot), so un-rotating it by qrot is exact.
    #  - CAPSULE-CAPSULE: an analytic closest-segment contact, with no portal or witness pair; the only available
    #    correction is the first-order twist of the perturbed normal back towards the unperturbed one.
    #  - MPR: it exposes no witness pair, only a portal; the un-rotated portal support simplex gives the exact normal
    #    as the Minkowski-triangle face normal (vertex-face / edge-edge contacts included).
    #  - GJK: same construction from the EPA polytope face nearest to the origin (its three support pairs).
    # is_exact reports whether the recovered penetration is exact (a true contact depth) rather than an approximate
    # first-order value. The caller uses it to pick the contact-acceptance threshold: an exact penetration can be
    # discarded as soon as it is non-positive (fictitious contact), while an approximate one keeps a negative tolerance.
    is_exact = False
    needs_twist = False
    if geoms_info.type[i_ga] == gs.GEOM_TYPE.PLANE:
        normal = R_inv @ normal
        penetration = normal.dot(contact_point_b - contact_point_a)
        is_exact = True
    elif geoms_info.type[i_ga] == gs.GEOM_TYPE.CAPSULE and geoms_info.type[i_gb] == gs.GEOM_TYPE.CAPSULE:
        # Analytic closest-segment contact: no portal or witness pair (and its portal_valid / nearest_face are stale,
        # since it runs neither MPR nor GJK), so the only correction available is the first-order twist.
        needs_twist = True
    elif used_gjk and gjk_state.nearest_face[i_scratch] < 0:
        # Shallow GJK contact (no EPA polytope was built): no support face, but the perturbed witness delta is the
        # perturbed normal by construction, so keep it; the +/- symmetry keeps the contact set unbiased in aggregate.
        pass
    elif not used_gjk and not mpr_state.portal_valid[i_scratch]:
        # MPR resolved the contact through a degenerate touch/segment path, so simplex_support holds no refined
        # contact-face portal; reconstructing from it would yield a spurious edge/corner normal.
        needs_twist = True
    else:
        # Support pairs of the contact face: the MPR portal (indices 1-3), or the GJK EPA face nearest to the origin.
        a1 = mpr_state.simplex_support.v1[1, i_scratch]
        b1 = mpr_state.simplex_support.v2[1, i_scratch]
        a2 = mpr_state.simplex_support.v1[2, i_scratch]
        b2 = mpr_state.simplex_support.v2[2, i_scratch]
        a3 = mpr_state.simplex_support.v1[3, i_scratch]
        b3 = mpr_state.simplex_support.v2[3, i_scratch]
        if used_gjk:
            i_f = gjk_state.nearest_face[i_scratch]
            iv1 = gjk_state.polytope_faces.verts_idx[i_scratch, i_f][0]
            iv2 = gjk_state.polytope_faces.verts_idx[i_scratch, i_f][1]
            iv3 = gjk_state.polytope_faces.verts_idx[i_scratch, i_f][2]
            a1 = gjk_state.polytope_verts.obj1[i_scratch, iv1]
            b1 = gjk_state.polytope_verts.obj2[i_scratch, iv1]
            a2 = gjk_state.polytope_verts.obj1[i_scratch, iv2]
            b2 = gjk_state.polytope_verts.obj2[i_scratch, iv2]
            a3 = gjk_state.polytope_verts.obj1[i_scratch, iv3]
            b3 = gjk_state.polytope_verts.obj2[i_scratch, iv3]
        # contact_pos_0 cancels in the edge differences, so the face normal needs only support-point deltas.
        edge1 = R_inv @ (a2 - a1) - R @ (b2 - b1)
        edge2 = R_inv @ (a3 - a1) - R @ (b3 - b1)
        portal_normal = edge1.cross(edge2)
        portal_norm_sqr = portal_normal.norm_sqr()
        # The face normal is reliable only when the support triangle is well-conditioned. For a nearly coplanar
        # contact (e.g. flat box-on-box) the support points can be almost collinear, making the face normal
        # numerically unstable; fall back to the twist there. Compared squared to avoid the edge-length square roots.
        if portal_norm_sqr > 0.01 * edge1.norm_sqr() * edge2.norm_sqr():
            normal = portal_normal / qd.sqrt(portal_norm_sqr)
            if normal.dot(normal_0) < 0.0:
                normal = -normal
            # m1 (one un-rotated Minkowski support point on the face) is only needed for the exact penetration depth.
            m1 = R_inv @ (a1 - contact_pos_0) - R @ (b1 - contact_pos_0)
            penetration = -normal.dot(m1)
            is_exact = True
        else:
            needs_twist = True

    # Single first-order fallback for every case that could not recover an exact normal (analytic capsule-capsule,
    # degenerate MPR, near-collinear portal). Computed once, and only when actually needed.
    if needs_twist:
        mc_perturbation = collider_info.mc_perturbation[None]
        twist_rotvec = qd.math.clamp(normal.cross(normal_0), -mc_perturbation, mc_perturbation)
        normal = normal + twist_rotvec.cross(normal)
    if not is_exact:
        penetration = normal.dot(contact_point_b - contact_point_a)

    # Apply the smooth-primitive position reconstruction here, after the perturbation has been reverted, so it uses the
    # final (corrected) normal and the unperturbed pose - the canonical state the solver stores.
    contact_pos = func_apply_smooth_refinement(
        i_ga,
        i_gb,
        normal,
        penetration,
        contact_pos,
        ga_pos_original,
        ga_quat_original,
        gb_pos_original,
        gb_quat_original,
        geoms_info,
        static_rigid_sim_config,
    )
    return normal, penetration, contact_pos, is_exact


@qd.func
def func_convex_convex_contact(
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
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: qd.template(),
    collider_state: array_class.ColliderState,
    collider_info: array_class.ColliderInfo,
    collider_static_config: qd.template(),
    mpr_state: array_class.MPRState,
    mpr_info: array_class.MPRInfo,
    gjk_state: array_class.GJKState,
    gjk_info: array_class.GJKInfo,
    gjk_static_config: qd.template(),
    support_field_info: array_class.SupportFieldInfo,
    # FIXME: Passing nested data structure as input argument is not supported for now.
    diff_contact_input: array_class.DiffContactInput,
    errno: qd.Tensor,
):
    if not (geoms_info.type[i_ga] == gs.GEOM_TYPE.PLANE and geoms_info.type[i_gb] == gs.GEOM_TYPE.BOX):
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
        if qd.static(static_rigid_sim_config.enable_mujoco_compatibility):
            tolerance = func_compute_mj_tolerance(
                i_ga, i_gb, collider_info.mc_tolerance[None], geoms_info, geoms_init_AABB
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

        # Pre-allocate some buffers
        # Note that the variables post-fixed with _0 are the values of these
        # variables for contact 0 (used for multi-contact).
        is_col_0 = False
        penetration_0 = gs.qd_float(0.0)
        normal_0 = qd.Vector.zero(gs.qd_float, 3)
        contact_pos_0 = qd.Vector.zero(gs.qd_float, 3)

        # Whether narrowphase detected a contact.
        is_col = False
        penetration = gs.qd_float(0.0)
        normal = qd.Vector.zero(gs.qd_float, 3)
        contact_pos = qd.Vector.zero(gs.qd_float, 3)

        n_con = gs.qd_int(0)
        axis_0 = qd.Vector.zero(gs.qd_float, 3)
        axis_1 = qd.Vector.zero(gs.qd_float, 3)
        qrot = qd.Vector.zero(gs.qd_float, 4)

        i_pair = collider_info.collision_pair_idx[(i_gb, i_ga) if i_ga > i_gb else (i_ga, i_gb)]
        for i_detection in range(5):
            prefer_gjk = (
                collider_static_config.ccd_algorithm == CCD_ALGORITHM_CODE.GJK
                or collider_static_config.ccd_algorithm == CCD_ALGORITHM_CODE.MJ_GJK
            )

            # Apply perturbations to thread-local state
            if multi_contact and is_col_0:
                if qd.static(static_rigid_sim_config.enable_mujoco_compatibility):
                    # Match MuJoCo's perturbation pattern: single axis at a time
                    # i_detection 1: (axis_0, -angle), 2: (axis_0, +angle),
                    # 3: (axis_1, -angle), 4: (axis_1, +angle)
                    axis_idx = (i_detection - 1) // 2
                    angle_sign = 2 * ((i_detection - 1) % 2) - 1
                    axis = axis_0 if axis_idx == 0 else axis_1
                    qrot = gu.qd_rotvec_to_quat(angle_sign * collider_info.mc_perturbation[None] * axis, EPS)
                else:
                    # Perturbation axis must not be aligned with the principal axes of inertia the geometry,
                    # otherwise it would be more sensitive to ill-conditioning.
                    axis = (2 * (i_detection % 2) - 1) * axis_0 + (1 - 2 * ((i_detection // 2) % 2)) * axis_1
                    qrot = gu.qd_rotvec_to_quat(collider_info.mc_perturbation[None] * axis, EPS)

                # Apply perturbation starting from original state
                ga_pos_current, ga_quat_current = func_rotate_frame(
                    ga_pos_original, ga_quat_original, contact_pos_0, qrot
                )
                gb_pos_current, gb_quat_current = func_rotate_frame(
                    gb_pos_original, gb_quat_original, contact_pos_0, gu.qd_inv_quat(qrot)
                )

            if (multi_contact and is_col_0) or (i_detection == 0):
                if geoms_info.type[i_ga] == gs.GEOM_TYPE.CAPSULE and geoms_info.type[i_gb] == gs.GEOM_TYPE.CAPSULE:
                    is_col, normal, contact_pos, penetration = capsule_contact.func_capsule_capsule_contact(
                        i_ga,
                        i_gb,
                        ga_pos_current,
                        ga_quat_current,
                        gb_pos_current,
                        gb_quat_current,
                        geoms_info,
                        rigid_global_info,
                    )
                elif geoms_info.type[i_ga] == gs.GEOM_TYPE.SPHERE and geoms_info.type[i_gb] == gs.GEOM_TYPE.CAPSULE:
                    is_col, normal, contact_pos, penetration = capsule_contact.func_sphere_capsule_contact(
                        i_ga,
                        i_gb,
                        ga_pos_current,
                        ga_quat_current,
                        gb_pos_current,
                        gb_quat_current,
                        geoms_info,
                        rigid_global_info,
                    )
                elif geoms_info.type[i_ga] == gs.GEOM_TYPE.SPHERE and geoms_info.type[i_gb] == gs.GEOM_TYPE.BOX:
                    is_col, normal, contact_pos, penetration = func_sphere_box_contact(
                        i_ga,
                        i_gb,
                        ga_pos_current,
                        ga_quat_current,
                        gb_pos_current,
                        gb_quat_current,
                        geoms_info,
                        rigid_global_info,
                    )
                elif geoms_info.type[i_ga] == gs.GEOM_TYPE.PLANE:
                    plane_dir = qd.Vector(
                        [geoms_info.data[i_ga][0], geoms_info.data[i_ga][1], geoms_info.data[i_ga][2]], dt=gs.qd_float
                    )
                    plane_dir = gu.qd_transform_by_quat(plane_dir, ga_quat_current)
                    normal = -plane_dir.normalized()

                    v1 = mpr.support_driver(
                        geoms_info,
                        collider_state,
                        collider_static_config,
                        support_field_info,
                        normal,
                        i_gb,
                        i_b,
                        gb_pos_current,
                        gb_quat_current,
                    )
                    penetration = normal.dot(v1 - ga_pos_current)
                    contact_pos = v1 - 0.5 * penetration * normal
                    is_col = penetration > 0.0
                else:
                    ### MPR, MJ_MPR
                    if qd.static(
                        collider_static_config.ccd_algorithm in (CCD_ALGORITHM_CODE.MPR, CCD_ALGORITHM_CODE.MJ_MPR)
                    ):
                        # Try using MPR before anything else
                        is_mpr_updated = False
                        normal_ws = collider_state.contact_cache.normal[i_pair, i_b]
                        is_mpr_guess_direction_available = (qd.abs(normal_ws) > EPS).any()
                        for i_mpr in range(2):
                            if i_mpr == 1:
                                # Try without warm-start if no contact was detected using it.
                                # When penetration depth is very shallow, MPR may wrongly classify two geometries as not
                                # in contact while they actually are. This helps to improve contact persistence without
                                # increasing much the overall computational cost since the fallback should not be
                                # triggered very often.
                                if qd.static(not static_rigid_sim_config.enable_mujoco_compatibility):
                                    if (i_detection == 0) and not is_col and is_mpr_guess_direction_available:
                                        normal_ws = qd.Vector.zero(gs.qd_float, 3)
                                        is_mpr_guess_direction_available = False
                                        is_mpr_updated = False

                            if not is_mpr_updated:
                                is_col, normal, penetration, contact_pos = mpr.func_mpr_contact(
                                    geoms_info,
                                    geoms_init_AABB,
                                    rigid_global_info,
                                    static_rigid_sim_config,
                                    collider_state,
                                    collider_static_config,
                                    mpr_state,
                                    mpr_info,
                                    support_field_info,
                                    i_ga,
                                    i_gb,
                                    i_b,
                                    normal_ws,
                                    ga_pos_current,
                                    ga_quat_current,
                                    gb_pos_current,
                                    gb_quat_current,
                                )
                                is_mpr_updated = True

                        # Fallback on GJK if collision is detected by MPR if the initial penetration is already quite
                        # large, and either no collision direction was cached or the geometries have large overlap. This
                        # contact information provided by MPR may be unreliable in these cases.
                        if qd.static(collider_static_config.ccd_algorithm == CCD_ALGORITHM_CODE.MPR):
                            if penetration > tolerance:
                                prefer_gjk = not is_mpr_guess_direction_available or (
                                    collider_info.mc_tolerance[None] * penetration
                                    >= collider_info.mpr_to_gjk_overlap_ratio[None] * tolerance
                                )

                    ### GJK, MJ_GJK
                    # TODO: Add support of smooth refinement to differentiable contact.
                    if qd.static(collider_static_config.ccd_algorithm != CCD_ALGORITHM_CODE.MJ_MPR):
                        if prefer_gjk:
                            if qd.static(static_rigid_sim_config.requires_grad):
                                diff_gjk.func_gjk_contact(
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
                                gjk.func_gjk_contact(
                                    geoms_state,
                                    geoms_info,
                                    verts_info,
                                    faces_info,
                                    rigid_global_info,
                                    static_rigid_sim_config,
                                    collider_state,
                                    collider_static_config,
                                    gjk_state,
                                    gjk_info,
                                    gjk_static_config,
                                    support_field_info,
                                    i_ga,
                                    i_gb,
                                    i_b,
                                    ga_pos_current,
                                    ga_quat_current,
                                    gb_pos_current,
                                    gb_quat_current,
                                )

                            is_col = gjk_state.is_col[i_b] == 1
                            penetration = gjk_state.penetration[i_b]
                            n_contacts = gjk_state.n_contacts[i_b]

                            if is_col:
                                if qd.static(static_rigid_sim_config.requires_grad):
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
                                            i_pair,
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
                                            if i_c < qd.static(collider_static_config.n_contacts_per_convex_pair):
                                                contact_pos = gjk_state.contact_pos[i_b, i_c]
                                                normal = gjk_state.normal[i_b, i_c]
                                                contact_pos = func_apply_smooth_refinement(
                                                    i_ga,
                                                    i_gb,
                                                    normal,
                                                    penetration,
                                                    contact_pos,
                                                    ga_pos_current,
                                                    ga_quat_current,
                                                    gb_pos_current,
                                                    gb_quat_current,
                                                    geoms_info,
                                                    static_rigid_sim_config,
                                                )
                                                func_add_contact(
                                                    i_ga,
                                                    i_gb,
                                                    normal,
                                                    contact_pos,
                                                    penetration,
                                                    i_b,
                                                    i_pair,
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

            # Refine the unperturbed (i_detection == 0) contact here; perturbed contacts are refined inside
            # func_recompute_perturbed_contact after the perturbation is reverted, on the canonical (unperturbed) pose.
            if is_col and i_detection == 0:
                contact_pos = func_apply_smooth_refinement(
                    i_ga,
                    i_gb,
                    normal,
                    penetration,
                    contact_pos,
                    ga_pos_current,
                    ga_quat_current,
                    gb_pos_current,
                    gb_quat_current,
                    geoms_info,
                    static_rigid_sim_config,
                )

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
                        i_pair,
                        geoms_state,
                        geoms_info,
                        collider_state,
                        collider_info,
                        errno,
                    )
                    if multi_contact:
                        # Perturb geom_a around two orthogonal axes to find multiple contacts
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

                    if qd.static(
                        collider_static_config.ccd_algorithm in (CCD_ALGORITHM_CODE.MPR, CCD_ALGORITHM_CODE.GJK)
                    ):
                        collider_state.contact_cache.normal[i_pair, i_b] = normal
                else:
                    # Clear collision normal cache if not in contact
                    collider_state.contact_cache.normal[i_pair, i_b] = qd.Vector.zero(gs.qd_float, 3)
            elif multi_contact and is_col:
                # For perturbed iterations (i_detection > 0), recompute the contact from the deepest contact points
                # discovered by the perturbed detection, evaluated on the unperturbed geometries. This applies to all
                # collision methods when multi-contact is enabled, except mujoco compatible. When the correction is
                # skipped (mujoco compatible), is_exact stays False so the lenient acceptance threshold is used.
                is_exact = False
                if qd.static(
                    collider_static_config.ccd_algorithm not in (CCD_ALGORITHM_CODE.MJ_MPR, CCD_ALGORITHM_CODE.MJ_GJK)
                ):
                    _used_gjk = prefer_gjk
                    normal, penetration, contact_pos, is_exact = func_recompute_perturbed_contact(
                        i_ga,
                        i_gb,
                        i_b,
                        normal,
                        penetration,
                        contact_pos,
                        normal_0,
                        contact_pos_0,
                        qrot,
                        ga_pos_original,
                        ga_quat_original,
                        gb_pos_original,
                        gb_quat_original,
                        _used_gjk,
                        geoms_info,
                        rigid_global_info,
                        collider_info,
                        mpr_state,
                        gjk_state,
                        static_rigid_sim_config,
                    )

                # For MuJoCo-compatible GJK, set penetration of perturbed contacts to equal the initial contact's
                # penetration, matching MuJoCo's behavior (engine_collision_convex.c:1010).
                if qd.static(
                    collider_static_config.ccd_algorithm in (CCD_ALGORITHM_CODE.MJ_MPR, CCD_ALGORITHM_CODE.MJ_GJK)
                ):
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
                    # When the correction is exact, a fictitious candidate (one that only touches because of the
                    # perturbation) reverts to a non-positive penetration and is discarded right away. When it is only
                    # approximate, keep the negative tolerance so a genuine contact is not dropped by first-order error.
                    if penetration > (0.0 if is_exact else -tolerance):
                        penetration = qd.max(penetration, 0.0)
                        func_add_contact(
                            i_ga,
                            i_gb,
                            normal,
                            contact_pos,
                            penetration,
                            i_b,
                            i_pair,
                            geoms_state,
                            geoms_info,
                            collider_state,
                            collider_info,
                            errno,
                        )
                        n_con = n_con + 1


@qd.func
def _func_multicontact_run_detection(
    i_ga,
    i_gb,
    i_scratch,
    i_b,
    ga_pos: qd.types.vector(3),
    ga_quat: qd.types.vector(4),
    gb_pos: qd.types.vector(3),
    gb_quat: qd.types.vector(4),
    geoms_state: array_class.GeomsState,
    geoms_info: array_class.GeomsInfo,
    geoms_init_AABB: array_class.GeomsInitAABB,
    verts_info: array_class.VertsInfo,
    faces_info: array_class.FacesInfo,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: qd.template(),
    collider_state: array_class.ColliderState,
    collider_info: array_class.ColliderInfo,
    collider_static_config: qd.template(),
    mpr_state: array_class.MPRState,
    mpr_info: array_class.MPRInfo,
    gjk_state: array_class.GJKState,
    gjk_info: array_class.GJKInfo,
    gjk_static_config: qd.template(),
    support_field_info: array_class.SupportFieldInfo,
    i_pair,
    use_gjk: qd.template(),
    is_initial_detection: qd.template(),
):
    """Run one detection (capsule/plane/MPR/GJK) and return (is_col, normal, contact_pos, penetration, used_gjk)."""
    EPS = rigid_global_info.EPS[None]
    is_col = False
    penetration = gs.qd_float(0.0)
    normal = qd.Vector.zero(gs.qd_float, 3)
    contact_pos = qd.Vector.zero(gs.qd_float, 3)
    used_gjk = False
    tolerance = func_compute_tolerance(i_ga, i_gb, i_b, collider_info.mc_tolerance[None], geoms_info, geoms_init_AABB)

    if geoms_info.type[i_ga] == gs.GEOM_TYPE.CAPSULE and geoms_info.type[i_gb] == gs.GEOM_TYPE.CAPSULE:
        is_col, normal, contact_pos, penetration = capsule_contact.func_capsule_capsule_contact(
            i_ga,
            i_gb,
            ga_pos,
            ga_quat,
            gb_pos,
            gb_quat,
            geoms_info,
            rigid_global_info,
        )
    elif geoms_info.type[i_ga] == gs.GEOM_TYPE.SPHERE and geoms_info.type[i_gb] == gs.GEOM_TYPE.CAPSULE:
        is_col, normal, contact_pos, penetration = capsule_contact.func_sphere_capsule_contact(
            i_ga,
            i_gb,
            ga_pos,
            ga_quat,
            gb_pos,
            gb_quat,
            geoms_info,
            rigid_global_info,
        )
    elif geoms_info.type[i_ga] == gs.GEOM_TYPE.SPHERE and geoms_info.type[i_gb] == gs.GEOM_TYPE.BOX:
        is_col, normal, contact_pos, penetration = func_sphere_box_contact(
            i_ga,
            i_gb,
            ga_pos,
            ga_quat,
            gb_pos,
            gb_quat,
            geoms_info,
            rigid_global_info,
        )
    elif geoms_info.type[i_ga] == gs.GEOM_TYPE.PLANE:
        plane_dir = qd.Vector(
            [geoms_info.data[i_ga][0], geoms_info.data[i_ga][1], geoms_info.data[i_ga][2]], dt=gs.qd_float
        )
        plane_dir = gu.qd_transform_by_quat(plane_dir, ga_quat)
        normal = -plane_dir.normalized()
        v1 = mpr.support_driver(
            geoms_info,
            collider_state,
            collider_static_config,
            support_field_info,
            normal,
            i_gb,
            i_b,
            gb_pos,
            gb_quat,
        )
        penetration = normal.dot(v1 - ga_pos)
        contact_pos = v1 - 0.5 * penetration * normal
        is_col = penetration > 0.0
    else:
        if qd.static(collider_static_config.ccd_algorithm in (CCD_ALGORITHM_CODE.MPR, CCD_ALGORITHM_CODE.MJ_MPR)):
            if not use_gjk:
                is_mpr_updated = False
                normal_ws = collider_state.contact_cache.normal[i_pair, i_b]
                is_mpr_guess_direction_available = (qd.abs(normal_ws) > EPS).any()
                for i_mpr in range(2):
                    if i_mpr == 1:
                        if qd.static(not static_rigid_sim_config.enable_mujoco_compatibility):
                            if is_initial_detection and not is_col and is_mpr_guess_direction_available:
                                normal_ws = qd.Vector.zero(gs.qd_float, 3)
                                is_mpr_guess_direction_available = False
                                is_mpr_updated = False

                    if not is_mpr_updated:
                        is_col, normal, penetration, contact_pos = mpr.func_mpr_contact(
                            geoms_info,
                            geoms_init_AABB,
                            rigid_global_info,
                            static_rigid_sim_config,
                            collider_state,
                            collider_static_config,
                            mpr_state,
                            mpr_info,
                            support_field_info,
                            i_ga,
                            i_gb,
                            i_scratch,
                            normal_ws,
                            ga_pos,
                            ga_quat,
                            gb_pos,
                            gb_quat,
                        )
                        is_mpr_updated = True

        if qd.static(collider_static_config.ccd_algorithm != CCD_ALGORITHM_CODE.MJ_MPR):
            if use_gjk:
                if qd.static(not static_rigid_sim_config.requires_grad):
                    gjk.func_gjk_contact(
                        geoms_state,
                        geoms_info,
                        verts_info,
                        faces_info,
                        rigid_global_info,
                        static_rigid_sim_config,
                        collider_state,
                        collider_static_config,
                        gjk_state,
                        gjk_info,
                        gjk_static_config,
                        support_field_info,
                        i_ga,
                        i_gb,
                        i_scratch,
                        ga_pos,
                        ga_quat,
                        gb_pos,
                        gb_quat,
                    )
                    is_col = gjk_state.is_col[i_scratch] == 1
                    penetration = gjk_state.penetration[i_scratch]
                    if is_col:
                        contact_pos = gjk_state.contact_pos[i_scratch, 0]
                        normal = gjk_state.normal[i_scratch, 0]
                    used_gjk = True

    return is_col, normal, contact_pos, penetration, used_gjk


@qd.func
def _func_multicontact_mpr(
    i_scratch,
    i_b,
    i_ga,
    i_gb,
    i_pair,
    contact_pos_0: qd.types.vector(3),
    normal_0: qd.types.vector(3),
    penetration_0,
    links_state: array_class.LinksState,
    links_info: array_class.LinksInfo,
    geoms_state: array_class.GeomsState,
    geoms_info: array_class.GeomsInfo,
    geoms_init_AABB: array_class.GeomsInitAABB,
    verts_info: array_class.VertsInfo,
    faces_info: array_class.FacesInfo,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: qd.template(),
    collider_state: array_class.ColliderState,
    collider_info: array_class.ColliderInfo,
    collider_static_config: qd.template(),
    mpr_state: array_class.MPRState,
    mpr_info: array_class.MPRInfo,
    gjk_state: array_class.GJKState,
    gjk_info: array_class.GJKInfo,
    gjk_static_config: qd.template(),
    support_field_info: array_class.SupportFieldInfo,
    errno: qd.Tensor,
):
    """Compute all contacts (0 through 4) for a pair and write them contiguously
    via a single atomic reservation, ensuring deterministic per-pair ordering.
    If any perturbation probe returns an unreliable large penetration, the pair
    is upgraded to the GJK queue for reprocessing instead."""
    EPS = rigid_global_info.EPS[None]

    ga_pos_original = geoms_state.pos[i_ga, i_b]
    ga_quat_original = geoms_state.quat[i_ga, i_b]
    gb_pos_original = geoms_state.pos[i_gb, i_b]
    gb_quat_original = geoms_state.quat[i_gb, i_b]

    tolerance = func_compute_tolerance(i_ga, i_gb, i_b, collider_info.mc_tolerance[None], geoms_info, geoms_init_AABB)

    axis_0, axis_1 = func_contact_orthogonals(
        i_ga,
        i_gb,
        normal_0,
        i_b,
        links_state,
        links_info,
        geoms_state,
        geoms_info,
        geoms_init_AABB,
        rigid_global_info,
        static_rigid_sim_config,
    )

    n_con = gs.qd_int(1)
    local_contact_pos = qd.Matrix.zero(gs.qd_float, 5, 3)
    local_normal = qd.Matrix.zero(gs.qd_float, 5, 3)
    local_penetration = qd.Matrix.zero(gs.qd_float, 5, 1)

    local_contact_pos[0, 0] = contact_pos_0[0]
    local_contact_pos[0, 1] = contact_pos_0[1]
    local_contact_pos[0, 2] = contact_pos_0[2]
    local_normal[0, 0] = normal_0[0]
    local_normal[0, 1] = normal_0[1]
    local_normal[0, 2] = normal_0[2]
    local_penetration[0, 0] = penetration_0

    needs_gjk_upgrade = False

    for i_detection in range(4):
        if not needs_gjk_upgrade:
            i_det = i_detection + 1
            axis = (2 * (i_det % 2) - 1) * axis_0 + (1 - 2 * ((i_det // 2) % 2)) * axis_1
            qrot = gu.qd_rotvec_to_quat(collider_info.mc_perturbation[None] * axis, EPS)

            ga_pos_current, ga_quat_current = func_rotate_frame(ga_pos_original, ga_quat_original, contact_pos_0, qrot)
            gb_pos_current, gb_quat_current = func_rotate_frame(
                gb_pos_original, gb_quat_original, contact_pos_0, gu.qd_inv_quat(qrot)
            )

            is_col, normal, contact_pos, penetration, _used_gjk = _func_multicontact_run_detection(
                i_ga,
                i_gb,
                i_scratch,
                i_b,
                ga_pos_current,
                ga_quat_current,
                gb_pos_current,
                gb_quat_current,
                geoms_state,
                geoms_info,
                geoms_init_AABB,
                verts_info,
                faces_info,
                rigid_global_info,
                static_rigid_sim_config,
                collider_state,
                collider_info,
                collider_static_config,
                mpr_state,
                mpr_info,
                gjk_state,
                gjk_info,
                gjk_static_config,
                support_field_info,
                i_pair,
                use_gjk=False,
                is_initial_detection=False,
            )

            if qd.static(collider_static_config.ccd_algorithm == CCD_ALGORITHM_CODE.MPR):
                if is_col and penetration > tolerance:
                    if (
                        collider_info.mc_tolerance[None] * penetration
                        >= collider_info.mpr_to_gjk_overlap_ratio[None] * tolerance
                    ):
                        needs_gjk_upgrade = True

            if is_col and not needs_gjk_upgrade:
                # The perturbed contact is refined inside func_recompute_perturbed_contact (after the perturbation is
                # reverted, on the canonical pose); no pre-reversal refinement is needed here.
                is_exact = False
                if qd.static(
                    collider_static_config.ccd_algorithm not in (CCD_ALGORITHM_CODE.MJ_MPR, CCD_ALGORITHM_CODE.MJ_GJK)
                ):
                    normal, penetration, contact_pos, is_exact = func_recompute_perturbed_contact(
                        i_ga,
                        i_gb,
                        i_scratch,
                        normal,
                        penetration,
                        contact_pos,
                        normal_0,
                        contact_pos_0,
                        qrot,
                        ga_pos_original,
                        ga_quat_original,
                        gb_pos_original,
                        gb_quat_original,
                        _used_gjk,
                        geoms_info,
                        rigid_global_info,
                        collider_info,
                        mpr_state,
                        gjk_state,
                        static_rigid_sim_config,
                    )

                repeated = False
                for i_c in range(n_con):
                    if not repeated:
                        prev = qd.Vector(
                            [local_contact_pos[i_c, 0], local_contact_pos[i_c, 1], local_contact_pos[i_c, 2]],
                            dt=gs.qd_float,
                        )
                        if (contact_pos - prev).norm() < tolerance:
                            repeated = True

                if not repeated:
                    if penetration > (0.0 if is_exact else -tolerance):
                        penetration = qd.max(penetration, 0.0)
                        local_contact_pos[n_con, 0] = contact_pos[0]
                        local_contact_pos[n_con, 1] = contact_pos[1]
                        local_contact_pos[n_con, 2] = contact_pos[2]
                        local_normal[n_con, 0] = normal[0]
                        local_normal[n_con, 1] = normal[1]
                        local_normal[n_con, 2] = normal[2]
                        local_penetration[n_con, 0] = penetration
                        n_con = n_con + 1

    if needs_gjk_upgrade:
        local_idx = qd.atomic_add(collider_state.narrowphase_work_queues.gjk_queue_size_k2[0], 1)
        idx = collider_state.narrowphase_work_queues.gjk_queue_size[0] + local_idx
        collider_state.narrowphase_work_queues.gjk_i_b[idx] = i_b
        collider_state.narrowphase_work_queues.gjk_i_ga[idx] = i_ga
        collider_state.narrowphase_work_queues.gjk_i_gb[idx] = i_gb
        collider_state.narrowphase_work_queues.gjk_i_pair[idx] = i_pair
    else:
        # Non-atomic pre-check to avoid reserving slots we cannot fill. A rare race between the read and the
        # atomic_add below may still overshoot; in that case we write only the contacts that fit and set errno.
        max_contact_pairs = collider_info.max_contact_pairs[None]
        if collider_state.n_contacts[i_b] + n_con > max_contact_pairs:
            errno[i_b] = errno[i_b] | array_class.ErrorCode.OVERFLOW_COLLISION_PAIRS
        else:
            start_idx = qd.atomic_add(collider_state.n_contacts[i_b], n_con)
            n_con = qd.math.clamp(max_contact_pairs - start_idx, 0, n_con)
            if n_con == 0:
                errno[i_b] = errno[i_b] | array_class.ErrorCode.OVERFLOW_COLLISION_PAIRS
            for i in range(n_con):
                i_c = start_idx + i
                pos_i = qd.Vector(
                    [local_contact_pos[i, 0], local_contact_pos[i, 1], local_contact_pos[i, 2]], dt=gs.qd_float
                )
                normal_i = qd.Vector([local_normal[i, 0], local_normal[i, 1], local_normal[i, 2]], dt=gs.qd_float)
                func_set_contact(
                    i_ga,
                    i_gb,
                    normal_i,
                    pos_i,
                    local_penetration[i, 0],
                    i_b,
                    i_c,
                    i_pair,
                    geoms_state,
                    geoms_info,
                    collider_state,
                    collider_info,
                )


@qd.func
def _func_multicontact_gjk_full(
    i_scratch,
    i_b,
    i_ga,
    i_gb,
    i_pair,
    links_state: array_class.LinksState,
    links_info: array_class.LinksInfo,
    geoms_state: array_class.GeomsState,
    geoms_info: array_class.GeomsInfo,
    geoms_init_AABB: array_class.GeomsInitAABB,
    verts_info: array_class.VertsInfo,
    faces_info: array_class.FacesInfo,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: qd.template(),
    collider_state: array_class.ColliderState,
    collider_info: array_class.ColliderInfo,
    collider_static_config: qd.template(),
    mpr_state: array_class.MPRState,
    mpr_info: array_class.MPRInfo,
    gjk_state: array_class.GJKState,
    gjk_info: array_class.GJKInfo,
    gjk_static_config: qd.template(),
    support_field_info: array_class.SupportFieldInfo,
    diff_contact_input: array_class.DiffContactInput,
    errno: qd.Tensor,
):
    """Run full contact detection (contacts 0-4) using GJK+EPA for pairs that MPR couldn't handle.
    All contacts are collected locally and written contiguously via a single atomic reservation."""
    EPS = rigid_global_info.EPS[None]

    multi_contact = (
        static_rigid_sim_config.enable_multi_contact
        and geoms_info.type[i_ga] != gs.GEOM_TYPE.SPHERE
        and geoms_info.type[i_ga] != gs.GEOM_TYPE.ELLIPSOID
        and geoms_info.type[i_gb] != gs.GEOM_TYPE.SPHERE
        and geoms_info.type[i_gb] != gs.GEOM_TYPE.ELLIPSOID
    )

    tolerance = func_compute_tolerance(i_ga, i_gb, i_b, collider_info.mc_tolerance[None], geoms_info, geoms_init_AABB)
    diff_pos_tolerance = func_compute_tolerance(
        i_ga, i_gb, i_b, collider_info.diff_pos_tolerance[None], geoms_info, geoms_init_AABB
    )
    diff_normal_tolerance = collider_info.diff_normal_tolerance[None]

    ga_pos_original = geoms_state.pos[i_ga, i_b]
    ga_quat_original = geoms_state.quat[i_ga, i_b]
    gb_pos_original = geoms_state.pos[i_gb, i_b]
    gb_quat_original = geoms_state.quat[i_gb, i_b]

    ga_pos_current = ga_pos_original
    ga_quat_current = ga_quat_original
    gb_pos_current = gb_pos_original
    gb_quat_current = gb_quat_original

    is_col_0 = False
    penetration_0 = gs.qd_float(0.0)
    normal_0 = qd.Vector.zero(gs.qd_float, 3)
    contact_pos_0 = qd.Vector.zero(gs.qd_float, 3)

    is_col = False
    penetration = gs.qd_float(0.0)
    normal = qd.Vector.zero(gs.qd_float, 3)
    contact_pos = qd.Vector.zero(gs.qd_float, 3)
    _used_gjk = False

    n_con = gs.qd_int(0)
    local_contact_pos = qd.Matrix.zero(gs.qd_float, 5, 3)
    local_normal = qd.Matrix.zero(gs.qd_float, 5, 3)
    local_penetration = qd.Matrix.zero(gs.qd_float, 5, 1)
    axis_0 = qd.Vector.zero(gs.qd_float, 3)
    axis_1 = qd.Vector.zero(gs.qd_float, 3)
    qrot = qd.Vector.zero(gs.qd_float, 4)
    gjk_multi_done = False

    for i_detection in range(5):
        if not gjk_multi_done:
            if multi_contact and is_col_0:
                axis = (2 * (i_detection % 2) - 1) * axis_0 + (1 - 2 * ((i_detection // 2) % 2)) * axis_1
                qrot = gu.qd_rotvec_to_quat(collider_info.mc_perturbation[None] * axis, EPS)
                ga_pos_current, ga_quat_current = func_rotate_frame(
                    ga_pos_original, ga_quat_original, contact_pos_0, qrot
                )
                gb_pos_current, gb_quat_current = func_rotate_frame(
                    gb_pos_original, gb_quat_original, contact_pos_0, gu.qd_inv_quat(qrot)
                )

            if (multi_contact and is_col_0) or (i_detection == 0):
                is_col, normal, contact_pos, penetration, _used_gjk = _func_multicontact_run_detection(
                    i_ga,
                    i_gb,
                    i_scratch,
                    i_b,
                    ga_pos_current,
                    ga_quat_current,
                    gb_pos_current,
                    gb_quat_current,
                    geoms_state,
                    geoms_info,
                    geoms_init_AABB,
                    verts_info,
                    faces_info,
                    rigid_global_info,
                    static_rigid_sim_config,
                    collider_state,
                    collider_info,
                    collider_static_config,
                    mpr_state,
                    mpr_info,
                    gjk_state,
                    gjk_info,
                    gjk_static_config,
                    support_field_info,
                    i_pair,
                    use_gjk=True,
                    is_initial_detection=i_detection == 0,
                )

                if is_col and _used_gjk:
                    n_contacts_gjk = gjk_state.n_contacts[i_scratch]
                    if gjk_state.multi_contact_flag[i_scratch]:
                        for i_c in range(n_contacts_gjk):
                            if i_c < qd.static(collider_static_config.n_contacts_per_convex_pair):
                                gjk_contact_pos = gjk_state.contact_pos[i_scratch, i_c]
                                gjk_normal = gjk_state.normal[i_scratch, i_c]
                                gjk_contact_pos = func_apply_smooth_refinement(
                                    i_ga,
                                    i_gb,
                                    gjk_normal,
                                    penetration,
                                    gjk_contact_pos,
                                    ga_pos_current,
                                    ga_quat_current,
                                    gb_pos_current,
                                    gb_quat_current,
                                    geoms_info,
                                    static_rigid_sim_config,
                                )
                                local_contact_pos[n_con, 0] = gjk_contact_pos[0]
                                local_contact_pos[n_con, 1] = gjk_contact_pos[1]
                                local_contact_pos[n_con, 2] = gjk_contact_pos[2]
                                local_normal[n_con, 0] = gjk_normal[0]
                                local_normal[n_con, 1] = gjk_normal[1]
                                local_normal[n_con, 2] = gjk_normal[2]
                                local_penetration[n_con, 0] = penetration
                                n_con = n_con + 1
                        gjk_multi_done = True

            if i_detection == 0 and not gjk_multi_done:
                is_col_0, normal_0, penetration_0, contact_pos_0 = is_col, normal, penetration, contact_pos
                if is_col_0:
                    contact_pos_0 = func_apply_smooth_refinement(
                        i_ga,
                        i_gb,
                        normal_0,
                        penetration_0,
                        contact_pos_0,
                        ga_pos_current,
                        ga_quat_current,
                        gb_pos_current,
                        gb_quat_current,
                        geoms_info,
                        static_rigid_sim_config,
                    )
                    local_contact_pos[0, 0] = contact_pos_0[0]
                    local_contact_pos[0, 1] = contact_pos_0[1]
                    local_contact_pos[0, 2] = contact_pos_0[2]
                    local_normal[0, 0] = normal_0[0]
                    local_normal[0, 1] = normal_0[1]
                    local_normal[0, 2] = normal_0[2]
                    local_penetration[0, 0] = penetration_0
                    n_con = 1
                    if multi_contact:
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

                    if qd.static(
                        collider_static_config.ccd_algorithm in (CCD_ALGORITHM_CODE.MPR, CCD_ALGORITHM_CODE.GJK)
                    ):
                        collider_state.contact_cache.normal[i_pair, i_b] = normal
                else:
                    collider_state.contact_cache.normal[i_pair, i_b] = qd.Vector.zero(gs.qd_float, 3)
            elif not gjk_multi_done and multi_contact and is_col:
                # The perturbed contact is refined inside func_recompute_perturbed_contact (after the perturbation is
                # reverted, on the canonical pose); no pre-reversal refinement is needed here.
                is_exact = False
                if qd.static(
                    collider_static_config.ccd_algorithm not in (CCD_ALGORITHM_CODE.MJ_MPR, CCD_ALGORITHM_CODE.MJ_GJK)
                ):
                    normal, penetration, contact_pos, is_exact = func_recompute_perturbed_contact(
                        i_ga,
                        i_gb,
                        i_scratch,
                        normal,
                        penetration,
                        contact_pos,
                        normal_0,
                        contact_pos_0,
                        qrot,
                        ga_pos_original,
                        ga_quat_original,
                        gb_pos_original,
                        gb_quat_original,
                        _used_gjk,
                        geoms_info,
                        rigid_global_info,
                        collider_info,
                        mpr_state,
                        gjk_state,
                        static_rigid_sim_config,
                    )

                repeated = False
                for i_c in range(n_con):
                    if not repeated:
                        prev = qd.Vector(
                            [local_contact_pos[i_c, 0], local_contact_pos[i_c, 1], local_contact_pos[i_c, 2]],
                            dt=gs.qd_float,
                        )
                        if (contact_pos - prev).norm() < tolerance:
                            repeated = True

                if not repeated:
                    if penetration > (0.0 if is_exact else -tolerance):
                        penetration = qd.max(penetration, 0.0)
                        local_contact_pos[n_con, 0] = contact_pos[0]
                        local_contact_pos[n_con, 1] = contact_pos[1]
                        local_contact_pos[n_con, 2] = contact_pos[2]
                        local_normal[n_con, 0] = normal[0]
                        local_normal[n_con, 1] = normal[1]
                        local_normal[n_con, 2] = normal[2]
                        local_penetration[n_con, 0] = penetration
                        n_con = n_con + 1

    if n_con > 0:
        # Non-atomic pre-check (see comment in _func_multicontact_mpr).
        max_contact_pairs = collider_info.max_contact_pairs[None]
        if collider_state.n_contacts[i_b] + n_con > max_contact_pairs:
            errno[i_b] = errno[i_b] | array_class.ErrorCode.OVERFLOW_COLLISION_PAIRS
        else:
            start_idx = qd.atomic_add(collider_state.n_contacts[i_b], n_con)
            n_con = qd.math.clamp(max_contact_pairs - start_idx, 0, n_con)
            if n_con == 0:
                errno[i_b] = errno[i_b] | array_class.ErrorCode.OVERFLOW_COLLISION_PAIRS
            for i in range(n_con):
                i_c = start_idx + i
                pos_i = qd.Vector(
                    [local_contact_pos[i, 0], local_contact_pos[i, 1], local_contact_pos[i, 2]], dt=gs.qd_float
                )
                normal_i = qd.Vector([local_normal[i, 0], local_normal[i, 1], local_normal[i, 2]], dt=gs.qd_float)
                func_set_contact(
                    i_ga,
                    i_gb,
                    normal_i,
                    pos_i,
                    local_penetration[i, 0],
                    i_b,
                    i_c,
                    i_pair,
                    geoms_state,
                    geoms_info,
                    collider_state,
                    collider_info,
                )


@qd.kernel(fastcache=True)
def _func_narrowphase_multicontact_mixed(
    links_state: array_class.LinksState,
    links_info: array_class.LinksInfo,
    geoms_state: array_class.GeomsState,
    geoms_info: array_class.GeomsInfo,
    geoms_init_AABB: array_class.GeomsInitAABB,
    verts_info: array_class.VertsInfo,
    faces_info: array_class.FacesInfo,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: qd.template(),
    collider_state: array_class.ColliderState,
    collider_info: array_class.ColliderInfo,
    collider_static_config: qd.template(),
    mpr_state: array_class.MPRState,
    mpr_info: array_class.MPRInfo,
    gjk_state: array_class.GJKState,
    gjk_info: array_class.GJKInfo,
    gjk_static_config: qd.template(),
    support_field_info: array_class.SupportFieldInfo,
    diff_contact_input: array_class.DiffContactInput,
    errno: qd.Tensor,
    n_gjk_threads: qd.template(),
    n_total_threads: qd.template(),
    max_items_per_thread: qd.template(),
):
    for i_tid in range(n_total_threads):
        if i_tid < qd.static(n_gjk_threads):
            # GJK partition: pull from gjk_queue
            for _iter in range(max_items_per_thread):
                idx = qd.atomic_add(collider_state.narrowphase_work_queues.gjk_work_counter[0], 1)
                if idx >= collider_state.narrowphase_work_queues.gjk_queue_size[0]:
                    break
                i_b = collider_state.narrowphase_work_queues.gjk_i_b[idx]
                i_ga = collider_state.narrowphase_work_queues.gjk_i_ga[idx]
                i_gb = collider_state.narrowphase_work_queues.gjk_i_gb[idx]
                i_pair = collider_state.narrowphase_work_queues.gjk_i_pair[idx]

                _func_multicontact_gjk_full(
                    i_tid,
                    i_b,
                    i_ga,
                    i_gb,
                    i_pair,
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
                    collider_info,
                    collider_static_config,
                    mpr_state,
                    mpr_info,
                    gjk_state,
                    gjk_info,
                    gjk_static_config,
                    support_field_info,
                    diff_contact_input,
                    errno,
                )
        else:
            # MPR partition: pull from mpr_queue
            for _iter in range(max_items_per_thread):
                idx = qd.atomic_add(collider_state.narrowphase_work_queues.mpr_work_counter[0], 1)
                if idx >= collider_state.narrowphase_work_queues.mpr_queue_size[0]:
                    break
                i_b = collider_state.narrowphase_work_queues.mpr_i_b[idx]
                i_ga = collider_state.narrowphase_work_queues.mpr_i_ga[idx]
                i_gb = collider_state.narrowphase_work_queues.mpr_i_gb[idx]
                i_pair = collider_state.narrowphase_work_queues.mpr_i_pair[idx]
                contact_pos_0 = collider_state.narrowphase_work_queues.mpr_contact_pos_0[idx]
                normal_0 = collider_state.narrowphase_work_queues.mpr_normal_0[idx]
                penetration_0 = collider_state.narrowphase_work_queues.mpr_penetration_0[idx]

                _func_multicontact_mpr(
                    i_tid,
                    i_b,
                    i_ga,
                    i_gb,
                    i_pair,
                    contact_pos_0,
                    normal_0,
                    penetration_0,
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
                    collider_info,
                    collider_static_config,
                    mpr_state,
                    mpr_info,
                    gjk_state,
                    gjk_info,
                    gjk_static_config,
                    support_field_info,
                    errno,
                )


@qd.kernel
def _func_reset_narrowphase_work_queues(
    collider_state: array_class.ColliderState,
):
    for _i in range(1):
        collider_state.narrowphase_work_queues.mpr_queue_size[0] = 0
        collider_state.narrowphase_work_queues.gjk_queue_size[0] = 0
        collider_state.narrowphase_work_queues.gjk_queue_size_k2[0] = 0
        collider_state.narrowphase_work_queues.mpr_work_counter[0] = 0
        collider_state.narrowphase_work_queues.gjk_work_counter[0] = 0


@qd.kernel
def _func_prepare_gjk_rerun(
    collider_state: array_class.ColliderState,
):
    for _i in range(1):
        k1_size = collider_state.narrowphase_work_queues.gjk_queue_size[0]
        k2_count = collider_state.narrowphase_work_queues.gjk_queue_size_k2[0]
        collider_state.narrowphase_work_queues.gjk_work_counter[0] = k1_size
        collider_state.narrowphase_work_queues.gjk_queue_size[0] = k1_size + k2_count
        collider_state.narrowphase_work_queues.mpr_queue_size[0] = 0
        collider_state.narrowphase_work_queues.mpr_work_counter[0] = 0


@qd.func
def _func_enqueue_for_multicontact(
    collider_state: array_class.ColliderState,
    i_b,
    i_ga,
    i_gb,
    i_pair,
    contact_pos_0: qd.types.vector(3),
    normal_0: qd.types.vector(3),
    penetration_0,
    prefer_gjk: bool,
):
    if prefer_gjk:
        idx = qd.atomic_add(collider_state.narrowphase_work_queues.gjk_queue_size[0], 1)
        collider_state.narrowphase_work_queues.gjk_i_b[idx] = i_b
        collider_state.narrowphase_work_queues.gjk_i_ga[idx] = i_ga
        collider_state.narrowphase_work_queues.gjk_i_gb[idx] = i_gb
        collider_state.narrowphase_work_queues.gjk_i_pair[idx] = i_pair
        collider_state.narrowphase_work_queues.gjk_contact_pos_0[idx] = contact_pos_0
        collider_state.narrowphase_work_queues.gjk_normal_0[idx] = normal_0
        collider_state.narrowphase_work_queues.gjk_penetration_0[idx] = penetration_0
    else:
        idx = qd.atomic_add(collider_state.narrowphase_work_queues.mpr_queue_size[0], 1)
        collider_state.narrowphase_work_queues.mpr_i_b[idx] = i_b
        collider_state.narrowphase_work_queues.mpr_i_ga[idx] = i_ga
        collider_state.narrowphase_work_queues.mpr_i_gb[idx] = i_gb
        collider_state.narrowphase_work_queues.mpr_i_pair[idx] = i_pair
        collider_state.narrowphase_work_queues.mpr_contact_pos_0[idx] = contact_pos_0
        collider_state.narrowphase_work_queues.mpr_normal_0[idx] = normal_0
        collider_state.narrowphase_work_queues.mpr_penetration_0[idx] = penetration_0


@qd.kernel(fastcache=True)
def _func_narrowphase_contact0(
    geoms_state: array_class.GeomsState,
    geoms_info: array_class.GeomsInfo,
    geoms_init_AABB: array_class.GeomsInitAABB,
    verts_info: array_class.VertsInfo,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: qd.template(),
    collider_state: array_class.ColliderState,
    collider_info: array_class.ColliderInfo,
    collider_static_config: qd.template(),
    mpr_state: array_class.MPRState,
    mpr_info: array_class.MPRInfo,
    gjk_state: array_class.GJKState,
    gjk_info: array_class.GJKInfo,
    support_field_info: array_class.SupportFieldInfo,
    errno: qd.Tensor,
    n_envs: qd.template(),
    n_chunks: qd.template(),
):
    _grid_size = n_envs * n_chunks
    max_broad_pairs = collider_state.broad_collision_pairs.shape[0]

    for flat_idx in range(_grid_size):
        i_b = flat_idx // n_chunks
        chunk = flat_idx % n_chunks
        n_pairs = collider_state.n_broad_pairs[i_b]
        pair_start = chunk * n_pairs // n_chunks
        pair_end = (chunk + 1) * n_pairs // n_chunks

        for i_pair_local in range(max_broad_pairs):
            i_pair_idx = pair_start + i_pair_local
            if i_pair_idx >= pair_end:
                break

            i_ga = collider_state.broad_collision_pairs[i_pair_idx, i_b][0]
            i_gb = collider_state.broad_collision_pairs[i_pair_idx, i_b][1]

            if geoms_info.type[i_ga] > geoms_info.type[i_gb]:
                i_ga, i_gb = i_gb, i_ga

            if not (
                geoms_info.is_convex[i_ga]
                and geoms_info.is_convex[i_gb]
                and not geoms_info.type[i_gb] == gs.GEOM_TYPE.TERRAIN
                and not (
                    qd.static(static_rigid_sim_config.box_box_detection)
                    and geoms_info.type[i_ga] == gs.GEOM_TYPE.BOX
                    and geoms_info.type[i_gb] == gs.GEOM_TYPE.BOX
                )
            ):
                continue

            if geoms_info.type[i_ga] == gs.GEOM_TYPE.PLANE and geoms_info.type[i_gb] == gs.GEOM_TYPE.BOX:
                continue

            EPS = rigid_global_info.EPS[None]

            multi_contact = (
                static_rigid_sim_config.enable_multi_contact
                and geoms_info.type[i_ga] != gs.GEOM_TYPE.SPHERE
                and geoms_info.type[i_ga] != gs.GEOM_TYPE.ELLIPSOID
                and geoms_info.type[i_gb] != gs.GEOM_TYPE.SPHERE
                and geoms_info.type[i_gb] != gs.GEOM_TYPE.ELLIPSOID
            )

            tolerance = func_compute_tolerance(
                i_ga, i_gb, i_b, collider_info.mc_tolerance[None], geoms_info, geoms_init_AABB
            )

            ga_pos = geoms_state.pos[i_ga, i_b]
            ga_quat = geoms_state.quat[i_ga, i_b]
            gb_pos = geoms_state.pos[i_gb, i_b]
            gb_quat = geoms_state.quat[i_gb, i_b]

            is_col = False
            penetration = gs.qd_float(0.0)
            normal = qd.Vector.zero(gs.qd_float, 3)
            contact_pos = qd.Vector.zero(gs.qd_float, 3)
            prefer_gjk = qd.static(collider_static_config.ccd_algorithm == CCD_ALGORITHM_CODE.GJK) or qd.static(
                collider_static_config.ccd_algorithm == CCD_ALGORITHM_CODE.MJ_GJK
            )

            i_pair = collider_info.collision_pair_idx[(i_gb, i_ga) if i_ga > i_gb else (i_ga, i_gb)]

            if geoms_info.type[i_ga] == gs.GEOM_TYPE.CAPSULE and geoms_info.type[i_gb] == gs.GEOM_TYPE.CAPSULE:
                is_col, normal, contact_pos, penetration = capsule_contact.func_capsule_capsule_contact(
                    i_ga,
                    i_gb,
                    ga_pos,
                    ga_quat,
                    gb_pos,
                    gb_quat,
                    geoms_info,
                    rigid_global_info,
                )
            elif geoms_info.type[i_ga] == gs.GEOM_TYPE.SPHERE and geoms_info.type[i_gb] == gs.GEOM_TYPE.CAPSULE:
                is_col, normal, contact_pos, penetration = capsule_contact.func_sphere_capsule_contact(
                    i_ga,
                    i_gb,
                    ga_pos,
                    ga_quat,
                    gb_pos,
                    gb_quat,
                    geoms_info,
                    rigid_global_info,
                )
            elif geoms_info.type[i_ga] == gs.GEOM_TYPE.SPHERE and geoms_info.type[i_gb] == gs.GEOM_TYPE.BOX:
                is_col, normal, contact_pos, penetration = func_sphere_box_contact(
                    i_ga,
                    i_gb,
                    ga_pos,
                    ga_quat,
                    gb_pos,
                    gb_quat,
                    geoms_info,
                    rigid_global_info,
                )
            elif geoms_info.type[i_ga] == gs.GEOM_TYPE.PLANE:
                plane_dir = qd.Vector(
                    [geoms_info.data[i_ga][0], geoms_info.data[i_ga][1], geoms_info.data[i_ga][2]], dt=gs.qd_float
                )
                plane_dir = gu.qd_transform_by_quat(plane_dir, ga_quat)
                normal = -plane_dir.normalized()
                v1 = mpr.support_driver(
                    geoms_info,
                    collider_state,
                    collider_static_config,
                    support_field_info,
                    normal,
                    i_gb,
                    i_b,
                    gb_pos,
                    gb_quat,
                )
                penetration = normal.dot(v1 - ga_pos)
                contact_pos = v1 - 0.5 * penetration * normal
                is_col = penetration > 0.0
            else:
                if qd.static(
                    collider_static_config.ccd_algorithm in (CCD_ALGORITHM_CODE.GJK, CCD_ALGORITHM_CODE.MJ_GJK)
                ):
                    gjk.clear_cache(gjk_state, flat_idx)
                    distance = gjk.func_gjk(
                        geoms_info,
                        verts_info,
                        static_rigid_sim_config,
                        collider_state,
                        collider_static_config,
                        gjk_state,
                        gjk_info,
                        support_field_info,
                        i_ga,
                        i_gb,
                        flat_idx,
                        ga_pos,
                        ga_quat,
                        gb_pos,
                        gb_quat,
                        shrink_sphere=False,
                    )
                    is_col = distance < gjk_info.collision_eps[None]

                if qd.static(
                    collider_static_config.ccd_algorithm in (CCD_ALGORITHM_CODE.MPR, CCD_ALGORITHM_CODE.MJ_MPR)
                ):
                    is_mpr_updated = False
                    normal_ws = collider_state.contact_cache.normal[i_pair, i_b]
                    is_mpr_guess_direction_available = (qd.abs(normal_ws) > EPS).any()
                    for i_mpr in range(2):
                        if i_mpr == 1:
                            if qd.static(not static_rigid_sim_config.enable_mujoco_compatibility):
                                if not is_col and is_mpr_guess_direction_available:
                                    normal_ws = qd.Vector.zero(gs.qd_float, 3)
                                    is_mpr_guess_direction_available = False
                                    is_mpr_updated = False

                        if not is_mpr_updated:
                            is_col, normal, penetration, contact_pos = mpr.func_mpr_contact(
                                geoms_info,
                                geoms_init_AABB,
                                rigid_global_info,
                                static_rigid_sim_config,
                                collider_state,
                                collider_static_config,
                                mpr_state,
                                mpr_info,
                                support_field_info,
                                i_ga,
                                i_gb,
                                flat_idx,
                                normal_ws,
                                ga_pos,
                                ga_quat,
                                gb_pos,
                                gb_quat,
                            )
                            is_mpr_updated = True

                    if qd.static(collider_static_config.ccd_algorithm == CCD_ALGORITHM_CODE.MPR):
                        if penetration > tolerance:
                            # Contact 0's normal always provides a warmstart for
                            # contacts 1-4, so only prefer GJK when penetration is
                            # genuinely large — not merely because the cache is cold.
                            prefer_gjk = (
                                collider_info.mc_tolerance[None] * penetration
                                >= collider_info.mpr_to_gjk_overlap_ratio[None] * tolerance
                            )

            if is_col:
                if qd.static(collider_static_config.ccd_algorithm in (CCD_ALGORITHM_CODE.MPR, CCD_ALGORITHM_CODE.GJK)):
                    collider_state.contact_cache.normal[i_pair, i_b] = normal
                # Refine the contact position before enqueueing or storing it. The downstream multicontact functions
                # store this as the initial contact (index 0 of local_contact_pos) without re-refining, so refinement
                # must happen here to stay consistent with the monolithic path's consolidated refinement at the start
                # of each i_detection iteration.
                contact_pos = func_apply_smooth_refinement(
                    i_ga,
                    i_gb,
                    normal,
                    penetration,
                    contact_pos,
                    geoms_state.pos[i_ga, i_b],
                    geoms_state.quat[i_ga, i_b],
                    geoms_state.pos[i_gb, i_b],
                    geoms_state.quat[i_gb, i_b],
                    geoms_info,
                    static_rigid_sim_config,
                )
                if prefer_gjk:
                    # GJK algorithm: always enqueue to GJK queue — multicontact
                    # runs full GJK detection regardless of multi_contact.
                    if qd.static(collider_static_config.ccd_algorithm != CCD_ALGORITHM_CODE.MJ_MPR):
                        _func_enqueue_for_multicontact(
                            collider_state,
                            i_b,
                            i_ga,
                            i_gb,
                            i_pair,
                            contact_pos,
                            normal,
                            penetration,
                            prefer_gjk=True,
                        )
                elif multi_contact:
                    # Enqueue for multicontact — multicontact will write all contacts
                    # (including contact 0) contiguously via a single atomic reservation.
                    _func_enqueue_for_multicontact(
                        collider_state,
                        i_b,
                        i_ga,
                        i_gb,
                        i_pair,
                        contact_pos,
                        normal,
                        penetration,
                        prefer_gjk=False,
                    )
                else:
                    func_add_contact(
                        i_ga,
                        i_gb,
                        normal,
                        contact_pos,
                        penetration,
                        i_b,
                        i_pair,
                        geoms_state,
                        geoms_info,
                        collider_state,
                        collider_info,
                        errno,
                        use_atomic=True,
                    )
            elif not is_col:
                collider_state.contact_cache.normal[i_pair, i_b] = qd.Vector.zero(gs.qd_float, 3)


@qd.kernel(fastcache=True)
def func_narrow_phase_convex_vs_convex(
    links_state: array_class.LinksState,
    links_info: array_class.LinksInfo,
    geoms_state: array_class.GeomsState,
    geoms_info: array_class.GeomsInfo,
    geoms_init_AABB: array_class.GeomsInitAABB,
    verts_info: array_class.VertsInfo,
    faces_info: array_class.FacesInfo,
    edges_info: array_class.EdgesInfo,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: qd.template(),
    collider_state: array_class.ColliderState,
    collider_info: array_class.ColliderInfo,
    collider_static_config: qd.template(),
    mpr_state: array_class.MPRState,
    mpr_info: array_class.MPRInfo,
    gjk_state: array_class.GJKState,
    gjk_info: array_class.GJKInfo,
    gjk_static_config: qd.template(),
    sdf_info: array_class.SDFInfo,
    support_field_info: array_class.SupportFieldInfo,
    diff_contact_input: array_class.DiffContactInput,
    errno: qd.Tensor,
):
    _B = collider_state.active_buffer.shape[1]

    qd.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
    for i_b in range(_B):
        for i_pair in range(collider_state.n_broad_pairs[i_b]):
            i_ga = collider_state.broad_collision_pairs[i_pair, i_b][0]
            i_gb = collider_state.broad_collision_pairs[i_pair, i_b][1]

            if geoms_info.type[i_ga] > geoms_info.type[i_gb]:
                i_ga, i_gb = i_gb, i_ga

            if (
                geoms_info.is_convex[i_ga]
                and geoms_info.is_convex[i_gb]
                and not geoms_info.type[i_gb] == gs.GEOM_TYPE.TERRAIN
                and not (
                    static_rigid_sim_config.box_box_detection
                    and geoms_info.type[i_ga] == gs.GEOM_TYPE.BOX
                    and geoms_info.type[i_gb] == gs.GEOM_TYPE.BOX
                )
            ):
                if not (geoms_info.type[i_ga] == gs.GEOM_TYPE.PLANE and geoms_info.type[i_gb] == gs.GEOM_TYPE.BOX):
                    func_convex_convex_contact(
                        i_ga=i_ga,
                        i_gb=i_gb,
                        i_b=i_b,
                        links_state=links_state,
                        links_info=links_info,
                        geoms_state=geoms_state,
                        geoms_info=geoms_info,
                        geoms_init_AABB=geoms_init_AABB,
                        verts_info=verts_info,
                        faces_info=faces_info,
                        rigid_global_info=rigid_global_info,
                        static_rigid_sim_config=static_rigid_sim_config,
                        collider_state=collider_state,
                        collider_info=collider_info,
                        collider_static_config=collider_static_config,
                        mpr_state=mpr_state,
                        mpr_info=mpr_info,
                        gjk_state=gjk_state,
                        gjk_info=gjk_info,
                        gjk_static_config=gjk_static_config,
                        support_field_info=support_field_info,
                        # FIXME: Passing nested data structure as input argument is not supported for now.
                        diff_contact_input=diff_contact_input,
                        errno=errno,
                    )


@qd.kernel(fastcache=True)
def func_narrow_phase_diff_convex_vs_convex(
    geoms_state: array_class.GeomsState,
    geoms_info: array_class.GeomsInfo,
    static_rigid_sim_config: qd.template(),
    collider_state: array_class.ColliderState,
    collider_info: array_class.ColliderInfo,
    gjk_info: array_class.GJKInfo,
    # FIXME: Passing nested data structure as input argument is not supported for now.
    diff_contact_input: array_class.DiffContactInput,
):
    # Compute reference contacts
    qd.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.PARTIAL)
    for i_c, i_b in qd.ndrange(collider_state.contact_data.pos.shape[0], collider_state.active_buffer.shape[1]):
        if i_c < collider_state.n_contacts[i_b]:
            ref_id = collider_state.diff_contact_input.ref_id[i_b, i_c]
            is_ref = i_c == ref_id
            i_ga = collider_state.diff_contact_input.geom_a[i_b, i_c]
            i_gb = collider_state.diff_contact_input.geom_b[i_b, i_c]

            if is_ref:
                ref_penetration = -1.0
                contact_pos, contact_normal, penetration, weight = diff_gjk.func_differentiable_contact(
                    geoms_state, diff_contact_input, gjk_info, i_ga, i_gb, i_b, i_c, ref_penetration
                )
                collider_state.diff_contact_input.ref_penetration[i_b, i_c] = penetration

                func_set_contact(
                    i_ga,
                    i_gb,
                    contact_normal,
                    contact_pos,
                    penetration * weight,
                    i_b,
                    i_c,
                    collider_state.contact_data.pair_idx[i_c, i_b],
                    geoms_state,
                    geoms_info,
                    collider_state,
                    collider_info,
                )

    # Compute other contacts
    for i_c, i_b in qd.ndrange(collider_state.contact_data.pos.shape[0], collider_state.active_buffer.shape[1]):
        if i_c < collider_state.n_contacts[i_b]:
            ref_id = collider_state.diff_contact_input.ref_id[i_b, i_c]
            is_ref = i_c == ref_id
            i_ga = collider_state.diff_contact_input.geom_a[i_b, i_c]
            i_gb = collider_state.diff_contact_input.geom_b[i_b, i_c]

            if not is_ref:
                ref_penetration = collider_state.diff_contact_input.ref_penetration[i_b, ref_id]
                contact_pos, contact_normal, penetration, weight = diff_gjk.func_differentiable_contact(
                    geoms_state, diff_contact_input, gjk_info, i_ga, i_gb, i_b, i_c, ref_penetration
                )

                func_set_contact(
                    i_ga,
                    i_gb,
                    contact_normal,
                    contact_pos,
                    penetration * weight,
                    i_b,
                    i_c,
                    collider_state.contact_data.pair_idx[i_c, i_b],
                    geoms_state,
                    geoms_info,
                    collider_state,
                    collider_info,
                )


@qd.kernel(fastcache=True)
def func_narrow_phase_convex_specializations(
    geoms_state: array_class.GeomsState,
    geoms_info: array_class.GeomsInfo,
    geoms_init_AABB: array_class.GeomsInitAABB,
    verts_info: array_class.VertsInfo,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: qd.template(),
    collider_state: array_class.ColliderState,
    collider_info: array_class.ColliderInfo,
    collider_static_config: qd.template(),
    errno: qd.Tensor,
):
    _B = collider_state.active_buffer.shape[1]
    qd.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
    for i_b in range(_B):
        for i_pair in range(collider_state.n_broad_pairs[i_b]):
            i_ga = collider_state.broad_collision_pairs[i_pair, i_b][0]
            i_gb = collider_state.broad_collision_pairs[i_pair, i_b][1]

            if geoms_info.type[i_ga] > geoms_info.type[i_gb]:
                i_ga, i_gb = i_gb, i_ga

            if geoms_info.type[i_ga] == gs.GEOM_TYPE.PLANE and geoms_info.type[i_gb] == gs.GEOM_TYPE.BOX:
                func_plane_box_contact(
                    i_ga,
                    i_gb,
                    i_b,
                    i_pair,
                    geoms_state,
                    geoms_info,
                    geoms_init_AABB,
                    verts_info,
                    static_rigid_sim_config,
                    collider_state,
                    collider_info,
                    collider_static_config,
                    errno,
                )

            if qd.static(static_rigid_sim_config.box_box_detection):
                if geoms_info.type[i_ga] == gs.GEOM_TYPE.BOX and geoms_info.type[i_gb] == gs.GEOM_TYPE.BOX:
                    func_box_box_contact(
                        i_ga,
                        i_gb,
                        i_b,
                        i_pair,
                        geoms_state,
                        geoms_info,
                        collider_state,
                        collider_info,
                        rigid_global_info,
                        collider_static_config,
                        errno,
                    )


@qd.kernel(fastcache=True)
def func_narrow_phase_any_vs_terrain(
    links_state: array_class.LinksState,
    links_info: array_class.LinksInfo,
    geoms_state: array_class.GeomsState,
    geoms_info: array_class.GeomsInfo,
    geoms_init_AABB: array_class.GeomsInitAABB,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: qd.template(),
    collider_state: array_class.ColliderState,
    collider_info: array_class.ColliderInfo,
    collider_static_config: qd.template(),
    mpr_state: array_class.MPRState,
    mpr_info: array_class.MPRInfo,
    support_field_info: array_class.SupportFieldInfo,
    errno: qd.Tensor,
):
    """
    NOTE: for a single non-batched scene with a lot of collisioin pairs, it will be faster if we also parallelize over `self.n_collision_pairs`. However, parallelize over both B and collisioin_pairs (instead of only over B) leads to significantly slow performance for batched scene. We can treat B=0 and B>0 separately, but we will end up with messier code.
    Therefore, for a big non-batched scene, users are encouraged to simply use `gs.cpu` backend.
    Updated NOTE & TODO: For a HUGE scene with numerous bodies, it's also reasonable to run on GPU. Let's save this for later.
    Update2: Now we use n_broad_pairs instead of n_collision_pairs, so we probably need to think about how to handle non-batched large scene better.
    """
    _B = collider_state.active_buffer.shape[1]
    qd.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
    for i_b in range(_B):
        for i_pair in range(collider_state.n_broad_pairs[i_b]):
            i_ga = collider_state.broad_collision_pairs[i_pair, i_b][0]
            i_gb = collider_state.broad_collision_pairs[i_pair, i_b][1]

            if qd.static(collider_static_config.has_terrain):
                if geoms_info.type[i_ga] == gs.GEOM_TYPE.TERRAIN:
                    i_ga, i_gb = i_gb, i_ga

                if geoms_info.type[i_gb] == gs.GEOM_TYPE.TERRAIN:
                    func_contact_mpr_terrain(
                        i_ga,
                        i_gb,
                        i_b,
                        links_state,
                        links_info,
                        geoms_state,
                        geoms_info,
                        geoms_init_AABB,
                        rigid_global_info,
                        static_rigid_sim_config,
                        collider_state,
                        collider_info,
                        collider_static_config,
                        mpr_state,
                        mpr_info,
                        support_field_info,
                        errno,
                    )


@qd.kernel(fastcache=True)
def func_narrow_phase_nonconvex_vs_nonterrain(
    links_state: array_class.LinksState,
    links_info: array_class.LinksInfo,
    geoms_state: array_class.GeomsState,
    geoms_info: array_class.GeomsInfo,
    geoms_init_AABB: array_class.GeomsInitAABB,
    verts_info: array_class.VertsInfo,
    edges_info: array_class.EdgesInfo,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: qd.template(),
    collider_state: array_class.ColliderState,
    collider_info: array_class.ColliderInfo,
    collider_static_config: qd.template(),
    sdf_info: array_class.SDFInfo,
    errno: qd.Tensor,
):
    EPS = rigid_global_info.EPS[None]

    _B = collider_state.active_buffer.shape[1]
    qd.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
    for i_b in range(_B):
        for i_pair in range(collider_state.n_broad_pairs[i_b]):
            i_ga = collider_state.broad_collision_pairs[i_pair, i_b][0]
            i_gb = collider_state.broad_collision_pairs[i_pair, i_b][1]

            if qd.static(collider_static_config.has_nonconvex_nonterrain):
                if (
                    not (geoms_info.is_convex[i_ga] and geoms_info.is_convex[i_gb])
                    and geoms_info.type[i_gb] != gs.GEOM_TYPE.TERRAIN
                ):
                    # Place the bounded side at i_ga: the polytope helper iterates A's verts and queries B's SDF.
                    # PLANE has infinite extent so it must be i_gb. For non-PLANE pairs the smaller-AABB side goes to
                    # i_ga (its verts are dense relative to the contact patch, while the larger side's scan wastes
                    # cycles on verts far from any contact). Type-ascending order is preserved as a tiebreaker so
                    # smooth primitives (SPHERE/ELLIPSOID/CAPSULE) stay at i_ga, matching the smooth-contact refinement
                    # convention shared with the convex-vs-convex narrowphase.
                    if geoms_info.type[i_ga] > geoms_info.type[i_gb]:
                        i_ga, i_gb = i_gb, i_ga
                    if geoms_info.type[i_ga] == gs.GEOM_TYPE.PLANE:
                        i_ga, i_gb = i_gb, i_ga
                    elif geoms_info.type[i_gb] != gs.GEOM_TYPE.PLANE:
                        diag_a_sq = (geoms_init_AABB[i_ga, 7] - geoms_init_AABB[i_ga, 0]).norm_sqr()
                        diag_b_sq = (geoms_init_AABB[i_gb, 7] - geoms_init_AABB[i_gb, 0]).norm_sqr()
                        if diag_a_sq > diag_b_sq:
                            i_ga, i_gb = i_gb, i_ga

                    tolerance = func_compute_tolerance(
                        i_ga, i_gb, i_b, collider_info.mc_tolerance[None], geoms_info, geoms_init_AABB
                    )

                    # enable_multi_contact controls how many contacts the helper emits per pair (n_max=1 vs
                    # n_contacts_per_convex_pair); the dispatch is unconditional so disabling multi-contact never drops
                    # collisions.
                    ga_pos = geoms_state.pos[i_ga, i_b]
                    ga_quat = geoms_state.quat[i_ga, i_b]
                    gb_pos = geoms_state.pos[i_gb, i_b]
                    gb_quat = geoms_state.quat[i_gb, i_b]
                    func_add_polytope_vertex_contacts_sdf(
                        i_ga,
                        i_gb,
                        i_b,
                        i_pair,
                        ga_pos,
                        ga_quat,
                        gb_pos,
                        gb_quat,
                        tolerance,
                        geoms_state,
                        geoms_info,
                        geoms_init_AABB,
                        verts_info,
                        rigid_global_info,
                        static_rigid_sim_config,
                        collider_static_config,
                        sdf_info,
                        collider_state,
                        collider_info,
                        errno,
                    )
