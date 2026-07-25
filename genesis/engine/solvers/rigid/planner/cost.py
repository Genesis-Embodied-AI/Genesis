import quadrants as qd

import genesis as gs
import genesis.utils.geom as gu
import genesis.utils.linalg as lu
from genesis.utils import array_class

from ..collider.gjk import clear_cache, func_gjk, func_is_sphere_swept_geom
from .kinematics import func_fk, func_ik_jacobian, func_spheres
from .world import func_world_aabb_skip, func_world_sd, func_world_sd_grad

# Validator flag bits (returned per candidate; 0 = certified feasible).
COLLISION = 1
JOINT_LIMIT = 2
GOAL_TOL = 4
GOAL_IN_COLLISION = 8
# Densification factor of the validator relative to the knot spacing.
_VALIDATE_UPSAMPLE = 8
# Local refinement factor for samples failing only through the sweep bound: the sample's covered interval is
# re-certified by this many sub-samples, shrinking the Lipschitz allowance accordingly. Even, so the refined
# sub-intervals split exactly at the sample (the only possible knot boundary inside the interval) and each one
# lies within a single knot segment.
_VALIDATE_REFINE = 16
# Window of margin-free clearance within which a goal contact may be excused: the allowance exists to absorb
# the sphere proxy's conservatism where the goal genuinely intends contact (grasp and place poses), so it only
# covers pairs within proxy-padding distance of contact at the goal. Above the band the pair is physically clear
# and merely violates the requested safety margin, which must stay enforced or the clearance contract would
# silently collapse at user-supplied goals; below the depth the goal is genuinely infeasible and must fail
# validation. The band matches the proxy padding scale; tighter values leave a dead zone of pairs too clear to
# excuse yet too snug for the optimizer and fallback demands, which walls off contact-rich goals. The start
# configuration is exempt from the window: it is the live physical state, so however deep its proxy violations
# run, they are proxy artifacts rather than real penetrations.
_EXCL_DEPTH_MAX = 0.05
_EXCL_CONTACT_BAND = 0.02
# Workspace displacement from the anchor geometry within which a boundary-contact allowance applies at full
# depth; beyond it the demand ramps back to the standard clearance at the Lipschitz rate (one meter of clearance
# per meter of displacement). Displacement is measured exactly - the excused sphere's world motion since its
# anchor (relative motion for self pairs) - so the excused depth is confined to the true approach neighborhood
# of its anchoring boundary: real penetration along a certified path is bounded by the anchor's own physical
# contact plus the plateau, which makes the plateau the certified leak budget. A straight approach keeps a
# positive margin of min(distance-to-anchor, plateau) under this ramp at ANY plateau width, and curved
# approaches that are genuinely clear certify through the exact rescue (see _EXACT_RESCUE_WINDOW), so the
# certification plateau stays at the leak budget. The optimizer enforces the same ramp (cost and gradient) to
# shape terminal approaches into the certified cone, but with a wide plateau: it has no rescue, its cone needs
# no soundness, and a budget-narrow funnel stalls descent. A joint-space travel bound was measured
# order-of-magnitude too conservative here (it walls off every contact-rich goal); the displacement must stay
# exact.
_EXCL_ANCHOR_SLACK_CERT = 0.02
_EXCL_ANCHOR_SLACK_OPT = 0.02
# Proxy deficit window within which a failing (robot sphere, world geom) pair is re-checked exactly
# (certification paths only). The sphere proxy over-covers the mesh by up to a few centimeters, and demanding
# proxy clearance outright walls off every corridor the real geometry clears - random sampling then cannot grow
# trees through them. Against a convex world geom the re-check is the collider's GJK distance on the sphere's
# own collision geom (see func_gjk_clearance) - exact, so a clear pair reads its true clearance and a
# sphere-swept pair its true shallow depth. The covering-sample sweep below remains the fallback for non-convex
# world geoms (grid signed distance fields) and for depth when GJK only reports intersection: the whole surface
# lies within _EXACT_SAMPLE_COV of some sample, so by the 1-Lipschitz world signed distance the true clearance
# is at least the sample minimum less that covering radius; the rescued demand subtracts it, keeping the rescue
# a strict lower bound of the true clearance (raw vertices alone are unsound: an obstacle edge can poke a face
# interior deeper than any of its vertices). Deficits deeper than the window are genuine collisions and skip
# the re-check, which bounds its cost. The sweep is two-level: the coarse covering decides the clear-cut cases
# at a fraction of the samples - its minimum is an upper bound of the fine one (fewer points), so a coarse
# sweep below the win threshold rules the rescue out, and a non-negative coarse covering bound certifies
# positives whose sign no finer sweep could flip - and only the marginal band pays the fine sweep.
_EXACT_RESCUE_WINDOW = 0.08
_EXACT_SAMPLE_COV = 0.005
_EXACT_SAMPLE_COV_COARSE = 0.02
# Real penetration budget of an accepted Cartesian goal hold: with the exact rescue active in the probe, a
# robot-world pair's exact reading is at least the real surface clearance minus _EXACT_SAMPLE_COV and the probe
# headroom, so goal acceptance bounds the hold's true world penetration by this constant alone - it must NOT
# track the anchor plateau, whose width is set by optimizer convergence, or widening the plateau silently
# re-admits really-penetrating goals. It only applies to the exact minimum: self and attached-entity pairs read
# raw proxy conservatism, which the acceptance gate bounds by boundary excusability instead (_EXCL_DEPTH_MAX).
_GOAL_REAL_PEN_MAX = 0.002

# Damped-least-squares budgets for the in-kernel Cartesian-goal solve (one Gauss-Newton solve per restart column):
# more iterations trade compute for reaching tighter poses, the damping trades step aggressiveness for robustness
# near singularities, and the step cap keeps a single update from overshooting. Tuned to the entity IK defaults.
_GOAL_IK_ITERS = 20
# Sub-batches of n_seeds restarts drawn per resolution pass (decouples goal-resolution restart depth from the
# trajectory candidate count): 4 sub-batches x n_seeds gives the restart diversity a rare collision-free goal
# branch needs, without inflating the trajectory-candidate pipeline.
_GOAL_IK_BATCHES = 4
_GOAL_IK_DAMPING = 0.01
_GOAL_IK_POS_TOL = 5e-4
_GOAL_IK_ROT_TOL = 5e-3
_GOAL_IK_MAX_STEP = 0.5


@qd.func
def func_hinge(sd, eps):
    """CHOMP smooth hinge: linear in penetration, quadratic within the activation band, zero beyond.

    Returns the pair (value, derivative with respect to sd).
    """
    cost = gs.qd_float(0.0)
    dcost = gs.qd_float(0.0)
    if sd <= 0.0:
        cost = 0.5 * eps - sd
        dcost = -1.0
    elif sd < eps:
        cost = (eps - sd) ** 2 / (2.0 * eps)
        dcost = (sd - eps) / eps
    return cost, dcost


@qd.func
def func_excl_world_offset(i_s, i_gw, i_bound, i_b, planner_info: array_class.PlannerEntityInfo):
    """Boundary-contact allowance of a (sphere, world geom) pair anchored at boundary i_bound (0 = start, 1 = goal).

    Pairs violating the margin at that boundary may keep their boundary clearance near it, never getting worse than
    the boundary configurations, which is what makes grasp and place goals plannable; everything else gets no
    allowance."""
    offset = gs.qd_float(0.0)
    for i_x in range(planner_info.cert.excl.world_count[i_b]):
        if (
            planner_info.cert.excl.world_pair[i_x, i_b][0] == i_s
            and planner_info.cert.excl.world_pair[i_x, i_b][1] == i_gw
            and planner_info.cert.excl.world_bound[i_x, i_b] == i_bound
        ):
            offset = qd.min(planner_info.cert.excl.world_sd[i_x, i_b], 0.0)
    return offset


@qd.func
def func_excl_self_offset(i_p, i_bound, i_b, planner_info: array_class.PlannerEntityInfo):
    """Boundary-contact allowance of a self-collision sphere pair (see func_excl_world_offset)."""
    offset = gs.qd_float(0.0)
    for i_x in range(planner_info.cert.excl.self_count[i_b]):
        if planner_info.cert.excl.self_pair[i_x, i_b] == i_p and planner_info.cert.excl.self_bound[i_x, i_b] == i_bound:
            offset = qd.min(planner_info.cert.excl.self_sd[i_x, i_b], 0.0)
    return offset


@qd.func
def func_excl_world_anchor(i_s, i_gw, i_bound, i_b, planner_info: array_class.PlannerEntityInfo):
    """Anchor world position and link orientation of an excluded (sphere, world geom) pair.

    Meaningful for pairs whose offset is negative (see _EXCL_ANCHOR_SLACK_CERT).
    """
    anchor = qd.Vector([0.0, 0.0, 0.0], dt=gs.qd_float)
    anchor_quat = qd.Vector([1.0, 0.0, 0.0, 0.0], dt=gs.qd_float)
    for i_x in range(planner_info.cert.excl.world_count[i_b]):
        if (
            planner_info.cert.excl.world_pair[i_x, i_b][0] == i_s
            and planner_info.cert.excl.world_pair[i_x, i_b][1] == i_gw
            and planner_info.cert.excl.world_bound[i_x, i_b] == i_bound
        ):
            anchor = planner_info.cert.excl.world_anchor[i_x, i_b]
            anchor_quat = planner_info.cert.excl.world_anchor_quat[i_x, i_b]
    return anchor, anchor_quat


@qd.func
def func_excl_self_anchor(i_p, i_bound, i_b, planner_info: array_class.PlannerEntityInfo):
    """Anchor relative offset of an excluded self-collision sphere pair (see func_excl_world_anchor)."""
    anchor = qd.Vector([0.0, 0.0, 0.0], dt=gs.qd_float)
    for i_x in range(planner_info.cert.excl.self_count[i_b]):
        if planner_info.cert.excl.self_pair[i_x, i_b] == i_p and planner_info.cert.excl.self_bound[i_x, i_b] == i_bound:
            anchor = planner_info.cert.excl.self_anchor[i_x, i_b]
    return anchor


@qd.func
def func_sphere_exact_sd(
    i_s,
    i_gw,
    i_col,
    i_b,
    sd_stop,
    verts_pos: qd.Tensor,
    verts_start: qd.Tensor,
    links_pos: qd.Tensor,
    links_quat: qd.Tensor,
    planner_world: array_class.PlannerWorldState,
    planner_info: array_class.PlannerEntityInfo,
    dyn_info: array_class.DynInfo,
    sdf_info: array_class.SDFInfo,
):
    """Exact clearance bound of proxy sphere i_s against world geom i_gw at forward-kinematics column i_col.

    The bound is the minimum world signed distance over the sphere's own covering surface samples of one level,
    verts_pos ranged by verts_start (see _EXACT_RESCUE_WINDOW); the caller subtracts the level's covering radius to
    bound the sampled surface. The sweep stops early once the minimum drops below sd_stop, where the rescue can no
    longer win.
    """
    i_l = planner_info.fk.spheres.link_idx[i_s]
    min_sd = gs.qd_float(qd.math.inf)
    i_v = verts_start[i_s]
    while i_v < verts_start[i_s + 1] and min_sd >= sd_stop:
        p = links_pos[i_l, i_col] + gu.qd_transform_by_quat(verts_pos[i_v], links_quat[i_l, i_col])
        min_sd = qd.min(min_sd, func_world_sd(i_gw, i_b, p, planner_world, dyn_info, sdf_info))
        i_v += 1
    return min_sd


@qd.func
def func_gjk_clearance(
    i_ga,
    i_gb,
    i_col,
    pos_a,
    quat_a,
    pos_b,
    quat_b,
    collider_state: array_class.ColliderState,
    gjk_state: array_class.GJKState,
    dyn_info: array_class.DynInfo,
    collider_info: array_class.ColliderInfo,
    rigid_config: qd.template(),
    collider_static_config: qd.template(),
):
    """Exact signed clearance between two convex geoms at explicit poses.

    Runs the collider's Gilbert-Johnson-Keerthi (GJK) distance query on the planner-owned scratch column i_col.
    Sphere-swept geoms (sphere, capsule) shrink to point / line supports, so their shallow penetrations read their
    exact negative depth; any other reading at or below the collision epsilon returns 0 - intersecting, depth
    unresolved (the covering-sample sweep supplies depth there, see _EXACT_RESCUE_WINDOW).
    """
    clear_cache(i_col, gjk_state)
    is_swept_a = func_is_sphere_swept_geom(i_ga, dyn_info)
    is_swept_b = func_is_sphere_swept_geom(i_gb, dyn_info)
    shrink_sphere = is_swept_a or is_swept_b
    dist = func_gjk(
        i_ga,
        i_gb,
        i_col,
        pos_a,
        quat_a,
        pos_b,
        quat_b,
        shrink_sphere,
        collider_state,
        gjk_state,
        dyn_info,
        collider_info,
        rigid_config,
        collider_static_config,
    )
    if dist <= collider_info.gjk.collision_eps[None]:
        dist = 0.0
    elif shrink_sphere:
        if is_swept_a:
            dist -= dyn_info.geoms.data[i_ga][0]
        if is_swept_b:
            dist -= dyn_info.geoms.data[i_gb][0]
    return dist


@qd.func
def func_sphere_radius(i_s, i_b, planner_info: array_class.PlannerEntityInfo, planner_config: qd.template()):
    """Radius of proxy sphere i_s, attached spheres included (inactive attached spheres read radius 0)."""
    radius = gs.qd_float(0.0)
    if i_s < qd.static(planner_config.n_spheres):
        radius = planner_info.fk.spheres.radius[i_s]
    elif planner_info.fk.attach.is_active[i_s - qd.static(planner_config.n_spheres), i_b]:
        radius = planner_info.fk.attach.radius[i_s - qd.static(planner_config.n_spheres)]
    return radius


@qd.func
def func_sphere_link(i_s, planner_info: array_class.PlannerEntityInfo, planner_config: qd.template()):
    """Entity-local link carrying proxy sphere i_s (attach link for attached spheres)."""
    i_l = gs.qd_int(0)
    if i_s < qd.static(planner_config.n_spheres):
        i_l = planner_info.fk.spheres.link_idx[i_s]
    else:
        i_l = planner_info.fk.attach.link_idx[i_s - qd.static(planner_config.n_spheres)]
    return i_l


@qd.func
def func_chain_grad(
    i_l,
    i_c,
    i_w,
    i_col_fk,
    i_b,
    x,
    g,
    scale,
    joints_xanchor: qd.Tensor,
    joints_xaxis: qd.Tensor,
    grad_traj: qd.Tensor,
    planner_info: array_class.PlannerEntityInfo,
    dyn_info: array_class.DynInfo,
    rigid_config: qd.template(),
    planner_config: qd.template(),
):
    """Accumulate a workspace gradient into the joint-space gradient of candidate i_c at knot i_w.

    The gradient g applies at point x on entity-local link i_l and is walked up the parent chain, which assembles the
    J^T v product without materializing J, the same way the entity jacobian does. Locked DOFs are skipped.
    """
    link_offset = qd.static(planner_config.link_offset)
    q_offset = qd.static(planner_config.q_offset)
    i_l_glob = i_l + link_offset
    while i_l_glob != -1:
        I_l = [gs.qd_int(i_l_glob), i_b] if qd.static(rigid_config.batch_links_info) else gs.qd_int(i_l_glob)
        if dyn_info.links.parent_idx[I_l] == -1 and dyn_info.links.is_fixed[I_l]:
            i_l_glob = -1
        else:
            for i_j_ in range(dyn_info.links.joint_start[I_l], dyn_info.links.joint_end[I_l]):
                i_j = gs.qd_int(i_j_)
                I_j = [i_j, i_b] if qd.static(rigid_config.batch_joints_info) else i_j
                joint_type = dyn_info.joints.type[I_j]
                if joint_type != gs.JOINT_TYPE.FIXED:
                    i_dp = dyn_info.joints.q_start[I_j] - q_offset
                    if not planner_info.fk.dofs.is_locked[i_dp, i_b]:
                        dq = gs.qd_float(0.0)
                        if joint_type == gs.JOINT_TYPE.REVOLUTE:
                            dq = (
                                joints_xaxis[i_j - qd.static(planner_config.joint_offset), i_col_fk]
                                .cross(x - joints_xanchor[i_j - qd.static(planner_config.joint_offset), i_col_fk])
                                .dot(g)
                            )
                        else:
                            dq = joints_xaxis[i_j - qd.static(planner_config.joint_offset), i_col_fk].dot(g)
                        grad_traj[i_dp, i_c, i_w] += scale * dq
            i_l_glob = dyn_info.links.parent_idx[I_l]


@qd.func
def func_collision_cost(
    i_col,
    i_b,
    swp,
    dq_inf,
    links_pos: qd.Tensor,
    links_quat: qd.Tensor,
    spheres_pos: qd.Tensor,
    planner_world: array_class.PlannerWorldState,
    collider_state: array_class.ColliderState,
    gjk_state: array_class.GJKState,
    planner_info: array_class.PlannerEntityInfo,
    dyn_info: array_class.DynInfo,
    collider_info: array_class.ColliderInfo,
    sdf_info: array_class.SDFInfo,
    rigid_config: qd.template(),
    collider_static_config: qd.template(),
    planner_config: qd.template(),
    use_exact: qd.template(),
):
    """Collision cost of one forward-kinematics column, over the world, self and attach hinges.

    The world hinge covers every (sphere, obstacle geom) pair within band, the other two the pair lists. The sweep
    allowance swp inflates every sphere so the cost covers the
    whole inter-sample segment (SDF is 1-Lipschitz); boundary-contact allowances ramp away from their anchor
    (see _EXCL_ANCHOR_SLACK_CERT). Certification callers pass use_exact=True so borderline pairs are re-checked
    exactly - robot world pairs and failing self link pairs through the collider's GJK, with the covering-sample
    sweep as the world fallback (see _EXACT_RESCUE_WINDOW). Returns (cost, min_sd_exact, min_sd_proxy): the exact
    minimum runs over robot-sphere world pairs, whose rescued reading is tied to the real surface clearance (see
    _EXACT_SAMPLE_COV); the proxy minimum runs over self and attached-entity pairs, whose reading carries the raw
    proxy conservatism except where the self rescue certifies an exact reading.
    """
    n_sph_tot = qd.static(planner_config.n_spheres + planner_config.n_attach_max)
    anchor_slack = (
        planner_info.cert.excl_anchor_slack_cert[None] if use_exact else planner_info.cert.excl_anchor_slack_opt[None]
    )
    cost = gs.qd_float(0.0)
    min_sd_exact = gs.qd_float(qd.math.inf)
    min_sd_proxy = gs.qd_float(qd.math.inf)

    for i_s in range(n_sph_tot):
        radius = func_sphere_radius(i_s, i_b, planner_info, planner_config)
        if radius > 0.0:
            x = spheres_pos[i_s, i_col]
            band = radius + planner_info.cost.d_safe[None] + planner_info.cost.eps_act[None] + swp
            for i_gw in range(planner_world.n_geoms[None]):
                if planner_world.geoms_is_active[i_gw, i_b] and not func_world_aabb_skip(
                    i_gw, i_b, x, band, planner_world
                ):
                    eps_act = qd.min(planner_info.cost.eps_act[None], planner_world.geoms_max_band[i_gw])
                    sd = func_world_sd(i_gw, i_b, x, planner_world, dyn_info, sdf_info)
                    sd_eff = sd - radius - planner_info.cost.d_safe[None] - swp
                    for i_bound in range(2):
                        offset = func_excl_world_offset(i_s, i_gw, i_bound, i_b, planner_info)
                        if offset < 0.0:
                            # Excluded pair: allowed to keep its anchoring boundary's clearance near it, ramped
                            # away from the anchor (see _EXCL_ANCHOR_SLACK_CERT); sweep-free (the allowance is exact
                            # at the boundary sample, and sweep-inflating it would flag pairs pinned at constant
                            # depth).
                            anchor, anchor_quat = func_excl_world_anchor(i_s, i_gw, i_bound, i_b, planner_info)
                            drift = (x - anchor).norm()
                            if qd.static(use_exact):
                                # Certification charges the link rotation too: the covered surface patch moves
                                # with the link frame, so a matching sphere position under a rotated link is a
                                # different contact (points within the sphere move at most radius * angle).
                                i_l_s = func_sphere_link(i_s, planner_info, planner_config)
                                quat_dot = qd.abs(links_quat[i_l_s, i_col].dot(anchor_quat))
                                drift += 2.0 * qd.acos(qd.min(quat_dot, 1.0)) * radius
                            ramp = qd.max(0.0, drift - anchor_slack)
                            sd_eff = qd.max(sd_eff, sd - radius - planner_info.cost.d_safe[None] - offset - ramp + 1e-4)
                    if qd.static(use_exact):
                        if (
                            sd_eff < 0.0
                            and sd_eff > -planner_info.cert.exact_rescue_window[None]
                            and i_s < qd.static(planner_config.n_spheres)
                        ):
                            # Borderline robot pair: the exact clearance replaces the proxy one (it is never
                            # smaller), which admits corridors the proxy conservatism walls off. Against a
                            # convex world geom the collider's GJK reads it directly on the sphere's own
                            # collision geom; a zero reading (intersecting, depth unresolved) and non-convex
                            # world geoms fall through to the covering-sample sweep.
                            is_exact_resolved = False
                            if planner_world.geoms_is_convex[i_gw]:
                                i_gr = planner_info.fk.spheres.geom_idx[i_s]
                                i_l_s = planner_info.fk.spheres.link_idx[i_s]
                                geom_pos = links_pos[i_l_s, i_col] + gu.qd_transform_by_quat(
                                    planner_info.fk.geoms.offset_pos[i_gr], links_quat[i_l_s, i_col]
                                )
                                geom_quat = gu.qd_transform_quat_by_quat(
                                    planner_info.fk.geoms.offset_quat[i_gr], links_quat[i_l_s, i_col]
                                )
                                d_exact = func_gjk_clearance(
                                    planner_info.fk.geoms.geoms_idx[i_gr],
                                    planner_world.geoms_idx[i_gw],
                                    i_col,
                                    geom_pos,
                                    geom_quat,
                                    planner_world.geoms_pos[i_gw, i_b],
                                    planner_world.geoms_quat[i_gw, i_b],
                                    collider_state=collider_state,
                                    gjk_state=gjk_state,
                                    dyn_info=dyn_info,
                                    collider_info=collider_info,
                                    rigid_config=rigid_config,
                                    collider_static_config=collider_static_config,
                                )
                                if d_exact != 0.0:
                                    sd_eff = qd.max(sd_eff, d_exact - planner_info.cost.d_safe[None] - swp)
                                    is_exact_resolved = True
                            if not is_exact_resolved:
                                # Two-level covering sweep: the coarse screening level settles the clear-cut
                                # cases; only its marginal band pays the fine sweep (see _EXACT_RESCUE_WINDOW).
                                sd_stop = (
                                    sd_eff
                                    + planner_info.cert.exact_sample_cov[None]
                                    + planner_info.cost.d_safe[None]
                                    + swp
                                )
                                sd_coarse = func_sphere_exact_sd(
                                    i_s,
                                    i_gw,
                                    i_col,
                                    i_b,
                                    sd_stop,
                                    verts_pos=planner_info.fk.verts.coarse_pos_local,
                                    verts_start=planner_info.fk.verts.spheres_coarse_start,
                                    links_pos=links_pos,
                                    links_quat=links_quat,
                                    planner_world=planner_world,
                                    planner_info=planner_info,
                                    dyn_info=dyn_info,
                                    sdf_info=sdf_info,
                                )
                                if sd_coarse >= sd_stop:
                                    bound_coarse = (
                                        sd_coarse
                                        - planner_info.cert.exact_sample_cov_coarse[None]
                                        - planner_info.cost.d_safe[None]
                                        - swp
                                    )
                                    if bound_coarse >= 0.0:
                                        sd_eff = qd.max(sd_eff, bound_coarse)
                                    else:
                                        sd_fine = func_sphere_exact_sd(
                                            i_s,
                                            i_gw,
                                            i_col,
                                            i_b,
                                            sd_stop,
                                            verts_pos=planner_info.fk.verts.pos_local,
                                            verts_start=planner_info.fk.verts.spheres_start,
                                            links_pos=links_pos,
                                            links_quat=links_quat,
                                            planner_world=planner_world,
                                            planner_info=planner_info,
                                            dyn_info=dyn_info,
                                            sdf_info=sdf_info,
                                        )
                                        sd_eff = qd.max(
                                            sd_eff,
                                            sd_fine
                                            - planner_info.cert.exact_sample_cov[None]
                                            - planner_info.cost.d_safe[None]
                                            - swp,
                                        )
                    if i_s < qd.static(planner_config.n_spheres):
                        min_sd_exact = qd.min(min_sd_exact, sd_eff)
                    else:
                        min_sd_proxy = qd.min(min_sd_proxy, sd_eff)
                    hinge, _ = func_hinge(sd_eff, eps_act)
                    cost += planner_info.cost.w_obs[None] * hinge

    for i_lp in range(planner_info.fk.self_pairs.link_pairs_idx.shape[0]):
        sd_pair = gs.qd_float(qd.math.inf)
        swp_pair_max = gs.qd_float(0.0)
        for i_p in range(
            planner_info.fk.self_pairs.link_pairs_start[i_lp], planner_info.fk.self_pairs.link_pairs_start[i_lp + 1]
        ):
            i_sa, i_sb = planner_info.fk.self_pairs.spheres_idx[i_p][0], planner_info.fk.self_pairs.spheres_idx[i_p][1]
            radius_a = func_sphere_radius(i_sa, i_b, planner_info, planner_config)
            radius_b = func_sphere_radius(i_sb, i_b, planner_info, planner_config)
            if radius_a > 0.0 and radius_b > 0.0:
                delta = spheres_pos[i_sa, i_col] - spheres_pos[i_sb, i_col]
                dist = delta.norm()
                swp_pair = qd.min(0.5 * planner_info.fk.self_pairs.reach[i_p] * dq_inf, planner_info.cost.eps_act[None])
                sd_eff = dist - radius_a - radius_b - planner_info.cost.d_safe[None] - swp_pair
                for i_bound in range(2):
                    offset = func_excl_self_offset(i_p, i_bound, i_b, planner_info)
                    if offset < 0.0:
                        # See the world-pair exclusion above.
                        anchor = func_excl_self_anchor(i_p, i_bound, i_b, planner_info)
                        ramp = qd.max(0.0, (delta - anchor).norm() - anchor_slack)
                        sd_eff = qd.max(
                            sd_eff, dist - radius_a - radius_b - planner_info.cost.d_safe[None] - offset - ramp + 1e-4
                        )
                sd_pair = qd.min(sd_pair, sd_eff)
                swp_pair_max = qd.max(swp_pair_max, swp_pair)
                hinge, _ = func_hinge(sd_eff, planner_info.cost.eps_self[None])
                cost += planner_info.cost.w_self[None] * hinge
        if qd.static(use_exact):
            if sd_pair < 0.0:
                # Failing self link pair: one exact GJK reading per geom pair replaces the proxy one when every
                # geom pair resolves (bounding-sphere clearance settles the far pairs query-free). A zero
                # reading leaves depth unresolved, so the proxy reading stands - self pairs have no sample
                # sweep, and unresolved intersections must keep their conservative depth for the boundary
                # allowances to stay sound.
                i_la = planner_info.fk.self_pairs.link_pairs_idx[i_lp][0]
                i_lb = planner_info.fk.self_pairs.link_pairs_idx[i_lp][1]
                d_exact_pair = gs.qd_float(qd.math.inf)
                is_unresolved = False
                for i_ga in range(planner_info.fk.geoms.links_start[i_la], planner_info.fk.geoms.links_start[i_la + 1]):
                    pos_ga = links_pos[i_la, i_col] + gu.qd_transform_by_quat(
                        planner_info.fk.geoms.offset_pos[i_ga], links_quat[i_la, i_col]
                    )
                    quat_ga = gu.qd_transform_quat_by_quat(
                        planner_info.fk.geoms.offset_quat[i_ga], links_quat[i_la, i_col]
                    )
                    center_a = links_pos[i_la, i_col] + gu.qd_transform_by_quat(
                        planner_info.fk.geoms.bound_center_local[i_ga], links_quat[i_la, i_col]
                    )
                    for i_gb in range(
                        planner_info.fk.geoms.links_start[i_lb], planner_info.fk.geoms.links_start[i_lb + 1]
                    ):
                        center_b = links_pos[i_lb, i_col] + gu.qd_transform_by_quat(
                            planner_info.fk.geoms.bound_center_local[i_gb], links_quat[i_lb, i_col]
                        )
                        bound = (
                            (center_a - center_b).norm()
                            - planner_info.fk.geoms.bound_radius[i_ga]
                            - planner_info.fk.geoms.bound_radius[i_gb]
                        )
                        if bound >= planner_info.cost.d_safe[None] + swp_pair_max:
                            d_exact_pair = qd.min(d_exact_pair, bound)
                        else:
                            pos_gb = links_pos[i_lb, i_col] + gu.qd_transform_by_quat(
                                planner_info.fk.geoms.offset_pos[i_gb], links_quat[i_lb, i_col]
                            )
                            quat_gb = gu.qd_transform_quat_by_quat(
                                planner_info.fk.geoms.offset_quat[i_gb], links_quat[i_lb, i_col]
                            )
                            d_exact = func_gjk_clearance(
                                planner_info.fk.geoms.geoms_idx[i_ga],
                                planner_info.fk.geoms.geoms_idx[i_gb],
                                i_col,
                                pos_ga,
                                quat_ga,
                                pos_gb,
                                quat_gb,
                                collider_state=collider_state,
                                gjk_state=gjk_state,
                                dyn_info=dyn_info,
                                collider_info=collider_info,
                                rigid_config=rigid_config,
                                collider_static_config=collider_static_config,
                            )
                            if d_exact == 0.0:
                                is_unresolved = True
                            d_exact_pair = qd.min(d_exact_pair, d_exact)
                if not is_unresolved:
                    sd_pair = qd.max(sd_pair, d_exact_pair - planner_info.cost.d_safe[None] - swp_pair_max)
        min_sd_proxy = qd.min(min_sd_proxy, sd_pair)

    return cost, min_sd_exact, min_sd_proxy


@qd.func
def func_pose_cost(
    i_col,
    i_b,
    links_pos: qd.Tensor,
    links_quat: qd.Tensor,
    planner_info: array_class.PlannerEntityInfo,
    rigid_info: array_class.RigidInfo,
    planner_config: qd.template(),
):
    """Terminal pose cost (position + rotation-vector geodesic) of the goal link for Cartesian goals."""
    i_l = planner_info.cost.boundary.goal_link_idx[None] - qd.static(planner_config.link_offset)
    err_pos = links_pos[i_l, i_col] - planner_info.cost.boundary.goal_pos[i_b]
    quat_rel = gu.qd_transform_quat_by_quat(
        links_quat[i_l, i_col], gu.qd_inv_quat(planner_info.cost.boundary.goal_quat[i_b])
    )
    err_rot = gu.qd_quat_to_rotvec(quat_rel, rigid_info.EPS[None])
    return (
        planner_info.cost.w_pose_pos[None] * err_pos.norm_sqr()
        + planner_info.cost.w_pose_rot[None] * err_rot.norm_sqr()
    )


@qd.func
def func_merge_boundary_exclusions(
    i_b_,
    i_b,
    i_bound,
    contact_band,
    depth_max,
    planner_state: array_class.PlannerState,
    planner_world: array_class.PlannerWorldState,
    planner_info: array_class.PlannerEntityInfo,
    dyn_info: array_class.DynInfo,
    sdf_info: array_class.SDFInfo,
    planner_config: qd.template(),
):
    """Merge the margin violations of the boundary configuration held in the eval scratch into the exclusion lists.

    A pair already listed keeps its most negative clearance, a new pair is appended. Only pairs whose margin-free
    clearance lies in (-depth_max, contact_band) are excused (see _EXCL_DEPTH_MAX). Returns nonzero when a list
    overflows.
    """
    n_sph_tot = qd.static(planner_config.n_spheres + planner_config.n_attach_max)
    n_excl_max = planner_info.cert.excl.world_pair.shape[0]
    is_overflow = gs.qd_int(0)

    for i_s in range(n_sph_tot):
        radius = func_sphere_radius(i_s, i_b, planner_info, planner_config)
        if radius > 0.0:
            x = planner_state.fk.eval.spheres_pos[i_s, i_b_]
            band = radius + planner_info.cost.d_safe[None]
            for i_gw in range(planner_world.n_geoms[None]):
                if planner_world.geoms_is_active[i_gw, i_b] and not func_world_aabb_skip(
                    i_gw, i_b, x, band, planner_world
                ):
                    sd = func_world_sd(i_gw, i_b, x, planner_world, dyn_info, sdf_info)
                    raw = sd - radius
                    sd_eff = raw - planner_info.cost.d_safe[None]
                    if sd_eff < 0.0 and raw < contact_band and raw > -depth_max:
                        link_quat = planner_state.fk.eval.links_quat[
                            func_sphere_link(i_s, planner_info, planner_config), i_b_
                        ]
                        n_world = planner_info.cert.excl.world_count[i_b]
                        is_listed = False
                        for i_x in range(n_world):
                            if (
                                planner_info.cert.excl.world_pair[i_x, i_b][0] == i_s
                                and planner_info.cert.excl.world_pair[i_x, i_b][1] == i_gw
                                and planner_info.cert.excl.world_bound[i_x, i_b] == i_bound
                            ):
                                if sd_eff < planner_info.cert.excl.world_sd[i_x, i_b]:
                                    planner_info.cert.excl.world_sd[i_x, i_b] = sd_eff
                                    planner_info.cert.excl.world_anchor[i_x, i_b] = x
                                    planner_info.cert.excl.world_anchor_quat[i_x, i_b] = link_quat
                                is_listed = True
                        if not is_listed:
                            if n_world < n_excl_max:
                                planner_info.cert.excl.world_pair[n_world, i_b] = qd.Vector([i_s, i_gw], dt=gs.qd_int)
                                planner_info.cert.excl.world_sd[n_world, i_b] = sd_eff
                                planner_info.cert.excl.world_bound[n_world, i_b] = i_bound
                                planner_info.cert.excl.world_anchor[n_world, i_b] = x
                                planner_info.cert.excl.world_anchor_quat[n_world, i_b] = link_quat
                                planner_info.cert.excl.world_count[i_b] = n_world + 1
                            else:
                                is_overflow = 1

    for i_p in range(planner_info.fk.self_pairs.spheres_idx.shape[0]):
        i_sa, i_sb = planner_info.fk.self_pairs.spheres_idx[i_p][0], planner_info.fk.self_pairs.spheres_idx[i_p][1]
        radius_a = func_sphere_radius(i_sa, i_b, planner_info, planner_config)
        radius_b = func_sphere_radius(i_sb, i_b, planner_info, planner_config)
        if radius_a > 0.0 and radius_b > 0.0:
            delta = planner_state.fk.eval.spheres_pos[i_sa, i_b_] - planner_state.fk.eval.spheres_pos[i_sb, i_b_]
            raw = delta.norm() - (radius_a + radius_b)
            sd_eff = raw - planner_info.cost.d_safe[None]
            if sd_eff < 0.0 and raw < contact_band and raw > -depth_max:
                n_self = planner_info.cert.excl.self_count[i_b]
                is_listed = False
                for i_x in range(n_self):
                    if (
                        planner_info.cert.excl.self_pair[i_x, i_b] == i_p
                        and planner_info.cert.excl.self_bound[i_x, i_b] == i_bound
                    ):
                        if sd_eff < planner_info.cert.excl.self_sd[i_x, i_b]:
                            planner_info.cert.excl.self_sd[i_x, i_b] = sd_eff
                            planner_info.cert.excl.self_anchor[i_x, i_b] = delta
                        is_listed = True
                if not is_listed:
                    if n_self < n_excl_max:
                        planner_info.cert.excl.self_pair[n_self, i_b] = i_p
                        planner_info.cert.excl.self_sd[n_self, i_b] = sd_eff
                        planner_info.cert.excl.self_bound[n_self, i_b] = i_bound
                        planner_info.cert.excl.self_anchor[n_self, i_b] = delta
                        planner_info.cert.excl.self_count[i_b] = n_self + 1
                    else:
                        is_overflow = 1

    return is_overflow


@qd.func
def func_boundary_exclusions(
    envs_idx: qd.types.ndarray(),
    planner_state: array_class.PlannerState,
    planner_world: array_class.PlannerWorldState,
    dyn_state: array_class.DynState,
    collider_state: array_class.ColliderState,
    gjk_state: array_class.GJKState,
    planner_info: array_class.PlannerEntityInfo,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    collider_info: array_class.ColliderInfo,
    sdf_info: array_class.SDFInfo,
    rigid_config: qd.template(),
    collider_static_config: qd.template(),
    planner_config: qd.template(),
    include_goal: qd.template(),
    errno: qd.Tensor,
):
    """Collect the contact exclusions of every env's boundary configurations, serial per env as the lists are tiny.

    Always covers the start, plus the goal when include_goal is set: a Cartesian goal is unknown until resolved, so
    its pass runs on a second call.
    """
    qd.loop_config(serialize=qd.static(planner_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_b_ in range(envs_idx.shape[0]):
        i_b = envs_idx[i_b_]
        planner_info.cert.excl.world_count[i_b] = 0
        planner_info.cert.excl.self_count[i_b] = 0
        is_overflow = gs.qd_int(0)
        for i_bound in range(qd.static(2 if include_goal else 1)):
            # FK of the boundary configuration through the eval scratch, one column per env.
            for i_dp in range(qd.static(planner_config.n_dp)):
                if i_bound == 0:
                    planner_state.fk.eval.qpos[i_dp, i_b_] = planner_info.cost.boundary.qpos_start[i_dp, i_b]
                else:
                    planner_state.fk.eval.qpos[i_dp, i_b_] = planner_info.cost.boundary.qpos_goal[i_dp, i_b]
            func_fk(
                i_b_,
                i_b_,
                i_b,
                qpos=planner_state.fk.eval.qpos,
                links_pos=planner_state.fk.eval.links_pos,
                links_quat=planner_state.fk.eval.links_quat,
                joints_xanchor=planner_state.fk.eval.joints_xanchor,
                joints_xaxis=planner_state.fk.eval.joints_xaxis,
                dyn_state=dyn_state,
                dyn_info=dyn_info,
                rigid_info=rigid_info,
                rigid_config=rigid_config,
                planner_config=planner_config,
            )
            func_spheres(
                i_b_,
                i_b,
                links_pos=planner_state.fk.eval.links_pos,
                links_quat=planner_state.fk.eval.links_quat,
                spheres_pos=planner_state.fk.eval.spheres_pos,
                planner_info=planner_info,
                planner_config=planner_config,
            )
            contact_band = qd.math.inf if i_bound == 0 else planner_info.cert.excl_contact_band[None]
            depth_max = qd.math.inf if i_bound == 0 else planner_info.cert.excl_depth_max[None]
            is_overflow |= func_merge_boundary_exclusions(
                i_b_,
                i_b,
                i_bound,
                contact_band,
                depth_max,
                planner_state,
                planner_world,
                planner_info,
                dyn_info,
                sdf_info,
                planner_config,
            )
        if is_overflow != 0:
            errno[i_b] = errno[i_b] | array_class.ErrorCode.OVERFLOW_PLANNER_EXCLUSIONS


@qd.kernel
def kernel_boundary_exclusions(
    envs_idx: qd.types.ndarray(),
    planner_state: array_class.PlannerState,
    planner_world: array_class.PlannerWorldState,
    dyn_state: array_class.DynState,
    collider_state: array_class.ColliderState,
    gjk_state: array_class.GJKState,
    planner_info: array_class.PlannerEntityInfo,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    collider_info: array_class.ColliderInfo,
    sdf_info: array_class.SDFInfo,
    rigid_config: qd.template(),
    collider_static_config: qd.template(),
    planner_config: qd.template(),
    include_goal: qd.template(),
    errno: qd.Tensor,
):
    func_boundary_exclusions(
        envs_idx,
        planner_state,
        planner_world,
        dyn_state,
        collider_state,
        gjk_state,
        planner_info,
        dyn_info,
        rigid_info,
        collider_info,
        sdf_info,
        rigid_config,
        collider_static_config,
        planner_config,
        include_goal=include_goal,
        errno=errno,
    )


@qd.func
def func_knot_cost_grad(
    i_c,
    i_w,
    i_cw,
    i_b,
    planner_state: array_class.PlannerState,
    planner_world: array_class.PlannerWorldState,
    planner_info: array_class.PlannerEntityInfo,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    sdf_info: array_class.SDFInfo,
    rigid_config: qd.template(),
    collider_static_config: qd.template(),
    planner_config: qd.template(),
):
    """Full cost and joint-space gradient of one (candidate, knot) column, from the forward-kinematics cache.

    Writes only its own cost_wp / grad_traj slices - no atomics, deterministic per backend. The sweep allowance is
    stop-gradient: it gates where the hinge activates without contributing cross-column gradient terms.
    """
    n_knots = qd.static(planner_config.n_knots)
    n_dp = qd.static(planner_config.n_dp)
    n_sph_tot = qd.static(planner_config.n_spheres + planner_config.n_attach_max)

    cost = gs.qd_float(0.0)
    for i_dp in range(n_dp):
        planner_state.cost.grad[i_dp, i_c, i_w] = 0.0

    # Sweep allowance: half the largest neighbor-knot travel of any sphere. Stop-gradient by design, so
    # its influence (and the induced gradient error) is clamped to the activation band; the validator
    # applies the full unclamped sweep, which is where rigor matters.
    swp = gs.qd_float(0.0)
    for i_s in range(n_sph_tot):
        if i_w > 0:
            swp = qd.max(
                swp,
                0.5 * (planner_state.fk.spheres_pos[i_s, i_cw] - planner_state.fk.spheres_pos[i_s, i_cw - 1]).norm(),
            )
        if i_w < n_knots - 1:
            swp = qd.max(
                swp,
                0.5 * (planner_state.fk.spheres_pos[i_s, i_cw + 1] - planner_state.fk.spheres_pos[i_s, i_cw]).norm(),
            )
    swp = qd.min(swp, planner_info.cost.eps_act[None])

    # L-inf joint travel toward the neighbor knots, for the per-pair self-collision sweep bound.
    dq_inf = gs.qd_float(0.0)
    for i_dp in range(n_dp):
        if i_w > 0:
            dq_inf = qd.max(
                dq_inf, qd.abs(planner_state.cost.qpos[i_dp, i_cw] - planner_state.cost.qpos[i_dp, i_cw - 1])
            )
        if i_w < n_knots - 1:
            dq_inf = qd.max(
                dq_inf, qd.abs(planner_state.cost.qpos[i_dp, i_cw + 1] - planner_state.cost.qpos[i_dp, i_cw])
            )

    # World collision, with gradient through the chain walk.
    for i_s in range(n_sph_tot):
        radius = func_sphere_radius(i_s, i_b, planner_info, planner_config)
        if radius > 0.0:
            x = planner_state.fk.spheres_pos[i_s, i_cw]
            band = radius + planner_info.cost.d_safe[None] + planner_info.cost.eps_act[None] + swp
            for i_gw in range(planner_world.n_geoms[None]):
                if planner_world.geoms_is_active[i_gw, i_b] and not func_world_aabb_skip(
                    i_gw, i_b, x, band, planner_world
                ):
                    eps_act = qd.min(planner_info.cost.eps_act[None], planner_world.geoms_max_band[i_gw])
                    sd = func_world_sd(i_gw, i_b, x, planner_world, dyn_info, sdf_info)
                    sd_eff = sd - radius - planner_info.cost.d_safe[None] - swp
                    ramp_dir = qd.Vector([0.0, 0.0, 0.0], dt=gs.qd_float)
                    for i_bound in range(2):
                        offset = func_excl_world_offset(i_s, i_gw, i_bound, i_b, planner_info)
                        if offset < 0.0:
                            # See the exclusion comment in func_collision_cost.
                            anchor, _ = func_excl_world_anchor(i_s, i_gw, i_bound, i_b, planner_info)
                            drift = x - anchor
                            drift_norm = drift.norm()
                            ramp = qd.max(0.0, drift_norm - planner_info.cert.excl_anchor_slack_opt[None])
                            sd_excused = sd - radius - planner_info.cost.d_safe[None] - offset - ramp + 1e-4
                            if sd_excused > sd_eff:
                                sd_eff = sd_excused
                                # The winning excused branch adds the ramp's own gradient - radially away from
                                # the anchor - to the signed-distance normal.
                                ramp_dir = qd.Vector([0.0, 0.0, 0.0], dt=gs.qd_float)
                                if ramp > 0.0:
                                    ramp_dir = drift / drift_norm
                    hinge, dhinge = func_hinge(sd_eff, eps_act)
                    if hinge > 0.0:
                        cost += planner_info.cost.w_obs[None] * hinge
                        normal = func_world_sd_grad(
                            i_gw, i_b, x, planner_world, dyn_info, rigid_info, sdf_info, collider_static_config
                        )
                        func_chain_grad(
                            func_sphere_link(i_s, planner_info, planner_config),
                            i_c,
                            i_w,
                            i_cw,
                            i_b,
                            x,
                            normal - ramp_dir,
                            planner_info.cost.w_obs[None] * dhinge,
                            joints_xanchor=planner_state.fk.joints_xanchor,
                            joints_xaxis=planner_state.fk.joints_xaxis,
                            grad_traj=planner_state.cost.grad,
                            planner_info=planner_info,
                            dyn_info=dyn_info,
                            rigid_config=rigid_config,
                            planner_config=planner_config,
                        )

    # Self / attach collision, gradient through both chains.
    for i_p in range(planner_info.fk.self_pairs.spheres_idx.shape[0]):
        i_sa, i_sb = planner_info.fk.self_pairs.spheres_idx[i_p][0], planner_info.fk.self_pairs.spheres_idx[i_p][1]
        radius_a = func_sphere_radius(i_sa, i_b, planner_info, planner_config)
        radius_b = func_sphere_radius(i_sb, i_b, planner_info, planner_config)
        if radius_a > 0.0 and radius_b > 0.0:
            delta = planner_state.fk.spheres_pos[i_sa, i_cw] - planner_state.fk.spheres_pos[i_sb, i_cw]
            dist = delta.norm()
            swp_pair = qd.min(0.5 * planner_info.fk.self_pairs.reach[i_p] * dq_inf, planner_info.cost.eps_act[None])
            sd_eff = dist - radius_a - radius_b - planner_info.cost.d_safe[None] - swp_pair
            ramp_dir = qd.Vector([0.0, 0.0, 0.0], dt=gs.qd_float)
            for i_bound in range(2):
                offset = func_excl_self_offset(i_p, i_bound, i_b, planner_info)
                if offset < 0.0:
                    # See the exclusion comment in func_collision_cost.
                    anchor = func_excl_self_anchor(i_p, i_bound, i_b, planner_info)
                    drift = delta - anchor
                    drift_norm = drift.norm()
                    ramp = qd.max(0.0, drift_norm - planner_info.cert.excl_anchor_slack_opt[None])
                    sd_excused = dist - radius_a - radius_b - planner_info.cost.d_safe[None] - offset - ramp + 1e-4
                    if sd_excused > sd_eff:
                        sd_eff = sd_excused
                        # See the world-pair ramp gradient above; here the ramp acts on the relative offset.
                        ramp_dir = qd.Vector([0.0, 0.0, 0.0], dt=gs.qd_float)
                        if ramp > 0.0:
                            ramp_dir = drift / drift_norm
            hinge, dhinge = func_hinge(sd_eff, planner_info.cost.eps_self[None])
            if hinge > 0.0 and dist > rigid_info.EPS[None]:
                cost += planner_info.cost.w_self[None] * hinge
                normal = delta / dist
                func_chain_grad(
                    func_sphere_link(i_sa, planner_info, planner_config),
                    i_c,
                    i_w,
                    i_cw,
                    i_b,
                    planner_state.fk.spheres_pos[i_sa, i_cw],
                    normal - ramp_dir,
                    planner_info.cost.w_self[None] * dhinge,
                    joints_xanchor=planner_state.fk.joints_xanchor,
                    joints_xaxis=planner_state.fk.joints_xaxis,
                    grad_traj=planner_state.cost.grad,
                    planner_info=planner_info,
                    dyn_info=dyn_info,
                    rigid_config=rigid_config,
                    planner_config=planner_config,
                )
                func_chain_grad(
                    func_sphere_link(i_sb, planner_info, planner_config),
                    i_c,
                    i_w,
                    i_cw,
                    i_b,
                    planner_state.fk.spheres_pos[i_sb, i_cw],
                    ramp_dir - normal,
                    planner_info.cost.w_self[None] * dhinge,
                    joints_xanchor=planner_state.fk.joints_xanchor,
                    joints_xaxis=planner_state.fk.joints_xaxis,
                    grad_traj=planner_state.cost.grad,
                    planner_info=planner_info,
                    dyn_info=dyn_info,
                    rigid_config=rigid_config,
                    planner_config=planner_config,
                )

    # Smoothness (acceleration + jerk finite-difference quadratics) and joint-limit hinge + posture
    # regularizer, all local stencils on qpos_traj. Column i_cw of candidate i_c spans knots
    # [i_c*n_knots, (i_c+1)*n_knots).
    i_w0 = i_c * n_knots
    for i_dp in range(n_dp):
        if not planner_info.fk.dofs.is_locked[i_dp, i_b]:
            q = planner_state.cost.qpos[i_dp, i_cw]
            # Acceleration stencil rows this knot participates in: cost = w_a * sum_w a_w^2 with
            # a_w = q_{w-1} - 2 q_w + q_{w+1}; d cost / d q_this = 2 w_a * sum_w a_w * c_w.
            for dw in qd.static(range(-1, 2)):
                i_wc = i_w + dw
                if 1 <= i_wc < n_knots - 1:
                    acc = (
                        planner_state.cost.qpos[i_dp, i_w0 + i_wc - 1]
                        - 2.0 * planner_state.cost.qpos[i_dp, i_w0 + i_wc]
                        + planner_state.cost.qpos[i_dp, i_w0 + i_wc + 1]
                    )
                    coeff = gs.qd_float(-2.0 if dw == 0 else 1.0)
                    if dw == 0:
                        cost += planner_info.cost.w_acc[None] * acc**2
                    planner_state.cost.grad[i_dp, i_c, i_w] += 2.0 * planner_info.cost.w_acc[None] * acc * coeff
            # Jerk stencil: j_w = q_{w+2} - 3 q_{w+1} + 3 q_w - q_{w-1}; knot i_w is term (dw) of row
            # i_wc = i_w - dw for dw in {-1, 0, 1, 2}.
            for dw in qd.static(range(-1, 3)):
                i_wc = i_w - dw
                if 1 <= i_wc < n_knots - 2:
                    jerk = (
                        planner_state.cost.qpos[i_dp, i_w0 + i_wc + 2]
                        - 3.0 * planner_state.cost.qpos[i_dp, i_w0 + i_wc + 1]
                        + 3.0 * planner_state.cost.qpos[i_dp, i_w0 + i_wc]
                        - planner_state.cost.qpos[i_dp, i_w0 + i_wc - 1]
                    )
                    coeff = gs.qd_float(0.0)
                    if dw == 2:
                        coeff = 1.0
                    elif dw == 1:
                        coeff = -3.0
                    elif dw == 0:
                        coeff = 3.0
                    else:
                        coeff = -1.0
                    if dw == 0:
                        cost += planner_info.cost.w_jerk[None] * jerk**2
                    planner_state.cost.grad[i_dp, i_c, i_w] += 2.0 * planner_info.cost.w_jerk[None] * jerk * coeff
            # Joint-limit quadratic hinge.
            over = q - planner_info.fk.dofs.q_limit_upper[i_dp]
            under = planner_info.fk.dofs.q_limit_lower[i_dp] - q
            if over > 0.0:
                cost += planner_info.cost.w_lim[None] * over**2
                planner_state.cost.grad[i_dp, i_c, i_w] += 2.0 * planner_info.cost.w_lim[None] * over
            if under > 0.0:
                cost += planner_info.cost.w_lim[None] * under**2
                planner_state.cost.grad[i_dp, i_c, i_w] -= 2.0 * planner_info.cost.w_lim[None] * under
            # Posture regularizer toward the straight-line reference (kept tiny; resolves redundancy).
            ref = planner_info.cost.boundary.qpos_start[i_dp, i_b] + (
                planner_info.cost.boundary.qpos_goal[i_dp, i_b] - planner_info.cost.boundary.qpos_start[i_dp, i_b]
            ) * (qd.cast(i_w, gs.qd_float) / float(n_knots - 1))
            cost += planner_info.cost.w_posture[None] * (q - ref) ** 2
            planner_state.cost.grad[i_dp, i_c, i_w] += 2.0 * planner_info.cost.w_posture[None] * (q - ref)

    # Terminal pose cost for Cartesian goals (last knot only), gradient through the ee chain.
    if planner_info.cost.boundary.has_pose_goal[None] and i_w == n_knots - 1:
        i_l = planner_info.cost.boundary.goal_link_idx[None] - qd.static(planner_config.link_offset)
        err_pos = planner_state.fk.links_pos[i_l, i_cw] - planner_info.cost.boundary.goal_pos[i_b]
        quat_rel = gu.qd_transform_quat_by_quat(
            planner_state.fk.links_quat[i_l, i_cw], gu.qd_inv_quat(planner_info.cost.boundary.goal_quat[i_b])
        )
        err_rot = gu.qd_quat_to_rotvec(quat_rel, rigid_info.EPS[None])
        cost += planner_info.cost.w_pose_pos[None] * err_pos.norm_sqr()
        cost += planner_info.cost.w_pose_rot[None] * err_rot.norm_sqr()
        # Position term: gradient 2 w (x - x*) applied at the link origin.
        func_chain_grad(
            i_l,
            i_c,
            i_w,
            i_cw,
            i_b,
            planner_state.fk.links_pos[i_l, i_cw],
            err_pos,
            2.0 * planner_info.cost.w_pose_pos[None],
            joints_xanchor=planner_state.fk.joints_xanchor,
            joints_xaxis=planner_state.fk.joints_xaxis,
            grad_traj=planner_state.cost.grad,
            planner_info=planner_info,
            dyn_info=dyn_info,
            rigid_config=rigid_config,
            planner_config=planner_config,
        )
        # Rotation term: the angular jacobian row of a revolute joint is its world axis, so the chain
        # walk with the pure-rotation "point" trick applies the axis dot product directly.
        link_offset = qd.static(planner_config.link_offset)
        q_offset = qd.static(planner_config.q_offset)
        i_l_glob = i_l + link_offset
        while i_l_glob != -1:
            I_l = [gs.qd_int(i_l_glob), i_b] if qd.static(rigid_config.batch_links_info) else gs.qd_int(i_l_glob)
            if dyn_info.links.parent_idx[I_l] == -1 and dyn_info.links.is_fixed[I_l]:
                i_l_glob = -1
            else:
                for i_j_ in range(dyn_info.links.joint_start[I_l], dyn_info.links.joint_end[I_l]):
                    i_j = gs.qd_int(i_j_)
                    I_j = [i_j, i_b] if qd.static(rigid_config.batch_joints_info) else i_j
                    if dyn_info.joints.type[I_j] == gs.JOINT_TYPE.REVOLUTE:
                        i_dp = dyn_info.joints.q_start[I_j] - q_offset
                        if not planner_info.fk.dofs.is_locked[i_dp, i_b]:
                            planner_state.cost.grad[i_dp, i_c, i_w] -= (
                                2.0
                                * planner_info.cost.w_pose_rot[None]
                                * planner_state.fk.joints_xaxis[i_j - qd.static(planner_config.joint_offset), i_cw].dot(
                                    err_rot
                                )
                            )
                i_l_glob = dyn_info.links.parent_idx[I_l]

    planner_state.cost.cost_wp[i_c, i_w] = cost


@qd.func
def func_eval_clearance(
    i_e_col,
    i_b,
    swp,
    dq_inf,
    planner_state: array_class.PlannerState,
    planner_world: array_class.PlannerWorldState,
    dyn_state: array_class.DynState,
    collider_state: array_class.ColliderState,
    gjk_state: array_class.GJKState,
    planner_info: array_class.PlannerEntityInfo,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    collider_info: array_class.ColliderInfo,
    sdf_info: array_class.SDFInfo,
    rigid_config: qd.template(),
    collider_static_config: qd.template(),
    planner_config: qd.template(),
):
    """Signed clearance of the configuration held in eval scratch column i_e_col, sweep allowance included.

    Returns the pair (min_sd_exact, min_sd_proxy); see func_collision_cost.
    """
    func_fk(
        i_e_col,
        i_e_col,
        i_b,
        qpos=planner_state.fk.eval.qpos,
        links_pos=planner_state.fk.eval.links_pos,
        links_quat=planner_state.fk.eval.links_quat,
        joints_xanchor=planner_state.fk.eval.joints_xanchor,
        joints_xaxis=planner_state.fk.eval.joints_xaxis,
        dyn_state=dyn_state,
        dyn_info=dyn_info,
        rigid_info=rigid_info,
        rigid_config=rigid_config,
        planner_config=planner_config,
    )
    func_spheres(
        i_e_col,
        i_b,
        links_pos=planner_state.fk.eval.links_pos,
        links_quat=planner_state.fk.eval.links_quat,
        spheres_pos=planner_state.fk.eval.spheres_pos,
        planner_info=planner_info,
        planner_config=planner_config,
    )
    _, min_sd_exact, min_sd_proxy = func_collision_cost(
        i_e_col,
        i_b,
        swp,
        dq_inf,
        links_pos=planner_state.fk.eval.links_pos,
        links_quat=planner_state.fk.eval.links_quat,
        spheres_pos=planner_state.fk.eval.spheres_pos,
        planner_world=planner_world,
        collider_state=collider_state,
        gjk_state=gjk_state,
        planner_info=planner_info,
        dyn_info=dyn_info,
        collider_info=collider_info,
        sdf_info=sdf_info,
        rigid_config=rigid_config,
        collider_static_config=collider_static_config,
        planner_config=planner_config,
        use_exact=True,
    )
    return min_sd_exact, min_sd_proxy


@qd.func
def func_validate(
    envs_idx: qd.types.ndarray(),
    planner_state: array_class.PlannerState,
    planner_world: array_class.PlannerWorldState,
    dyn_state: array_class.DynState,
    collider_state: array_class.ColliderState,
    gjk_state: array_class.GJKState,
    planner_info: array_class.PlannerEntityInfo,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    collider_info: array_class.ColliderInfo,
    sdf_info: array_class.SDFInfo,
    rigid_config: qd.template(),
    collider_static_config: qd.template(),
    planner_config: qd.template(),
    check_start: qd.template(),
):
    """Certify every active candidate independently of the optimizer costs.

    Joint limits and collision clearance are checked at _VALIDATE_UPSAMPLE x knot density, each sample carrying the
    swept allowance from the per-DOF
    Lipschitz reach bounds so the whole continuous path is covered, plus the goal tolerance. A sample failing
    only through the sweep allowance is re-certified at _VALIDATE_REFINE x local density before being flagged,
    so the bound's conservatism costs extra samples instead of false rejections. Writes the per-candidate flags
    bitfield and the exact / proxy min clearances (see func_collision_cost); one thread per candidate,
    serial inside (deterministic).
    """
    n_knots = qd.static(planner_config.n_knots)
    n_seeds = qd.static(planner_config.n_seeds)
    n_dp = qd.static(planner_config.n_dp)
    n_upsample = qd.static(planner_config.n_upsample)
    n_refine = qd.static(planner_config.n_refine)
    n_samples = qd.static(planner_config.n_upsample * (planner_config.n_knots - 1) + 1)

    # Every candidate is validated, frozen ones included: the raw sampling-fallback path is deliberately kept
    # unrefined as an insurance candidate, and re-validating already-solved candidates is idempotent.
    qd.loop_config(serialize=qd.static(planner_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_c in range(envs_idx.shape[0] * n_seeds):
        if True:
            i_b = envs_idx[i_c // n_seeds]
            # Eval scratch column owned by this thread.
            i_e_col = i_c
            flags = gs.qd_int(0)
            min_clearance_exact = gs.qd_float(qd.math.inf)
            min_clearance_proxy = gs.qd_float(qd.math.inf)

            for i_smp in range(n_samples):
                # Linear interpolation between knots at the densified sample, and the per-segment sweep bound.
                t = qd.cast(i_smp, gs.qd_float) / float(n_samples - 1) * float(n_knots - 1)
                i_w = qd.min(gs.qd_int(t), n_knots - 2)
                alpha = t - qd.cast(i_w, gs.qd_float)
                swp = gs.qd_float(0.0)
                dq_inf = gs.qd_float(0.0)
                for i_dp in range(n_dp):
                    q0 = planner_state.cost.qpos[i_dp, i_c * n_knots + i_w]
                    q1 = planner_state.cost.qpos[i_dp, i_c * n_knots + i_w + 1]
                    q = q0 + alpha * (q1 - q0)
                    planner_state.fk.eval.qpos[i_dp, i_e_col] = q
                    # Half the inter-sample joint travel bounds the workspace motion of every sphere.
                    swp += 0.5 * planner_info.fk.dofs.reach[i_dp] * qd.abs(q1 - q0) / float(n_upsample)
                    dq_inf = qd.max(dq_inf, qd.abs(q1 - q0) / float(n_upsample))
                    # A start (or goal) configuration supplied beyond a limit is a boundary condition, so a knot
                    # is only flagged when it exceeds the limit by more than the boundary conditions already do.
                    over_allow = qd.max(
                        planner_info.cost.boundary.qpos_start[i_dp, i_b] - planner_info.fk.dofs.q_limit_upper[i_dp],
                        planner_info.cost.boundary.qpos_goal[i_dp, i_b] - planner_info.fk.dofs.q_limit_upper[i_dp],
                    )
                    under_allow = qd.max(
                        planner_info.fk.dofs.q_limit_lower[i_dp] - planner_info.cost.boundary.qpos_start[i_dp, i_b],
                        planner_info.fk.dofs.q_limit_lower[i_dp] - planner_info.cost.boundary.qpos_goal[i_dp, i_b],
                    )
                    if (
                        q > planner_info.fk.dofs.q_limit_upper[i_dp] + qd.max(over_allow, 0.0) + 1e-5
                        or q < planner_info.fk.dofs.q_limit_lower[i_dp] - qd.max(under_allow, 0.0) - 1e-5
                    ):
                        flags |= planner_config.flag_joint_limit

                min_sd_exact, min_sd_proxy = func_eval_clearance(
                    i_e_col,
                    i_b,
                    swp,
                    dq_inf,
                    planner_state=planner_state,
                    planner_world=planner_world,
                    dyn_state=dyn_state,
                    collider_state=collider_state,
                    gjk_state=gjk_state,
                    planner_info=planner_info,
                    dyn_info=dyn_info,
                    rigid_info=rigid_info,
                    collider_info=collider_info,
                    sdf_info=sdf_info,
                    rigid_config=rigid_config,
                    collider_static_config=collider_static_config,
                    planner_config=planner_config,
                )
                min_sd = qd.min(min_sd_exact, min_sd_proxy)
                if min_sd < 0.0 and min_sd + swp > 0.0:
                    # Borderline: the raw clearance is positive and only the sweep allowance fails, so the
                    # covered interval [t - h, t + h] is re-certified by _VALIDATE_REFINE sub-samples whose
                    # allowance shrinks accordingly (see _VALIDATE_REFINE).
                    min_sd_exact = qd.math.inf
                    min_sd_proxy = qd.math.inf
                    h = 0.5 / float(n_upsample)
                    for i_r in range(n_refine):
                        t_r = t + h * ((2.0 * qd.cast(i_r, gs.qd_float) + 1.0) / float(n_refine) - 1.0)
                        t_r = qd.max(qd.min(t_r, float(n_knots - 1)), 0.0)
                        i_w_r = qd.min(gs.qd_int(t_r), n_knots - 2)
                        alpha_r = t_r - qd.cast(i_w_r, gs.qd_float)
                        swp_r = gs.qd_float(0.0)
                        dq_inf_r = gs.qd_float(0.0)
                        for i_dp in range(n_dp):
                            q0r = planner_state.cost.qpos[i_dp, i_c * n_knots + i_w_r]
                            q1r = planner_state.cost.qpos[i_dp, i_c * n_knots + i_w_r + 1]
                            planner_state.fk.eval.qpos[i_dp, i_e_col] = q0r + alpha_r * (q1r - q0r)
                            dq_r = qd.abs(q1r - q0r) / float(n_upsample * n_refine)
                            swp_r += 0.5 * planner_info.fk.dofs.reach[i_dp] * dq_r
                            dq_inf_r = qd.max(dq_inf_r, dq_r)
                        min_sd_exact_r, min_sd_proxy_r = func_eval_clearance(
                            i_e_col,
                            i_b,
                            swp_r,
                            dq_inf_r,
                            planner_state=planner_state,
                            planner_world=planner_world,
                            dyn_state=dyn_state,
                            collider_state=collider_state,
                            gjk_state=gjk_state,
                            planner_info=planner_info,
                            dyn_info=dyn_info,
                            rigid_info=rigid_info,
                            collider_info=collider_info,
                            sdf_info=sdf_info,
                            rigid_config=rigid_config,
                            collider_static_config=collider_static_config,
                            planner_config=planner_config,
                        )
                        min_sd_exact = qd.min(min_sd_exact, min_sd_exact_r)
                        min_sd_proxy = qd.min(min_sd_proxy, min_sd_proxy_r)
                min_clearance_exact = qd.min(min_clearance_exact, min_sd_exact)
                min_clearance_proxy = qd.min(min_clearance_proxy, min_sd_proxy)
                if qd.min(min_sd_exact, min_sd_proxy) < 0.0:
                    flags |= planner_config.flag_collision

            # Boundary reaching: every candidate must depart from qpos_start, and land on qpos_goal (joint
            # goals) or within the pose tolerance (Cartesian goals) - otherwise a trajectory holding inside its
            # own boundary allowances (e.g. a probe hold-at-goal) would certify as a plan. The terminal knot is
            # re-evaluated explicitly, since local refinement may have left another sample in the eval scratch.
            if qd.static(check_start):
                for i_dp in range(n_dp):
                    if (
                        qd.abs(
                            planner_state.cost.qpos[i_dp, i_c * n_knots]
                            - planner_info.cost.boundary.qpos_start[i_dp, i_b]
                        )
                        > 1e-4
                    ):
                        flags |= planner_config.flag_goal_tol
            if planner_info.cost.boundary.has_pose_goal[None]:
                for i_dp in range(n_dp):
                    planner_state.fk.eval.qpos[i_dp, i_e_col] = planner_state.cost.qpos[
                        i_dp, i_c * n_knots + n_knots - 1
                    ]
                func_fk(
                    i_e_col,
                    i_e_col,
                    i_b,
                    qpos=planner_state.fk.eval.qpos,
                    links_pos=planner_state.fk.eval.links_pos,
                    links_quat=planner_state.fk.eval.links_quat,
                    joints_xanchor=planner_state.fk.eval.joints_xanchor,
                    joints_xaxis=planner_state.fk.eval.joints_xaxis,
                    dyn_state=dyn_state,
                    dyn_info=dyn_info,
                    rigid_info=rigid_info,
                    rigid_config=rigid_config,
                    planner_config=planner_config,
                )
                i_l = planner_info.cost.boundary.goal_link_idx[None] - qd.static(planner_config.link_offset)
                err_pos = planner_state.fk.eval.links_pos[i_l, i_e_col] - planner_info.cost.boundary.goal_pos[i_b]
                quat_rel = gu.qd_transform_quat_by_quat(
                    planner_state.fk.eval.links_quat[i_l, i_e_col],
                    gu.qd_inv_quat(planner_info.cost.boundary.goal_quat[i_b]),
                )
                err_rot = gu.qd_quat_to_rotvec(quat_rel, rigid_info.EPS[None])
                if err_pos.norm() > 5e-3 or err_rot.norm() > 5e-2:
                    flags |= planner_config.flag_goal_tol
            else:
                for i_dp in range(n_dp):
                    q_end = planner_state.cost.qpos[i_dp, i_c * n_knots + n_knots - 1]
                    if qd.abs(q_end - planner_info.cost.boundary.qpos_goal[i_dp, i_b]) > 1e-4:
                        flags |= planner_config.flag_goal_tol

            planner_state.cert.valid_flags[i_c] = flags
            planner_state.cert.min_clearance_exact[i_c] = min_clearance_exact
            planner_state.cert.min_clearance_proxy[i_c] = min_clearance_proxy


@qd.kernel
def kernel_validate(
    envs_idx: qd.types.ndarray(),
    planner_state: array_class.PlannerState,
    planner_world: array_class.PlannerWorldState,
    dyn_state: array_class.DynState,
    collider_state: array_class.ColliderState,
    gjk_state: array_class.GJKState,
    planner_info: array_class.PlannerEntityInfo,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    collider_info: array_class.ColliderInfo,
    sdf_info: array_class.SDFInfo,
    rigid_config: qd.template(),
    collider_static_config: qd.template(),
    planner_config: qd.template(),
    check_start: qd.template(),
):
    func_validate(
        envs_idx,
        planner_state,
        planner_world,
        dyn_state,
        collider_state,
        gjk_state,
        planner_info,
        dyn_info,
        rigid_info,
        collider_info,
        sdf_info,
        rigid_config,
        collider_static_config,
        planner_config,
        check_start=check_start,
    )


@qd.func
def func_resolve_goal(
    graph_counter: qd.types.ndarray(),
    envs_idx: qd.types.ndarray(),
    planner_state: array_class.PlannerState,
    planner_world: array_class.PlannerWorldState,
    dyn_state: array_class.DynState,
    collider_state: array_class.ColliderState,
    gjk_state: array_class.GJKState,
    planner_info: array_class.PlannerEntityInfo,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    collider_info: array_class.ColliderInfo,
    sdf_info: array_class.SDFInfo,
    rigid_config: qd.template(),
    collider_static_config: qd.template(),
    planner_config: qd.template(),
    ignore_collision: qd.template(),
):
    """Resolve the Cartesian goal of each unsolved env by parallel multi-restart inverse kinematics.

    Every candidate column is one restart (seed 0 warm-starts from the plan start, the rest sample uniformly in
    the joint limits keyed by the attempt), solved independently by damped least squares on the per-column planner
    forward kinematics. Each restart's converged configuration is validated as a hold-at-goal candidate, and the
    closest-to-start certified restart per env - preferring collision-free over excusably-contacting - becomes that
    env's goal configuration. Envs with no acceptable restart keep their previous goal.
    """
    EPS = rigid_info.EPS[None]
    n_seeds = qd.static(planner_config.n_seeds)
    n_knots = qd.static(planner_config.n_knots)
    n_dp = qd.static(planner_config.n_dp)
    link_offset = qd.static(planner_config.link_offset)
    goal_link = planner_info.cost.boundary.goal_link_idx[None]
    attempt = graph_counter[()]
    margin = planner_info.cost.d_safe[None]

    # Reset the per-pass best goal score: each pass keeps the best branch across its restart sub-batches
    # before overwriting the goal, so a pass that finds no acceptable branch still leaves the next pass to
    # retry with fresh restarts (the adaptive re-resolution).
    qd.loop_config(name="planner_resolve_reset")
    for i_b_reset in range(envs_idx.shape[0]):
        planner_state.goal_resolve_score[envs_idx[i_b_reset]] = 1e30

    # Draw goal_ik_batches sub-batches of n_seeds restarts, reusing the candidate columns: this decouples the
    # goal-resolution restart depth from the trajectory candidate count, so a rare collision-free goal branch
    # is sampled reliably without inflating the far more expensive trajectory-candidate pipeline. Each
    # sub-batch is validated as hold-at-goal candidates and its best acceptable restart competes for the
    # env's goal (collision-free preferred), the pass keeping the best branch across all its sub-batches.
    for i_batch in range(qd.static(planner_config.goal_ik_batches)):
        attempt_key = attempt * qd.static(planner_config.goal_ik_batches) + i_batch
        # One damped-least-squares solve per restart column - all restarts of all envs run in parallel.
        qd.loop_config(serialize=qd.static(planner_config.para_level < gs.PARA_LEVEL.PARTIAL))
        for i_c in range(envs_idx.shape[0] * n_seeds):
            i_b = envs_idx[i_c // n_seeds]
            if not planner_state.is_env_solved[i_b]:
                i_r = i_c % n_seeds
                for i_dp in range(n_dp):
                    q = planner_info.cost.boundary.qpos_start[i_dp, i_b]
                    if i_r > 0 and not planner_info.fk.dofs.is_locked[i_dp, i_b]:
                        lo = planner_info.fk.dofs.q_limit_lower[i_dp]
                        hi = planner_info.fk.dofs.q_limit_upper[i_dp]
                        q = lo + gu.qd_hash01(planner_info.mppi.seed_key[None], i_c, i_dp, attempt_key) * (hi - lo)
                    planner_state.fk.eval.qpos[i_dp, i_c] = q

                for _ in range(qd.static(planner_config.goal_ik_iters)):
                    func_fk(
                        i_c,
                        i_c,
                        i_b,
                        planner_state.fk.eval.qpos,
                        planner_state.fk.eval.links_pos,
                        planner_state.fk.eval.links_quat,
                        planner_state.fk.eval.joints_xanchor,
                        planner_state.fk.eval.joints_xaxis,
                        dyn_state,
                        dyn_info,
                        rigid_info,
                        rigid_config,
                        planner_config,
                    )
                    ee_pos = planner_state.fk.eval.links_pos[goal_link - link_offset, i_c]
                    ee_quat = planner_state.fk.eval.links_quat[goal_link - link_offset, i_c]
                    err_pos = planner_info.cost.boundary.goal_pos[i_b] - ee_pos
                    err_rot = gu.qd_quat_to_rotvec(
                        gu.qd_transform_quat_by_quat(
                            gu.qd_inv_quat(ee_quat), planner_info.cost.boundary.goal_quat[i_b]
                        ),
                        EPS,
                    )
                    for k in qd.static(range(3)):
                        planner_state.ik.err_pose[k, i_c] = err_pos[k]
                        planner_state.ik.err_pose[k + 3, i_c] = err_rot[k]
                    if (
                        err_pos.norm() <= planner_config.goal_ik_pos_tol
                        and err_rot.norm() <= planner_config.goal_ik_rot_tol
                    ):
                        break

                    func_ik_jacobian(
                        i_c,
                        i_b,
                        goal_link,
                        ee_pos,
                        planner_state.ik.jacobian_stacked,
                        planner_state.fk.eval.joints_xanchor,
                        planner_state.fk.eval.joints_xaxis,
                        dyn_info,
                        rigid_config,
                        planner_config,
                    )
                    lu.mat_transpose(
                        planner_state.ik.jacobian_stacked, planner_state.ik.jacobian_stacked_t, 6, n_dp, i_c
                    )
                    lu.mat_mul(
                        planner_state.ik.jacobian_stacked,
                        planner_state.ik.jacobian_stacked_t,
                        planner_state.ik.mat,
                        6,
                        n_dp,
                        6,
                        i_c,
                    )
                    lu.mat_add_eye(planner_state.ik.mat, planner_config.goal_ik_damping**2, 6, i_c)
                    lu.mat_inverse(
                        planner_state.ik.mat,
                        planner_state.ik.lu_lower,
                        planner_state.ik.lu_upper,
                        planner_state.ik.lu_y,
                        planner_state.ik.inv,
                        6,
                        i_c,
                    )
                    lu.mat_mul_vec(planner_state.ik.inv, planner_state.ik.err_pose, planner_state.ik.vec, 6, 6, i_c)
                    for i_dp in range(n_dp):
                        if not planner_info.fk.dofs.is_locked[i_dp, i_b]:
                            dq = gs.qd_float(0.0)
                            for j in range(6):
                                dq += planner_state.ik.jacobian_stacked_t[i_dp, j, i_c] * planner_state.ik.vec[j, i_c]
                            planner_state.fk.eval.qpos[i_dp, i_c] = qd.math.clamp(
                                planner_state.fk.eval.qpos[i_dp, i_c]
                                + qd.math.clamp(dq, -planner_config.goal_ik_max_step, planner_config.goal_ik_max_step),
                                planner_info.fk.dofs.q_limit_lower[i_dp],
                                planner_info.fk.dofs.q_limit_upper[i_dp],
                            )

                # Store the restart's solution and lay it down as a hold-at-goal candidate for the shared validator.
                col_base = i_c * n_knots
                for i_dp in range(n_dp):
                    planner_state.ik.qpos_best[i_dp, i_c] = planner_state.fk.eval.qpos[i_dp, i_c]
                    for i_w in range(n_knots):
                        planner_state.cost.qpos[i_dp, col_base + i_w] = planner_state.fk.eval.qpos[i_dp, i_c]
                planner_state.cert.is_active[i_c] = True

        func_validate(
            envs_idx,
            planner_state,
            planner_world,
            dyn_state,
            collider_state,
            gjk_state,
            planner_info,
            dyn_info,
            rigid_info,
            collider_info,
            sdf_info,
            rigid_config,
            collider_static_config,
            planner_config,
            check_start=False,
        )

        # Adopt each env's goal: the closest-to-start certified restart, preferring collision-free over excusable.
        qd.loop_config(serialize=qd.static(planner_config.para_level < gs.PARA_LEVEL.PARTIAL))
        for i_b_ in range(envs_idx.shape[0]):
            i_b = envs_idx[i_b_]
            if not planner_state.is_env_solved[i_b]:
                best_c = gs.qd_int(-1)
                best_score = gs.qd_float(1e30)
                for i_r in range(n_seeds):
                    i_c = i_b_ * n_seeds + i_r
                    flags = planner_state.cert.valid_flags[i_c]
                    is_acceptable = (flags & planner_config.flag_goal_tol) == 0
                    # With collision checking disabled, a converged restart is accepted regardless of clearance;
                    # otherwise a converged solution is acceptable only if collision-free or excusably contacting.
                    if qd.static(not ignore_collision):
                        if is_acceptable:
                            is_acceptable = planner_state.cert.min_clearance_exact[i_c] + margin + 0.01 > -(
                                planner_info.cert.exact_sample_cov[None] + planner_info.cert.goal_real_pen_max[None]
                            )
                        if is_acceptable:
                            is_acceptable = (
                                planner_state.cert.min_clearance_proxy[i_c] + margin + 0.01
                                > -planner_info.cert.excl_depth_max[None]
                            )
                    if is_acceptable:
                        dist = gs.qd_float(0.0)
                        for i_dp in range(n_dp):
                            dist += qd.abs(
                                planner_state.ik.qpos_best[i_dp, i_c] - planner_info.cost.boundary.qpos_start[i_dp, i_b]
                            )
                        if qd.static(not ignore_collision) and (flags & planner_config.flag_collision) != 0:
                            dist += 1e6
                        if dist < best_score:
                            best_score = dist
                            best_c = i_c
                if best_c != -1 and best_score < planner_state.goal_resolve_score[i_b]:
                    planner_state.goal_resolve_score[i_b] = best_score
                    for i_dp in range(n_dp):
                        planner_info.cost.boundary.qpos_goal[i_dp, i_b] = planner_state.ik.qpos_best[i_dp, best_c]
