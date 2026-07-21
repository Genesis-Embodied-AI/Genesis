import quadrants as qd

import genesis as gs
import genesis.utils.geom as gu
from genesis.utils import array_class

from .kinematics import func_planner_fk, func_planner_spheres
from .world import func_planner_world_aabb_skip, func_planner_world_sd, func_planner_world_sd_grad

# Validator flag bits (returned per candidate; 0 = certified feasible).
COLLISION = 1
JOINT_LIMIT = 2
GOAL_TOL = 4
GOAL_IN_COLLISION = 8
# Densification factor of the validator relative to the knot spacing.
_VALIDATE_UPSAMPLE = 8


@qd.func
def func_planner_hinge(sd, eps):
    """CHOMP smooth hinge: linear in penetration, quadratic within the activation band, zero beyond. Returns
    (value, d/d sd)."""
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
def func_planner_excl_world_offset(i_s, i_gw, i_b, plan_info: array_class.PlannerEntityInfo):
    """Start-contact allowance of a (sphere, world geom) pair: pairs violating the margin at qpos_start may keep
    their start clearance (never get worse); everything else gets no allowance."""
    offset = gs.qd_float(0.0)
    for i_x in range(plan_info.excl_world_count[i_b]):
        if plan_info.excl_world_pair[i_x, i_b][0] == i_s and plan_info.excl_world_pair[i_x, i_b][1] == i_gw:
            offset = qd.min(plan_info.excl_world_sd[i_x, i_b], 0.0)
    return offset


@qd.func
def func_planner_excl_self_offset(i_p, i_b, plan_info: array_class.PlannerEntityInfo):
    """Start-contact allowance of a self-collision sphere pair (see func_planner_excl_world_offset)."""
    offset = gs.qd_float(0.0)
    for i_x in range(plan_info.excl_self_count[i_b]):
        if plan_info.excl_self_pair[i_x, i_b] == i_p:
            offset = qd.min(plan_info.excl_self_sd[i_x, i_b], 0.0)
    return offset


@qd.func
def func_planner_sphere_radius(i_s, i_b, plan_info: array_class.PlannerEntityInfo, planner_config: qd.template()):
    """Radius of proxy sphere i_s, attached spheres included (inactive attached spheres read radius 0)."""
    radius = gs.qd_float(0.0)
    if i_s < qd.static(planner_config.n_spheres):
        radius = plan_info.spheres_radius[i_s]
    elif plan_info.attach_spheres_is_active[i_s - qd.static(planner_config.n_spheres), i_b]:
        radius = plan_info.attach_spheres_radius[i_s - qd.static(planner_config.n_spheres)]
    return radius


@qd.func
def func_planner_sphere_link(i_s, plan_info: array_class.PlannerEntityInfo, planner_config: qd.template()):
    """Entity-local link carrying proxy sphere i_s (attach link for attached spheres)."""
    i_l = gs.qd_int(0)
    if i_s < qd.static(planner_config.n_spheres):
        i_l = plan_info.spheres_link_idx[i_s]
    else:
        i_l = plan_info.attach_spheres_link_idx[i_s - qd.static(planner_config.n_spheres)]
    return i_l


@qd.func
def func_planner_chain_grad(
    i_l,
    i_c,
    i_col_fk,
    i_b,
    x,
    g,
    scale,
    joints_xanchor: qd.Tensor,
    joints_xaxis: qd.Tensor,
    grad_traj: qd.Tensor,
    i_col_grad,
    plan_info: array_class.PlannerEntityInfo,
    dyn_info: array_class.DynInfo,
    rigid_config: qd.template(),
    planner_config: qd.template(),
):
    """
    Accumulate a workspace gradient g applied at point x on entity-local link i_l into the joint-space gradient
    column i_col_grad, by walking the parent chain (the J^T v product assembled without materializing J - pattern
    of the entity jacobian, rigid_entity.py). Locked DOFs are skipped.
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
                    if not plan_info.dof_is_locked[i_dp, i_b]:
                        dq = gs.qd_float(0.0)
                        if joint_type == gs.JOINT_TYPE.REVOLUTE:
                            dq = (
                                joints_xaxis[i_j - qd.static(planner_config.joint_offset), i_col_fk]
                                .cross(x - joints_xanchor[i_j - qd.static(planner_config.joint_offset), i_col_fk])
                                .dot(g)
                            )
                        else:
                            dq = joints_xaxis[i_j - qd.static(planner_config.joint_offset), i_col_fk].dot(g)
                        grad_traj[i_dp, i_col_grad] += scale * dq
            i_l_glob = dyn_info.links.parent_idx[I_l]


@qd.func
def func_planner_collision_cost(
    i_col,
    i_b,
    swp,
    dq_inf,
    spheres_pos: qd.Tensor,
    plan_info: array_class.PlannerEntityInfo,
    plan_world: array_class.PlannerWorldState,
    dyn_info: array_class.DynInfo,
    sdf_info: array_class.SDFInfo,
    planner_config: qd.template(),
):
    """
    Collision cost of one FK column (cost-only path): world hinge over every (sphere, obstacle geom) within band +
    self/attach hinge over the pair lists. The sweep allowance swp inflates every sphere so the cost covers the
    whole inter-sample segment (SDF is 1-Lipschitz). Returns (cost, min signed clearance).
    """
    n_sph_tot = qd.static(planner_config.n_spheres + planner_config.n_attach_max)
    cost = gs.qd_float(0.0)
    min_sd = gs.qd_float(qd.math.inf)

    for i_s in range(n_sph_tot):
        radius = func_planner_sphere_radius(i_s, i_b, plan_info, planner_config)
        if radius > 0.0:
            x = spheres_pos[i_s, i_col]
            band = radius + plan_info.d_safe[None] + plan_info.eps_act[None] + swp
            for i_gw in range(plan_world.n_geoms[None]):
                if plan_world.geoms_is_active[i_gw, i_b] and not func_planner_world_aabb_skip(
                    i_gw, i_b, x, band, plan_world
                ):
                    eps_act = qd.min(plan_info.eps_act[None], plan_world.geoms_max_band[i_gw])
                    sd = func_planner_world_sd(i_gw, i_b, x, plan_world, dyn_info, sdf_info)
                    sd_eff = sd - radius - plan_info.d_safe[None] - swp
                    offset = func_planner_excl_world_offset(i_s, i_gw, i_b, plan_info)
                    if offset < 0.0:
                        # Excluded pair: allowed to keep its start clearance, sweep-free (the allowance is exact
                        # at the start sample, and sweep-inflating it would flag pairs pinned at constant depth).
                        sd_eff = sd - radius - plan_info.d_safe[None] - offset + 1e-4
                    min_sd = qd.min(min_sd, sd_eff)
                    hinge, _ = func_planner_hinge(sd_eff, eps_act)
                    cost += plan_info.w_obs[None] * hinge

    for i_p in range(plan_info.self_pairs.shape[0]):
        i_sa, i_sb = plan_info.self_pairs[i_p][0], plan_info.self_pairs[i_p][1]
        radius_a = func_planner_sphere_radius(i_sa, i_b, plan_info, planner_config)
        radius_b = func_planner_sphere_radius(i_sb, i_b, plan_info, planner_config)
        if radius_a > 0.0 and radius_b > 0.0:
            dist = (spheres_pos[i_sa, i_col] - spheres_pos[i_sb, i_col]).norm()
            swp_pair = qd.min(0.5 * plan_info.self_pairs_reach[i_p] * dq_inf, plan_info.eps_act[None])
            sd_eff = dist - radius_a - radius_b - plan_info.d_safe[None] - swp_pair
            offset = func_planner_excl_self_offset(i_p, i_b, plan_info)
            if offset < 0.0:
                # See the world-pair exclusion above.
                sd_eff = dist - radius_a - radius_b - plan_info.d_safe[None] - offset + 1e-4
            min_sd = qd.min(min_sd, sd_eff)
            hinge, _ = func_planner_hinge(sd_eff, plan_info.eps_self[None])
            cost += plan_info.w_self[None] * hinge

    return cost, min_sd


@qd.func
def func_planner_pose_cost(
    i_col,
    i_b,
    links_pos: qd.Tensor,
    links_quat: qd.Tensor,
    plan_info: array_class.PlannerEntityInfo,
    rigid_info: array_class.RigidInfo,
    planner_config: qd.template(),
):
    """Terminal pose cost (position + rotation-vector geodesic) of the goal link for Cartesian goals."""
    i_l = plan_info.goal_link_idx[None] - qd.static(planner_config.link_offset)
    err_pos = links_pos[i_l, i_col] - plan_info.goal_pos[i_b]
    quat_rel = gu.qd_transform_quat_by_quat(links_quat[i_l, i_col], gu.qd_inv_quat(plan_info.goal_quat[i_b]))
    err_rot = gu.qd_quat_to_rotvec(quat_rel, rigid_info.EPS[None])
    return plan_info.w_pose_pos[None] * err_pos.norm_sqr() + plan_info.w_pose_rot[None] * err_rot.norm_sqr()


@qd.kernel
def kernel_planner_start_exclusions(
    envs_idx: qd.types.ndarray(),
    plan_state: array_class.PlannerState,
    plan_info: array_class.PlannerEntityInfo,
    plan_world: array_class.PlannerWorldState,
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    sdf_info: array_class.SDFInfo,
    rigid_config: qd.template(),
    planner_config: qd.template(),
    errno: qd.Tensor,
):
    """Collect the start-config contact exclusions per env (serial per env - the lists are tiny)."""
    n_sph_tot = qd.static(planner_config.n_spheres + planner_config.n_attach_max)
    n_excl_max = plan_info.excl_world_pair.shape[0]

    qd.loop_config(serialize=qd.static(planner_config.para_level < gs.PARA_LEVEL.ALL))
    for i_b_ in range(envs_idx.shape[0]):
        i_b = envs_idx[i_b_]
        # FK of the start configuration through the eval scratch, one column per env.
        for i_dp in range(qd.static(planner_config.n_dp)):
            plan_state.eval_qpos[i_dp, i_b_] = plan_info.qpos_start[i_dp, i_b]
        func_planner_fk(
            i_b_,
            i_b_,
            i_b,
            qpos=plan_state.eval_qpos,
            links_pos=plan_state.eval_links_pos,
            links_quat=plan_state.eval_links_quat,
            joints_xanchor=plan_state.eval_joints_xanchor,
            joints_xaxis=plan_state.eval_joints_xaxis,
            dyn_state=dyn_state,
            dyn_info=dyn_info,
            rigid_info=rigid_info,
            rigid_config=rigid_config,
            planner_config=planner_config,
        )
        func_planner_spheres(
            i_b_,
            i_b,
            links_pos=plan_state.eval_links_pos,
            links_quat=plan_state.eval_links_quat,
            spheres_pos=plan_state.eval_spheres_pos,
            plan_info=plan_info,
            planner_config=planner_config,
        )

        n_world = 0
        for i_s in range(n_sph_tot):
            radius = func_planner_sphere_radius(i_s, i_b, plan_info, planner_config)
            if radius > 0.0:
                x = plan_state.eval_spheres_pos[i_s, i_b_]
                band = radius + plan_info.d_safe[None]
                for i_gw in range(plan_world.n_geoms[None]):
                    if plan_world.geoms_is_active[i_gw, i_b] and not func_planner_world_aabb_skip(
                        i_gw, i_b, x, band, plan_world
                    ):
                        sd = func_planner_world_sd(i_gw, i_b, x, plan_world, dyn_info, sdf_info)
                        sd_eff = sd - radius - plan_info.d_safe[None]
                        if sd_eff < 0.0:
                            if n_world < n_excl_max:
                                plan_info.excl_world_pair[n_world, i_b] = qd.Vector([i_s, i_gw], dt=gs.qd_int)
                                plan_info.excl_world_sd[n_world, i_b] = sd_eff
                            n_world += 1
        if n_world > n_excl_max:
            errno[i_b] = errno[i_b] | array_class.ErrorCode.OVERFLOW_PLANNER_EXCLUSIONS
        plan_info.excl_world_count[i_b] = qd.min(n_world, n_excl_max)

        n_self = 0
        for i_p in range(plan_info.self_pairs.shape[0]):
            i_sa, i_sb = plan_info.self_pairs[i_p][0], plan_info.self_pairs[i_p][1]
            radius_a = func_planner_sphere_radius(i_sa, i_b, plan_info, planner_config)
            radius_b = func_planner_sphere_radius(i_sb, i_b, plan_info, planner_config)
            if radius_a > 0.0 and radius_b > 0.0:
                sd_eff = (
                    (plan_state.eval_spheres_pos[i_sa, i_b_] - plan_state.eval_spheres_pos[i_sb, i_b_]).norm()
                    - radius_a
                    - radius_b
                    - plan_info.d_safe[None]
                )
                if sd_eff < 0.0:
                    if n_self < n_excl_max:
                        plan_info.excl_self_pair[n_self, i_b] = i_p
                        plan_info.excl_self_sd[n_self, i_b] = sd_eff
                    n_self += 1
        if n_self > n_excl_max:
            errno[i_b] = errno[i_b] | array_class.ErrorCode.OVERFLOW_PLANNER_EXCLUSIONS
        plan_info.excl_self_count[i_b] = qd.min(n_self, n_excl_max)


@qd.func
def func_planner_knot_cost_grad(
    i_c,
    i_w,
    i_cw,
    i_b,
    plan_state: array_class.PlannerState,
    plan_info: array_class.PlannerEntityInfo,
    plan_world: array_class.PlannerWorldState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    sdf_info: array_class.SDFInfo,
    rigid_config: qd.template(),
    collider_static_config: qd.template(),
    planner_config: qd.template(),
):
    """
    Full cost and joint-space gradient of one (candidate, knot) column, from the FK cache. Writes only its own
    cost_wp / grad_traj slices - no atomics, deterministic per backend. The sweep allowance is stop-gradient (see
    the design plan): it gates where the hinge activates without contributing cross-column gradient terms.
    """
    n_knots = qd.static(planner_config.n_knots)
    n_dp = qd.static(planner_config.n_dp)
    n_sph_tot = qd.static(planner_config.n_spheres + planner_config.n_attach_max)

    cost = gs.qd_float(0.0)
    for i_dp in range(n_dp):
        plan_state.grad_traj[i_dp, i_cw] = 0.0

    # Sweep allowance: half the largest neighbor-knot travel of any sphere. Stop-gradient by design, so
    # its influence (and the induced gradient error) is clamped to the activation band; the validator
    # applies the full unclamped sweep, which is where rigor matters.
    swp = gs.qd_float(0.0)
    for i_s in range(n_sph_tot):
        if i_w > 0:
            swp = qd.max(swp, 0.5 * (plan_state.spheres_pos[i_s, i_cw] - plan_state.spheres_pos[i_s, i_cw - 1]).norm())
        if i_w < n_knots - 1:
            swp = qd.max(swp, 0.5 * (plan_state.spheres_pos[i_s, i_cw + 1] - plan_state.spheres_pos[i_s, i_cw]).norm())
    swp = qd.min(swp, plan_info.eps_act[None])

    # L-inf joint travel toward the neighbor knots, for the per-pair self-collision sweep bound.
    dq_inf = gs.qd_float(0.0)
    for i_dp in range(n_dp):
        if i_w > 0:
            dq_inf = qd.max(dq_inf, qd.abs(plan_state.qpos_traj[i_dp, i_cw] - plan_state.qpos_traj[i_dp, i_cw - 1]))
        if i_w < n_knots - 1:
            dq_inf = qd.max(dq_inf, qd.abs(plan_state.qpos_traj[i_dp, i_cw + 1] - plan_state.qpos_traj[i_dp, i_cw]))

    # World collision, with gradient through the chain walk.
    for i_s in range(n_sph_tot):
        radius = func_planner_sphere_radius(i_s, i_b, plan_info, planner_config)
        if radius > 0.0:
            x = plan_state.spheres_pos[i_s, i_cw]
            band = radius + plan_info.d_safe[None] + plan_info.eps_act[None] + swp
            for i_gw in range(plan_world.n_geoms[None]):
                if plan_world.geoms_is_active[i_gw, i_b] and not func_planner_world_aabb_skip(
                    i_gw, i_b, x, band, plan_world
                ):
                    eps_act = qd.min(plan_info.eps_act[None], plan_world.geoms_max_band[i_gw])
                    sd = func_planner_world_sd(i_gw, i_b, x, plan_world, dyn_info, sdf_info)
                    sd_eff = sd - radius - plan_info.d_safe[None] - swp
                    offset = func_planner_excl_world_offset(i_s, i_gw, i_b, plan_info)
                    if offset < 0.0:
                        # See the exclusion comment in func_planner_collision_cost.
                        sd_eff = sd - radius - plan_info.d_safe[None] - offset + 1e-4
                    hinge, dhinge = func_planner_hinge(sd_eff, eps_act)
                    if hinge > 0.0:
                        cost += plan_info.w_obs[None] * hinge
                        normal = func_planner_world_sd_grad(
                            i_gw, i_b, x, plan_world, dyn_info, rigid_info, sdf_info, collider_static_config
                        )
                        func_planner_chain_grad(
                            func_planner_sphere_link(i_s, plan_info, planner_config),
                            i_c,
                            i_cw,
                            i_b,
                            x,
                            normal,
                            plan_info.w_obs[None] * dhinge,
                            joints_xanchor=plan_state.joints_xanchor,
                            joints_xaxis=plan_state.joints_xaxis,
                            grad_traj=plan_state.grad_traj,
                            i_col_grad=i_cw,
                            plan_info=plan_info,
                            dyn_info=dyn_info,
                            rigid_config=rigid_config,
                            planner_config=planner_config,
                        )

    # Self / attach collision, gradient through both chains.
    for i_p in range(plan_info.self_pairs.shape[0]):
        i_sa, i_sb = plan_info.self_pairs[i_p][0], plan_info.self_pairs[i_p][1]
        radius_a = func_planner_sphere_radius(i_sa, i_b, plan_info, planner_config)
        radius_b = func_planner_sphere_radius(i_sb, i_b, plan_info, planner_config)
        if radius_a > 0.0 and radius_b > 0.0:
            delta = plan_state.spheres_pos[i_sa, i_cw] - plan_state.spheres_pos[i_sb, i_cw]
            dist = delta.norm()
            swp_pair = qd.min(0.5 * plan_info.self_pairs_reach[i_p] * dq_inf, plan_info.eps_act[None])
            sd_eff = dist - radius_a - radius_b - plan_info.d_safe[None] - swp_pair
            offset = func_planner_excl_self_offset(i_p, i_b, plan_info)
            if offset < 0.0:
                # See the exclusion comment in func_planner_collision_cost.
                sd_eff = dist - radius_a - radius_b - plan_info.d_safe[None] - offset + 1e-4
            hinge, dhinge = func_planner_hinge(sd_eff, plan_info.eps_self[None])
            if hinge > 0.0 and dist > gs.EPS:
                cost += plan_info.w_self[None] * hinge
                normal = delta / dist
                func_planner_chain_grad(
                    func_planner_sphere_link(i_sa, plan_info, planner_config),
                    i_c,
                    i_cw,
                    i_b,
                    plan_state.spheres_pos[i_sa, i_cw],
                    normal,
                    plan_info.w_self[None] * dhinge,
                    joints_xanchor=plan_state.joints_xanchor,
                    joints_xaxis=plan_state.joints_xaxis,
                    grad_traj=plan_state.grad_traj,
                    i_col_grad=i_cw,
                    plan_info=plan_info,
                    dyn_info=dyn_info,
                    rigid_config=rigid_config,
                    planner_config=planner_config,
                )
                func_planner_chain_grad(
                    func_planner_sphere_link(i_sb, plan_info, planner_config),
                    i_c,
                    i_cw,
                    i_b,
                    plan_state.spheres_pos[i_sb, i_cw],
                    -normal,
                    plan_info.w_self[None] * dhinge,
                    joints_xanchor=plan_state.joints_xanchor,
                    joints_xaxis=plan_state.joints_xaxis,
                    grad_traj=plan_state.grad_traj,
                    i_col_grad=i_cw,
                    plan_info=plan_info,
                    dyn_info=dyn_info,
                    rigid_config=rigid_config,
                    planner_config=planner_config,
                )

    # Smoothness (acceleration + jerk finite-difference quadratics) and joint-limit hinge + posture
    # regularizer, all local stencils on qpos_traj. Column i_cw of candidate i_c spans knots
    # [i_c*n_knots, (i_c+1)*n_knots).
    i_w0 = i_c * n_knots
    for i_dp in range(n_dp):
        if not plan_info.dof_is_locked[i_dp, i_b]:
            q = plan_state.qpos_traj[i_dp, i_cw]
            # Acceleration stencil rows this knot participates in: cost = w_a * sum_w a_w^2 with
            # a_w = q_{w-1} - 2 q_w + q_{w+1}; d cost / d q_this = 2 w_a * sum_w a_w * c_w.
            for dw in qd.static(range(-1, 2)):
                i_wc = i_w + dw
                if 1 <= i_wc < n_knots - 1:
                    acc = (
                        plan_state.qpos_traj[i_dp, i_w0 + i_wc - 1]
                        - 2.0 * plan_state.qpos_traj[i_dp, i_w0 + i_wc]
                        + plan_state.qpos_traj[i_dp, i_w0 + i_wc + 1]
                    )
                    coeff = gs.qd_float(-2.0 if dw == 0 else 1.0)
                    if dw == 0:
                        cost += plan_info.w_acc[None] * acc**2
                    plan_state.grad_traj[i_dp, i_cw] += 2.0 * plan_info.w_acc[None] * acc * coeff
            # Jerk stencil: j_w = q_{w+2} - 3 q_{w+1} + 3 q_w - q_{w-1}; knot i_w is term (dw) of row
            # i_wc = i_w - dw for dw in {-1, 0, 1, 2}.
            for dw in qd.static(range(-1, 3)):
                i_wc = i_w - dw
                if 1 <= i_wc < n_knots - 2:
                    jerk = (
                        plan_state.qpos_traj[i_dp, i_w0 + i_wc + 2]
                        - 3.0 * plan_state.qpos_traj[i_dp, i_w0 + i_wc + 1]
                        + 3.0 * plan_state.qpos_traj[i_dp, i_w0 + i_wc]
                        - plan_state.qpos_traj[i_dp, i_w0 + i_wc - 1]
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
                        cost += plan_info.w_jerk[None] * jerk**2
                    plan_state.grad_traj[i_dp, i_cw] += 2.0 * plan_info.w_jerk[None] * jerk * coeff
            # Joint-limit quadratic hinge.
            over = q - plan_info.q_limit_upper[i_dp]
            under = plan_info.q_limit_lower[i_dp] - q
            if over > 0.0:
                cost += plan_info.w_lim[None] * over**2
                plan_state.grad_traj[i_dp, i_cw] += 2.0 * plan_info.w_lim[None] * over
            if under > 0.0:
                cost += plan_info.w_lim[None] * under**2
                plan_state.grad_traj[i_dp, i_cw] -= 2.0 * plan_info.w_lim[None] * under
            # Posture regularizer toward the straight-line reference (kept tiny; resolves redundancy).
            ref = plan_info.qpos_start[i_dp, i_b] + (
                plan_info.qpos_goal[i_dp, i_b] - plan_info.qpos_start[i_dp, i_b]
            ) * (qd.cast(i_w, gs.qd_float) / float(n_knots - 1))
            cost += plan_info.w_posture[None] * (q - ref) ** 2
            plan_state.grad_traj[i_dp, i_cw] += 2.0 * plan_info.w_posture[None] * (q - ref)

    # Terminal pose cost for Cartesian goals (last knot only), gradient through the ee chain.
    if plan_info.has_pose_goal[None] and i_w == n_knots - 1:
        i_l = plan_info.goal_link_idx[None] - qd.static(planner_config.link_offset)
        err_pos = plan_state.links_pos[i_l, i_cw] - plan_info.goal_pos[i_b]
        quat_rel = gu.qd_transform_quat_by_quat(
            plan_state.links_quat[i_l, i_cw], gu.qd_inv_quat(plan_info.goal_quat[i_b])
        )
        err_rot = gu.qd_quat_to_rotvec(quat_rel, rigid_info.EPS[None])
        cost += plan_info.w_pose_pos[None] * err_pos.norm_sqr()
        cost += plan_info.w_pose_rot[None] * err_rot.norm_sqr()
        # Position term: gradient 2 w (x - x*) applied at the link origin.
        func_planner_chain_grad(
            i_l,
            i_c,
            i_cw,
            i_b,
            plan_state.links_pos[i_l, i_cw],
            err_pos,
            2.0 * plan_info.w_pose_pos[None],
            joints_xanchor=plan_state.joints_xanchor,
            joints_xaxis=plan_state.joints_xaxis,
            grad_traj=plan_state.grad_traj,
            i_col_grad=i_cw,
            plan_info=plan_info,
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
                        if not plan_info.dof_is_locked[i_dp, i_b]:
                            plan_state.grad_traj[i_dp, i_cw] += (
                                2.0
                                * plan_info.w_pose_rot[None]
                                * plan_state.joints_xaxis[i_j - qd.static(planner_config.joint_offset), i_cw].dot(
                                    err_rot
                                )
                            )
                i_l_glob = dyn_info.links.parent_idx[I_l]

    plan_state.cost_wp[i_cw] = cost


@qd.kernel
def kernel_planner_cost_grad(
    envs_idx: qd.types.ndarray(),
    plan_state: array_class.PlannerState,
    plan_info: array_class.PlannerEntityInfo,
    plan_world: array_class.PlannerWorldState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    sdf_info: array_class.SDFInfo,
    rigid_config: qd.template(),
    collider_static_config: qd.template(),
    planner_config: qd.template(),
):
    """Cost and gradient of every active (candidate, knot) column, one thread per column."""
    n_knots = qd.static(planner_config.n_knots)
    n_seeds = qd.static(planner_config.n_seeds)

    qd.loop_config(serialize=qd.static(planner_config.para_level < gs.PARA_LEVEL.ALL))
    for i_cw in range(envs_idx.shape[0] * n_seeds * n_knots):
        i_c = i_cw // n_knots
        if plan_state.is_active[i_c]:
            func_planner_knot_cost_grad(
                i_c,
                i_cw - i_c * n_knots,
                i_cw,
                envs_idx[i_c // n_seeds],
                plan_state=plan_state,
                plan_info=plan_info,
                plan_world=plan_world,
                dyn_info=dyn_info,
                rigid_info=rigid_info,
                sdf_info=sdf_info,
                rigid_config=rigid_config,
                collider_static_config=collider_static_config,
                planner_config=planner_config,
            )


@qd.kernel
def kernel_planner_reduce_cost(
    envs_idx: qd.types.ndarray(),
    plan_state: array_class.PlannerState,
    planner_config: qd.template(),
):
    """Serial per-candidate sum of the per-knot costs (deterministic reduction order)."""
    n_knots = qd.static(planner_config.n_knots)
    n_seeds = qd.static(planner_config.n_seeds)

    qd.loop_config(serialize=qd.static(planner_config.para_level < gs.PARA_LEVEL.ALL))
    for i_c in range(envs_idx.shape[0] * n_seeds):
        if plan_state.is_active[i_c]:
            cost = gs.qd_float(0.0)
            for i_w in range(n_knots):
                cost += plan_state.cost_wp[i_c * n_knots + i_w]
            plan_state.cost[i_c] = cost


@qd.kernel
def kernel_planner_validate(
    envs_idx: qd.types.ndarray(),
    plan_state: array_class.PlannerState,
    plan_info: array_class.PlannerEntityInfo,
    plan_world: array_class.PlannerWorldState,
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    sdf_info: array_class.SDFInfo,
    rigid_config: qd.template(),
    planner_config: qd.template(),
):
    """
    Certify every active candidate independently of the optimizer costs: joint limits and collision clearance
    checked at _VALIDATE_UPSAMPLE x knot density, each sample carrying the swept allowance from the per-DOF
    Lipschitz reach bounds so the whole continuous path is covered, plus the goal tolerance. Writes the per-
    candidate flags bitfield and min clearance; one thread per candidate, serial inside (deterministic).
    """
    n_knots = qd.static(planner_config.n_knots)
    n_seeds = qd.static(planner_config.n_seeds)
    n_dp = qd.static(planner_config.n_dp)
    n_samples = qd.static(_VALIDATE_UPSAMPLE * (planner_config.n_knots - 1) + 1)

    qd.loop_config(serialize=qd.static(planner_config.para_level < gs.PARA_LEVEL.ALL))
    for i_c in range(envs_idx.shape[0] * n_seeds):
        if plan_state.is_active[i_c]:
            i_b = envs_idx[i_c // n_seeds]
            i_e_col = i_c  # eval scratch column owned by this thread
            flags = gs.qd_int(0)
            min_clearance = gs.qd_float(qd.math.inf)

            for i_smp in range(n_samples):
                # Linear interpolation between knots at the densified sample, and the per-segment sweep bound.
                t = qd.cast(i_smp, gs.qd_float) / float(n_samples - 1) * float(n_knots - 1)
                i_w = qd.min(gs.qd_int(t), n_knots - 2)
                alpha = t - qd.cast(i_w, gs.qd_float)
                swp = gs.qd_float(0.0)
                dq_inf = gs.qd_float(0.0)
                for i_dp in range(n_dp):
                    q0 = plan_state.qpos_traj[i_dp, i_c * n_knots + i_w]
                    q1 = plan_state.qpos_traj[i_dp, i_c * n_knots + i_w + 1]
                    q = q0 + alpha * (q1 - q0)
                    plan_state.eval_qpos[i_dp, i_e_col] = q
                    # Half the inter-sample joint travel bounds the workspace motion of every sphere.
                    swp += 0.5 * plan_info.dof_reach[i_dp] * qd.abs(q1 - q0) / float(_VALIDATE_UPSAMPLE)
                    dq_inf = qd.max(dq_inf, qd.abs(q1 - q0) / float(_VALIDATE_UPSAMPLE))
                    # A start (or goal) configuration supplied beyond a limit is a boundary condition, so a knot
                    # is only flagged when it exceeds the limit by more than the boundary conditions already do.
                    over_allow = qd.max(
                        plan_info.qpos_start[i_dp, i_b] - plan_info.q_limit_upper[i_dp],
                        plan_info.qpos_goal[i_dp, i_b] - plan_info.q_limit_upper[i_dp],
                    )
                    under_allow = qd.max(
                        plan_info.q_limit_lower[i_dp] - plan_info.qpos_start[i_dp, i_b],
                        plan_info.q_limit_lower[i_dp] - plan_info.qpos_goal[i_dp, i_b],
                    )
                    if (
                        q > plan_info.q_limit_upper[i_dp] + qd.max(over_allow, 0.0) + 1e-5
                        or q < plan_info.q_limit_lower[i_dp] - qd.max(under_allow, 0.0) - 1e-5
                    ):
                        flags |= JOINT_LIMIT

                func_planner_fk(
                    i_e_col,
                    i_e_col,
                    i_b,
                    qpos=plan_state.eval_qpos,
                    links_pos=plan_state.eval_links_pos,
                    links_quat=plan_state.eval_links_quat,
                    joints_xanchor=plan_state.eval_joints_xanchor,
                    joints_xaxis=plan_state.eval_joints_xaxis,
                    dyn_state=dyn_state,
                    dyn_info=dyn_info,
                    rigid_info=rigid_info,
                    rigid_config=rigid_config,
                    planner_config=planner_config,
                )
                func_planner_spheres(
                    i_e_col,
                    i_b,
                    links_pos=plan_state.eval_links_pos,
                    links_quat=plan_state.eval_links_quat,
                    spheres_pos=plan_state.eval_spheres_pos,
                    plan_info=plan_info,
                    planner_config=planner_config,
                )
                cost, min_sd = func_planner_collision_cost(
                    i_e_col,
                    i_b,
                    swp,
                    dq_inf,
                    spheres_pos=plan_state.eval_spheres_pos,
                    plan_info=plan_info,
                    plan_world=plan_world,
                    dyn_info=dyn_info,
                    sdf_info=sdf_info,
                    planner_config=planner_config,
                )
                min_clearance = qd.min(min_clearance, min_sd)
                if min_sd < 0.0:
                    flags |= COLLISION

            # Goal reaching: joint goals must land on qpos_goal; Cartesian goals within the pose tolerance.
            if plan_info.has_pose_goal[None]:
                pose_cost = func_planner_pose_cost(
                    i_e_col,
                    i_b,
                    links_pos=plan_state.eval_links_pos,
                    links_quat=plan_state.eval_links_quat,
                    plan_info=plan_info,
                    rigid_info=rigid_info,
                    planner_config=planner_config,
                )
                i_l = plan_info.goal_link_idx[None] - qd.static(planner_config.link_offset)
                err_pos = plan_state.eval_links_pos[i_l, i_e_col] - plan_info.goal_pos[i_b]
                quat_rel = gu.qd_transform_quat_by_quat(
                    plan_state.eval_links_quat[i_l, i_e_col], gu.qd_inv_quat(plan_info.goal_quat[i_b])
                )
                err_rot = gu.qd_quat_to_rotvec(quat_rel, rigid_info.EPS[None])
                if err_pos.norm() > 5e-3 or err_rot.norm() > 5e-2:
                    flags |= GOAL_TOL
            else:
                for i_dp in range(n_dp):
                    q_end = plan_state.qpos_traj[i_dp, i_c * n_knots + n_knots - 1]
                    if qd.abs(q_end - plan_info.qpos_goal[i_dp, i_b]) > 1e-4:
                        flags |= GOAL_TOL

            plan_state.valid_flags[i_c] = flags
            plan_state.min_clearance[i_c] = min_clearance
