import numpy as np
import quadrants as qd

import genesis as gs
import genesis.utils.geom as gu
from genesis.utils import array_class

from .cost import func_collision_cost, func_knot_cost_gradient, func_pose_cost
from .kinematics import func_forward_kinematics, func_sphere_positions

# Fixed step-size ladder of the parallel noisy line search (jittered per candidate and iteration).
_LS_LADDER = (1.0, 0.5, 0.2, 0.08, 0.03, 0.01, 0.003, 0.001)
# MPPI noise annealing factor per iteration.
_MPPI_ANNEAL = 0.7


def build_noise_basis(n_knots):
    """Smooth MPPI noise basis: Gaussian bumps over the knot axis, one column per noise knot.

    Zero rows at the clamped boundary knots keep perturbations inside the boundary conditions. Deterministic pure
    numpy.
    """
    n_noise = array_class.PLANNER_N_NOISE_KNOTS
    t = np.linspace(0.0, 1.0, n_knots)
    centers = np.linspace(0.0, 1.0, n_noise + 2)[1:-1]
    width = 1.0 / (n_noise + 1)
    basis = np.exp(-0.5 * ((t[:, None] - centers[None, :]) / width) ** 2)
    basis[:2] = 0.0
    basis[-2:] = 0.0
    # Unit peak response per knot so mppi_sigma is expressed in joint units.
    scale = basis.sum(axis=1).max()
    return (basis / scale).astype(gs.np_float)


@qd.func
def func_trajectory_cost(
    i_c,
    i_b,
    i_lane,
    i_e_col,
    qpos_cols: qd.Tensor,
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
    is_tiled: qd.template(),
):
    """Total cost of one full trajectory read from qpos_cols.

    This is the cost-only evaluator of MPPI rollouts and line-search trials. The sweep allowance uses the joint-space
    Lipschitz reach bound, needing no neighbor forward kinematics, and is clamped like the gradient path.

    Under is_tiled, lane i_lane of n_cost_lanes takes every n_cost_lanes-th knot, evaluating it through its own
    scratch column, and the partial costs are summed across the lanes so every lane returns the whole trajectory's
    cost; the caller must have written the knots the lane reads - its own and their neighbors - and made them visible
    to the subgroup. A caller that is not tiled walks every knot in one lane.
    """
    n_knots = qd.static(planner_config.n_knots)
    n_dp = qd.static(planner_config.n_dp)
    n_lanes = qd.static(planner_config.n_cost_lanes if is_tiled else 1)
    n_knot_chunks = qd.static((planner_config.n_knots + n_lanes - 1) // n_lanes)
    cost = gs.qd_float(0.0)

    for i_knot_chunk in range(n_knot_chunks):
        i_w = i_knot_chunk * n_lanes + i_lane
        if i_w < n_knots:
            # Sweep allowance and L-inf joint travel toward both neighbors.
            swp = gs.qd_float(0.0)
            dq_inf = gs.qd_float(0.0)
            for i_dp in range(n_dp):
                q = qpos_cols[i_dp, i_c, i_w]
                planner_state.fk.eval.qpos[i_dp, i_e_col] = q
                if i_w > 0:
                    swp = qd.max(
                        swp, 0.5 * planner_info.fk.dofs.reach[i_dp] * qd.abs(q - qpos_cols[i_dp, i_c, i_w - 1])
                    )
                    dq_inf = qd.max(dq_inf, qd.abs(q - qpos_cols[i_dp, i_c, i_w - 1]))
                if i_w < n_knots - 1:
                    swp = qd.max(
                        swp, 0.5 * planner_info.fk.dofs.reach[i_dp] * qd.abs(qpos_cols[i_dp, i_c, i_w + 1] - q)
                    )
                    dq_inf = qd.max(dq_inf, qd.abs(qpos_cols[i_dp, i_c, i_w + 1] - q))
            swp = qd.min(swp, planner_info.cost.eps_act[None])

            func_forward_kinematics(
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
            func_sphere_positions(
                i_e_col,
                i_b,
                links_pos=planner_state.fk.eval.links_pos,
                links_quat=planner_state.fk.eval.links_quat,
                spheres_pos=planner_state.fk.eval.spheres_pos,
                planner_info=planner_info,
                planner_config=planner_config,
            )
            coll_cost, _, _ = func_collision_cost(
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
                use_exact=False,
            )
            cost += coll_cost

            # Smoothness / limit / posture terms, identical to the gradient path (see func_knot_cost_gradient).
            for i_dp in range(n_dp):
                q = qpos_cols[i_dp, i_c, i_w]
                if 1 <= i_w < n_knots - 1:
                    acc = qpos_cols[i_dp, i_c, i_w - 1] - 2.0 * q + qpos_cols[i_dp, i_c, i_w + 1]
                    cost += planner_info.cost.w_acc[None] * acc**2
                if 1 <= i_w < n_knots - 2:
                    jerk = (
                        qpos_cols[i_dp, i_c, i_w + 2]
                        - 3.0 * qpos_cols[i_dp, i_c, i_w + 1]
                        + 3.0 * q
                        - qpos_cols[i_dp, i_c, i_w - 1]
                    )
                    cost += planner_info.cost.w_jerk[None] * jerk**2
                over = q - planner_info.fk.dofs.q_limit_upper[i_dp]
                under = planner_info.fk.dofs.q_limit_lower[i_dp] - q
                if over > 0.0:
                    cost += planner_info.cost.w_lim[None] * over**2
                if under > 0.0:
                    cost += planner_info.cost.w_lim[None] * under**2
                ref = planner_info.cost.boundary.qpos_start[i_dp, i_b] + (
                    planner_info.cost.boundary.qpos_goal[i_dp, i_b] - planner_info.cost.boundary.qpos_start[i_dp, i_b]
                ) * (qd.cast(i_w, gs.qd_float) / float(n_knots - 1))
                cost += planner_info.cost.w_posture[None] * (q - ref) ** 2

            if planner_info.cost.boundary.has_pose_goal[None] and i_w == n_knots - 1:
                cost += func_pose_cost(
                    i_e_col,
                    i_b,
                    links_pos=planner_state.fk.eval.links_pos,
                    links_quat=planner_state.fk.eval.links_quat,
                    planner_info=planner_info,
                    rigid_info=rigid_info,
                    planner_config=planner_config,
                )
    # Sum the lanes' partial costs. The butterfly leaves the total in every lane, so each one can drive the
    # candidate's serial follow-up without a broadcast. The subgroup intrinsics exist only on the GPU backends, so
    # the single-lane arm must not emit this at all.
    if qd.static(is_tiled and planner_config.n_cost_lanes > 1):
        cost = qd.simt.subgroup.reduce_all_add_tiled(cost, qd.static(planner_config.n_cost_lanes.bit_length() - 1))
    return cost


@qd.func
def func_candidate_cost_gradient(
    i_c,
    i_b,
    planner_state: array_class.PlannerState,
    planner_world: array_class.PlannerWorldState,
    dyn_state: array_class.DynState,
    planner_info: array_class.PlannerEntityInfo,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    sdf_info: array_class.SDFInfo,
    rigid_config: qd.template(),
    collider_static_config: qd.template(),
    planner_config: qd.template(),
):
    """Forward-kinematics cache, cost and gradient of one candidate's whole trajectory, serial over its knots.

    Returns the cost. Each knot is staged through the candidate's evaluation column before the forward kinematics
    reads it, which is what lets the trajectory carry the optimizer's (candidate, knot) layout while the shared
    kinematics keeps its single-column addressing.
    """
    n_knots = qd.static(planner_config.n_knots)
    n_dp = qd.static(planner_config.n_dp)
    for i_w in range(n_knots):
        i_cw = i_c * n_knots + i_w
        for i_dp in range(n_dp):
            planner_state.fk.eval.qpos[i_dp, i_c] = planner_state.cost.qpos[i_dp, i_c, i_w]
        func_forward_kinematics(
            i_cw,
            i_c,
            i_b,
            qpos=planner_state.fk.eval.qpos,
            links_pos=planner_state.fk.links_pos,
            links_quat=planner_state.fk.links_quat,
            joints_xanchor=planner_state.fk.joints_xanchor,
            joints_xaxis=planner_state.fk.joints_xaxis,
            dyn_state=dyn_state,
            dyn_info=dyn_info,
            rigid_info=rigid_info,
            rigid_config=rigid_config,
            planner_config=planner_config,
        )
        func_sphere_positions(
            i_cw,
            i_b,
            links_pos=planner_state.fk.links_pos,
            links_quat=planner_state.fk.links_quat,
            spheres_pos=planner_state.fk.spheres_pos,
            planner_info=planner_info,
            planner_config=planner_config,
        )
    cost = gs.qd_float(0.0)
    for i_w in range(n_knots):
        i_cw = i_c * n_knots + i_w
        func_knot_cost_gradient(
            i_c,
            i_w,
            i_cw,
            i_b,
            planner_state=planner_state,
            planner_world=planner_world,
            planner_info=planner_info,
            dyn_info=dyn_info,
            rigid_info=rigid_info,
            sdf_info=sdf_info,
            rigid_config=rigid_config,
            collider_static_config=collider_static_config,
            planner_config=planner_config,
        )
        cost += planner_state.cost.cost_wp[i_c, i_w]
    return cost


@qd.func
def func_is_knot_clamped(i_c, i_w, i_b, planner_info: array_class.PlannerEntityInfo, planner_config: qd.template()):
    """True when knot i_w is clamped, which the start and goal knot pairs are.

    A Cartesian goal clamps to its inverse-kinematics solution, whose residual is far below the goal tolerance.
    """
    return i_w <= 1 or i_w >= qd.static(planner_config.n_knots) - 2


@qd.func
def func_seed_trajectories(
    graph_counter: qd.types.ndarray(),
    envs_idx: qd.types.ndarray(),
    planner_state: array_class.PlannerState,
    planner_info: array_class.PlannerEntityInfo,
    planner_config: qd.template(),
):
    """Seed each not-yet-seeded env's candidate trajectories with straight start-to-goal lines.

    Seed 0 is the straight start-to-goal line; the others add smooth basis-weighted noise drawn from the
    counter-based hash keyed by the attempt (the ladder counter), so distinct candidates explore distinct basins.
    Locked DOFs hold the start value, interior knots clamp to the joint limits, and the two boundary knot pairs
    pin to start and goal. Only envs the ladder has not seeded yet are touched: an env is seeded once from
    straight lines, then escalates to RRT-Connect seeds on later passes, so this leaves already-escalated envs
    untouched.
    """
    n_knots = qd.static(planner_config.n_knots)
    n_seeds = qd.static(planner_config.n_seeds)
    n_dp = qd.static(planner_config.n_dp)
    n_noise = qd.static(planner_config.n_noise_knots)
    attempt = graph_counter[()]

    qd.loop_config(serialize=qd.static(planner_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_c in range(envs_idx.shape[0] * n_seeds):
        i_b = envs_idx[i_c // n_seeds]
        if not planner_state.is_env_solved[i_b] and not planner_state.is_env_seeded[i_b]:
            i_s = i_c % n_seeds
            for i_w in range(n_knots):
                alpha = qd.cast(i_w, gs.qd_float) / (n_knots - 1)
                for i_dp in range(n_dp):
                    q_start = planner_info.cost.boundary.qpos_start[i_dp, i_b]
                    q_goal = planner_info.cost.boundary.qpos_goal[i_dp, i_b]
                    q = q_start
                    if i_w >= n_knots - 2:
                        q = q_goal
                    elif i_w > 1:
                        q = q_start * (1.0 - alpha) + q_goal * alpha
                        if i_s > 0 and not planner_info.fk.dofs.is_locked[i_dp, i_b]:
                            delta = gs.qd_float(0.0)
                            for i_k in range(n_noise):
                                key = i_k * n_dp + i_dp
                                delta += planner_info.mppi.noise_basis[i_w, i_k] * gu.qd_hash_gauss(
                                    planner_info.mppi.seed_key[None], i_c, key, attempt
                                )
                            q += 0.5 * delta
                        q = qd.math.clamp(
                            q, planner_info.fk.dofs.q_limit_lower[i_dp], planner_info.fk.dofs.q_limit_upper[i_dp]
                        )
                    planner_state.cost.qpos[i_dp, i_c, i_w] = q
            planner_state.cert.is_active[i_c] = True


@qd.func
def func_seed_trajectories_from_rrt(
    envs_idx: qd.types.ndarray(),
    planner_state: array_class.PlannerState,
    planner_config: qd.template(),
):
    """Seed each escalating env's candidates from its connected RRT-Connect trees.

    Every connected tree's certified polyline seeds its own candidate column (cycling when fewer trees than
    columns), so distinct corridors seed distinct candidates. A polyline with no more vertices than knots keeps
    every vertex as a knot and holds the goal on the trailing knots, so consecutive knots stay on the certified
    polyline; a longer polyline falls back to an arclength resample, whose corner cutting the refiner repairs.
    Seed 0 stays unrefined as an insurance candidate the certifier still checks. Envs with no connected tree keep
    their previous candidates untouched.
    """
    n_knots = qd.static(planner_config.n_knots)
    n_seeds = qd.static(planner_config.n_seeds)
    n_dp = qd.static(planner_config.n_dp)
    n_trees = qd.static(planner_config.n_rrt_trees)
    n_nodes = qd.static(planner_config.n_rrt_nodes)

    qd.loop_config(serialize=qd.static(planner_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_b_ in range(envs_idx.shape[0]):
        i_b = envs_idx[i_b_]
        if not planner_state.is_env_solved[i_b] and planner_state.is_env_seeded[i_b]:
            n_conn = 0
            for i_t_ in range(n_trees):
                i_t = i_b_ * n_trees + i_t_
                if planner_state.rrt.is_done[i_t] and planner_state.rrt.path_len[i_t] >= 2:
                    n_conn += 1
            if n_conn > 0:
                for i_s in range(n_seeds):
                    # The (i_s mod n_conn)-th connected tree seeds column i_s.
                    target = i_s % n_conn
                    i_t_sel = i_b_ * n_trees
                    seen = 0
                    for i_t_ in range(n_trees):
                        i_t = i_b_ * n_trees + i_t_
                        if planner_state.rrt.is_done[i_t] and planner_state.rrt.path_len[i_t] >= 2:
                            if seen == target:
                                i_t_sel = i_t
                            seen += 1
                    n_path = planner_state.rrt.path_len[i_t_sel]
                    rrt_col0 = i_t_sel * n_nodes
                    i_c = i_b_ * n_seeds + i_s
                    if n_path <= n_knots:
                        # Vertex-preserving subdivision: every vertex is a knot and the leftover knots subdivide
                        # the segments proportionally to their length, so consecutive knots stay on the certified
                        # polyline AND no segment stays long enough for the validator's swept allowance to reject
                        # it. Counts are placed by a running proportional accumulator (no per-segment scratch);
                        # the last segment absorbs the rounding remainder so exactly n_knots knots are written.
                        total = gs.qd_float(0.0)
                        for i_seg in range(n_path - 1):
                            d = gs.qd_float(0.0)
                            for i_dp in range(n_dp):
                                diff = (
                                    planner_state.rrt.path[i_dp, rrt_col0 + i_seg + 1]
                                    - planner_state.rrt.path[i_dp, rrt_col0 + i_seg]
                                )
                                d += diff * diff
                            total += qd.sqrt(d)
                        extra = n_knots - n_path
                        for i_dp in range(n_dp):
                            planner_state.cost.qpos[i_dp, i_c, 0] = planner_state.rrt.path[i_dp, rrt_col0]
                        i_w = 1
                        placed = 0
                        acc = gs.qd_float(0.0)
                        for i_seg in range(n_path - 1):
                            d = gs.qd_float(0.0)
                            for i_dp in range(n_dp):
                                diff = (
                                    planner_state.rrt.path[i_dp, rrt_col0 + i_seg + 1]
                                    - planner_state.rrt.path[i_dp, rrt_col0 + i_seg]
                                )
                                d += diff * diff
                            acc += qd.sqrt(d) * qd.cast(extra, gs.qd_float) / qd.max(total, 1e-9)
                            n_sub = gs.qd_int(qd.floor(acc)) - placed
                            if i_seg == n_path - 2:
                                n_sub = extra - placed
                            placed += n_sub
                            for k in range(n_sub):
                                u = qd.cast(k + 1, gs.qd_float) / qd.cast(n_sub + 1, gs.qd_float)
                                for i_dp in range(n_dp):
                                    q_a = planner_state.rrt.path[i_dp, rrt_col0 + i_seg]
                                    q_b = planner_state.rrt.path[i_dp, rrt_col0 + i_seg + 1]
                                    planner_state.cost.qpos[i_dp, i_c, i_w] = q_a * (1.0 - u) + q_b * u
                                i_w += 1
                            for i_dp in range(n_dp):
                                planner_state.cost.qpos[i_dp, i_c, i_w] = planner_state.rrt.path[
                                    i_dp, rrt_col0 + i_seg + 1
                                ]
                            i_w += 1
                    else:
                        # Arclength resample: total polyline length in one pass, then a single walk placing the
                        # interior knots; the endpoints pin to the polyline start and goal.
                        total = gs.qd_float(0.0)
                        for i_seg in range(n_path - 1):
                            d = gs.qd_float(0.0)
                            for i_dp in range(n_dp):
                                diff = (
                                    planner_state.rrt.path[i_dp, rrt_col0 + i_seg + 1]
                                    - planner_state.rrt.path[i_dp, rrt_col0 + i_seg]
                                )
                                d += diff * diff
                            total += qd.sqrt(d)
                        for i_dp in range(n_dp):
                            planner_state.cost.qpos[i_dp, i_c, 0] = planner_state.rrt.path[i_dp, rrt_col0]
                            planner_state.cost.qpos[i_dp, i_c, n_knots - 1] = planner_state.rrt.path[
                                i_dp, rrt_col0 + n_path - 1
                            ]
                        i_seg = 0
                        s_cum = gs.qd_float(0.0)
                        seg_len = gs.qd_float(0.0)
                        for i_w in range(1, n_knots - 1):
                            s_tgt = total * qd.cast(i_w, gs.qd_float) / (n_knots - 1)
                            advancing = True
                            while advancing:
                                seg_len = gs.qd_float(0.0)
                                for i_dp in range(n_dp):
                                    diff = (
                                        planner_state.rrt.path[i_dp, rrt_col0 + i_seg + 1]
                                        - planner_state.rrt.path[i_dp, rrt_col0 + i_seg]
                                    )
                                    seg_len += diff * diff
                                seg_len = qd.sqrt(seg_len)
                                if i_seg < n_path - 2 and s_cum + seg_len < s_tgt:
                                    s_cum += seg_len
                                    i_seg += 1
                                else:
                                    advancing = False
                            u = (s_tgt - s_cum) / qd.max(seg_len, 1e-9)
                            for i_dp in range(n_dp):
                                q_a = planner_state.rrt.path[i_dp, rrt_col0 + i_seg]
                                q_b = planner_state.rrt.path[i_dp, rrt_col0 + i_seg + 1]
                                planner_state.cost.qpos[i_dp, i_c, i_w] = q_a * (1.0 - u) + q_b * u
                    planner_state.cert.is_active[i_b_ * n_seeds + i_s] = True
                # Seed 0 keeps the raw fallback polyline unrefined, so refinement can only add better candidates.
                planner_state.cert.is_active[i_b_ * n_seeds] = False
                planner_state.cost.cost[i_b_ * n_seeds] = 1e30


@qd.func
def func_mppi(
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
):
    """Iteration-resident MPPI warmup: every candidate refines its mean trajectory through annealed rollouts.

    A candidate advances through annealed smooth-noise particle rollouts and softmax-weighted updates in one kernel
    launch per phase, deterministic per backend since all randomness comes from the counter-based hash keyed by
    content indices. Particle noise is regenerated on demand instead of stored: pass one scores the particles, pass
    two rebuilds the winning mixture.

    A candidate is worked by n_cost_lanes lanes of one subgroup, each owning every n_cost_lanes-th knot; the lanes
    barrier after writing a rollout so each of them sees its knots' neighbors, and the trajectory cost sums across
    them, leaving every lane with the same scores and so the same serial softmax. With one lane this is one thread
    per candidate throughout.
    """
    n_knots = qd.static(planner_config.n_knots)
    n_seeds = qd.static(planner_config.n_seeds)
    n_dp = qd.static(planner_config.n_dp)
    n_noise = qd.static(planner_config.n_noise_knots)
    n_particles_max = qd.static(planner_config.n_mppi_particles_max)
    n_lanes = qd.static(planner_config.n_cost_lanes)
    n_knot_chunks = qd.static((planner_config.n_knots + planner_config.n_cost_lanes - 1) // planner_config.n_cost_lanes)

    qd.loop_config(serialize=qd.static(planner_config.para_level < gs.PARA_LEVEL.PARTIAL), block_dim=qd.static(n_lanes))
    for i_cl in range(envs_idx.shape[0] * n_seeds * n_lanes):
        i_lane = i_cl % n_lanes
        i_c = i_cl // n_lanes
        # Uniform across the candidate's lanes, so the barriers and the cross-lane sum below stay convergent.
        if planner_state.cert.is_active[i_c]:
            i_b = envs_idx[i_c // n_seeds]
            n_particles = qd.min(planner_info.mppi.n_particles[None], n_particles_max)
            costs = qd.Vector.zero(gs.qd_float, n_particles_max)

            for it in range(planner_info.mppi.n_iters[None]):
                anneal = planner_info.mppi.anneal[None] ** it
                # Pass 1: score every particle (particle 0 is the unperturbed mean).
                for i_p in range(n_particles):
                    for i_dp in range(n_dp):
                        # A particle's noise coefficients are keyed by DOF and basis knot, never by trajectory
                        # knot: the perturbation of the whole trajectory is one draw per DOF projected through the
                        # basis. Drawing them per DOF instead of per knot is what keeps the Box-Muller transcen-
                        # dentals off the knot loop, which otherwise redraws identical values for every knot.
                        coeffs = qd.Vector.zero(gs.qd_float, n_noise)
                        if i_p > 0 and not planner_info.fk.dofs.is_locked[i_dp, i_b]:
                            for i_k in range(n_noise):
                                key = ((it * n_particles_max + i_p) * n_noise + i_k) * n_dp + i_dp
                                coeffs[i_k] = gu.qd_hash_gauss(planner_info.mppi.seed_key[None], i_c, key, 0)
                        for i_knot_chunk in range(n_knot_chunks):
                            i_w = i_knot_chunk * n_lanes + i_lane
                            if i_w < n_knots:
                                q = planner_state.cost.qpos[i_dp, i_c, i_w]
                                if not func_is_knot_clamped(i_c, i_w, i_b, planner_info, planner_config):
                                    delta = gs.qd_float(0.0)
                                    for i_k in range(n_noise):
                                        delta += planner_info.mppi.noise_basis[i_w, i_k] * coeffs[i_k]
                                    delta *= anneal * planner_info.mppi.sigma[i_dp]
                                    q = qd.math.clamp(
                                        q + delta,
                                        planner_info.fk.dofs.q_limit_lower[i_dp],
                                        planner_info.fk.dofs.q_limit_upper[i_dp],
                                    )
                                planner_state.lbfgs.trial_qpos[i_dp, i_c, i_w] = q
                    # The cost of a lane's knot reads its neighbors, which other lanes wrote.
                    if qd.static(n_lanes > 1):
                        qd.simt.block.sync()
                    costs[i_p] = func_trajectory_cost(
                        i_c,
                        i_b,
                        i_lane,
                        i_c * n_lanes + i_lane,
                        qpos_cols=planner_state.lbfgs.trial_qpos,
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
                        is_tiled=True,
                    )

                # Softmax weights with a median-scaled temperature, run identically by every lane of the candidate.
                cost_min = costs[0]
                cost_mean = gs.qd_float(0.0)
                for i_p in range(n_particles):
                    cost_min = qd.min(cost_min, costs[i_p])
                    cost_mean += costs[i_p]
                cost_mean /= qd.cast(n_particles, gs.qd_float)
                beta = 0.5 * (cost_mean - cost_min) + 1e-6
                weight_sum = gs.qd_float(0.0)
                exps = qd.Vector.zero(gs.qd_float, n_particles_max)
                for i_p in range(n_particles):
                    exps[i_p] = qd.exp(-(costs[i_p] - cost_min) / beta)
                    weight_sum += exps[i_p]

                # Pass 2: rebuild the weighted noise mixture and fold it into the mean. The mixture accumulates
                # per knot so the DOF and its particles stay the outer loops, sharing one draw of the coefficients
                # (see pass 1) and one weight evaluation across the trajectory.
                for i_dp in range(n_dp):
                    if not planner_info.fk.dofs.is_locked[i_dp, i_b]:
                        delta_mean = qd.Vector.zero(gs.qd_float, n_knots)
                        for i_p in range(1, n_particles):
                            weight = exps[i_p] / weight_sum
                            coeffs = qd.Vector.zero(gs.qd_float, n_noise)
                            for i_k in range(n_noise):
                                key = ((it * n_particles_max + i_p) * n_noise + i_k) * n_dp + i_dp
                                coeffs[i_k] = gu.qd_hash_gauss(planner_info.mppi.seed_key[None], i_c, key, 0)
                            for i_knot_chunk in range(n_knot_chunks):
                                i_w = i_knot_chunk * n_lanes + i_lane
                                if i_w < n_knots and not func_is_knot_clamped(
                                    i_c, i_w, i_b, planner_info, planner_config
                                ):
                                    delta = gs.qd_float(0.0)
                                    for i_k in range(n_noise):
                                        delta += planner_info.mppi.noise_basis[i_w, i_k] * coeffs[i_k]
                                    delta_mean[i_w] += weight * delta * anneal * planner_info.mppi.sigma[i_dp]
                        for i_knot_chunk in range(n_knot_chunks):
                            i_w = i_knot_chunk * n_lanes + i_lane
                            if i_w < n_knots and not func_is_knot_clamped(i_c, i_w, i_b, planner_info, planner_config):
                                planner_state.cost.qpos[i_dp, i_c, i_w] = qd.math.clamp(
                                    planner_state.cost.qpos[i_dp, i_c, i_w] + delta_mean[i_w],
                                    planner_info.fk.dofs.q_limit_lower[i_dp],
                                    planner_info.fk.dofs.q_limit_upper[i_dp],
                                )
                # The next iteration's rollouts read every knot of the mean, so the lanes' updates must land first.
                if qd.static(n_lanes > 1):
                    qd.simt.block.sync()


@qd.func
def func_lbfgs(
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
):
    """Iteration-resident limited-memory Broyden-Fletcher-Goldfarb-Shanno (L-BFGS) refinement of every candidate.

    The whole two-loop recursion and step-ladder line search runs inside the candidate's own thread, serial and so
    deterministic. Clamped knots and locked DOFs never move, their gradient entries being zeroed.
    """
    n_knots = qd.static(planner_config.n_knots)
    n_seeds = qd.static(planner_config.n_seeds)
    n_dp = qd.static(planner_config.n_dp)
    m_hist = qd.static(planner_config.n_lbfgs_hist)

    qd.loop_config(serialize=qd.static(planner_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_c in range(envs_idx.shape[0] * n_seeds):
        if planner_state.cert.is_active[i_c]:
            i_b = envs_idx[i_c // n_seeds]

            cost = func_candidate_cost_gradient(
                i_c,
                i_b,
                planner_state=planner_state,
                planner_world=planner_world,
                dyn_state=dyn_state,
                planner_info=planner_info,
                dyn_info=dyn_info,
                rigid_info=rigid_info,
                sdf_info=sdf_info,
                rigid_config=rigid_config,
                collider_static_config=collider_static_config,
                planner_config=planner_config,
            )
            # Zero the gradient of clamped knots so the whole optimizer sees the reduced problem.
            for i_w in range(n_knots):
                if func_is_knot_clamped(i_c, i_w, i_b, planner_info, planner_config):
                    for i_dp in range(n_dp):
                        planner_state.cost.grad[i_dp, i_c, i_w] = 0.0

            n_hist = 0
            i_next = 0
            n_stall = 0
            it = 0
            while it < planner_info.lbfgs.n_iters[None] and n_stall < 4:
                # Two-loop recursion over the ring-buffer history -> descent direction in dir_traj.
                for i_w in range(n_knots):
                    for i_dp in range(n_dp):
                        planner_state.lbfgs.dir_traj[i_dp, i_c, i_w] = planner_state.cost.grad[i_dp, i_c, i_w]
                alphas = qd.Vector.zero(gs.qd_float, m_hist)
                for i_h_ in range(n_hist):
                    i_h = (i_next - 1 - i_h_) % m_hist
                    dot_sd = gs.qd_float(0.0)
                    for i_w in range(n_knots):
                        for i_dp in range(n_dp):
                            dot_sd += (
                                planner_state.lbfgs.dqpos_hist[i_h, i_dp, i_c, i_w]
                                * planner_state.lbfgs.dir_traj[i_dp, i_c, i_w]
                            )
                    alpha_h = planner_state.lbfgs.rho_hist[i_h, i_c] * dot_sd
                    alphas[i_h] = alpha_h
                    for i_w in range(n_knots):
                        for i_dp in range(n_dp):
                            planner_state.lbfgs.dir_traj[i_dp, i_c, i_w] -= (
                                alpha_h * planner_state.lbfgs.dgrad_hist[i_h, i_dp, i_c, i_w]
                            )
                # Fresh history: scale steepest descent so the ladder's unit step moves the largest joint by
                # one radian - the raw gradient magnitude is arbitrary and would overshoot every trial.
                if n_hist == 0:
                    dir_inf = gs.qd_float(0.0)
                    for i_w in range(n_knots):
                        for i_dp in range(n_dp):
                            dir_inf = qd.max(dir_inf, qd.abs(planner_state.lbfgs.dir_traj[i_dp, i_c, i_w]))
                    scale_sd = 1.0 / qd.max(dir_inf, 1e-9)
                    for i_w in range(n_knots):
                        for i_dp in range(n_dp):
                            planner_state.lbfgs.dir_traj[i_dp, i_c, i_w] *= scale_sd
                # Initial scaling gamma = (s.y) / (y.y) of the most recent pair.
                if n_hist > 0:
                    i_h = (i_next - 1) % m_hist
                    dot_sy = gs.qd_float(0.0)
                    dot_yy = gs.qd_float(0.0)
                    for i_w in range(n_knots):
                        for i_dp in range(n_dp):
                            dot_sy += (
                                planner_state.lbfgs.dqpos_hist[i_h, i_dp, i_c, i_w]
                                * planner_state.lbfgs.dgrad_hist[i_h, i_dp, i_c, i_w]
                            )
                            dot_yy += planner_state.lbfgs.dgrad_hist[i_h, i_dp, i_c, i_w] ** 2
                    gamma = qd.math.clamp(dot_sy / qd.max(dot_yy, 1e-12), 1e-4, 1e2)
                    for i_w in range(n_knots):
                        for i_dp in range(n_dp):
                            planner_state.lbfgs.dir_traj[i_dp, i_c, i_w] *= gamma
                for i_h_ in range(n_hist):
                    i_h = (i_next - n_hist + i_h_) % m_hist
                    dot_yd = gs.qd_float(0.0)
                    for i_w in range(n_knots):
                        for i_dp in range(n_dp):
                            dot_yd += (
                                planner_state.lbfgs.dgrad_hist[i_h, i_dp, i_c, i_w]
                                * planner_state.lbfgs.dir_traj[i_dp, i_c, i_w]
                            )
                    beta_h = planner_state.lbfgs.rho_hist[i_h, i_c] * dot_yd
                    for i_w in range(n_knots):
                        for i_dp in range(n_dp):
                            planner_state.lbfgs.dir_traj[i_dp, i_c, i_w] += (
                                alphas[i_h] - beta_h
                            ) * planner_state.lbfgs.dqpos_hist[i_h, i_dp, i_c, i_w]

                # Step-ladder line search on trial trajectories, jittered per (candidate, iteration).
                jitter = 0.8 + 0.4 * gu.qd_hash01(planner_info.mppi.seed_key[None], i_c, it, 12345)
                cost_best = cost
                alpha_best = gs.qd_float(0.0)
                for i_t in range(
                    qd.min(planner_info.lbfgs.n_ls_trials[None], qd.static(planner_config.n_ls_trials_max))
                ):
                    alpha = planner_info.lbfgs.ls_ladder[i_t] * jitter
                    for i_w in range(n_knots):
                        is_clamped = func_is_knot_clamped(i_c, i_w, i_b, planner_info, planner_config)
                        for i_dp in range(n_dp):
                            q = planner_state.cost.qpos[i_dp, i_c, i_w]
                            if not is_clamped:
                                q = qd.math.clamp(
                                    q - alpha * planner_state.lbfgs.dir_traj[i_dp, i_c, i_w],
                                    planner_info.fk.dofs.q_limit_lower[i_dp],
                                    planner_info.fk.dofs.q_limit_upper[i_dp],
                                )
                            planner_state.lbfgs.trial_qpos[i_dp, i_c, i_w] = q
                    cost_trial = func_trajectory_cost(
                        i_c,
                        i_b,
                        0,
                        i_c * qd.static(planner_config.n_cost_lanes),
                        qpos_cols=planner_state.lbfgs.trial_qpos,
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
                        is_tiled=False,
                    )
                    if cost_trial < cost_best:
                        cost_best = cost_trial
                        alpha_best = alpha

                if alpha_best == 0.0:
                    # No improving step: drop the history (bad curvature model) before giving up.
                    n_stall += 1
                    n_hist = 0
                else:
                    n_stall = 0
                    # Accept the step; stage s = x_new - x_old into the history slot.
                    for i_w in range(n_knots):
                        is_clamped = func_is_knot_clamped(i_c, i_w, i_b, planner_info, planner_config)
                        for i_dp in range(n_dp):
                            q_old = planner_state.cost.qpos[i_dp, i_c, i_w]
                            planner_state.lbfgs.qpos_prev[i_dp, i_c, i_w] = q_old
                            planner_state.lbfgs.grad_prev[i_dp, i_c, i_w] = planner_state.cost.grad[i_dp, i_c, i_w]
                            if not is_clamped:
                                planner_state.cost.qpos[i_dp, i_c, i_w] = qd.math.clamp(
                                    q_old - alpha_best * planner_state.lbfgs.dir_traj[i_dp, i_c, i_w],
                                    planner_info.fk.dofs.q_limit_lower[i_dp],
                                    planner_info.fk.dofs.q_limit_upper[i_dp],
                                )
                    cost = func_candidate_cost_gradient(
                        i_c,
                        i_b,
                        planner_state=planner_state,
                        planner_world=planner_world,
                        dyn_state=dyn_state,
                        planner_info=planner_info,
                        dyn_info=dyn_info,
                        rigid_info=rigid_info,
                        sdf_info=sdf_info,
                        rigid_config=rigid_config,
                        collider_static_config=collider_static_config,
                        planner_config=planner_config,
                    )
                    for i_w in range(n_knots):
                        if func_is_knot_clamped(i_c, i_w, i_b, planner_info, planner_config):
                            for i_dp in range(n_dp):
                                planner_state.cost.grad[i_dp, i_c, i_w] = 0.0
                    # Curvature-guarded history push.
                    dot_sy = gs.qd_float(0.0)
                    for i_w in range(n_knots):
                        for i_dp in range(n_dp):
                            s_e = (
                                planner_state.cost.qpos[i_dp, i_c, i_w] - planner_state.lbfgs.qpos_prev[i_dp, i_c, i_w]
                            )
                            y_e = (
                                planner_state.cost.grad[i_dp, i_c, i_w] - planner_state.lbfgs.grad_prev[i_dp, i_c, i_w]
                            )
                            planner_state.lbfgs.dqpos_hist[i_next % m_hist, i_dp, i_c, i_w] = s_e
                            planner_state.lbfgs.dgrad_hist[i_next % m_hist, i_dp, i_c, i_w] = y_e
                            dot_sy += s_e * y_e
                    if dot_sy > 1e-10:
                        planner_state.lbfgs.rho_hist[i_next % m_hist, i_c] = 1.0 / dot_sy
                        i_next += 1
                        n_hist = qd.min(n_hist + 1, m_hist)
                it += 1

            planner_state.cost.cost[i_c] = cost


@qd.kernel
def kernel_lbfgs(
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
):
    func_lbfgs(
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
    )
