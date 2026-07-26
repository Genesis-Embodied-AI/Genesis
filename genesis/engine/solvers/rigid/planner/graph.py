import quadrants as qd

import genesis as gs
import genesis.utils.geom as gu
from genesis.utils import array_class

from .cost import func_collision_cost
from .kinematics import func_forward_kinematics, func_sphere_positions

# Joint-space steer step (L-inf, radians), the goal-bias probability of extending straight at the other tree,
# and the number of shortcut splices applied to an extracted path.
_RRT_STEER_STEP = 0.2
_RRT_GOAL_BIAS = 0.2
_RRT_N_SHORTCUT = 24
# Consecutive iterations without an accepted node on either side before a tree gives up its attempt: a walled
# tree burns its whole iteration budget on rejected extensions (each one paying certified edge checks), while
# later ladder attempts re-roll it with fresh sampling streams anyway - probabilistic completeness is preserved
# across the ladder, and growing trees are never cut.
_RRT_STALL_ITERS = 200


@qd.func
def func_rrt_config_is_free(
    i_t,
    i_lane,
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
    """Collision check of the configuration staged in the lane's eval column, with sweep allowance swp."""
    i_e_col = i_t * qd.static(planner_config.n_cost_lanes) + i_lane
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
    return qd.min(min_sd_exact, min_sd_proxy) >= 0.0


@qd.func
def func_rrt_edge_is_free(
    i_t,
    i_lane,
    i_b,
    col_from,
    col_to,
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
    """Certified collision check of the edge between two stored nodes.

    The segment is sampled densely enough that each sample's sweep allowance, from the per-DOF Lipschitz reach
    bounds, covers the whole inter-sample motion, so a passing edge is collision-free along its continuum, at the
    required clearance.
    """
    n_dp = qd.static(planner_config.n_dp)
    reach = gs.qd_float(0.0)
    for i_dp in range(n_dp):
        reach = reach + (
            planner_info.fk.dofs.reach[i_dp]
            * qd.abs(planner_state.rrt.qpos[i_dp, col_to] - planner_state.rrt.qpos[i_dp, col_from])
        )
    # Quarter-band granularity: each sample's sweep allowance stays ~eps_act/8, so edges certify while only
    # requiring modest true clearance (the demand IS the allowance - coarser sampling would reject any edge
    # whose clearance is below half the activation band).
    n_sub = gs.qd_int(qd.ceil(4.0 * reach / qd.max(planner_info.cost.eps_act[None], 1e-3))) + 1
    swp = 0.5 * reach / qd.cast(n_sub, gs.qd_float)
    dq_inf = gs.qd_float(0.0)
    for i_dp in range(n_dp):
        dq_inf = qd.max(
            dq_inf,
            qd.abs(planner_state.rrt.qpos[i_dp, col_to] - planner_state.rrt.qpos[i_dp, col_from])
            / qd.cast(n_sub, gs.qd_float),
        )

    # Lane i_lane takes a share of the segment's samples (see the visiting order below) and the lanes' verdicts
    # are combined, so an edge costs its samples spread across a subgroup rather than walked one by one. A lane
    # stops at the first blocked sample of its own share; the lanes agree only at the end, which is the trade that
    # buys the parallelism.
    n_lanes = qd.static(planner_config.n_cost_lanes)
    i_e_col = i_t * n_lanes + i_lane
    # A lane walks its share in one of two orders, and either way the visited set is the same {1..n_sub}, so the
    # verdict is the conjunction over identical samples and every edge decides identically - the choice only moves
    # when a rejected edge meets its blocker. Both live in one loop because the body inlines the whole collision
    # cost, so a second call site would duplicate it in every kernel that grows a tree.
    #   serial arm: bit-reversed, so a blocked stretch is met after a number of samples set by its width as a
    #     fraction of the edge rather than by the blocker's distance from the start. Long edges are the ones that
    #     reject, so they stop paying for their length; the first index visited is the far endpoint, which for a
    #     growth edge is the new node the check exists to qualify.
    #   lane arm: start to end, strided. A lane holds only n_sub/n_cost_lanes samples, too few for the reordering
    #     to save a pass over one, while its skipped indices - the bit-reversed walk covers a power-of-two range
    #     and drops what overshoots - cost more than they buy.
    n_steps = (n_sub + n_lanes - 1) // n_lanes
    n_bits = gs.qd_int(1)
    n_perm = gs.qd_int(2)
    if qd.static(n_lanes == 1):
        while n_perm < n_sub:
            n_bits = n_bits + 1
            n_perm = n_perm * 2
        n_steps = n_perm
    is_free_lane = True
    i_step = gs.qd_int(0)
    while is_free_lane and i_step < n_steps:
        i_sub = gs.qd_int(1 + i_lane) + i_step * n_lanes
        if qd.static(n_lanes == 1):
            i_sub = gs.qd_int(0)
            i_rev = i_step
            for _ in range(n_bits):
                i_sub = (i_sub << 1) | (i_rev & 1)
                i_rev = i_rev >> 1
            if i_sub == 0:
                i_sub = n_perm
        if i_sub <= n_sub:
            alpha = qd.cast(i_sub, gs.qd_float) / qd.cast(n_sub, gs.qd_float)
            for i_dp in range(n_dp):
                planner_state.fk.eval.qpos[i_dp, i_e_col] = planner_state.rrt.qpos[i_dp, col_from] + alpha * (
                    planner_state.rrt.qpos[i_dp, col_to] - planner_state.rrt.qpos[i_dp, col_from]
                )
            is_free_lane = func_rrt_config_is_free(
                i_t,
                i_lane,
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
        i_step = i_step + 1
    # An edge is free only if every lane's share was: a minimum over 0/1 is their conjunction.
    if qd.static(planner_config.n_cost_lanes > 1):
        return (
            qd.simt.subgroup.reduce_all_min_tiled(
                gs.qd_int(1) if is_free_lane else gs.qd_int(0),
                qd.static(planner_config.n_cost_lanes.bit_length() - 1),
            )
            == 1
        )
    return is_free_lane


@qd.func
def func_rrt_connect(
    envs_idx: qd.types.ndarray(),
    trees_is_active: qd.types.ndarray(),
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
    """Batched RRT-Connect seed generator, one thread per tree pair.

    A thread runs its whole pair (sample -> nearest -> steer -> certified edge -> insert -> connect) serially, so
    every accepted edge is certified and the result is deterministic per backend, all randomness coming from the
    counter-based hash. The start tree grows in
    node columns [0, n/2), the goal tree in [n/2, n); on success the bridge pair joins them and the path is
    backtracked into rrt_path, start to goal.
    """
    n_dp = qd.static(planner_config.n_dp)
    n_trees = qd.static(planner_config.n_rrt_trees)
    n_nodes = qd.static(planner_config.n_rrt_nodes)
    n_half = qd.static(planner_config.n_rrt_nodes // 2)

    # One subgroup per tree pair: trees write disjoint node columns and draw counter-hashed streams, so tree
    # parallelism preserves determinism. The lanes of a pair replicate its bookkeeping - sampling, steering, the
    # node insertions - which keeps their control flow convergent for the certified edge checks they share, where
    # the samples of an edge are split across them.
    n_lanes = qd.static(planner_config.n_cost_lanes)
    qd.loop_config(serialize=qd.static(planner_config.para_level < gs.PARA_LEVEL.PARTIAL), block_dim=qd.static(n_lanes))
    for i_tl in range(envs_idx.shape[0] * n_trees * n_lanes):
        i_lane = i_tl % n_lanes
        i_t = i_tl // n_lanes
        if trees_is_active[i_t] == 1:
            i_b = envs_idx[i_t // n_trees]
            col0 = i_t * n_nodes

            # Roots: node 0 = start, node n_half = goal.
            for i_dp in range(n_dp):
                planner_state.rrt.qpos[i_dp, col0] = planner_info.cost.boundary.qpos_start[i_dp, i_b]
                planner_state.rrt.qpos[i_dp, col0 + n_half] = planner_info.cost.boundary.qpos_goal[i_dp, i_b]
            planner_state.rrt.parent[col0] = -1
            planner_state.rrt.parent[col0 + n_half] = -1
            planner_state.rrt.n_nodes[2 * i_t] = 1
            planner_state.rrt.n_nodes[2 * i_t + 1] = 1
            planner_state.rrt.is_done[i_t] = False
            planner_state.rrt.path_len[i_t] = 0

            # Tree growth and the shortcut pass share one certified-edge slot: the check inlines the whole
            # collision cost, so every extra call site duplicates that cost in each kernel reaching the tree (see
            # func_rrt_edge_is_free). Growth fills up to two slots per iteration - the extend edge, then the
            # connect edge, the latter only when the former was certified - and the shortcut pass fills one.
            side = 0
            it = 0
            n_stall = 0
            n_path = 0
            i_cut = 0
            i_near = gs.qd_int(0)
            i_new = gs.qd_int(0)
            i_join = gs.qd_int(0)
            i_from = gs.qd_int(0)
            i_to = gs.qd_int(0)
            base = gs.qd_int(0)
            other = gs.qd_int(0)
            col_new = gs.qd_int(0)
            n_edges = gs.qd_int(0)
            is_growing = True
            is_shortcutting = False
            while is_growing or is_shortcutting:
                if is_growing and (
                    it >= planner_info.rrt.n_iters[None]
                    or n_stall >= planner_info.rrt.n_stall_iters[None]
                    or planner_state.rrt.is_done[i_t]
                ):
                    is_growing = False
                    # Path extraction: start-tree chain reversed in place, then the goal-tree chain appended.
                    if planner_state.rrt.is_done[i_t]:
                        n_path = 0
                        node = planner_state.rrt.bridge[i_t][0]
                        while node != -1:
                            for i_dp in range(n_dp):
                                planner_state.rrt.path[i_dp, col0 + n_path] = planner_state.rrt.qpos[i_dp, col0 + node]
                            n_path = n_path + 1
                            node = planner_state.rrt.parent[col0 + node]
                        for i_swap in range(n_path // 2):
                            for i_dp in range(n_dp):
                                tmp = planner_state.rrt.path[i_dp, col0 + i_swap]
                                planner_state.rrt.path[i_dp, col0 + i_swap] = planner_state.rrt.path[
                                    i_dp, col0 + n_path - 1 - i_swap
                                ]
                                planner_state.rrt.path[i_dp, col0 + n_path - 1 - i_swap] = tmp
                        node = planner_state.rrt.bridge[i_t][1]
                        while node != -1 and n_path < n_nodes:
                            for i_dp in range(n_dp):
                                planner_state.rrt.path[i_dp, col0 + n_path] = planner_state.rrt.qpos[i_dp, col0 + node]
                            n_path = n_path + 1
                            node = planner_state.rrt.parent[col0 + node]
                        is_shortcutting = i_cut < planner_info.rrt.n_shortcut[None] and n_path > 3

                n_edges = 0
                if is_growing:
                    base = side * n_half
                    other = (1 - side) * n_half

                    # Sample a target: goal-biased toward the other tree's newest node, else uniform in limits
                    # (locked DOFs pinned to the start value).
                    if gu.qd_hash01(planner_info.mppi.seed_key[None], i_t, it, 777) < planner_info.rrt.goal_bias[None]:
                        newest = other + planner_state.rrt.n_nodes[2 * i_t + 1 - side] - 1
                        for i_dp in range(n_dp):
                            planner_state.fk.eval.qpos[i_dp, i_t] = planner_state.rrt.qpos[i_dp, col0 + newest]
                    else:
                        for i_dp in range(n_dp):
                            u = gu.qd_hash01(planner_info.mppi.seed_key[None], i_t, it, i_dp)
                            q = planner_info.fk.dofs.q_limit_lower[i_dp] + u * (
                                planner_info.fk.dofs.q_limit_upper[i_dp] - planner_info.fk.dofs.q_limit_lower[i_dp]
                            )
                            if planner_info.fk.dofs.is_locked[i_dp, i_b]:
                                q = planner_info.cost.boundary.qpos_start[i_dp, i_b]
                            planner_state.fk.eval.qpos[i_dp, i_t] = q

                    # Nearest node of the growing side (deterministic lowest-index tie-break).
                    i_near = 0
                    d_near = gs.qd_float(qd.math.inf)
                    for i_n in range(planner_state.rrt.n_nodes[2 * i_t + side]):
                        d = gs.qd_float(0.0)
                        for i_dp in range(n_dp):
                            d = d + (
                                (
                                    planner_state.fk.eval.qpos[i_dp, i_t]
                                    - planner_state.rrt.qpos[i_dp, col0 + base + i_n]
                                )
                                ** 2
                            )
                        if d < d_near:
                            d_near = d
                            i_near = i_n

                    # Steer from the nearest node toward the sample, one L-inf step.
                    d_inf = gs.qd_float(0.0)
                    for i_dp in range(n_dp):
                        d_inf = qd.max(
                            d_inf,
                            qd.abs(
                                planner_state.fk.eval.qpos[i_dp, i_t]
                                - planner_state.rrt.qpos[i_dp, col0 + base + i_near]
                            ),
                        )
                    scale = qd.min(1.0, planner_info.rrt.steer_step[None] / qd.max(d_inf, 1e-9))
                    i_new = planner_state.rrt.n_nodes[2 * i_t + side]
                    if i_new < n_half:
                        col_new = col0 + base + i_new
                        for i_dp in range(n_dp):
                            q_near = planner_state.rrt.qpos[i_dp, col0 + base + i_near]
                            planner_state.rrt.qpos[i_dp, col_new] = q_near + scale * (
                                planner_state.fk.eval.qpos[i_dp, i_t] - q_near
                            )
                        n_edges = 2
                elif is_shortcutting:
                    # Shortcut pass: splice certified straight edges over random sub-chains. Downstream the path
                    # is arclength-resampled to the knot count, and resampled chords cut the corners of the raw
                    # polyline - straightening it first is what keeps the resampled trajectory certifiable. The
                    # tree storage is disposable now, so its first two columns serve as edge-check scratch.
                    u0 = gu.qd_hash01(planner_info.mppi.seed_key[None], i_t, i_cut, 555)
                    u1 = gu.qd_hash01(planner_info.mppi.seed_key[None], i_t, i_cut, 556)
                    i_from = gs.qd_int(u0 * qd.cast(n_path - 3, gs.qd_float))
                    i_to = i_from + 2 + gs.qd_int(u1 * qd.cast(n_path - i_from - 3, gs.qd_float))
                    for i_dp in range(n_dp):
                        planner_state.rrt.qpos[i_dp, col0] = planner_state.rrt.path[i_dp, col0 + i_from]
                        planner_state.rrt.qpos[i_dp, col0 + 1] = planner_state.rrt.path[i_dp, col0 + i_to]
                    n_edges = 1

                is_extended = False
                for i_edge in range(n_edges):
                    col_from = col0
                    col_to = col0 + 1
                    is_wanted = True
                    if is_growing:
                        col_from = col0 + base + i_near
                        col_to = col_new
                        if i_edge == 1:
                            is_wanted = is_extended
                            if is_wanted:
                                # Connect attempt: nearest node of the other tree tries to join the new node.
                                d_join = gs.qd_float(qd.math.inf)
                                for i_n in range(planner_state.rrt.n_nodes[2 * i_t + 1 - side]):
                                    d = gs.qd_float(0.0)
                                    for i_dp in range(n_dp):
                                        d = d + (
                                            (
                                                planner_state.rrt.qpos[i_dp, col_new]
                                                - planner_state.rrt.qpos[i_dp, col0 + other + i_n]
                                            )
                                            ** 2
                                        )
                                    if d < d_join:
                                        d_join = d
                                        i_join = i_n
                                col_from = col0 + other + i_join
                    is_free = False
                    if is_wanted:
                        is_free = func_rrt_edge_is_free(
                            i_t,
                            i_lane,
                            i_b,
                            col_from,
                            col_to,
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
                    if is_free:
                        if not is_growing:
                            n_cut = i_to - i_from - 1
                            for i_n in range(i_from + 1, n_path - n_cut):
                                for i_dp in range(n_dp):
                                    planner_state.rrt.path[i_dp, col0 + i_n] = planner_state.rrt.path[
                                        i_dp, col0 + i_n + n_cut
                                    ]
                            n_path = n_path - n_cut
                        elif i_edge == 0:
                            planner_state.rrt.parent[col_new] = base + i_near
                            planner_state.rrt.n_nodes[2 * i_t + side] = i_new + 1
                            n_stall = -1
                            is_extended = True
                        else:
                            # Bridge stored as (start-tree node, goal-tree node).
                            if side == 0:
                                planner_state.rrt.bridge[i_t] = qd.Vector([base + i_new, other + i_join], dt=gs.qd_int)
                            else:
                                planner_state.rrt.bridge[i_t] = qd.Vector([other + i_join, base + i_new], dt=gs.qd_int)
                            planner_state.rrt.is_done[i_t] = True

                if is_growing:
                    side = 1 - side
                    it = it + 1
                    n_stall = n_stall + 1
                elif is_shortcutting:
                    i_cut = i_cut + 1
                    is_shortcutting = i_cut < planner_info.rrt.n_shortcut[None] and n_path > 3

            planner_state.rrt.path_len[i_t] = n_path
