import quadrants as qd

import genesis as gs
import genesis.utils.geom as gu
from genesis.utils import array_class

from .cost import func_collision_cost
from .kinematics import func_forward_kinematics, func_sphere_positions

# Joint-space steer step (L-inf, radians), the goal-bias probability of extending straight at the other tree,
# and the number of shortcut splices applied to an extracted path.
_RRT_STEER_STEP = 0.4
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
    i_node_from,
    i_node_to,
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
    i_tree = i_t % qd.static(planner_config.n_rrt_trees)
    i_b_ = i_t // qd.static(planner_config.n_rrt_trees)
    reach = gs.qd_float(0.0)
    for i_dp in range(n_dp):
        reach = reach + (
            planner_info.fk.dofs.reach[i_dp]
            * qd.abs(
                planner_state.rrt.qpos[i_dp, i_tree, i_node_to, i_b_]
                - planner_state.rrt.qpos[i_dp, i_tree, i_node_from, i_b_]
            )
        )
    # The density sets each sample's sweep allowance to ~eps_act/(2*density), and the demand IS the allowance:
    # sampling more coarsely rejects any edge whose true clearance falls below it (see planner_edge_check_density).
    n_sub = (
        gs.qd_int(qd.ceil(planner_info.rrt.edge_density[None] * reach / qd.max(planner_info.cost.eps_act[None], 1e-3)))
        + 1
    )
    swp = 0.5 * reach / qd.cast(n_sub, gs.qd_float)
    dq_inf = gs.qd_float(0.0)
    for i_dp in range(n_dp):
        dq_inf = qd.max(
            dq_inf,
            qd.abs(
                planner_state.rrt.qpos[i_dp, i_tree, i_node_to, i_b_]
                - planner_state.rrt.qpos[i_dp, i_tree, i_node_from, i_b_]
            )
            / qd.cast(n_sub, gs.qd_float),
        )

    # Lane i_lane takes a share of the segment's samples and the lanes agree after every round, so an edge spreads
    # its samples across a subgroup and a blocked edge stops at the first round any lane rejects it, instead of each
    # lane walking to its own blocker. The agreed verdict drives the loop, keeping the lanes convergent through the
    # round's reduction, and it cannot change once taken: leaving early means a blocker was found.
    n_lanes = qd.static(planner_config.n_cost_lanes)
    i_e_col = i_t * n_lanes + i_lane
    # Both walk orders visit the same set {1..n_sub}, so the verdict is a conjunction over identical samples and
    # every edge decides identically - the order only moves when a rejected edge meets its blocker. They share one
    # loop because the body inlines the whole collision cost, which a second call site would duplicate in every
    # kernel that grows a tree.
    #   serial arm: bit-reversed, so a blocked stretch is met after a sample count set by its width as a fraction
    #     of the edge rather than by the blocker's distance from the start. Long edges are the ones that reject, so
    #     they stop paying for their length, and the first index visited is the far endpoint - for a growth edge,
    #     the new node the check exists to qualify.
    #   lane arm: start to end, strided. A lane holds only n_sub/n_cost_lanes samples, too few for the reordering
    #     to save a pass, and its skipped indices cost more than they buy (the bit-reversed walk covers a
    #     power-of-two range and drops the overshoot).
    n_steps = (n_sub + n_lanes - 1) // n_lanes
    n_bits = gs.qd_int(1)
    n_perm = gs.qd_int(2)
    if qd.static(n_lanes == 1):
        while n_perm < n_sub:
            n_bits = n_bits + 1
            n_perm = n_perm * 2
        n_steps = n_perm
    is_free_lane = True
    is_free_agreed = True
    i_step = gs.qd_int(0)
    while is_free_agreed and i_step < n_steps:
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
                planner_state.fk.eval.qpos[i_dp, i_e_col] = planner_state.rrt.qpos[
                    i_dp, i_tree, i_node_from, i_b_
                ] + alpha * (
                    planner_state.rrt.qpos[i_dp, i_tree, i_node_to, i_b_]
                    - planner_state.rrt.qpos[i_dp, i_tree, i_node_from, i_b_]
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
        if qd.static(n_lanes > 1):
            is_free_agreed = (
                qd.simt.subgroup.reduce_all_min_tiled(
                    gs.qd_int(1) if is_free_lane else gs.qd_int(0),
                    qd.static(planner_config.n_cost_lanes.bit_length() - 1),
                )
                == 1
            )
        else:
            is_free_agreed = is_free_lane
    return is_free_agreed


@qd.func
def func_rrt_connect(
    i_env_offset,
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
    # The sampling stream keys on the ladder attempt as well as the tree, so a tree that gets walled is grown from
    # a different stream by the next attempt. Without it every attempt regrows the same tree and an env that needs
    # a tree can never be rescued by the ladder, however many attempts it is given. Folding the attempt into the
    # key keeps the draws counter-based, so they stay deterministic under any parallel execution.
    seed_key = planner_info.mppi.seed_key[None] + planner_state.pass_index[None] * 7919
    n_half = qd.static(planner_config.n_rrt_nodes // 2)
    n_path_max = qd.static(planner_config.n_rrt_path)
    # The straightening pass needs two node slots to hold the chord it is testing, and a pass now inherits the
    # nodes the previous one grew, so those slots cannot be tree storage. The top two of the start half are kept
    # out of both sides' reach instead, which costs two nodes of capacity and no allocation.
    n_grow_max = qd.static(planner_config.n_rrt_nodes // 2 - 2)
    i_scratch = qd.static(planner_config.n_rrt_nodes // 2 - 2)

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
            i_b_ = i_t // n_trees
            i_tree = i_t % n_trees
            i_b = envs_idx[i_b_]
            # Draws key on the tree's position in the PASS, not on its slot in the pool: a pass runs in waves
            # over a pool smaller than the batch, so an env's slot depends on how the waves fall and keying on
            # it would make the tree an env grows depend on the pool size rather than on the problem.
            i_draw_key = (i_env_offset + i_b_) * n_trees + i_tree

            # A pass CONTINUES the pair the previous one grew instead of replanting it: an env only reaches
            # escalation because no single search settled it, so re-rooting spends every pass on the first few
            # hundred nodes rather than on the ones that would meet, and the sampling stream keys on the pass so
            # the growth never repeats itself. The pair is rooted afresh only when it holds nothing yet, or when
            # its goal root is no longer the goal - a Cartesian goal is re-resolved per pass, and nodes grown
            # toward a branch that has been dropped hang off a root the plan no longer has.
            is_fresh = planner_state.rrt.n_nodes[0, i_tree, i_b_] == 0
            for i_dp in range(n_dp):
                if (
                    planner_state.rrt.qpos[i_dp, i_tree, n_half, i_b_]
                    != planner_info.cost.boundary.qpos_goal[i_dp, i_b]
                ):
                    is_fresh = True
            if is_fresh:
                # Roots: node 0 = start, node n_half = goal.
                for i_dp in range(n_dp):
                    planner_state.rrt.qpos[i_dp, i_tree, 0, i_b_] = planner_info.cost.boundary.qpos_start[i_dp, i_b]
                    planner_state.rrt.qpos[i_dp, i_tree, n_half, i_b_] = planner_info.cost.boundary.qpos_goal[i_dp, i_b]
                planner_state.rrt.parent[i_tree, 0, i_b_] = -1
                planner_state.rrt.parent[i_tree, n_half, i_b_] = -1
                planner_state.rrt.n_nodes[0, i_tree, i_b_] = 1
                planner_state.rrt.n_nodes[1, i_tree, i_b_] = 1
            # The bridge is this pass's to find. An already joined pair re-joins within its first iterations, and a
            # pass that joins nothing leaves the candidates as the previous one left them.
            planner_state.rrt.is_done[i_tree, i_b_] = False
            planner_state.rrt.path_len[i_tree, i_b_] = 0

            # Tree growth and the shortcut pass share one certified-edge slot: the check inlines the whole
            # collision cost, so every extra call site duplicates that cost in each kernel reaching the tree (see
            # func_rrt_edge_is_free). An iteration issues the extend edge, then - only if it certified - the
            # connect walk's edges one at a time; the shortcut pass issues one.
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
            i_node_new = gs.qd_int(0)
            i_conn = gs.qd_int(0)
            i_cand = gs.qd_int(0)
            i_node_from = gs.qd_int(0)
            i_node_to = gs.qd_int(0)
            is_wanted = False
            is_growing = True
            is_shortcutting = False
            is_reaching = False
            is_both_full = False
            while is_growing or is_shortcutting:
                if is_growing and (
                    it >= planner_info.rrt.n_iters[None]
                    or n_stall >= planner_info.rrt.n_stall_iters[None]
                    or planner_state.rrt.is_done[i_tree, i_b_]
                    or is_both_full
                ):
                    is_growing = False
                    # Path extraction: start-tree chain reversed in place, then the goal-tree chain appended. The
                    # path buffer holds a chain, not a tree, so it is sized for one - a bridge whose chains do
                    # not fit reports no path at all rather than a truncated one, which would silently start or
                    # end somewhere other than the boundary it is supposed to join.
                    if planner_state.rrt.is_done[i_tree, i_b_]:
                        n_path = 0
                        node = planner_state.rrt.bridge[i_tree, i_b_][0]
                        while node != -1 and n_path < n_path_max:
                            for i_dp in range(n_dp):
                                planner_state.rrt.path[i_dp, i_tree, n_path, i_b_] = planner_state.rrt.qpos[
                                    i_dp, i_tree, node, i_b_
                                ]
                            n_path = n_path + 1
                            node = planner_state.rrt.parent[i_tree, node, i_b_]
                        if node == -1:
                            for i_swap in range(n_path // 2):
                                for i_dp in range(n_dp):
                                    tmp = planner_state.rrt.path[i_dp, i_tree, i_swap, i_b_]
                                    planner_state.rrt.path[i_dp, i_tree, i_swap, i_b_] = planner_state.rrt.path[
                                        i_dp, i_tree, n_path - 1 - i_swap, i_b_
                                    ]
                                    planner_state.rrt.path[i_dp, i_tree, n_path - 1 - i_swap, i_b_] = tmp
                            node = planner_state.rrt.bridge[i_tree, i_b_][1]
                            while node != -1 and n_path < n_path_max:
                                for i_dp in range(n_dp):
                                    planner_state.rrt.path[i_dp, i_tree, n_path, i_b_] = planner_state.rrt.qpos[
                                        i_dp, i_tree, node, i_b_
                                    ]
                                n_path = n_path + 1
                                node = planner_state.rrt.parent[i_tree, node, i_b_]
                        if node != -1:
                            planner_state.rrt.is_done[i_tree, i_b_] = False
                            n_path = 0
                        is_shortcutting = (
                            planner_state.rrt.is_done[i_tree, i_b_]
                            and i_cut < planner_info.rrt.n_shortcut[None]
                            and n_path > 3
                        )

                is_wanted = False
                if is_growing:
                    base = side * n_half
                    other = (1 - side) * n_half

                    # Sample a target: goal-biased toward the other tree's newest node, else uniform in limits
                    # (locked DOFs pinned to the start value).
                    if gu.qd_hash01(seed_key, i_draw_key, it, 777) < planner_info.rrt.goal_bias[None]:
                        newest = other + planner_state.rrt.n_nodes[1 - side, i_tree, i_b_] - 1
                        for i_dp in range(n_dp):
                            planner_state.fk.eval.qpos[i_dp, i_t] = planner_state.rrt.qpos[i_dp, i_tree, newest, i_b_]
                    else:
                        for i_dp in range(n_dp):
                            u = gu.qd_hash01(seed_key, i_draw_key, it, i_dp)
                            q = planner_info.fk.dofs.q_limit_lower[i_dp] + u * (
                                planner_info.fk.dofs.q_limit_upper[i_dp] - planner_info.fk.dofs.q_limit_lower[i_dp]
                            )
                            if planner_info.fk.dofs.is_locked[i_dp, i_b]:
                                q = planner_info.cost.boundary.qpos_start[i_dp, i_b]
                            planner_state.fk.eval.qpos[i_dp, i_t] = q

                    # Nearest node of the growing side (deterministic lowest-index tie-break).
                    i_near = 0
                    d_near = gs.qd_float(qd.math.inf)
                    for i_n in range(planner_state.rrt.n_nodes[side, i_tree, i_b_]):
                        d = gs.qd_float(0.0)
                        for i_dp in range(n_dp):
                            d = d + (
                                (
                                    planner_state.fk.eval.qpos[i_dp, i_t]
                                    - planner_state.rrt.qpos[i_dp, i_tree, base + i_n, i_b_]
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
                                - planner_state.rrt.qpos[i_dp, i_tree, base + i_near, i_b_]
                            ),
                        )
                    scale = qd.min(1.0, planner_info.rrt.steer_step[None] / qd.max(d_inf, 1e-9))
                    i_new = planner_state.rrt.n_nodes[side, i_tree, i_b_]
                    if i_new < n_grow_max:
                        i_node_new = base + i_new
                        for i_dp in range(n_dp):
                            q_near = planner_state.rrt.qpos[i_dp, i_tree, base + i_near, i_b_]
                            planner_state.rrt.qpos[i_dp, i_tree, i_node_new, i_b_] = q_near + scale * (
                                planner_state.fk.eval.qpos[i_dp, i_t] - q_near
                            )
                        is_wanted = True
                elif is_shortcutting:
                    # Shortcut pass: splice certified straight edges over random sub-chains. Downstream the path
                    # is arclength-resampled to the knot count, and resampled chords cut the corners of the raw
                    # polyline - straightening it first is what keeps the resampled trajectory certifiable. The
                    # chord goes in the reserved scratch slots (see i_scratch), which no node ever occupies.
                    u0 = gu.qd_hash01(seed_key, i_draw_key, i_cut, 555)
                    u1 = gu.qd_hash01(seed_key, i_draw_key, i_cut, 556)
                    i_from = gs.qd_int(u0 * qd.cast(n_path - 3, gs.qd_float))
                    i_to = i_from + 2 + gs.qd_int(u1 * qd.cast(n_path - i_from - 3, gs.qd_float))
                    for i_dp in range(n_dp):
                        planner_state.rrt.qpos[i_dp, i_tree, i_scratch, i_b_] = planner_state.rrt.path[
                            i_dp, i_tree, i_from, i_b_
                        ]
                        planner_state.rrt.qpos[i_dp, i_tree, i_scratch + 1, i_b_] = planner_state.rrt.path[
                            i_dp, i_tree, i_to, i_b_
                        ]
                    is_wanted = True

                # The extend edge comes first; the connect walk below then issues one request per steer step, so
                # both go through the single certified-edge slot.
                is_extended = False
                is_connecting = False
                while is_wanted:
                    i_node_from = i_scratch
                    i_node_to = i_scratch + 1
                    if is_growing:
                        i_node_from = base + i_near
                        i_node_to = i_node_new
                        if is_connecting:
                            # Greedy connect: the other tree walks toward the new node one steer step at a time,
                            # keeping every step that certifies, and bridges when the last step lands within one
                            # step of it. Joining through a single long edge instead leaves the other tree unable
                            # to gain a node at all in clutter - the far endpoint is the whole gap away, so the
                            # edge almost never certifies and the search degenerates to one tree with a goal bias.
                            d_conn = gs.qd_float(0.0)
                            for i_dp in range(n_dp):
                                d_conn = qd.max(
                                    d_conn,
                                    qd.abs(
                                        planner_state.rrt.qpos[i_dp, i_tree, i_node_new, i_b_]
                                        - planner_state.rrt.qpos[i_dp, i_tree, i_conn, i_b_]
                                    ),
                                )
                            is_reaching = d_conn <= planner_info.rrt.steer_step[None]
                            i_node_from = i_conn
                            if is_reaching:
                                i_node_to = i_node_new
                            else:
                                i_cand = planner_state.rrt.n_nodes[1 - side, i_tree, i_b_]
                                # A full tree stops walking rather than overwriting a node another edge relies on.
                                is_wanted = i_cand < n_grow_max
                                if is_wanted:
                                    i_node_to = other + i_cand
                                    scale_conn = planner_info.rrt.steer_step[None] / d_conn
                                    for i_dp in range(n_dp):
                                        q_conn = planner_state.rrt.qpos[i_dp, i_tree, i_conn, i_b_]
                                        planner_state.rrt.qpos[i_dp, i_tree, i_node_to, i_b_] = q_conn + scale_conn * (
                                            planner_state.rrt.qpos[i_dp, i_tree, i_node_new, i_b_] - q_conn
                                        )
                    is_free = False
                    if is_wanted:
                        is_free = func_rrt_edge_is_free(
                            i_t,
                            i_lane,
                            i_b,
                            i_node_from,
                            i_node_to,
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
                                    planner_state.rrt.path[i_dp, i_tree, i_n, i_b_] = planner_state.rrt.path[
                                        i_dp, i_tree, i_n + n_cut, i_b_
                                    ]
                            n_path = n_path - n_cut
                        elif not is_connecting:
                            planner_state.rrt.parent[i_tree, i_node_new, i_b_] = base + i_near
                            planner_state.rrt.n_nodes[side, i_tree, i_b_] = i_new + 1
                            n_stall = -1
                            is_extended = True
                        elif is_reaching:
                            # Bridge stored as (start-tree node, goal-tree node).
                            if side == 0:
                                planner_state.rrt.bridge[i_tree, i_b_] = qd.Vector([i_node_new, i_conn], dt=gs.qd_int)
                            else:
                                planner_state.rrt.bridge[i_tree, i_b_] = qd.Vector([i_conn, i_node_new], dt=gs.qd_int)
                            planner_state.rrt.is_done[i_tree, i_b_] = True
                        else:
                            # A certified step is a node the other tree keeps, and the walk resumes from it.
                            planner_state.rrt.parent[i_tree, i_node_to, i_b_] = i_conn
                            planner_state.rrt.n_nodes[1 - side, i_tree, i_b_] = i_cand + 1
                            i_conn = i_node_to
                            n_stall = -1

                    # The extend edge hands over to the connect walk, which continues while its steps certify.
                    if not is_growing:
                        is_wanted = False
                    elif not is_connecting:
                        is_wanted = is_free
                        if is_wanted:
                            # The walk starts at the other tree's node nearest the new one.
                            d_join = gs.qd_float(qd.math.inf)
                            for i_n in range(planner_state.rrt.n_nodes[1 - side, i_tree, i_b_]):
                                d = gs.qd_float(0.0)
                                for i_dp in range(n_dp):
                                    d = d + (
                                        (
                                            planner_state.rrt.qpos[i_dp, i_tree, i_node_new, i_b_]
                                            - planner_state.rrt.qpos[i_dp, i_tree, other + i_n, i_b_]
                                        )
                                        ** 2
                                    )
                                if d < d_join:
                                    d_join = d
                                    i_join = i_n
                            i_conn = other + i_join
                            is_connecting = True
                    else:
                        is_wanted = is_free and not is_reaching

                if is_growing:
                    side = 1 - side
                    # A side that has filled its half can store nothing, so it hands its turn to the other one
                    # rather than spending the iteration on an extension with nowhere to go. The two sides fill at
                    # very different rates - the one rooted in open space runs away with its half while the one
                    # walled in clutter is still near its root - so this is what lets the walled side keep growing
                    # once its partner is done. Growth ends when neither side has room.
                    if planner_state.rrt.n_nodes[side, i_tree, i_b_] >= n_grow_max:
                        side = 1 - side
                        is_both_full = planner_state.rrt.n_nodes[side, i_tree, i_b_] >= n_grow_max
                    it = it + 1
                    n_stall = n_stall + 1
                elif is_shortcutting:
                    i_cut = i_cut + 1
                    is_shortcutting = i_cut < planner_info.rrt.n_shortcut[None] and n_path > 3

            planner_state.rrt.path_len[i_tree, i_b_] = n_path
