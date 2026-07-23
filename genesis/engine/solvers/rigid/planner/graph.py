import quadrants as qd

import genesis as gs
import genesis.utils.geom as gu
from genesis.utils import array_class

from .cost import func_planner_collision_cost
from .kinematics import func_planner_fk, func_planner_spheres

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
def func_planner_rrt_config_is_free(
    i_t,
    i_b,
    swp,
    dq_inf,
    planner_state: array_class.PlannerState,
    planner_info: array_class.PlannerEntityInfo,
    planner_world: array_class.PlannerWorldState,
    dyn_state: array_class.DynState,
    collider_state: array_class.ColliderState,
    gjk_state: array_class.GJKState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    collider_info: array_class.ColliderInfo,
    sdf_info: array_class.SDFInfo,
    rigid_config: qd.template(),
    collider_static_config: qd.template(),
    planner_config: qd.template(),
):
    """Collision check of the configuration staged in the tree's eval column, with sweep allowance swp."""
    func_planner_fk(
        i_t,
        i_t,
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
    func_planner_spheres(
        i_t,
        i_b,
        links_pos=planner_state.fk.eval.links_pos,
        links_quat=planner_state.fk.eval.links_quat,
        spheres_pos=planner_state.fk.eval.spheres_pos,
        planner_info=planner_info,
        planner_config=planner_config,
    )
    _, min_sd_exact, min_sd_proxy = func_planner_collision_cost(
        i_t,
        i_b,
        swp,
        dq_inf,
        links_pos=planner_state.fk.eval.links_pos,
        links_quat=planner_state.fk.eval.links_quat,
        spheres_pos=planner_state.fk.eval.spheres_pos,
        planner_info=planner_info,
        planner_world=planner_world,
        collider_state=collider_state,
        gjk_state=gjk_state,
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
def func_planner_rrt_edge_is_free(
    i_t,
    i_b,
    col_from,
    col_to,
    planner_state: array_class.PlannerState,
    planner_info: array_class.PlannerEntityInfo,
    planner_world: array_class.PlannerWorldState,
    dyn_state: array_class.DynState,
    collider_state: array_class.ColliderState,
    gjk_state: array_class.GJKState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    collider_info: array_class.ColliderInfo,
    sdf_info: array_class.SDFInfo,
    rigid_config: qd.template(),
    collider_static_config: qd.template(),
    planner_config: qd.template(),
):
    """
    Certified edge check between two stored nodes: sample the segment densely enough that each sample's sweep
    allowance (from the per-DOF Lipschitz reach bounds) covers the whole inter-sample motion, so a passing edge
    is collision-free along its continuum, at the required clearance.
    """
    n_dp = qd.static(planner_config.n_dp)
    reach = gs.qd_float(0.0)
    for i_dp in range(n_dp):
        reach += planner_info.fk.dofs.reach[i_dp] * qd.abs(
            planner_state.rrt.qpos[i_dp, col_to] - planner_state.rrt.qpos[i_dp, col_from]
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

    is_free = True
    i_sub = 1
    while is_free and i_sub <= n_sub:
        alpha = qd.cast(i_sub, gs.qd_float) / qd.cast(n_sub, gs.qd_float)
        for i_dp in range(n_dp):
            planner_state.fk.eval.qpos[i_dp, i_t] = planner_state.rrt.qpos[i_dp, col_from] + alpha * (
                planner_state.rrt.qpos[i_dp, col_to] - planner_state.rrt.qpos[i_dp, col_from]
            )
        is_free = func_planner_rrt_config_is_free(
            i_t,
            i_b,
            swp,
            dq_inf,
            planner_state=planner_state,
            planner_info=planner_info,
            planner_world=planner_world,
            dyn_state=dyn_state,
            collider_state=collider_state,
            gjk_state=gjk_state,
            dyn_info=dyn_info,
            rigid_info=rigid_info,
            collider_info=collider_info,
            sdf_info=sdf_info,
            rigid_config=rigid_config,
            collider_static_config=collider_static_config,
            planner_config=planner_config,
        )
        i_sub += 1
    return is_free


@qd.func
def func_planner_rrt_connect(
    envs_idx: qd.types.ndarray(),
    trees_is_active: qd.types.ndarray(),
    planner_state: array_class.PlannerState,
    planner_info: array_class.PlannerEntityInfo,
    planner_world: array_class.PlannerWorldState,
    dyn_state: array_class.DynState,
    collider_state: array_class.ColliderState,
    gjk_state: array_class.GJKState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    collider_info: array_class.ColliderInfo,
    sdf_info: array_class.SDFInfo,
    rigid_config: qd.template(),
    collider_static_config: qd.template(),
    planner_config: qd.template(),
):
    """
    Batched RRT-Connect seed generator: one thread runs one whole tree pair (sample -> nearest -> steer ->
    certified edge -> insert -> connect), entirely serial inside, so every accepted edge is certified and the
    result is deterministic per backend (all randomness from the counter-based hash). The start tree grows in
    node columns [0, n/2), the goal tree in [n/2, n); on success the bridge pair joins them and the path is
    backtracked into rrt_path, start to goal.
    """
    n_dp = qd.static(planner_config.n_dp)
    n_trees = qd.static(planner_config.n_rrt_trees)
    n_nodes = qd.static(planner_config.n_rrt_nodes)
    n_half = qd.static(planner_config.n_rrt_nodes // 2)

    # One thread per tree pair: trees write disjoint node columns and draw counter-hashed streams, so tree
    # parallelism preserves determinism (a hard-serial loop runs the whole batch on a single GPU thread).
    qd.loop_config(serialize=qd.static(planner_config.para_level < gs.PARA_LEVEL.ALL))
    for i_t in range(envs_idx.shape[0] * n_trees):
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

            side = 0
            it = 0
            n_stall = 0
            while (
                it < planner_info.rrt.n_iters[None]
                and n_stall < planner_info.rrt.n_stall_iters[None]
                and not planner_state.rrt.is_done[i_t]
            ):
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
                        d += (
                            planner_state.fk.eval.qpos[i_dp, i_t] - planner_state.rrt.qpos[i_dp, col0 + base + i_n]
                        ) ** 2
                    if d < d_near:
                        d_near = d
                        i_near = i_n

                # Steer from the nearest node toward the sample, one L-inf step.
                d_inf = gs.qd_float(0.0)
                for i_dp in range(n_dp):
                    d_inf = qd.max(
                        d_inf,
                        qd.abs(
                            planner_state.fk.eval.qpos[i_dp, i_t] - planner_state.rrt.qpos[i_dp, col0 + base + i_near]
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
                    if func_planner_rrt_edge_is_free(
                        i_t,
                        i_b,
                        col0 + base + i_near,
                        col_new,
                        planner_state=planner_state,
                        planner_info=planner_info,
                        planner_world=planner_world,
                        dyn_state=dyn_state,
                        collider_state=collider_state,
                        gjk_state=gjk_state,
                        dyn_info=dyn_info,
                        rigid_info=rigid_info,
                        collider_info=collider_info,
                        sdf_info=sdf_info,
                        rigid_config=rigid_config,
                        collider_static_config=collider_static_config,
                        planner_config=planner_config,
                    ):
                        planner_state.rrt.parent[col_new] = base + i_near
                        planner_state.rrt.n_nodes[2 * i_t + side] = i_new + 1
                        n_stall = -1

                        # Connect attempt: nearest node of the other tree tries to join the new node.
                        i_join = 0
                        d_join = gs.qd_float(qd.math.inf)
                        for i_n in range(planner_state.rrt.n_nodes[2 * i_t + 1 - side]):
                            d = gs.qd_float(0.0)
                            for i_dp in range(n_dp):
                                d += (
                                    planner_state.rrt.qpos[i_dp, col_new]
                                    - planner_state.rrt.qpos[i_dp, col0 + other + i_n]
                                ) ** 2
                            if d < d_join:
                                d_join = d
                                i_join = i_n
                        if func_planner_rrt_edge_is_free(
                            i_t,
                            i_b,
                            col0 + other + i_join,
                            col_new,
                            planner_state=planner_state,
                            planner_info=planner_info,
                            planner_world=planner_world,
                            dyn_state=dyn_state,
                            collider_state=collider_state,
                            gjk_state=gjk_state,
                            dyn_info=dyn_info,
                            rigid_info=rigid_info,
                            collider_info=collider_info,
                            sdf_info=sdf_info,
                            rigid_config=rigid_config,
                            collider_static_config=collider_static_config,
                            planner_config=planner_config,
                        ):
                            # Bridge stored as (start-tree node, goal-tree node).
                            if side == 0:
                                planner_state.rrt.bridge[i_t] = qd.Vector([base + i_new, other + i_join], dt=gs.qd_int)
                            else:
                                planner_state.rrt.bridge[i_t] = qd.Vector([other + i_join, base + i_new], dt=gs.qd_int)
                            planner_state.rrt.is_done[i_t] = True
                side = 1 - side
                it += 1
                n_stall += 1

            # Path extraction: start-tree chain reversed in place, then the goal-tree chain appended.
            if planner_state.rrt.is_done[i_t]:
                n_path = 0
                node = planner_state.rrt.bridge[i_t][0]
                while node != -1:
                    for i_dp in range(n_dp):
                        planner_state.rrt.path[i_dp, col0 + n_path] = planner_state.rrt.qpos[i_dp, col0 + node]
                    n_path += 1
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
                    n_path += 1
                    node = planner_state.rrt.parent[col0 + node]

                # Shortcut pass: splice certified straight edges over random sub-chains. Downstream the path is
                # arclength-resampled to the knot count, and resampled chords cut the corners of the raw
                # polyline - straightening it first is what keeps the resampled trajectory certifiable. The tree
                # storage is disposable now, so its first two columns serve as edge-check scratch.
                for i_cut in range(planner_info.rrt.n_shortcut[None]):
                    if n_path > 3:
                        u0 = gu.qd_hash01(planner_info.mppi.seed_key[None], i_t, i_cut, 555)
                        u1 = gu.qd_hash01(planner_info.mppi.seed_key[None], i_t, i_cut, 556)
                        i_from = gs.qd_int(u0 * qd.cast(n_path - 3, gs.qd_float))
                        i_to = i_from + 2 + gs.qd_int(u1 * qd.cast(n_path - i_from - 3, gs.qd_float))
                        for i_dp in range(n_dp):
                            planner_state.rrt.qpos[i_dp, col0] = planner_state.rrt.path[i_dp, col0 + i_from]
                            planner_state.rrt.qpos[i_dp, col0 + 1] = planner_state.rrt.path[i_dp, col0 + i_to]
                        if func_planner_rrt_edge_is_free(
                            i_t,
                            i_b,
                            col0,
                            col0 + 1,
                            planner_state=planner_state,
                            planner_info=planner_info,
                            planner_world=planner_world,
                            dyn_state=dyn_state,
                            collider_state=collider_state,
                            gjk_state=gjk_state,
                            dyn_info=dyn_info,
                            rigid_info=rigid_info,
                            collider_info=collider_info,
                            sdf_info=sdf_info,
                            rigid_config=rigid_config,
                            collider_static_config=collider_static_config,
                            planner_config=planner_config,
                        ):
                            n_cut = i_to - i_from - 1
                            for i_n in range(i_from + 1, n_path - n_cut):
                                for i_dp in range(n_dp):
                                    planner_state.rrt.path[i_dp, col0 + i_n] = planner_state.rrt.path[
                                        i_dp, col0 + i_n + n_cut
                                    ]
                            n_path -= n_cut
                planner_state.rrt.path_len[i_t] = n_path


@qd.kernel
def kernel_planner_rrt_connect(
    envs_idx: qd.types.ndarray(),
    trees_is_active: qd.types.ndarray(),
    planner_state: array_class.PlannerState,
    planner_info: array_class.PlannerEntityInfo,
    planner_world: array_class.PlannerWorldState,
    dyn_state: array_class.DynState,
    collider_state: array_class.ColliderState,
    gjk_state: array_class.GJKState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    collider_info: array_class.ColliderInfo,
    sdf_info: array_class.SDFInfo,
    rigid_config: qd.template(),
    collider_static_config: qd.template(),
    planner_config: qd.template(),
):
    func_planner_rrt_connect(
        envs_idx,
        trees_is_active,
        planner_state,
        planner_info,
        planner_world,
        dyn_state,
        collider_state,
        gjk_state,
        dyn_info,
        rigid_info,
        collider_info,
        sdf_info,
        rigid_config,
        collider_static_config,
        planner_config,
    )
