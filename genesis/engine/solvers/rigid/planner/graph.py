import quadrants as qd

import genesis as gs
from genesis.utils import array_class

from .cost import func_planner_collision_cost
from .kinematics import func_planner_fk, func_planner_spheres
from .trajopt import func_planner_hash01

# Joint-space steer step (L-inf, radians) and the goal-bias probability of extending straight at the other tree.
_RRT_STEER_STEP = 0.2
_RRT_GOAL_BIAS = 0.2


@qd.func
def func_planner_rrt_config_is_free(
    i_t,
    i_b,
    swp,
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
    """Collision check of the configuration staged in the tree's eval column, with sweep allowance swp."""
    func_planner_fk(
        i_t,
        i_t,
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
        i_t,
        i_b,
        links_pos=plan_state.eval_links_pos,
        links_quat=plan_state.eval_links_quat,
        spheres_pos=plan_state.eval_spheres_pos,
        plan_info=plan_info,
        planner_config=planner_config,
    )
    _, min_sd = func_planner_collision_cost(
        i_t,
        i_b,
        swp,
        spheres_pos=plan_state.eval_spheres_pos,
        plan_info=plan_info,
        plan_world=plan_world,
        dyn_info=dyn_info,
        sdf_info=sdf_info,
        planner_config=planner_config,
    )
    return min_sd >= 0.0


@qd.func
def func_planner_rrt_edge_is_free(
    i_t,
    i_b,
    col_from,
    col_to,
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
    Certified edge check between two stored nodes: sample the segment densely enough that each sample's sweep
    allowance (from the per-DOF Lipschitz reach bounds) covers the whole inter-sample motion, so a passing edge
    is collision-free along its continuum, at the required clearance.
    """
    n_dp = qd.static(planner_config.n_dp)
    reach = gs.qd_float(0.0)
    for i_dp in range(n_dp):
        reach += plan_info.dof_reach[i_dp] * qd.abs(
            plan_state.rrt_qpos[i_dp, col_to] - plan_state.rrt_qpos[i_dp, col_from]
        )
    # Quarter-band granularity: each sample's sweep allowance stays ~eps_act/8, so edges certify while only
    # requiring modest true clearance (the demand IS the allowance - coarser sampling would reject any edge
    # whose clearance is below half the activation band).
    n_sub = gs.qd_int(qd.ceil(4.0 * reach / qd.max(plan_info.eps_act[None], 1e-3))) + 1
    swp = 0.5 * reach / qd.cast(n_sub, gs.qd_float)

    is_free = True
    i_sub = 1
    while is_free and i_sub <= n_sub:
        alpha = qd.cast(i_sub, gs.qd_float) / qd.cast(n_sub, gs.qd_float)
        for i_dp in range(n_dp):
            plan_state.eval_qpos[i_dp, i_t] = plan_state.rrt_qpos[i_dp, col_from] + alpha * (
                plan_state.rrt_qpos[i_dp, col_to] - plan_state.rrt_qpos[i_dp, col_from]
            )
        is_free = func_planner_rrt_config_is_free(
            i_t,
            i_b,
            swp,
            plan_state=plan_state,
            plan_info=plan_info,
            plan_world=plan_world,
            dyn_state=dyn_state,
            dyn_info=dyn_info,
            rigid_info=rigid_info,
            sdf_info=sdf_info,
            rigid_config=rigid_config,
            planner_config=planner_config,
        )
        i_sub += 1
    return is_free


@qd.kernel
def kernel_planner_rrt_connect(
    envs_idx: qd.types.ndarray(),
    trees_is_active: qd.types.ndarray(),
    n_iters: int,
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

    qd.loop_config(serialize=qd.static(planner_config.para_level < gs.PARA_LEVEL.ALL))
    for i_t in range(envs_idx.shape[0] * n_trees):
        if trees_is_active[i_t] == 1:
            i_b = envs_idx[i_t // n_trees]
            col0 = i_t * n_nodes

            # Roots: node 0 = start, node n_half = goal.
            for i_dp in range(n_dp):
                plan_state.rrt_qpos[i_dp, col0] = plan_info.qpos_start[i_dp, i_b]
                plan_state.rrt_qpos[i_dp, col0 + n_half] = plan_info.qpos_goal[i_dp, i_b]
            plan_state.rrt_parent[col0] = -1
            plan_state.rrt_parent[col0 + n_half] = -1
            plan_state.rrt_n_nodes[2 * i_t] = 1
            plan_state.rrt_n_nodes[2 * i_t + 1] = 1
            plan_state.rrt_is_done[i_t] = False
            plan_state.rrt_path_len[i_t] = 0

            side = 0
            it = 0
            while it < n_iters and not plan_state.rrt_is_done[i_t]:
                base = side * n_half
                other = (1 - side) * n_half

                # Sample a target: goal-biased toward the other tree's newest node, else uniform in limits
                # (locked DOFs pinned to the start value).
                if func_planner_hash01(plan_info.seed_key[None], i_t, it, 777) < _RRT_GOAL_BIAS:
                    newest = other + plan_state.rrt_n_nodes[2 * i_t + 1 - side] - 1
                    for i_dp in range(n_dp):
                        plan_state.eval_qpos[i_dp, i_t] = plan_state.rrt_qpos[i_dp, col0 + newest]
                else:
                    for i_dp in range(n_dp):
                        u = func_planner_hash01(plan_info.seed_key[None], i_t, it, i_dp)
                        q = plan_info.q_limit_lower[i_dp] + u * (
                            plan_info.q_limit_upper[i_dp] - plan_info.q_limit_lower[i_dp]
                        )
                        if plan_info.dof_is_locked[i_dp, i_b]:
                            q = plan_info.qpos_start[i_dp, i_b]
                        plan_state.eval_qpos[i_dp, i_t] = q

                # Nearest node of the growing side (deterministic lowest-index tie-break).
                i_near = 0
                d_near = gs.qd_float(qd.math.inf)
                for i_n in range(plan_state.rrt_n_nodes[2 * i_t + side]):
                    d = gs.qd_float(0.0)
                    for i_dp in range(n_dp):
                        d += (plan_state.eval_qpos[i_dp, i_t] - plan_state.rrt_qpos[i_dp, col0 + base + i_n]) ** 2
                    if d < d_near:
                        d_near = d
                        i_near = i_n

                # Steer from the nearest node toward the sample, one L-inf step.
                d_inf = gs.qd_float(0.0)
                for i_dp in range(n_dp):
                    d_inf = qd.max(
                        d_inf, qd.abs(plan_state.eval_qpos[i_dp, i_t] - plan_state.rrt_qpos[i_dp, col0 + base + i_near])
                    )
                scale = qd.min(1.0, _RRT_STEER_STEP / qd.max(d_inf, 1e-9))
                i_new = plan_state.rrt_n_nodes[2 * i_t + side]
                if i_new < n_half:
                    col_new = col0 + base + i_new
                    for i_dp in range(n_dp):
                        q_near = plan_state.rrt_qpos[i_dp, col0 + base + i_near]
                        plan_state.rrt_qpos[i_dp, col_new] = q_near + scale * (plan_state.eval_qpos[i_dp, i_t] - q_near)
                    if func_planner_rrt_edge_is_free(
                        i_t,
                        i_b,
                        col0 + base + i_near,
                        col_new,
                        plan_state=plan_state,
                        plan_info=plan_info,
                        plan_world=plan_world,
                        dyn_state=dyn_state,
                        dyn_info=dyn_info,
                        rigid_info=rigid_info,
                        sdf_info=sdf_info,
                        rigid_config=rigid_config,
                        planner_config=planner_config,
                    ):
                        plan_state.rrt_parent[col_new] = base + i_near
                        plan_state.rrt_n_nodes[2 * i_t + side] = i_new + 1

                        # Connect attempt: nearest node of the other tree tries to join the new node.
                        i_join = 0
                        d_join = gs.qd_float(qd.math.inf)
                        for i_n in range(plan_state.rrt_n_nodes[2 * i_t + 1 - side]):
                            d = gs.qd_float(0.0)
                            for i_dp in range(n_dp):
                                d += (
                                    plan_state.rrt_qpos[i_dp, col_new] - plan_state.rrt_qpos[i_dp, col0 + other + i_n]
                                ) ** 2
                            if d < d_join:
                                d_join = d
                                i_join = i_n
                        if func_planner_rrt_edge_is_free(
                            i_t,
                            i_b,
                            col0 + other + i_join,
                            col_new,
                            plan_state=plan_state,
                            plan_info=plan_info,
                            plan_world=plan_world,
                            dyn_state=dyn_state,
                            dyn_info=dyn_info,
                            rigid_info=rigid_info,
                            sdf_info=sdf_info,
                            rigid_config=rigid_config,
                            planner_config=planner_config,
                        ):
                            # Bridge stored as (start-tree node, goal-tree node).
                            if side == 0:
                                plan_state.rrt_bridge[i_t] = qd.Vector([base + i_new, other + i_join], dt=gs.qd_int)
                            else:
                                plan_state.rrt_bridge[i_t] = qd.Vector([other + i_join, base + i_new], dt=gs.qd_int)
                            plan_state.rrt_is_done[i_t] = True
                side = 1 - side
                it += 1

            # Path extraction: start-tree chain reversed in place, then the goal-tree chain appended.
            if plan_state.rrt_is_done[i_t]:
                n_path = 0
                node = plan_state.rrt_bridge[i_t][0]
                while node != -1:
                    for i_dp in range(n_dp):
                        plan_state.rrt_path[i_dp, col0 + n_path] = plan_state.rrt_qpos[i_dp, col0 + node]
                    n_path += 1
                    node = plan_state.rrt_parent[col0 + node]
                for i_swap in range(n_path // 2):
                    for i_dp in range(n_dp):
                        tmp = plan_state.rrt_path[i_dp, col0 + i_swap]
                        plan_state.rrt_path[i_dp, col0 + i_swap] = plan_state.rrt_path[i_dp, col0 + n_path - 1 - i_swap]
                        plan_state.rrt_path[i_dp, col0 + n_path - 1 - i_swap] = tmp
                node = plan_state.rrt_bridge[i_t][1]
                while node != -1 and n_path < n_nodes:
                    for i_dp in range(n_dp):
                        plan_state.rrt_path[i_dp, col0 + n_path] = plan_state.rrt_qpos[i_dp, col0 + node]
                    n_path += 1
                    node = plan_state.rrt_parent[col0 + node]
                plan_state.rrt_path_len[i_t] = n_path
