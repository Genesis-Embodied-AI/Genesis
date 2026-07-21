import numpy as np
import quadrants as qd

import genesis as gs
from genesis.utils import array_class

from .cost import func_planner_collision_cost, func_planner_knot_cost_grad, func_planner_pose_cost
from .kinematics import func_planner_fk, func_planner_spheres

# Fixed step-size ladder of the parallel noisy line search (jittered per candidate and iteration).
_LS_LADDER = (1.0, 0.5, 0.2, 0.08, 0.03, 0.01, 0.003, 0.001)
# MPPI noise annealing factor per iteration.
_MPPI_ANNEAL = 0.7


def build_noise_basis(n_knots, has_clamped_end):
    """
    Smooth MPPI noise basis: Gaussian bumps over the knot axis, one column per noise knot, with zero rows at the
    clamped knots (start pair always; goal pair for joint goals) so perturbations preserve the boundary
    conditions. Deterministic pure numpy.
    """
    n_noise = array_class.PLANNER_N_NOISE_KNOTS
    t = np.linspace(0.0, 1.0, n_knots)
    centers = np.linspace(0.0, 1.0, n_noise + 2)[1:-1]
    width = 1.0 / (n_noise + 1)
    basis = np.exp(-0.5 * ((t[:, None] - centers[None, :]) / width) ** 2)
    basis[:2] = 0.0
    if has_clamped_end:
        basis[-2:] = 0.0
    # Unit peak response per knot so mppi_sigma is expressed in joint units.
    scale = basis.sum(axis=1).max()
    return (basis / scale).astype(gs.np_float)


@qd.func
def func_planner_hash01(k0, k1, k2, k3):
    """Counter-based hash to [0, 1): value depends only on the four keys, never on thread scheduling."""
    h = qd.cast(k0, qd.u32) * qd.u32(0x9E3779B1)
    h = (h ^ qd.cast(k1, qd.u32)) * qd.u32(0x85EBCA77)
    h = (h ^ qd.cast(k2, qd.u32)) * qd.u32(0xC2B2AE3D)
    h = (h ^ qd.cast(k3, qd.u32)) * qd.u32(0x27D4EB2F)
    h = h ^ (h >> qd.u32(15))
    h = h * qd.u32(0x2C1B3C6D)
    h = h ^ (h >> qd.u32(12))
    return qd.cast(h >> qd.u32(8), gs.qd_float) * (1.0 / 16777216.0)


@qd.func
def func_planner_gauss(k0, k1, k2, k3):
    """Standard normal draw from the counter-based hash (Box-Muller on two uniform lanes)."""
    u1 = qd.max(func_planner_hash01(k0, k1, k2, 2 * k3), 1e-7)
    u2 = func_planner_hash01(k0, k1, k2, 2 * k3 + 1)
    return qd.sqrt(-2.0 * qd.log(u1)) * qd.cos(6.2831853 * u2)


@qd.func
def func_planner_traj_cost(
    i_c,
    i_b,
    i_e_col,
    col_base,
    qpos_cols: qd.Tensor,
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
    Total cost of one full trajectory read from qpos_cols[:, col_base + w] - the cost-only evaluator of MPPI
    rollouts and line-search trials. Serial over knots through the thread's private eval column; the sweep
    allowance uses the joint-space Lipschitz reach bound (no neighbor FK needed), clamped like the gradient path.
    """
    n_knots = qd.static(planner_config.n_knots)
    n_dp = qd.static(planner_config.n_dp)
    cost = gs.qd_float(0.0)

    for i_w in range(n_knots):
        # Sweep allowance and L-inf joint travel toward both neighbors.
        swp = gs.qd_float(0.0)
        dq_inf = gs.qd_float(0.0)
        for i_dp in range(n_dp):
            q = qpos_cols[i_dp, col_base + i_w]
            plan_state.eval_qpos[i_dp, i_e_col] = q
            if i_w > 0:
                swp = qd.max(swp, 0.5 * plan_info.dof_reach[i_dp] * qd.abs(q - qpos_cols[i_dp, col_base + i_w - 1]))
                dq_inf = qd.max(dq_inf, qd.abs(q - qpos_cols[i_dp, col_base + i_w - 1]))
            if i_w < n_knots - 1:
                swp = qd.max(swp, 0.5 * plan_info.dof_reach[i_dp] * qd.abs(qpos_cols[i_dp, col_base + i_w + 1] - q))
                dq_inf = qd.max(dq_inf, qd.abs(qpos_cols[i_dp, col_base + i_w + 1] - q))
        swp = qd.min(swp, plan_info.eps_act[None])

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
        coll_cost, _ = func_planner_collision_cost(
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
        cost += coll_cost

        # Smoothness / limit / posture terms, identical to the gradient path (see func_planner_knot_cost_grad).
        for i_dp in range(n_dp):
            q = qpos_cols[i_dp, col_base + i_w]
            if 1 <= i_w < n_knots - 1:
                acc = qpos_cols[i_dp, col_base + i_w - 1] - 2.0 * q + qpos_cols[i_dp, col_base + i_w + 1]
                cost += plan_info.w_acc[None] * acc**2
            if 1 <= i_w < n_knots - 2:
                jerk = (
                    qpos_cols[i_dp, col_base + i_w + 2]
                    - 3.0 * qpos_cols[i_dp, col_base + i_w + 1]
                    + 3.0 * q
                    - qpos_cols[i_dp, col_base + i_w - 1]
                )
                cost += plan_info.w_jerk[None] * jerk**2
            over = q - plan_info.q_limit_upper[i_dp]
            under = plan_info.q_limit_lower[i_dp] - q
            if over > 0.0:
                cost += plan_info.w_lim[None] * over**2
            if under > 0.0:
                cost += plan_info.w_lim[None] * under**2
            ref = plan_info.qpos_start[i_dp, i_b] + (
                plan_info.qpos_goal[i_dp, i_b] - plan_info.qpos_start[i_dp, i_b]
            ) * (qd.cast(i_w, gs.qd_float) / float(n_knots - 1))
            cost += plan_info.w_posture[None] * (q - ref) ** 2

        if plan_info.has_pose_goal[None] and i_w == n_knots - 1:
            cost += func_planner_pose_cost(
                i_e_col,
                i_b,
                links_pos=plan_state.eval_links_pos,
                links_quat=plan_state.eval_links_quat,
                plan_info=plan_info,
                rigid_info=rigid_info,
                planner_config=planner_config,
            )
    return cost


@qd.func
def func_planner_candidate_cost_grad(
    i_c,
    i_b,
    plan_state: array_class.PlannerState,
    plan_info: array_class.PlannerEntityInfo,
    plan_world: array_class.PlannerWorldState,
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    sdf_info: array_class.SDFInfo,
    rigid_config: qd.template(),
    collider_static_config: qd.template(),
    planner_config: qd.template(),
):
    """FK cache + cost + gradient of one candidate's whole trajectory, serial over its knots. Returns the cost."""
    n_knots = qd.static(planner_config.n_knots)
    for i_w in range(n_knots):
        i_cw = i_c * n_knots + i_w
        func_planner_fk(
            i_cw,
            i_cw,
            i_b,
            qpos=plan_state.qpos_traj,
            links_pos=plan_state.links_pos,
            links_quat=plan_state.links_quat,
            joints_xanchor=plan_state.joints_xanchor,
            joints_xaxis=plan_state.joints_xaxis,
            dyn_state=dyn_state,
            dyn_info=dyn_info,
            rigid_info=rigid_info,
            rigid_config=rigid_config,
            planner_config=planner_config,
        )
        func_planner_spheres(
            i_cw,
            i_b,
            links_pos=plan_state.links_pos,
            links_quat=plan_state.links_quat,
            spheres_pos=plan_state.spheres_pos,
            plan_info=plan_info,
            planner_config=planner_config,
        )
    cost = gs.qd_float(0.0)
    for i_w in range(n_knots):
        i_cw = i_c * n_knots + i_w
        func_planner_knot_cost_grad(
            i_c,
            i_w,
            i_cw,
            i_b,
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
        cost += plan_state.cost_wp[i_cw]
    return cost


@qd.func
def func_planner_mask_clamped(i_c, i_w, i_b, plan_info: array_class.PlannerEntityInfo, planner_config: qd.template()):
    """True when knot i_w is clamped: the start pair always, the goal pair for joint-space goals."""
    n_knots = qd.static(planner_config.n_knots)
    is_clamped = i_w <= 1
    if not plan_info.has_pose_goal[None] and i_w >= n_knots - 2:
        is_clamped = True
    return is_clamped


@qd.kernel
def kernel_planner_mppi(
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
    Iteration-resident MPPI warmup: every candidate refines its mean trajectory through annealed smooth-noise
    particle rollouts and softmax-weighted updates, entirely inside its own thread (one kernel launch per phase,
    deterministic per backend - all randomness comes from the counter-based hash keyed by content indices).
    Particle noise is regenerated on demand instead of stored: pass one scores the particles, pass two rebuilds
    the winning mixture.
    """
    n_knots = qd.static(planner_config.n_knots)
    n_seeds = qd.static(planner_config.n_seeds)
    n_dp = qd.static(planner_config.n_dp)
    n_noise = qd.static(array_class.PLANNER_N_NOISE_KNOTS)

    qd.loop_config(serialize=qd.static(planner_config.para_level < gs.PARA_LEVEL.ALL))
    for i_c in range(envs_idx.shape[0] * n_seeds):
        if plan_state.is_active[i_c]:
            i_b = envs_idx[i_c // n_seeds]
            col_base = i_c * n_knots
            n_particles = qd.min(plan_info.mppi_n_particles[None], array_class.PLANNER_MPPI_P_MAX)
            costs = qd.Vector.zero(gs.qd_float, array_class.PLANNER_MPPI_P_MAX)

            for it in range(plan_info.mppi_n_iters[None]):
                anneal = _MPPI_ANNEAL**it
                # Pass 1: score every particle (particle 0 is the unperturbed mean).
                for i_p in range(n_particles):
                    for i_w in range(n_knots):
                        is_clamped = func_planner_mask_clamped(i_c, i_w, i_b, plan_info, planner_config)
                        for i_dp in range(n_dp):
                            q = plan_state.qpos_traj[i_dp, col_base + i_w]
                            if not is_clamped:
                                delta = gs.qd_float(0.0)
                                if i_p > 0 and not plan_info.dof_is_locked[i_dp, i_b]:
                                    for i_k in range(n_noise):
                                        key = (
                                            (it * array_class.PLANNER_MPPI_P_MAX + i_p) * n_noise + i_k
                                        ) * n_dp + i_dp
                                        delta += plan_info.noise_basis[i_w, i_k] * func_planner_gauss(
                                            plan_info.seed_key[None], i_c, key, 0
                                        )
                                    delta *= anneal * plan_info.mppi_sigma[i_dp]
                                q = qd.math.clamp(
                                    q + delta, plan_info.q_limit_lower[i_dp], plan_info.q_limit_upper[i_dp]
                                )
                            plan_state.trial_qpos[i_dp, col_base + i_w] = q
                    costs[i_p] = func_planner_traj_cost(
                        i_c,
                        i_b,
                        i_c,
                        col_base,
                        qpos_cols=plan_state.trial_qpos,
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

                # Softmax weights with a median-scaled temperature.
                cost_min = costs[0]
                cost_mean = gs.qd_float(0.0)
                for i_p in range(n_particles):
                    cost_min = qd.min(cost_min, costs[i_p])
                    cost_mean += costs[i_p]
                cost_mean /= qd.cast(n_particles, gs.qd_float)
                beta = 0.5 * (cost_mean - cost_min) + 1e-6
                weight_sum = gs.qd_float(0.0)
                for i_p in range(n_particles):
                    weight_sum += qd.exp(-(costs[i_p] - cost_min) / beta)

                # Pass 2: rebuild the weighted noise mixture and fold it into the mean.
                for i_w in range(n_knots):
                    if not func_planner_mask_clamped(i_c, i_w, i_b, plan_info, planner_config):
                        for i_dp in range(n_dp):
                            if not plan_info.dof_is_locked[i_dp, i_b]:
                                delta_mean = gs.qd_float(0.0)
                                for i_p in range(1, n_particles):
                                    weight = qd.exp(-(costs[i_p] - cost_min) / beta) / weight_sum
                                    delta = gs.qd_float(0.0)
                                    for i_k in range(n_noise):
                                        key = (
                                            (it * array_class.PLANNER_MPPI_P_MAX + i_p) * n_noise + i_k
                                        ) * n_dp + i_dp
                                        delta += plan_info.noise_basis[i_w, i_k] * func_planner_gauss(
                                            plan_info.seed_key[None], i_c, key, 0
                                        )
                                    delta_mean += weight * delta * anneal * plan_info.mppi_sigma[i_dp]
                                plan_state.qpos_traj[i_dp, col_base + i_w] = qd.math.clamp(
                                    plan_state.qpos_traj[i_dp, col_base + i_w] + delta_mean,
                                    plan_info.q_limit_lower[i_dp],
                                    plan_info.q_limit_upper[i_dp],
                                )


@qd.kernel
def kernel_planner_lbfgs(
    envs_idx: qd.types.ndarray(),
    plan_state: array_class.PlannerState,
    plan_info: array_class.PlannerEntityInfo,
    plan_world: array_class.PlannerWorldState,
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    sdf_info: array_class.SDFInfo,
    rigid_config: qd.template(),
    collider_static_config: qd.template(),
    planner_config: qd.template(),
):
    """
    Iteration-resident limited-memory Broyden-Fletcher-Goldfarb-Shanno (L-BFGS) refinement: the whole two-loop
    recursion + step-ladder line search runs inside each candidate's thread. Everything is serial per candidate
    (deterministic); clamped knots and locked DOFs never move (their gradient entries are zeroed).
    """
    n_knots = qd.static(planner_config.n_knots)
    n_seeds = qd.static(planner_config.n_seeds)
    n_dp = qd.static(planner_config.n_dp)
    m_hist = qd.static(array_class.PLANNER_LBFGS_M)

    qd.loop_config(serialize=qd.static(planner_config.para_level < gs.PARA_LEVEL.ALL))
    for i_c in range(envs_idx.shape[0] * n_seeds):
        if plan_state.is_active[i_c]:
            i_b = envs_idx[i_c // n_seeds]
            col_base = i_c * n_knots

            cost = func_planner_candidate_cost_grad(
                i_c,
                i_b,
                plan_state=plan_state,
                plan_info=plan_info,
                plan_world=plan_world,
                dyn_state=dyn_state,
                dyn_info=dyn_info,
                rigid_info=rigid_info,
                sdf_info=sdf_info,
                rigid_config=rigid_config,
                collider_static_config=collider_static_config,
                planner_config=planner_config,
            )
            # Zero the gradient of clamped knots so the whole optimizer sees the reduced problem.
            for i_w in range(n_knots):
                if func_planner_mask_clamped(i_c, i_w, i_b, plan_info, planner_config):
                    for i_dp in range(n_dp):
                        plan_state.grad_traj[i_dp, col_base + i_w] = 0.0

            n_hist = 0
            i_next = 0
            n_stall = 0
            it = 0
            while it < plan_info.lbfgs_n_iters[None] and n_stall < 4:
                # Two-loop recursion over the ring-buffer history -> descent direction in dir_traj.
                for i_w in range(n_knots):
                    for i_dp in range(n_dp):
                        plan_state.dir_traj[i_dp, col_base + i_w] = plan_state.grad_traj[i_dp, col_base + i_w]
                alphas = qd.Vector.zero(gs.qd_float, array_class.PLANNER_LBFGS_M)
                for i_h_ in range(n_hist):
                    i_h = (i_next - 1 - i_h_) % m_hist
                    dot_sd = gs.qd_float(0.0)
                    for i_w in range(n_knots):
                        for i_dp in range(n_dp):
                            dot_sd += (
                                plan_state.lbfgs_s[i_h, i_dp, col_base + i_w]
                                * plan_state.dir_traj[i_dp, col_base + i_w]
                            )
                    alpha_h = plan_state.lbfgs_rho[i_h, i_c] * dot_sd
                    alphas[i_h] = alpha_h
                    for i_w in range(n_knots):
                        for i_dp in range(n_dp):
                            plan_state.dir_traj[i_dp, col_base + i_w] -= (
                                alpha_h * plan_state.lbfgs_y[i_h, i_dp, col_base + i_w]
                            )
                # Fresh history: scale steepest descent so the ladder's unit step moves the largest joint by
                # one radian - the raw gradient magnitude is arbitrary and would overshoot every trial.
                if n_hist == 0:
                    dir_inf = gs.qd_float(0.0)
                    for i_w in range(n_knots):
                        for i_dp in range(n_dp):
                            dir_inf = qd.max(dir_inf, qd.abs(plan_state.dir_traj[i_dp, col_base + i_w]))
                    scale_sd = 1.0 / qd.max(dir_inf, 1e-9)
                    for i_w in range(n_knots):
                        for i_dp in range(n_dp):
                            plan_state.dir_traj[i_dp, col_base + i_w] *= scale_sd
                # Initial scaling gamma = (s.y) / (y.y) of the most recent pair.
                if n_hist > 0:
                    i_h = (i_next - 1) % m_hist
                    dot_sy = gs.qd_float(0.0)
                    dot_yy = gs.qd_float(0.0)
                    for i_w in range(n_knots):
                        for i_dp in range(n_dp):
                            dot_sy += (
                                plan_state.lbfgs_s[i_h, i_dp, col_base + i_w]
                                * plan_state.lbfgs_y[i_h, i_dp, col_base + i_w]
                            )
                            dot_yy += plan_state.lbfgs_y[i_h, i_dp, col_base + i_w] ** 2
                    gamma = qd.math.clamp(dot_sy / qd.max(dot_yy, 1e-12), 1e-4, 1e2)
                    for i_w in range(n_knots):
                        for i_dp in range(n_dp):
                            plan_state.dir_traj[i_dp, col_base + i_w] *= gamma
                for i_h_ in range(n_hist):
                    i_h = (i_next - n_hist + i_h_) % m_hist
                    dot_yd = gs.qd_float(0.0)
                    for i_w in range(n_knots):
                        for i_dp in range(n_dp):
                            dot_yd += (
                                plan_state.lbfgs_y[i_h, i_dp, col_base + i_w]
                                * plan_state.dir_traj[i_dp, col_base + i_w]
                            )
                    beta_h = plan_state.lbfgs_rho[i_h, i_c] * dot_yd
                    for i_w in range(n_knots):
                        for i_dp in range(n_dp):
                            plan_state.dir_traj[i_dp, col_base + i_w] += (alphas[i_h] - beta_h) * plan_state.lbfgs_s[
                                i_h, i_dp, col_base + i_w
                            ]

                # Step-ladder line search on trial trajectories, jittered per (candidate, iteration).
                jitter = 0.8 + 0.4 * func_planner_hash01(plan_info.seed_key[None], i_c, it, 12345)
                cost_best = cost
                alpha_best = gs.qd_float(0.0)
                for i_t in range(qd.min(plan_info.ls_n_trials[None], array_class.PLANNER_LS_TRIALS_MAX)):
                    alpha = gs.qd_float(0.0)
                    for i_l in qd.static(range(len(_LS_LADDER))):
                        if i_l == i_t:
                            alpha = _LS_LADDER[i_l]
                    alpha *= jitter
                    for i_w in range(n_knots):
                        is_clamped = func_planner_mask_clamped(i_c, i_w, i_b, plan_info, planner_config)
                        for i_dp in range(n_dp):
                            q = plan_state.qpos_traj[i_dp, col_base + i_w]
                            if not is_clamped:
                                q = qd.math.clamp(
                                    q - alpha * plan_state.dir_traj[i_dp, col_base + i_w],
                                    plan_info.q_limit_lower[i_dp],
                                    plan_info.q_limit_upper[i_dp],
                                )
                            plan_state.trial_qpos[i_dp, col_base + i_w] = q
                    cost_trial = func_planner_traj_cost(
                        i_c,
                        i_b,
                        i_c,
                        col_base,
                        qpos_cols=plan_state.trial_qpos,
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
                        is_clamped = func_planner_mask_clamped(i_c, i_w, i_b, plan_info, planner_config)
                        for i_dp in range(n_dp):
                            q_old = plan_state.qpos_traj[i_dp, col_base + i_w]
                            plan_state.qpos_prev[i_dp, col_base + i_w] = q_old
                            plan_state.grad_prev[i_dp, col_base + i_w] = plan_state.grad_traj[i_dp, col_base + i_w]
                            if not is_clamped:
                                plan_state.qpos_traj[i_dp, col_base + i_w] = qd.math.clamp(
                                    q_old - alpha_best * plan_state.dir_traj[i_dp, col_base + i_w],
                                    plan_info.q_limit_lower[i_dp],
                                    plan_info.q_limit_upper[i_dp],
                                )
                    cost = func_planner_candidate_cost_grad(
                        i_c,
                        i_b,
                        plan_state=plan_state,
                        plan_info=plan_info,
                        plan_world=plan_world,
                        dyn_state=dyn_state,
                        dyn_info=dyn_info,
                        rigid_info=rigid_info,
                        sdf_info=sdf_info,
                        rigid_config=rigid_config,
                        collider_static_config=collider_static_config,
                        planner_config=planner_config,
                    )
                    for i_w in range(n_knots):
                        if func_planner_mask_clamped(i_c, i_w, i_b, plan_info, planner_config):
                            for i_dp in range(n_dp):
                                plan_state.grad_traj[i_dp, col_base + i_w] = 0.0
                    # Curvature-guarded history push.
                    dot_sy = gs.qd_float(0.0)
                    for i_w in range(n_knots):
                        for i_dp in range(n_dp):
                            s_e = (
                                plan_state.qpos_traj[i_dp, col_base + i_w] - plan_state.qpos_prev[i_dp, col_base + i_w]
                            )
                            y_e = (
                                plan_state.grad_traj[i_dp, col_base + i_w] - plan_state.grad_prev[i_dp, col_base + i_w]
                            )
                            plan_state.lbfgs_s[i_next % m_hist, i_dp, col_base + i_w] = s_e
                            plan_state.lbfgs_y[i_next % m_hist, i_dp, col_base + i_w] = y_e
                            dot_sy += s_e * y_e
                    if dot_sy > 1e-10:
                        plan_state.lbfgs_rho[i_next % m_hist, i_c] = 1.0 / dot_sy
                        i_next += 1
                        n_hist = qd.min(n_hist + 1, m_hist)
                it += 1

            plan_state.cost[i_c] = cost
