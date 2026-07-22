import torch

import genesis as gs

# Workspace deviation budget of the corner blends (meters). Each interior knot's velocity-direction change is
# smoothed by a parabolic blend whose peak deviation from the certified polyline is |dv| * T_blend / 8 per DOF;
# the corner speed cap below keeps the reach-weighted total within this budget, so a certified path's real
# penetration stays bounded by it. It spends part of the certified leak budget instead of charging the validator
# demand: boundary-contact allowances are exact at the certification margin and the path rests at both ends, so
# a uniform extra demand would flag every excused grasp contact while the blends never deviate there anyway.
_BLEND_DEV_MAX = 1.5e-3


def retime_trajectory(qpos_knots, vel_limit, acc_limit, dofs_reach, num_waypoints, scene_dt):
    """
    Batched retiming and resampling of the knot trajectory by velocity profiling along the knot polyline: the
    output positions follow the certified piecewise-linear path (what the validator sweeps), so retiming cannot
    overshoot the knots, and the only deviation is the corner blends' bounded one (see _BLEND_DEV_MAX).

    The profile is trapezoidal in arc length (max-norm of the per-step joint motion) with the speed varying at
    uniform acceleration within each step: per-step speed caps keep every DOF within vel_limit, a
    forward-backward pass over the knot speeds keeps the per-DOF acceleration along the steps within acc_limit
    from rest to rest, and each interior knot caps its speed so the parabolic blend that rotates the velocity
    across the direction change respects both acc_limit and the blend deviation budget.

    When num_waypoints is None the waypoints are spaced exactly at scene_dt (per-env duration, padded with a
    zero-velocity terminal hold so a shared [N, ...] batch shape holds); an integer keeps that exact count with
    the true per-env spacing returned.

    Parameters
    ----------
    qpos_knots : torch.Tensor
        Knot positions [B, W, n_dp].
    vel_limit, acc_limit : torch.Tensor
        Per-DOF limits [n_dp] (infinite entries are ignored).
    dofs_reach : torch.Tensor
        Per-DOF workspace reach bounds [n_dp] (see PlannerDofsInfo), weighting joint deviation into workspace.
    num_waypoints : int | None
        Output waypoint count, or None for scene_dt spacing.
    scene_dt : float
        Output spacing when num_waypoints is None.

    Returns
    -------
    qpos, dofs_vel, dofs_acc : torch.Tensor
        Waypoint positions, velocities, and accelerations [B, N, n_dp].
    dt : torch.Tensor
        Waypoint spacing in seconds [B].
    """
    B, W, n_dp = qpos_knots.shape
    steps = qpos_knots.diff(dim=1)
    arc = steps.abs().amax(dim=-1)
    is_moving = arc > gs.EPS
    arc_safe = arc.clamp(min=gs.EPS)
    # Unit path direction in arc time: per-DOF speed = |dir| * arc speed.
    direction = steps / arc_safe[..., None]

    # Per-step arc-speed cap from the per-DOF velocity limits, and the arc acceleration whose per-DOF
    # acceleration respects acc_limit along the step.
    vel_ratio = torch.where(vel_limit.isfinite(), direction.abs() / vel_limit, torch.zeros_like(direction))
    v_cap = 1.0 / vel_ratio.amax(dim=-1).clamp(min=gs.EPS)
    acc_ratio = torch.where(acc_limit.isfinite(), direction.abs() / acc_limit, torch.zeros_like(direction))
    a_step = 1.0 / acc_ratio.amax(dim=-1).clamp(min=gs.EPS)

    # Knot speed caps: rest at both ends, the adjacent steps' velocity caps in between. At a direction change
    # the blend duration keeping the per-DOF blend acceleration within acc_limit is T = v * r_acc, and its peak
    # deviation v * r_dev * T / 8, so the deviation budget fixes the corner speed in closed form. Degenerate
    # steps read direction zero, which only makes the caps conservative.
    v_node = torch.full((B, W), torch.inf, dtype=gs.tc_float, device=gs.device)
    v_node[:, 0] = 0.0
    v_node[:, -1] = 0.0
    v_node[:, 1:] = torch.minimum(v_node[:, 1:], v_cap)
    v_node[:, :-1] = torch.minimum(v_node[:, :-1], v_cap)
    ddir = (direction[:, 1:] - direction[:, :-1]).abs()
    r_acc = torch.where(acc_limit.isfinite(), ddir / acc_limit, torch.zeros_like(ddir)).amax(dim=-1)
    r_dev = (ddir * dofs_reach).sum(dim=-1)
    v_corner = (8.0 * _BLEND_DEV_MAX / (r_acc * r_dev).clamp(min=gs.EPS)).sqrt()
    v_node[:, 1:-1] = torch.minimum(v_node[:, 1:-1], v_corner)

    # Forward-backward pass over the knot speeds: v_next^2 <= v^2 + 2 a s per step (uniform acceleration along
    # the step, per-DOF acceleration within acc_limit by construction of a_step).
    accel_sq = 2.0 * a_step * arc
    v_sq = v_node.square()
    for i_seg in range(W - 1):
        v_sq[:, i_seg + 1] = torch.minimum(v_sq[:, i_seg + 1], v_sq[:, i_seg] + accel_sq[:, i_seg])
    for i_seg in range(W - 2, -1, -1):
        v_sq[:, i_seg] = torch.minimum(v_sq[:, i_seg], v_sq[:, i_seg + 1] + accel_sq[:, i_seg])
    v_node = v_sq.sqrt()

    # Step durations (uniform acceleration: dt = 2 s / (v0 + v1)) and the cumulative schedule; degenerate steps
    # take no time.
    v_pair = (v_node[:, :-1] + v_node[:, 1:]).clamp(min=gs.EPS)
    dt_step = torch.where(is_moving, 2.0 * arc / v_pair, torch.zeros_like(arc))
    t_knots = torch.zeros(B, W, dtype=gs.tc_float, device=gs.device)
    t_knots[:, 1:] = dt_step.cumsum(dim=1)
    duration = t_knots[:, -1]

    # Blend windows, centered on the interior knots: the acc-limited duration, shrunk to the adjacent step
    # durations so neighboring blends never overlap (shrinking only shrinks the deviation).
    t_blend = torch.zeros(B, W, dtype=gs.tc_float, device=gs.device)
    t_blend[:, 1:-1] = torch.minimum(v_node[:, 1:-1] * r_acc, torch.minimum(dt_step[:, :-1], dt_step[:, 1:]))
    # Velocity jump the blend rotates through, one vector per interior knot.
    dv_node = torch.zeros(B, W, n_dp, dtype=gs.tc_float, device=gs.device)
    dv_node[:, 1:-1] = (direction[:, 1:] - direction[:, :-1]) * v_node[:, 1:-1, None]

    if num_waypoints is None:
        n_out = max(int(torch.ceil(duration.max() / scene_dt).item()) + 1, 2)
        dt_out = torch.full_like(duration, scene_dt)
    else:
        n_out = num_waypoints
        dt_out = duration / (n_out - 1)

    # Uniform-time resample on the polyline (clamped to the duration: terminal hold at rest). Within a step the
    # arc progress is v0 tau + a tau^2 / 2 with the step's uniform acceleration.
    t_out = torch.arange(n_out, dtype=gs.tc_float, device=gs.device)[None, :] * dt_out[:, None]
    t_out = t_out.clamp(max=duration[:, None])
    i_seg = (torch.searchsorted(t_knots, t_out, right=True) - 1).clamp(min=0, max=W - 2)
    tau = t_out - t_knots.gather(1, i_seg)
    v_seg = v_node.gather(1, i_seg)
    arc_seg = arc.gather(1, i_seg)
    a_seg = (v_node.gather(1, i_seg + 1).square() - v_seg.square()) / (2.0 * arc_seg.clamp(min=gs.EPS))
    u = (((v_seg + 0.5 * a_seg * tau) * tau) / arc_seg.clamp(min=gs.EPS)).clamp(0.0, 1.0)[..., None]

    idx = i_seg[..., None].expand(B, n_out, n_dp)
    qpos = qpos_knots.gather(1, idx) * (1 - u) + qpos_knots.gather(1, idx + 1) * u
    dir_seg = direction.gather(1, idx)
    dofs_vel = dir_seg * (v_seg + a_seg * tau)[..., None]
    dofs_acc = dir_seg * a_seg[..., None]

    # Corner blends: an additive parabolic bump around each interior knot turns the velocity-direction step into
    # a linear rotation over the window - zero at both window ends, C0 velocity across the knot, peak deviation
    # |dv| * T / 8. A sample sits in at most one half-window per side (windows never overlap).
    for is_approaching, i_knot in ((True, i_seg + 1), (False, i_seg)):
        t_b = t_blend.gather(1, i_knot)
        t_b_safe = t_b.clamp(min=gs.EPS)
        dv = dv_node.gather(1, i_knot[..., None].expand(B, n_out, n_dp))
        if is_approaching:
            # Approaching the right knot: first blend half, tau_b in [0, T/2).
            tau_b = (0.5 * t_b - (t_knots.gather(1, i_seg + 1) - t_out)).clamp(min=0.0)
            is_blend = (tau_b > 0.0) & (t_out < duration[:, None])
            bump = tau_b.square() / (2.0 * t_b_safe)
            bump_vel = tau_b / t_b_safe
        else:
            # Leaving the left knot: second blend half, tau_b in [T/2, T].
            tau_b = 0.5 * t_b + tau
            is_blend = (tau < 0.5 * t_b) & (t_out < duration[:, None])
            bump = tau_b.square() / (2.0 * t_b_safe) - tau_b + 0.5 * t_b
            bump_vel = tau_b / t_b_safe - 1.0
        qpos = qpos + torch.where(is_blend[..., None], dv * bump[..., None], torch.zeros_like(dv))
        dofs_vel = dofs_vel + torch.where(is_blend[..., None], dv * bump_vel[..., None], torch.zeros_like(dv))
        dofs_acc = dofs_acc + torch.where(is_blend[..., None], dv / t_b_safe[..., None], torch.zeros_like(dv))

    is_held = (t_out >= duration[:, None])[..., None]
    dofs_vel = torch.where(is_held, torch.zeros_like(dofs_vel), dofs_vel)
    dofs_acc = torch.where(is_held, torch.zeros_like(dofs_acc), dofs_acc)
    return qpos, dofs_vel, dofs_acc, dt_out
