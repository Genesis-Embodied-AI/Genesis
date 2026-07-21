import torch

import genesis as gs


def retime_trajectory(qpos_knots, vel_limit, acc_limit, jerk_limit):
    """
    Batched TOPP-lite dt-scaling: pick the per-env knot spacing so the finite-difference velocity, acceleration,
    and jerk of the knot trajectory respect the per-DOF limits.

    Scaling time by k divides velocities by k, accelerations by k^2, and jerks by k^3, so the binding axis fixes
    k exactly; uniform scaling is conservative elsewhere, and the velocity-shaping cost keeps little on the table.

    Parameters
    ----------
    qpos_knots : torch.Tensor
        Knot positions [B, W, n_dp].
    vel_limit, acc_limit, jerk_limit : torch.Tensor
        Per-DOF limits [n_dp] (infinite entries are ignored).

    Returns
    -------
    dt_knot : torch.Tensor
        Per-env knot spacing in seconds [B].
    """
    vel_nom = qpos_knots.diff(dim=1).abs().amax(dim=1)
    acc_nom = qpos_knots.diff(n=2, dim=1).abs().amax(dim=1) if qpos_knots.shape[1] > 2 else torch.zeros_like(vel_nom)
    jerk_nom = qpos_knots.diff(n=3, dim=1).abs().amax(dim=1) if qpos_knots.shape[1] > 3 else torch.zeros_like(vel_nom)

    k_vel = torch.where(vel_limit.isfinite(), vel_nom / vel_limit, torch.zeros_like(vel_nom)).amax(dim=-1)
    k_acc = torch.where(acc_limit.isfinite(), acc_nom / acc_limit, torch.zeros_like(acc_nom)).amax(dim=-1).sqrt()
    k_jerk = (
        torch.where(jerk_limit.isfinite(), jerk_nom / jerk_limit, torch.zeros_like(jerk_nom))
        .amax(dim=-1)
        .pow(1.0 / 3.0)
    )
    return torch.maximum(torch.maximum(k_vel, k_acc), k_jerk).clamp(min=1e-3)


def resample_trajectory(qpos_knots, dt_knot, num_waypoints, scene_dt):
    """
    Resample the retimed knot trajectory to the output waypoints via cubic Hermite interpolation, with velocities
    and accelerations evaluated from the same polynomial.

    When num_waypoints is None the waypoints are spaced exactly at scene_dt (per-env duration, padded with a
    zero-velocity terminal hold so a shared [N, ...] batch shape holds); an integer keeps that exact count with
    the true per-env spacing returned.

    Returns
    -------
    qpos, dofs_vel, dofs_acc : torch.Tensor
        Waypoint positions, velocities, and accelerations [B, N, n_dp].
    dt : torch.Tensor
        Waypoint spacing in seconds [B].
    """
    B, W, n_dp = qpos_knots.shape
    duration = dt_knot * (W - 1)

    if num_waypoints is None:
        n_out = max(int(torch.ceil(duration.max() / scene_dt).item()) + 1, 2)
        dt_out = torch.full_like(dt_knot, scene_dt)
    else:
        n_out = num_waypoints
        dt_out = duration / (n_out - 1)

    # Knot velocities from central differences (zero at both ends, matching the clamped boundary knots).
    vel_knots = torch.zeros_like(qpos_knots)
    vel_knots[:, 1:-1] = (qpos_knots[:, 2:] - qpos_knots[:, :-2]) / (2.0 * dt_knot[:, None, None])

    # Segment lookup at the output timestamps (clamped to the duration: terminal hold).
    t_knots = torch.arange(W, dtype=gs.tc_float, device=gs.device)[None, :] * dt_knot[:, None]
    t_out = torch.arange(n_out, dtype=gs.tc_float, device=gs.device)[None, :] * dt_out[:, None]
    t_out = t_out.clamp(max=duration[:, None])
    i_seg = (torch.searchsorted(t_knots, t_out, right=True) - 1).clamp(min=0, max=W - 2)
    u = ((t_out - t_knots.gather(1, i_seg)) / dt_knot[:, None])[..., None]

    idx = i_seg[..., None].expand(B, n_out, n_dp)
    q0 = qpos_knots.gather(1, idx)
    q1 = qpos_knots.gather(1, idx + 1)
    # Hermite tangents in normalized segment time.
    v0 = vel_knots.gather(1, idx) * dt_knot[:, None, None]
    v1 = vel_knots.gather(1, idx + 1) * dt_knot[:, None, None]

    qpos = (
        (2 * u**3 - 3 * u**2 + 1) * q0 + (u**3 - 2 * u**2 + u) * v0 + (-2 * u**3 + 3 * u**2) * q1 + (u**3 - u**2) * v1
    )
    dofs_vel = (
        (6 * u**2 - 6 * u) * q0 + (3 * u**2 - 4 * u + 1) * v0 + (-6 * u**2 + 6 * u) * q1 + (3 * u**2 - 2 * u) * v1
    ) / dt_knot[:, None, None]
    dofs_acc = ((12 * u - 6) * q0 + (6 * u - 4) * v0 + (-12 * u + 6) * q1 + (6 * u - 2) * v1) / dt_knot[
        :, None, None
    ] ** 2

    # Terminal hold: past the duration the trajectory parks at the last knot at rest.
    is_held = (t_out >= duration[:, None])[..., None]
    qpos = torch.where(is_held, qpos_knots[:, -1:], qpos)
    dofs_vel = torch.where(is_held, torch.zeros_like(dofs_vel), dofs_vel)
    dofs_acc = torch.where(is_held, torch.zeros_like(dofs_acc), dofs_acc)
    return qpos, dofs_vel, dofs_acc, dt_out
