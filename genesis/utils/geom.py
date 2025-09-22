import math

import numpy as np
import numba as nb
import torch
import torch.nn.functional as F

import gstaichi as ti

import genesis as gs

# ------------------------------------------------------------------------------------
# ------------------------------------- taichi ----------------------------------------
# ------------------------------------------------------------------------------------


@ti.func
def ti_xyz_to_quat(xyz):
    """
    Convert intrinsic x-y-z Euler angles to quaternion.
    """
    ai, aj, ak = 0.5 * xyz[2], -0.5 * xyz[1], 0.5 * xyz[0]
    si, sj, sk = ti.sin(ai), ti.sin(aj), ti.sin(ak)
    ci, cj, ck = ti.cos(ai), ti.cos(aj), ti.cos(ak)
    cc, cs = ci * ck, ci * sk
    sc, ss = si * ck, si * sk

    quat = ti.Vector(
        [
            +cj * cc + sj * ss,
            +cj * cs - sj * sc,
            -cj * ss - sj * cc,
            +cj * sc - sj * cs,
        ],
        dt=gs.ti_float,
    )
    return quat


@ti.func
def ti_R_to_xyz(R):
    """
    Convert a rotation matrix into intrinsic x-y-z Euler angles.
    """
    xyz = ti.Vector.zero(gs.ti_float, 3)

    cy = ti.sqrt(R[2, 2] ** 2 + R[1, 2] ** 2)
    if cy > gs.EPS:
        xyz[0] = -ti.atan2(R[1, 2], R[2, 2])
        xyz[1] = -ti.atan2(-R[0, 2], cy)
        xyz[2] = -ti.atan2(R[0, 1], R[0, 0])
    else:
        xyz[0] = 0.0
        xyz[1] = -ti.atan2(-R[0, 2], cy)
        xyz[2] = -ti.atan2(-R[1, 0], R[1, 1])
    return xyz


@ti.func
def ti_rotvec_to_R(rotvec):
    R = ti.Matrix.identity(gs.ti_float, 3)

    angle = rotvec.norm()
    if angle > gs.EPS:
        c = ti.cos(angle)
        s = ti.sqrt(1.0 - c**2)
        t = 1.0 - c
        x, y, z = rotvec / angle

        R = ti.Matrix(
            [
                [t * x * x + c, t * x * y - z * s, t * x * z + y * s],
                [t * x * y + z * s, t * y * y + c, t * y * z - x * s],
                [t * x * z - y * s, t * y * z + x * s, t * z * z + c],
            ],
            dt=gs.ti_float,
        )

    return R


@ti.func
def ti_rotvec_to_quat(rotvec):
    quat = ti.Vector.zero(gs.ti_float, 4)

    theta = rotvec.norm()
    if theta > gs.EPS:
        theta_half = 0.5 * theta
        c, s = ti.cos(theta_half), ti.sin(theta_half)

        quat[0] = c
        xyz = s / theta * rotvec
        for i in ti.static(range(3)):
            quat[i + 1] = xyz[i]

        # First order quaternion normalization is accurate enough yet necessary
        quat *= 0.5 * (3.0 - quat.norm_sqr())
    else:
        quat[0] = 1.0

    return quat


@ti.func
def ti_quat_to_R(quat):
    """
    Converts quaternion to 3x3 rotation matrix.
    """
    R = ti.Matrix.identity(gs.ti_float, 3)

    d = quat.norm_sqr()
    if d > gs.EPS:
        s = 2.0 / d
        w, x, y, z = quat
        xs, ys, zs = x * s, y * s, z * s
        wx, wy, wz = w * xs, w * ys, w * zs
        xx, xy, xz = x * xs, x * ys, x * zs
        yy, yz, zz = y * ys, y * zs, z * zs

        R = ti.Matrix(
            [
                [1.0 - (yy + zz), xy - wz, xz + wy],
                [xy + wz, 1.0 - (xx + zz), yz - wx],
                [xz - wy, yz + wx, 1.0 - (xx + yy)],
            ],
            dt=gs.ti_float,
        )

    return R


@ti.func
def ti_quat_to_xyz(quat):
    """
    Convert a quaternion into intrinsic x-y-z Euler angles.
    """
    roll = gs.ti_float(0.0)
    pitch = gs.ti_float(0.0)
    yaw = gs.ti_float(0.0)

    quat_norm_sqr = quat.norm_sqr()
    if quat_norm_sqr > gs.EPS:
        s = 2.0 / quat_norm_sqr
        q_w, q_x, q_y, q_z = quat
        q_xs, q_ys, q_zs = q_x * s, q_y * s, q_z * s
        q_wx, q_wy, q_wz = q_w * q_xs, q_w * q_ys, q_w * q_zs
        q_xx, q_xy, q_xz = q_x * q_xs, q_x * q_ys, q_x * q_zs
        q_yy, q_yz, q_zz = q_y * q_ys, q_y * q_zs, q_z * q_zs

        sinycosp = q_wz - q_xy
        cosycosp = 1.0 - (q_yy + q_zz)
        cosp = ti.sqrt(cosycosp**2 + sinycosp**2)

        pitch = ti.atan2(q_xz + q_wy, cosp)
        if cosp > gs.EPS:
            roll = ti.atan2(q_wx - q_yz, 1.0 - (q_xx + q_yy))
            yaw = ti.atan2(sinycosp, cosycosp)
        else:
            yaw = ti.atan2(q_wz + q_xy, 1.0 - (q_xx + q_zz))

    return ti.Vector([roll, pitch, yaw], dt=gs.ti_float)


@ti.func
def ti_quat_to_rotvec(quat):
    q_w, q_x, q_y, q_z = quat
    rotvec = ti.Vector([q_x, q_y, q_z], dt=gs.ti_float)

    s2 = rotvec.norm()
    if s2 > gs.EPS:
        angle = 2.0 * ti.atan2(s2, ti.abs(q_w))
        inv_sinc = angle / s2
        rotvec = (-1.0 if q_w < 0.0 else 1.0) * inv_sinc * rotvec

    return rotvec


@ti.func
def ti_trans_quat_to_T(trans, quat):
    T = ti.Matrix.identity(gs.ti_float, 4)
    T[:3, :3] = ti_quat_to_R(quat)
    T[:3, 3] = trans
    return T


@ti.func
def ti_inv_quat(quat):
    return ti.Vector([quat[0], -quat[1], -quat[2], -quat[3]], dt=gs.ti_float)


@ti.func
def ti_quat_mul_axis(q, axis):
    return ti.Vector(
        [
            -q[1] * axis[0] - q[2] * axis[1] - q[3] * axis[2],
            +q[0] * axis[0] + q[2] * axis[2] - q[3] * axis[1],
            +q[0] * axis[1] + q[3] * axis[0] - q[1] * axis[2],
            +q[0] * axis[2] + q[1] * axis[1] - q[2] * axis[0],
        ]
    )


@ti.func
def ti_quat_mul(u, v):
    vu = u.outer_product(v)
    w = vu[0, 0] - vu[1, 1] - vu[2, 2] - vu[3, 3]
    x = vu[0, 1] + vu[1, 0] + vu[2, 3] - vu[3, 2]
    y = vu[0, 2] - vu[1, 3] + vu[2, 0] + vu[3, 1]
    z = vu[0, 3] + vu[1, 2] - vu[2, 1] + vu[3, 0]
    return ti.Vector([w, x, y, z], dt=gs.ti_float)


@ti.func
def ti_transform_quat_by_quat(v, u):
    """Transforms quat_v by quat_u.

    This is equivalent to quatmul(quat_u, quat_v) or R_u @ R_v
    """
    vec = ti_quat_mul(u, v)
    return vec.normalized()


@ti.func
def ti_transform_by_quat(v, quat):
    q_w, q_x, q_y, q_z = quat
    q_xx, q_xy, q_xz, q_wx = q_x * q_x, q_x * q_y, q_x * q_z, q_x * q_w
    q_yy, q_yz, q_wy = q_y * q_y, q_y * q_z, q_y * q_w
    q_zz, q_wz = q_z * q_z, q_z * q_w
    q_ww = q_w * q_w

    return ti.Vector(
        [
            v.x * (q_xx + q_ww - q_yy - q_zz) + v.y * (2.0 * q_xy - 2.0 * q_wz) + v.z * (2.0 * q_xz + 2.0 * q_wy),
            v.x * (2.0 * q_wz + 2.0 * q_xy) + v.y * (q_ww - q_xx + q_yy - q_zz) + v.z * (-2.0 * q_wx + 2.0 * q_yz),
            v.x * (-2.0 * q_wy + 2.0 * q_xz) + v.y * (2.0 * q_wx + 2.0 * q_yz) + v.z * (q_ww - q_xx - q_yy + q_zz),
        ],
        dt=gs.ti_float,
    ) / (q_ww + q_xx + q_yy + q_zz)


@ti.func
def ti_inv_transform_by_quat(v, quat):
    return ti_transform_by_quat(v, ti_inv_quat(quat))


@ti.func
def ti_transform_by_T(pos, T):
    return T[:3, :3] @ pos + T[:3, 3]


@ti.func
def ti_inv_transform_by_T(pos, T):
    return T[:3, :3].transpose() @ (pos - T[:3, 3])


@ti.func
def ti_transform_by_trans_quat(pos, trans, quat):
    return ti_transform_by_quat(pos, quat) + trans


@ti.func
def ti_inv_transform_by_trans_quat(pos, trans, quat):
    return ti_transform_by_quat(pos - trans, ti_inv_quat(quat))


@ti.func
def ti_transform_motion_by_trans_quat(m_ang, m_vel, trans, quat):
    quat_inv = ti_inv_quat(quat)
    ang = ti_transform_by_quat(m_ang, quat_inv)
    vel = ti_transform_by_quat(m_vel - trans.cross(m_ang), quat_inv)
    return ang, vel


@ti.func
def ti_inv_transform_motion_by_trans_quat(m_ang, m_vel, trans, quat):
    ang = ti_transform_by_quat(m_ang, quat)
    vel = ti_transform_by_quat(m_vel, quat) + trans.cross(ang)
    return ang, vel


@ti.func
def ti_transform_pos_quat_by_trans_quat(pos, quat, t_trans, t_quat):
    new_pos = t_trans + ti_transform_by_quat(pos, t_quat)
    new_quat = ti_transform_quat_by_quat(quat, t_quat)
    return new_pos, new_quat


@ti.func
def ti_transform_inertia_by_trans_quat(i_inertial, i_mass, trans, quat):
    x, y, z = trans.x, trans.y, trans.z
    xx, xy, xz, yy, yz, zz = x * x, x * y, x * z, y * y, y * z, z * z
    hhT = ti.Matrix(
        [
            [yy + zz, -xy, -xz],
            [-xy, xx + zz, -yz],
            [-xz, -yz, xx + yy],
        ]
    )

    R = ti_quat_to_R(quat)
    i = R @ i_inertial @ R.transpose() + hhT * i_mass
    trans = trans * i_mass

    return i, trans, quat, i_mass


@ti.func
def ti_normalize(v):
    return v / (v.norm(gs.EPS))


@ti.func
def ti_identity_quat():
    return ti.Vector([1.0, 0.0, 0.0, 0.0], dt=gs.ti_float)


@ti.func
def ti_vec3(val):
    return ti.Vector([val, val, val], dt=gs.ti_float)


@ti.func
def ti_nowhere():
    # let's inject a bit of humor here
    return ti.Vector([-2333333, -6666666, -5201314], dt=gs.ti_float)


@ti.func
def ti_tet_vol(p0, p1, p2, p3):
    return (p1 - p0).cross(p2 - p0).dot(p3 - p0) / 6.0


@ti.func
def inertial_mul(pos, i, mass, vel, ang):
    _ang = i @ ang + pos.cross(vel)
    _vel = mass * vel - pos.cross(ang)
    return _ang, _vel


@ti.func
def motion_cross_force(m_ang, m_vel, f_ang, f_vel):
    vel = m_ang.cross(f_vel)
    ang = m_ang.cross(f_ang) + m_vel.cross(f_vel)
    return ang, vel


@ti.func
def motion_cross_motion(s_ang, s_vel, m_ang, m_vel):
    vel = s_ang.cross(m_vel) + s_vel.cross(m_ang)
    ang = s_ang.cross(m_ang)
    return ang, vel


@ti.func
def ti_orthogonals(a):
    """
    Returns orthogonal vectors `b` and `c`, given a normal vector `a`.
    """
    b = ti.Vector.zero(gs.ti_float, 3)
    if ti.abs(a[1]) < 0.5:
        b[0] = -a[0] * a[1]
        b[1] = 1.0 - a[1] ** 2
        b[2] = -a[2] * a[1]
    else:
        b[0] = -a[0] * a[2]
        b[1] = -a[1] * a[2]
        b[2] = 1.0 - a[2] ** 2
    b = b.normalized()
    return b, a.cross(b)


@ti.func
def imp_aref(params, neg_penetration, vel, pos):
    timeconst, dampratio, dmin, dmax, width, mid, power = params

    imp_x = ti.abs(neg_penetration) / width
    imp_a = (1.0 / mid ** (power - 1)) * imp_x**power
    imp_b = 1.0 - (1.0 / (1.0 - mid) ** (power - 1)) * (1.0 - imp_x) ** power
    imp_y = imp_a if imp_x < mid else imp_b

    imp = dmin + imp_y * (dmax - dmin)
    imp = ti.math.clamp(imp, dmin, dmax)
    imp = dmax if imp_x > 1.0 else imp

    b = 2.0 / (dmax * timeconst)
    k = 1.0 / (dmax * dmax * timeconst * timeconst * dampratio * dampratio)

    aref = -b * vel - k * imp * pos

    return imp, aref


# ------------------------------------------------------------------------------------
# -------------------------------- torch and numpy -----------------------------------
# ------------------------------------------------------------------------------------


def inv_quat(quat):
    if isinstance(quat, torch.Tensor):
        _quat = quat.clone()
        _quat[..., 1:].neg_()
    elif isinstance(quat, np.ndarray):
        _quat = quat.copy()
        _quat[..., 1:] *= -1
    else:
        gs.raise_exception(f"the input must be either torch.Tensor or np.ndarray. got: {type(quat)=}")
    return _quat


def inv_T(T):
    if isinstance(T, torch.Tensor):
        T_inv = torch.zeros_like(T)
    elif isinstance(T, np.ndarray):
        T_inv = np.zeros_like(T)
    else:
        gs.raise_exception(f"the input must be torch.Tensor or np.ndarray. got: {type(T)=}")

    trans, R = T[..., :3, 3], T[..., :3, :3]
    T_inv[..., 3, 3] = 1.0
    T_inv[..., :3, 3] = -R.T @ trans
    T_inv[..., :3, :3] = R.T

    return T_inv


def normalize(x, eps: float = 1e-12):
    if isinstance(x, torch.Tensor):
        return x / x.norm(p=2, dim=-1).clamp(min=eps, max=None).unsqueeze(-1)
    elif isinstance(x, np.ndarray):
        return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), eps)
    else:
        gs.raise_exception(f"the input must be either torch.Tensor or np.ndarray. got: {type(x)=}")


def rot6d_to_R(d6):
    """
    Converts 6D rotation representation by Zhou et al. [1] to rotation matrix
    using Gram--Schmidt orthogonalization per Section B of [1].
    Args:
        d6: 6D rotation representation, of size (*, 6)

    Returns:
        batch of rotation matrices of size (*, 3, 3)

    [1] http://arxiv.org/abs/1812.07035
    """
    if isinstance(d6, torch.Tensor):
        a1, a2 = d6[..., :3], d6[..., 3:]
        b1 = F.normalize(a1, dim=-1)
        b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
        b2 = F.normalize(b2, dim=-1)
        b3 = torch.cross(b1, b2, dim=-1)
        return torch.stack((b1, b2, b3), dim=-2)
    elif isinstance(d6, np.ndarray):
        a1, a2 = d6[..., :3], d6[..., 3:]
        b1 = a1 / np.linalg.norm(a1, axis=-1, keepdims=True)
        dot = np.sum(b1 * a2, axis=-1, keepdims=True)
        b2 = a2 - dot * b1
        b2 = b2 / np.linalg.norm(b2, axis=-1, keepdims=True)
        b3 = np.cross(b1, b2, axis=-1)
        return np.stack((b1, b2, b3), axis=-2)
    else:
        gs.raise_exception(f"the input must be either torch.Tensor or np.ndarray. got: {type(d6)=}")


def R_to_rot6d(R):
    """
    Converts rotation matrices to 6D rotation representation by Zhou et al. [1]
    by dropping the last row. Note that 6D representation is not unique.
    Args:
        R: batch of rotation matrices of size (*, 3, 3)

    Returns:
        6D rotation representation, of size (*, 6)

    [1] http://arxiv.org/abs/1812.07035
    """
    if isinstance(R, torch.Tensor):
        return R[..., :2, :].flatten(start_dim=-2)
    elif isinstance(R, np.ndarray):
        return R[..., :2, :].reshape((*R.shape[:-2], 6))
    else:
        gs.raise_exception(f"the input must be either torch.Tensor or np.ndarray. got: {type(R)=}")


@nb.jit(nopython=True, cache=True)
def _np_xyz_to_quat(xyz: np.ndarray, rpy: bool = False, out: np.ndarray | None = None) -> np.ndarray:
    """Compute the (qw, qx, qy, qz) Quaternion representation of a single or a
    batch of Yaw-Pitch-Roll Euler angles.

    :param xyz: N-dimensional array whose first dimension gathers the 3
                Yaw-Pitch-Roll Euler angles [Roll, Pitch, Yaw].
    :param out: Pre-allocated array in which to store the result. If not
                provided, a new array is freshly-allocated and returned, which
                is slower.
    """
    assert xyz.ndim >= 1
    if out is None:
        out_ = np.empty((*xyz.shape[:-1], 4), dtype=xyz.dtype)
    else:
        assert out.shape == (*xyz.shape[:-1], 4)
        out_ = out

    rpy2 = 0.5 * xyz
    roll2, pitch2, yaw2 = rpy2[..., 0], rpy2[..., 1], rpy2[..., 2]
    cosr, sinr = np.cos(roll2), np.sin(roll2)
    cosp, sinp = np.cos(pitch2), np.sin(pitch2)
    cosy, siny = np.cos(yaw2), np.sin(yaw2)
    sign = 1.0 if rpy else -1.0

    out_[..., 0] = cosr * cosp * cosy + sign * sinr * sinp * siny
    out_[..., 1] = sinr * cosp * cosy - sign * cosr * sinp * siny
    out_[..., 2] = cosr * sinp * cosy + sign * sinr * cosp * siny
    out_[..., 3] = cosr * cosp * siny - sign * sinr * sinp * cosy

    return out_


def _tc_xyz_to_quat(xyz: torch.Tensor, rpy: bool = False, *, out: torch.Tensor | None = None) -> torch.Tensor:
    if out is None:
        out = torch.empty((*xyz.shape[:-1], 4), dtype=xyz.dtype, device=xyz.device)

    roll2, pitch2, yaw2 = (0.5 * xyz).unbind(-1)
    cosr, sinr = roll2.cos(), roll2.sin()
    cosp, sinp = pitch2.cos(), pitch2.sin()
    cosy, siny = yaw2.cos(), yaw2.sin()
    sign = 1.0 if rpy else -1.0

    out[..., 0] = cosr * cosp * cosy + sign * sinr * sinp * siny
    out[..., 1] = sinr * cosp * cosy - sign * cosr * sinp * siny
    out[..., 2] = cosr * sinp * cosy + sign * sinr * cosp * siny
    out[..., 3] = cosr * cosp * siny - sign * sinr * sinp * cosy

    return out


def xyz_to_quat(xyz, rpy=False, degrees=False):
    if isinstance(xyz, torch.Tensor):
        if degrees:
            xyz = torch.deg2rad(xyz)
        return _tc_xyz_to_quat(xyz, rpy)
    elif isinstance(xyz, np.ndarray):
        if degrees:
            xyz = np.deg2rad(xyz)
        return _np_xyz_to_quat(xyz, rpy)
    else:
        gs.raise_exception(f"the input must be either torch.Tensor or np.ndarray. got: {type(xyz)=}")


@nb.jit(nopython=True, cache=True)
def _np_quat_to_R(quat: np.ndarray, out: np.ndarray | None = None) -> np.ndarray:
    """Compute the Rotation Matrix representation of a single or a batch of quaternions.

    :param quat: N-dimensional array whose last dimension gathers the 4 quaternion coordinates (qw, qx, qy, qz).
    :param out: Pre-allocated array in which to store the result. If not provided, a new array is freshly-allocated and
                returned, which is slower.
    """
    assert quat.ndim >= 1
    if out is None:
        out_ = np.empty((*quat.shape[:-1], 3, 3), dtype=quat.dtype)
    else:
        assert out.shape == (*quat.shape[:-1], 3, 3)
        out_ = out

    s = 2.0 / np.sum(np.square(quat), -1)
    q_w, q_x, q_y, q_z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    q_xs, q_ys, q_zs = q_x * s, q_y * s, q_z * s
    q_wx, q_wy, q_wz = q_w * q_xs, q_w * q_ys, q_w * q_zs
    q_xx, q_xy, q_xz = q_x * q_xs, q_x * q_ys, q_x * q_zs
    q_yy, q_yz = q_y * q_ys, q_y * q_zs
    q_zz = q_z * q_zs

    out_[..., 0, 0] = 1.0 - (q_yy + q_zz)
    out_[..., 0, 1] = q_xy - q_wz
    out_[..., 0, 2] = q_xz + q_wy
    out_[..., 1, 0] = q_xy + q_wz
    out_[..., 1, 1] = 1.0 - (q_xx + q_zz)
    out_[..., 1, 2] = q_yz - q_wx
    out_[..., 2, 0] = q_xz - q_wy
    out_[..., 2, 1] = q_yz + q_wx
    out_[..., 2, 2] = 1.0 - (q_xx + q_yy)

    return out_


def _tc_quat_to_R(quat, out=None):
    if out is None:
        out = torch.empty((*quat.shape[:-1], 3, 3), dtype=quat.dtype, device=quat.device)

    q_w, q_x, q_y, q_z = torch.tensor_split(quat, 4, dim=-1)

    s = 2.0 / (quat**2).sum(dim=-1, keepdim=True)
    q_vec_s = s * quat[..., 1:]
    q_wx, q_wy, q_wz = torch.unbind(q_w * q_vec_s, -1)
    q_xx, q_xy, q_xz = torch.unbind(q_x * q_vec_s, -1)
    q_yy, q_yz = torch.unbind(q_y * q_vec_s[..., 1:], -1)
    q_zz = q_z[..., 0] * q_vec_s[..., -1]

    out[..., 0, 0] = 1.0 - (q_yy + q_zz)
    out[..., 0, 1] = q_xy - q_wz
    out[..., 0, 2] = q_xz + q_wy
    out[..., 1, 0] = q_xy + q_wz
    out[..., 1, 1] = 1.0 - (q_xx + q_zz)
    out[..., 1, 2] = q_yz - q_wx
    out[..., 2, 0] = q_xz - q_wy
    out[..., 2, 1] = q_yz + q_wx
    out[..., 2, 2] = 1.0 - (q_xx + q_yy)

    return out


def quat_to_R(quat, *, out=None):
    # NOTE: Ignore zero-norm quaternion for efficiency

    if all(isinstance(e, torch.Tensor) for e in (quat, out) if e is not None):
        return _tc_quat_to_R(quat, out=out)
    elif all(isinstance(e, np.ndarray) for e in (quat, out) if e is not None):
        return _np_quat_to_R(quat, out=out)
    else:
        gs.raise_exception(f"the input must be either torch.Tensor or np.ndarray. got: {type(quat)=}")


@nb.jit(nopython=True, cache=True)
def _np_quat_to_xyz(quat, rpy=False, out=None):
    """Compute the Yaw-Pitch-Roll Euler angles representation of a single or a batch of quaternions.

    The Roll, Pitch and Yaw angles are guaranteed to be within range [-pi,pi], [-pi/2,pi/2], [-pi,pi], respectively.

    :param quat: N-dimensional array whose last dimension gathers the 4 quaternion coordinates (qw, qx, qy, qz).
    :param out: Pre-allocated array in which to store the result. If not provided, a new array is freshly-allocated
                and returned, which is slower.
    """
    assert quat.ndim >= 1
    if out is None:
        out_ = np.empty((*quat.shape[:-1], 3), dtype=quat.dtype)
    else:
        assert out.shape == (*quat.shape[:-1], 3)
        out_ = out

    # Flatten batch dimensions
    quat_2d = quat.reshape((-1, 4))
    out_2d = out_.reshape((-1, 3))

    s = 2.0 / np.sum(np.square(quat_2d), -1)
    q_w, q_x, q_y, q_z = quat_2d[:, 0], quat_2d[:, 1], quat_2d[:, 2], quat_2d[:, 3]
    q_xs, q_ys, q_zs = q_x * s, q_y * s, q_z * s
    q_wx, q_wy, q_wz = q_w * q_xs, q_w * q_ys, q_w * q_zs
    q_xx, q_xy, q_xz = q_x * q_xs, q_x * q_ys, q_x * q_zs
    q_yy, q_yz, q_zz = q_y * q_ys, q_y * q_zs, q_z * q_zs

    if rpy:
        sinp = q_wy - q_xz
        sinrcosp = q_wx + q_yz
        sinycosp = q_wz + q_xy
    else:
        sinp = q_xz + q_wy
        sinrcosp = q_wx - q_yz
        sinycosp = q_wz - q_xy
    cosr_cosp = 1.0 - (q_xx + q_yy)
    cosycosp = 1.0 - (q_yy + q_zz)
    cosp = np.sqrt(cosycosp**2 + sinycosp**2)

    out_2d[:, 0] = np.arctan2(sinrcosp, cosr_cosp)
    out_2d[:, 1] = np.arctan2(sinp, cosp)
    out_2d[:, 2] = np.arctan2(sinycosp, cosycosp)

    cosp_mask = cosp < gs.EPS
    if rpy:
        sinycosp_sinrsinpcosy = q_wz[cosp_mask] - q_xy[cosp_mask]
    else:
        sinycosp_sinrsinpcosy = q_wz[cosp_mask] + q_xy[cosp_mask]
    cospcosy_sinrsinpsiny = 1.0 - (q_xx[cosp_mask] + q_zz[cosp_mask])
    out_2d[cosp_mask, 0] = 0.0
    out_2d[cosp_mask, 2] = np.arctan2(sinycosp_sinrsinpcosy, cospcosy_sinrsinpsiny)

    return out_


def _tc_quat_to_xyz(quat, rpy=False, out=None):
    if out is None:
        out = torch.empty((*quat.shape[:-1], 3), dtype=quat.dtype, device=quat.device)

    # Extract quaternion components
    q_w, q_x, q_y, q_z = torch.tensor_split(quat, 4, dim=-1)

    s = 2.0 / (quat**2).sum(dim=-1, keepdim=True)
    q_vec_s = s * quat[..., 1:]
    q_wx, q_wy, q_wz = torch.unbind(q_w * q_vec_s, -1)
    q_xx, q_xy, q_xz = torch.unbind(q_x * q_vec_s, -1)
    q_yy, q_yz = torch.unbind(q_y * q_vec_s[..., 1:], -1)
    q_zz = q_z[..., 0] * q_vec_s[..., 2]

    # Compute some intermediary quantities.
    # Numerical robustness of 'cos(pitch)' could be improved using 'hypot' implementation from Eigen:
    # https://gitlab.com/libeigen/eigen/-/blob/master/Eigen/src/Core/MathFunctionsImpl.h#L149
    if rpy:
        sinp = q_wy - q_xz
        sinrcosp = q_wx + q_yz
        sinycosp = q_wz + q_xy
    else:
        sinp = q_xz + q_wy
        sinrcosp = q_wx - q_yz
        sinycosp = q_wz - q_xy
    cosr_cosp = 1.0 - (q_xx + q_yy)
    cosycosp = 1.0 - (q_yy + q_zz)
    cosp = torch.sqrt(cosycosp**2 + sinycosp**2)

    # Roll (x-axis rotation)
    out[..., 0] = torch.atan2(sinrcosp, cosr_cosp)

    # Pitch (y-axis rotation)
    out[..., 1] = torch.atan2(sinp, cosp)

    # Yaw (z-axis rotation)
    out[..., 2] = torch.atan2(sinycosp, cosycosp)

    # Special treatment of nearly singular rotations
    cosp_mask = cosp < gs.EPS
    if cosp_mask.any():
        if rpy:
            sinycosp_sinrsinpcosy = q_wz[cosp_mask] - q_xy[cosp_mask]
        else:
            sinycosp_sinrsinpcosy = q_wz[cosp_mask] + q_xy[cosp_mask]
        cospcosy_sinrsinpsiny = 1.0 - (q_xx[cosp_mask] + q_zz[cosp_mask])

        out[..., 0].masked_fill_(cosp_mask, 0.0)
        out[cosp_mask, 2] = torch.arctan2(sinycosp_sinrsinpcosy, cospcosy_sinrsinpsiny)

    return out


def quat_to_xyz(quat, rpy=False, degrees=False):
    if isinstance(quat, torch.Tensor):
        rpy = _tc_quat_to_xyz(quat, rpy)
        if degrees:
            rpy = torch.rad2deg(rpy)
    elif isinstance(quat, np.ndarray):
        rpy = _np_quat_to_xyz(quat, rpy)
        if degrees:
            rpy = np.rad2deg(rpy)
    else:
        gs.raise_exception(f"the input must be either torch.Tensor or np.ndarray. got: {type(quat)=}")
    return rpy


@nb.jit(nopython=True, cache=True)
def _np_R_to_quat(R, out=None):
    assert R.ndim >= 2

    if out is None:
        out_ = np.empty((*R.shape[:-2], 4), dtype=R.dtype)
    else:
        assert out.shape == (*R.shape[:-2], 4)
        out_ = out

    for i in np.ndindex(R.shape[:-2]):
        R_i = R[i]
        quat_i = out_[i]

        if R_i[2, 2] < 0.0:
            if R_i[0, 0] > R_i[1, 1]:
                t = 1.0 + R_i[0, 0] - R_i[1, 1] - R_i[2, 2]
                quat_i[0] = R_i[2, 1] - R_i[1, 2]
                quat_i[1] = t
                quat_i[2] = R_i[1, 0] + R_i[0, 1]
                quat_i[3] = R_i[0, 2] + R_i[2, 0]
            else:
                t = 1.0 - R_i[0, 0] + R_i[1, 1] - R_i[2, 2]
                quat_i[0] = R_i[0, 2] - R_i[2, 0]
                quat_i[1] = R_i[1, 0] + R_i[0, 1]
                quat_i[2] = t
                quat_i[3] = R_i[2, 1] + R_i[1, 2]
        else:
            if R_i[0, 0] < -R_i[1, 1]:
                t = 1.0 - R_i[0, 0] - R_i[1, 1] + R_i[2, 2]
                quat_i[0] = R_i[1, 0] - R_i[0, 1]
                quat_i[1] = R_i[0, 2] + R_i[2, 0]
                quat_i[2] = R_i[2, 1] + R_i[1, 2]
                quat_i[3] = t
            else:
                t = 1.0 + R_i[0, 0] + R_i[1, 1] + R_i[2, 2]
                quat_i[0] = t
                quat_i[1] = R_i[2, 1] - R_i[1, 2]
                quat_i[2] = R_i[0, 2] - R_i[2, 0]
                quat_i[3] = R_i[1, 0] - R_i[0, 1]
        quat_i /= 2.0 * np.sqrt(t)

    return out_


def _tc_R_to_quat(R, out=None):
    if out is None:
        out = torch.zeros((*R.shape[:-2], 4), dtype=R.dtype, device=R.device)

    # Flattening batch dimensions because multi-dimensional masking is acting weird
    out_ = out.reshape((-1, 4))
    R = R.reshape((-1, 3, 3))

    diag = torch.diagonal(R, dim1=-2, dim2=-1)
    trace = diag.sum(-1)

    # Compute quaternion based on the trace of the matrix
    mask1 = trace > 0.0
    mask2 = ~mask1 & (diag[:, 0] >= diag[:, 1]) & (diag[:, 0] >= diag[:, 2])
    mask3 = ~mask1 & ~mask2 & (diag[:, 1] >= diag[:, 2])
    mask4 = ~mask1 & ~mask2 & ~mask3

    S = 2.0 * torch.sqrt(trace[mask1] + 1.0)
    out_[mask1, 0] = 0.25 * S
    out_[mask1, 1] = (R[mask1, 2, 1] - R[mask1, 1, 2]) / S
    out_[mask1, 2] = (R[mask1, 0, 2] - R[mask1, 2, 0]) / S
    out_[mask1, 3] = (R[mask1, 1, 0] - R[mask1, 0, 1]) / S

    S = 2.0 * torch.sqrt(1.0 + diag[mask2, 0] - diag[mask2, 1] - diag[mask2, 2])
    out_[mask2, 0] = (R[mask2, 2, 1] - R[mask2, 1, 2]) / S
    out_[mask2, 1] = 0.25 * S
    out_[mask2, 2] = (R[mask2, 0, 1] + R[mask2, 1, 0]) / S
    out_[mask2, 3] = (R[mask2, 0, 2] + R[mask2, 2, 0]) / S

    S = 2.0 * torch.sqrt(1.0 + diag[mask3, 1] - diag[mask3, 0] - diag[mask3, 2])
    out_[mask3, 0] = (R[mask3, 0, 2] - R[mask3, 2, 0]) / S
    out_[mask3, 1] = (R[mask3, 0, 1] + R[mask3, 1, 0]) / S
    out_[mask3, 2] = 0.25 * S
    out_[mask3, 3] = (R[mask3, 1, 2] + R[mask3, 2, 1]) / S

    S = 2.0 * torch.sqrt(1.0 + diag[mask4, 2] - diag[mask4, 0] - diag[mask4, 1])
    out_[mask4, 0] = (R[mask4, 1, 0] - R[mask4, 0, 1]) / S
    out_[mask4, 1] = (R[mask4, 0, 2] + R[mask4, 2, 0]) / S
    out_[mask4, 2] = (R[mask4, 1, 2] + R[mask4, 2, 1]) / S
    out_[mask4, 3] = 0.25 * S

    return out


def R_to_quat(R, *, out=None):
    if all(isinstance(e, torch.Tensor) for e in (R, out) if e is not None):
        return _tc_R_to_quat(R, out=out)
    elif all(isinstance(e, np.ndarray) for e in (R, out) if e is not None):
        return _np_R_to_quat(R, out=out)
    else:
        gs.raise_exception(f"the input must be either torch.Tensor or np.ndarray. got: {type(R)=}")


def trans_R_to_T(trans=None, R=None, *, out=None):
    is_torch = all(isinstance(e, torch.Tensor) for e in (trans, R) if e is not None)
    is_numpy = not is_torch and all(isinstance(e, np.ndarray) for e in (trans, R) if e is not None)

    T = out
    B = () if trans is None and R is None else R.shape[:-2] if trans is None else trans.shape[:-1]
    dtype = R.dtype if trans is None else trans.dtype
    if is_torch:
        if T is None:
            device = R.device if trans is None else trans.device
            T = torch.zeros((*B, 4, 4), dtype=dtype, device=device)
    elif is_numpy:
        if T is None:
            T = np.zeros((*B, 4, 4), dtype=dtype)
    else:
        gs.raise_exception(f"both of the inputs must be torch.Tensor or np.ndarray. got: {type(trans)=} and {type(R)=}")
    if B:
        assert T.shape == (*B, 4, 4)

    T[..., 3, 3] = 1.0
    if trans is not None:
        T[..., :3, 3] = trans
    if R is None:
        if is_torch:
            torch.diagonal(T, dim1=-2, dim2=-1).fill_(1.0)
        else:
            T[..., [0, 1, 2], [0, 1, 2]] = 1.0
    else:
        T[..., :3, :3] = R

    return T


def R_to_T(R, *, out=None):
    return trans_R_to_T(None, R, out=out)


def trans_to_T(trans, *, out=None):
    return trans_R_to_T(trans, None, out=out)


def trans_quat_to_T(trans=None, quat=None, *, out=None):
    is_torch = all(isinstance(e, torch.Tensor) for e in (trans, quat) if e is not None)
    is_numpy = not is_torch and all(isinstance(e, np.ndarray) for e in (trans, quat) if e is not None)

    T = out
    B = () if trans is None and quat is None else quat.shape[:-2] if trans is None else trans.shape[:-1]
    if is_torch:
        if T is None:
            T = torch.zeros((*B, 4, 4), dtype=trans.dtype, device=trans.device)
    elif is_numpy:
        if T is None:
            T = np.zeros((*B, 4, 4), dtype=trans.dtype)
    else:
        gs.raise_exception(
            f"both of the inputs must be torch.Tensor or np.ndarray. got: {type(trans)=} and {type(quat)=}"
        )
    if B:
        assert T.shape == (*B, 4, 4)

    T[..., 3, 3] = 1.0
    if trans is not None:
        T[..., :3, 3] = trans
    if quat is not None:
        quat_to_R(quat, out=T[..., :3, :3])

    return T


def T_to_trans_quat(T, *, out=None):
    trans = T[..., :3, 3]
    quat = R_to_quat(T[..., :3, :3])
    return trans, quat


@nb.jit(nopython=True, cache=True)
def _np_quat_mul(u, v, out=None):
    assert u.shape == v.shape

    if out is None:
        out_ = np.empty(u.shape, dtype=u.dtype)
    else:
        assert out.shape == u.shape
        out_ = out

    w1, x1, y1, z1 = u[..., 0], u[..., 1], u[..., 2], u[..., 3]
    w2, x2, y2, z2 = v[..., 0], v[..., 1], v[..., 2], v[..., 3]
    ww = (z1 + x1) * (x2 + y2)
    yy = (w1 - y1) * (w2 + z2)
    zz = (w1 + y1) * (w2 - z2)
    xx = ww + yy + zz
    qq = 0.5 * (xx + (z1 - x1) * (x2 - y2))

    out_[..., 0] = qq - ww + (z1 - y1) * (y2 - z2)
    out_[..., 1] = qq - xx + (x1 + w1) * (x2 + w2)
    out_[..., 2] = qq - yy + (w1 - x1) * (y2 + z2)
    out_[..., 3] = qq - zz + (z1 + y1) * (w2 - x2)

    return out_


def _tc_quat_mul(u, v, out=None):
    if out is None:
        out = torch.empty(torch.broadcast_shapes(u.shape, v.shape), dtype=v.dtype, device=v.device)

    w1, x1, y1, z1 = u[..., 0], u[..., 1], u[..., 2], u[..., 3]
    w2, x2, y2, z2 = v[..., 0], v[..., 1], v[..., 2], v[..., 3]
    ww = (z1 + x1) * (x2 + y2)
    yy = (w1 - y1) * (w2 + z2)
    zz = (w1 + y1) * (w2 - z2)
    xx = ww + yy + zz
    qq = 0.5 * (xx + (z1 - x1) * (x2 - y2))

    out[..., 0] = qq - ww + (z1 - y1) * (y2 - z2)
    out[..., 1] = qq - xx + (x1 + w1) * (x2 + w2)
    out[..., 2] = qq - yy + (w1 - x1) * (y2 + z2)
    out[..., 3] = qq - zz + (z1 + y1) * (w2 - x2)

    return out


def transform_quat_by_quat(v, u):
    """
    This method transforms quat_v by quat_u.

    This is equivalent to quatmul(quat_u, quat_v) or R_u @ R_v
    """
    assert u.ndim >= 1

    if all(isinstance(e, torch.Tensor) for e in (u, v)):
        quat = _tc_quat_mul(u, v)
    elif all(isinstance(e, np.ndarray) for e in (u, v)):
        quat = _np_quat_mul(u, v, out=None)
    else:
        gs.raise_exception(f"The inputs must all be torch.Tensor or np.ndarray. got: {type(v)=} and {type(quat)=}")

    return normalize(quat)


@nb.jit(nopython=True, cache=True)
def _np_transform_by_quat(v, quat, out=None):
    if out is None:
        out_ = np.empty(v.shape, dtype=v.dtype)
    else:
        assert out.shape == v.shape
        out_ = out

    v_T, quat_T, out_T = v.T, quat.T, out_.T
    v_x, v_y, v_z = v_T
    q_ww, q_wx, q_wy, q_wz = quat_T[0] * quat_T
    q_xx, q_xy, q_xz = quat_T[1] * quat_T[1:]
    q_yy, q_yz = quat_T[2] * quat_T[2:]
    q_zz = quat_T[3] * quat_T[3]

    out_T[0] = v_x * (q_xx + q_ww - q_yy - q_zz) + v_y * (2.0 * q_xy - 2.0 * q_wz) + v_z * (2.0 * q_xz + 2.0 * q_wy)
    out_T[1] = v_x * (2.0 * q_wz + 2.0 * q_xy) + v_y * (q_ww - q_xx + q_yy - q_zz) + v_z * (2.0 * q_yz - 2.0 * q_wx)
    out_T[2] = v_x * (2.0 * q_xz - 2.0 * q_wy) + v_y * (2.0 * q_wx + 2.0 * q_yz) + v_z * (q_ww - q_xx - q_yy + q_zz)

    out_T /= q_ww + q_xx + q_yy + q_zz

    return out_


def _tc_transform_by_quat(v, quat, out=None):
    if out is None:
        out = torch.empty(v.shape, dtype=v.dtype, device=v.device)

    v_x, v_y, v_z = torch.unbind(v, dim=-1)
    q_w, q_x, q_y, q_z = torch.tensor_split(quat, 4, dim=-1)
    q_ww, q_wx, q_wy, q_wz = torch.unbind(q_w * quat, -1)
    q_xx, q_xy, q_xz = torch.unbind(q_x * quat[..., 1:], -1)
    q_yy, q_yz = torch.unbind(q_y * quat[..., 2:], -1)
    q_zz = q_z[..., 0] * quat[..., 3]

    out[..., 0] = v_x * (q_xx + q_ww - q_yy - q_zz) + v_y * (2.0 * q_xy - 2.0 * q_wz) + v_z * (2.0 * q_xz + 2.0 * q_wy)
    out[..., 1] = v_x * (2.0 * q_wz + 2.0 * q_xy) + v_y * (q_ww - q_xx + q_yy - q_zz) + v_z * (2.0 * q_yz - 2.0 * q_wx)
    out[..., 2] = v_x * (2.0 * q_xz - 2.0 * q_wy) + v_y * (2.0 * q_wx + 2.0 * q_yz) + v_z * (q_ww - q_xx - q_yy + q_zz)

    out /= (q_ww + q_xx + q_yy + q_zz)[..., None]

    return out


def transform_by_quat(v, quat):
    """
    This method transforms quat_v by quat_u.

    This is equivalent to quatmul(quat_u, quat_v) or R_u @ R_v
    """
    assert v.ndim >= 1 and quat.ndim >= 1

    if all(isinstance(e, torch.Tensor) for e in (v, quat)):
        return _tc_transform_by_quat(v, quat)
    elif all(isinstance(e, np.ndarray) for e in (v, quat)):
        return _np_transform_by_quat(v, quat, out=None)
    else:
        gs.raise_exception(f"The inputs must all be torch.Tensor or np.ndarray. got: {type(v)=} and {type(quat)=}")


def axis_angle_to_quat(angle, axis):
    if isinstance(angle, torch.Tensor) and isinstance(axis, torch.Tensor):
        theta = (0.5 * angle).unsqueeze(-1)
        xyz = normalize(axis) * theta.sin()
        w = theta.cos()
        return normalize(torch.cat([w, xyz], dim=-1))
    elif isinstance(angle, np.ndarray) and isinstance(axis, np.ndarray):
        theta = (0.5 * angle)[..., None]
        xyz = normalize(axis) * np.sin(theta)
        w = np.cos(theta)
        return normalize(np.concatenate([w, xyz], axis=-1))
    else:
        gs.raise_exception(
            f"both of the inputs must be torch.Tensor or np.ndarray. got: {type(angle)=} and {type(axis)=}"
        )


def transform_by_xyz(pos, xyz):
    return transform_by_quat(pos, xyz_to_quat(xyz))


def transform_by_trans_quat(pos, trans, quat):
    return transform_by_quat(pos, quat) + trans


def inv_transform_by_quat(pos, quat):
    return transform_by_quat(pos, inv_quat(quat))


def inv_transform_by_trans_quat(pos, trans, quat):
    return inv_transform_by_quat(pos - trans, quat)


def transform_pos_quat_by_trans_quat(pos, quat, t_trans, t_quat):
    new_pos = t_trans + transform_by_quat(pos, t_quat)
    new_quat = transform_quat_by_quat(quat, t_quat)
    return new_pos, new_quat


def transform_by_R(pos, R):
    """
    Transforms 3D points by a 3x3 rotation matrix or a batch of matrices, supporting both NumPy arrays and PyTorch
    tensors.

    Parameters
    ----------
    pos: np.ndarray | torch.Tensor
        A numpy array or torch tensor of 3D points. Can be a single point (3,), a batch of points (B, 3), or a batched
        batch of points (B, N, 3).
    R: np.ndarray | torch.Tensor
        The 3x3 rotation matrix or a batch of B rotation matrices of shape (B, 3, 3). Must be of the same type as `pos`.

    Returns
    -------
        The transformed points in a shape corresponding to the input dimensions.
    """
    assert pos.shape[-1] == 3 and R.shape[-2:] == (3, 3)
    assert R.ndim - pos.ndim in (0, 1)

    B = R.shape[:-2]
    N = pos.shape[-2] if pos.ndim == R.ndim else 1
    R = R.reshape((-1, 3, 3))
    pos_ = pos.reshape((math.prod(B), N, 3))

    if all(isinstance(e, torch.Tensor) for e in (pos, R) if e is not None):
        new_pos = torch.bmm(R, pos_.swapaxes(1, 2)).swapaxes(1, 2)
    elif all(isinstance(e, np.ndarray) for e in (pos, R) if e is not None):
        new_pos = np.matmul(R, pos_.swapaxes(1, 2)).swapaxes(1, 2)
    else:
        gs.raise_exception(f"both of the inputs must be torch.Tensor or np.ndarray. got: {type(pos)=} and {type(R)=}")

    new_pos = new_pos.reshape(pos.shape)

    return new_pos


def transform_by_trans_R(pos, trans, R):
    assert trans.shape[:-1] == R.shape[:-2]

    B = trans.shape[:-1]
    if trans.ndim < pos.ndim:
        trans = trans[..., None, :]

    new_pos = transform_by_R(pos, R)
    new_pos += trans

    return new_pos


def transform_by_T(pos, T):
    """
    Transforms 3D points by a 4x4 transformation matrix or a batch of matrices, supporting both NumPy arrays and
    PyTorch tensors.

    Parameters
    ----------
    pos: np.ndarray | torch.Tensor
        A numpy array or torch tensor of 3D points. Can be a single point (3,), a batch of points (B, 3), or a
        batched batch of points (B, N, 3).
    T: np.ndarray | torch.Tensor
        The 4x4 transformation matrix or a batch of B transformation matrices of shape (B, 4, 4). Must be of the
        same type as `pos`.

    Returns
    -------
        The transformed points in a shape corresponding to the input dimensions.
    """
    return transform_by_trans_R(pos, T[..., :3, 3], T[..., :3, :3])


def inv_transform_by_T(pos, T):
    trans, R = T[..., :3, 3], T[..., :3, :3]

    R_inv = R.swapaxes(-1, -2)
    if pos.ndim == T.ndim:
        trans = trans.reshape((-1, 1, 3))

    return transform_by_R(pos - trans, R_inv)


# ------------------------------------------------------------------------------------
# ------------------------------------- numpy ----------------------------------------
# ------------------------------------------------------------------------------------


def scale_to_T(scale):
    T = np.eye(4, dtype=scale.dtype)
    T[[0, 1, 2], [0, 1, 2]] = scale
    return T


@nb.jit(nopython=True, cache=True)
def _np_z_up_to_R(z, up=None, out=None):
    B = z.shape[:-1]
    if out is None:
        out_ = np.empty((*B, 3, 3), dtype=z.dtype)
    else:
        assert out.shape == (*B, 3, 3)
        out_ = out

    z_norm = np.sqrt(np.sum(np.square(z.reshape((-1, 3))), -1)).reshape(B)

    out_[..., 2] = z
    for i in np.ndindex(B):
        z_norm_i = z_norm[i]
        R = out_[i]
        x, y, z = R.T

        if z_norm_i > gs.EPS:
            z /= z_norm_i
        else:
            if up is None or abs(up[i][1]) < 0.5:
                z[:] = 0.0, 1.0, 0.0
            else:
                z[:] = 0.0, 0.0, 1.0

        if up is not None:
            x[:] = np.cross(up[i], z)
        else:
            if abs(z[2]) < 1.0 - gs.EPS:
                # up = (0.0, 0.0, 1.0)
                x[0] = z[1]
                x[1] = -z[0]
                x[2] = 0.0
            else:
                # up = (0.0, 1.0, 0.0)
                x[0] = z[2]
                x[1] = 0.0
                x[2] = -z[0]

        x_norm = np.linalg.norm(x)
        if x_norm > gs.EPS:
            x /= x_norm
            y[:] = np.cross(z, x)
        else:
            # This would only occurs if the user specified non-zero colinear z and up
            R[:] = np.eye(3, dtype=R.dtype)

    return out_


def _tc_z_up_to_R(z, up=None, out=None):
    B = z.shape[:-1]
    if out is None:
        R = torch.empty((*B, 3, 3), dtype=z.dtype, device=z.device)
    else:
        assert out.shape == (*B, 3, 3)
        R = out

    # Set z as the third column of rotation matrix
    R[..., 2] = z

    # Handle batch dimension properly
    x, y, z = R[..., 0], R[..., 1], R[..., 2]

    # Normalize z vectors
    z_norm = torch.linalg.norm(z, dim=-1, keepdim=True)
    z /= z_norm.clamp(min=gs.EPS)

    # Handle zero norm cases
    zero_mask = z_norm[..., 0] < gs.EPS
    if zero_mask.any():
        if up is None:
            z[zero_mask] = torch.tensor((0.0, 1.0, 0.0), device=z.device, dtype=z.dtype)
        else:
            up_mask = up[..., 1].abs() < 0.5
            z[zero_mask & up_mask] = torch.tensor((0.0, 1.0, 0.0), device=z.device, dtype=z.dtype)
            z[zero_mask & ~up_mask] = torch.tensor((0.0, 0.0, 1.0), device=z.device, dtype=z.dtype)

    # Compute x vectors (first column)
    if up is not None:
        x[:] = torch.cross(up, z, dim=-1)
    else:
        up_mask = z[..., 2].abs() < 1.0 - gs.EPS
        _zero = torch.tensor(0.0, device=z.device, dtype=z.dtype)
        torch.where(up_mask, z[..., 1], z[..., 2], out=x[..., 0])
        torch.where(up_mask, -z[..., 0], _zero, out=x[..., 1])
        torch.where(up_mask, _zero, -z[..., 0], out=x[..., 2])

    # Normalize x vectors
    x_norm = torch.norm(x, dim=-1, keepdim=True)
    x /= x_norm.clamp(min=gs.EPS)

    # Handle zero x norm cases
    zero_x_mask = x_norm[..., 0] < gs.EPS
    zero_x_num = zero_x_mask.sum()
    if zero_x_num:
        # For zero x norm, set identity matrix
        R[zero_x_mask] = torch.eye(3, device=z.device, dtype=z.dtype).unsqueeze(0).expand((zero_x_num, 3, 3))

        # Continue with non-zero cases
        valid_mask = ~zero_x_mask
        if zero_x_num < zero_x_mask.numel():
            z_valid = z[valid_mask]
            x_valid = x[valid_mask]
            y[valid_mask] = torch.cross(z_valid, x_valid, dim=-1)
    else:
        # All x norms are valid, compute y vectors
        y[:] = torch.cross(z, x, dim=-1)

    return R


def z_up_to_R(z, up=None, out=None):
    if isinstance(z, torch.Tensor):
        return _tc_z_up_to_R(z, up, out)
    else:
        return _np_z_up_to_R(z, up, out)


def pos_lookat_up_to_T(pos, lookat, up):
    if all(isinstance(e, torch.Tensor) for e in (pos, lookat, up) if e is not None):
        T = torch.zeros((*pos.shape[:-1], 4, 4), dtype=pos.dtype, device=pos.device)
        T[..., 3, 3] = 1.0
        T[..., :3, 3] = pos

        z = pos - lookat
        z_up_to_R(z, up=up, out=T[..., :3, :3])
        return T
    elif all(isinstance(e, np.ndarray) for e in (pos, lookat, up) if e is not None):
        T = np.zeros((*pos.shape[:-1], 4, 4), dtype=pos.dtype)
        T[..., 3, 3] = 1.0
        T[..., :3, 3] = pos

        z = pos - lookat
        z_up_to_R(z, up=up, out=T[..., :3, :3])
        return T
    else:
        gs.raise_exception(
            f"all of the inputs must be torch.Tensor or np.ndarray. got: {type(pos)=}, {type(lookat)=}, {type(up)=}"
        )


def T_to_pos_lookat_up(T):
    pos = T[..., :3, 3]
    lookat = T[..., :3, 3] - T[..., :3, 2]
    up = T[..., :3, 1]
    return pos, lookat, up


def euler_to_quat(euler_xyz):
    return xyz_to_quat(np.asarray(euler_xyz), rpy=True, degrees=True)


@nb.jit(nopython=True, cache=True)
def _np_euler_to_R(rpy: np.ndarray, out: np.ndarray | None = None) -> np.ndarray:
    """Compute the Rotation Matrix representation of a single or a batch of Yaw-Pitch-Roll Euler angles.

    :param rpy: N-dimensional array whose last dimension gathers the 3 Yaw-Pitch-Roll Euler angles [Roll, Pitch, Yaw].
    :param out: Pre-allocated array in which to store the result. If not provided, a new array is freshly-allocated and
                returned, which is slower.
    """
    assert rpy.ndim >= 1
    if out is None:
        out_ = np.empty((*rpy.shape[1:], 3, 3), dtype=rpy.dtype)
    else:
        assert out.shape == (*rpy.shape[1:], 3, 3)
        out_ = out

    cos_rpy, sin_rpy = np.cos(rpy), np.sin(rpy)
    cos_roll, cos_pitch, cos_yaw = cos_rpy[..., 0], cos_rpy[..., 1], cos_rpy[..., 2]
    sin_roll, sin_pitch, sin_yaw = sin_rpy[..., 0], sin_rpy[..., 1], sin_rpy[..., 2]

    out_[..., 0, 0] = cos_pitch * cos_yaw
    out_[..., 0, 1] = -cos_roll * sin_yaw + sin_roll * sin_pitch * cos_yaw
    out_[..., 0, 2] = sin_roll * sin_yaw + cos_roll * sin_pitch * cos_yaw
    out_[..., 1, 0] = cos_pitch * sin_yaw
    out_[..., 1, 1] = cos_roll * cos_yaw + sin_roll * sin_pitch * sin_yaw
    out_[..., 1, 2] = -sin_roll * cos_yaw + cos_roll * sin_pitch * sin_yaw
    out_[..., 2, 0] = -sin_pitch
    out_[..., 2, 1] = sin_roll * cos_pitch
    out_[..., 2, 2] = cos_roll * cos_pitch

    return out_


def euler_to_R(euler_xyz):
    return _np_euler_to_R(np.deg2rad(euler_xyz))


@nb.jit(nopython=True, cache=True)
def quat_to_rotvec(quat: np.ndarray, out: np.ndarray | None = None) -> np.ndarray:
    """Compute the angle-axis representation of a single or a batch of quaternions (qw, qx, qy, qz).

    :param quat: N-dimensional array whose last dimension gathers the 4 quaternion coordinates (qw, qx, qy, qz).
    :param out: Pre-allocated array into which to store the result. If not provided, a new array is freshly-allocated
                and returned, which is slower.
    """
    assert quat.ndim >= 1
    if out is None:
        out_ = np.empty((*quat.shape[:-1], 3), dtype=quat.dtype)
    else:
        assert out.shape == (*quat.shape[:-1], 3)
        out_ = out

    # Split real (qw,) and imaginary (qx, qy, qz) quaternion parts
    q_w, q_vec = quat[..., 0], quat[..., 1:]

    # Compute the angle-axis representation of the relative rotation
    s2 = np.sqrt(np.sum(np.square(q_vec), -1))
    angle = 2.0 * np.arctan2(s2, np.abs(q_w))
    # FIXME: Ideally, a taylor expansion should be used to handle angle ~ 0.0
    inv_sinc = angle / np.maximum(s2, gs.EPS)
    out_[:] = (-1.0 if q_w < 0.0 else 1.0) * inv_sinc * q_vec

    if out is None:
        return out_
    return None


@nb.jit(nopython=True, cache=True)
def rotvec_to_quat(rotvec: np.ndarray, out: np.ndarray | None = None) -> np.ndarray:
    """Compute the quaternion representation (qw, qx, qy, qz) of a single or a batch of angle-axis vectors.

    :param rotvec: N-dimensional array whose last dimension gathers the 3 angle-axis components
                   angle * (ax, ay, az).
    :param out: Pre-allocated array into which to store the result. If not provided, a new array is freshly-allocated
                and returned, which is slower.
    """
    assert rotvec.ndim >= 1
    if out is None:
        out_ = np.empty((*rotvec.shape[:-1], 4), dtype=rotvec.dtype)
    else:
        assert out.shape == (*rotvec.shape[:-1], 4)
        out_ = out

    # Compute unit axis and positive angle separately
    angle = np.sqrt(np.sum(np.square(rotvec), -1))
    # FIXME: Taylor expansion should be used to handle angle ~ 0.0
    axis = rotvec / np.maximum(angle, gs.EPS)

    # Compute the quaternion representation
    out_[..., 0] = np.cos(0.5 * angle)
    out_[..., 1:] = np.sin(0.5 * angle) * axis

    return out_


@nb.jit(nopython=True, cache=True)
def _np_axis_cos_angle_to_R(axis: np.ndarray, cos_theta: np.ndarray, out: np.ndarray | None = None) -> np.ndarray:
    if isinstance(cos_theta, (float, np.float32, np.float64)):
        assert axis.ndim == 1
    else:
        assert axis.ndim - 1 == cos_theta.ndim
    if out is None:
        out_ = np.empty((*axis.shape[:-1], 3, 3), dtype=axis.dtype)
    else:
        assert out.shape == (*axis.shape[:-1], 3, 3)
        out_ = out

    axis_norm = np.sqrt(np.sum(np.square(axis.reshape((-1, 3))), -1).reshape((*axis.shape[:-1], 1)))
    axis = axis / axis_norm
    if not isinstance(cos_theta, (float, np.float32, np.float64)):
        cos_theta = cos_theta[..., None]
    sin_theta = np.sqrt(1.0 - cos_theta**2)
    cos1_axis = (1.0 - cos_theta) * axis
    sin_axis = sin_theta * axis

    tmp = cos1_axis[..., 0] * axis[..., 1]
    out_[..., 0, 1] = tmp - sin_axis[..., 2]
    out_[..., 1, 0] = tmp + sin_axis[..., 2]
    tmp = cos1_axis[..., 0] * axis[..., 2]
    out_[..., 0, 2] = tmp + sin_axis[..., 1]
    out_[..., 2, 0] = tmp - sin_axis[..., 1]
    tmp = cos1_axis[..., 1] * axis[..., 2]
    out_[..., 1, 2] = tmp - sin_axis[..., 0]
    out_[..., 2, 1] = tmp + sin_axis[..., 0]
    tmp = cos1_axis * axis + cos_theta
    out_[..., 0, 0] = tmp[..., 0]
    out_[..., 1, 1] = tmp[..., 1]
    out_[..., 2, 2] = tmp[..., 2]

    return out_


def axis_angle_to_R(axis: np.ndarray, theta: np.ndarray) -> np.ndarray:
    return _np_axis_cos_angle_to_R(axis, np.cos(theta))


def rotvec_to_R(rotvec: np.ndarray) -> np.ndarray:
    return axis_angle_to_R(rotvec, np.linalg.norm(rotvec, axis=-1))


@nb.jit(nopython=True, cache=True)
def z_to_R(v_a: np.ndarray, out: np.ndarray | None = None) -> np.ndarray:
    """
    Compute the "smallest" rotation transforming vector 'v_a' in 'e_z' as a rotation matrix.

    This operation is computed by rotating the world frame by moving the original z-axis to the given vector via the
    shortest path.
    """
    B = v_a.shape[:-1]

    v_a_norm = np.sqrt(np.sum(np.square(v_a.reshape((-1, 3))), -1).reshape((*B, 1)))
    v_a = v_a / v_a_norm

    axis = np.empty((*B, 3), dtype=v_a.dtype)
    cos_theta = np.empty(B, dtype=v_a.dtype)
    for i in np.ndindex(B):
        axis_i = axis[i]
        v_x, v_y, v_z = v_a[i]
        cos_theta[i] = v_z
        if abs(cos_theta[i]) < 1.0 - gs.EPS:
            axis_i[0] = -v_y
            axis_i[1] = v_x
            axis_i[2] = 0.0
        else:
            axis_i[:] = 0.0, 1.0, 0.0

    return _np_axis_cos_angle_to_R(axis, cos_theta, out)


@nb.jit(nopython=True, cache=True)
def z_to_quat(v_a: np.ndarray, out: np.ndarray | None = None) -> np.ndarray:
    """
    Compute the "smallest" rotation transforming vector 'v_a' in 'e_z' as a quaternion (q_w, q_x, q_y, q_z).

    This method is (surprisingly) slower than `z_up_to_R`, w/ and w/o chaining `transform_by_(quat|R)`.
    """
    if out is None:
        out_ = np.empty((*v_a.shape[:-1], 4), dtype=v_a.dtype)
    else:
        assert out.shape == (*v_a.shape[:-1], 4)
        out_ = out

    v_a_norm = np.sqrt(np.sum(np.square(v_a.reshape((-1, 3))), -1).reshape((*v_a.shape[:-1], 1)))
    v_a = v_a / v_a_norm
    v_x, v_y, v_z = v_a[..., 0], v_a[..., 1], v_a[..., 2]

    for i in np.ndindex(v_a.shape[:-1]):
        v_x, v_y, v_z = v_a[i]
        quat = out_[i]

        if v_a_norm[i] < gs.EPS:
            quat[:] = 1.0, 0.0, 0.0, 0.0
        elif v_z > -1.0 + gs.EPS:
            s = np.sqrt(2.0 * (1.0 + v_z))
            s_inv = 1.0 / s
            quat[:] = 0.5 * s, -v_y * s_inv, v_x * s_inv, 0.0
        else:
            eps_thr = np.sqrt(gs.EPS)
            eps_x = abs(v_x) < gs.EPS
            eps_y = abs(v_y) < gs.EPS
            if not eps_y:
                ratio = v_x / v_y
                if eps_x:
                    esp_ratio = abs(ratio) < eps_thr
            elif eps_y and not eps_x:
                ratio = v_y / v_x
                esp_ratio = abs(ratio) < eps_thr
            w_2 = 0.5 * (1.0 + max(v_z, -1.0))
            quat[0] = np.sqrt(w_2)
            if eps_x and eps_y:
                # Both q_x and q_y would do fine. Picking q_y arbitrarily.
                quat[1] = 0.0
                quat[2] = 1.0
            elif esp_ratio:
                coef_abs = np.sqrt(1.0 - w_2) * (1.0 - 0.5 * ratio**2)
                if eps_x:
                    quat[1] = -np.sign(v_y) * coef_abs
                    quat[2] = -quat[1] * ratio
                else:
                    quat[2] = np.sign(v_x) * coef_abs
                    quat[1] = -quat[2] * ratio
            else:
                q_x_abs = np.sqrt((1.0 - w_2) / (1.0 + ratio**2))
                quat[1] = -np.sign(v_y) * q_x_abs
                quat[2] = +np.sign(v_x) * q_x_abs * ratio
            quat[3] = 0.0

            # First order quaternion normalization is accurate enough
            quat *= 0.5 * (3.0 - np.sum(np.square(quat), -1))

    return out_


def compute_camera_angle(camera_pos, camera_lookat):
    camera_dir = np.asarray(camera_lookat) - np.asarray(camera_pos)

    # rotation around vertical (y) axis
    angle_x = np.arctan2(-camera_dir[0], -camera_dir[2])

    # rotation w.r.t horizontal plane
    angle_y = np.arctan2(camera_dir[1], np.linalg.norm([camera_dir[0], camera_dir[2]]))

    angle_z = 0.0

    return np.array([angle_x, angle_y, angle_z])


def transform_inertia_by_T(inertia_tensor, T, mass):
    """
    Transform the inertia tensor to the new reference frame.
    """
    R = T[:3, :3]
    t = T[:3, 3]

    # Parallel axis theorem
    translation_inertia = mass * (np.dot(t, t) * np.eye(3) - np.outer(t, t))

    return R @ inertia_tensor @ R.T + translation_inertia


def slerp(q0, q1, t):
    """
    Perform spherical linear interpolation between two quaternions.

    Parameters:
    q0 : numpy.array
        The start quaternion (4-dimensional vector).
    q1 : numpy.array
        The end quaternion (4-dimensional vector).
    t : float
        The interpolation parameter between 0 and 1.

    Returns:
    numpy.array
        The interpolated quaternion (4-dimensional vector).
    """
    q0 = q0 / np.linalg.norm(q0)
    q1 = q1 / np.linalg.norm(q1)

    dot_product = np.dot(q0, q1)

    if dot_product < 0.0:
        q1 = -q1
        dot_product = -dot_product

    dot_product = np.clip(dot_product, -1.0, 1.0)

    theta_0 = np.arccos(dot_product)
    sin_theta_0 = np.sin(theta_0)

    if sin_theta_0 < 1e-6:
        return (1.0 - t) * q0 + t * q1

    theta = theta_0 * t
    sin_theta = np.sin(theta)

    s0 = np.cos(theta) - dot_product * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0

    return s0 * q0 + s1 * q1


def random_quaternion(batch_size):
    # Generate three uniform random numbers for each quaternion in the batch
    u1, u2, u3 = np.random.rand(3, batch_size)
    q1 = np.sqrt(1 - u1) * np.sin(2 * np.pi * u2)
    q2 = np.sqrt(1 - u1) * np.cos(2 * np.pi * u2)
    q3 = np.sqrt(u1) * np.sin(2 * np.pi * u3)
    q4 = np.sqrt(u1) * np.cos(2 * np.pi * u3)
    return np.stack((q1, q2, q3, q4), axis=1)


# ------------------------------------------------------------------------------------
# ------------------------------------- misc ----------------------------------------
# ------------------------------------------------------------------------------------


def zero_pos():
    return np.zeros(3, dtype=gs.np_float)


def identity_quat():
    return np.array([1.0, 0.0, 0.0, 0.0], dtype=gs.np_float)


def tc_zero_pos():
    return torch.zeros(3, dtype=gs.tc_float, device=gs.device)


def tc_identity_quat():
    return torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=gs.tc_float, device=gs.device)


def nowhere():
    # let's inject a bit of humor here
    return np.array([2333333, 6666666, 5201314], dtype=gs.np_float)


def default_solver_params():
    """
    Default solver parameters (timeconst, dampratio, dmin, dmax, width, mid, power).

    Reference: https://mujoco.readthedocs.io/en/latest/modeling.html#solver-parameters
    """
    return np.array([0.0, 1.0e00, 9.0e-01, 9.5e-01, 1.0e-03, 5.0e-01, 2.0e00])


def default_friction():
    return 1.0


def default_dofs_kp(n=6):
    return np.full((n,), fill_value=100.0, dtype=gs.np_float)


def default_dofs_kv(n=6):
    return np.full((n,), fill_value=10.0, dtype=gs.np_float)


@ti.data_oriented
class SpatialHasher:
    def __init__(self, cell_size, grid_res, n_slots=None):
        self.cell_size = cell_size
        self.grid_res = grid_res

        if n_slots is None:
            self.n_slots = np.prod(grid_res)
        else:
            self.n_slots = n_slots

    def build(self, n_batch):
        self._B = n_batch
        # number of elements in each slot
        self.slot_size = ti.field(gs.ti_int, shape=(self.n_slots, self._B))
        # element index offset in each slot
        self.slot_start = ti.field(gs.ti_int, shape=(self.n_slots, self._B))
        self.cur_cnt = ti.field(gs.ti_int, shape=self._B)

    @ti.func
    def compute_reordered_idx(self, n, pos, active, reordered_idx):
        """
        Reordered element idx based on the given positions and active flags.

        Parameters:
            n (int)       : The number of elements in the positions and active arrays.
            pos           : The array of positions.
            active        : The array of active flags.
            reordered_idx : The array to store the computed reordered indices.

        Returns:
            None
        """

        self.slot_size.fill(0)
        self.slot_start.fill(0)
        self.cur_cnt.fill(0)

        for i_n, i_b in ti.ndrange(n, self._B):
            if active[i_n, i_b]:
                slot_idx = self.pos_to_slot(pos[i_n, i_b])
                ti.atomic_add(self.slot_size[slot_idx, i_b], 1)

        for i_n in range(self.n_slots):
            for i_b in range(self._B):
                self.slot_start[i_n, i_b] = ti.atomic_add(self.cur_cnt[i_b], self.slot_size[i_n, i_b])

        for i_n, i_b in ti.ndrange(n, self._B):
            if active[i_n, i_b]:
                slot_idx = self.pos_to_slot(pos[i_n, i_b])
                reordered_idx[i_n, i_b] = ti.atomic_add(self.slot_start[slot_idx, i_b], 1)

        # recover slot_start
        for i_s, i_b in ti.ndrange(self.n_slots, self._B):
            self.slot_start[i_s, i_b] -= self.slot_size[i_s, i_b]

    @ti.func
    def for_all_neighbors(self, i, pos, task_range, ret: ti.template(), task: ti.template(), i_b):
        """
        Iterates over all neighbors of a given position and performs a task on each neighbor.
        Elements are considered neighbors if they are within task_range.

        Parameters:
            i (int)    : Index of the querying particle.
            pos        : Template for the positions of all particles.
            task       : Template for the task to be performed on each neighbor of the querying particle.
            task_range : Range within which the task should be performed.
            ret        : Template for the return value of the task.

        Returns:
            None
        """
        base = self.pos_to_grid(pos[i, i_b])
        for offset in ti.grouped(ti.ndrange((-1, 2), (-1, 2), (-1, 2))):
            slot_idx = self.grid_to_slot(base + offset)
            for j in range(
                self.slot_start[slot_idx, i_b], self.slot_size[slot_idx, i_b] + self.slot_start[slot_idx, i_b]
            ):
                if i != j and (pos[i, i_b] - pos[j, i_b]).norm() < task_range:
                    task(i, j, ret, i_b)

    @ti.func
    def pos_to_grid(self, pos):
        return ti.floor(pos / self.cell_size, gs.ti_int)

    @ti.func
    def grid_to_pos(self, grid_id):
        return (grid_id + 0.5) * self.cell_size

    @ti.func
    def grid_to_slot(self, grid_id):
        return (
            grid_id[0] * self.grid_res[1] * self.grid_res[2] + grid_id[1] * self.grid_res[2] + grid_id[2]
        ) % self.n_slots

    @ti.func
    def pos_to_slot(self, pos):
        return self.grid_to_slot(self.pos_to_grid(pos))
