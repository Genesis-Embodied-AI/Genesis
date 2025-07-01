import numpy as np
import taichi as ti
import torch
import torch.nn.functional as F
from scipy.spatial.transform import Rotation
import genesis as gs

# ------------------------------------------------------------------------------------
# ------------------------------------- taichi ----------------------------------------
# ------------------------------------------------------------------------------------


@ti.func
def ti_quat_mul(u, v):
    terms = v.outer_product(u)
    w = terms[0, 0] - terms[1, 1] - terms[2, 2] - terms[3, 3]
    x = terms[0, 1] + terms[1, 0] - terms[2, 3] + terms[3, 2]
    y = terms[0, 2] + terms[1, 3] + terms[2, 0] - terms[3, 1]
    z = terms[0, 3] - terms[1, 2] + terms[2, 1] + terms[3, 0]
    return ti.Vector([w, x, y, z])


@ti.func
def ti_transform_quat_by_quat(v, u):
    # This method transforms quat_v by quat_u
    # This is equivalent to quatmul(quat_u, quat_v) or R_u @ R_v
    vec = ti_quat_mul(u, v)
    return vec.normalized()


@ti.func
def ti_rotvec_to_quat(rotvec):
    theta = rotvec.norm()
    v = ti.Vector.zero(gs.ti_float, 3)
    if theta > gs.EPS:
        axis = rotvec / theta
        v = axis * ti.sin(theta / 2)
    return ti.Vector([ti.cos(theta / 2), v[0], v[1], v[2]]).normalized()


@ti.func
def ti_rotvec_to_R(rotvec):
    return ti_quat_to_R(ti_rotvec_to_quat(rotvec))


@ti.func
def ti_quat_to_rotvec(quat):
    v = ti.Vector([quat[1], quat[2], quat[3]])
    axis = v / v.norm(gs.EPS)
    theta = 2 * ti.atan2(v.norm(), quat[0])

    # when axis-angle is larger than pi, rotation is in the opposite direction
    if theta > ti.math.pi:
        theta -= 2 * ti.math.pi

    return axis * theta


@ti.func
def ti_inv_quat(quat):
    return ti.Vector([quat[0], -quat[1], -quat[2], -quat[3]])


@ti.func
def ti_normalize(v):
    return v / (v.norm(gs.EPS))


@ti.func
def ti_transform_by_quat(v, quat):
    qvec = ti.Vector([quat[1], quat[2], quat[3]])
    uv = qvec.cross(v)
    uuv = qvec.cross(uv)
    return v + 2 * (quat[0] * uv + uuv)


@ti.func
def ti_inv_transform_by_quat(v, quat):
    return ti_transform_by_quat(v, ti_inv_quat(quat))


@ti.func
def ti_transform_by_T(pos, T):
    new_pos = ti.Vector([pos[0], pos[1], pos[2], 1.0], dt=gs.ti_float)
    new_pos = T @ new_pos
    return new_pos[:3]


@ti.func
def ti_inv_transform_by_T(pos, T):
    T_inv = T.inverse()
    return ti_transform_by_T(pos, T_inv)


@ti.func
def ti_transform_by_trans_quat(pos, trans, quat):
    return ti_transform_by_quat(pos, quat) + trans


@ti.func
def ti_inv_transform_by_trans_quat(pos, trans, quat):
    return ti_transform_by_quat(pos - trans, ti_inv_quat(quat))


@ti.func
def ti_quat_to_xyz(quat):
    """
    Convert a quaternion into roll, pitch, and yaw angles.
    """
    return ti_R_to_xyz(ti_quat_to_R(quat))


@ti.func
def ti_R_to_xyz(R):
    """
    Convert a rotation matrix into intrinsic x-y-z Euler angles.
    Reference: https://github.com/openai/mujoco-worldgen/blob/master/mujoco_worldgen/util/rotation.py
    """
    cy = ti.sqrt(R[2, 2] * R[2, 2] + R[1, 2] * R[1, 2])
    xyz = ti.Vector.zero(gs.ti_float, 3)
    if cy > 1e-6:
        xyz[2] = -ti.atan2(R[0, 1], R[0, 0])
        xyz[1] = -ti.atan2(-R[0, 2], cy)
        xyz[0] = -ti.atan2(R[1, 2], R[2, 2])
    else:
        xyz[2] = -ti.atan2(-R[1, 0], R[1, 1])
        xyz[1] = -ti.atan2(-R[0, 2], cy)
        xyz[0] = 0.0
    return xyz


@ti.func
def ti_xyz_to_quat(xyz):
    """
    Convert intrinsic x-y-z Euler angles to quaternion.
    Reference: https://github.com/openai/mujoco-worldgen/blob/master/mujoco_worldgen/util/rotation.py
    """
    ai, aj, ak = xyz[2] / 2, -xyz[1] / 2, xyz[0] / 2
    si, sj, sk = ti.sin(ai), ti.sin(aj), ti.sin(ak)
    ci, cj, ck = ti.cos(ai), ti.cos(aj), ti.cos(ak)
    cc, cs = ci * ck, ci * sk
    sc, ss = si * ck, si * sk

    quat = ti.Vector([cj * cc + sj * ss, cj * cs - sj * sc, -(cj * ss + sj * cc), cj * sc - sj * cs], dt=gs.ti_float)
    return quat


@ti.func
def ti_quat_to_R(q):
    """Converts quaternion to 3x3 rotation matrix."""
    d = q.dot(q)
    w, x, y, z = q
    s = 2 / d
    xs, ys, zs = x * s, y * s, z * s
    wx, wy, wz = w * xs, w * ys, w * zs
    xx, xy, xz = x * xs, x * ys, x * zs
    yy, yz, zz = y * ys, y * zs, z * zs

    return ti.Matrix(
        [[1 - (yy + zz), xy - wz, xz + wy], [xy + wz, 1 - (xx + zz), yz - wx], [xz - wy, yz + wx, 1 - (xx + yy)]]
    )


@ti.func
def ti_trans_quat_to_T(trans, quat):
    w, x, y, z = quat
    T = ti.Matrix(
        [
            [1 - 2 * y**2 - 2 * z**2, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y, trans[0]],
            [2 * x * y + 2 * w * z, 1 - 2 * x**2 - 2 * z**2, 2 * y * z - 2 * w * x, trans[1]],
            [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x**2 - 2 * y**2, trans[2]],
            [0, 0, 0, 1],
        ],
        dt=gs.ti_float,
    )
    return T


@ti.func
def ti_trans_to_T(trans):
    T = ti.Matrix([[1, 0, 0, trans[0]], [0, 1, 0, trans[1]], [0, 0, 1, trans[2]], [0, 0, 0, 1]], dt=gs.ti_float)
    return T


@ti.func
def ti_quat_to_T(quat):
    w, x, y, z = quat
    T = ti.Matrix(
        [
            [1 - 2 * y**2 - 2 * z**2, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y, 0.0],
            [2 * x * y + 2 * w * z, 1 - 2 * x**2 - 2 * z**2, 2 * y * z - 2 * w * x, 0.0],
            [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x**2 - 2 * y**2, 0.0],
            [0, 0, 0, 1],
        ],
        dt=gs.ti_float,
    )
    return T


@ti.func
def ti_transform_motion_by_trans_quat(m_ang, m_vel, trans, quat):
    quat_t = ti_inv_quat(quat)
    ang = ti_transform_by_quat(m_ang, quat_t)
    vel = ti_transform_by_quat(m_vel - trans.cross(m_ang), quat_t)
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
    h = ti.Matrix.rows(
        [
            trans.cross(ti.Vector([-1.0, 0.0, 0.0])),
            trans.cross(ti.Vector([0.0, -1.0, 0.0])),
            trans.cross(ti.Vector([0.0, 0.0, -1.0])),
        ]
    )

    R = ti_quat_to_R(quat)
    i = R @ i_inertial @ R.transpose() + h @ h.transpose() * i_mass
    trans = trans * i_mass
    quat = quat
    return i, trans, quat, i_mass


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
def orthogonals(a):
    """Returns orthogonal vectors `b` and `c`, given a normal vector `a`."""
    y, z = ti.Vector([0.0, 1.0, 0.0], dt=gs.ti_float), ti.Vector([0.0, 0.0, 1.0], dt=gs.ti_float)
    b = z
    if -0.5 < a[1] and a[1] < 0.5:
        b = y
    b = b - a * a.dot(b)
    b = b.normalized()
    return b, a.cross(b)


@ti.func
def imp_aref(params, neg_penetration, vel, pos):
    timeconst, dampratio, dmin, dmax, width, mid, power = params

    imp_x = ti.abs(neg_penetration) / width
    imp_a = (1.0 / mid ** (power - 1)) * imp_x**power
    imp_b = 1 - (1.0 / (1 - mid) ** (power - 1)) * (1 - imp_x) ** power
    imp_y = imp_a if imp_x < mid else imp_b

    imp = dmin + imp_y * (dmax - dmin)
    imp = ti.math.clamp(imp, dmin, dmax)
    imp = dmax if imp_x > 1.0 else imp

    b = 2 / (dmax * timeconst)
    k = 1 / (dmax * dmax * timeconst * timeconst * dampratio * dampratio)

    aref = -b * vel - k * imp * pos

    return imp, aref


@ti.func
def closest_segment_point(a, b, pt):
    ab = b - a
    t = (pt - a).dot(ab) / (ab.dot(ab) + 1e-6)
    return a + ti.math.clamp(t, 0.0, 1.0) * ab


@ti.func
def get_face_norm(v0, v1, v2):
    edge0 = v1 - v0
    edge1 = v2 - v0
    face_norm = edge0.cross(edge1)
    face_norm = face_norm.normalized()
    return face_norm


@ti.func
def ti_quat_mul_axis(q, axis):
    return ti.Vector(
        [
            -q[1] * axis[0] - q[2] * axis[1] - q[3] * axis[2],
            q[0] * axis[0] + q[2] * axis[2] - q[3] * axis[1],
            q[0] * axis[1] + q[3] * axis[0] - q[1] * axis[2],
            q[0] * axis[2] + q[1] * axis[1] - q[2] * axis[0],
        ]
    )


# ------------------------------------------------------------------------------------
# -------------------------------- torch and numpy -----------------------------------
# ------------------------------------------------------------------------------------
def xyzw_to_wxyz(xyzw):
    if xyzw.ndim == 1:
        return xyzw[[3, 0, 1, 2]]
    elif xyzw.ndim == 2:
        return xyzw[:, [3, 0, 1, 2]]
    else:
        gs.raise_exception(f"ndim expected to be 1 or 2, but got {xyzw.ndim=}")


def wxyz_to_xyzw(wxyz):
    if wxyz.ndim == 1:
        return wxyz[[1, 2, 3, 0]]
    elif wxyz.ndim == 2:
        return wxyz[:, [1, 2, 3, 0]]
    else:
        gs.raise_exception(f"ndim expected to be 1 or 2, but got {wxyz.ndim=}")


def transform_quat_by_quat(v, u):
    if isinstance(v, torch.Tensor) and isinstance(u, torch.Tensor):
        assert v.shape == u.shape, f"{v.shape} != {u.shape}"
        w1, x1, y1, z1 = u[..., 0], u[..., 1], u[..., 2], u[..., 3]
        w2, x2, y2, z2 = v[..., 0], v[..., 1], v[..., 2], v[..., 3]
        ww = (z1 + x1) * (x2 + y2)
        yy = (w1 - y1) * (w2 + z2)
        zz = (w1 + y1) * (w2 - z2)
        xx = ww + yy + zz
        qq = 0.5 * (xx + (z1 - x1) * (x2 - y2))
        w = qq - ww + (z1 - y1) * (y2 - z2)
        x = qq - xx + (x1 + w1) * (x2 + w2)
        y = qq - yy + (w1 - x1) * (y2 + z2)
        z = qq - zz + (z1 + y1) * (w2 - x2)
        quat = torch.stack([w, x, y, z], dim=-1)
        return quat
    elif isinstance(v, np.ndarray) and isinstance(u, np.ndarray):
        assert v.shape == u.shape, f"{v.shape} != {u.shape}"
        w1, x1, y1, z1 = u[..., 0], u[..., 1], u[..., 2], u[..., 3]
        w2, x2, y2, z2 = v[..., 0], v[..., 1], v[..., 2], v[..., 3]
        # This method transforms quat_v by quat_u
        # This is equivalent to quatmul(quat_u, quat_v) or R_u @ R_v
        ww = (z1 + x1) * (x2 + y2)
        yy = (w1 - y1) * (w2 + z2)
        zz = (w1 + y1) * (w2 - z2)
        xx = ww + yy + zz
        qq = 0.5 * (xx + (z1 - x1) * (x2 - y2))
        w = qq - ww + (z1 - y1) * (y2 - z2)
        x = qq - xx + (x1 + w1) * (x2 + w2)
        y = qq - yy + (w1 - x1) * (y2 + z2)
        z = qq - zz + (z1 + y1) * (w2 - x2)
        quat = np.stack([w, x, y, z], axis=-1)
        return quat
    else:
        gs.raise_exception(f"both of the inputs must be torch.Tensor or np.ndarray. got: {type(v)=} and {type(u)=}")


def inv_quat(quat):
    if isinstance(quat, torch.Tensor):
        scaling = torch.tensor([1, -1, -1, -1], device=quat.device)
        return quat * scaling
    elif isinstance(quat, np.ndarray):
        scaling = np.array([1, -1, -1, -1], dtype=quat.dtype)
        return quat * scaling
    else:
        gs.raise_exception(f"the input must be either torch.Tensor or np.ndarray. got: {type(quat)=}")


def normalize(x, eps: float = 1e-9):
    if isinstance(x, torch.Tensor):
        return x / x.norm(p=2, dim=-1).clamp(min=eps, max=None).unsqueeze(-1)
    elif isinstance(x, np.ndarray):
        return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), eps)
    else:
        gs.raise_exception(f"the input must be either torch.Tensor or np.ndarray. got: {type(x)=}")


def rot6d_to_quat(d6):
    R = rot6d_to_R(d6)
    return R_to_quat(R)


def quat_to_rot6d(quat):
    R = quat_to_R(quat)
    return R_to_rot6d(R)


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
        return R[..., :2, :].clone().flatten(start_dim=-2)
    elif isinstance(R, np.ndarray):
        return R[..., :2, :].reshape(*R.shape[:-2], 6)
    else:
        gs.raise_exception(f"the input must be either torch.Tensor or np.ndarray. got: {type(R)=}")


def quat_to_R(quat):
    if isinstance(quat, torch.Tensor):
        qw, qx, qy, qz = torch.unbind(quat, -1)
        # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
        two_s = 2.0 / (quat * quat).sum(-1)
        return torch.stack(
            (
                1 - two_s * (qy * qy + qz * qz),
                two_s * (qx * qy - qz * qw),
                two_s * (qx * qz + qy * qw),
                two_s * (qx * qy + qz * qw),
                1 - two_s * (qx * qx + qz * qz),
                two_s * (qy * qz - qx * qw),
                two_s * (qx * qz - qy * qw),
                two_s * (qy * qz + qx * qw),
                1 - two_s * (qx * qx + qy * qy),
            ),
            -1,
        ).reshape(quat.shape[:-1] + (3, 3))
    elif isinstance(quat, np.ndarray):
        return Rotation.from_quat(quat, scalar_first=True).as_matrix().astype(quat.dtype)
    else:
        gs.raise_exception(f"the input must be either torch.Tensor or np.ndarray. got: {type(quat)=}")


def R_to_quat(R):
    if isinstance(R, torch.Tensor):
        batch = R.shape[:-2]  # Support batch dimension
        quat_xyzw = torch.zeros((*batch, 4), dtype=R.dtype, device=R.device)

        trace = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]

        # Compute quaternion based on the trace of the matrix
        mask1 = trace > 0
        mask2 = ~mask1 & (R[..., 0, 0] >= R[..., 1, 1]) & (R[..., 0, 0] >= R[..., 2, 2])
        mask3 = ~mask1 & ~mask2 & (R[..., 1, 1] >= R[..., 2, 2])
        mask4 = ~mask1 & ~mask2 & ~mask3

        S = torch.zeros_like(trace)

        S[mask1] = torch.sqrt(trace[mask1] + 1.0) * 2
        quat_xyzw[mask1, 0] = (R[mask1, 2, 1] - R[mask1, 1, 2]) / S[mask1]
        quat_xyzw[mask1, 1] = (R[mask1, 0, 2] - R[mask1, 2, 0]) / S[mask1]
        quat_xyzw[mask1, 2] = (R[mask1, 1, 0] - R[mask1, 0, 1]) / S[mask1]
        quat_xyzw[mask1, 3] = 0.25 * S[mask1]

        S[mask2] = torch.sqrt(1.0 + R[mask2, 0, 0] - R[mask2, 1, 1] - R[mask2, 2, 2]) * 2
        quat_xyzw[mask2, 0] = 0.25 * S[mask2]
        quat_xyzw[mask2, 1] = (R[mask2, 0, 1] + R[mask2, 1, 0]) / S[mask2]
        quat_xyzw[mask2, 2] = (R[mask2, 0, 2] + R[mask2, 2, 0]) / S[mask2]
        quat_xyzw[mask2, 3] = (R[mask2, 2, 1] - R[mask2, 1, 2]) / S[mask2]

        S[mask3] = torch.sqrt(1.0 + R[mask3, 1, 1] - R[mask3, 0, 0] - R[mask3, 2, 2]) * 2
        quat_xyzw[mask3, 0] = (R[mask3, 0, 1] + R[mask3, 1, 0]) / S[mask3]
        quat_xyzw[mask3, 1] = 0.25 * S[mask3]
        quat_xyzw[mask3, 2] = (R[mask3, 1, 2] + R[mask3, 2, 1]) / S[mask3]
        quat_xyzw[mask3, 3] = (R[mask3, 0, 2] - R[mask3, 2, 0]) / S[mask3]

        S[mask4] = torch.sqrt(1.0 + R[mask4, 2, 2] - R[mask4, 0, 0] - R[mask4, 1, 1]) * 2
        quat_xyzw[mask4, 0] = (R[mask4, 0, 2] + R[mask4, 2, 0]) / S[mask4]
        quat_xyzw[mask4, 1] = (R[mask4, 1, 2] + R[mask4, 2, 1]) / S[mask4]
        quat_xyzw[mask4, 2] = 0.25 * S[mask4]
        quat_xyzw[mask4, 3] = (R[mask4, 1, 0] - R[mask4, 0, 1]) / S[mask4]

        return xyzw_to_wxyz(quat_xyzw)
    elif isinstance(R, np.ndarray):
        quat_xyzw = Rotation.from_matrix(R).as_quat().astype(R.dtype)
        return xyzw_to_wxyz(quat_xyzw)
    else:
        gs.raise_exception(f"the input must be either torch.Tensor or np.ndarray. got: {type(R)=}")


def trans_to_T(trans):
    if isinstance(trans, torch.Tensor):
        T = torch.eye(4, dtype=trans.dtype, device=trans.device)
        if trans.ndim == 1:
            T[:3, 3] = trans
        elif trans.ndim == 2:
            T = T.unsqueeze(0).repeat(trans.shape[0], 1, 1)
            T[:, :3, 3] = trans
        else:
            gs.raise_exception(f"ndim expected to be 1 or 2, but got {trans.ndim=}")
        return T
    elif isinstance(trans, np.ndarray):
        T = np.eye(4, dtype=trans.dtype)
        if trans.ndim == 1:
            T[:3, 3] = trans
        elif trans.ndim == 2:
            T = np.tile(T, [trans.shape[0], 1, 1])
            T[:, :3, 3] = trans
        else:
            gs.raise_exception(f"ndim expected to be 1 or 2, but got {trans.ndim=}")
        return T
    else:
        gs.raise_exception(f"the input must be either torch.Tensor or np.ndarray. got: {type(trans)=}")


def trans_quat_to_T(trans, quat):
    if isinstance(trans, torch.Tensor) and isinstance(quat, torch.Tensor):
        T = torch.eye(4, dtype=quat.dtype, device=quat.device)
        if trans.ndim == 1:
            T[:3, 3] = trans
            T[:3, :3] = quat_to_R(quat)
        elif trans.ndim == 2:
            assert quat.ndim == 2
            T = T.unsqueeze(0).repeat(trans.shape[0], 1, 1)
            T[:, :3, 3] = trans
            T[:, :3, :3] = quat_to_R(quat)
        else:
            gs.raise_exception(f"ndim expected to be 1 or 2, but got {trans.ndim=}")
        return T
    elif isinstance(trans, np.ndarray) and isinstance(quat, np.ndarray):
        T = np.eye(4, dtype=np.result_type(trans, quat))
        if trans.ndim == 1:
            T[:3, 3] = trans
            T[:3, :3] = quat_to_R(quat)
        elif trans.ndim == 2:
            assert quat.ndim == 2
            T = np.tile(T, [trans.shape[0], 1, 1])
            T[:, :3, 3] = trans
            T[:, :3, :3] = quat_to_R(quat)
        else:
            gs.raise_exception(f"ndim expected to be 1 or 2, but got {trans.ndim=}")
        return T
    else:
        gs.raise_exception(
            f"both of the inputs must be torch.Tensor or np.ndarray. got: {type(trans)=} and {type(quat)=}"
        )


def T_to_trans_quat(T):
    if isinstance(T, torch.Tensor):
        if T.ndim == 2:
            trans = T[:3, 3]
            quat = R_to_quat(T[:3, :3])
        elif T.ndim == 3:
            trans = T[:, :3, 3]
            quat = R_to_quat(T[:, :3, :3])
        else:
            gs.raise_exception(f"ndim expected to be 2 or 3, but got {T.ndim=}")
        return trans, quat
    elif isinstance(T, np.ndarray):
        if T.ndim == 2:
            trans = T[:3, 3]
            quat = Rotation.from_matrix(T[:3, :3]).as_quat()
            quat = xyzw_to_wxyz(quat)
        elif T.ndim == 3:
            trans = T[:, :3, 3]
            quat = Rotation.from_matrix(T[:, :3, :3]).as_quat()
            quat = xyzw_to_wxyz(quat)
        else:
            gs.raise_exception(f"ndim expected to be 2 or 3, but got {T.ndim=}")
        return trans, quat
    else:
        raise TypeError(f"Input must be a torch.Tensor or np.ndarray. Got: {type(T)}")


def trans_R_to_T(trans, R):
    if isinstance(trans, torch.Tensor) and isinstance(R, torch.Tensor):
        T = torch.eye(4, dtype=R.dtype, device=R.device)
        if trans.ndim == 1:
            T[:3, 3] = trans
            T[:3, :3] = R
        elif trans.ndim == 2:
            assert R.ndim == 3
            T = T.unsqueeze(0).repeat(trans.shape[0], 1, 1)
            T[:, :3, 3] = trans
            T[:, :3, :3] = R
        else:
            gs.raise_exception(f"ndim expected to be 1 or 2, but got {trans.ndim=}")
        return T
    elif isinstance(trans, np.ndarray) and isinstance(R, np.ndarray):
        T = np.eye(4, dtype=np.result_type(trans, R))
        if trans.ndim == 1:
            T[:3, 3] = trans
            T[:3, :3] = R
        elif trans.ndim == 2:
            assert R.ndim == 3
            T = np.tile(T, [trans.shape[0], 1, 1])
            T[:, :3, 3] = trans
            T[:, :3, :3] = R
        else:
            gs.raise_exception(f"ndim expected to be 1 or 2, but got {trans.ndim=}")
        return T
    else:
        gs.raise_exception(f"both of the inputs must be torch.Tensor or np.ndarray. got: {type(trans)=} and {type(R)=}")


def R_to_T(R):
    if isinstance(R, torch.Tensor):
        T = torch.eye(4, dtype=R.dtype, device=R.device)
        if R.ndim == 2:
            T[:3, :3] = R
        elif R.ndim == 3:
            T = T.unsqueeze(0).repeat(R.shape[0], 1, 1)
            T[:, :3, :3] = R
        else:
            gs.raise_exception(f"ndim expected to be 2 or 3, but got {R.ndim=}")
        return T
    elif isinstance(R, np.ndarray):
        T = np.eye(4, dtype=R.dtype)
        if R.ndim == 2:
            T[:3, :3] = R
        elif R.ndim == 3:
            T = np.tile(T, [R.shape[0], 1, 1])
            T[:, :3, :3] = R
        else:
            gs.raise_exception(f"ndim expected to be 2 or 3, but got {R.ndim=}")
        return T
    else:
        gs.raise_exception(f"the input must be either torch.Tensor or np.ndarray. got: {type(R)=}")


def quat_to_T(quat):
    if isinstance(quat, torch.Tensor):
        T = torch.eye(4, dtype=quat.dtype, device=quat.device)
        if quat.ndim == 1:
            T[:3, :3] = quat_to_R(quat)
        elif quat.ndim == 2:
            T = T.unsqueeze(0).repeat(quat.shape[0], 1, 1)
            T[:, :3, :3] = quat_to_R(quat)
        else:
            gs.raise_exception(f"ndim expected to be 1 or 2, but got {quat.ndim=}")
        return T
    elif isinstance(quat, np.ndarray):
        T = np.eye(4, dtype=quat.dtype)
        if quat.ndim == 1:
            T[:3, :3] = Rotation.from_quat(quat, scalar_first=True).as_matrix()
        elif quat.ndim == 2:
            T = np.tile(T, [quat.shape[0], 1, 1])
            T[:, :3, :3] = Rotation.from_quat(quat, scalar_first=True).as_matrix()
        else:
            gs.raise_exception(f"ndim expected to be 1 or 2, but got {quat.ndim=}")
        return T
    else:
        gs.raise_exception(f"the input must be either torch.Tensor or np.ndarray. got: {type(quat)=}")


def quat_to_xyz(quat, rpy=False, degrees=False):
    if isinstance(quat, torch.Tensor):
        # Extract quaternion components
        qw, qx, qy, qz = quat.unbind(-1)

        # Roll (x-axis rotation)
        if rpy:
            sinr_cosp = 2 * (qw * qx + qy * qz)
        else:
            sinr_cosp = -2 * (qy * qz - qw * qx)
        cosr_cosp = 1 - 2 * (qx * qx + qy * qy)
        roll = torch.atan2(sinr_cosp, cosr_cosp)

        # Pitch (y-axis rotation)
        if rpy:
            sinp = 2 * (qw * qy - qz * qx)
        else:
            sinp = 2 * (qx * qz + qw * qy)
        pitch = torch.where(torch.abs(sinp) >= 1, torch.sign(sinp) * (torch.pi / 2), torch.asin(sinp))

        # Yaw (z-axis rotation)
        if rpy:
            siny_cosp = 2 * (qw * qz + qx * qy)
        else:
            siny_cosp = -2 * (qx * qy - qw * qz)
        cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
        yaw = torch.atan2(siny_cosp, cosy_cosp)

        rpy = torch.stack([roll, pitch, yaw], dim=-1)
        if degrees:
            rpy *= 180.0 / torch.pi
        return rpy
    elif isinstance(quat, np.ndarray):
        rot = Rotation.from_quat(quat, scalar_first=True)
        if rpy:
            return rot.as_euler("xyz", degrees=degrees)
        return rot.as_euler("zyx", degrees=degrees)[::-1]
    else:
        gs.raise_exception(f"the input must be either torch.Tensor or np.ndarray. got: {type(quat)=}")


def xyz_to_quat(euler_xyz, rpy=False, degrees=False):
    if isinstance(euler_xyz, torch.Tensor):
        if degrees:
            euler_xyz *= torch.pi / 180.0
        roll, pitch, yaw = euler_xyz.unbind(-1)
        cosr = (roll * 0.5).cos()
        sinr = (roll * 0.5).sin()
        cosp = (pitch * 0.5).cos()
        sinp = (pitch * 0.5).sin()
        cosy = (yaw * 0.5).cos()
        siny = (yaw * 0.5).sin()
        sign = 1.0 if rpy else -1.0
        qw = cosr * cosp * cosy + sign * sinr * sinp * siny
        qx = sinr * cosp * cosy - sign * cosr * sinp * siny
        qy = cosr * sinp * cosy + sign * sinr * cosp * siny
        qz = cosr * cosp * siny - sign * sinr * sinp * cosy
        return torch.stack([qw, qx, qy, qz], dim=-1)
    elif isinstance(euler_xyz, np.ndarray):
        if rpy:
            rot = Rotation.from_euler("xyz", euler_xyz, degrees=degrees)
        else:
            rot = Rotation.from_euler("zyx", euler_xyz[::-1], degrees=degrees)
        return rot.as_quat(scalar_first=True)
    else:
        gs.raise_exception(f"the input must be either torch.Tensor or np.ndarray. got: {type(euler_xyz)=}")


def transform_by_quat(v, quat):
    if isinstance(v, torch.Tensor) and isinstance(quat, torch.Tensor):
        qvec = quat[..., 1:]
        t = qvec.cross(v, dim=-1) * 2
        return v + quat[..., :1] * t + qvec.cross(t, dim=-1)
    elif isinstance(v, np.ndarray) and isinstance(quat, np.ndarray):
        return transform_by_R(v, quat_to_R(quat))
    else:
        gs.raise_exception(f"both of the inputs must be torch.Tensor or np.ndarray. got: {type(v)=} and {type(quat)=}")


def axis_angle_to_quat(angle, axis):
    if isinstance(angle, torch.Tensor) and isinstance(axis, torch.Tensor):
        theta = (angle / 2).unsqueeze(-1)
        xyz = normalize(axis) * theta.sin()
        w = theta.cos()
        return normalize(torch.cat([w, xyz], dim=-1))
    elif isinstance(angle, np.ndarray) and isinstance(axis, np.ndarray):
        theta = (angle / 2)[..., None]
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


def transform_by_T(pos, T):
    """
    Transforms 3D points by a 4x4 transformation matrix or a batch of matrices,
    supporting both NumPy arrays and PyTorch tensors.

    Parameters
    ----------
    pos: np.ndarray | torch.Tensor
        A numpy array or torch tensor of 3D points. Can be a single point
         (3,), a batch of points (B, 3), or a batched batch of points (B, N, 3).
    T: np.ndarray | torch.Tensor
        The 4x4 transformation matrix or a batch of B transformation
        matrices of shape (B, 4, 4). Must be of the same type as `pos`.

    Returns
    -------
        The transformed points in a shape corresponding to the input dimensions.
    """
    assert pos.shape[-1] == 3, "Input positions must have 3 dimensions"

    if T.ndim == 2:
        T = T.reshape(1, 4, 4)

    if isinstance(pos, torch.Tensor) and isinstance(T, torch.Tensor):
        if pos.ndim > 1:
            ones_shape = pos.shape[:-1] + (1,)
            pos_hom = torch.cat([pos, torch.ones(ones_shape, dtype=pos.dtype, device=pos.device)], dim=-1)
        else:
            pos_hom = torch.cat([pos, torch.tensor([1.0], dtype=pos.dtype, device=pos.device)])
    elif isinstance(pos, np.ndarray) and isinstance(T, np.ndarray):
        if pos.ndim > 1:
            ones_shape = pos.shape[:-1] + (1,)
            pos_hom = np.concatenate([pos, np.ones(ones_shape, dtype=pos.dtype)], axis=-1)
        else:
            pos_hom = np.append(pos, 1)
    else:
        gs.raise_exception(f"Inputs must be both torch.Tensor or both np.ndarray. Got: {type(pos)=} and {type(T)=}")

    if pos_hom.ndim == 1:
        pos_hom = pos_hom.reshape(1, 1, -1)
    elif pos_hom.ndim == 2:
        assert T.shape[0] == 1 or T.shape[0] == pos.shape[0], f"{T.shape}, {pos.shape}"
        pos_hom = pos_hom.reshape(-1, 1, 4)

    pos_hom_t = pos_hom.swapaxes(-1, -2)  # (..., N, 4) -> (..., 4, N)
    transformed_hom = T @ pos_hom_t
    transformed_hom = transformed_hom.swapaxes(-1, -2)[..., :3]

    if pos.ndim == 1:
        transformed_hom = transformed_hom.reshape(-1)
    elif pos.ndim == 2:
        transformed_hom = transformed_hom.reshape(-1, 3)
    return transformed_hom


def transform_by_R(pos, R):
    """
    Transforms 3D points by a 3x3 rotation matrix or a batch of matrices,
    supporting both NumPy arrays and PyTorch tensors.

    Parameters
    ----------
    pos: np.ndarray | torch.Tensor
        A numpy array or torch tensor of 3D points. Can be a single point
         (3,), a batch of points (B, 3), or a batched batch of points (B, N, 3).
    T: np.ndarray | torch.Tensor
        The 3x3 rotation matrix or a batch of B rotation
        matrices of shape (B, 3, 3). Must be of the same type as `pos`.

    Returns
    -------
        The transformed points in a shape corresponding to the input dimensions.
    """
    assert pos.shape[-1] == 3

    dim_added = False
    if R.ndim == 2:
        R = R[None]
        dim_added = True
    if pos.ndim == 3:
        new_pos = (R @ pos.swapaxes(-1, -2)).swapaxes(-1, -2)
    elif pos.ndim == 2:
        new_pos = (R @ pos[:, :, None])[..., 0]
    elif pos.ndim == 1:
        new_pos = (R @ pos[None, :, None])[..., 0]
        if dim_added:
            new_pos = new_pos[0]
    else:
        gs.raise_exception(f"invalid input dim for pos: {pos.shape=}")
    return new_pos


def inv_transform_by_T(pos, T):
    if isinstance(T, torch.Tensor):
        T_inv = torch.linalg.inv(T)
    elif isinstance(T, np.ndarray):
        T_inv = np.linalg.inv(T)
    else:
        gs.raise_exception(f"Inputs must be both torch.Tensor or both np.ndarray. Got: {type(pos)=} and {type(T)=}")
    return transform_by_T(pos, T_inv)


# ------------------------------------------------------------------------------------
# ------------------------------------- numpy ----------------------------------------
# ------------------------------------------------------------------------------------


def euler_to_quat(euler_xyz):
    # added for backward compatibility
    if isinstance(euler_xyz, tuple):
        euler_xyz = np.array(euler_xyz)
    if isinstance(euler_xyz, list):
        euler_xyz = np.array(euler_xyz)
    return xyz_to_quat(euler_xyz)


def scale_to_T(scale):
    T = np.eye(4, dtype=scale.dtype)
    T[[0, 1, 2], [0, 1, 2]] = scale
    return T


def euler_to_R(euler_xyz):
    return Rotation.from_euler("xyz", euler_xyz, degrees=True).as_matrix()


def z_up_to_R(z, up=np.array([0, 0, 1])):
    z = normalize(z)
    up = normalize(up)
    x = np.cross(up, z)
    if np.linalg.norm(x) == 0:
        R = np.eye(3)
    else:
        x = normalize(x)
        y = normalize(np.cross(z, x))
        R = np.vstack([x, y, z]).T
    return R


def pos_lookat_up_to_T(pos, lookat, up):
    pos = np.array(pos)
    lookat = np.array(lookat)
    up = np.array(up)
    if np.all(pos == lookat):
        z = np.array([1, 0, 0])
    else:
        z = pos - lookat
    R = z_up_to_R(z, up=up)
    return trans_R_to_T(pos, R)


def T_to_pos_lookat_up(T):
    pos = T[:3, 3]
    lookat = T[:3, 3] - T[:3, 2]
    up = T[:3, 1]
    return pos, lookat, up


def z_to_R(z):
    """
    Convert a vector to a rotation matrix such that the z-axis points to the vector.
    This operation is computed by rotating the world frame by moving the original z-axis to the given vector via the shortest path.
    """
    z = np.array(z)
    if z.ndim == 1:
        if np.linalg.norm(z) == 0:
            z = np.array([1, 0, 0])
    elif z.ndim == 2:
        z[np.linalg.norm(z, axis=-1) == 0] = np.array([1, 0, 0])

    z = z / np.linalg.norm(z, axis=-1, keepdims=True)
    # angle between z and world z
    angle = np.arccos(np.dot(z, np.array([0, 0, 1])))
    # axis of rotation
    axis = np.cross(np.array([0, 0, 1]), z)

    if axis.ndim == 1:
        if np.linalg.norm(axis) == 0:
            axis = np.array([1, 0, 0])
    elif axis.ndim == 2:
        axis[np.linalg.norm(axis, axis=-1) == 0] = np.array([1, 0, 0])

    return axis_angle_to_R(axis, angle)


def axis_angle_to_R(axis, theta):
    axis = np.array(axis)
    axis = axis / np.linalg.norm(axis, axis=-1, keepdims=True)
    if axis.ndim == 2:
        theta = theta[:, None]
    return rotvec_to_R(axis * theta)


def rotvec_to_R(rotvec):
    return Rotation.from_rotvec(rotvec).as_matrix()


def quat_to_rotvec(quat):
    return Rotation.from_quat(quat, scalar_first=True).as_rotvec()


def rotvec_to_quat(rotvec):
    return Rotation.from_rotvec(rotvec).as_quat(scalar_first=True)


def compute_camera_angle(camera_pos, camera_lookat):
    camera_dir = np.array(camera_lookat) - np.array(camera_pos)

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
    return np.zeros(3)


def identity_quat():
    return np.array([1.0, 0.0, 0.0, 0.0])


def tc_zero_pos():
    return torch.zeros(3, device=gs.device)


def tc_identity_quat():
    return torch.tensor([1.0, 0.0, 0.0, 0.0], device=gs.device)


def nowhere():
    # let's inject a bit of humor here
    return np.array([2333333, 6666666, 5201314])


def default_solver_params():
    """
    Default solver parameters (timeconst, dampratio, dmin, dmax, width, mid, power).

    Reference: https://mujoco.readthedocs.io/en/latest/modeling.html#solver-parameters
    """
    return np.array([0.0, 1.0e00, 9.0e-01, 9.5e-01, 1.0e-03, 5.0e-01, 2.0e00])


def default_friction():
    return 1.0


def default_dofs_kp(n=6):
    return np.tile(100.0, n).astype(gs.np_float)


def default_dofs_kv(n=6):
    return np.tile(10.0, n).astype(gs.np_float)


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
