import numpy as np
import taichi as ti
import torch
from scipy.spatial.transform import Rotation
import genesis as gs

# ------------------------------------------------------------------------------------
# ------------------------------------- taichi ----------------------------------------
# ------------------------------------------------------------------------------------


@ti.func
def ti_transform_quat_by_quat(v, u):
    # This method transforms quat_v by quat_u
    # This is equivalent to quatmul(quat_u, quat_v) or R_u @ R_v
    terms = v.outer_product(u)
    w = terms[0, 0] - terms[1, 1] - terms[2, 2] - terms[3, 3]
    x = terms[0, 1] + terms[1, 0] - terms[2, 3] + terms[3, 2]
    y = terms[0, 2] + terms[1, 3] + terms[2, 0] - terms[3, 1]
    z = terms[0, 3] - terms[1, 2] + terms[2, 1] + terms[3, 0]
    return ti.Vector([w, x, y, z]).normalized()


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
    # make b a normal vector. however if a is a zero vector, zero b as well.
    b = b.normalized()
    if a.norm() < gs.EPS:
        b = b * 0.0
    return b, a.cross(b)


@ti.func
def orthogonals2(a):
    """Returns orthogonal vectors `b` and `c`, given a normal vector `a`."""
    y, z = ti.Vector([0.0, 1.0, 0.0], dt=gs.ti_float), ti.Vector([0.0, 0.0, 1.0], dt=gs.ti_float)
    b = z
    if -0.5 < a[1] and a[1] < 0.5:
        b = y
    b = b - a * a.dot(b)
    # make b a normal vector. however if a is a zero vector, zero b as well.
    b = b.normalized()
    if a.norm() < gs.EPS:
        b = b * 0.0

    # perturb with some noise so that they do not align with world axes
    c = (a.cross(b) + 0.1 * b).normalized()
    b = c.cross(a).normalized()
    return b, c


@ti.func
def imp_aref(params, neg_penetration, vel, pos):
    # The first term in parms is the timeconst parsed from mjcf. However, we don't use it here but use the one passed in, which is 2*substep_dt.
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
        return Rotation.from_quat(wxyz_to_xyzw(quat)).as_matrix().astype(quat.dtype)
    else:
        gs.raise_exception(f"the input must be either torch.Tensor or np.ndarray. got: {type(quat)=}")


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
            T[:3, :3] = Rotation.from_quat(wxyz_to_xyzw(quat)).as_matrix()
        elif trans.ndim == 2:
            assert quat.ndim == 2
            T = np.tile(T, [trans.shape[0], 1, 1])
            T[:, :3, 3] = trans
            T[:, :3, :3] = Rotation.from_quat(wxyz_to_xyzw(quat)).as_matrix()
        else:
            gs.raise_exception(f"ndim expected to be 1 or 2, but got {trans.ndim=}")
        return T
    else:
        gs.raise_exception(
            f"both of the inputs must be torch.Tensor or np.ndarray. got: {type(trans)=} and {type(quat)=}"
        )


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
            T[:3, :3] = Rotation.from_quat(wxyz_to_xyzw(quat)).as_matrix()
        elif quat.ndim == 2:
            T = np.tile(T, [quat.shape[0], 1, 1])
            T[:, :3, :3] = Rotation.from_quat(wxyz_to_xyzw(quat)).as_matrix()
        else:
            gs.raise_exception(f"ndim expected to be 1 or 2, but got {quat.ndim=}")
        return T
    else:
        gs.raise_exception(f"the input must be either torch.Tensor or np.ndarray. got: {type(quat)=}")


def quat_to_xyz(quat):
    if isinstance(quat, torch.Tensor):
        # Extract quaternion components
        qw, qx, qy, qz = quat.unbind(-1)
        # Roll (x-axis rotation)
        sinr_cosp = 2 * (qw * qx + qy * qz)
        cosr_cosp = 1 - 2 * (qx * qx + qy * qy)
        roll = torch.atan2(sinr_cosp, cosr_cosp)
        # Pitch (y-axis rotation)
        sinp = 2 * (qw * qy - qz * qx)
        pitch = torch.where(
            torch.abs(sinp) >= 1,
            torch.sign(sinp) * torch.tensor(torch.pi / 2),
            torch.asin(sinp),
        )
        # Yaw (z-axis rotation)
        siny_cosp = 2 * (qw * qz + qx * qy)
        cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
        yaw = torch.atan2(siny_cosp, cosy_cosp)
        return torch.stack([roll, pitch, yaw], dim=-1) * 180.0 / torch.tensor(np.pi)
    elif isinstance(quat, np.ndarray):
        return Rotation.from_quat(wxyz_to_xyzw(quat)).as_euler("xyz", degrees=True)
    else:
        gs.raise_exception(f"the input must be either torch.Tensor or np.ndarray. got: {type(quat)=}")


def xyz_to_quat(euler_xyz):
    if isinstance(euler_xyz, torch.Tensor):
        euler_xyz = euler_xyz * torch.tensor(np.pi) / 180.0
        roll, pitch, yaw = euler_xyz.unbind(-1)
        cosr = (roll * 0.5).cos()
        sinr = (roll * 0.5).sin()
        cosp = (pitch * 0.5).cos()
        sinp = (pitch * 0.5).sin()
        cosy = (yaw * 0.5).cos()
        siny = (yaw * 0.5).sin()
        qw = cosr * cosp * cosy + sinr * sinp * siny
        qx = sinr * cosp * cosy - cosr * sinp * siny
        qy = cosr * sinp * cosy + sinr * cosp * siny
        qz = cosr * cosp * siny - sinr * sinp * cosy
        return torch.stack([qw, qx, qy, qz], dim=-1)
    elif isinstance(euler_xyz, np.ndarray):
        return xyzw_to_wxyz(Rotation.from_euler("xyz", euler_xyz, degrees=True).as_quat())
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


def R_to_quat(R):
    quat_xyzw = Rotation.from_matrix(R).as_quat().astype(R.dtype)
    return xyzw_to_wxyz(quat_xyzw)


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
    return Rotation.from_quat(wxyz_to_xyzw(quat)).as_rotvec()


def rotvec_to_quat(rotvec):
    return xyzw_to_wxyz(Rotation.from_rotvec(rotvec).as_quat())


def compute_camera_angle(camera_pos, camera_lookat):
    camera_dir = np.array(camera_lookat) - np.array(camera_pos)

    # rotation around vertical (y) axis
    angle_x = np.arctan2(-camera_dir[0], -camera_dir[2])

    # rotation w.r.t horizontal plane
    angle_y = np.arctan2(camera_dir[1], np.linalg.norm([camera_dir[0], camera_dir[2]]))

    angle_z = 0.0

    return np.array([angle_x, angle_y, angle_z])


def transform_by_R(pos, R):
    assert pos.shape[-1] == 3

    if R.ndim == 2:
        if pos.ndim == 2:
            new_pos = (R @ pos.T).T
        elif pos.ndim == 1:
            new_pos = R @ pos
        else:
            assert False
    elif R.ndim == 3:  # batched R and pos
        if pos.ndim == 2:
            new_pos = (R @ pos[:, :, None]).squeeze(-1)
        else:
            assert False

    return new_pos


def transform_by_T(pos, T):
    assert pos.shape[-1] == 3
    assert T.ndim == 2  # TODO: handle batched T

    if pos.ndim == 2:
        new_pos = np.hstack([pos, np.ones_like(pos[:, :1])]).T
        new_pos = (T @ new_pos).T
        new_pos = new_pos[:, :3]
    elif pos.ndim == 1:
        new_pos = np.append(pos, np.array(1, dtype=pos.dtype))
        new_pos = T @ new_pos
        new_pos = new_pos[:3]
    else:
        assert False

    return new_pos


def inv_transform_by_T(pos, T):
    T_inv = np.linalg.inv(T)
    return transform_by_T(pos, T_inv)


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


def default_dofs_kp(n=6):
    return np.tile(100.0, n).astype(gs.np_float)


def default_dofs_kv(n=6):
    return np.tile(10.0, n).astype(gs.np_float)


def default_dofs_force_range(n=6):
    # TODO: This is big enough for robot arms, but is this general?
    return np.tile([[-100.0, 100.0]], [n, 1])


def default_dofs_limit(n=6):
    return np.tile([[-np.inf, np.inf]], [n, 1])


def default_dofs_invweight(n=6):
    return np.ones(n)


def default_dofs_damping(n=6):
    return np.ones(n)


def free_dofs_damping(n=6):
    return np.zeros(n)


def default_dofs_motion_ang(n=6):
    if n == 6:
        return np.array(
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        )
    elif n == 0:
        return np.zeros((0, 3))
    else:
        assert False


def default_dofs_motion_vel(n=6):
    if n == 6:
        return np.array(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
        )
    elif n == 0:
        return np.zeros((0, 3))
    else:
        return False


def default_dofs_stiffness(n=6):
    return np.zeros(n)


def default_solver_params(n=6, substep_dt=0.01):
    """
    Default solver parameters (timeconst, dampratio, dmin, dmax, width, mid, power). Reference: https://mujoco.readthedocs.io/en/latest/modeling.html#solver-parameters
    Note that timeconst here will not be used in the current workflow. Instead, it will be computed using 2 * substep_dt.
    """

    solver_params = np.array([2 * substep_dt, 1.0e00, 9.0e-01, 9.5e-01, 1.0e-03, 5.0e-01, 2.0e00])
    return np.repeat(solver_params[None], n, axis=0)


def default_friction():
    return 1.0


def default_dofs_armature(n=6):
    return np.full(n, 0.1)


def free_dofs_armature(n=6):
    return np.zeros(n)


@ti.data_oriented
class SpatialHasher:
    def __init__(self, cell_size, grid_res, n_slots=None):
        self.cell_size = cell_size
        self.grid_res = grid_res

        if n_slots is None:
            self.n_slots = np.prod(grid_res)
        else:
            self.n_slots = n_slots

    def build(self):
        # number of elements in each slot
        self.slot_size = ti.field(gs.ti_int, shape=self.n_slots)
        # element index offset in each slot
        self.slot_start = ti.field(gs.ti_int, shape=self.n_slots)

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

        for i in range(n):
            if active[i]:
                slot_idx = self.pos_to_slot(pos[i])
                ti.atomic_add(self.slot_size[slot_idx], 1)

        cur_cnt = 0
        for i in range(self.n_slots):
            self.slot_start[i] = ti.atomic_add(cur_cnt, self.slot_size[i])

        for i in range(n):
            if active[i]:
                slot_idx = self.pos_to_slot(pos[i])
                reordered_idx[i] = ti.atomic_add(self.slot_start[slot_idx], 1)

        # recover slot_start
        for i in range(self.n_slots):
            self.slot_start[i] -= self.slot_size[i]

    @ti.func
    def for_all_neighbors(
        self,
        i,
        pos,
        task_range,
        ret: ti.template(),
        task: ti.template(),
    ):
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
        base = self.pos_to_grid(pos[i])
        for offset in ti.grouped(ti.ndrange((-1, 2), (-1, 2), (-1, 2))):
            slot_idx = self.grid_to_slot(base + offset)
            for j in range(self.slot_start[slot_idx], self.slot_size[slot_idx] + self.slot_start[slot_idx]):
                if i != j and (pos[i] - pos[j]).norm() < task_range:
                    task(i, j, ret)

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
