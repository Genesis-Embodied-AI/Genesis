from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import mujoco
import scipy.spatial.transform as Rotation
import trimesh as libtrimesh

@dataclass
class CollisionBody:
    body_id: int
    name: str
    vertices: np.ndarray
    faces: np.ndarray
    position: np.ndarray
    rotation: np.ndarray


@dataclass
class JointInfo:
    joint_id: int
    name: str
    joint_type: int
    body_id: int
    parent_body_id: int
    world_pos: np.ndarray
    world_axis: np.ndarray

def _transform_points(points: np.ndarray, rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    return (points @ rotation.T) + translation

def _quat_wxyz_to_mat(quat_wxyz: np.ndarray) -> np.ndarray:
    return Rotation.Rotation.from_quat([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]]).as_matrix()

def _accumulate_body_to_parent_transform(model: mujoco.MjModel, body_id: int) -> Tuple[np.ndarray, np.ndarray]:
    pos = model.body_pos[body_id].astype(np.float32, copy=False)
    quat = model.body_quat[body_id].astype(np.float32, copy=False)
    rot = _quat_wxyz_to_mat(quat)
    return rot, pos

def _merge_target_and_transform(
    model: mujoco.MjModel, body_id: int
) -> Tuple[int, np.ndarray, np.ndarray]:
    # Walk up fixed-to-parent bodies (no joint) and accumulate transform to target frame.
    target_id = body_id
    R_acc = np.eye(3, dtype=np.float32)
    t_acc = np.zeros(3, dtype=np.float32)

    while model.body_jntnum[target_id] == 0 and model.body_parentid[target_id] != 0:
        R_local, t_local = _accumulate_body_to_parent_transform(model, target_id)
        R_acc = R_local @ R_acc
        t_acc = R_local @ t_acc + t_local
        target_id = int(model.body_parentid[target_id])

    return target_id, R_acc, t_acc


def _merge_target_id(model: mujoco.MjModel, body_id: int) -> int:
    target_id = body_id
    while model.body_jntnum[target_id] == 0 and model.body_parentid[target_id] != 0:
        target_id = int(model.body_parentid[target_id])
    return target_id
def _get_geom_mesh_id(model: mujoco.MjModel, geom_id: int) -> int:
    if hasattr(model, "geom_meshid"):
        return int(model.geom_meshid[geom_id])
    return int(model.geom_dataid[geom_id])

def _mesh_from_primitive(geom_type: int, geom_size: np.ndarray) -> Tuple[np.ndarray, np.ndarray] | None:
    size = np.asarray(geom_size, dtype=np.float32)
    if geom_type == mujoco.mjtGeom.mjGEOM_BOX:
        # MuJoCo box size is half-extent.
        mesh = libtrimesh.creation.box(extents=2.0 * size[:3])
    elif geom_type == mujoco.mjtGeom.mjGEOM_SPHERE:
        mesh = libtrimesh.creation.icosphere(subdivisions=2, radius=float(size[0]))
    elif geom_type == mujoco.mjtGeom.mjGEOM_CYLINDER:
        radius = float(size[0])
        height = float(2.0 * size[1])
        mesh = libtrimesh.creation.cylinder(radius=radius, height=height, sections=24)
    elif geom_type == mujoco.mjtGeom.mjGEOM_CAPSULE:
        radius = float(size[0])
        height = float(2.0 * size[1])
        mesh = libtrimesh.creation.capsule(radius=radius, height=height, count=[24, 12])
    elif geom_type == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
        mesh = libtrimesh.creation.icosphere(subdivisions=2, radius=1.0)
        mesh.apply_scale(size[:3])
    else:
        return None

    return mesh.vertices.astype(np.float32), mesh.faces.astype(np.int32)


def _extract_geom_mesh_in_body_frame(model: mujoco.MjModel, geom_id: int) -> Tuple[np.ndarray, np.ndarray] | None:
    geom_type = int(model.geom_type[geom_id])
    if geom_type == mujoco.mjtGeom.mjGEOM_MESH:
        if model.geom_group[geom_id] == 2:
            return None
        mesh_id = _get_geom_mesh_id(model, geom_id)
        mesh_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, mesh_id)
        if mesh_name == "finger_0" or mesh_name == "finger_1":
            return None
        vert_start = int(model.mesh_vertadr[mesh_id])
        vert_count = int(model.mesh_vertnum[mesh_id])
        face_start = int(model.mesh_faceadr[mesh_id])
        face_count = int(model.mesh_facenum[mesh_id])

        vertices = model.mesh_vert[vert_start : vert_start + vert_count].astype(np.float32, copy=True)
        faces = model.mesh_face[face_start : face_start + face_count].astype(np.int32, copy=True)
        faces = faces.reshape(-1, 3)
    else:
        primitive = _mesh_from_primitive(geom_type, model.geom_size[geom_id])
        if primitive is None:
            return None
        vertices, faces = primitive

    geom_pos = model.geom_pos[geom_id].astype(np.float32, copy=False)
    geom_quat = model.geom_quat[geom_id].astype(np.float32, copy=False)
    geom_rot = _quat_wxyz_to_mat(geom_quat)
    vertices = _transform_points(vertices, geom_rot, geom_pos)
    return vertices, faces


def _is_collision_geom(model: mujoco.MjModel, geom_id: int) -> bool:
    if model.geom_group[geom_id] == 3:
        return True
    if model.geom_contype[geom_id] != 0:
        return True
    if model.geom_conaffinity[geom_id] != 0:
        return True
    return False


def load_collision_robot(xml_path: str) -> Tuple[mujoco.MjModel, mujoco.MjData, List[CollisionBody], List[JointInfo]]:
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    body_meshes: Dict[int, Dict[str, np.ndarray]] = {}

    for geom_id in range(model.ngeom):
        if not _is_collision_geom(model, geom_id):
            continue

        body_id = int(model.geom_bodyid[geom_id])
        target_body_id, body_to_target_R, body_to_target_t = _merge_target_and_transform(model, body_id)
        geom_mesh = _extract_geom_mesh_in_body_frame(model, geom_id)
        if geom_mesh is None:
            continue
        vertices, faces = geom_mesh
        if target_body_id != body_id:
            vertices = _transform_points(vertices, body_to_target_R, body_to_target_t)

        entry = body_meshes.get(target_body_id)
        if entry is None:
            entry = {"vertices": np.empty((0, 3), dtype=np.float32), "faces": np.empty((0, 3), dtype=np.int32)}
            body_meshes[target_body_id] = entry

        vert_offset = entry["vertices"].shape[0]
        entry["vertices"] = np.vstack([entry["vertices"], vertices])
        entry["faces"] = np.vstack([entry["faces"], faces + vert_offset])

    collision_bodies: List[CollisionBody] = []
    for body_id, entry in body_meshes.items():
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        pos = data.xpos[body_id].astype(np.float32, copy=True)
        rot = data.xmat[body_id].reshape(3, 3).astype(np.float32, copy=True)
        collision_bodies.append(
            CollisionBody(
                body_id=body_id,
                name=name,
                vertices=entry["vertices"],
                faces=entry["faces"],
                position=pos,
                rotation=rot,
            )
        )

    joints: List[JointInfo] = []
    for joint_id in range(model.njnt):
        joint_type = int(model.jnt_type[joint_id])
        if joint_type not in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
            continue

        body_id = int(model.jnt_bodyid[joint_id])
        parent_body_id = int(model.body_parentid[body_id])
        parent_body_id = _merge_target_id(model, parent_body_id)
        body_pos = data.xpos[body_id]
        body_rot = data.xmat[body_id].reshape(3, 3)
        joint_pos = model.jnt_pos[joint_id]
        joint_axis = model.jnt_axis[joint_id]

        world_pos = body_pos + body_rot @ joint_pos
        world_axis = body_rot @ joint_axis
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)

        joints.append(
            JointInfo(
                joint_id=joint_id,
                name=name,
                joint_type=joint_type,
                body_id=body_id,
                parent_body_id=parent_body_id,
                world_pos=world_pos.astype(np.float32),
                world_axis=world_axis.astype(np.float32),
            )
        )

    return model, data, collision_bodies, joints
