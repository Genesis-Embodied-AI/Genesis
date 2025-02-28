import os

import mujoco
import numpy as np
from PIL import Image

import genesis as gs
from genesis.ext import trimesh
from genesis.ext.trimesh.visual.texture import TextureVisuals

from . import geom as gu
from .misc import get_assets_dir


def parse_mjcf(path):
    path = os.path.join(get_assets_dir(), path)
    mj = mujoco.MjModel.from_xml_path(path)
    return mj


def parse_link(mj, i_l, q_offset, dof_offset, qpos0_offset, scale):

    # mj.body
    l_info = dict()

    name_start = mj.name_bodyadr[i_l]
    if i_l + 1 < mj.nbody:
        name_end = mj.name_bodyadr[i_l + 1]
        l_info["name"] = mj.names[name_start:name_end].decode("utf-8").replace("\x00", "")
    else:
        l_info["name"] = mj.names[name_start:].decode("utf-8").split("\x00")[0]

    l_info["pos"] = mj.body_pos[i_l]
    l_info["quat"] = mj.body_quat[i_l]
    l_info["inertial_pos"] = mj.body_ipos[i_l]
    l_info["inertial_quat"] = mj.body_iquat[i_l]
    l_info["inertial_i"] = np.diag(mj.body_inertia[i_l])
    l_info["inertial_mass"] = float(mj.body_mass[i_l])
    l_info["parent_idx"] = int(mj.body_parentid[i_l] - 1)
    l_info["invweight"] = float(mj.body_invweight0[i_l, 0])

    l_info["pos"] *= scale
    l_info["inertial_pos"] *= scale
    l_info["inertial_mass"] *= scale**3
    l_info["inertial_i"] *= scale**5
    l_info["invweight"] /= scale**3

    # mj.jnt =================================
    def add_actuator(j_info, i_j=None):
        # mj.actuator
        j_info["dofs_kp"] = gu.default_dofs_kp(j_info["n_dofs"])
        j_info["dofs_kv"] = gu.default_dofs_kv(j_info["n_dofs"])
        j_info["dofs_force_range"] = gu.default_dofs_force_range(j_info["n_dofs"])

        if i_j is not None:
            for i_a in range(len(mj.actuator_trnid)):
                if mj.actuator_trnid[i_a, 0] == i_j and mj.actuator_trntype[i_a] == mujoco.mjtTrn.mjTRN_JOINT:
                    if mj.actuator_gainprm[i_a, 0] != -mj.actuator_biasprm[i_a, 1]:
                        gs.logger.warning("`kp` in `gainprm` doesn't match `-kp` in `biasprm`.")
                    j_info["dofs_kp"] = np.tile(mj.actuator_gainprm[i_a, 0], j_info["n_dofs"])
                    j_info["dofs_kv"] = np.tile(-mj.actuator_biasprm[i_a, 2], j_info["n_dofs"])
                    j_info["dofs_force_range"] = np.tile(mj.actuator_forcerange[i_a], (j_info["n_dofs"], 1))
                    break

        return j_info

    def add_more_joint_info(j_info, jnt_offset=0):
        d_off = dof_offset + jnt_offset
        qpos0_off = qpos0_offset + jnt_offset

        j_info["dofs_damping"] = np.array(mj.dof_damping[d_off : d_off + j_info["n_dofs"]])
        j_info["dofs_invweight"] = np.array(mj.dof_invweight0[d_off : d_off + j_info["n_dofs"]])
        j_info["dofs_armature"] = np.array(mj.dof_armature[d_off : d_off + j_info["n_dofs"]])
        if j_info["n_qpos0"] == 4 and j_info["type"] == gs.JOINT_TYPE.SPHERICAL:
            # this is a real mujoco ball joint
            j_info["init_qpos"] = gu.quat_to_xyz(mj.qpos0[qpos0_off : qpos0_off + 4])
        else:
            j_info["init_qpos"] = np.array(mj.qpos0[qpos0_off : qpos0_off + j_info["n_qpos0"]])

        # apply scale
        j_info["pos"] *= scale
        return j_info

    jnt_adr = mj.body_jntadr[i_l]
    jnt_num = mj.body_jntnum[i_l]

    final_joint_list = []
    if jnt_adr == -1:  # fixed joint
        j_info = dict()
        j_info["dofs_motion_ang"] = np.zeros((0, 3))
        j_info["dofs_motion_vel"] = np.zeros((0, 3))
        j_info["dofs_limit"] = np.zeros((0, 2))
        j_info["dofs_stiffness"] = np.zeros((0))
        j_info["dofs_sol_params"] = np.zeros((0, 7))

        j_info["name"] = f'{l_info["name"]}_joint'
        j_info["type"] = gs.JOINT_TYPE.FIXED
        j_info["pos"] = np.array([0.0, 0.0, 0.0])
        j_info["quat"] = np.array([1.0, 0.0, 0.0, 0.0])
        j_info["n_qs"] = 0
        j_info["n_dofs"] = 0
        j_info["n_qpos0"] = 0

        j_info = add_more_joint_info(add_actuator(j_info))
        final_joint_list.append(j_info)
    else:
        j_info_list = []
        for i_j in range(jnt_adr, jnt_adr + jnt_num):
            j_info = dict()
            j_info["quat"] = np.array([1.0, 0.0, 0.0, 0.0])
            name_start = mj.name_jntadr[i_j]
            if i_j + 1 < mj.njnt:
                name_end = mj.name_jntadr[i_j + 1]
            else:
                name_end = mj.name_geomadr[0]
            j_info["name"] = mj.names[name_start:name_end].decode("utf-8").replace("\x00", "")
            j_info["pos"] = np.array(mj.jnt_pos[i_j])

            if len(j_info["name"]) == 0:
                j_info["name"] = f'{l_info["name"]}_joint'

            mj_type = mj.jnt_type[i_j]
            mj_stiffness = mj.jnt_stiffness[i_j]
            mj_limit = mj.jnt_range[i_j] if mj.jnt_limited[i_j] == 1 else np.array([-np.inf, np.inf])
            mj_axis = mj.jnt_axis[i_j]
            mj_sol_params = np.concatenate((mj.jnt_solref[i_j], mj.jnt_solimp[i_j]))

            if mj_type == mujoco.mjtJoint.mjJNT_HINGE:
                j_info["dofs_motion_ang"] = np.array([mj_axis])
                j_info["dofs_motion_vel"] = np.zeros((1, 3))
                j_info["dofs_limit"] = np.array([mj_limit])
                j_info["dofs_stiffness"] = np.array([mj_stiffness])
                j_info["dofs_sol_params"] = np.array([mj_sol_params])

                j_info["type"] = gs.JOINT_TYPE.REVOLUTE
                j_info["n_qs"] = 1
                j_info["n_dofs"] = 1

            elif mj_type == mujoco.mjtJoint.mjJNT_SLIDE:
                j_info["dofs_motion_ang"] = np.zeros((1, 3))
                j_info["dofs_motion_vel"] = np.array([mj_axis])
                j_info["dofs_limit"] = np.array([mj_limit])
                j_info["dofs_stiffness"] = np.array([mj_stiffness])
                j_info["dofs_sol_params"] = np.array([mj_sol_params])

                j_info["type"] = gs.JOINT_TYPE.PRISMATIC
                j_info["n_qs"] = 1
                j_info["n_dofs"] = 1

            elif mj_type == mujoco.mjtJoint.mjJNT_BALL:
                if np.any(~np.isinf(mj_limit)):
                    gs.logger.warning("joint limit is ignored for ball joints")

                j_info["dofs_motion_ang"] = np.eye(3)
                j_info["dofs_motion_vel"] = np.zeros((3, 3))
                j_info["dofs_limit"] = np.tile([-np.inf, np.inf], (3, 1))
                j_info["dofs_stiffness"] = np.repeat(mj_stiffness[None], 3, axis=0)
                j_info["dofs_sol_params"] = np.repeat(mj_sol_params[None], 3, axis=0)

                j_info["type"] = gs.JOINT_TYPE.SPHERICAL
                j_info["n_qs"] = 3
                j_info["n_dofs"] = 3

            elif mj_type == mujoco.mjtJoint.mjJNT_FREE:
                if mj_stiffness > 0:
                    raise gs.raise_exception("does not support stiffness for free joints")

                j_info["dofs_motion_ang"] = np.eye(6, 3, -3)
                j_info["dofs_motion_vel"] = np.eye(6, 3)
                j_info["dofs_limit"] = np.tile([-np.inf, np.inf], (6, 1))
                j_info["dofs_stiffness"] = np.zeros(6)
                j_info["dofs_sol_params"] = np.zeros((6, 7))

                j_info["type"] = gs.JOINT_TYPE.FREE
                j_info["n_qs"] = 7
                j_info["n_dofs"] = 6

            else:
                gs.raise_exception(f"Unsupported MJCF joint type: {mj_type}")

            j_info_list.append(add_actuator(j_info, i_j))

        j_info = dict()
        j_info["n_qs"] = sum([j["n_qs"] for j in j_info_list])
        j_info["n_dofs"] = sum([j["n_dofs"] for j in j_info_list])
        j_info["dofs_motion_ang"] = np.concatenate([j["dofs_motion_ang"] for j in j_info_list], axis=0)
        j_info["dofs_motion_vel"] = np.concatenate([j["dofs_motion_vel"] for j in j_info_list], axis=0)
        j_info["dofs_limit"] = np.concatenate([j["dofs_limit"] for j in j_info_list], axis=0)
        j_info["dofs_stiffness"] = np.concatenate([j["dofs_stiffness"] for j in j_info_list], axis=0)
        j_info["dofs_sol_params"] = np.concatenate([j["dofs_sol_params"] for j in j_info_list], axis=0)

        j_info["dofs_kp"] = np.concatenate([j["dofs_kp"] for j in j_info_list], axis=0)
        j_info["dofs_kv"] = np.concatenate([j["dofs_kv"] for j in j_info_list], axis=0)
        j_info["dofs_force_range"] = np.concatenate([j["dofs_force_range"] for j in j_info_list], axis=0)

        if j_info["n_dofs"] == 1:
            j_info["type"] = j_info_list[0]["type"]
        elif j_info["n_dofs"] == 2:
            j_info["type"] = gs.JOINT_TYPE.PLANAR
        elif j_info["n_dofs"] == 3:
            j_info["type"] = gs.JOINT_TYPE.SPHERICAL
        elif j_info["n_dofs"] == 6:
            j_info["type"] = gs.JOINT_TYPE.FREE

        j_info["n_qpos0"] = j_info["n_qs"]
        if j_info["type"] == gs.JOINT_TYPE.SPHERICAL and len(j_info_list) == 1:
            # for real ball joint, mujoco uses quaternion for qpos0
            # however, we could merge multiple hinge joints into a single ball joint
            # in this case, we need to use xyz for qpos0
            j_info["n_qpos0"] = 4

        j_info["quat"] = j_info_list[0]["quat"]
        j_info["pos"] = j_info_list[0]["pos"]
        j_info["name"] = j_info_list[0]["name"]

        final_joint_list.append(j_info)

    j_info = add_more_joint_info(final_joint_list[0])

    return l_info, j_info


def parse_geom(mj, i_g, scale, convexify, surface, xml_path):
    mj_geom = mj.geom(i_g)

    is_col = bool(mj_geom.conaffinity or mj_geom.contype)
    geom_size = mj_geom.size
    if is_col:
        gs.logger.warning(
            f"Collision mesh in MJCF is not visualized by default. To visualize "
            + "collision mesh, please use `vis_mode='collision'` when scene.add_entity."
        )

    visual = None
    if mj_geom.type == mujoco.mjtGeom.mjGEOM_PLANE:
        plan_size = 100.0
        r = plan_size / 2.0
        tmesh = trimesh.Trimesh(
            vertices=np.array([[-r, r, 0.0], [r, r, 0.0], [-r, -r, 0.0], [r, -r, 0.0]]),
            faces=np.array([[0, 2, 3], [0, 3, 1]]),
            face_normals=np.array(
                [
                    [0, 0, 1],
                    [0, 0, 1],
                    [0, 0, 1],
                    [0, 0, 1],
                ]
            ),
        )
        gs_type = gs.GEOM_TYPE.PLANE

    elif mj_geom.type == mujoco.mjtGeom.mjGEOM_SPHERE:
        radius = geom_size[0]
        if is_col:
            tmesh = trimesh.creation.icosphere(radius=radius, subdivisions=2)
        else:
            tmesh = trimesh.creation.icosphere(radius=radius)
        gs_type = gs.GEOM_TYPE.SPHERE

    elif mj_geom.type == mujoco.mjtGeom.mjGEOM_CAPSULE:
        radius = geom_size[0]
        halflength = geom_size[1]
        if is_col:
            tmesh = trimesh.creation.capsule(radius=radius, height=halflength * 2, count=(8, 12))
        else:
            tmesh = trimesh.creation.capsule(radius=radius, height=halflength * 2)
        geom_size[1] *= 2
        gs_type = gs.GEOM_TYPE.CAPSULE

    elif mj_geom.type == mujoco.mjtGeom.mjGEOM_CYLINDER:
        radius = geom_size[0]
        halflength = geom_size[1]
        geom_size[1] *= 2
        tmesh = trimesh.creation.cylinder(radius=radius, height=halflength * 2)
        gs_type = gs.GEOM_TYPE.CYLINDER

    elif mj_geom.type == mujoco.mjtGeom.mjGEOM_BOX:
        tmesh = trimesh.creation.box(extents=geom_size * 2)
        geom_size *= 2
        gs_type = gs.GEOM_TYPE.BOX
        if mj_geom.matid >= 0:
            mj_mat = mj.mat(mj_geom.matid)
            tex_id_RGB = mj_mat.texid[mujoco.mjtTextureRole.mjTEXROLE_RGB]
            tex_id_RGBA = mj_mat.texid[mujoco.mjtTextureRole.mjTEXROLE_RGBA]
            tex_id = tex_id_RGB if tex_id_RGB >= 0 else tex_id_RGBA
            if tex_id >= 0:
                mj_tex = mj.tex(tex_id)
                # assert mj_tex.type == mujoco.mjtTexture.mjTEXTURE_2D
                uv_coordinates = tmesh.vertices[:, :2].copy()
                uv_coordinates -= uv_coordinates.min(axis=0)
                uv_coordinates /= uv_coordinates.max(axis=0)
                H, W, C = mj_tex.height[0], mj_tex.width[0], mj_tex.nchannel[0]
                image_array = mj.tex_data[mj_tex.adr[0] : mj_tex.adr[0] + H * W * C].reshape(H, W, C)
                uv_coordinates = uv_coordinates * mj_mat.texrepeat
                visual = TextureVisuals(uv=uv_coordinates, image=Image.fromarray(image_array))
                tmesh.visual = visual

    elif mj_geom.type == mujoco.mjtGeom.mjGEOM_MESH:
        mj_mesh = mj.mesh(mj_geom.dataid)

        vert_start = int(mj_mesh.vertadr)
        vert_num = int(mj_mesh.vertnum)
        vert_end = vert_start + vert_num

        face_start = int(mj_mesh.faceadr)
        face_num = int(mj_mesh.facenum)
        face_end = face_start + face_num

        vertices = mj.mesh_vert[vert_start:vert_end]
        faces = mj.mesh_face[face_start:face_end]
        face_normals = mj.mesh_normal[vert_start:vert_end]
        visual = None

        if mj_geom.matid >= 0:
            mj_mat = mj.mat(mj_geom.matid)
            tex_id_RGB = mj_mat.texid[mujoco.mjtTextureRole.mjTEXROLE_RGB]
            tex_id_RGBA = mj_mat.texid[mujoco.mjtTextureRole.mjTEXROLE_RGBA]
            tex_id = tex_id_RGB if tex_id_RGB >= 0 else tex_id_RGBA
            if tex_id >= 0:
                mj_tex = mj.tex(tex_id)
                tex_vert_start = int(mj.mesh_texcoordadr[mj_mesh.id])
                num_tex_vert = int(mj.mesh_texcoordnum[mj_mesh.id])
                if tex_vert_start != -1:  # -1 means no texcoord
                    vertices = np.zeros((num_tex_vert, 3))
                    faces = mj.mesh_facetexcoord[face_start:face_end]
                    for face_id in range(face_start, face_end):
                        for i in range(3):
                            mesh_vert_id = mj.mesh_face[face_id, i]
                            tex_vert_id = mj.mesh_facetexcoord[face_id, i]
                            vertices[tex_vert_id] = mj.mesh_vert[mesh_vert_id + vert_start]

                    uv = mj.mesh_texcoord[tex_vert_start : tex_vert_start + num_tex_vert]
                    uv[:, 1] = 1 - uv[:, 1]

                    H, W, C = mj_tex.height[0], mj_tex.width[0], mj_tex.nchannel[0]
                    image_array = mj.tex_data[mj_tex.adr[0] : mj_tex.adr[0] + H * W * C].reshape(H, W, C)
                    uv = uv * mj_mat.texrepeat
                    visual = TextureVisuals(uv=uv, image=Image.fromarray(image_array))

        tmesh = trimesh.Trimesh(
            vertices=vertices,
            faces=faces,
            face_normals=face_normals,
            process=False,
            visual=visual,
        )
        gs_type = gs.GEOM_TYPE.MESH

    else:
        gs.logger.warning(f"Unsupported MJCF geom type: {mj_geom.type}")
        return None

    mesh = gs.Mesh.from_trimesh(
        tmesh,
        scale=scale,
        convexify=is_col and convexify,
        surface=gs.surfaces.Collision() if is_col else surface,
    )

    if surface.diffuse_texture is None and visual is None:  # user input will override mjcf color
        if mj_geom.matid >= 0:
            mesh.set_color(mj.mat(mj_geom.matid).rgba)
        else:
            mesh.set_color(mj_geom.rgba)

    info = {
        "type": gs_type,
        "pos": mj_geom.pos * scale,
        "quat": mj_geom.quat,
        "mesh": mesh,
        "is_col": is_col,
        "is_convex": True,
        "data": geom_size,
        "friction": mj_geom.friction[0],
        "sol_params": np.concatenate((mj_geom.solref, mj_geom.solimp)),
    }

    return info


def parse_equality(mj, i_e, scale, ordered_links_idx):
    e_info = dict()
    mj_equality = mj.equality(i_e)
    e_info["name"] = mj_equality.name

    if mj.eq_type[i_e] == mujoco.mjtEq.mjEQ_CONNECT:
        e_info["type"] = gs.EQUALITY_TYPE.CONNECT
        e_info["link1_idx"] = -1 if mj.eq_obj1id[i_e] == 0 else ordered_links_idx.index(mj.eq_obj1id[i_e] - 1)
        e_info["link2_idx"] = -1 if mj.eq_obj2id[i_e] == 0 else ordered_links_idx.index(mj.eq_obj2id[i_e] - 1)
        e_info["anchor1_pos"] = mj.eq_data[i_e][0:3] * scale
        e_info["anchor2_pos"] = mj.eq_data[i_e][3:6] * scale
        e_info["rel_pose"] = mj.eq_data[i_e][6:10]
        e_info["torque_scale"] = mj.eq_data[i_e][10]
        e_info["sol_params"] = np.concatenate((mj.eq_solref[i_e], mj.eq_solimp[i_e]))

    elif mj.eq_type[i_e] == mujoco.mjtEq.mjEQ_WELD:
        e_info["type"] = gs.EQUALITY_TYPE.WELD
    elif mj.eq_type[i_e] == mujoco.mjtEq.mjEQ_JOINT:
        e_info["type"] = gs.EQUALITY_TYPE.JOINT
    else:
        raise gs.raise_exception(f"Unsupported MJCF equality type: {mj.eq_type[i_e]}")

    return e_info
