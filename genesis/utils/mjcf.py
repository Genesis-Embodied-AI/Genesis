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


def parse_link(mj, i_l, q_offset, dof_offset, scale):
    mj_body = mj.body(i_l)
    l_info = dict()
    l_info["name"] = mj_body.name

    l_info["pos"] = mj_body.pos
    l_info["quat"] = mj_body.quat
    l_info["inertial_pos"] = mj_body.ipos
    l_info["inertial_quat"] = mj_body.iquat
    l_info["inertial_i"] = np.diag(mj_body.inertia)
    l_info["inertial_mass"] = mj_body.mass[0]
    l_info["parent_idx"] = mj_body.parentid[0] - 1
    l_info["invweight"] = mj_body.invweight0[0]

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
            for i_a in range(mj.nu):
                mj_actuator = mj.actuator(i_a)
                if mj_actuator.trnid[0] == i_j and mj_actuator.trntype[0] == mujoco.mjtTrn.mjTRN_JOINT:
                    if mj_actuator.gainprm[0] != -mj_actuator.biasprm[1]:
                        gs.logger.warning("`kp` in `gainprm` doesn't match `-kp` in `biasprm`.")
                    j_info["dofs_kp"] = np.tile(mj_actuator.gainprm[0], j_info["n_dofs"])
                    j_info["dofs_kv"] = np.tile(-mj_actuator.biasprm[2], j_info["n_dofs"])
                    j_info["dofs_force_range"] = np.tile(mj_actuator.forcerange, (j_info["n_dofs"], 1))
                    break

        return j_info

    def add_more_joint_info(j_info, jnt_offset=0):
        d_off = dof_offset + jnt_offset
        q_off = q_offset + jnt_offset

        j_info["dofs_damping"] = np.array(mj.dof_damping[d_off : d_off + j_info["n_dofs"]])
        j_info["dofs_invweight"] = np.array(mj.dof_invweight0[d_off : d_off + j_info["n_dofs"]])
        j_info["dofs_armature"] = np.array(mj.dof_armature[d_off : d_off + j_info["n_dofs"]])
        j_info["init_qpos"] = np.array(mj.qpos0[q_off : q_off + j_info["n_qs"]])

        # apply scale
        j_info["pos"] *= scale
        return j_info

    jnt_adr = mj_body.jntadr[0]
    jnt_num = mj_body.jntnum[0]

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
        j_info = add_more_joint_info(add_actuator(j_info))
        final_joint_list.append(j_info)
    else:
        j_info_list = []
        for i_j in range(jnt_adr, jnt_adr + jnt_num):
            mj_joint = mj.joint(i_j)
            j_info = dict()
            j_info["name"] = mj_joint.name if len(mj_joint.name) > 0 else f'{l_info["name"]}_joint'
            j_info["pos"] = mj_joint.pos
            j_info["quat"] = np.array([1.0, 0.0, 0.0, 0.0])

            mj_limit = mj_joint.range if mj_joint.limited[0] == 1 else np.array([-np.inf, np.inf])
            mj_sol_params = np.concatenate((mj_joint.solref, mj_joint.solimp), axis=1)[0]  # only get the first element]

            if mj_joint.type == mujoco.mjtJoint.mjJNT_HINGE:
                j_info["dofs_motion_ang"] = mj_joint.axis[None]
                j_info["dofs_motion_vel"] = np.zeros((1, 3))
                j_info["dofs_limit"] = np.array([mj_limit])
                j_info["dofs_stiffness"] = mj_joint.stiffness
                j_info["dofs_sol_params"] = np.array([mj_sol_params])

                j_info["type"] = gs.JOINT_TYPE.REVOLUTE
                j_info["n_qs"] = 1
                j_info["n_dofs"] = 1

            elif mj_joint.type == mujoco.mjtJoint.mjJNT_SLIDE:
                j_info["dofs_motion_ang"] = np.zeros((1, 3))
                j_info["dofs_motion_vel"] = mj_joint.axis[None]
                j_info["dofs_limit"] = np.array([mj_limit])
                j_info["dofs_stiffness"] = mj_joint.stiffness
                j_info["dofs_sol_params"] = np.array([mj_sol_params])

                j_info["type"] = gs.JOINT_TYPE.PRISMATIC
                j_info["n_qs"] = 1
                j_info["n_dofs"] = 1

            elif mj_joint.type == mujoco.mjtJoint.mjJNT_BALL:
                if np.any(~np.isinf(mj_limit)):
                    gs.logger.warning("joint limit is ignored for ball joints")

                j_info["dofs_motion_ang"] = np.eye(3)
                j_info["dofs_motion_vel"] = np.zeros((3, 3))
                j_info["dofs_limit"] = np.tile([-np.inf, np.inf], (3, 1))
                j_info["dofs_stiffness"] = np.repeat(mj_joint.stiffness, 3, axis=0)
                j_info["dofs_sol_params"] = np.repeat(mj_sol_params[None], 3, axis=0)

                j_info["type"] = gs.JOINT_TYPE.SPHERICAL
                j_info["n_qs"] = 3
                j_info["n_dofs"] = 3

            elif mj_joint.type == mujoco.mjtJoint.mjJNT_FREE:
                if mj_joint.stiffness[0] > 0:
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
                gs.raise_exception(f"Unsupported MJCF joint type: {mj_joint.type}")

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
                assert mj_tex.type == mujoco.mjtTexture.mjTEXTURE_2D
                uv_coordinates = tmesh.vertices[:, :2].copy()
                uv_coordinates -= uv_coordinates.min(axis=0)
                uv_coordinates /= uv_coordinates.max(axis=0)
                image = Image.open(os.path.join(assets_dir, tex_path)).convert("RGBA")
                image_array = np.array(image)
                tex_repeat = np.ceil(mj.mat_texrepeat[mat_id]).astype(int)
                image_array = np.tile(image_array, (tex_repeat[0], tex_repeat[1], 1))
                visual = TextureVisuals(uv=uv_coordinates, image=Image.fromarray(image_array, mode="RGBA"))
                tmesh.visual = visual

    elif mj_geom.type == mujoco.mjtGeom.mjGEOM_MESH:
        mj_mesh = mj.mesh(mj_geom.dataid)

        vert_start = int(mj_mesh.vertadr)
        vert_num = int(mj_mesh.vertnum)
        vert_end = vert_start + vert_num

        face_start = int(mj_mesh.faceadr)
        face_num = int(mj_mesh.facenum)
        face_end = face_start + face_num

        if mj_geom.matid >= 0:
            mj_mat = mj.mat(mj_geom.matid)
            tex_vert_start = int(mj_mesh.texcoordadr)
            tex_id_RGB = mj_mat.texid[mujoco.mjtTextureRole.mjTEXROLE_RGB]
            tex_id_RGBA = mj_mat.texid[mujoco.mjtTextureRole.mjTEXROLE_RGBA]
            tex_id = tex_id_RGB if tex_id_RGB >= 0 else tex_id_RGBA
            if tex_id >= 0:
                mj_tex = mj.tex(tex_id)

                # remap texture coordinates
                uv = np.zeros((vert_num, 2))
                for face_id in range(face_start, face_end):
                    for i in range(3):
                        tex_face_id = mj.mesh_facetexcoord[face_id, i] + tex_vert_start
                        uv[mj.mesh_face[face_id, i]] = mj.mesh_texcoord[
                            tex_face_id
                        ]  # this may overwrite the same vertex
                uv[:, 1] = 1 - uv[:, 1]

                # TODO: check if we can parse <compiler> tag with mj model
                texturedir = extract_compiler_attributes(xml_path)["texturedir"]
                assets_dir = os.path.join(get_assets_dir(), os.path.join(os.path.dirname(xml_path), texturedir))

                image = Image.open(os.path.join(assets_dir, tex_path)).convert("RGBA")
                image_array = np.array(image)
                tex_repeat = np.ceil(mj.mat_texrepeat[mat_id]).astype(int)
                image_array = np.tile(image_array, (tex_repeat[0], tex_repeat[1], 1))
                visual = TextureVisuals(uv=uv, image=Image.fromarray(image_array, mode="RGBA"))

        tmesh = trimesh.Trimesh(
            vertices=mj.mesh_vert[vert_start:vert_end],
            faces=mj.mesh_face[face_start:face_end],
            face_normals=mj.mesh_normal[vert_start:vert_end],
            process=False,
            visual=visual,
        )
        gs_type = gs.GEOM_TYPE.MESH
        # import ipdb; ipdb.set_trace()

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

    if mj_equality.type == mujoco.mjtEq.mjEQ_CONNECT:
        e_info["type"] = gs.EQUALITY_TYPE.CONNECT
        e_info["link1_idx"] = -1 if mj_equality.obj1id[0] == 0 else ordered_links_idx[mj_equality.obj1id[0] - 1]
        e_info["link2_idx"] = -1 if mj_equality.obj2id[0] == 0 else ordered_links_idx[mj_equality.obj2id[0] - 1]
        e_info["anchor1_pos"] = mj_equality.data[0:3] * scale
        e_info["anchor2_pos"] = mj_equality.data[3:6] * scale
        e_info["rel_pose"] = mj_equality.data[6:10]
        e_info["torque_scale"] = mj_equality.data[10]
        e_info["sol_params"] = np.concatenate((mj_equality.solref, mj_equality.solimp))

    elif mj_equality.type == mujoco.mjtEq.mjEQ_WELD:
        e_info["type"] = gs.EQUALITY_TYPE.WELD
    elif mj_equality.type == mujoco.mjtEq.mjEQ_JOINT:
        e_info["type"] = gs.EQUALITY_TYPE.JOINT
    else:
        raise gs.raise_exception(f"Unsupported MJCF equality type: {mj_equality.type}")

    return e_info
