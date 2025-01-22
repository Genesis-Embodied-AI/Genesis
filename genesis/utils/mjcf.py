import os
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from PIL import Image

import genesis as gs
from genesis.ext import trimesh
from genesis.ext.trimesh.visual.texture import TextureVisuals

from . import geom as gu
from . import mesh as mu
from .misc import get_assets_dir


def extract_compiler_attributes(xml_path):
    # Parse the XML file
    tree = ET.parse(xml_path)
    root = tree.getroot()
    # Find the <compiler> tag
    compiler_tag = root.find(".//compiler")
    # Check if the tag exists and has the desired attributes
    res = {"meshdir": "", "texturedir": ""}
    if compiler_tag is not None:
        meshdir = compiler_tag.get("meshdir")
        texturedir = compiler_tag.get("texturedir")
        if meshdir:
            res["meshdir"] = meshdir
        if texturedir:
            res["texturedir"] = texturedir
    return res


def parse_mjcf(path):
    path = os.path.join(get_assets_dir(), path)
    mj = mujoco.MjModel.from_xml_path(path)

    # for trntype in mj.actuator_trntype:
    #     if trntype != mujoco.mjtTrn.mjTRN_JOINT:
    #         gs.logger.warning(f'Unsupported MJCF actuator type: {mujoco.mjtTrn(trntype)}')

    return mj


def parse_link(mj, i_l, q_offset, dof_offset, scale):
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
        q_off = q_offset + jnt_offset

        j_info["dofs_damping"] = np.array(mj.dof_damping[d_off : d_off + j_info["n_dofs"]])
        j_info["dofs_invweight"] = np.array(mj.dof_invweight0[d_off : d_off + j_info["n_dofs"]])
        j_info["dofs_armature"] = np.array(mj.dof_armature[d_off : d_off + j_info["n_dofs"]])
        j_info["init_qpos"] = np.array(mj.qpos0[q_off : q_off + j_info["n_qs"]])

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

        j_info["quat"] = j_info_list[0]["quat"]
        j_info["pos"] = j_info_list[0]["pos"]
        j_info["name"] = j_info_list[0]["name"]

        final_joint_list.append(j_info)

    j_info = add_more_joint_info(final_joint_list[0])

    return l_info, j_info


def parse_geom(mj, i_g, scale, convexify, surface, xml_path):
    mj_type = mj.geom_type[i_g]
    is_col = bool(mj.geom_conaffinity[i_g] or mj.geom_contype[i_g])
    dim_data = 7
    data = np.zeros([dim_data])
    for gi in range(min(dim_data, len(mj.geom_size[i_g, :]))):
        data[gi] = mj.geom_size[i_g, gi]

    is_convex = True
    visual = None
    if mj_type == mujoco.mjtGeom.mjGEOM_PLANE:
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

    elif mj_type == mujoco.mjtGeom.mjGEOM_SPHERE:
        if is_col:
            tmesh = trimesh.creation.icosphere(radius=mj.geom_size[i_g, 0], subdivisions=2)
        else:
            tmesh = trimesh.creation.icosphere(radius=mj.geom_size[i_g, 0])
        gs_type = gs.GEOM_TYPE.SPHERE

    elif mj_type == mujoco.mjtGeom.mjGEOM_CAPSULE:
        radius = mj.geom_size[i_g, 0]
        halflength = mj.geom_size[i_g, 1]
        if is_col:
            tmesh = trimesh.creation.capsule(radius=radius, height=halflength * 2, count=(8, 12))
        else:
            tmesh = trimesh.creation.capsule(radius=radius, height=halflength * 2)
        data[1] *= 2
        gs_type = gs.GEOM_TYPE.CAPSULE

    elif mj_type == mujoco.mjtGeom.mjGEOM_CYLINDER:
        radius = mj.geom_size[i_g, 0]
        halflength = mj.geom_size[i_g, 1]
        data[1] *= 2
        tmesh = trimesh.creation.cylinder(radius=radius, height=halflength * 2)
        gs_type = gs.GEOM_TYPE.CYLINDER

    elif mj_type == mujoco.mjtGeom.mjGEOM_BOX:
        tmesh = trimesh.creation.box(extents=mj.geom_size[i_g, :3] * 2)
        data *= 2
        gs_type = gs.GEOM_TYPE.BOX

        # TODO: not sure if it is the right way to load texture for box
        mat_id = mj.geom_matid[i_g]
        if mat_id >= 0:
            mat_id = mj.geom_matid[i_g]
            tex_id = next((x for x in mj.mat_texid[mat_id] if x != -1), None)

            if tex_id is not None:
                tex_path = mj.paths[mj.tex_pathadr[tex_id] :].decode("utf-8").split("\x00")[0]
                texturedir = extract_compiler_attributes(xml_path)["texturedir"]
                assets_dir = os.path.join(get_assets_dir(), os.path.join(os.path.dirname(xml_path), texturedir))

                uv_coordinates = tmesh.vertices[:, :2].copy()
                uv_coordinates -= uv_coordinates.min(axis=0)
                uv_coordinates /= uv_coordinates.max(axis=0)
                image = Image.open(os.path.join(assets_dir, tex_path)).convert("RGBA")
                image_array = np.array(image)
                tex_repeat = mj.mat_texrepeat[mat_id].astype(int)
                image_array = np.tile(image_array, (tex_repeat[0], tex_repeat[1], 1))
                visual = TextureVisuals(uv=uv_coordinates, image=Image.fromarray(image_array, mode="RGBA"))
                tmesh.visual = visual

    elif mj_type == mujoco.mjtGeom.mjGEOM_MESH:
        i = mj.geom_dataid[i_g]
        last = (i + 1) >= mj.nmesh

        vert_start = mj.mesh_vertadr[i]
        vert_end = mj.mesh_vertadr[i + 1] if not last else mj.mesh_vert.shape[0]
        face_start = mj.mesh_faceadr[i]
        face_end = mj.mesh_faceadr[i + 1] if not last else mj.mesh_face.shape[0]
        tex_start = mj.mesh_texcoordadr[i]

        if tex_start >= 0:
            tex_end = mj.mesh_texcoordadr[i + 1] if not last else mj.mesh_texcoord.shape[0]
            if tex_end == -1:
                tex_end = tex_start + (vert_end - vert_start)
            assert tex_end - tex_start == vert_end - vert_start

            mat_id = mj.geom_matid[i_g]
            tex_id = next((x for x in mj.mat_texid[mat_id] if x != -1), None)
            if not tex_id is None:
                tex_path = mj.paths[mj.tex_pathadr[tex_id] :].decode("utf-8").split("\x00")[0]
                uv = mj.mesh_texcoord[tex_start:tex_end]
                uv[:, 1] = 1 - uv[:, 1]

                # TODO: check if we can parse <compiler> tag with mj model
                texturedir = extract_compiler_attributes(xml_path)["texturedir"]
                assets_dir = os.path.join(get_assets_dir(), os.path.join(os.path.dirname(xml_path), texturedir))

                image = Image.open(os.path.join(assets_dir, tex_path)).convert("RGBA")
                image_array = np.array(image)
                tex_repeat = mj.mat_texrepeat[mat_id].astype(int)
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

    else:
        gs.logger.warning(f"Unsupported MJCF geom type: {mj_type}")
        return None

    mesh = gs.Mesh.from_trimesh(
        tmesh,
        scale=scale,
        convexify=is_col and convexify,
        surface=gs.surfaces.Collision() if is_col else surface,
    )

    if surface.diffuse_texture is None and visual is None:  # user input will override mjcf color
        if mj.geom_matid[i_g] >= 0:
            mesh.set_color(mj.mat_rgba[mj.geom_matid[i_g]])
        else:
            mesh.set_color(mj.geom_rgba[i_g])

    pos = mj.geom_pos[i_g]
    quat = mj.geom_quat[i_g]

    # apply scale
    pos = np.copy(pos) * scale

    # other params
    friction = mj.geom_friction[i_g, 0]
    sol_params = np.concatenate((mj.geom_solref[i_g], mj.geom_solimp[i_g]))

    info = {
        "type": gs_type,
        "pos": pos,
        "quat": quat,
        "mesh": mesh,
        "is_col": is_col,
        "is_convex": is_convex,
        "data": data,
        "friction": friction,
        "sol_params": sol_params,
    }

    return info
