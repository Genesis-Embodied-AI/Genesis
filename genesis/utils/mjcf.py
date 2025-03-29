import os
from bisect import bisect_right

import numpy as np
import trimesh
from trimesh.visual.texture import TextureVisuals
from PIL import Image

import mujoco
import genesis as gs

from . import geom as gu
from .misc import get_assets_dir


def parse_mjcf(path):
    path = os.path.join(get_assets_dir(), path)
    mj = mujoco.MjModel.from_xml_path(path)
    return mj


def parse_link(mj, i_l, scale):
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

    jnt_adr = mj.body_jntadr[i_l]
    jnt_num = mj.body_jntnum[i_l]

    j_infos = []
    for i_j in range(jnt_adr, jnt_adr + max(jnt_num, 1)):
        j_info = dict()

        # Parsing joint parameters that are type-specific
        if i_j == -1:
            j_info["dofs_motion_ang"] = np.zeros((0, 3))
            j_info["dofs_motion_vel"] = np.zeros((0, 3))
            j_info["dofs_limit"] = np.zeros((0, 2))
            j_info["dofs_stiffness"] = np.zeros((0))

            j_info["name"] = l_info["name"]
            j_info["type"] = gs.JOINT_TYPE.FIXED
            j_info["pos"] = np.array([0.0, 0.0, 0.0])
            j_info["n_qs"] = 0
            j_info["n_dofs"] = 0
        else:
            name_start = mj.name_jntadr[i_j]
            if i_j + 1 < mj.njnt:
                name_end = mj.name_jntadr[i_j + 1]
            else:
                name_end = mj.name_geomadr[0]
            j_info["name"] = mj.names[name_start:name_end].decode("utf-8").replace("\x00", "")

            j_info["pos"] = mj.jnt_pos[i_j]

            mj_type = mj.jnt_type[i_j]
            mj_stiffness = mj.jnt_stiffness[i_j]
            mj_limit = mj.jnt_range[i_j] if mj.jnt_limited[i_j] == 1 else np.array([-np.inf, np.inf])
            mj_axis = mj.jnt_axis[i_j]

            if mj_type == mujoco.mjtJoint.mjJNT_HINGE:
                j_info["dofs_motion_ang"] = np.array([mj_axis])
                j_info["dofs_motion_vel"] = np.zeros((1, 3))
                j_info["dofs_limit"] = np.array([mj_limit])
                j_info["dofs_stiffness"] = np.array([mj_stiffness])

                j_info["type"] = gs.JOINT_TYPE.REVOLUTE
                j_info["n_qs"] = 1
                j_info["n_dofs"] = 1

            elif mj_type == mujoco.mjtJoint.mjJNT_SLIDE:
                j_info["dofs_motion_ang"] = np.zeros((1, 3))
                j_info["dofs_motion_vel"] = np.array([mj_axis])
                j_info["dofs_limit"] = np.array([mj_limit])
                j_info["dofs_stiffness"] = np.array([mj_stiffness])

                j_info["type"] = gs.JOINT_TYPE.PRISMATIC
                j_info["n_qs"] = 1
                j_info["n_dofs"] = 1

            elif mj_type == mujoco.mjtJoint.mjJNT_BALL:
                if not np.all(np.isinf(mj_limit)):
                    gs.logger.warning("(MJCF) Joint limit ignored for ball joints")

                j_info["dofs_motion_ang"] = np.eye(3)
                j_info["dofs_motion_vel"] = np.zeros((3, 3))
                j_info["dofs_limit"] = np.tile([-np.inf, np.inf], (3, 1))
                j_info["dofs_stiffness"] = np.repeat(mj_stiffness[None], 3, axis=0)

                j_info["type"] = gs.JOINT_TYPE.SPHERICAL
                j_info["n_qs"] = 4
                j_info["n_dofs"] = 3

            elif mj_type == mujoco.mjtJoint.mjJNT_FREE:
                if mj_stiffness > 0:
                    raise gs.raise_exception("(MJCF) Joint stiffness not supported for free joints")

                j_info["dofs_motion_ang"] = np.eye(6, 3, -3)
                j_info["dofs_motion_vel"] = np.eye(6, 3)
                j_info["dofs_limit"] = np.tile([-np.inf, np.inf], (6, 1))
                j_info["dofs_stiffness"] = np.zeros(6)

                j_info["type"] = gs.JOINT_TYPE.FREE
                j_info["n_qs"] = 7
                j_info["n_dofs"] = 6

            else:
                gs.raise_exception(f"Unsupported MJCF joint type: {mj_type}")

        # Parsing joint parameters that are type-agnostic
        mj_jnt_offset = i_j if i_j != -1 else 0
        mj_dof_offset = mj.jnt_dofadr[i_j] if i_j != -1 else 0
        mj_qpos_offset = mj.jnt_qposadr[i_j] if i_j != -1 else 0
        n_dofs = j_info["n_dofs"]
        j_info["quat"] = np.array([1.0, 0.0, 0.0, 0.0])
        j_info["init_qpos"] = np.array(mj.qpos0[mj_qpos_offset : (mj_qpos_offset + j_info["n_qs"])])
        j_info["dofs_damping"] = mj.dof_damping[mj_dof_offset : (mj_dof_offset + n_dofs)]
        j_info["dofs_invweight"] = mj.dof_invweight0[mj_dof_offset : (mj_dof_offset + n_dofs)]
        j_info["dofs_armature"] = mj.dof_armature[mj_dof_offset : (mj_dof_offset + n_dofs)]
        j_info["sol_params"] = np.concatenate(
            (
                mj.jnt_solref[mj_jnt_offset : (mj_jnt_offset + 1)],
                mj.jnt_solimp[mj_jnt_offset : (mj_jnt_offset + 1)],
            ),
            axis=1,
        )
        j_info["dofs_sol_params"] = np.concatenate(
            (
                mj.dof_solref[mj_dof_offset : (mj_dof_offset + n_dofs)],
                mj.dof_solimp[mj_dof_offset : (mj_dof_offset + n_dofs)],
            ),
            axis=1,
        )

        if (mj.dof_frictionloss[mj_dof_offset : (mj_dof_offset + n_dofs)] > 0.0).any():
            gs.logger.warning("(MJCF) Joint Coulomb friction not supported.")

        # Parsing actuator parameters
        j_info["dofs_kp"] = np.zeros((n_dofs,), dtype=gs.np_float)
        j_info["dofs_kv"] = np.zeros((n_dofs,), dtype=gs.np_float)
        j_info["dofs_force_range"] = np.zeros((n_dofs, 2), dtype=gs.np_float)

        i_a = -1
        try:
            actuator_mask_j = (mj.actuator_trnid[:, 0] == i_j) & (mj.actuator_trntype == mujoco.mjtTrn.mjTRN_JOINT)
            if actuator_mask_j.any():
                (i_a,) = np.nonzero(actuator_mask_j)[0]
            else:  # No actuator directly attached to the joint via mechanical transmission
                # Special case where all tendon are attached to joint. Very common in practice.
                if (mj.wrap_type == mujoco.mjtWrap.mjWRAP_JOINT).all():
                    if i_j in mj.wrap_objid:
                        (m,) = np.nonzero(mj.wrap_objid == i_j)[0]
                        i_t = bisect_right(np.cumsum(mj.tendon_num), m)
                        actuator_mask_t = (mj.actuator_trnid[:, 0] == i_t) & (
                            mj.actuator_trntype == mujoco.mjtTrn.mjTRN_TENDON
                        )
                        (i_a,) = np.nonzero(actuator_mask_t)[0]
                        gs.logger.warning(f"(MJCF) Approximating tendon by joint actuator for `{j_info['name']}`")
        except ValueError:
            gs.logger.warning(f"(MJCF) Failed to parse actuator for joint `{j_info['name']}`.")

        if i_a >= 0:
            if mj.actuator_dyntype[i_a] != mujoco.mjtDyn.mjDYN_NONE:
                gs.logger.warning(f"(MJCF) Actuator internal dynamics not supported")
            gaintype = mujoco.mjtGain(mj.actuator_gaintype[i_a])
            if gaintype != mujoco.mjtGain.mjGAIN_FIXED:
                gs.logger.warning(f"(MJCF) Actuator control gain of type '{gaintype}' not supported")
            biastype = mujoco.mjtBias(mj.actuator_biastype[i_a])
            if biastype not in (mujoco.mjtBias.mjBIAS_NONE, mujoco.mjtBias.mjBIAS_AFFINE):
                gs.logger.warning(f"(MJCF) Actuator control bias of type '{biastype}' not supported")
            if n_dofs > 1 and not (mj.actuator_gear[i_a, :n_dofs] == 1.0).all():
                gs.logger.warning("(MJCF) Actuator transmission gear is only supported of 1DoF joints")

            if biastype == mujoco.mjtBias.mjBIAS_NONE:
                # Direct-drive
                actuator_kp = 0.0
                actuator_kv = 0.0
            else:  # U = gain_term * ctrl + bias_term
                # PD control
                gainprm = mj.actuator_gainprm[i_a]
                biasprm = mj.actuator_biasprm[i_a]
                if gainprm[1:].any() or biasprm[0]:
                    gs.logger.warning(
                        "(MJCF) Actuator control gain and bias parameters not supported. Using default values."
                    )
                    actuator_kp = gu.default_dofs_kp(1)[0]
                    actuator_kv = gu.default_dofs_kv(1)[0]
                elif gainprm[0] != -biasprm[1]:
                    # Doing our best to approximate the expected behavior: g0 * p_target + b1 * p_mes + b2 * v_mes
                    gs.logger.warning(
                        "(MJCF) Actuator control gain and bias parameters cannot be reduced to a unique PD control "
                        "position gain. Using max between gain and bias."
                    )
                    actuator_kp = min(-gainprm[0], biasprm[1])
                    actuator_kv = biasprm[2]
                else:
                    actuator_kp, actuator_kv = biasprm[1], biasprm[2]

            gear = mj.actuator_gear[i_a, 0]
            j_info["dofs_kp"] = np.tile(-gear * actuator_kp, (n_dofs,))
            j_info["dofs_kv"] = np.tile(-gear * actuator_kv, (n_dofs,))
            if mj.actuator_forcelimited[i_a]:
                j_info["dofs_force_range"] = np.tile(mj.actuator_forcerange[i_a], (n_dofs, 1))
            if mj.actuator_ctrllimited[i_a] and biastype == mujoco.mjtBias.mjBIAS_NONE:
                j_info["dofs_force_range"] = np.minimum(
                    j_info["dofs_force_range"], np.tile(gear * mj.actuator_ctrlrange[i_a], (n_dofs, 1))
                )
        else:
            gs.logger.debug(f"(MJCF) No actuator found for joint `{j_info['name']}`")

        j_infos.append(j_info)

    # Applying scale
    l_info["pos"] *= scale
    l_info["inertial_pos"] *= scale
    l_info["inertial_mass"] *= scale**3
    l_info["inertial_i"] *= scale**5
    l_info["invweight"] /= scale**3
    for j_info in j_infos:
        j_info["pos"] *= scale
    # exclude joints with 0 dofs in MJCF models to align with mujoco
    j_infos = [j_info for j_info in j_infos if j_info["n_dofs"] > 0]

    return l_info, j_infos


def parse_links(mj, scale):
    l_infos = []
    j_infos = []

    for i_l in range(mj.nbody):
        l_info, j_info = parse_link(mj, i_l, scale)

        l_infos.append(l_info)
        j_infos.append(j_info)

    return l_infos, j_infos


def parse_geom(mj, i_g, scale, surface, xml_path):
    mj_geom = mj.geom(i_g)

    geom_size = mj_geom.size
    is_col = mj_geom.contype or mj_geom.conaffinity

    visual = None
    if mj_geom.type == mujoco.mjtGeom.mjGEOM_PLANE:
        length, width, _ = geom_size
        length = length or 1e3
        width = width or 1e3

        tmesh = trimesh.Trimesh(
            vertices=np.array(
                [[-length, width, 0.0], [length, width, 0.0], [-length, -width, 0.0], [length, -width, 0.0]]
            ),
            faces=np.array([[0, 2, 3], [0, 3, 1]]),
            face_normals=np.array(
                [
                    [0, 0, 1],
                    [0, 0, 1],
                ]
            ),
        )
        geom_data = np.array([0.0, 0.0, 1.0])
        gs_type = gs.GEOM_TYPE.PLANE

    elif mj_geom.type == mujoco.mjtGeom.mjGEOM_SPHERE:
        radius = geom_size[0]
        if is_col:
            tmesh = trimesh.creation.icosphere(radius=radius, subdivisions=2)
        else:
            tmesh = trimesh.creation.icosphere(radius=radius)
        gs_type = gs.GEOM_TYPE.SPHERE
        geom_data = np.array([radius])

    elif mj_geom.type == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
        if is_col:
            tmesh = trimesh.creation.icosphere(radius=1.0, subdivisions=2)
        else:
            tmesh = trimesh.creation.icosphere(radius=1.0)
        tmesh.apply_transform(np.diag([*geom_size, 1]))
        gs_type = gs.GEOM_TYPE.ELLIPSOID
        geom_data = geom_size

    elif mj_geom.type == mujoco.mjtGeom.mjGEOM_CAPSULE:
        radius = geom_size[0]
        height = geom_size[1] * 2
        if is_col:
            tmesh = trimesh.creation.capsule(radius=radius, height=height, count=(8, 12))
        else:
            tmesh = trimesh.creation.capsule(radius=radius, height=height)
        gs_type = gs.GEOM_TYPE.CAPSULE
        geom_data = np.array([radius, height])

    elif mj_geom.type == mujoco.mjtGeom.mjGEOM_CYLINDER:
        radius = geom_size[0]
        height = geom_size[1] * 2
        tmesh = trimesh.creation.cylinder(radius=radius, height=height)
        gs_type = gs.GEOM_TYPE.CYLINDER
        geom_data = np.array([radius, height])

    elif mj_geom.type == mujoco.mjtGeom.mjGEOM_BOX:
        tmesh = trimesh.creation.box(extents=geom_size * 2)
        gs_type = gs.GEOM_TYPE.BOX
        if mj_geom.matid >= 0:
            mj_mat = mj.mat(mj_geom.matid[0])
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
                image_array = mj.tex_data[mj_tex.adr[0] : (mj_tex.adr[0] + H * W * C)].reshape(H, W, C)
                uv_coordinates = uv_coordinates * mj_mat.texrepeat
                visual = TextureVisuals(uv=uv_coordinates, image=Image.fromarray(image_array))
                tmesh.visual = visual
        geom_data = 2 * geom_size

    elif mj_geom.type == mujoco.mjtGeom.mjGEOM_MESH:
        mj_mesh = mj.mesh(mj_geom.dataid[0])

        vert_start = mj_mesh.vertadr[0]
        vert_num = mj_mesh.vertnum[0]
        vert_end = vert_start + vert_num

        face_start = mj_mesh.faceadr[0]
        face_num = mj_mesh.facenum[0]
        face_end = face_start + face_num

        vertices = mj.mesh_vert[vert_start:vert_end]
        faces = mj.mesh_face[face_start:face_end]
        face_normals = mj.mesh_normal[vert_start:vert_end]
        visual = None

        if mj_geom.matid >= 0:
            mj_mat = mj.mat(mj_geom.matid[0])
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

                    uv = mj.mesh_texcoord[tex_vert_start : (tex_vert_start + num_tex_vert)]
                    uv[:, 1] = 1 - uv[:, 1]

                    H, W, C = mj_tex.height[0], mj_tex.width[0], mj_tex.nchannel[0]
                    image_array = mj.tex_data[mj_tex.adr[0] : (mj_tex.adr[0] + H * W * C)].reshape(H, W, C)
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
        geom_data = None

    else:
        gs.logger.warning(f"Unsupported MJCF geom type '{mj_geom.type}'.")
        return None

    mesh = gs.Mesh.from_trimesh(
        tmesh,
        scale=scale,
        surface=gs.surfaces.Collision() if is_col else surface,
    )

    if surface.diffuse_texture is None and visual is None:  # user input will override mjcf color
        if mj_geom.matid >= 0:
            mesh.set_color(mj.mat(mj_geom.matid[0]).rgba)
        else:
            mesh.set_color(mj_geom.rgba)

    info = {
        "type": gs_type,
        "pos": mj_geom.pos * scale,
        "quat": mj_geom.quat,
        "contype": mj_geom.contype[0],
        "conaffinity": mj_geom.conaffinity[0],
        "group": mj_geom.group[0],
        "data": geom_data,
        "friction": mj_geom.friction[0],
        "sol_params": np.concatenate((mj_geom.solref, mj_geom.solimp)),
    }
    if is_col:
        info["mesh"] = mesh
    else:
        info["vmesh"] = mesh

    return info


def parse_geoms(mj, scale, surface, xml_path):
    links_g_info = [[] for _ in range(mj.nbody)]

    # Loop over all geometries sequentially
    is_any_col = False
    for i_g in range(mj.ngeom):
        if mj.geom_bodyid[i_g] < 0:
            continue

        # try parsing a given geometry
        g_info = parse_geom(mj, i_g, scale, surface, xml_path)
        if g_info is None:
            continue

        # Ignore world when looking for collision geometries
        if mj.geom_bodyid[i_g] == 0:
            is_any_col |= g_info["contype"] or g_info["conaffinity"]

        # assign geoms to link
        link_idx = mj.geom_bodyid[i_g]
        links_g_info[link_idx].append(g_info)

    # Inform the user that collision geometries are not displayed by default
    if is_any_col and surface.vis_mode != "collision":
        gs.logger.info(
            "Collision meshes are not visualized by default. To visualize them, please use `vis_mode='collision'` "
            "when calling `scene.add_entity`."
        )

    # Parse geometry group if available.
    # Duplicate collision geometries as visual for bodies not having dedicated visual geometries as a fallback.
    for link_g_info in links_g_info:
        has_visual_group = any(g_info["group"] > 0 for g_info in link_g_info)
        is_all_col = all(g_info["contype"] or g_info["conaffinity"] for g_info in link_g_info)
        for g_info in link_g_info.copy():
            group = g_info.pop("group")
            is_col = g_info["contype"] or g_info["conaffinity"]
            if (has_visual_group and group in (1, 2) and is_col) or (not has_visual_group and is_all_col):
                g_info = g_info.copy()
                mesh = g_info.pop("mesh")
                vmesh = gs.Mesh(
                    mesh=mesh.trimesh,
                    surface=surface,
                    uvs=mesh.uvs,
                    metadata=mesh.metadata,
                )
                g_info = {**g_info, "vmesh": vmesh, "contype": 0, "conaffinity": 0}
                link_g_info.append(g_info)

    return links_g_info


def parse_equality(mj, i_e, scale, ordered_links_idx):
    e_info = dict()
    mj_equality = mj.equality(i_e)
    e_info["name"] = mj_equality.name

    e_info["eq_data"] = mj.eq_data[i_e]
    e_info["eq_data"][:6] *= scale
    e_info["sol_params"] = np.concatenate((mj.eq_solref[i_e], mj.eq_solimp[i_e]))

    if mj.eq_type[i_e] == mujoco.mjtEq.mjEQ_CONNECT:
        e_info["type"] = gs.EQUALITY_TYPE.CONNECT
        e_info["eq_obj1id"] = -1 if mj.eq_obj1id[i_e] == 0 else ordered_links_idx.index(mj.eq_obj1id[i_e] - 1)
        e_info["eq_obj2id"] = -1 if mj.eq_obj2id[i_e] == 0 else ordered_links_idx.index(mj.eq_obj2id[i_e] - 1)
    elif mj.eq_type[i_e] == mujoco.mjtEq.mjEQ_WELD:
        e_info["type"] = gs.EQUALITY_TYPE.WELD
        e_info["eq_obj1id"] = -1 if mj.eq_obj1id[i_e] == 0 else ordered_links_idx.index(mj.eq_obj1id[i_e] - 1)
        e_info["eq_obj2id"] = -1 if mj.eq_obj2id[i_e] == 0 else ordered_links_idx.index(mj.eq_obj2id[i_e] - 1)
    elif mj.eq_type[i_e] == mujoco.mjtEq.mjEQ_JOINT:
        e_info["eq_obj1id"] = mj.eq_obj1id[i_e]
        e_info["eq_obj2id"] = mj.eq_obj2id[i_e]
        # y -y0 = a0 + a1 * (x-x0) + a2 * (x-x0)^2 + a3 * (x-x0)^3 + a4 * (x-x0)^4
        e_info["type"] = gs.EQUALITY_TYPE.JOINT
    else:
        raise gs.raise_exception(f"Unsupported MJCF equality type: {mj.eq_type[i_e]}")

    return e_info
