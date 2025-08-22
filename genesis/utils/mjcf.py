import os
import xml.etree.ElementTree as ET
import contextlib
from pathlib import Path
from itertools import chain
from bisect import bisect_right
import io

import numpy as np
import trimesh
from trimesh.visual.texture import TextureVisuals
from PIL import Image

import z3
import mujoco
import genesis as gs
from genesis.ext import urdfpy

from . import geom as gu
from . import urdf as uu
from .misc import get_assets_dir, redirect_libc_stderr


MIN_TIMECONST = np.finfo(np.double).eps


def build_model(xml, discard_visual, default_armature=None, merge_fixed_links=False, links_to_keep=()):
    if isinstance(xml, (str, Path)):
        # Make sure that it is pointing to a valid XML content (either file path or string)
        path = os.path.join(get_assets_dir(), xml)
        is_valid_path = False
        try:
            if os.path.exists(path):
                xml = ET.parse(path)
                is_valid_path = True
            else:
                xml = ET.fromstring(xml)
        except ET.ParseError:
            gs.raise_exception_from(f"'{xml}' is not a valid XML file path or string.")

        # Best guess for the search path
        asset_path = os.path.dirname(path) if is_valid_path else os.getcwd()

        # Detect whether it is a URDF file or a Mujoco MJCF file
        root = xml.getroot()
        is_urdf_file = root.tag == "robot"

        # Make sure compiler options are defined
        mjcf = ET.SubElement(root, "mujoco") if is_urdf_file else root
        compiler = mjcf.find("compiler")
        if compiler is None:
            compiler = ET.SubElement(mjcf, "compiler")

        # Set absolute asset search directory
        for name in ("assetdir", "meshdir", "texturedir"):
            compiler.attrib[name] = str(Path(asset_path) / compiler.attrib.get(name, ""))

        # Set default constraint solver time constant and motor armature.
        # Note that these default options are ignored when parsing URDF files.
        default = mjcf.find("default")
        if default is None:
            default = ET.SubElement(mjcf, "default")
        for group_name, params_name in (
            ("geom", ("solref",)),
            ("joint", ("solreflimit", "solreffriction")),
            ("equality", ("solref",)),
        ):
            group = default.find(group_name)
            if group is None:
                group = ET.SubElement(default, group_name)
            for param_name in params_name:
                # 0.0 cannot be used because it is considered as an error, so that it will fallback to the original
                # default value...
                group.attrib.setdefault(param_name, str(MIN_TIMECONST))
        if default_armature is not None:
            default.find("joint").attrib.setdefault("armature", str(default_armature))

        # Must pre-process URDF to overwrite default Mujoco compile flags
        if is_urdf_file:
            robot = urdfpy.URDF._from_xml(root, root, asset_path)

            # Merge fixed links if requested
            if merge_fixed_links:
                robot = uu.merge_fixed_links(robot, links_to_keep)
                root = robot._to_xml(None, asset_path)
                root.append(mjcf)

            # Enforce some compiler options
            compiler.attrib |= dict(
                fusestatic="false",
                strippath="false",
                inertiafromgeom="false",  # This option is unreliable, so doing this ourselves
                balanceinertia="false",
                discardvisual="true" if discard_visual else "false",
                autolimits="true",
            )

            # Bound mass and inertia if necessary
            if not all(link.inertial is not None for link in robot.links):
                compiler.attrib |= dict(
                    boundmass=str(MIN_TIMECONST),
                    boundinertia=str(MIN_TIMECONST),
                )

            # Resolve relative mesh paths
            for elem in root.findall(".//mesh"):
                mesh_path = elem.get("filename")
                if mesh_path.startswith("package://"):
                    mesh_path = mesh_path[10:]
                elem.set("filename", os.path.abspath(os.path.join(asset_path, mesh_path)))

        with open(os.devnull, "w") as stderr, redirect_libc_stderr(stderr):
            # Parse updated URDF file as a string
            data = ET.tostring(root, encoding="utf8")
            mj = mujoco.MjModel.from_xml_string(data)
            # Special treatment for URDF
            if is_urdf_file:
                # Discard placeholder inertias that were used to avoid parsing failure
                for i, link in enumerate(robot.links):
                    if link.inertial is None:
                        body = mj.body(link.name)
                        body.inertia[:] = 0.0
                        body.mass[:] = 0.0
                        body.invweight0[:] = 0.0

                # Set default constraint solver time constant
                mj.jnt_solref[:, 0] = MIN_TIMECONST
                mj.geom_solref[:, 0] = MIN_TIMECONST
                mj.eq_solref[:, 0] = MIN_TIMECONST

                # Set default rotor armature inertia
                if default_armature is not None:
                    mj.dof_armature[:] = default_armature
                    mj.body_invweight0[:] = 0.0
                    mj.dof_invweight0[:] = 0.0
    elif isinstance(xml, mujoco.MjModel):
        mj = xml
    else:
        raise gs.raise_exception(f"'{xml}' is not a valid MJCF file.")

    return mj


def parse_xml(morph, surface):
    # Always merge fixed links unless explicitly asked not to do so
    merge_fixed_links, links_to_keep = False, ()
    if isinstance(morph, (gs.morphs.URDF, gs.morphs.Drone)):
        merge_fixed_links = morph.merge_fixed_links
        links_to_keep = morph.links_to_keep

    # Build model from XML (either URDF or MJCF)
    mj = build_model(morph.file, not morph.visualization, morph.default_armature, merge_fixed_links, links_to_keep)

    # We have another more informative warning later so we suppress this one
    # gs.logger.warning(f"(MJCF) Approximating tendon by joint actuator for `{j_info['name']}`")
    # if mj.ntendon:
    #     gs.logger.warning("(MJCF) Tendon not supported")

    # Parse all geometries grouped by parent joint (or world)
    links_g_infos = parse_geoms(mj, morph.scale, surface, morph.file)

    # Parse all bodies (links and joints)
    l_infos, links_j_infos = parse_links(mj, morph.scale)

    # Re-order kinematic tree info
    l_infos, links_j_infos, links_g_infos, _ = uu._order_links(l_infos, links_j_infos, links_g_infos)

    # Parsing all equality constraints
    eqs_info = parse_equalities(mj, morph.scale)

    return l_infos, links_j_infos, links_g_infos, eqs_info


def parse_link(mj, i_l, scale):
    # mj.body
    l_info = dict()

    name_start = mj.name_bodyadr[i_l]
    l_info["name"], *_ = filter(None, mj.names[name_start:].decode("utf-8").split("\x00"))

    l_info["pos"] = mj.body_pos[i_l]
    l_info["quat"] = mj.body_quat[i_l]
    l_info["inertial_pos"] = mj.body_ipos[i_l]
    l_info["inertial_quat"] = mj.body_iquat[i_l]
    l_info["inertial_i"] = np.diag(mj.body_inertia[i_l])
    l_info["inertial_mass"] = float(mj.body_mass[i_l])
    if mj.body_parentid[i_l] == i_l:
        l_info["parent_idx"] = -1
    else:
        l_info["parent_idx"] = int(mj.body_parentid[i_l])
    l_info["root_idx"] = int(mj.body_rootid[i_l])
    l_info["invweight"] = mj.body_invweight0[i_l]

    jnt_adr = mj.body_jntadr[i_l]
    jnt_num = mj.body_jntnum[i_l]

    j_infos = []
    for i_j in range(jnt_adr, jnt_adr + max(jnt_num, 1)):
        j_info = dict()

        # Parsing joint type
        mj_type = mj.jnt_type[i_j] if i_j != -1 else None
        if mj_type is None:
            gs_type = gs.JOINT_TYPE.FIXED
            n_qs, n_dofs = 0, 0
        elif mj_type == mujoco.mjtJoint.mjJNT_FREE:
            gs_type = gs.JOINT_TYPE.FREE
            n_qs, n_dofs = 7, 6
        elif mj_type == mujoco.mjtJoint.mjJNT_HINGE:
            gs_type = gs.JOINT_TYPE.REVOLUTE
            n_qs, n_dofs = 1, 1
        elif mj_type == mujoco.mjtJoint.mjJNT_SLIDE:
            gs_type = gs.JOINT_TYPE.PRISMATIC
            n_qs, n_dofs = 1, 1
        elif mj_type == mujoco.mjtJoint.mjJNT_BALL:
            gs_type = gs.JOINT_TYPE.SPHERICAL
            n_qs, n_dofs = 4, 3
        else:
            gs.raise_exception(f"Unsupported MJCF joint type: {mj_type}")
        j_info["type"], j_info["n_qs"], j_info["n_dofs"] = gs_type, n_qs, n_dofs

        # Parsing joint parameters that are type-agnostic
        mj_dof_offset = mj.jnt_dofadr[i_j] if i_j != -1 else 0
        mj_qpos_offset = mj.jnt_qposadr[i_j] if i_j != -1 else 0
        if i_j == -1:
            j_info["name"] = l_info["name"]
            j_info["pos"] = np.array([0.0, 0.0, 0.0])
        else:
            name_start = mj.name_jntadr[i_j]
            j_info["name"], *_ = filter(None, mj.names[name_start:].decode("utf-8").split("\x00"))
            j_info["pos"] = mj.jnt_pos[i_j]
        j_info["quat"] = np.array([1.0, 0.0, 0.0, 0.0])
        j_info["init_qpos"] = np.array(mj.qpos0[mj_qpos_offset : (mj_qpos_offset + n_qs)])
        j_info["dofs_damping"] = mj.dof_damping[mj_dof_offset : (mj_dof_offset + n_dofs)]
        j_info["dofs_invweight"] = mj.dof_invweight0[mj_dof_offset : (mj_dof_offset + n_dofs)]
        j_info["dofs_armature"] = mj.dof_armature[mj_dof_offset : (mj_dof_offset + n_dofs)]
        j_info["dofs_frictionloss"] = mj.dof_frictionloss[mj_dof_offset : (mj_dof_offset + n_dofs)]
        if mj.njnt > 0:
            mj_jnt_offset = i_j if i_j != -1 else 0
            j_info["sol_params"] = np.concatenate((mj.jnt_solref[mj_jnt_offset], mj.jnt_solimp[mj_jnt_offset]))
        else:
            j_info["sol_params"] = gu.default_solver_params()  # Placeholder. It will not be used anyway.

        # Parsing joint parameters that are type-specific
        mj_stiffness = mj.jnt_stiffness[i_j] if i_j != -1 else 0.0
        mj_is_limited = mj.jnt_limited[i_j] == 1 if i_j != -1 else False
        if gs_type == gs.JOINT_TYPE.FIXED:
            j_info["dofs_motion_ang"] = np.zeros((0, 3))
            j_info["dofs_motion_vel"] = np.zeros((0, 3))
            j_info["dofs_limit"] = np.zeros((0, 2))
            j_info["dofs_stiffness"] = np.zeros((0))
        elif gs_type == gs.JOINT_TYPE.FREE:
            if mj_stiffness > 0.0:
                raise gs.raise_exception("(MJCF) Joint stiffness not supported for free joints")

            j_info["dofs_motion_ang"] = np.eye(6, 3, -3)
            j_info["dofs_motion_vel"] = np.eye(6, 3)
            j_info["dofs_limit"] = np.tile([-np.inf, np.inf], (6, 1))
            j_info["dofs_stiffness"] = np.zeros(6)

            j_info["init_qpos"][:3] *= scale
        elif gs_type == gs.JOINT_TYPE.SPHERICAL:
            if mj_is_limited:
                gs.logger.warning("(MJCF) Joint limit ignored for ball joints")

            j_info["dofs_motion_ang"] = np.eye(3)
            j_info["dofs_motion_vel"] = np.zeros((3, 3))
            j_info["dofs_limit"] = np.tile([-np.inf, np.inf], (3, 1))
            j_info["dofs_stiffness"] = np.full((3,), mj_stiffness)
        else:
            mj_axis = mj.jnt_axis[i_j]
            mj_limit = mj.jnt_range[i_j] if mj_is_limited else np.array([-np.inf, np.inf])

            if gs_type == gs.JOINT_TYPE.REVOLUTE:
                j_info["dofs_motion_ang"] = np.array([mj_axis])
                j_info["dofs_motion_vel"] = np.zeros((1, 3))
                j_info["dofs_limit"] = np.array([mj_limit])
                j_info["dofs_stiffness"] = np.array([mj_stiffness])
            else:  # gs_type == gs.JOINT_TYPE.PRISMATIC:
                j_info["dofs_motion_ang"] = np.zeros((1, 3))
                j_info["dofs_motion_vel"] = np.array([mj_axis])
                j_info["dofs_limit"] = np.array([mj_limit]) * scale
                j_info["dofs_stiffness"] = np.array([mj_stiffness])

                j_info["init_qpos"] *= scale

        # Parsing actuator parameters
        j_info["dofs_kp"] = np.zeros((n_dofs,), dtype=gs.np_float)
        j_info["dofs_kv"] = np.zeros((n_dofs,), dtype=gs.np_float)
        j_info["dofs_force_range"] = np.tile([-np.inf, np.inf], (n_dofs, 1))

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
                        "(MJCF) Actuator control gain and bias parameters not supported. "
                        f"Using default values for joint `{j_info['name']}`"
                    )
                    actuator_kp = gu.default_dofs_kp(1)[0]
                    actuator_kv = gu.default_dofs_kv(1)[0]
                elif gainprm[0] != -biasprm[1]:
                    # Doing our best to approximate the expected behavior: g0 * p_target + b1 * p_mes + b2 * v_mes
                    gs.logger.warning(
                        "(MJCF) Actuator control gain and bias parameters cannot be reduced to a unique PD control "
                        f"position gain. Using max between gain and bias for joint `{j_info['name']}`."
                    )
                    actuator_kp = min(-gainprm[0], biasprm[1])
                    actuator_kv = biasprm[2]
                else:
                    actuator_kp, actuator_kv = biasprm[1], biasprm[2]

            gear = mj.actuator_gear[i_a, 0]
            j_info["dofs_kp"] = np.tile(-gear * actuator_kp * scale**3, (n_dofs,))
            j_info["dofs_kv"] = np.tile(-gear * actuator_kv * scale**3, (n_dofs,))
            if mj.actuator_forcelimited[i_a]:
                j_info["dofs_force_range"] = np.tile(mj.actuator_forcerange[i_a], (n_dofs, 1))
            if mj.actuator_ctrllimited[i_a] and biastype == mujoco.mjtBias.mjBIAS_NONE:
                j_info["dofs_force_range"] = np.minimum(
                    j_info["dofs_force_range"], np.tile(gear * mj.actuator_ctrlrange[i_a], (n_dofs, 1))
                )
        elif gs_type not in (gs.JOINT_TYPE.FIXED, gs.JOINT_TYPE.FREE):
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
    metadata = {}
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

        mesh_path_start = mj.mesh_pathadr[mj_mesh.id]
        metadata["mesh_path"], *_ = filter(None, mj.paths[mesh_path_start:].decode("utf-8").split("\x00"))
    else:
        gs.logger.warning(f"Unsupported MJCF geom type '{mj_geom.type}'.")
        return None

    mesh = gs.Mesh.from_trimesh(
        tmesh, scale=scale, surface=gs.surfaces.Collision() if is_col else surface, metadata=metadata
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

    # Update contype and conaffinity to take into account any additional list of explicitly excluded collision pairs
    if mj.nexclude:
        # Extract the list of collision geometries
        cg_infos = []
        for g_info in chain.from_iterable(links_g_info):
            if g_info["contype"] or g_info["conaffinity"]:
                cg_infos.append(g_info)

        # Compute the original of all the excluded collision pairs
        invalid_set = set()
        for i, g_info_1 in enumerate(cg_infos):
            for j, g_info_2 in enumerate(cg_infos):
                if i >= j:
                    continue
                if g_info_1["contype"] & g_info_2["conaffinity"]:
                    continue
                if g_info_2["contype"] & g_info_1["conaffinity"]:
                    continue
                invalid_set.add(frozenset((i, j)))

        # Append all the explicitly excluded collision pairs
        for exclude_signature in mj.exclude_signature:
            body_1 = (exclude_signature >> 16) & 0xFFFF
            body_2 = exclude_signature & 0xFFFF

            geoms_1, geoms_2 = [], []
            for body_idx, geoms_idx in ((body_1, geoms_1), (body_2, geoms_2)):
                for g_info in links_g_info[body_idx]:
                    for geom_idx, cg_info in enumerate(cg_infos):
                        if g_info is cg_info:
                            geoms_idx.append(geom_idx)
                            break

            for geom_1 in geoms_1:
                for geom_2 in geoms_2:
                    invalid_set.add(frozenset((geom_1, geom_2)))

        # Compute updated contype and conaffinity from the complete list of invalid collision pairs
        is_success = False
        N = len(cg_infos)
        for K in range(1, 32):
            s = z3.Solver()
            contype_bits = [[z3.Bool(f"contype_{i}_{b}") for b in range(K)] for i in range(N)]
            conaffinity_bits = [[z3.Bool(f"conaffinity_{i}_{b}") for b in range(K)] for i in range(N)]
            for i in range(N):
                for j in range(i + 1, N):
                    cond1 = z3.Or([z3.And(contype_bits[i][b], conaffinity_bits[j][b]) for b in range(K)])
                    cond2 = z3.Or([z3.And(contype_bits[j][b], conaffinity_bits[i][b]) for b in range(K)])
                    pair = frozenset((i, j))
                    if pair in invalid_set:
                        s.add(z3.Not(cond1), z3.Not(cond2))
                    else:
                        s.add(z3.Or(cond1, cond2))
            if s.check() == z3.sat:
                is_success = True
                model = s.model()
                for g_info, contype_bits_i, conaffinity_bits_i in zip(cg_infos, contype_bits, conaffinity_bits):
                    g_info["contype"], g_info["conaffinity"] = (
                        sum((1 << b) if z3.is_true(model[e]) else 0 for b, e in enumerate(bits))
                        for bits in (contype_bits_i, conaffinity_bits_i)
                    )
                break

        if not is_success:
            gs.logger.warning(
                "Compatible collision geometries cannot be described using bitmasks 'contype' and 'conaffinity'. "
                "Using default values..."
            )
            for g_info in cg_infos:
                g_info["contype"], g_info["conaffinity"] = 1, 1

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


def parse_equalities(mj, scale):
    eqs_info = []
    for i_e in range(mj.neq):
        mj_equality = mj.equality(i_e)

        eq_info = dict()
        eq_info["name"] = mj_equality.name
        eq_info["data"] = mj.eq_data[i_e]
        eq_info["sol_params"] = np.concatenate((mj.eq_solref[i_e], mj.eq_solimp[i_e]))

        if mj.eq_type[i_e] == mujoco.mjtEq.mjEQ_CONNECT:
            eq_info["type"] = gs.EQUALITY_TYPE.CONNECT
            eq_info["data"][:6] *= scale
            name_objadr = mj.name_bodyadr
        elif mj.eq_type[i_e] == mujoco.mjtEq.mjEQ_WELD:
            eq_info["type"] = gs.EQUALITY_TYPE.WELD
            eq_info["data"][:6] *= scale
            name_objadr = mj.name_bodyadr
        elif mj.eq_type[i_e] == mujoco.mjtEq.mjEQ_JOINT:
            # y -y0 = a0 + a1 * (x-x0) + a2 * (x-x0)^2 + a3 * (x-x0)^3 + a4 * (x-x0)^4
            eq_info["type"] = gs.EQUALITY_TYPE.JOINT
            name_objadr = mj.name_jntadr
        else:
            raise gs.raise_exception(f"Unsupported MJCF equality type: {mj.eq_type[i_e]}")

        objs_name = []
        for obj_idx in (mj.eq_obj1id[i_e], mj.eq_obj2id[i_e]):
            if obj_idx < 0:
                obj_name = None
            else:
                name_start = name_objadr[obj_idx]
                obj_name, *_ = filter(None, mj.names[name_start:].decode("utf-8").split("\x00"))
            objs_name.append(obj_name)
        eq_info["objs_name"] = objs_name

        eqs_info.append(eq_info)

    return eqs_info
