import os
from itertools import chain
from pathlib import Path

import trimesh
import numpy as np

import genesis as gs
from genesis.ext import urdfpy

from . import geom as gu
from . import mesh as mu
from .misc import get_assets_dir


def _order_links(l_infos, j_infos, links_g_infos=None):
    # re-order links based on depth in the kinematic tree, so that parent links are always before child links
    n_links = len(l_infos)
    dict_child = {k: [] for k in range(n_links)}
    for lc in range(n_links):
        if "parent_idx" not in l_infos[lc]:
            l_infos[lc]["parent_idx"] = -1
        lp = l_infos[lc]["parent_idx"]
        if lp != -1:
            dict_child[lp].append(lc)

    ordered_links_idx = []
    n_level = 0
    stack_topology = [lc for lc in range(n_links) if l_infos[lc]["parent_idx"] == -1]
    while len(stack_topology) > 0:
        next_stack = []
        ordered_links_idx.append([])
        for link in stack_topology:
            ordered_links_idx[n_level].append(link)
            next_stack += dict_child[link]
        n_level += 1
        stack_topology = next_stack
    ordered_links_idx = tuple(chain.from_iterable(ordered_links_idx))

    if not ordered_links_idx:
        # avoid case with worldbody without any body (geom directly assigned to worldbody)
        return [], [], [], []

    for l_info in l_infos:
        if l_info["parent_idx"] >= 0:  # non-base link
            l_info["parent_idx"] = ordered_links_idx.index(l_info["parent_idx"])

    new_l_infos = [l_infos[i] for i in ordered_links_idx]
    new_j_infos = [j_infos[i] for i in ordered_links_idx]

    if links_g_infos is not None:
        links_g_infos = [links_g_infos[i] for i in ordered_links_idx]

    return new_l_infos, new_j_infos, links_g_infos, ordered_links_idx


def parse_urdf(morph, surface):
    if isinstance(morph.file, (str, Path)):
        path = os.path.join(get_assets_dir(), morph.file)
        robot = urdfpy.URDF.load(path)
    else:
        robot = morph.file

    # Merge links connected by fixed joints
    if morph.merge_fixed_links:
        robot = merge_fixed_links(robot, morph.links_to_keep)

    link_name_to_idx = dict()
    for idx, link in enumerate(robot.links):
        link_name_to_idx[link.name] = idx

    # Note that each link corresponds to one joint
    n_links = len(robot.links)
    assert n_links == len(robot.joints) + 1
    l_infos = [dict() for _ in range(n_links)]
    links_j_infos = [[] for _ in range(n_links)]
    links_g_infos = [[] for _ in range(n_links)]

    for link, l_info, link_g_infos in zip(robot.links, l_infos, links_g_infos):
        l_info["name"] = link.name

        # No parent by default. It will be overwritten latter on if appropriate.
        l_info["parent_idx"] = -1

        # Neutral pose by default. It will be overwritten latter on if necessary.
        l_info["pos"] = gu.zero_pos()
        l_info["quat"] = gu.identity_quat()

        # we compute urdf's invweight later
        l_info["invweight"] = np.full((2,), fill_value=-1.0)

        if link.inertial is None:
            l_info["inertial_pos"] = gu.zero_pos()
            l_info["inertial_quat"] = gu.identity_quat()
            l_info["inertial_i"] = None
            l_info["inertial_mass"] = None

        else:
            l_info["inertial_pos"] = link.inertial.origin[:3, 3]
            l_info["inertial_quat"] = gu.R_to_quat(link.inertial.origin[:3, :3])
            l_info["inertial_i"] = link.inertial.inertia
            l_info["inertial_mass"] = link.inertial.mass

        for geom in (*link.collisions, *link.visuals):
            link_g_infos_ = []
            geom_is_col = not isinstance(geom, urdfpy.Visual)
            if isinstance(geom.geometry.geometry, urdfpy.Mesh):
                geom_type = gs.GEOM_TYPE.MESH
                geom_data = None

                # One asset (.obj) can contain multiple meshes. Each mesh is one RigidGeom in genesis.
                for tmesh in geom.geometry.meshes:
                    scale = float(morph.scale)
                    if geom.geometry.geometry.scale is not None:
                        scale *= geom.geometry.geometry.scale

                    mesh = gs.Mesh.from_trimesh(
                        tmesh,
                        scale=scale,
                        surface=gs.surfaces.Collision() if geom_is_col else surface,
                        metadata={
                            "mesh_path": urdfpy.utils.get_filename(
                                os.path.dirname(path), geom.geometry.geometry.filename
                            )
                        },
                    )

                    if not geom_is_col and (morph.prioritize_urdf_material or not tmesh.visual.defined):
                        if geom.material is not None and geom.material.color is not None:
                            mesh.set_color(geom.material.color)

                    g_info = {"mesh" if geom_is_col else "vmesh": mesh}
                    link_g_infos_.append(g_info)
            else:
                # Each geometry primitive is one RigidGeom in genesis.
                if isinstance(geom.geometry.geometry, urdfpy.Box):
                    tmesh = trimesh.creation.box(extents=geom.geometry.geometry.size)
                    geom_type = gs.GEOM_TYPE.BOX
                    geom_data = np.array(geom.geometry.geometry.size)
                elif isinstance(geom.geometry.geometry, urdfpy.Cylinder):
                    tmesh = trimesh.creation.cylinder(
                        radius=geom.geometry.geometry.radius, height=geom.geometry.geometry.length
                    )
                    geom_type = gs.GEOM_TYPE.CYLINDER
                    geom_data = None
                elif isinstance(geom.geometry.geometry, urdfpy.Sphere):
                    if geom_is_col:
                        tmesh = trimesh.creation.icosphere(radius=geom.geometry.geometry.radius, subdivisions=2)
                    else:
                        tmesh = trimesh.creation.icosphere(radius=geom.geometry.geometry.radius)
                    geom_type = gs.GEOM_TYPE.SPHERE
                    geom_data = np.array([geom.geometry.geometry.radius])

                mesh = gs.Mesh.from_trimesh(
                    tmesh,
                    scale=morph.scale,
                    surface=gs.surfaces.Collision() if geom_is_col else surface,
                )

                if not geom_is_col:
                    if geom.material is not None and geom.material.color is not None:
                        mesh.set_color(geom.material.color)

                g_info = {"mesh" if geom_is_col else "vmesh": mesh}
                link_g_infos_.append(g_info)

            for g_info in link_g_infos_:
                g_info["type"] = geom_type
                g_info["data"] = geom_data
                g_info["pos"] = geom.origin[:3, 3].copy()
                g_info["quat"] = gu.R_to_quat(geom.origin[:3, :3])
                g_info["contype"] = 1 if geom_is_col else 0
                g_info["conaffinity"] = 1 if geom_is_col else 0
                g_info["friction"] = gu.default_friction()
                g_info["sol_params"] = gu.default_solver_params()
            link_g_infos += link_g_infos_

    #########################  non-base joints and links #########################
    for joint in robot.joints:
        idx = link_name_to_idx[joint.child]
        l_info = l_infos[idx]
        j_info = dict()
        links_j_infos[idx].append(j_info)

        j_info["name"] = joint.name
        j_info["pos"] = gu.zero_pos()
        j_info["quat"] = gu.identity_quat()

        l_info["parent_idx"] = link_name_to_idx[joint.parent]
        l_info["pos"] = joint.origin[:3, 3]
        l_info["quat"] = gu.R_to_quat(joint.origin[:3, :3])

        if joint.joint_type == "fixed":
            j_info["dofs_motion_ang"] = np.zeros((0, 3))
            j_info["dofs_motion_vel"] = np.zeros((0, 3))
            j_info["dofs_limit"] = np.zeros((0, 2))
            j_info["dofs_stiffness"] = np.zeros((0))

            j_info["type"] = gs.JOINT_TYPE.FIXED
            j_info["n_qs"] = 0
            j_info["n_dofs"] = 0
            j_info["init_qpos"] = np.zeros(0)

        elif joint.joint_type == "revolute":
            j_info["dofs_motion_ang"] = np.array([joint.axis])
            j_info["dofs_motion_vel"] = np.zeros((1, 3))
            j_info["dofs_limit"] = np.array(
                [
                    [
                        joint.limit.lower if joint.limit.lower is not None else -np.inf,
                        joint.limit.upper if joint.limit.upper is not None else np.inf,
                    ]
                ]
            )
            j_info["dofs_stiffness"] = np.array([0.0])

            j_info["type"] = gs.JOINT_TYPE.REVOLUTE
            j_info["n_qs"] = 1
            j_info["n_dofs"] = 1
            j_info["init_qpos"] = np.zeros(1)

        elif joint.joint_type == "continuous":
            j_info["dofs_motion_ang"] = np.array([joint.axis])
            j_info["dofs_motion_vel"] = np.zeros((1, 3))
            j_info["dofs_limit"] = np.array([[-np.inf, np.inf]])
            j_info["dofs_stiffness"] = np.array([0.0])

            j_info["type"] = gs.JOINT_TYPE.REVOLUTE
            j_info["n_qs"] = 1
            j_info["n_dofs"] = 1
            j_info["init_qpos"] = np.zeros(1)

        elif joint.joint_type == "prismatic":
            j_info["dofs_motion_ang"] = np.zeros((1, 3))
            j_info["dofs_motion_vel"] = np.array([joint.axis])
            j_info["dofs_limit"] = np.array(
                [
                    [
                        morph.scale * joint.limit.lower if joint.limit.lower is not None else -np.inf,
                        morph.scale * joint.limit.upper if joint.limit.upper is not None else np.inf,
                    ]
                ]
            )
            j_info["dofs_stiffness"] = np.array([0.0])

            j_info["type"] = gs.JOINT_TYPE.PRISMATIC
            j_info["n_qs"] = 1
            j_info["n_dofs"] = 1
            j_info["init_qpos"] = np.zeros(1)

        elif joint.joint_type == "floating":
            j_info["dofs_motion_ang"] = np.eye(6, 3, -3)
            j_info["dofs_motion_vel"] = np.eye(6, 3)
            j_info["dofs_limit"] = np.tile([-np.inf, np.inf], (6, 1))
            j_info["dofs_stiffness"] = np.zeros(6)

            j_info["type"] = gs.JOINT_TYPE.FREE
            j_info["n_qs"] = 7
            j_info["n_dofs"] = 6
            j_info["init_qpos"] = np.concatenate([gu.zero_pos(), gu.identity_quat()])

        else:
            gs.raise_exception(f"Unsupported URDF joint type: {joint.joint_type}")

        j_info["dofs_invweight"] = np.full((j_info["n_dofs"],), fill_value=-1.0)
        j_info["dofs_frictionloss"] = np.zeros(j_info["n_dofs"])
        j_info["sol_params"] = gu.default_solver_params()
        j_info["dofs_kp"] = gu.default_dofs_kp(j_info["n_dofs"])
        j_info["dofs_kv"] = gu.default_dofs_kv(j_info["n_dofs"])
        j_info["dofs_force_range"] = np.tile([-np.inf, np.inf], (j_info["n_dofs"], 1))

        j_info["dofs_damping"] = np.zeros(j_info["n_dofs"])
        j_info["dofs_armature"] = np.zeros(j_info["n_dofs"])
        if joint.joint_type not in ("floating", "fixed") and morph.default_armature is not None:
            j_info["dofs_armature"] = np.full((j_info["n_dofs"],), morph.default_armature)

        if joint.safety_controller is not None:
            if joint.safety_controller.k_position is not None:
                j_info["dofs_kp"] = np.tile(joint.safety_controller.k_position, j_info["n_dofs"])
            if joint.safety_controller.k_velocity is not None:
                j_info["dofs_kv"] = np.tile(joint.safety_controller.k_velocity, j_info["n_dofs"])

        if joint.limit is not None and joint.limit.effort is not None:
            j_info["dofs_force_range"] = np.tile([-joint.limit.effort, joint.limit.effort], (j_info["n_dofs"], 1))

    # Apply scaling factor
    for l_info, link_j_infos, link_g_infos in zip(l_infos, links_j_infos, links_g_infos):
        l_info["pos"] *= morph.scale
        l_info["inertial_pos"] *= morph.scale

        if l_info["inertial_mass"] is not None:
            l_info["inertial_mass"] *= morph.scale**3
        if l_info["inertial_i"] is not None:
            l_info["inertial_i"] *= morph.scale**5

        for j_info in link_j_infos:
            j_info["pos"] *= morph.scale

        for g_info in link_g_infos:
            g_info["pos"] *= morph.scale

    # Re-order kinematic tree info
    l_infos, links_j_infos, links_g_infos, _ = _order_links(l_infos, links_j_infos, links_g_infos)

    eqs_info = parse_equalities(robot, morph)

    return l_infos, links_j_infos, links_g_infos, eqs_info


def parse_equalities(robot, morph):
    eqs_info = []

    for joint in robot.joints:
        if joint.mimic:
            gs.logger.debug(
                f"Joint '{joint.name}' mimics '{joint.mimic.joint}' with multiplier {joint.mimic.multiplier} and offset {joint.mimic.offset}"
            )

            eq_info = dict()
            eq_info["type"] = gs.EQUALITY_TYPE.JOINT
            eq_info["name"] = f"mimic_{joint.name}_to_{joint.mimic.joint}"
            eq_info["objs_name"] = (joint.name, joint.mimic.joint)
            eq_info["sol_params"] = gu.default_solver_params()

            eq_info["data"] = np.zeros([11])
            eq_info["data"][0] = joint.mimic.offset
            eq_info["data"][1] = joint.mimic.multiplier
            eq_info["data"][:6] *= morph.scale

            eqs_info.append(eq_info)

    return eqs_info


def merge_fixed_links(robot, links_to_keep):
    links = robot.links.copy()
    joints = robot.joints.copy()
    link_name_to_idx = {link.name: idx for idx, link in enumerate(links)}
    original_to_merged = {}

    while True:
        fixed_joint_found = False
        for joint in joints:
            if joint.joint_type == "fixed" and joint.child not in links_to_keep:
                parent_name = joint.parent
                child_name = joint.child

                if parent_name in original_to_merged:
                    parent_name = original_to_merged[parent_name]
                if child_name in original_to_merged:
                    child_name = original_to_merged[child_name]

                parent_idx = link_name_to_idx.get(parent_name)
                child_idx = link_name_to_idx.get(child_name)

                if parent_idx is None or child_idx is None:
                    continue

                parent_link = links[parent_idx]
                child_link = links[child_idx]

                if parent_link.name not in original_to_merged:
                    original_to_merged[parent_link.name] = parent_link.name
                original_to_merged[child_link.name] = original_to_merged[parent_link.name]

                update_subtree(links, joints, child_link.name, joint.origin)
                merge_inertia(parent_link, child_link)
                parent_link.visuals.extend(child_link.visuals)
                parent_link.collisions.extend(child_link.collisions)

                links.pop(child_idx)
                joints.remove(joint)

                link_name_to_idx = {link.name: idx for idx, link in enumerate(links)}

                fixed_joint_found = True
                break

        if not fixed_joint_found:
            break

    for joint in joints:
        if joint.parent in original_to_merged:
            joint.parent = original_to_merged[joint.parent]
        if joint.child in original_to_merged:
            joint.child = original_to_merged[joint.child]

    return urdfpy.URDF(robot.name, links=links, joints=joints, materials=robot.materials)


def translate_inertia(I, m, dist):
    """Translate inertia tensor I by dist for a body with mass m."""
    dist = np.array(dist)
    dist_squared = np.dot(dist, dist)
    identity_matrix = np.eye(3)
    translation_matrix = m * (dist_squared * identity_matrix - np.outer(dist, dist))
    return I + translation_matrix


def rotate_inertia(I, R):
    """Rotate inertia tensor I by rotation matrix R."""
    return R @ I @ R.T


def merge_inertia(link1, link2):
    """Combine two links with fixed joint."""
    if link2.inertial is None:
        return

    if link1.inertial is None:
        link1.inertial = link2.inertial
        return

    m1 = link1.inertial.mass
    m2 = link2.inertial.mass

    com1 = link1.inertial.origin[:3, 3]
    com2 = link2.inertial.origin[:3, 3]

    R1 = link1.inertial.origin[:3, :3]
    R2 = link2.inertial.origin[:3, :3]

    combined_mass = m1 + m2
    if combined_mass > 0:
        combined_com = (m1 * com1 + m2 * com2) / combined_mass
    else:
        combined_com = com1

    # Rotate and translate inertia tensors to the new center of mass
    inertia1_rotated = rotate_inertia(link1.inertial.inertia, R1)
    inertia2_rotated = rotate_inertia(link2.inertial.inertia, R2)

    inertia1_new = translate_inertia(inertia1_rotated, m1, combined_com - com1)
    inertia2_new = translate_inertia(inertia2_rotated, m2, combined_com - com2)

    # Combine the inertia tensors
    combined_inertia = inertia1_new + inertia2_new

    # Set the properties of the combined link
    link1.inertial.mass = combined_mass
    link1.inertial.origin[:3, 3] = combined_com
    link1.inertial.origin[:3, :3] = np.eye(3)  # Reset rotation to identity since it's now aligned
    link1.inertial.inertia = combined_inertia


def transform_inertial(inertial, transform):
    if inertial is None:
        return None

    new_origin = transform @ inertial

    return urdfpy.Inertial(origin=new_origin, mass=inertial.mass, inertia=inertial.inertia)


def update_subtree(links, joints, root_name, transform):
    current_name = root_name
    current_idx = next((idx for idx, link in enumerate(links) if link.name == current_name), None)
    if current_idx is None:
        return
    current_link = links[current_idx]

    # Apply the transformation to the current link
    if current_link.inertial is not None:
        current_link.inertial.origin = transform @ current_link.inertial.origin

    for geom in current_link.visuals:
        geom.origin = transform @ geom.origin

    for geom in current_link.collisions:
        geom.origin = transform @ geom.origin

    for joint in joints:
        if joint.parent == current_name:
            joint.origin = transform @ joint.origin
