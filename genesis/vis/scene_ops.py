"""Shared scene manipulation operations for ImGui overlay and Web GUI.

These functions encapsulate rendering-context updates that both the ImGui overlay plugin and the web server need.
Each accepts the relevant Genesis objects explicitly so callers can use their own accessor pattern
(e.g. ``viewer.gs_context`` vs ``scene.visualizer._rasterizer._context``).
"""

import numpy as np

import genesis as gs

FREE_JOINT_POS_LIMIT = 10.0
QUATERNION_COMPONENT_LIMIT = 1.0


def refresh_visual_transforms(scene, ctx):
    """Refresh render transforms so visuals reflect the latest qpos.

    Call after ``entity.set_qpos()`` or ``entity.set_dofs_position()``. Forward kinematics updates joint poses,
    but the rasterizer caches need an explicit poke.

    Args:
        scene: The Genesis scene (needs ``rigid_solver``).
        ctx: The ``RasterizerContext`` instance.
    """
    rigid_solver = scene.rigid_solver
    if not rigid_solver.is_active:
        return
    rigid_solver.update_geoms_render_T()
    rigid_solver.update_vgeoms()
    rigid_solver.update_vgeoms_render_T()
    ctx.update_link_frame(ctx.buffer)
    ctx.update_rigid(ctx.buffer)


def switch_entity_vis_mode(scene, ctx, entity, new_mode):
    """Switch *entity* between ``'visual'`` and ``'collision'`` rendering.

    Removes the old geom nodes from *ctx*, updates transforms, then re-adds geom nodes for *new_mode*.

    Args:
        scene: The Genesis scene.
        ctx: The ``RasterizerContext``.
        entity: A rigid entity with ``vgeoms`` / ``geoms``.
        new_mode: ``"visual"`` or ``"collision"``.
    """
    from genesis.ext import pyrender

    if not hasattr(entity, "surface"):
        return
    old_mode = entity.surface.vis_mode
    if old_mode == new_mode:
        return

    rigid_solver = scene.rigid_solver

    # Remove old geom nodes
    old_geoms = entity.vgeoms if old_mode == "visual" else entity.geoms
    for geom in old_geoms:
        if geom.uid in ctx.rigid_nodes:
            ctx.remove_node(ctx.rigid_nodes[geom.uid])
            del ctx.rigid_nodes[geom.uid]

    entity.surface.vis_mode = new_mode

    rigid_solver.update_geoms_render_T()
    rigid_solver.update_vgeoms()
    rigid_solver.update_vgeoms_render_T()

    if new_mode == "visual":
        geoms = entity.vgeoms
        geoms_T = rigid_solver._vgeoms_render_T
    else:
        geoms = entity.geoms
        geoms_T = rigid_solver._geoms_render_T

    for geom in geoms:
        geom_envs_idx = ctx._get_geom_active_envs_idx(geom, ctx.rendered_envs_idx)
        if len(geom_envs_idx) == 0:
            continue
        mesh = geom.get_trimesh()
        geom_T = geoms_T[geom.idx][geom_envs_idx]
        is_collision = new_mode == "collision"
        ctx.add_rigid_node(
            geom,
            pyrender.Mesh.from_trimesh(
                mesh=mesh,
                poses=geom_T,
                smooth=geom.surface.smooth if not is_collision else False,
                double_sided=geom.surface.double_sided if not is_collision else False,
                is_floor=isinstance(entity._morph, gs.morphs.Plane),
                env_shared=not ctx.env_separate_rigid,
            ),
        )


def set_entity_wireframe(ctx, entity, enable):
    """Toggle wireframe rendering for all geom nodes of *entity*.

    Args:
        ctx: The ``RasterizerContext``.
        entity: A rigid entity.
        enable: ``True`` to enable wireframe.
    """
    geoms = (
        entity.vgeoms
        if hasattr(entity, "surface") and entity.surface.vis_mode == "visual"
        else entity.geoms
        if hasattr(entity, "geoms")
        else []
    )
    for geom in geoms:
        if geom.uid in ctx.rigid_nodes:
            node = ctx.rigid_nodes[geom.uid]
            for primitive in node.mesh.primitives:
                if primitive.material is not None:
                    primitive.material.wireframe = enable
    ctx._scene._meshes_updated = True


def set_entity_contact_viz(entity, enable):
    """Toggle contact-force arrow rendering for *entity* and its links.

    Args:
        entity: A rigid entity.
        enable: ``True`` to show contacts.
    """
    entity._visualize_contact = enable
    if hasattr(entity, "links"):
        for link in entity.links:
            link._visualize_contact = enable


def build_entity_joint_data(entity):
    """Build rich joint metadata for *entity*.

    Handles free joints (7 qpos), spherical joints (4 qpos quaternion), and regular joints correctly
    using ``n_qs`` (not ``n_dofs``) for indexing.

    Returns:
        dict with keys: ``q_names``, ``q_limits_lower``, ``q_limits_upper``, ``q_is_quaternion``, ``quat_groups``,
        ``has_free_joint``, ``free_joint_q_start``.
    """
    q_names = []
    q_limits_lower = []
    q_limits_upper = []
    q_is_quaternion = []
    quat_groups = []
    has_free_joint = False
    free_joint_q_start = -1

    for joint in entity.joints:
        if joint.n_qs == 0 or joint.type == gs.JOINT_TYPE.FIXED:
            continue

        if joint.type == gs.JOINT_TYPE.FREE:
            has_free_joint = True
            free_joint_q_start = len(q_names)
            q_names.extend(
                [
                    f"{joint.name}_x",
                    f"{joint.name}_y",
                    f"{joint.name}_z",
                    f"{joint.name}_qw",
                    f"{joint.name}_qx",
                    f"{joint.name}_qy",
                    f"{joint.name}_qz",
                ]
            )
            q_limits_lower.extend([-FREE_JOINT_POS_LIMIT] * 3 + [-QUATERNION_COMPONENT_LIMIT] * 4)
            q_limits_upper.extend([FREE_JOINT_POS_LIMIT] * 3 + [QUATERNION_COMPONENT_LIMIT] * 4)
            q_is_quaternion.extend([False, False, False, True, True, True, True])
            quat_groups.append([len(q_names) - 4, len(q_names)])
        elif joint.type == gs.JOINT_TYPE.SPHERICAL:
            quat_start = len(q_names)
            q_names.extend(
                [
                    f"{joint.name}_qw",
                    f"{joint.name}_qx",
                    f"{joint.name}_qy",
                    f"{joint.name}_qz",
                ]
            )
            q_limits_lower.extend([-QUATERNION_COMPONENT_LIMIT] * 4)
            q_limits_upper.extend([QUATERNION_COMPONENT_LIMIT] * 4)
            q_is_quaternion.extend([True, True, True, True])
            quat_groups.append([quat_start, quat_start + 4])
        else:
            # Revolute, prismatic, or other joints: n_qs == n_dofs
            for i in range(joint.n_qs):
                name = joint.name if joint.n_qs == 1 else f"{joint.name}[{i}]"
                q_names.append(name)
                lo = float(joint.dofs_limit[i, 0])
                hi = float(joint.dofs_limit[i, 1])
                if not np.isfinite(lo):
                    lo = -1e6
                if not np.isfinite(hi):
                    hi = 1e6
                q_limits_lower.append(lo)
                q_limits_upper.append(hi)
                q_is_quaternion.append(False)

    return {
        "q_names": q_names,
        "q_limits_lower": q_limits_lower,
        "q_limits_upper": q_limits_upper,
        "q_is_quaternion": q_is_quaternion,
        "quat_groups": quat_groups,
        "has_free_joint": has_free_joint,
        "free_joint_q_start": free_joint_q_start,
    }
