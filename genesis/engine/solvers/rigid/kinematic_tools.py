"""System-unaware host tools shared by kinematic model owners, the standalone KinematicSolver and RigidDynamics."""

from collections.abc import Callable, Sequence
from dataclasses import asdict
from typing import TYPE_CHECKING, TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray
import quadrants as qd
import torch

import genesis as gs
import genesis.utils.array_class as array_class
import genesis.utils.geom as gu
from genesis.engine.solvers.rigid.abd.accessor import kernel_get_state_grad, kernel_get_terrain_height
from genesis.engine.solvers.rigid.abd.inverse_kinematics import (
    kernel_forward_kinematics_query,
    kernel_get_jacobian,
    kernel_get_jacobian_zero,
)
from genesis.engine.solvers.rigid.abd.misc import (
    kernel_init_dof_fields,
    kernel_init_joint_fields,
    kernel_init_link_fields,
    kernel_init_vgeom_fields,
    kernel_init_vvert_fields,
)
from genesis.utils.misc import broadcast_tensor, indices_to_mask, qd_to_numpy, qd_to_torch

if TYPE_CHECKING:
    from genesis.engine.entities import KinematicEntity
    from genesis.engine.entities.rigid_entity.rigid_geom import RigidGeom, RigidVisGeom
    from genesis.engine.entities.rigid_entity.rigid_joint import RigidJoint
    from genesis.engine.entities.rigid_entity.rigid_link import RigidLink
    from genesis.engine.states.solvers import KinematicSolverState, RigidSolverState

HostArray: TypeAlias = NDArray[np.floating] | torch.Tensor
HostIndex: TypeAlias = ArrayLike | torch.Tensor | None
SolParamsSanitizer: TypeAlias = Callable[[NDArray[np.floating]], NDArray[np.floating]]

TERRAIN_HEIGHT_QUERY_TILT_TOLERANCE = 1e-3


def balanced_variant_mapping(n_variants: int, batch_size: int) -> NDArray[np.integer]:
    """Map variants to environments using balanced contiguous blocks."""
    if batch_size >= n_variants:
        base = batch_size // n_variants
        extra = batch_size % n_variants
        sizes = np.r_[np.full(extra, base + 1), np.full(n_variants - extra, base)]
        return np.repeat(np.arange(n_variants), sizes)
    return np.arange(batch_size)


def select_links_offset(offset: HostArray, links_idx: HostIndex, envs_idx: HostIndex) -> HostArray:
    """Index a base-link forward offset with the same mask as a transposed state query."""
    if offset.ndim == 2:
        return offset[indices_to_mask(links_idx)]
    return offset[indices_to_mask(envs_idx, links_idx)]


def offset_world_shift(offset_pos: HostArray, offset_quat: HostArray, world_quat: HostArray) -> HostArray:
    """Return the world displacement contributed by a body-frame offset position."""
    authored_quat = gu.transform_quat_by_quat(gu.inv_quat(offset_quat), world_quat)
    return gu.transform_by_quat(offset_pos, authored_quat)


def fill_base_link_geom_offsets(
    offset_pos: NDArray[np.floating],
    offset_quat: NDArray[np.floating],
    entity: "KinematicEntity",
    geoms: Sequence["RigidGeom | RigidVisGeom"],
    ranges: Sequence[tuple[int, int]] | None,
) -> None:
    """Fill per-geom forward offsets for one entity's collision or visual geoms."""
    for geom in geoms:
        if ranges is None:
            link_off_pos = geom.link.desc.offset_pos
            link_off_quat = geom.link.desc.offset_quat
            if np.allclose(link_off_pos, 0.0, atol=gs.EPS) and np.allclose(
                link_off_quat, gu.identity_quat(), atol=gs.EPS
            ):
                continue
            morph = entity._morph
        else:
            i_variant = next((i for i, (start, end) in enumerate(ranges) if start <= geom.idx < end), None)
            if i_variant is None:
                continue
            link_off_pos = entity._desc.variants[i_variant].offset_pos
            link_off_quat = entity._desc.variants[i_variant].offset_quat
            morph = entity._morph if i_variant == 0 else entity._morph_heterogeneous[i_variant - 1]
        frame_pos, frame_quat = gu.transform_pos_quat_by_trans_quat(
            geom.init_pos,
            geom.init_quat,
            link_off_pos,
            link_off_quat,
        )
        offset_frame_pos, offset_frame_quat = gu.transform_pos_quat_by_trans_quat(
            frame_pos,
            frame_quat,
            np.array(morph.offset_pos, dtype=gs.np_float),
            np.array(morph.offset_quat, dtype=gs.np_float),
        )
        offset_pos[geom.idx], offset_quat[geom.idx] = gu.inv_transform_pos_quat_by_trans_quat(
            offset_frame_pos,
            offset_frame_quat,
            frame_pos,
            frame_quat,
        )


def init_dof_fields(
    joints: Sequence["RigidJoint"],
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    rigid_config: array_class.RigidSimStaticConfig,
) -> None:
    has_dofs = sum(joint.n_dofs for joint in joints) > 0
    if has_dofs:
        kernel_init_dof_fields(
            np.concatenate(
                [(joint.link._entity_idx_in_solver,) * joint.n_dofs for joint in joints if joint.n_dofs],
                dtype=gs.np_int,
            ),
            np.concatenate([joint.desc.dofs_motion_ang for joint in joints], dtype=gs.np_float),
            np.concatenate([joint.desc.dofs_motion_vel for joint in joints], dtype=gs.np_float),
            np.concatenate([joint.desc.dofs_limit for joint in joints], dtype=gs.np_float),
            np.concatenate([joint.desc.dofs_invweight for joint in joints], dtype=gs.np_float),
            np.concatenate([joint.desc.dofs_stiffness for joint in joints], dtype=gs.np_float),
            np.concatenate([joint.desc.dofs_damping for joint in joints], dtype=gs.np_float),
            np.concatenate([joint.desc.dofs_frictionloss for joint in joints], dtype=gs.np_float),
            np.concatenate([joint.desc.dofs_armature for joint in joints], dtype=gs.np_float),
            np.concatenate([joint.desc.dofs_act_gain for joint in joints], dtype=gs.np_float),
            np.concatenate([joint.desc.dofs_act_bias for joint in joints], dtype=gs.np_float),
            np.concatenate([joint.desc.dofs_force_range for joint in joints], dtype=gs.np_float),
            dyn_state,
            dyn_info,
            rigid_info,
            rigid_config,
        )

    dyn_state.dofs.force.fill(0)


def init_tree_fields(
    links: Sequence["RigidLink"],
    n_dofs: int,
    links_root_idx: NDArray[np.integer],
    links_parent_idx: NDArray[np.integer],
    links_tree_root_idx: NDArray[np.integer],
    links_tree_level: NDArray[np.integer],
    links_root_level: NDArray[np.integer],
    rigid_info: array_class.RigidInfo,
    rigid_config: array_class.RigidSimStaticConfig,
) -> None:
    """Initialize the fields describing the kinematic roots and trees (see roots_link_idx and trees_root_idx in
    array_class.py).

    The links come parent first and each branch occupies a contiguous index range, so the dofs of a tree form one
    contiguous range and the trees are disjoint in dof space.
    """
    n_links = len(links)
    n_links_ = max(1, n_links)
    n_roots = len({link.root_idx for link in links})
    n_trees = np.unique(links_tree_root_idx[links_tree_root_idx >= 0]).size
    # A static link belongs to no tree, and a scene without any tree keeps its padded tree slot empty (see
    # trees_root_idx in array_class.py)
    links_tree_idx = np.full(n_links_, -1, dtype=gs.np_int)
    if n_roots:
        roots_link_idx, links_root_rank = np.unique(links_root_idx, return_inverse=True)
        roots_link_end = np.zeros(n_roots, dtype=gs.np_int)
        np.maximum.at(roots_link_end, links_root_rank, np.arange(1, n_links + 1, dtype=gs.np_int))
        rigid_info.roots_link_idx.from_numpy(roots_link_idx)
        rigid_info.links_root_end.from_numpy(roots_link_end[links_root_rank])
    if n_trees:
        links_n_dofs = np.array([link.n_dofs for link in links], dtype=gs.np_int)
        links_dof_start = np.array([link.dof_start for link in links], dtype=gs.np_int)
        links_dof_end = np.array([link.dof_end for link in links], dtype=gs.np_int)
        tree_links = np.flatnonzero(links_tree_root_idx >= 0)
        trees_root_idx, links_tree_rank = np.unique(links_tree_root_idx[tree_links], return_inverse=True)
        trees_n_dofs = np.zeros(n_trees, dtype=gs.np_int)
        np.add.at(trees_n_dofs, links_tree_rank, links_n_dofs[tree_links])
        # A dof-less link carries no dof range of its own
        dof_links = np.flatnonzero(links_n_dofs[tree_links])
        trees_dof_start = np.full(n_trees, n_dofs, dtype=gs.np_int)
        np.minimum.at(trees_dof_start, links_tree_rank[dof_links], links_dof_start[tree_links][dof_links])
        trees_dof_end = np.zeros(n_trees, dtype=gs.np_int)
        np.maximum.at(trees_dof_end, links_tree_rank[dof_links], links_dof_end[tree_links][dof_links])
        if (trees_dof_end - trees_dof_start != trees_n_dofs).any():
            gs.raise_exception("The dofs of a kinematic tree must be contiguous.")
        trees_link_end = np.zeros(n_trees, dtype=gs.np_int)
        np.maximum.at(trees_link_end, links_tree_rank, tree_links + 1)
        trees_n_links = np.zeros(n_trees, dtype=gs.np_int)
        np.add.at(trees_n_links, links_tree_rank, 1)
        trees_order = np.argsort(trees_dof_start)
        trees_rank = np.empty(n_trees, dtype=gs.np_int)
        trees_rank[trees_order] = np.arange(n_trees, dtype=gs.np_int)
        links_tree_idx[tree_links] = trees_rank[links_tree_rank]
        rigid_info.trees_root_idx.from_numpy(trees_root_idx[trees_order])
        rigid_info.trees_link_end.from_numpy(trees_link_end[trees_order])
        rigid_info.trees_n_links.from_numpy(trees_n_links[trees_order])
        rigid_info.trees_dof_start.from_numpy(trees_dof_start[trees_order])
        rigid_info.trees_n_dofs.from_numpy(trees_n_dofs[trees_order])
    else:
        rigid_info.trees_root_idx.fill(0)
        rigid_info.trees_link_end.fill(0)
    rigid_info.links_tree_idx.from_numpy(links_tree_idx)

    # Fill the level tables of every tree and root for the level sweep (see trees_levels_links_idx in
    # array_class.py). A scene without any tree keeps its padded tree slot at zero levels.
    if n_roots and rigid_config.enable_level_sweep:
        for links_group_idx, links_group_level, n_groups, groups_fields in (
            (
                links_tree_idx,
                links_tree_level,
                max(1, n_trees),
                (
                    rigid_info.trees_level_start,
                    rigid_info.trees_n_levels,
                    rigid_info.trees_levels_links_idx,
                    None,
                    rigid_info.trees_levels_links_end,
                ),
            ),
            (
                links_root_rank,
                links_root_level,
                n_roots,
                (
                    rigid_info.roots_level_start,
                    rigid_info.roots_n_levels,
                    rigid_info.roots_levels_links_idx,
                    rigid_info.roots_levels_links_start,
                    rigid_info.roots_levels_links_end,
                ),
            ),
        ):
            group_links = np.arange(n_links, dtype=gs.np_int)[links_group_idx >= 0]
            links_group = links_group_idx[group_links]
            links_level = links_group_level[group_links]
            groups_n_levels = np.zeros(n_groups, dtype=gs.np_int)
            np.maximum.at(groups_n_levels, links_group, links_level + 1)
            groups_level_start = np.zeros(n_groups, dtype=gs.np_int)
            np.cumsum(np.bincount(links_group, minlength=n_groups)[:-1], out=groups_level_start[1:])
            # The links of every group, group after group, level after level, and each level in ascending order
            entries_order = np.lexsort((group_links, links_level, links_group))
            groups_levels_links_idx = group_links[entries_order]
            entries_group = links_group[entries_order]
            entries_level = links_level[entries_order]
            entries_idx = np.arange(len(groups_levels_links_idx), dtype=gs.np_int)
            # An entry ends its level where the next entry starts another level or another group
            is_level_last = np.ones(len(groups_levels_links_idx), dtype=bool)
            is_level_last[:-1] = (np.diff(entries_group) != 0) | (np.diff(entries_level) != 0)
            levels_last_entry = entries_idx[is_level_last]
            groups_levels_links_end = levels_last_entry[np.searchsorted(levels_last_entry, entries_idx)] + 1
            is_level_first = np.ones(len(groups_levels_links_idx), dtype=bool)
            is_level_first[1:] = is_level_last[:-1]
            levels_first_entry = entries_idx[is_level_first]
            groups_levels_links_start = levels_first_entry[
                np.searchsorted(levels_first_entry, entries_idx, side="right") - 1
            ]
            for field, value in zip(
                groups_fields,
                (
                    groups_level_start,
                    groups_n_levels,
                    groups_levels_links_idx,
                    groups_levels_links_start,
                    groups_levels_links_end,
                ),
            ):
                if field is not None and len(value):
                    field.from_numpy(value)
        child_links = np.arange(n_links, dtype=gs.np_int)[links_parent_idx >= 0]
        links_child_start = np.zeros(n_links_ + 1, dtype=gs.np_int)
        np.cumsum(np.bincount(links_parent_idx[child_links], minlength=n_links_), out=links_child_start[1:])
        rigid_info.links_child_start.from_numpy(links_child_start)
        if len(child_links):
            rigid_info.links_child_idx.from_numpy(
                child_links[np.lexsort((-child_links, links_parent_idx[child_links]))]
            )


def init_link_fields(
    links: Sequence["RigidLink"],
    joints: Sequence["RigidJoint"],
    entities: Sequence["KinematicEntity"],
    init_qpos: NDArray[np.floating],
    n_dofs_: int,
    batch_size: int,
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    rigid_config: array_class.RigidSimStaticConfig,
    sanitize_joint_sol_params: SolParamsSanitizer,
) -> None:
    """Upload the links and the joints, the frames of the heterogeneous variants, and the initial configuration."""
    if links:
        kernel_init_link_fields(
            np.array([link.parent_idx for link in links], dtype=gs.np_int),
            np.array([link.root_idx for link in links], dtype=gs.np_int),
            np.array([link.q_start for link in links], dtype=gs.np_int),
            np.array([link.dof_start for link in links], dtype=gs.np_int),
            np.array([link.joint_start for link in links], dtype=gs.np_int),
            np.array([link.q_end for link in links], dtype=gs.np_int),
            np.array([link.dof_end for link in links], dtype=gs.np_int),
            np.array([link.joint_end for link in links], dtype=gs.np_int),
            np.array([link._entity_idx_in_solver for link in links], dtype=gs.np_int),
            np.array([link.geom_start for link in links], dtype=gs.np_int),
            np.array([link.geom_end for link in links], dtype=gs.np_int),
            np.array([link.vgeom_start for link in links], dtype=gs.np_int),
            np.array([link.vgeom_end for link in links], dtype=gs.np_int),
            np.array([link.is_fixed for link in links], dtype=gs.np_bool),
            np.array([link.desc.pos for link in links], dtype=gs.np_float),
            np.array([link.desc.quat for link in links], dtype=gs.np_float),
            dyn_state,
            dyn_info,
            rigid_info,
            rigid_config,
        )

    if joints:
        joints_sol_params = np.array([joint.desc.sol_params for joint in joints], dtype=gs.np_float)
        joints_sol_params = sanitize_joint_sol_params(joints_sol_params)

        kernel_init_joint_fields(
            np.array([joint.q_start for joint in joints], dtype=gs.np_int),
            np.array([joint.dof_start for joint in joints], dtype=gs.np_int),
            np.array([joint.q_end for joint in joints], dtype=gs.np_int),
            np.array([joint.dof_end for joint in joints], dtype=gs.np_int),
            np.array([joint.type for joint in joints], dtype=gs.np_int),
            joints_sol_params,
            np.array([joint.pos for joint in joints], dtype=gs.np_float),
            dyn_info,
            rigid_config,
        )

    dispatch_heterogeneous_kinematics(entities, batch_size, dyn_state, dyn_info, sanitize_joint_sol_params)

    # Set initial qpos
    if len(init_qpos) > 0:
        init_qpos = np.tile(np.expand_dims(init_qpos, -1), (1, batch_size))

        # Dispatch per-variant init_qpos for heterogeneous entities
        for entity in entities:
            if not entity._desc.variants:
                continue
            n_variants = len(entity._desc.variants)
            variant_idx = balanced_variant_mapping(n_variants, batch_size)
            q_s, q_e = entity.q_start, entity.q_start + entity.n_qs
            for i_b in range(batch_size):
                init_qpos[q_s:q_e, i_b] = entity._desc.variants[variant_idx[i_b]].init_qpos

        rigid_info.qpos0.from_numpy(init_qpos)
        dofs_limit = np.broadcast_to(qd_to_numpy(dyn_info.dofs.limit, transpose=True), (batch_size, n_dofs_, 2))
        is_init_qpos_out_of_bounds = False
        for joint in joints:
            if joint.type in (gs.JOINT_TYPE.REVOLUTE, gs.JOINT_TYPE.PRISMATIC):
                is_init_qpos_out_of_bounds |= (dofs_limit[:, joint.dof_start, 0] > init_qpos[joint.q_start]).any()
                is_init_qpos_out_of_bounds |= (init_qpos[joint.q_start] > dofs_limit[:, joint.dof_start, 1]).any()
        if is_init_qpos_out_of_bounds:
            gs.logger.warning("Neutral robot position (qpos0) exceeds joint limits.")
        rigid_info.qpos.from_numpy(init_qpos)


def dispatch_heterogeneous_kinematics(
    entities: Sequence["KinematicEntity"],
    batch_size: int,
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    sanitize_joint_sol_params: SolParamsSanitizer,
) -> None:
    """Write the link frames and the joint and dof parameters of each heterogeneous variant to the envs carrying it.

    The init kernels broadcast those of the primary to every environment, while heterogeneous simulation batches
    the link, joint and dof info (see 'build').
    """
    entities = [entity for entity in entities if entity._desc.variants]
    if not entities:
        return

    # Field layout, batch second, since the arrays are written back as fields
    links_pos = qd_to_numpy(dyn_info.links.pos, transpose=False, copy=True)
    links_quat = qd_to_numpy(dyn_info.links.quat, transpose=False, copy=True)
    joints_pos = qd_to_numpy(dyn_info.joints.pos, transpose=False, copy=True)
    joints_sol_params = qd_to_numpy(dyn_info.joints.sol_params, transpose=False, copy=True)
    dofs_info = dyn_info.dofs
    dofs_fields = (
        (dofs_info.motion_ang, "dofs_motion_ang"),
        (dofs_info.motion_vel, "dofs_motion_vel"),
        (dofs_info.limit, "dofs_limit"),
        (dofs_info.stiffness, "dofs_stiffness"),
        (dofs_info.damping, "dofs_damping"),
        (dofs_info.frictionloss, "dofs_frictionloss"),
        (dofs_info.armature, "dofs_armature"),
        (dofs_info.act_gain, "dofs_act_gain"),
        (dofs_info.act_bias, "dofs_act_bias"),
        (dofs_info.force_range, "dofs_force_range"),
    )
    dofs_values = [qd_to_numpy(field, transpose=False, copy=True) for field, _ in dofs_fields]

    for entity in entities:
        variants_links = [entity._desc.links, *(variant.links for variant in entity._desc.variants[1:])]
        envs_variant_idx = balanced_variant_mapping(len(variants_links), batch_size)
        links_slice = slice(entity.link_start, entity.link_end)
        variants_links_pos = np.stack([[l_desc.pos for l_desc in links] for links in variants_links])
        variants_links_quat = np.stack([[l_desc.quat for l_desc in links] for links in variants_links])
        links_pos[links_slice] = variants_links_pos[envs_variant_idx].swapaxes(0, 1)
        links_quat[links_slice] = variants_links_quat[envs_variant_idx].swapaxes(0, 1)

        # A file morph drops its joints without degrees of freedom, so a fixed one may leave the entity without any
        if entity.n_joints == 0:
            continue
        variants_joints = [[asdict(j_desc) for l_desc in links for j_desc in l_desc.joints] for links in variants_links]
        joints_slice = slice(entity.joint_start, entity.joint_end)
        variants_joints_pos = np.stack([[j_fields["pos"] for j_fields in joints] for joints in variants_joints])
        joints_pos[joints_slice] = variants_joints_pos[envs_variant_idx].swapaxes(0, 1)
        variants_sol_params = np.stack([[j_fields["sol_params"] for j_fields in joints] for joints in variants_joints])
        variants_sol_params = sanitize_joint_sol_params(variants_sol_params).reshape(variants_sol_params.shape)
        joints_sol_params[joints_slice] = variants_sol_params[envs_variant_idx].swapaxes(0, 1)
        dofs_slice = slice(entity.dof_start, entity.dof_end)
        for dofs_value, (_, name) in zip(dofs_values, dofs_fields):
            variants_value = np.stack(
                [np.concatenate([j_fields[name] for j_fields in joints]) for joints in variants_joints]
            )
            dofs_value[dofs_slice] = variants_value[envs_variant_idx].swapaxes(0, 1)

    dyn_info.links.pos.from_numpy(links_pos)
    dyn_info.links.quat.from_numpy(links_quat)
    dyn_info.joints.pos.from_numpy(joints_pos)
    dyn_info.joints.sol_params.from_numpy(joints_sol_params)
    for (field, _), dofs_value in zip(dofs_fields, dofs_values):
        field.from_numpy(dofs_value)

    # The state of a fixed root link is set at init alone (see kernel_init_link_fields), so it takes the frame of
    # the variant its environment carries too
    fixed_roots_idx = [entity.link_start for entity in entities if entity.base_link.is_fixed]
    if fixed_roots_idx:
        links_state_pos = qd_to_numpy(dyn_state.links.pos, transpose=False, copy=True)
        links_state_quat = qd_to_numpy(dyn_state.links.quat, transpose=False, copy=True)
        links_state_pos[fixed_roots_idx] = links_pos[fixed_roots_idx]
        links_state_quat[fixed_roots_idx] = links_quat[fixed_roots_idx]
        dyn_state.links.pos.from_numpy(links_state_pos)
        dyn_state.links.quat.from_numpy(links_state_quat)


def init_vvert_fields(
    vgeoms: Sequence["RigidVisGeom"],
    entities: Sequence["KinematicEntity"],
    n_vverts: int,
    dyn_info: array_class.DynInfo,
    rigid_config: array_class.RigidSimStaticConfig,
) -> None:
    if n_vverts == 0:
        return

    vverts = np.concatenate([vg.init_vverts for vg in vgeoms], dtype=gs.np_float)
    vnormals = np.concatenate([vg.init_vnormals for vg in vgeoms], dtype=gs.np_float)
    vfaces = np.concatenate([vg.init_vfaces + vg._vvert_start for vg in vgeoms], dtype=gs.np_int)
    vverts_vgeom_idx = np.concatenate([np.full(vg.n_vverts, vg.idx) for vg in vgeoms], dtype=gs.np_int)
    vverts_state_idx = np.full(n_vverts, -1, dtype=gs.np_int)
    for entity in entities:
        if not entity._morph.enable_custom_vverts:
            continue
        entity_custom_offset = entity._custom_vvert_start - entity._vvert_start
        for vgeom in entity.vgeoms:
            local = np.arange(vgeom.n_vverts, dtype=gs.np_int)
            vverts_state_idx[vgeom._vvert_start + local] = vgeom._vvert_start + entity_custom_offset + local
    kernel_init_vvert_fields(vverts_vgeom_idx, vverts_state_idx, vverts, vfaces, vnormals, dyn_info, rigid_config)


def init_vgeom_fields(
    vgeoms: Sequence["RigidVisGeom"], dyn_info: array_class.DynInfo, rigid_config: array_class.RigidSimStaticConfig
) -> None:
    if vgeoms:
        kernel_init_vgeom_fields(
            np.array([vgeom.link.idx for vgeom in vgeoms], dtype=gs.np_int),
            np.array([vgeom.vvert_start for vgeom in vgeoms], dtype=gs.np_int),
            np.array([vgeom.vface_start for vgeom in vgeoms], dtype=gs.np_int),
            np.array([vgeom.vvert_end for vgeom in vgeoms], dtype=gs.np_int),
            np.array([vgeom.vface_end for vgeom in vgeoms], dtype=gs.np_int),
            np.array([vgeom.init_pos for vgeom in vgeoms], dtype=gs.np_float),
            np.array([vgeom.init_quat for vgeom in vgeoms], dtype=gs.np_float),
            np.array([vgeom.desc.vmesh.color for vgeom in vgeoms], dtype=gs.np_float),
            dyn_info,
            rigid_config,
        )


def init_vfaces_raycast_mask(
    entities: Sequence["KinematicEntity"], n_vgeoms_: int, dyn_info: array_class.DynInfo
) -> torch.Tensor:
    """Fit the static per-vface visual-raycasting opt-in mask.

    Sized over the padded vgeom count so the gather also covers a solver holding no vgeom at all, whose vface
    fields still carry one padding slot: nothing opts in there, so every entry stays 0.
    """
    vgeom_enabled = torch.zeros(n_vgeoms_, dtype=gs.tc_bool, device=gs.device)
    for entity in entities:
        if not entity.material.use_visual_raycasting:
            continue
        for vgeom in entity.vgeoms:
            vgeom_enabled[vgeom.idx] = 1
    return vgeom_enabled[qd_to_torch(dyn_info.vfaces.vgeom_idx)]


def add_grad_from_state(
    state: "KinematicSolverState | RigidSolverState",
    dyn_state: array_class.DynState,
    rigid_info: array_class.RigidInfo,
    rigid_config: array_class.RigidSimStaticConfig,
) -> None:
    """Accumulate the gradients a queried state received into the configuration, velocity and link poses."""
    qpos_grad = gs.zeros_like(state.qpos)
    dofs_vel_grad = gs.zeros_like(state.dofs_vel)
    links_pos_grad = gs.zeros_like(state.links_pos)
    links_quat_grad = gs.zeros_like(state.links_quat)

    if state.qpos.grad is not None:
        qpos_grad = state.qpos.grad
    if state.dofs_vel.grad is not None:
        dofs_vel_grad = state.dofs_vel.grad
    if state.links_pos.grad is not None:
        links_pos_grad = state.links_pos.grad
    if state.links_quat.grad is not None:
        links_quat_grad = state.links_quat.grad

    kernel_get_state_grad(
        qpos_grad, dofs_vel_grad, links_pos_grad, links_quat_grad, dyn_state, rigid_info, rigid_config
    )


def get_terrain_height(
    terrain: "KinematicEntity",
    link_idx: int,
    positions: ArrayLike | torch.Tensor,
    envs_idx: torch.Tensor,
    is_batched: bool,
    dyn_state: array_class.DynState,
    rigid_config: array_class.RigidSimStaticConfig,
) -> torch.Tensor:
    """Height of the terrain under each horizontal position, in the sanitized environments."""
    positions = torch.as_tensor(positions, dtype=gs.tc_float, device=gs.device)
    if positions.ndim == 0 or positions.ndim > 3 or positions.shape[-1] != 2:
        gs.raise_exception("`positions` must have shape (2,), (n_points, 2), or (n_envs, n_points, 2).")

    n_envs = len(envs_idx)
    is_single_position = positions.ndim == 1
    n_points = 1 if is_single_position else positions.shape[-2]
    if n_points == 0:
        gs.raise_exception("`positions` must contain at least one position.")

    is_per_env = positions.ndim == 3 and positions.shape[0] != 1
    positions = broadcast_tensor(
        positions, gs.tc_float, (n_envs if is_per_env else 1, n_points, 2), ("envs_idx", "positions", "")
    )

    heights = torch.empty((n_envs, n_points), dtype=gs.tc_float, device=gs.device)
    kernel_get_terrain_height(
        envs_idx,
        link_idx,
        terrain._morph.horizontal_scale,
        positions.contiguous(),
        terrain._terrain_height_field,
        heights,
        dyn_state,
        rigid_config,
        tilt_tolerance=TERRAIN_HEIGHT_QUERY_TILT_TOLERANCE,
        is_per_env=is_per_env,
    )

    if not is_batched:
        heights = heights[0]
    if is_single_position:
        heights = heights[..., 0]
    return heights


def get_links_jacobian(
    jacobian: qd.Tensor,
    link_idx: int,
    dof_start: int,
    n_dofs: int,
    local_point: ArrayLike | torch.Tensor | None,
    batch_size: int,
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    rigid_config: array_class.RigidSimStaticConfig,
) -> torch.Tensor:
    """Spatial Jacobian of a link-local point, as (n_envs, 6, n_dofs), for the entity owning that link.

    dof_start offsets the entity degrees of freedom into the solver and n_dofs narrows the shared buffer down to
    that entity. local_point defaults to the link origin.
    """
    if local_point is None:
        kernel_get_jacobian_zero(link_idx, dof_start, jacobian, dyn_state, dyn_info, rigid_config, batch_size)
    else:
        kernel_get_jacobian(link_idx, dof_start, local_point, jacobian, dyn_state, dyn_info, rigid_config, batch_size)
    return qd_to_torch(jacobian, transpose=True, copy=True)[..., :n_dofs]


def forward_kinematics_query(
    entity: "KinematicEntity",
    qpos: ArrayLike | torch.Tensor,
    envs_idx: torch.Tensor,
    is_batched: bool,
    qpos_cache: qd.Tensor,
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    rigid_config: array_class.RigidSimStaticConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Link poses one entity would have at the given configuration, leaving the live configuration as it was.

    Distinct from the forward kinematics the step propagates: this evaluates a configuration the solver does not
    hold, and restores the one it does. Returns positions (n_envs, n_links, 3) and orientations
    (n_envs, n_links, 4), without the batch dimension when the scene is not batched.
    """
    qpos = broadcast_tensor(qpos, gs.tc_float, (len(envs_idx), entity.n_qs), ("envs_idx", "qs_idx")).contiguous()

    qs_idx = torch.arange(entity._q_start, entity._q_start + entity.n_qs, dtype=gs.tc_int, device=gs.device)
    links_idx = torch.arange(entity._link_start, entity._link_start + entity.n_links, dtype=gs.tc_int, device=gs.device)
    links_pos = torch.empty((len(envs_idx), entity.n_links, 3), dtype=gs.tc_float, device=gs.device)
    links_quat = torch.empty((len(envs_idx), entity.n_links, 4), dtype=gs.tc_float, device=gs.device)
    kernel_forward_kinematics_query(
        entity._idx_in_solver,
        qs_idx,
        links_idx,
        envs_idx,
        links_pos,
        links_quat,
        qpos,
        qpos_cache,
        dyn_state,
        dyn_info,
        rigid_info,
        rigid_config,
    )

    if not is_batched:
        links_pos = links_pos[0]
        links_quat = links_quat[0]
    return links_pos, links_quat
