"""System-unaware Rigid numerical tools shared across backend and coupling code."""

from collections.abc import Sequence
from typing import TYPE_CHECKING, TypeAlias

import numpy as np
from numpy.typing import NDArray
import quadrants as qd
import torch

import genesis as gs
import genesis.utils.array_class as array_class
import genesis.utils.geom as gu
from genesis.constants import link_ref_frame
from genesis.options.solvers import RigidOptions

if TYPE_CHECKING:
    from genesis.engine.entities import RigidEntity

HostTensor: TypeAlias = NDArray | torch.Tensor

IMP_MIN = 0.0001
IMP_MAX = 0.9999


def resolve_broadphase_traversal(
    options: RigidOptions,
    enable_heterogeneous: bool,
) -> gs.broadphase_traversal:
    """Resolve broad-phase traversal from host configuration only."""
    if options.broadphase_traversal is not None:
        return options.broadphase_traversal
    if gs.backend == gs.cpu or enable_heterogeneous:
        return gs.broadphase_traversal.SAP
    return gs.broadphase_traversal.ALL_VS_ALL


def sanitize_ref_frame(ref: link_ref_frame, *, has_root_com: bool = True) -> int:
    """Validate a public link reference frame and return its template-safe integer value."""
    frames = tuple(link_ref_frame) if has_root_com else (link_ref_frame.link_origin, link_ref_frame.link_COM)
    if not isinstance(ref, link_ref_frame) or ref not in frames:
        refs = ", ".join(f"'gs.link_ref_frame.{frame.name}'" for frame in frames)
        gs.raise_exception(f"'ref' must be one of {refs}.")
    return int(ref)


def sanitize_sol_params(
    sol_params: HostTensor,
    min_timeconst: float,
    default_timeconst: float | None = None,
    *,
    floor_timeconst: bool = True,
) -> HostTensor:
    """Sanitize explicit solver parameters without depending on a Solver or System."""
    timeconst, dampratio, dmin, dmax, width, mid, power = sol_params.reshape((-1, 7)).T
    direct_mask = timeconst < 0.0
    if direct_mask.any():
        gs.logger.warning(
            "Constraint solver `timeconst` is negative, which parameterizes the constraint by its stiffness and "
            "damping directly. Genesis does not support it for now, so the default time constant is used instead."
        )
        timeconst[direct_mask] = 0.0
        dampratio[direct_mask] = gu.default_solver_params()[1]
    if (timeconst < gs.EPS).any() and default_timeconst is not None:
        gs.logger.debug(
            f"Constraint solver time constant not specified. Using default value (`{default_timeconst:0.6g}`)."
        )
    invalid_mask = (timeconst > gs.EPS) & (timeconst + gs.EPS < min_timeconst)
    if invalid_mask.any():
        gs.logger.warning(
            "Constraint solver time constant should be greater than 2*substep_dt. timeconst is changed from "
            f"`{min(timeconst[invalid_mask]):0.6g}` to `{min_timeconst:0.6g}`). Decrease simulation timestep or "
            "increase timeconst to avoid altering the original value."
        )
    if default_timeconst is not None:
        timeconst[timeconst < gs.EPS] = default_timeconst
    if floor_timeconst:
        timeconst[:] = timeconst.clip(min_timeconst)
    if (dampratio < gs.EPS).any():
        gs.raise_exception(
            "Constraint solver `dampratio` must be strictly positive. Despite its name, it controls spring stiffness, "
            "not damping. See `genesis.utils.geom.default_solver_params` for details."
        )
    dmin[:] = dmin.clip(IMP_MIN, IMP_MAX)
    dmax[:] = dmax.clip(IMP_MIN, IMP_MAX)
    mid[:] = mid.clip(IMP_MIN, IMP_MAX)
    width[:] = width.clip(0.0)
    power[:] = power.clip(1)
    return sol_params


def jacobi_mass_spread_bound(entities: Sequence["RigidEntity"]) -> float:
    """Upper-bound the mass-matrix diagonal spread over every model configuration."""
    children = {}
    for entity in entities:
        for link in entity.links:
            children.setdefault(link.parent_idx, []).append(link)

    entities_lower, entities_upper = [], []
    for entity in entities:
        rotor_links = [
            link
            for link in entity.links
            if link.n_dofs == 1 and link.joints[0].type in (gs.JOINT_TYPE.REVOLUTE, gs.JOINT_TYPE.PRISMATIC)
        ]
        if entity.desc.variants:
            variants_default = np.array([variant.default_armature or 0.0 for variant in entity.desc.variants])
        elif rotor_links:
            variants_default = np.array([entity.main_morph.default_armature or 0.0])
        else:
            variants_default = np.zeros(1)
        variants_links = [entity.desc.links, *(variant.links for variant in entity.desc.variants[1:])]
        variants_lower, variants_upper = [], []
        for variant_default, variant_links in zip(variants_default, variants_links):
            links_desc = {link.idx: link_desc for link, link_desc in zip(entity.links, variant_links)}
            lower, upper = [], []
            for link in entity.links:
                if link.is_fixed:
                    continue
                link_desc = links_desc[link.idx]
                is_rotor = link in rotor_links
                for joint, joint_desc in zip(link.joints, link_desc.joints):
                    if joint.type == gs.JOINT_TYPE.FIXED:
                        continue
                    anchor = joint_desc.pos
                    is_sole_joint = len(link.joints) == 1
                    if is_sole_joint:
                        dist_com = {link.idx: np.linalg.norm(link_desc.inertial_pos - anchor)}
                    else:
                        dist_com = {link.idx: np.linalg.norm(anchor) + np.linalg.norm(link_desc.inertial_pos)}
                    dist_origin = {link.idx: np.linalg.norm(anchor)}
                    subtree, stack = [], [link]
                    while stack:
                        current = stack.pop()
                        subtree.append(current)
                        for child in children.get(current.idx, []):
                            child_desc = links_desc.get(child.idx, child.desc)
                            hop = np.linalg.norm(child_desc.pos) + 2.0 * sum(
                                np.linalg.norm(child_joint.pos) for child_joint in child_desc.joints
                            )
                            hop += sum(
                                np.ptp(child_joint.dofs_limit)
                                for child_joint in child_desc.joints
                                if child_joint.type == gs.JOINT_TYPE.PRISMATIC
                            )
                            dist_origin[child.idx] = dist_origin[current.idx] + hop
                            dist_com[child.idx] = dist_origin[child.idx] + np.linalg.norm(child_desc.inertial_pos)
                            stack.append(child)
                    subtree_desc = [links_desc.get(subtree_link.idx, subtree_link.desc) for subtree_link in subtree]
                    subtree_mass = sum(subtree_link_desc.mass for subtree_link_desc in subtree_desc)
                    eigvals = np.linalg.eigvalsh(link_desc.inertia)
                    rotational_descendant_upper = sum(
                        np.linalg.eigvalsh(subtree_link_desc.inertia)[-1]
                        + subtree_link_desc.mass * dist_com[subtree_link.idx] ** 2
                        for subtree_link, subtree_link_desc in zip(subtree, subtree_desc)
                        if subtree_link is not link
                    )
                    if is_sole_joint:
                        inertial_rotation = gu.quat_to_R(link_desc.inertial_quat)
                        inertia_com = inertial_rotation @ link_desc.inertia @ inertial_rotation.T
                        offset_com = link_desc.inertial_pos - anchor
                        rotational_self_lower = eigvals[0]
                        rotational_self_upper = eigvals[-1] + link_desc.mass * np.dot(offset_com, offset_com)
                    else:
                        inertia_com = None
                        offset_com = None
                        rotational_self_lower = eigvals[0]
                        rotational_self_upper = eigvals[-1] + link_desc.mass * dist_com[link.idx] ** 2
                    for i_d, armature_d in enumerate(joint_desc.dofs_armature):
                        armature = variant_default if is_rotor and armature_d <= 0.0 else armature_d
                        if joint.type == gs.JOINT_TYPE.PRISMATIC or (
                            joint.type == gs.JOINT_TYPE.FREE and i_d < 3
                        ):
                            lower.append(armature + subtree_mass)
                            upper.append(armature + subtree_mass)
                        elif joint.type == gs.JOINT_TYPE.REVOLUTE and is_sole_joint:
                            axis = joint_desc.dofs_motion_ang[i_d]
                            lever = np.cross(axis, offset_com)
                            rotational_self = axis @ inertia_com @ axis + link_desc.mass * np.dot(lever, lever)
                            lower.append(armature + rotational_self)
                            upper.append(armature + rotational_self + rotational_descendant_upper)
                        else:
                            lower.append(armature + max(rotational_self_lower, 0.0))
                            upper.append(armature + rotational_self_upper + rotational_descendant_upper)
            variants_lower.append(lower)
            variants_upper.append(upper)
        if variants_lower[0]:
            lower, upper = np.array(variants_lower).T, np.array(variants_upper).T
            entities_lower.append(np.where(lower > 0.0, lower, np.inf).min(axis=0))
            entities_upper.append(upper.max(axis=0))
    if not entities_lower:
        return 0.0

    lower_min = np.array([entity_lower.min() for entity_lower in entities_lower])
    mass_spread = 0.0
    for i_e, (entity_lower, entity_upper) in enumerate(zip(entities_lower, entities_upper)):
        lower_paired = np.minimum(entity_lower, np.delete(lower_min, i_e).min(initial=np.inf))
        is_bounded = np.isfinite(lower_paired)
        mass_spread_entity = np.zeros_like(entity_upper)
        np.divide(entity_upper, lower_paired, out=mass_spread_entity, where=is_bounded)
        mass_spread = max(mass_spread, mass_spread_entity.max())
    return mass_spread


@qd.func
def func_vel_at_point(
    link_idx: int,
    i_b: int,
    pos_world: qd.types.vector(3),
    links_state: array_class.LinksState,
) -> qd.types.vector(3):
    """Velocity of a point on a rigid link."""
    vel_rot = links_state.cd_ang[link_idx, i_b].cross(pos_world - links_state.root_COM[link_idx, i_b])
    vel_lin = links_state.cd_vel[link_idx, i_b]
    return vel_rot + vel_lin


@qd.func
def func_apply_coupling_force(
    link_idx: int, env_idx: int, pos: qd.types.vector(3), force: qd.types.vector(3), links_state: array_class.LinksState
):
    """Accumulate a coupling force and its torque on a rigid link."""
    torque = (pos - links_state.root_COM[link_idx, env_idx]).cross(force)
    links_state.cfrc_coupling_ang[link_idx, env_idx] -= torque
    links_state.cfrc_coupling_vel[link_idx, env_idx] -= force
