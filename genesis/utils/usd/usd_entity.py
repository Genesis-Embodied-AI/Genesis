"""
USD entity parsing for scene.add_entity().

This module provides functionality for parsing a single USD entity when called via scene.add_entity().
It analyzes the entity structure and determines whether it's a pure rigid body, pure articulation,
or mixed (which is an error case).
"""

import genesis as gs
from pxr import Usd

from .usd_context import (
    UsdContext,
    extract_links_referenced_by_joints,
    find_joints_in_range,
    find_rigid_bodies_in_range,
)
from .usd_rigid_entity import parse_usd_rigid_entity


def parse_usd_single_entity(morph: gs.morphs.USD, surface: gs.surfaces.Surface):
    """
    Parse a single USD entity for scene.add_entity().

    This function processes a USD morph and determines whether it represents:
    - A pure rigid body (no joints in subtree)
    - A pure articulation (has joints, all links are connected)
    - Mixed case (error - has both joints and unreferenced rigid bodies)

    Note: This function assumes morph.joint_prims is None. If joint_prims is set
    (from parse_usd_stage), parse_usd_rigid_entity should be called directly.

    Parameters
    ----------
    morph : gs.morphs.USD
        The USD morph to parse. joint_prims should be None - this function will
        analyze the prim_path subtree to find joints and determine the entity type.
    surface : gs.surfaces.Surface
        Surface configuration.

    Returns
    -------
    l_infos : list
        List of link info dictionaries.
    links_j_infos : list
        List of lists of joint info dictionaries.
    links_g_infos : list
        List of lists of geometry info dictionaries.
    eqs_info : list
        List of equality constraint info dictionaries.
    """
    # joint_prims should be None at this point - it's an internal field set by parser functions
    # parse_usd_entity() will detect and set it if needed
    assert morph.joint_prims is None, (
        f"morph.joint_prims should be None when entering parse_usd_entity(). "
        f"This is an internal field set by parser functions, not user input. Got: {morph.joint_prims}."
    )

    context: UsdContext = morph.usd_ctx
    stage: Usd.Stage = context.stage

    # Get the entity prim
    if morph.prim_path is None:
        gs.logger.debug("USD morph has no prim path. Fallback to its default prim path.")
        entity_prim = stage.GetDefaultPrim()
    else:
        entity_prim = stage.GetPrimAtPath(morph.prim_path)
    if not entity_prim.IsValid():
        if morph.prim_path is None:
            err_msg = (
                f"Invalid default prim path {entity_prim} in USD file {morph.file}. Please specify 'morph.prim_path'."
            )
        else:
            err_msg = f"Invalid user-specified prim path {entity_prim} in USD file {morph.file}."
        gs.raise_exception(err_msg)

    # Analyze the entity_prim subtree
    # Find all joints in the subtree
    joints_in_subtree = find_joints_in_range(Usd.PrimRange(entity_prim))

    # Find all rigid bodies in the subtree
    rigid_bodies_in_subtree = find_rigid_bodies_in_range(Usd.PrimRange(entity_prim))

    # Extract links referenced by joints (don't check rigid body here, we'll filter later)
    links_referenced_by_joints = extract_links_referenced_by_joints(stage, joints_in_subtree, check_rigid_body=False)

    # Determine entity type
    has_joints = len(joints_in_subtree) > 0
    has_unreferenced_rigid_bodies = len(rigid_bodies_in_subtree - links_referenced_by_joints) > 0

    # Check for mixed case error, because scene.add_entity(...) only return 1 entity.
    if has_joints and has_unreferenced_rigid_bodies:
        unreferenced = rigid_bodies_in_subtree - links_referenced_by_joints
        gs.raise_exception(
            f"Mixed entity detected at {entity_prim.GetPath()}: "
            f"has {len(joints_in_subtree)} joints but also has {len(unreferenced)} rigid bodies "
            f"not referenced by joints: {list(unreferenced)[:5]}. "
            "Use scene.add_stage() to handle mixed entities, or ensure all rigid bodies are connected by joints."
        )

    if not has_joints:
        # Pure rigid body case (no joints)
        morph.joint_prims = None
    else:
        # Pure articulation case (has joints, all rigid bodies are referenced)
        morph.joint_prims = [str(joint.GetPath()) for joint in joints_in_subtree]

    return parse_usd_rigid_entity(morph, surface)
