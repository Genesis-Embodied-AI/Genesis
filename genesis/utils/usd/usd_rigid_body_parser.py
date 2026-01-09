"""
USD Rigid Body Parser

Parser for extracting rigid body information from USD stages.
The parser is agnostic to genesis structures, focusing only on USD rigid body structure.

Also includes Genesis-specific parsing functions that translate USD structures into Genesis info structures.
"""

import re
from typing import Dict, List, Literal, Tuple

import numpy as np
from pxr import Sdf, Usd, UsdPhysics
from scipy.spatial.transform import Rotation as R

import genesis as gs

from .. import geom as gu
from .usd_geo_adapter import UsdGeometryAdapter, create_geo_info_from_prim, create_geo_infos_from_subtree
from .usd_parser_context import UsdParserContext
from .usd_parser_utils import compute_gs_global_transform


class UsdRigidBodyParser:
    """
    A parser to extract rigid body information from a USD stage.

    The Parser is agnostic to genesis structures, it only focuses on USD rigid body structure.
    """

    def __init__(self, stage: Usd.Stage, rigid_body_prim: Usd.Prim):
        """
        Initialize the rigid body parser.

        Parameters
        ----------
        stage : Usd.Stage
            The USD stage.
        rigid_body_prim : Usd.Prim
            The rigid body prim (must have RigidBodyAPI or CollisionAPI).
        """
        self._stage: Usd.Stage = stage
        self._root: Usd.Prim = rigid_body_prim

        is_rigid_body = UsdRigidBodyParser.is_rigid_body(rigid_body_prim)
        if not is_rigid_body:
            gs.raise_exception(
                f"Provided prim {rigid_body_prim.GetPath()} is not a rigid body, APIs found: {rigid_body_prim.GetAppliedSchemas()}"
            )

        collision_api_only = rigid_body_prim.HasAPI(UsdPhysics.CollisionAPI) and not rigid_body_prim.HasAPI(
            UsdPhysics.RigidBodyAPI
        )
        kinematic_enabled = False
        if rigid_body_prim.HasAPI(UsdPhysics.RigidBodyAPI):
            rigid_body_api = UsdPhysics.RigidBodyAPI(rigid_body_prim)
            kinematic_enabled = bool(rigid_body_api.GetKinematicEnabledAttr().Get())

        self._is_fixed = collision_api_only or kinematic_enabled

    # ==================== Properties ====================

    @property
    def stage(self) -> Usd.Stage:
        return self._stage

    @property
    def rigid_body_prim(self) -> Usd.Prim:
        """
        Get the rigid body Usd.Prim.

        Returns
        -------
        Usd.Prim
            The rigid body Usd.Prim with RigidBodyAPI or CollisionAPI applied.
        """
        return self._root

    @property
    def is_fixed(self) -> bool:
        """
        Get whether the rigid body is fixed.
        """
        return self._is_fixed

    # ==================== Static Methods: Finding Rigid Bodies ====================

    @staticmethod
    def is_rigid_body(prim: Usd.Prim) -> bool:
        """
        Check if a prim should be regarded as a rigid body.

        Note: We regard CollisionAPI also as rigid body (they are fixed rigid body).

        Parameters
        ----------
        prim : Usd.Prim
            The prim to check.

        Returns
        -------
        bool
            True if the prim should be regarded as a rigid body, False otherwise.
        """
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            return False

        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            return True

        if prim.HasAPI(UsdPhysics.CollisionAPI):
            return True

        return False

    @staticmethod
    def find_all_rigid_bodies(stage: Usd.Stage, context: UsdParserContext = None) -> List[Usd.Prim]:
        """
        Find all top-most rigid body prims that are not part of an articulation.
        Only finds the head of each rigid body subtree (prims with RigidBodyAPI or CollisionAPI),
        and skips descendants to avoid recursive finding.

        Parameters
        ----------
        stage : Usd.Stage
            The USD stage.
        context : UsdParserContext, optional
            If provided, rigid body top prims will be added to the context.

        Returns
        -------
        List[Usd.Prim]
            List of top-most rigid body prims.
        """
        rigid_bodies = []

        # Use Usd.PrimRange for traversal
        it = iter(Usd.PrimRange(stage.GetPseudoRoot()))
        for prim in it:
            # Early break if we come across an ArticulationRootAPI (don't go deeper)
            if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
                it.PruneChildren()
                continue

            # Early break if we come across a rigid body
            if UsdRigidBodyParser.is_rigid_body(prim):
                rigid_bodies.append(prim)
                if context:
                    context.add_rigid_body(prim)
                # Skip descendants (they will be merged, not treated as separate rigid bodies)
                it.PruneChildren()

        return rigid_bodies


# ==================== Genesis-Specific Geometry Collection Functions ====================

# Pattern matching for geometry collection
_visual_pattern = re.compile(r"^(visual|Visual).*")
_collision_pattern = re.compile(r"^(collision|Collision).*")
_all_pattern = re.compile(r"^.*")


def _create_geo_infos(
    context: UsdParserContext, rigid_body_prim: Usd.Prim, pattern: re.Pattern, mesh_type: Literal["mesh", "vmesh"]
) -> List[Dict]:
    """
    Create geometry info dictionaries from a rigid body prim and its children that match the pattern.

    Parameters
    ----------
    context : UsdParserContext
        The parser context.
    rigid_body_prim : Usd.Prim
        The rigid body prim.
    pattern : re.Pattern
        Pattern to match child prim names.
    mesh_type : Literal["mesh", "vmesh"]
        The mesh type to create geometry info for.

    Returns
    -------
    List[Dict]
        List of geometry info dictionaries.
    """
    # if the rigid body itself is a geometry
    geo_infos: List[Dict] = []
    rigid_body_geo_info = create_geo_info_from_prim(context, rigid_body_prim, rigid_body_prim, mesh_type)
    if rigid_body_geo_info is not None:
        geo_infos.append(rigid_body_geo_info)

    # - RigidBody
    #     - Visuals
    #     - Collisions
    # find all direct children of the rigid body that match the pattern
    search_roots: list[Usd.Prim] = []
    for child in rigid_body_prim.GetChildren():
        child: Usd.Prim
        if pattern.match(str(child.GetName())):
            search_roots.append(child)

    for search_root in search_roots:
        geo_infos.extend(create_geo_infos_from_subtree(context, search_root, rigid_body_prim, mesh_type))

    return geo_infos


def _get_visual_geometries(context: UsdParserContext, rigid_body_prim: Usd.Prim) -> List[Dict]:
    """
    Get visual geometries from a rigid body prim.

    Parameters
    ----------
    context : UsdParserContext
        The parser context.
    rigid_body_prim : Usd.Prim
        The rigid body prim.

    Returns
    -------
    List[Dict]
        List of visual geometry info dictionaries.
    """
    if context.vis_mode == "visual":
        vis_geo_infos = _create_geo_infos(context, rigid_body_prim, _visual_pattern, "vmesh")
        if len(vis_geo_infos) == 0:
            # if no visual geometries found, use any pattern to find visual geometries
            gs.logger.warning(
                f"No visual geometries found in {rigid_body_prim.GetPath()}, using any pattern to find visual "
                "geometries."
            )
            vis_geo_infos = _create_geo_infos(context, rigid_body_prim, _all_pattern, "vmesh")
    elif context.vis_mode == "collision":
        vis_geo_infos = _create_geo_infos(context, rigid_body_prim, _collision_pattern, "vmesh")
    else:
        gs.raise_exception(f"Unsupported visualization mode {context.vis_mode}.")
    return vis_geo_infos


def _get_collision_geometries(context: UsdParserContext, rigid_body_prim: Usd.Prim) -> List[Dict]:
    """
    Get collision geometries from a rigid body prim.

    Parameters
    ----------
    context : UsdParserContext
        The parser context.
    rigid_body_prim : Usd.Prim
        The rigid body prim.

    Returns
    -------
    List[Dict]
        List of collision geometry info dictionaries.
    """
    return _create_geo_infos(context, rigid_body_prim, _collision_pattern, "mesh")


# ==================== Helper Functions for Genesis Parsing ====================


def _create_rigid_body_link_info(rigid_body_prim: Usd.Prim, is_fixed: bool) -> Tuple[Dict, Dict]:
    """
    Create link and joint info dictionaries for a rigid body.

    Parameters
    ----------
    rigid_body_prim : Usd.Prim
        The rigid body prim.
    is_fixed : bool
        Whether the rigid body is fixed.

    Returns
    -------
    l_info : dict
        Link info dictionary.
    j_info : dict
        Joint info dictionary.
    """
    Q_w, S = compute_gs_global_transform(rigid_body_prim)
    body_pos = Q_w[:3, 3]
    body_quat = gu.R_to_quat(Q_w[:3, :3])

    if is_fixed:
        joint_type = gs.JOINT_TYPE.FIXED
        n_qs = 0
        n_dofs = 0
        init_qpos = np.zeros(0)
    else:
        joint_type = gs.JOINT_TYPE.FREE
        n_qs = 7
        n_dofs = 6
        init_qpos = np.concatenate([body_pos, body_quat])

    link_name = str(rigid_body_prim.GetPath())

    l_info = dict(
        is_robot=False,
        name=f"{link_name}",
        pos=body_pos,
        quat=body_quat,
        inertial_pos=None,  # we will compute the COM later based on the geometry
        inertial_quat=gu.identity_quat(),
        parent_idx=-1,
    )

    j_info = dict(
        name=f"{link_name}_joint",  # we only have one joint for the rigid body
        n_qs=n_qs,
        n_dofs=n_dofs,
        type=joint_type,
        init_qpos=init_qpos,
    )

    return l_info, j_info


# ==================== Main Parsing Function ====================


def parse_usd_rigid_body(morph: gs.morphs.USD, surface: gs.surfaces.Surface):
    """
    Parse USD rigid body from the given USD file and prim path, returning info structures
    suitable for mesh-style loading (similar to gs.morph.Mesh).

    Returns
    -------
    l_info : dict
        Link info dictionary.
    j_infos : list
        List of joint info dictionaries (single joint for rigid body).
    g_infos : list
        List of geometry info dictionaries.
    """
    # Validate inputs and setup
    scale = getattr(morph, "scale", 1.0)
    if scale != 1.0:
        gs.logger.warning("USD rigid body parsing currently only supports scale=1.0. Scale will be ignored.")
        scale = 1.0

    assert morph.parser_ctx is not None, "USDRigidBody must have a parser context."
    assert morph.prim_path is not None, "USDRigidBody must have a prim path."

    context: UsdParserContext = morph.parser_ctx
    stage: Usd.Stage = context.stage
    rigid_body_prim = stage.GetPrimAtPath(Sdf.Path(morph.prim_path))
    if not rigid_body_prim.IsValid():
        gs.raise_exception(f"Invalid prim path {morph.prim_path} in USD file {morph.file}.")

    rigid_body = UsdRigidBodyParser(stage, rigid_body_prim)

    g_infos = []
    g_infos.extend(_get_visual_geometries(context, rigid_body_prim))
    g_infos.extend(_get_collision_geometries(context, rigid_body_prim))

    l_info, j_info = _create_rigid_body_link_info(rigid_body_prim, rigid_body.is_fixed)
    j_infos = [j_info]

    return l_info, j_infos, g_infos
