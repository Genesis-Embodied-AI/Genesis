"""
USD Rigid Body Parser

Parser for extracting rigid body information from USD stages.
The parser is agnostic to genesis structures, focusing only on USD rigid body structure.

Also includes Genesis-specific parsing functions that translate USD structures into Genesis info structures.
"""

import re
from typing import Dict, List, Literal, Tuple

import numpy as np
from pxr import Sdf, Usd, UsdGeom, UsdPhysics, UsdShade
from scipy.spatial.transform import Rotation as R

import genesis as gs

from .. import geom as gu
from .. import mesh as mu
from .usd_articulation_parser import UsdArticulationParser
from .usd_geo_adapter import UsdGeometryAdapter, create_geo_info_from_prim, create_geo_infos_from_prim_tree
from .usd_parser_context import UsdParserContext
from .usd_parser_utils import bfs_iterator, compute_gs_global_transform, compute_gs_relative_transform


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

        is_rigid_body = UsdRigidBodyParser.regard_as_rigid_body(rigid_body_prim)
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
            kinematic_enabled = (
                rigid_body_api.GetKinematicEnabledAttr().Get()
                if rigid_body_api.GetKinematicEnabledAttr().Get()
                else False
            )
        self.is_fixed = collision_api_only or kinematic_enabled

    # ==================== Properties ====================

    @property
    def stage(self) -> Usd.Stage:
        """Get the USD stage."""
        return self._stage

    @property
    def rigid_body_prim(self) -> Usd.Prim:
        """Get the rigid body prim."""
        return self._root

    # ==================== Geometry Collection Methods ====================

    visual_pattern = re.compile(r"^(visual|Visual).*")
    collision_pattern = re.compile(r"^(collision|Collision).*")
    all_pattern = re.compile(r"^.*")

    @staticmethod
    def is_geo_prim(prim: Usd.Prim) -> bool:
        return any(prim.IsA(geo_type) for geo_type in UsdGeometryAdapter.SupportedUsdGeoms)

    @staticmethod
    def create_gs_geo_infos(
        context: UsdParserContext, rigid_body_prim: Usd.Prim, pattern, mesh_type: Literal["mesh", "vmesh"]
    ) -> List[Dict]:
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
            if pattern.match(child.GetName()):
                search_roots.append(child)

        for search_root in search_roots:
            geo_infos.extend(create_geo_infos_from_prim_tree(context, search_root, rigid_body_prim, mesh_type))

        return geo_infos

    def get_visual_geometries(self, context: UsdParserContext) -> List[Dict]:
        vis_geo_infos = UsdRigidBodyParser.create_gs_geo_infos(
            context, self._root, UsdRigidBodyParser.visual_pattern, "vmesh"
        )
        if len(vis_geo_infos) == 0:
            # if no visual geometries found, use any pattern to find visual geometries
            gs.logger.info(
                f"No visual geometries found, using any pattern to find visual geometries in {self._root.GetPath()}"
            )
            vis_geo_infos = UsdRigidBodyParser.create_gs_geo_infos(
                context, self._root, UsdRigidBodyParser.all_pattern, "vmesh"
            )
        return vis_geo_infos

    def get_collision_geometries(self, context: UsdParserContext) -> List[Dict]:
        col_geo_infos = UsdRigidBodyParser.create_gs_geo_infos(
            context, self._root, UsdRigidBodyParser.collision_pattern, "mesh"
        )
        return col_geo_infos

    def get_visual_and_collision_geometries(self, context: UsdParserContext) -> tuple[List[Dict], List[Dict]]:
        """
        Get visual and collision geometries.

        Parameters
        ----------
        context : UsdParserContext
            The parser context.

        Returns
        -------
        visual_g_infos : List[Dict]
            List of visual geometry info dictionaries.
        collision_g_infos : List[Dict]
            List of collision geometry info dictionaries.
        """
        visual_g_infos = self.get_visual_geometries(context)
        collision_g_infos = self.get_collision_geometries(context)

        return visual_g_infos, collision_g_infos

    # ==================== Static Methods: Finding Rigid Bodies ====================

    @staticmethod
    def regard_as_rigid_body(prim: Usd.Prim) -> bool:
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
        # First, collect all articulation roots and their descendants
        articulation_roots = UsdArticulationParser.find_all_articulation_roots(stage, context)
        articulation_descendants = set()
        for root in articulation_roots:
            for prim in bfs_iterator(root):
                articulation_descendants.add(prim.GetPath())

        # Track rigid body subtrees to skip descendants
        rigid_body_descendants = set()

        # Find all rigid bodies that are not part of any articulation
        rigid_bodies = []

        def should_continue(prim: Usd.Prim) -> bool:
            # Don't process children if already part of a rigid body subtree or articulation
            return prim.GetPath() not in rigid_body_descendants and prim.GetPath() not in articulation_descendants

        for prim in bfs_iterator(stage.GetPseudoRoot(), should_continue=should_continue):
            # Skip if already part of a rigid body subtree
            if prim.GetPath() in rigid_body_descendants:
                continue

            # Check for RigidBodyAPI or CollisionAPI
            if UsdRigidBodyParser.regard_as_rigid_body(prim):
                # Check if this prim is a descendant of any articulation
                is_descendant = False
                current_prim = prim
                while current_prim and current_prim != stage.GetPseudoRoot():
                    if current_prim.GetPath() in articulation_descendants:
                        is_descendant = True
                        break
                    current_prim = current_prim.GetParent()

                if not is_descendant:
                    rigid_bodies.append(prim)
                    if context:
                        context._rigid_body_top_prims[prim.GetPath()] = prim
                    # Mark all descendants of this rigid body as part of its subtree
                    # (they will be merged, not treated as separate rigid bodies)
                    for descendant in bfs_iterator(prim):
                        if descendant != prim:  # Don't mark the head itself
                            rigid_body_descendants.add(descendant.GetPath())

        return rigid_bodies


# ==================== Helper Functions for Genesis Parsing ====================


def _finalize_geometry_info(g_info: Dict, morph: gs.morphs.USD, is_visual: bool) -> Dict:
    """
    Finalize geometry info dictionary by adding parser-specific fields.

    Parameters
    ----------
    g_info : dict
        Geometry info dictionary from adapter.
    morph : gs.morphs.USDRigidBody
        The rigid body morph (for visualization/collision flags).
    is_visual : bool
        Whether this is a visual geometry.

    Returns
    -------
    dict or None
        Finalized geometry info dictionary, or None if should be skipped.
    """
    visualization = getattr(morph, "visualization", True)
    collision = getattr(morph, "collision", True)

    # Return None if geometry should not be added
    if is_visual and not visualization:
        return None
    if not is_visual and not collision:
        return None

    # Add solver params if not present (for collision geometries)
    if not is_visual and "sol_params" not in g_info:
        g_info["sol_params"] = gu.default_solver_params()

    # Override contype/conaffinity for collision if specified in morph
    if not is_visual:
        g_info["contype"] = getattr(morph, "contype", g_info.get("contype", 1))
        g_info["conaffinity"] = getattr(morph, "conaffinity", g_info.get("conaffinity", 1))

    return g_info


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
    # Get the global position and quaternion of the rigid body
    Q_w, S = compute_gs_global_transform(rigid_body_prim)
    body_pos = Q_w[:3, 3]
    body_quat = gu.R_to_quat(Q_w[:3, :3])

    # Determine joint type and init_qpos
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

    # Generate link name from prim path
    link_name = str(rigid_body_prim.GetPath())

    # Create link info
    l_info = dict(
        is_robot=False,
        name=f"{link_name}",
        pos=body_pos,
        quat=body_quat,
        inertial_pos=None,  # we will compute the COM later based on the geometry
        inertial_quat=gu.identity_quat(),
        parent_idx=-1,
    )

    # Create joint info
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
    assert rigid_body_prim.IsValid(), f"Invalid prim path {morph.prim_path} in USD file {morph.file}."

    # Create parser for USD-agnostic extraction
    rigid_body = UsdRigidBodyParser(stage, rigid_body_prim)

    # Build geometry infos using BatchedUsdGeometryAdapter
    g_infos = []

    # Get visual and collision geometries
    visual_g_infos, collision_g_infos = rigid_body.get_visual_and_collision_geometries(context)

    # Finalize and add visual geometries
    for g_info in visual_g_infos:
        finalized = _finalize_geometry_info(g_info, morph, is_visual=True)
        if finalized:
            g_infos.append(finalized)

    # Finalize and add collision geometries
    for g_info in collision_g_infos:
        finalized = _finalize_geometry_info(g_info, morph, is_visual=False)
        if finalized:
            g_infos.append(finalized)

    # Create link and joint info
    l_info, j_info = _create_rigid_body_link_info(rigid_body_prim, rigid_body.is_fixed)
    j_infos = [j_info]

    return l_info, j_infos, g_infos
