"""
USD Parser Context

Context class for tracking materials, articulations, and rigid bodies during USD parsing.
"""

from pxr import Usd, UsdShade
from typing import Set
import genesis as gs
from .usd_parser_utils import bfs_iterator
from typing import Literal


class UsdParserContext:
    """
    A context class for USD Parsing.

    Tracks:
    - Materials: rendering materials parsed from the stage
    - Articulation roots: prims with ArticulationRootAPI
    - Prims in articulation: flattened set of all prims in articulations
    - Rigid body top prims: top-most rigid body prims (including CollisionAPI)
    """

    def __init__(self, stage: Usd.Stage):
        """
        Initialize the parser context.

        Parameters
        ----------
        stage : Usd.Stage
            The USD stage being parsed.
        """
        self._stage = stage
        self._materials: dict[str, tuple[gs.surfaces.Surface, str]] = {}  # material_id -> (material_surface, uv_name)
        self._articulation_root_prims: dict[str, Usd.Prim] = {}  # prim_path -> articulation_root_prim
        self._prims_in_articulation: dict[str, Usd.Prim] = {}  # prim_path -> prim (flattened set)
        self._rigid_body_top_prims: dict[str, Usd.Prim] = {}  # prim_path -> rigid_body_top_prim
        self._vis_mode: Literal["visual", "collision"] = "visual"

    @property
    def stage(self) -> Usd.Stage:
        """Get the USD stage."""
        return self._stage

    @property
    def vis_mode(self) -> Literal["visual", "collision"]:
        """Get the visualization mode."""
        return self._vis_mode

    @property
    def materials(self) -> dict[str, tuple[gs.surfaces.Surface, str]]:
        """
        Get the parsed materials dictionary.

        Returns
        -------
        dict
            Key: material_id (str)
            Value: tuple of (material_surface, uv_name)
        """
        return self._materials

    def find_material(self, mesh_prim: Usd.Prim):
        mesh_material = gs.surfaces.Default()
        if mesh_prim.HasRelationship("material:binding"):
            if not mesh_prim.HasAPI(UsdShade.MaterialBindingAPI):
                UsdShade.MaterialBindingAPI.Apply(mesh_prim)
            prim_bindings = UsdShade.MaterialBindingAPI(mesh_prim)
            material_usd = prim_bindings.ComputeBoundMaterial()[0]
            if material_usd.GetPrim().IsValid():
                material_spec = material_usd.GetPrim().GetPrimStack()[-1]
                material_id = material_spec.layer.identifier + material_spec.path.pathString
                material_result = self._materials.get(material_id)
                if material_result is not None:
                    mesh_material, _ = material_result
        return mesh_material

    @property
    def articulation_root_prim(self) -> dict[str, Usd.Prim]:
        """
        Get the articulation root prims dictionary.

        Returns
        -------
        dict
            Key: prim_path (str)
            Value: articulation_root_prim
        """
        return self._articulation_root_prims

    @property
    def prims_in_articulation(self) -> dict[str, Usd.Prim]:
        """
        Get the flattened set of all prims in articulations.

        Returns
        -------
        dict
            Key: prim_path (str)
            Value: prim
        """
        return self._prims_in_articulation

    @property
    def rigid_body_top_prims(self) -> dict[str, Usd.Prim]:
        """
        Get the top-most rigid body prims dictionary.

        Returns
        -------
        dict
            Key: prim_path (str)
            Value: rigid_body_top_prim
        """
        return self._rigid_body_top_prims

    def add_articulation_root(self, prim: Usd.Prim):
        """
        Add an articulation root prim and flatten all its descendants.

        Parameters
        ----------
        prim : Usd.Prim
            The articulation root prim to add.
        """
        self._articulation_root_prims[prim.GetPath()] = prim
        # Flatten the prims in articulation to a dictionary (include the root prim itself)
        self._prims_in_articulation[prim.GetPath()] = prim
        for prim_in_articulation in bfs_iterator(prim):
            if prim_in_articulation != prim:  # Don't duplicate the root
                self._prims_in_articulation[prim_in_articulation.GetPath()] = prim_in_articulation

    def is_prim_in_articulation(self, prim: Usd.Prim) -> bool:
        """
        Check if a prim is in an articulation.

        Parameters
        ----------
        prim : Usd.Prim
            The prim to check.

        Returns
        -------
        bool
            True if the prim is in an articulation, False otherwise.
        """
        return prim.GetPath() in self._prims_in_articulation

    def get_material(self, material_id: str):
        """
        Get a parsed material by its ID.

        Parameters
        ----------
        material_id : str
            The material ID.

        Returns
        -------
        tuple or None
            Tuple of (material_surface, uv_name) if found, None otherwise.
        """
        return self._materials.get(material_id)
