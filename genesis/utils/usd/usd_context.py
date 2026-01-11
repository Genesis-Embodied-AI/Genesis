"""
USD Parser Context

Context class for tracking materials, articulations, and rigid bodies during USD parsing.
"""

from collections import deque
from typing import Set
from pathlib import Path
import shutil
import os
import subprocess
import logging

from pxr import Usd, UsdShade, UsdPhysics, Sdf

import genesis as gs
import genesis.utils.mesh as mu

from .usd_parser_utils import bfs_iterator
from .usd_material import parse_material_preview_surface
from .usd_stage import decompress_usdz


class UsdContext:
    """
    A context class for USD Parsing.

    Tracks:
    - Materials: rendering materials parsed from the stage
    - Articulation roots: prims with ArticulationRootAPI
    - Prims in articulation: flattened set of all prims in articulations
    - Rigid body top prims: top-most rigid body prims (including CollisionAPI)
    """

    def __init__(self, stage_file: str, bake_cache: bool = True):
        """
        Initialize the parser context.

        Parameters
        ----------
        stage : Usd.Stage
            The USD stage being parsed.
        """
        # decompress usdz
        if stage_file.lower().endswith(gs.options.morphs.USD_FORMATS[-1]):
            stage_file = decompress_usdz(stage_file)

        # detect bake file caches
        if bake_cache:
            self._need_bake = True
            self._bake_folder = mu.get_usd_bake_path(stage_file)
            self._bake_stage_file = os.path.join(self._bake_folder, os.path.basename(stage_file))
            if os.path.exists(self._bake_stage_file):
                self._need_bake = False
                gs.logger.info(f"Baked assets detected and used: {self._bake_stage_file}")
                stage_file = self._bake_stage_file
        else:
            self._need_bake = False
            self._bake_folder = None
            self._bake_stage_file = None

        self._stage_file = stage_file
        self._stage = Usd.Stage.Open(self._stage_file)
        self._articulation_roots: list[Usd.Prim] = []  # prim_path -> articulation_root_prim
        self._rigid_bodies: list[Usd.Prim] = []  # prim_path -> rigid_body_top_prim
        self._material_preview_surfaces: dict[str, tuple[dict, str]] = {}  # material_id -> (material_dict, uv_name)
        self._bake_materials: dict[str, str] = {}  # material_id -> bake_material_path

    @property
    def stage(self) -> Usd.Stage:
        """Get the USD stage."""
        return self._stage

    # @property
    # def vis_mode(self) -> Literal["visual", "collision"]:
    #     """Get the visualization mode."""
    #     return self._vis_mode

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

    @property
    def articulation_roots(self) -> list[Usd.Prim]:
        """
        Get the articulation root prims.
        """
        return self._articulation_roots

    @property
    def rigid_bodies(self) -> list[Usd.Prim]:
        """
        Get the rigid body prims.
        """
        return self._rigid_bodies

    def get_prim_id(self, prim: Usd.Prim):
        """Get a unique identifier for a prim based on its first SpecifierOver spec."""
        prim_stack = prim.GetPrimStack()
        spec = next((s for s in prim_stack if s.specifier == Sdf.SpecifierOver), prim_stack[-1])
        spec_path = self._stage_file if spec.layer.identifier == self._bake_stage_file else spec.layer.identifier
        return spec_path + spec.path.pathString

    def find_all_rigids(self):
        """
        Find all prims with ArticulationRootAPI and RigidBody in the stage.
        """
        queue = deque([self._stage.GetPseudoRoot()])
        while queue:
            prim = queue.popleft()
            if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
                self._articulation_roots.append(prim)
            elif prim.HasAPI(UsdPhysics.RigidBodyAPI) or prim.HasAPI(UsdPhysics.CollisionAPI):
                self._rigid_bodies.append(prim)
            else:
                for child in prim.GetChildren():
                    queue.append(child)

    def find_all_materials(self):
        """
        Parse all materials in the USD stage.
        """
        # parse materials
        for prim in self._stage.Traverse():
            if prim.IsA(UsdShade.Material):
                material_usd = UsdShade.Material(prim)
                material_id = self.get_prim_id(prim)
                material_dict, uv_name = parse_material_preview_surface(material_usd)
                self._material_preview_surfaces[material_id] = material_dict, uv_name
                if self._need_bake and material_dict is None:
                    self._bake_materials[material_id] = str(prim.GetPath())

        if not self._bake_materials:
            return

        device = gs.device
        if device.type == "cpu":
            try:
                device, *_ = gs.utils.get_device(gs.cuda)
            except gs.GenesisException as e:
                gs.raise_exception_from("USD baking requires CUDA GPU.", e)

        self.replace_asset_symlinks()
        os.makedirs(self._bake_folder, exist_ok=True)

        # Note that it is necessary to call 'bake_usd_material' via a subprocess to ensure proper isolation of
        # omninerse kit, otherwise the global conversion registry of some Python bindings will be conflicting between
        # each, ultimately leading to segfault...
        commands = [
            "python",
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "usd_bake.py"),
            "--input_file",
            self._stage_file,
            "--output_dir",
            self._bake_folder,
            "--usd_material_paths",
            *self._bake_materials.values(),
            "--device",
            str(device.index if device.index is not None else 0),
            "--log_level",
            logging.getLevelName(gs.logger.level).lower(),
        ]
        gs.logger.debug(f"Execute: {' '.join(commands)}")

        try:
            result = subprocess.run(
                commands,
                capture_output=True,
                check=True,
                text=True,
            )
            if result.stdout:
                gs.logger.debug(result.stdout)
            if result.stderr:
                gs.logger.warning(result.stderr)
        except (subprocess.CalledProcessError, OSError) as e:
            gs.logger.warning(f"Baking process failed: {e} (Note that USD baking may only support Python 3.10 now.)")

        if os.path.exists(self._bake_stage_file):
            gs.logger.warning(f"USD materials baked to file {self._bake_stage_file}")
            self._stage = Usd.Stage.Open(self._bake_stage_file)
            for bake_material_id, bake_material_path in self._bake_materials.items():
                bake_material_usd = UsdShade.Material(self._stage.GetPrimAtPath(bake_material_path))
                bake_material_dict, uv_name = parse_material_preview_surface(bake_material_usd)
                self._material_preview_surfaces[bake_material_id] = bake_material_dict, uv_name
            for baked_texture_obj in Path(self._bake_folder).glob("baked_textures*"):
                shutil.rmtree(baked_texture_obj)

    def replace_asset_symlinks(self):
        asset_paths = set()

        for prim in self._stage.TraverseAll():
            for attr in prim.GetAttributes():
                value = attr.Get()
                if isinstance(value, Sdf.AssetPath):
                    asset_paths.add(value.resolvedPath)
                elif isinstance(value, list):
                    for v in value:
                        if isinstance(v, Sdf.AssetPath):
                            asset_paths.add(v.resolvedPath)

        for asset_path in map(Path, asset_paths):
            if not asset_path.is_symlink():
                continue

            real_path = asset_path.resolve()
            if asset_path.suffix.lower() == real_path.suffix.lower():
                continue

            asset_path.unlink()
            if real_path.is_file():
                gs.logger.warning(f"Replacing symlink {asset_path} with real file {real_path}.")
                shutil.copy2(real_path, asset_path)

    def apply_surface(self, material_id: str, surface: gs.surfaces.Surface):
        material_surface = surface.copy()
        material_dict, uv_name = self._material_preview_surfaces.get(material_id, (None, "st"))
        if material_dict is not None:
            material_surface.update_texture(
                color_texture=material_dict.get("color_texture"),
                opacity_texture=material_dict.get("opacity_texture"),
                roughness_texture=material_dict.get("roughness_texture"),
                metallic_texture=material_dict.get("metallic_texture"),
                normal_texture=material_dict.get("normal_texture"),
                emissive_texture=material_dict.get("emissive_texture"),
                ior=material_dict.get("ior"),
            )
        return material_surface, uv_name

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
