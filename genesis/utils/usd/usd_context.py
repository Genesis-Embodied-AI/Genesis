"""
USD Parser Context

Context class for tracking materials, articulations, and rigid bodies during USD parsing.
"""

from pathlib import Path
import shutil
import os
import io
import subprocess
import logging

import numpy as np
from pxr import Usd, UsdShade, UsdPhysics, UsdGeom, Sdf

import genesis as gs
import genesis.utils.mesh as mu

from .usd_material import parse_material_preview_surface
from .usd_parser_utils import extract_scale


def decompress_usdz(usdz_path: str):
    usdz_folder = mu.get_usd_zip_path(usdz_path)

    # The first file in the package must be a native usd file.
    # See https://openusd.org/docs/Usdz-File-Format-Specification.html
    zip_files = Usd.ZipFile.Open(usdz_path)
    zip_filelist = zip_files.GetFileNames()
    root_file = zip_filelist[0]
    if not root_file.lower().endswith(gs.options.morphs.USD_FORMATS[:-1]):
        gs.raise_exception(f"Invalid usdz root file: {root_file}")
    root_path = os.path.join(usdz_folder, root_file)

    if not os.path.exists(root_path):
        for file_name in zip_filelist:
            file_data = io.BytesIO(zip_files.GetFile(file_name))
            file_path = os.path.join(usdz_folder, file_name)
            file_folder = os.path.dirname(file_path)
            os.makedirs(file_folder, exist_ok=True)
            with open(file_path, "wb") as out:
                out.write(file_data.read())
        gs.logger.warning(f"USDZ file {usdz_path} decompressed to {root_path}.")
    else:
        gs.logger.info(f"Decompressed assets detected and used: {root_path}.")
    return root_path


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
        self._material_properties: dict[str, tuple[dict, str]] = {}  # material_id -> (material_dict, uv_name)
        self._material_parsed = False
        self._xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
        self._is_yup = UsdGeom.GetStageUpAxis(self._stage) == "Y"
        self._meter_scale = UsdGeom.GetStageMetersPerUnit(self._stage)

    @property
    def stage(self) -> Usd.Stage:
        """Get the USD stage."""
        return self._stage

    @property
    def stage_file(self) -> str:
        """Get the USD stage file."""
        return self._stage_file

    def get_prim_id(self, prim: Usd.Prim):
        """Get a unique identifier for a prim based on its first SpecifierOver spec."""
        prim_stack = prim.GetPrimStack()
        spec = next((s for s in prim_stack if s.specifier == Sdf.SpecifierOver), prim_stack[-1])
        spec_path = self._stage_file if spec.layer.identifier == self._bake_stage_file else spec.layer.identifier
        return spec_path + spec.path.pathString

    def find_all_rigid_entities(self) -> list[Usd.Prim]:
        """
        Find all prims with ArticulationRootAPI and RigidBody in the stage.
        """
        entity_prims = []
        stage_iter = iter(Usd.PrimRange(self._stage.GetPseudoRoot()))
        for prim in stage_iter:
            # Early break if we come across an ArticulationRootAPI (don't go deeper)
            if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
                entity_prims.append(prim)
                stage_iter.PruneChildren()
            elif prim.HasAPI(UsdPhysics.RigidBodyAPI) or prim.HasAPI(UsdPhysics.CollisionAPI):
                entity_prims.append(prim)
                stage_iter.PruneChildren()

        return entity_prims

    def find_all_materials(self):
        """
        Parse all materials in the USD stage.
        """
        if self._material_parsed:
            return

        bake_material_paths: dict[str, str] = {}  # material_id -> bake_material_path

        # parse materials
        bound_prims = []
        for prim in self._stage.Traverse():
            if prim.IsA(UsdGeom.Gprim) or prim.IsA(UsdGeom.Subset):
                bound_prims.append(prim)
        materials = UsdShade.MaterialBindingAPI.ComputeBoundMaterials(bound_prims)[0]
        for material in materials:
            material_prim = material.GetPrim()
            if material_prim.IsValid():
                # TODO: material_id is reserved for group_by_material option.
                material_id = self.get_prim_id(material_prim)
                if material_id not in self._material_properties:
                    material_dict, uv_name = parse_material_preview_surface(material)
                    self._material_properties[material_id] = material_dict, uv_name
                    if self._need_bake and material_dict is None:
                        bake_material_paths[material_id] = str(material_prim.GetPath())
        self._material_parsed = True

        if not bake_material_paths:
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
            *bake_material_paths.values(),
            "--device",
            str(device.index if device.index is not None else 0),
            "--log_level",
            logging.getLevelName(gs.logger.level).lower(),
        ]
        gs.logger.debug(f"Execute: {' '.join(commands)}")

        print(' '.join(commands))

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
            for bake_material_id, bake_material_path in bake_material_paths.items():
                bake_material_usd = UsdShade.Material(self._stage.GetPrimAtPath(bake_material_path))
                bake_material_dict, uv_name = parse_material_preview_surface(bake_material_usd)
                self._material_properties[bake_material_id] = bake_material_dict, uv_name
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

    def apply_surface(self, surface_id: str, surface: gs.surfaces.Surface):
        applied_surface = surface.copy()
        surface_dict, uv_name = self._material_properties.get(surface_id, (None, "st"))
        if surface_dict is not None:
            applied_surface.update_texture(
                color_texture=surface_dict.get("color_texture"),
                opacity_texture=surface_dict.get("opacity_texture"),
                roughness_texture=surface_dict.get("roughness_texture"),
                metallic_texture=surface_dict.get("metallic_texture"),
                normal_texture=surface_dict.get("normal_texture"),
                emissive_texture=surface_dict.get("emissive_texture"),
                ior=surface_dict.get("ior"),
            )
        return applied_surface, uv_name

    def compute_transform(self, prim: Usd.Prim) -> np.ndarray:
        transform = self._xform_cache.GetLocalToWorldTransform(prim)
        T_usd = np.asarray(transform, dtype=np.float32)  # translation on the bottom row
        if self._is_yup:
            T_usd @= mu.Y_UP_TRANSFORM
        T_usd[:, :3] *= self._meter_scale
        return T_usd.transpose()

    def compute_gs_transform(self, prim: Usd.Prim, ref_prim: Usd.Prim = None):
        Q, S = extract_scale(self.compute_transform(prim))

        if ref_prim is None:
            return Q, S

        Q_ref, S_ref = self.compute_gs_transform(ref_prim)
        Q_rel = np.linalg.inv(Q_ref) @ Q
        return Q_rel, S
