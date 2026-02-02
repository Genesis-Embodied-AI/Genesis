import io
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from pxr import Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

import genesis as gs
import genesis.utils.mesh as mu

from .usd_material import parse_material_preview_surface
from .usd_utils import extract_scale

# Check for Omniverse Kit support (required for USD baking)
# Note: CI workflows should set OMNI_KIT_ACCEPT_EULA=yes in their env section
try:
    import omni.kit_app

    HAS_OMNIVERSE_KIT_SUPPORT = True
except ImportError:
    HAS_OMNIVERSE_KIT_SUPPORT = False


def decompress_usdz(usdz_path: str):
    usdz_folder = mu.get_usd_zip_path(usdz_path)

    # The first file in the package must be a native usd file.
    # See https://openusd.org/docs/Usdz-File-Format-Specification.html
    zip_files = Sdf.ZipFile.Open(usdz_path)
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
    Context manager for USD stage parsing and material processing.

    This class provides a centralized context for parsing USD files, managing materials,
    computing transforms, and handling asset preprocessing. It supports USDZ decompression,
    material baking, coordinate system conversion, and asset symlink resolution.

    Parameters
    ----------
    stage_file : str
        Path to the USD stage file (.usd, .usda, .usdc) or USDZ archive (.usdz).
        If a USDZ file is provided, it will be automatically decompressed.
    usd_bake_cache : bool, optional
        If True, enables material baking and uses last time baked assets if available.
        Otherwise, will re-bake materials every time.
        Default is True.

    Notes
    -----
    - USDZ files are automatically decompressed to a temporary directory
    - The stage's up-axis and meter scale are detected and stored for transform computations
    - Material parsing is lazy (only when find_all_materials() is called)
    """

    def __init__(self, stage_file: str, use_bake_cache: bool = True):
        # decompress usdz
        if stage_file.lower().endswith(gs.options.morphs.USD_FORMATS[-1]):
            stage_file = decompress_usdz(stage_file)

        # detect if baking is needed
        self._need_bake = HAS_OMNIVERSE_KIT_SUPPORT
        if HAS_OMNIVERSE_KIT_SUPPORT:
            if not torch.cuda.is_available():
                gs.logger.warning("USD baking requires CUDA GPU. USD baking will be disabled.")
                self._need_bake = False
        else:
            gs.logger.warning(
                "omniverse-kit not found. USD baking will be disabled. "
                "Please install it with `pip install --extra-index-url https://pypi.nvidia.com omniverse-kit`. "
                "See https://genesis-world.readthedocs.io/en/latest/user_guide/getting_started/usd_import.html."
            )

        # detect bake file caches
        self._bake_folder = mu.get_usd_bake_path(stage_file)
        self._bake_stage_file = os.path.join(self._bake_folder, os.path.basename(stage_file))
        if use_bake_cache:
            if os.path.exists(self._bake_stage_file):
                self._need_bake = False
                gs.logger.info(f"Baked assets detected and used: {self._bake_stage_file}")
                stage_file = self._bake_stage_file
        else:
            if os.path.exists(self._bake_stage_file):
                shutil.rmtree(self._bake_folder)

        self._stage_file = stage_file
        self._stage = Usd.Stage.Open(self._stage_file)
        self._material_properties: dict[str, tuple[dict, str]] = {}  # material_id -> (material_dict, uv_name)
        self._material_parsed = False
        self._bake_material_paths: dict[str, str] = {}  # material_id -> bake_material_path
        self._prim_material_bindings: dict[str, str] = {}  # prim_path -> material_path
        self._xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
        self._is_yup = UsdGeom.GetStageUpAxis(self._stage) == "Y"
        self._meter_scale = UsdGeom.GetStageMetersPerUnit(self._stage)

    @property
    def stage(self) -> Usd.Stage:
        """
        Get the USD stage object.
        """
        return self._stage

    @property
    def stage_file(self) -> str:
        """
        Get the path to the USD stage file.
        """
        return self._stage_file

    def get_prim_id(self, prim: Usd.Prim) -> str:
        """
        Get a unique identifier for a prim based on its layer specification.

        The identifier is constructed from the layer file path and the prim's path
        string. This ensures uniqueness even when the same prim appears in multiple
        layers or when using baked stages.
        """
        prim_stack = prim.GetPrimStack()
        spec = next((s for s in prim_stack if s.specifier == Sdf.SpecifierOver), prim_stack[-1])
        spec_path = self._stage_file if spec.layer.identifier == self._bake_stage_file else spec.layer.identifier
        return spec_path + spec.path.pathString

    def get_binding_material(self, prim: Usd.Prim) -> UsdShade.Material | None:
        """
        Get the material bound to a geometry prim.
        """
        prim_path = str(prim.GetPath())
        if prim_path in self._prim_material_bindings:
            return UsdShade.Material(self._stage.GetPrimAtPath(self._prim_material_bindings[prim_path]))
        return None

    def compute_transform(self, prim: Usd.Prim) -> np.ndarray:
        """
        Compute the local-to-world transformation matrix for a prim.
        """
        transform = self._xform_cache.GetLocalToWorldTransform(prim)
        T_usd = np.asarray(transform, dtype=np.float32)  # translation on the bottom row
        if self._is_yup:
            T_usd @= mu.Y_UP_TRANSFORM
        T_usd[:, :3] *= self._meter_scale
        return T_usd.transpose()

    def compute_gs_transform(self, prim: Usd.Prim, ref_prim: Usd.Prim = None) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute the Genesis transform (pose and scale) for a prim.
        """
        Q, S = extract_scale(self.compute_transform(prim))
        if ref_prim is None:
            return Q, S

        Q_ref, S_ref = self.compute_gs_transform(ref_prim)
        Q_rel = np.linalg.inv(Q_ref) @ Q
        return Q_rel, S

    def apply_surface(self, geom_prim: Usd.Prim, surface: gs.surfaces.Surface):
        """
        Apply material properties from USD to a Genesis surface object.
        """
        geom_path = str(geom_prim.GetPath())
        applied_surface = surface.copy()

        if geom_path in self._prim_material_bindings:
            surface_id = self._prim_material_bindings[geom_path]
            surface_dict, uv_name = self._material_properties.get(surface_id, ({}, "st"))
            # accepted keys: color_texture, opacity_texture, roughness_texture, metallic_texture, normal_texture, emissive_texture, ior
            applied_surface.update_texture(**surface_dict)
            if surface_id in self._bake_material_paths:
                bake_success = True if surface_dict else False
            else:
                bake_success = None
        else:
            uv_name, surface_id = "st", None
            bake_success = None

        return applied_surface, uv_name, surface_id, bake_success

    def find_all_rigid_entities(self) -> list[Usd.Prim]:
        """
        Find all rigid body entities in the USD stage.
        """
        entity_prims = []
        stage_iter = iter(Usd.PrimRange(self._stage.GetPseudoRoot()))
        for prim in stage_iter:
            if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
                entity_prims.append(prim)
                stage_iter.PruneChildren()
            elif prim.HasAPI(UsdPhysics.RigidBodyAPI) or prim.HasAPI(UsdPhysics.CollisionAPI):
                entity_prims.append(prim)
                stage_iter.PruneChildren()
        return entity_prims

    def find_all_materials(self):
        """
        Parse all materials in the USD stage and optionally bake complex materials.
        """
        if self._material_parsed:
            return

        # parse materials
        bound_prims = []
        for prim in self._stage.Traverse():
            if prim.IsA(UsdGeom.Gprim) or prim.IsA(UsdGeom.Subset):
                bound_prims.append(prim)
        materials = UsdShade.MaterialBindingAPI.ComputeBoundMaterials(bound_prims)[0]
        for bound_prim, material in zip(bound_prims, materials):
            geom_path = str(bound_prim.GetPath())
            material_prim = material.GetPrim()

            if material_prim.IsValid():
                # TODO: material_id is also reserved for group_by_material option.
                material_id = self.get_prim_id(material_prim)
                if material_id not in self._material_properties:
                    material_dict, uv_name = parse_material_preview_surface(material)
                    self._material_properties[material_id] = material_dict, uv_name
                    if self._need_bake and not material_dict:
                        self._bake_material_paths[material_id] = str(material_prim.GetPath())
                self._prim_material_bindings[geom_path] = material_id
            else:
                if bound_prim.IsA(UsdGeom.Gprim):
                    gprim = UsdGeom.Gprim(bound_prim)
                    display_colors = np.asarray(gprim.GetDisplayColorPrimvar().Get() or [], dtype=np.float32)
                    if display_colors.size > 0:
                        material_id = self.get_prim_id(bound_prim)
                        color_texture = gs.textures.ColorTexture(color=tuple(display_colors[0]))
                        self._material_properties[material_id] = {"color_texture": color_texture}, "st"
                        self._prim_material_bindings[geom_path] = material_id
        self._material_parsed = True

        if not self._bake_material_paths:
            return

        device = gs.utils.get_device(gs.cuda)[0] if gs.device.type == "cpu" else gs.device
        self.replace_asset_symlinks()
        os.makedirs(self._bake_folder, exist_ok=True)

        # Note that it is necessary to call 'bake_usd_material' as a subprocess to ensure proper isolation of omniverse
        # kit, otherwise the global conversion registry of some Python bindings will be conflicting with each other,
        # ultimately leading to segfault...
        commands = [
            sys.executable,
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "usd_bake.py"),
            "--input_file",
            self._stage_file,
            "--output_dir",
            self._bake_folder,
            "--usd_material_paths",
            *self._bake_material_paths.values(),
            "--device",
            str(device.index if device.index is not None else 0),
            "--log_level",
            logging.getLevelName(gs.logger.level).lower(),
        ]
        gs.logger.debug(f"Execute: {' '.join(commands)}")

        env = dict(os.environ)
        env["OMNI_KIT_ALLOW_ROOT"] = "1"

        try:
            result = subprocess.run(commands, capture_output=True, check=True, text=True, env=env)
            if result.stdout:
                gs.logger.debug(result.stdout)
            if result.stderr:
                gs.logger.warning(result.stderr)
        except (subprocess.CalledProcessError, OSError) as e:
            gs.logger.warning(
                f"Baking process failed: {e} A few possible reasons:"
                "\n\t1. The first launch may require accepting the Omniverse EULA. "
                "Set `OMNI_KIT_ACCEPT_EULA=yes` to accept it automatically."
                "\n\t2. The first launch may install additional dependencies, which can cause a timeout."
                "\n\t3. If you have multiple Python environments (especially with different Python versions), "
                "Omniverse Kit extensions may conflict across environments. Try to remove the shared omniverse "
                "extension folder (e.g. `~/.local/share/ov/data/ext` in Linux) and try again."
            )

        if os.path.exists(self._bake_stage_file):
            gs.logger.warning(f"USD materials baked to file {self._bake_stage_file}")
            self._stage = Usd.Stage.Open(self._bake_stage_file)
            for bake_material_id, bake_material_path in self._bake_material_paths.items():
                bake_material_usd = UsdShade.Material(self._stage.GetPrimAtPath(bake_material_path))
                bake_material_dict, uv_name = parse_material_preview_surface(bake_material_usd)
                self._material_properties[bake_material_id] = bake_material_dict, uv_name
            for baked_texture_obj in Path(self._bake_folder).glob("baked_textures*"):
                shutil.rmtree(baked_texture_obj)

    def replace_asset_symlinks(self):
        """
        Replace asset path symlinks with actual file copies when file extensions differ.

        Some USD assets use symlinks that point to files with different extensions
        (e.g., .png symlink pointing to .exr). This method finds such symlinks and
        replaces them with actual file copies to ensure compatibility.
        """
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
