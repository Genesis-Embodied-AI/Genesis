"""
USD Stage

Main parser entrance for importing USD stages into Genesis scenes.
Provides the parse pipeline: materials -> articulations -> rigid bodies.
"""

from typing import List
import os
import io

import genesis as gs
import genesis.utils.mesh as mu

try:
    from pxr import Usd, Sdf
except ImportError:
    gs.raise_exception("pxr module not found. Please install it with `pip install genesis[usd]`.")

from .usd_context import UsdContext


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


def parse_usd_stage(stage: gs.morphs.USDStage, bake_cache=True) -> List[gs.morphs.FileMorph]:
    # create context
    context = UsdContext(stage.file, bake_cache)

    # parse all rendering materials
    context.find_all_materials()
    gs.logger.info(f"Parsed {len(context.materials)} materials from USD stage.")

    # find all articulations and rigid bodies
    context.find_all_rigids()
    gs.logger.info(
        f"Found {len(context.articulation_roots)} articulation(s) and {len(context.rigid_bodies)} rigid body(ies) in USD stage."
    )

    morphs: List[gs.morphs.FileMorph] = []
    stage_params = stage.model_dump()
    for articulation_root in context.articulation_roots:
        articulation_params = stage_params | {
            "parser_ctx": context,
            "prim_path": str(articulation_root.GetPath()),
        }
        articulation_morph = gs.morphs.USDArticulation(**articulation_params)
        morphs.append(articulation_morph)
        gs.logger.info(f"Imported articulation from prim: {articulation_root.GetPath()}")

    for rigid_body_prim in context.rigid_bodies:
        rigid_body_params = stage_params | {
            "parser_ctx": context,
            "prim_path": str(rigid_body_prim.GetPath()),
        }
        rigid_body_morph = gs.morphs.USDRigidBody(**rigid_body_params)
        morphs.append(rigid_body_morph)
        gs.logger.info(f"Imported rigid body from prim: {rigid_body_prim.GetPath()}")

    if not morphs:
        gs.logger.warning("No articulations or rigid bodies found in USD stage.")

    return morphs
