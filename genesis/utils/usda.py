import io
import os
import shutil
import subprocess
import logging

from PIL import Image
import numpy as np
import trimesh

try:
    from pxr import Usd, UsdGeom, UsdShade, Sdf
except ImportError as e:
    gs.raise_exception_from(
        "Failed to import USD related libs. Please make sure that installing Genesis with [usd] option.", e
    )


import genesis as gs
from . import mesh as mu
from .usda_bake import BAKE_EXT

cs_encode = {
    "raw": "linear",
    "sRGB": "srgb",
    "auto": None,
    "": None,
}

yup_rotation = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])


def get_input_attribute_value(shader, input_name, input_type=None):
    shader_input = shader.GetInput(input_name)

    if input_type != "value":
        if shader_input.GetPrim().IsValid() and shader_input.HasConnectedSource():
            shader_input_connect, shader_input_name = shader_input.GetConnectedSource()[:2]
            return UsdShade.Shader(shader_input_connect.GetPrim()), shader_input_name

    if input_type != "attribute":
        return shader_input.Get(), None
    return None, None


def get_texture_image(image_path, zipfiles):
    if zipfiles is not None:
        image_data = zipfiles.GetFile(image_path.path)
        if image_data is not None:
            return np.asarray(Image.open(io.BytesIO(image_data)))
    return np.asarray(Image.open(image_path.resolvedPath))


def parse_preview_surface(shader, output_name, zipfiles):
    shader_id = shader.GetShaderId()
    if shader_id == "UsdPreviewSurface":
        uvname = None

        def parse_component(component_name, component_encode):
            component, component_output = get_input_attribute_value(shader, component_name)
            if component_output is None:  # constant value
                component_factor = component
                component_image = None
                component_uvname = None
            else:  # texture shader
                component_image, component_overencode, component_uvname = parse_preview_surface(
                    component, component_output, zipfiles
                )
                if component_overencode is not None:
                    component_encode = component_overencode
                component_factor = None

            component_texture = mu.create_texture(component_image, component_factor, component_encode)
            return component_texture, component_uvname

        # parse color
        color_texture, color_uvname = parse_component("diffuseColor", "srgb")
        if color_uvname is not None:
            uvname = color_uvname

        # parse opacity
        opacity_texture, opacity_uvname = parse_component("opacity", "linear")
        if opacity_uvname is not None and uvname is None:
            uvname = opacity_uvname
        if opacity_texture is not None:
            alpha_cutoff = get_input_attribute_value(shader, "opacityThreshold", "value")[0]
            opacity_texture.apply_cutoff(alpha_cutoff)

        # parse emissive
        emissive_texture, emissive_uvname = parse_component("emissiveColor", "srgb")
        if emissive_texture is not None and emissive_texture.is_black():
            emissive_texture = None
        if emissive_uvname is not None and uvname is None:
            uvname = emissive_uvname

        # parse mertalic
        use_specular = get_input_attribute_value(shader, "useSpecularWorkflow", "value")[0]
        if not use_specular:
            metallic_texture, metallic_uvname = parse_component("metallic", "linear")
            if metallic_uvname is not None and uvname is None:
                uvname = metallic_uvname
        else:
            metallic_texture = None

        # parse roughness
        roughness_texture, roughness_uvname = parse_component("roughness", "linear")
        if roughness_uvname is not None and uvname is None:
            uvname = roughness_uvname

        # parse normal
        normal_texture, normal_uvname = parse_component("normal", "linear")
        if normal_uvname is not None and uvname is None:
            uvname = normal_uvname

        # parse ior
        ior = get_input_attribute_value(shader, "ior", "value")[0]

        if uvname is None:
            uvname = "st"
        return {
            "color_texture": color_texture,
            "opacity_texture": opacity_texture,
            "roughness_texture": roughness_texture,
            "metallic_texture": metallic_texture,
            "emissive_texture": emissive_texture,
            "normal_texture": normal_texture,
            "ior": ior,
        }, uvname

    elif shader_id == "UsdUVTexture":
        texture = get_input_attribute_value(shader, "file", "value")[0]
        if texture is not None:
            texture_image = get_texture_image(texture, zipfiles)
            if output_name == "r":
                texture_image = texture_image[:, :, 0]
            elif output_name == "g":
                texture_image = texture_image[:, :, 1]
            elif output_name == "b":
                texture_image = texture_image[:, :, 2]
            elif output_name == "a":
                texture_image = texture_image[:, :, 3]
            elif output_name == "rgb":
                texture_image = texture_image[:, :, :3]
            else:
                gs.raise_exception(f"Invalid output channel for UsdUVTexture: {output_name}.")
        else:
            texture_image = None

        texture_encode = get_input_attribute_value(shader, "sourceColorSpace", "value")[0] or "sRGB"
        texture_encode = cs_encode[texture_encode]
        texture_uvs_shader, texture_uvs_output = get_input_attribute_value(shader, "st", "attribute")
        texture_uvs_name = parse_preview_surface(texture_uvs_shader, texture_uvs_output, zipfiles)

        return texture_image, texture_encode, texture_uvs_name

    elif shader_id.startswith("UsdPrimvarReader"):
        primvar_name = get_input_attribute_value(shader, "varname", "value")[0]
        return primvar_name


def parse_usd_material(material, surface, zipfiles):
    surface_outputs = material.GetSurfaceOutputs()
    material_dict, uv_name = None, None
    material_surface = None

    require_bake = False
    material_candidates = []
    for surface_output in surface_outputs:
        if not surface_output.HasConnectedSource():
            continue
        surface_output_connectable, surface_output_name, _ = surface_output.GetConnectedSource()
        surface_shader = UsdShade.Shader(surface_output_connectable.GetPrim())
        surface_shader_implement = surface_shader.GetImplementationSource()
        surface_shader_id = surface_shader.GetShaderId()

        if surface_shader_implement == "id" and surface_shader_id == "UsdPreviewSurface":
            material_dict, uv_name = parse_preview_surface(surface_shader, surface_output_name, zipfiles)
            require_bake = False
            break

        require_bake = True
        material_candidates.append((surface_shader.GetPath(), surface_shader_id, surface_shader_implement))

    if material_dict is not None:
        material_surface = surface.copy()
        material_surface.update_texture(
            color_texture=material_dict.get("color_texture"),
            opacity_texture=material_dict.get("opacity_texture"),
            roughness_texture=material_dict.get("roughness_texture"),
            metallic_texture=material_dict.get("metallic_texture"),
            normal_texture=material_dict.get("normal_texture"),
            emissive_texture=material_dict.get("emissive_texture"),
            ior=material_dict.get("ior"),
        )

    if require_bake:
        candidates_str = "\n".join(
            [
                f"\tShader at {shader_path} with implement {shader_impl} and ID {shader_id}."
                for shader_path, shader_id, shader_impl in material_candidates
            ]
        )
        gs.logger.warning(f"Material require baking:\n{candidates_str}")
    return material_surface, uv_name, require_bake


def replace_asset_symlinks(stage):
    asset_paths = set()

    for prim in stage.TraverseAll():
        for attr in prim.GetAttributes():
            value = attr.Get()
            if isinstance(value, Sdf.AssetPath):
                asset_paths.add(value.resolvedPath)
            elif isinstance(value, list):
                for v in value:
                    if isinstance(v, Sdf.AssetPath):
                        asset_paths.add(v.resolvedPath)

    for asset_path in asset_paths:
        if not os.path.islink(asset_path):
            return

        real_path = os.path.realpath(asset_path)
        path_ext = os.path.splitext(asset_path)[1]
        real_path_ext = os.path.splitext(real_path)[1]
        if path_ext.lower() == real_path_ext.lower():
            return

        os.unlink(asset_path)
        if os.path.isfile(real_path):
            shutil.copy2(real_path, asset_path)
            gs.logger.warning(f"Replace symlink {asset_path} with real file {real_path}.")


def parse_mesh_usd(path, group_by_material, scale, surface, bake_cache=True):
    zipfiles = Usd.ZipFile.Open(path) if path.endswith(".usdz") else None

    is_bake_cache_found = False
    baked_path = f"{os.path.splitext(path)[0]}_baked.{BAKE_EXT}"
    if bake_cache:
        if os.path.exists(baked_path):
            path = baked_path
            is_bake_cache_found = True

    stage = Usd.Stage.Open(path)
    scale *= UsdGeom.GetStageMetersPerUnit(stage)
    yup = UsdGeom.GetStageUpAxis(stage) == "Y"
    xform_cache = UsdGeom.XformCache()

    mesh_infos = mu.MeshInfoGroup()
    materials = {}
    baked_materials = {}

    # parse materials
    for prim in stage.Traverse():
        if prim.IsA(UsdShade.Material):
            material_usd = UsdShade.Material(prim)
            material_spec = prim.GetPrimStack()[-1]
            material_id = material_spec.layer.identifier + material_spec.path.pathString
            material_pack = materials.get(material_id, None)

            if material_pack is None:
                material, uv_name, require_bake = parse_usd_material(material_usd, surface, zipfiles)
                materials[material_id] = (material, uv_name)
                if not is_bake_cache_found and require_bake:
                    baked_materials[material_id] = material_usd.GetPath()

    if baked_materials:
        baked_material_paths = list(map(str, baked_materials.values()))
        baker_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "usda_bake.py")

        try:
            if gs.device.type == "cpu":
                raise OSError("USD baking does not support backend CPU.")

            replace_asset_symlinks(stage)

            commands = [
                "python",
                baker_file,
                "--usd_file",
                path,
                "--usd_material_paths",
                *baked_material_paths,
                "--device",
                str(gs.device.index),
                "--log_level",
                logging.getLevelName(gs.logger.level).lower(),
            ]

            # Each material is estimated to bake at most 4s, and boostrap and initialization for 10s.
            result = subprocess.run(
                commands,
                timeout=len(baked_material_paths) * 4 + 10,
                capture_output=True,
                check=True,
                text=True,
            )
            if result.stdout:
                gs.logger.debug(result.stdout)
            if result.stderr:
                gs.logger.warning(result.stderr)
        except subprocess.TimeoutExpired:
            gs.logger.warning(f"Timeout. Terminate baking for USD file {path}.")
        except (subprocess.CalledProcessError, OSError) as e:
            gs.logger.warning(f"Baking process failed: {e}. (Note that USD baking may only support Python 3.10 now.)")

        if os.path.exists(baked_path):
            baked_folder = os.path.dirname(path)
            gs.logger.warning(f"USD materials baked to file {baked_path}")
            stage = Usd.Stage.Open(baked_path)
            for baked_material_id, baked_material_path in baked_materials.items():
                baked_material_usd = UsdShade.Material(stage.GetPrimAtPath(baked_material_path))
                baked_material, uv_name, require_bake = parse_usd_material(baked_material_usd, surface, zipfiles)
                materials[baked_material_id] = (baked_material, uv_name)

            if not bake_cache:
                os.remove(baked_path)
                for file in os.listdir(baked_folder):
                    if file.startswith("baked_textures"):
                        shutil.rmtree(os.path.join(baked_folder, file))

    # parse geometries
    for prim in stage.Traverse():
        if prim.HasRelationship("material:binding"):
            if not prim.HasAPI(UsdShade.MaterialBindingAPI):
                UsdShade.MaterialBindingAPI.Apply(prim)
    for i, prim in enumerate(stage.Traverse()):
        if prim.IsA(UsdGeom.Mesh):
            matrix = np.array(xform_cache.GetLocalToWorldTransform(prim))
            if yup:
                matrix[:3, :3] @= yup_rotation
            mesh_usd = UsdGeom.Mesh(prim)
            mesh_spec = prim.GetPrimStack()[-1]
            mesh_id = mesh_spec.layer.identifier + mesh_spec.path.pathString

            if not mesh_usd.GetPointsAttr().HasValue():
                continue
            points = np.array(mesh_usd.GetPointsAttr().Get(), dtype=np.float32)
            faces = np.array(mesh_usd.GetFaceVertexIndicesAttr().Get(), dtype=np.int32)
            faces_vertex_counts = np.array(mesh_usd.GetFaceVertexCountsAttr().Get())
            points_faces_varying = False

            # parse normals
            normals = None
            normal_attr = mesh_usd.GetNormalsAttr()
            if normal_attr.HasValue():
                normals = np.array(normal_attr.Get(), dtype=np.float32)
                if normals.shape[0] != points.shape[0]:
                    if normals.shape[0] == faces.shape[0]:  # face varying meshes, adjacent faces do not share vertices
                        points_faces_varying = True
                    else:
                        gs.raise_exception(f"Size of normals mismatch for mesh {mesh_id} in usd file {path}.")

            # parse materials
            prim_bindings = UsdShade.MaterialBindingAPI(prim)
            material_usd = prim_bindings.ComputeBoundMaterial()[0]
            if material_usd.GetPrim().IsValid():
                material_spec = material_usd.GetPrim().GetPrimStack()[-1]
                material_file = material_spec.layer.identifier
                material_file = path if material_file == baked_path else material_file
                material_id = material_file + material_spec.path.pathString
                material, uv_name = materials.get(material_id, (None, "st"))
            else:
                material, uv_name, material_id = None, "st", None

            # parse uvs
            uvs = None
            uv_var = UsdGeom.PrimvarsAPI(prim).GetPrimvar(uv_name)
            if uv_var.IsDefined() and uv_var.HasValue():
                uvs = np.array(uv_var.ComputeFlattened(), dtype=np.float32)
                if uvs.shape[0] != points.shape[0]:
                    if uvs.shape[0] == faces.shape[0]:
                        points_faces_varying = True
                    else:
                        gs.raise_exception(f"Size of uvs mismatch for mesh {mesh_id} in usd file {path}.")
                uvs[:, 1] = 1.0 - uvs[:, 1]

            # rearrange points and faces
            if points_faces_varying:
                points = points[faces]
                faces = np.arange(faces.shape[0])

            # triangulate faces
            if np.max(faces_vertex_counts) > 3:
                triangles = []
                bi = 0
                for face_vertex_count in faces_vertex_counts:
                    if face_vertex_count == 3:
                        triangles.append([faces[bi + 0], faces[bi + 1], faces[bi + 2]])
                    elif face_vertex_count > 3:
                        for i in range(1, face_vertex_count - 1):
                            triangles.append([faces[bi + 0], faces[bi + i], faces[bi + i + 1]])
                    bi += face_vertex_count
                triangles = np.array(triangles, dtype=np.int32)
                gs.logger.warning(f"Mesh {mesh_usd} has non-triangle faces.")
            else:
                triangles = faces.reshape(-1, 3)

            # process mesh
            processed_mesh = trimesh.Trimesh(
                vertices=points,
                faces=triangles,
                vertex_normals=normals,
                visual=trimesh.visual.TextureVisuals(uv=uvs),
                process=True,
            )
            points = processed_mesh.vertices
            triangles = processed_mesh.faces
            normals = processed_mesh.vertex_normals
            uvs = processed_mesh.visual.uv

            # apply tranform
            points, normals = mu.apply_transform(matrix, points, normals)

            group_idx = material_id if group_by_material else i
            mesh_info, first_created = mesh_infos.get(group_idx)
            if first_created:
                mesh_info.set_property(
                    surface=material,
                    metadata={
                        "path": path,
                        "name": material_id if group_by_material else mesh_id,
                        "baked": material_id in baked_materials,
                    },
                )
            mesh_info.append(points, triangles, normals, uvs)

    return mesh_infos.export_meshes(scale=scale)


def parse_instance_usd(path):
    stage = Usd.Stage.Open(path)
    xform_cache = UsdGeom.XformCache()

    instance_list = []
    for i, prim in enumerate(stage.Traverse()):
        if prim.IsA(UsdGeom.Xformable):
            if len(prim.GetPrimStack()) > 1:
                assert len(prim.GetPrimStack()) == 2, f"Invalid instance {prim.GetPath()} in usd file {path}."
                if prim.GetPrimStack()[0].hasReferences:
                    matrix = np.array(xform_cache.GetLocalToWorldTransform(prim))
                    instance_spec = prim.GetPrimStack()[-1]
                    instance_list.append((matrix.T, instance_spec.layer.identifier))

    return instance_list
