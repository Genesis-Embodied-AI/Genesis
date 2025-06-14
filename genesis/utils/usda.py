import genesis as gs
from . import mesh as mu

from pxr import Usd, UsdGeom, UsdShade
import trimesh
import numpy as np
from PIL import Image
import io

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
    if zipfiles is None:
        return np.array(Image.open(image_path.resolvedPath))
    else:
        return np.array(Image.open(io.BytesIO(zipfiles.GetFile(image_path.path))))


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

        # parse io
        ior = get_input_attribute_value(shader, "ior", "value")[0]

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


def parse_gltf_surface(shader, source_type, output_name, zipfiles):
    shader_subid = shader.GetSourceAssetSubIdentifier(source_type)
    if shader_subid == "gltf_material":
        # Parse color
        color_factor = get_input_attribute_value(shader, "base_color_factor", "value")[0]  # Gf.Vec3f(1.0, 1.0, 1.0)
        color_texture_shader, color_texture_output = get_input_attribute_value(
            shader, "base_color_texture", "attribute"
        )
        if color_texture_shader is not None:
            color_image = parse_gltf_surface(color_texture_shader, source_type, color_texture_output, zipfiles)
        else:
            color_image = None
        color_texture = mu.create_texture(color_image, color_factor, "srgb")

        # parse opacity
        opacity_factor = get_input_attribute_value(shader, "base_alpha", "value")[0]
        opacity_texture = mu.create_texture(None, opacity_factor, "linear")
        alpha_cutoff = get_input_attribute_value(shader, "alpha_cutoff", "value")[0]
        alpha_mode = get_input_attribute_value(shader, "alpha_mode", "value")[0]
        alpha_cutoff = mu.adjust_alpha_cutoff(alpha_cutoff, alpha_mode)
        opacity_texture.apply_cutoff(alpha_cutoff)

        # parse roughness and metaillic
        metallic_factor = get_input_attribute_value(shader, "metallic_factor", "value")[0]
        roughness_factor = get_input_attribute_value(shader, "roughness_factor", "value")[0]
        combined_texture_shader, combined_texture_output = get_input_attribute_value(
            shader, "metallic_roughness_texture", "attribute"
        )
        if combined_texture_shader is not None:
            combined_image = parse_gltf_surface(combined_texture_shader, source_type, combined_texture_output, zipfiles)
            roughness_image = combined_image[:, :, 1]
            metallic_image = combined_image[:, :, 2]
        else:
            roughness_image, metallic_image = None, None
        metallic_texture = mu.create_texture(metallic_image, metallic_factor, "linear")
        roughness_texture = mu.create_texture(roughness_image, roughness_factor, "linear")

        # parse emissive
        emissive_strength = get_input_attribute_value(shader, "emissive_strength", "value")[0]
        emissive_texture = mu.create_texture(None, emissive_strength, "srgb") if emissive_strength else None

        occlusion_texture_shader, occlusion_texture_output = get_input_attribute_value(
            shader, "occlusion_texture", "attribute"
        )
        if occlusion_texture_shader is not None:
            occlusion_image = parse_gltf_surface(
                occlusion_texture_shader, source_type, occlusion_texture_output, zipfiles
            )

        return {
            "color_texture": color_texture,
            "opacity_texture": opacity_texture,
            "roughness_texture": roughness_texture,
            "metallic_texture": metallic_texture,
            "emissive_texture": emissive_texture,
        }, "st"

    elif shader_subid == "gltf_texture_lookup":
        texture = get_input_attribute_value(shader, "texture", "value")[0]
        if texture is not None:
            texture_image = get_texture_image(texture, zipfiles)
        else:
            texture_image = None
        return texture_image

    else:
        raise Exception(f"Fail to parse gltf Shader {shader_subid}.")


def parse_omni_surface(shader, source_type, output_name, zipfiles):

    def parse_component(component_name, component_encode, adjust=None):
        component_usetex = get_input_attribute_value(shader, f"Is{component_name}Tex", "value")[0] == 1
        if component_usetex:
            component_tex_name = f"{component_name}_Tex"
            component_tex = get_input_attribute_value(shader, component_tex_name, "value")[0]
            if component_tex is not None:
                component_image = get_texture_image(component_tex, zipfiles)
                if adjust is not None:
                    component_image = (adjust(component_image / 255.0) * 255.0).astype(np.uint8)
            component_cs = shader.GetInput(component_tex_name).GetAttr().GetColorSpace()
            component_overencode = cs_encode[component_cs]
            if component_overencode is not None:
                component_encode = component_overencode
            component_factor = None
        else:
            component_color_name = f"{component_name}_Color"
            component_factor = get_input_attribute_value(shader, component_color_name, "value")[0]
            if adjust is not None and component_factor is not None:
                component_factor = tuple([adjust(c) for c in component_factor])
            component_image = None

        component_texture = mu.create_texture(component_image, component_factor, component_encode)
        return component_texture

    color_texture = parse_component("BaseColor", "srgb")
    opacity_texture = color_texture.check_dim(3) if color_texture else None
    emissive_intensity = get_input_attribute_value(shader, "EmissiveIntensity", "value")[0]
    emissive_texture = (
        parse_component("Emissive", "srgb", lambda x: x * emissive_intensity) if emissive_intensity else None
    )
    if emissive_texture is not None:
        emissive_texture.check_dim(3)
    metallic_texture = parse_component("Metallic", "linear")
    normal_texture = parse_component("Normal", "linear")
    roughness_texture = parse_component("Gloss", "linear", lambda x: (2 / (x + 2)) ** (1.0 / 4.0))

    return {
        "color_texture": color_texture,
        "opacity_texture": opacity_texture,
        "roughness_texture": roughness_texture,
        "metallic_texture": metallic_texture,
        "emissive_texture": emissive_texture,
        "normal_texture": normal_texture,
    }, "st"


def parse_usd_material(material, surface, zipfiles):
    surface_outputs = material.GetSurfaceOutputs()
    material_dict, uv_name = None, None
    material_surface = None
    for surface_output in surface_outputs:
        if not surface_output.HasConnectedSource():
            continue
        surface_output_connectable, surface_output_name, _ = surface_output.GetConnectedSource()
        surface_shader = UsdShade.Shader(surface_output_connectable.GetPrim())
        surface_shader_implement = surface_shader.GetImplementationSource()

        if surface_shader_implement == "id":
            shader_id = surface_shader.GetShaderId()
            if shader_id == "UsdPreviewSurface":
                material_dict, uv_name = parse_preview_surface(surface_shader, surface_output_name, zipfiles)
                break
            gs.logger.warning(f"Fail to parse Shader {surface_shader.GetPath()} with ID {shader_id}.")
            continue

        elif surface_shader_implement == "sourceAsset":
            source_types = surface_shader.GetSourceTypes()
            for source_type in source_types:
                source_asset = surface_shader.GetSourceAsset(source_type).resolvedPath
                if "gltf/pbr" in source_asset:
                    material_dict, uv_name = parse_gltf_surface(
                        surface_shader, source_type, surface_output_name, zipfiles
                    )
                    break
                try:
                    material_dict, uv_name = parse_omni_surface(
                        surface_shader, source_type, surface_output_name, zipfiles
                    )
                except Exception as e:
                    gs.logger.warning(
                        f"Fail to parse Shader {surface_shader.GetPath()} of asset {source_asset} with message: {e}."
                    )
                    continue

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
    return material_surface, uv_name


def parse_mesh_usd(path, group_by_material, scale, surface):
    zipfiles = Usd.ZipFile.Open(path) if path.endswith(".usdz") else None
    stage = Usd.Stage.Open(path)
    scale *= UsdGeom.GetStageMetersPerUnit(stage)
    yup = UsdGeom.GetStageUpAxis(stage) == "Y"
    xform_cache = UsdGeom.XformCache()

    mesh_infos = mu.MeshInfoGroup()
    materials = dict()

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
                material_id = material_spec.layer.identifier + material_spec.path.pathString
                material, uv_name = materials.get(material_id, (None, "st"))
                if material is None:
                    material, uv_name = materials.setdefault(
                        material_id, parse_usd_material(material_usd, surface, zipfiles)
                    )
            else:
                material, uv_name, material_id = None, "st", None

            # parse uvs
            uv_var = UsdGeom.PrimvarsAPI(prim).GetPrimvar(uv_name)
            uvs = None
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
                    surface=material, metadata={"path": path, "name": material_id if group_by_material else mesh_id}
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
