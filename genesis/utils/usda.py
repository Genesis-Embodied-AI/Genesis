import genesis as gs
from . import mesh as mu

from pxr import Usd, UsdGeom, UsdShade
import trimesh
import numpy as np
from PIL import Image

cs_encode = {
    "raw": "linear",
    "sRGB": "srgb",
    "auto": None,
    "": None,
}


def make_tuple(value):
    if value is None:
        return None
    else:
        return (value,)


def flip(image):
    return None if image is None else np.flipud(image)


def get_input_attribute_value(shader, input_name, input_type=None):
    shader_input = shader.GetInput(input_name)

    if input_type != "value":
        shader_input_attr = shader_input.GetValueProducingAttribute()[0]
        if shader_input_attr.IsValid():
            return UsdShade.Shader(shader_input_attr.GetPrim()), shader_input_attr.GetBaseName()

    if input_type != "attribute":
        return shader_input.Get(), None
    return None, None


def parse_preview_surface(shader, output_name):
    shader_id = shader.GetShaderId()
    if shader_id == "UsdPreviewSurface":
        uvname = None

        def parse_component(component_name, component_encode):
            component, component_output = get_input_attribute_value(shader, component_name)
            if component_output is None:  # constant value
                component_factor = component
                component_image = None
            else:  # texture shader
                component_image, component_overencode, component_uvname = parse_preview_surface(
                    component, component_output
                )
                if component_overencode is not None:
                    component_encode = component_overencode
                component_factor = None

            component_texture = mu.create_texture(flip(component_image), component_factor, component_encode)
            return component_texture, component_uvname

        # parse color
        color_texture, color_uvname = parse_component("diffuseColor", "srgb")
        if color_uvname is not None:
            uvname = color_uvname

        # parse opacity
        opacity_texture, opacity_uvname = parse_component("opacity", "linear")
        if opacity_uvname is not None and uvname is None:
            uvname = opacity_uvname
        alpha_cutoff = get_input_attribute_value(shader, "opacityThreshold", "value")[0]
        opacity_texture.apply_cutoff(alpha_cutoff)

        # parse emissive
        emissive_texture, emissive_uvname = parse_component("emissiveColor", "srgb")
        if emissive_uvname is not None and uvname is None:
            uvname = emissive_uvname

        # parse mertalic
        use_metallic = get_input_attribute_value(shader, "useSpecularWorkflow", "value")[0] == 0
        if use_metallic:
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
            texture_path = texture.resolvedPath
        texture_image = np.array(Image.open(texture_path))
        texture_encode = get_input_attribute_value(shader, "sourceColorSpace", "value")[0]
        texture_encode = cs_encode[texture_encode]

        texture_uvs_shader, texture_uvs_output = get_input_attribute_value(shader, "st", "attribute")
        texture_uvs_name = parse_preview_surface(texture_uvs_shader, texture_uvs_output)

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

        return texture_image, texture_encode, texture_uvs_name

    elif shader_id.startswith("UsdPrimvarReader"):
        primvar_name = get_input_attribute_value(shader, "varname", "value")[0]
        return primvar_name


def parse_gltf_surface(shader, source_type, output_name):
    shader_subid = shader.GetSourceAssetSubIdentifier(source_type)
    if shader_subid == "gltf_material":
        # Parse color
        color_factor = get_input_attribute_value(shader, "base_color_factor", "value")[0]  # Gf.Vec3f(1.0, 1.0, 1.0)
        color_texture_shader, color_texture_output = get_input_attribute_value(
            shader, "base_color_texture", "attribute"
        )
        if color_texture_shader is not None:
            color_image = parse_gltf_surface(color_texture_shader, source_type, color_texture_output)
        else:
            color_image = None
        color_texture = mu.create_texture(flip(color_image), color_factor, "srgb")

        # parse opacity
        opacity_factor = make_tuple(get_input_attribute_value(shader, "base_alpha", "value")[0])
        opacity_texture = mu.create_texture(None, opacity_factor, "linear")
        alpha_cutoff = get_input_attribute_value(shader, "alpha_cutoff", "value")[0]
        alpha_mode = get_input_attribute_value(shader, "alpha_mode", "value")[0]
        alpha_cutoff = mu.adjust_alpha_cutoff(alpha_cutoff, alpha_mode)
        opacity_texture.apply_cutoff(alpha_cutoff)

        # parse roughness and metaillic
        metallic_factor = make_tuple(get_input_attribute_value(shader, "metallic_factor", "value")[0])
        roughness_factor = make_tuple(get_input_attribute_value(shader, "roughness_factor", "value")[0])
        combined_texture_shader, combined_texture_output = get_input_attribute_value(
            shader, "metallic_roughness_texture", "attribute"
        )
        if combined_texture_shader is not None:
            combined_image = parse_gltf_surface(combined_texture_shader, source_type, combined_texture_output)
            roughness_image = combined_image[:, :, 1]
            metallic_image = combined_image[:, :, 2]
        else:
            roughness_image, metallic_image = None, None
        metallic_texture = mu.create_texture(flip(metallic_image), metallic_factor, "linear")
        roughness_texture = mu.create_texture(flip(roughness_image), roughness_factor, "linear")

        # parse emissive
        emissive_factor = make_tuple(get_input_attribute_value(shader, "emissive_strength", "value")[0])
        emissive_texture = mu.create_texture(None, emissive_factor, "srgb")

        occlusion_texture_shader, occlusion_texture_output = get_input_attribute_value(
            shader, "occlusion_texture", "attribute"
        )
        if occlusion_texture_shader is not None:
            occlusion_image = parse_gltf_surface(occlusion_texture_shader, source_type, occlusion_texture_output)

        return {
            "color_texture": color_texture,
            "opacity_texture": opacity_texture,
            "roughness_texture": roughness_texture,
            "metallic_texture": metallic_texture,
            "emissive_texture": emissive_texture,
        }, "st"

    elif shader_subid == "gltf_texture_lookup":
        # offset = shader.GetInput("offset").Get()
        # rotation = shader.GetInput("rotation").Get()
        # scale = shader.GetInput("scale").Get()
        # tex_coord_index = shader.GetInput("tex_coord_index").Get()

        texture = get_input_attribute_value(shader, "texture", "value")[0]
        if texture is not None:
            texture_path = texture.resolvedPath
        texture_image = np.array(Image.open(texture_path))
        return texture_image

    else:
        raise Exception(f"Fail to parse gltf Shader {shader_subid}.")


def parse_omni_surface(shader, source_type, output_name):

    def parse_component(component_name, component_encode, adjust=None):
        component_usetex = get_input_attribute_value(shader, f"Is{component_name}Tex", "value")[0] == 1
        if component_usetex:
            component_tex_name = f"{component_name}_Tex"
            component_texture = get_input_attribute_value(shader, component_tex_name, "value")[0]
            if component_texture is not None:
                component_image = np.array(Image.open(component_texture.resolvedPath))
            component_cs = shader.GetInput(component_tex_name).GetAttr().GetColorSpace()
            component_overencode = cs_encode[component_cs]
            if component_overencode is not None:
                component_encode = component_overencode
            if adjust is not None:
                component_image = (adjust(component_image / 255.0) * 255.0).astype(np.uint8)
            component_factor = None
        else:
            component_color_name = f"{component_name}_Color"
            component_factor = get_input_attribute_value(shader, component_color_name, "value")[0]
            if adjust is not None and component_factor is not None:
                component_factor = tuple([adjust(c) for c in component_factor])
            component_image = None

        component_texture = mu.create_texture(flip(component_image), component_factor, component_encode)
        return component_texture

    color_texture = parse_component("BaseColor", "srgb")
    opacity_texture = color_texture.check_dim(3) if color_texture else None
    emissive_intensity = get_input_attribute_value(shader, "EmissiveIntensity", "value")[0]
    emissive_texture = parse_component("Emissive", "srgb", lambda x: x * emissive_intensity)
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


def parse_usd_material(material):
    surface_outputs = material.GetSurfaceOutputs()
    for surface_output in surface_outputs:
        if not surface_output.HasConnectedSource():
            continue
        surface_output_connectable, surface_output_name, _ = surface_output.GetConnectedSource()
        surface_shader = UsdShade.Shader(surface_output_connectable.GetPrim())
        surface_shader_implement = surface_shader.GetImplementationSource()

        if surface_shader_implement == "id":
            if surface_shader.GetShaderId() == "UsdPreviewSurface":
                return parse_preview_surface(surface_shader, surface_output_name)
            gs.logger.warning(
                f"Fail to parse Shader {surface_shader.GetPath()} with ID {surface_shader.GetShaderId()}."
            )
            continue

        elif surface_shader_implement == "sourceAsset":
            source_types = surface_shader.GetSourceTypes()
            for source_type in source_types:
                source_asset = surface_shader.GetSourceAsset(source_type).resolvedPath
                if "gltf/pbr" in source_asset:
                    return parse_gltf_surface(surface_shader, source_type, surface_output_name)
                return parse_omni_surface(surface_shader, source_type, surface_output_name)
                # try:
                #     return parse_omni_surface(surface_shader, source_type, surface_output_name)
                # except Exception as e:
                #     gs.logger.warning(f"Fail to parse Shader {surface_shader.GetPath()} of asset {source_asset} with message: {e}.")
                #     continue

    return None, None


def parse_mesh_usd(path, group_by_material, scale, surface):
    stage = Usd.Stage.Open(path)
    xform_cache = UsdGeom.XformCache()

    mesh_infos = mu.MeshInfoGroup()
    materials = dict()
    uv_names = dict()

    for prim in stage.Traverse():
        if prim.HasRelationship("material:binding"):
            if not prim.HasAPI(UsdShade.MaterialBindingAPI):
                UsdShade.MaterialBindingAPI.Apply(prim)
    for i, prim in enumerate(stage.Traverse()):
        if prim.IsA(UsdGeom.Mesh):
            matrix = np.array(xform_cache.GetLocalToWorldTransform(prim))
            usd_mesh = UsdGeom.Mesh(prim)
            mesh_path = prim.GetPath().pathString

            if not usd_mesh.GetPointsAttr().HasValue():
                continue
            points = np.array(usd_mesh.GetPointsAttr().Get(), dtype=np.float32)
            faces = np.array(usd_mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int32)
            faces_vertex_counts = np.array(usd_mesh.GetFaceVertexCountsAttr().Get())
            points_faces_varying = False

            # parse normals
            normals = None
            normal_attr = usd_mesh.GetNormalsAttr()
            if normal_attr.HasValue():
                normals = np.array(normal_attr.Get(), dtype=np.float32)
                if normals.shape[0] != points.shape[0]:
                    if normals.shape[0] == faces.shape[0]:  # face varying
                        points_faces_varying = True
                    else:
                        gs.raise_exception(f"Size of normals mismatch for mesh {mesh_path} in usd file {path}.")

            # parse materials
            prim_bindings = UsdShade.MaterialBindingAPI(prim)
            material = prim_bindings.ComputeBoundMaterial()[0]
            group_idx = ""
            if material.GetPrim().IsValid():
                material_spec = material.GetPrim().GetPrimStack()[-1]
                material_id = material_spec.layer.identifier + material_spec.path.pathString

                if material_id not in materials:
                    material_dict, uv_names[material_id] = parse_usd_material(material)
                    material_surface = surface.copy()

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
                    materials[material_id] = material_surface
                group_idx = material_id if group_by_material else i
                uv_name = uv_names[material_id]

            # parse uvs
            # print(uv_name, material_id, len(uv_names))

            uv_attr = prim.GetAttribute(f"primvars:{uv_name}")
            uvs = None
            if uv_attr.HasValue():
                uvs = np.array(uv_attr.Get(), dtype=np.float32)
                if uvs.shape[0] != points.shape[0]:
                    if uvs.shape[0] == faces.shape[0]:
                        points_faces_varying = True
                    else:
                        gs.raise_exception(f"Size of uvs mismatch for mesh {mesh_path} in usd file {path}.")

            # rearrange points and faces
            if points_faces_varying:
                points = points[faces]
                faces = np.arange(faces.shape[0])

            if np.max(faces_vertex_counts) > 3:
                triangles = list()
                bi = 0
                for fi in range(len(faces_vertex_counts)):
                    if faces_vertex_counts[fi] == 3:
                        triangles.append([faces[bi + 0], faces[bi + 1], faces[bi + 2]])
                        bi += 3
                    elif faces_vertex_counts[fi] == 4:
                        triangles.append([faces[bi + 0], faces[bi + 1], faces[bi + 2]])
                        triangles.append([faces[bi + 0], faces[bi + 2], faces[bi + 3]])
                        bi += 4
                triangles = np.array(triangles, dtype=np.int32)
            else:
                triangles = faces.reshape(-1, 3)

            if normals is None:
                normals = trimesh.Trimesh(points, triangles, process=False).vertex_normals
            points, normals = mu.apply_transform(matrix, points, normals)

            mesh_infos.append(group_idx, points, triangles, normals, uvs, materials[material_id])

    return mesh_infos.export_meshes(scale=scale, path=path)


def parse_instance_usd(path):
    stage = Usd.Stage.Open(path)
    xform_cache = UsdGeom.XformCache()

    instance_list = list()
    for i, prim in enumerate(stage.Traverse()):
        if prim.IsA(UsdGeom.Xformable):
            if len(prim.GetPrimStack()) > 1:
                assert len(prim.GetPrimStack()) == 2, f"Invalid instance {prim.GetPath()} in usd file {path}."
                if prim.GetPrimStack()[0].hasReferences:
                    matrix = np.array(xform_cache.GetLocalToWorldTransform(prim))
                    instance_spec = prim.GetPrimStack()[-1]
                    instance_list.append((matrix.T, instance_spec.layer.identifier))

    return instance_list


if __name__ == "__main__":
    file_path = "table_scene.usd"
    grouped_meshes = parse_mesh_usd(file_path)
    for mesh in grouped_meshes:
        print(mesh)
