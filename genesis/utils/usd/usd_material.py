"""
USD Rendering Material Parser

Parser for extracting and parsing rendering materials from USD stages.
"""

from pxr import Usd, UsdShade, Sdf
from typing import List
import genesis as gs

# from . import usda


# class UsdRenderingMaterialParser:
#     """
#     Parser for rendering materials in USD stages.
#     """

#     def __init__(self, context: UsdParserContext):
#         """
#         Initialize the material parser.

#         Parameters
#         ----------
#         context : UsdParserContext
#             The parser context to store materials in.
#         surface : gs.surfaces.Surface
#             The base surface to use for material parsing.
#         """
#         self._context = context

#     def parse_all_materials(self) -> dict:
#         """
#         Find all materials in the USD stage and parse them.

#         Returns
#         -------
#         dict
#             The materials dictionary (same as context.materials).
#             Key: material_id (str) - unique identifier for the material
#             Value: tuple of (material_surface, uv_name) - parsed material surface and UV name
#         """
#         stage = self._context.stage
#         materials = self._context.materials
#         default_surface = gs.surfaces.Default()

#         # Parse materials from the stage
#         for prim in stage.Traverse():
#             if prim.IsA(UsdShade.Material):
#                 material_usd = UsdShade.Material(prim)
#                 material_spec = prim.GetPrimStack()[-1]
#                 material_id = material_spec.layer.identifier + material_spec.path.pathString

#                 if material_id not in materials:
#                     material, uv_name, require_bake = usda.parse_usd_material(material_usd, default_surface)
#                     materials[material_id] = (material, uv_name)
#                     if require_bake:
#                         gs.logger.debug(
#                             f"Material {material_id} requires baking (not yet implemented in context-based parsing)"
#                         )

#         return materials


cs_encode = {
    "raw": "linear",
    "sRGB": "srgb",
    "auto": None,
    "": None,
}


def get_input_attribute_value(shader: UsdShade.Shader, input_name, input_type=None):
    shader_input = shader.GetInput(input_name)

    if input_type != "value":
        if shader_input.GetPrim().IsValid() and shader_input.HasConnectedSource():
            shader_input_connect, shader_input_name = shader_input.GetConnectedSource()[:2]
            return shader_input_connect.GetPrim(), shader_input_name

    if input_type != "attribute":
        return shader_input.Get(), None
    return None, None


def get_shader(prim: Usd.Prim, output_name):
    if prim.IsA(UsdShade.Shader):
        return UsdShade.Shader(prim)
    elif prim.IsA(UsdShade.NodeGraph):
        return UsdShade.NodeGraph(prim).ComputeOutputSource(output_name)[0]
    else:
        gs.raise_exception(f"Invalid shader type: {prim.GetTypeName()} at {prim.GetPath()}.")


def parse_preview_surface(prim: Usd.Prim, output_name):
    shader = get_shader(prim, output_name)
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
                    component, component_output
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
            texture_image = np.asarray(Image.open(texture.resolvedPath))
            if texture_image.ndim == 3:
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
        texture_uvs, texture_uvs_output = get_input_attribute_value(shader, "st", "attribute")
        texture_uvs_name = parse_preview_surface(texture_uvs, texture_uvs_output)

        return texture_image, texture_encode, texture_uvs_name

    elif shader_id.startswith("UsdPrimvarReader"):
        primvar_name = get_input_attribute_value(shader, "varname", "value")[0]
        return primvar_name


def parse_material_preview_surface(material: UsdShade.Material) -> tuple[dict, str]:
    """Find the preview surface for a material."""
    surface_outputs = material.GetSurfaceOutputs()
    candidates_surfaces = []
    material_dict, uv_name = None, None
    for surface_output in surface_outputs:
        if not surface_output.HasConnectedSource():
            continue
        surface_output_connectable, surface_output_name, _ = surface_output.GetConnectedSource()
        surface_output_connect = surface_output_connectable.GetPrim()
        surface_shader = get_shader(surface_output_connect, "surface")
        surface_shader_implement = surface_shader.GetImplementationSource()
        surface_shader_id = surface_shader.GetShaderId()
        if surface_shader_implement == "id" and surface_shader_id == "UsdPreviewSurface":
            material_dict, uv_name = parse_preview_surface(surface_output_connect, surface_output_name)
            break
        candidates_surfaces.append((surface_shader.GetPath(), surface_shader_id, surface_shader_implement))

    if material_dict is None:
        candidates_str = "\n".join(
            f"\tShader at {shader_path} with implement {shader_impl} and ID {shader_id}."
            for shader_path, shader_id, shader_impl in candidates_surfaces
        )
        gs.logger.debug(f"Material require baking:\n{candidates_str}")
    return material_dict, uv_name
