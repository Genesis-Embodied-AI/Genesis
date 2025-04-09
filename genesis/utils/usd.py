import os
import numpy as np
import re
from PIL import Image
from pxr import Usd, UsdGeom, UsdShade, Gf, Sdf

import genesis as gs
import genesis.utils.mesh as mu

def parse_mesh_usd(path, group_by_material, scale, surface):
    """
    Parse mesh from USD file.
    This function only supports reading geometries from the default prim of the USD stage.
    Especially, for NVIDIA's SimReady USD dataset: https://developer.nvidia.com/omniverse/simready-assets
    Refer to the book for more details: https://www.manning.com/books/universal-scene-description-in-action

    Parameters
    ----------
    path : str
        usd file path.
    group_by_material : bool
        Whether to group meshes by material. #TODO: not implemented yet.
    scale : float
        Scale factor for the mesh.
    surface : Surface
        Surface object to be used for the mesh.
    """

    def get_world_transform(prim):
        """Get the world transform of a given prim in the USD stage."""
        xform = UsdGeom.Xformable(prim)
        time_code = Usd.TimeCode.Default()
        world_transform = xform.ComputeLocalToWorldTransform(time_code)
        return world_transform

    def extract_texture_paths_from_mdl(mdl_file_path):
        """
        Extract diffuse and normal map texture paths and diffuse tint (as color_factor)
        from an MDL file using regex. Since the NVIDIA MDL SDK doesn't have a Python API,
        we parse the file to extract these values.
        """
        with open(mdl_file_path, "r", encoding="utf-8") as f:
            mdl_content = f.read()

        # Regex patterns to match diffuse and normal map textures
        diffuse_pattern = re.search(r'diffuse_texture:\s*texture_2d\("([^"]+)"', mdl_content)
        normal_pattern = re.search(r'normalmap_texture:\s*texture_2d\("([^"]+)"', mdl_content)
        orm_pattern = re.search(r'ORM_texture:\s*texture_2d\("([^"]+)"', mdl_content)

        # Regex pattern to match diffuse_tint color values
        # This expects a pattern like: diffuse_tint: color(0.47876447, 0.47875968, 0.47875968)
        diffuse_tint_pattern = re.search(r"diffuse_tint:\s*color\(([^)]+)\)", mdl_content)

        diffuse_texture = diffuse_pattern.group(1) if diffuse_pattern else None
        normal_texture = normal_pattern.group(1) if normal_pattern else None
        orm_texture = orm_pattern.group(1) if orm_pattern else None

        if diffuse_tint_pattern:
            # Extract the comma-separated color values from inside the parentheses
            tint_values_str = diffuse_tint_pattern.group(1)
            # Convert each value to a float and form a tuple
            diffuse_tint = tuple(float(x.strip()) for x in tint_values_str.split(","))
        else:
            diffuse_tint = None

        return diffuse_texture, normal_texture, diffuse_tint, orm_texture

    """parse mesh from USD file"""
    meshes = list()
    stage = Usd.Stage.Open(path)
    default_prim_path = stage.GetDefaultPrim().GetPath().pathString
    for prim in stage.Traverse():
        # only load mesh under the default prim
        if UsdGeom.Mesh(prim) and (default_prim_path in prim.GetPath().pathString):
            # check surface
            mesh_surface = gs.surfaces.Default() if surface is None else surface.copy()

            # load mesh basic information
            mesh = UsdGeom.Mesh(prim)
            points = mesh.GetPointsAttr().Get()
            normals = mesh.GetNormalsAttr().Get()
            face_vertex_counts = mesh.GetFaceVertexCountsAttr().Get()
            face_vertex_indices = mesh.GetFaceVertexIndicesAttr().Get()
            sts = mesh.GetPrim().GetAttribute("primvars:st").Get()
            st_indices = mesh.GetPrim().GetAttribute("primvars:st:indices").Get()

            # calculate transform
            transform = np.array(get_world_transform(mesh.GetPrim()))
            points_homogeneous = np.hstack([points, np.ones((len(points), 1))])

            verts = (points_homogeneous @ transform)[:, :3]
            normals = np.array(normals, dtype=float)
            indices = np.array(face_vertex_indices, dtype=int)

            faces = []
            index_offset = 0
            face_count = {}

            for count in face_vertex_counts:
                face_count[count] = face_count.get(count, 0) + 1
                face = indices[index_offset : index_offset + count].tolist()

                # If the face is a triangle, keep it as is
                if count == 3:
                    faces.append(face)
                # If the face is a quad or more, triangulate it
                elif count > 3:
                    for i in range(1, count - 1):
                        faces.append([face[0], face[i], face[i + 1]])  # Fan triangulation

                index_offset += count  # Move to the next set of indices

            # if face count is not 3, means it is not all triangle faces
            if len(face_count) >= 2 or (3 not in face_count):
                gs.logger.warning(f"Mesh {mesh} has non-triangle faces. Face count {face_count}")

            faces = np.array(faces, dtype=int)

            # load uvs
            if sts is not None:
                uvs = []
                gs.logger.info("Parsing OpenUSD Face st into Vertex uv")
                # Create a mapping of face vertex indices to their UV
                if st_indices:
                    assert len(face_vertex_indices) == len(
                        st_indices
                    ), "face vertice count should be the same as st indices"
                    index_to_uv = {fv_idx: sts[st_indices[idx]] for idx, fv_idx in enumerate(face_vertex_indices)}
                else:
                    assert len(sts) == len(face_vertex_indices), "face vertice count should be the same as st"
                    index_to_uv = {fv_idx: sts[idx] for idx, fv_idx in enumerate(face_vertex_indices)}

                # Collect the UVs using the mapping
                uvs = [index_to_uv[i] for i in range(len(points)) if i in index_to_uv]
                uvs = np.array(uvs)
                uvs[:, 1] = 1.0 - uvs[:, 1]  # flip y axis for trimesh
            else:
                uvs = None

            # load material
            material_binding = UsdShade.MaterialBindingAPI(mesh.GetPrim())
            material, relation = material_binding.ComputeBoundMaterial()

            # if has material
            if material.GetPrim().IsValid():
                # normal usd shader
                shader, _, _ = material.ComputeSurfaceSource()
                # compatible with NVIDIA mdl: https://github.com/NVIDIA/MDL-SDK
                if not shader:
                    shader, _, _ = material.ComputeSurfaceSource("mdl")

                if shader:
                    diffuse_texture_attr = shader.GetPrim().GetAttribute("inputs:diffuse_texture").Get()

                    # handle the diffuse map texture from USD attribute
                    if diffuse_texture_attr:
                        diffuse_texture_path = diffuse_texture_attr.resolvedPath
                        diffuse_image = mu.PIL_to_array(Image.open(diffuse_texture_path).convert("RGB"))

                        # get diffuse tint color
                        diffuse_tint = shader.GetPrim().GetAttribute("inputs:diffuse_tint").Get()
                        if diffuse_tint:
                            diffuse_tint = np.array(list(diffuse_tint) + [1.0], dtype=float)
                        else:
                            diffuse_tint = np.ones(4, dtype=float)

                        color_texture = None
                        opacity_texture = None
                        roughness_texture = None
                        metallic_texture = None
                        normal_texture_path = None

                        # handle the diffuse map texture from USD attribute
                        if diffuse_image.ndim == 2:
                            diffuse_image = diffuse_image[:, :, np.newaxis]
                        elif diffuse_image.shape[2] == 4:
                            # get opacity texture from alpha channel
                            opacity_texture = mu.create_texture(diffuse_image[:, :, 3], None, "linear")

                            # remove alpha channel
                            diffuse_image = diffuse_image[:, :, :3]

                        color_texture = mu.create_texture(diffuse_image, diffuse_tint, "srgb")
                        gs.logger.info(f"Loading Diffuse texture: {diffuse_texture_path}")

                        # handle the normal map texture from USD attribute
                        normal_texture_attr = shader.GetPrim().GetAttribute("inputs:normal_texture").Get()
                        if normal_texture_attr:
                            normal_texture_path = normal_texture_attr.resolvedPath
                            normal_image = mu.PIL_to_array(Image.open(normal_texture_path))
                            normal_texture_path = mu.create_texture(normal_image, None, "linear")
                            gs.logger.info(f"Loading Normal texture: {normal_texture_path}")

                        # handle ORM texture from USD attribute
                        orm_texture_attr = shader.GetPrim().GetAttribute("inputs:ORM_texture").Get()
                        if orm_texture_attr:
                            orm_texture_path = orm_texture_attr.resolvedPath
                            orm_image = mu.PIL_to_array(Image.open(orm_texture_path))
                            # Split ORM image into separate textures
                            opacity_texture = mu.create_texture(orm_image[:, :, 0], None, "linear")
                            roughness_texture = mu.create_texture(orm_image[:, :, 1], None, "linear")
                            metallic_texture = mu.create_texture(orm_image[:, :, 2], None, "linear")
                            gs.logger.info(f"Loading ORM texture: {orm_texture_path}")
                        else:
                            # no ORM texture， handle roughness and metallic texture from USD attribute
                            opacity_texture_attr = shader.GetPrim().GetAttribute("inputs:opacity_texture").Get()
                            if opacity_texture_attr:
                                opacity_texture_path = opacity_texture_attr.resolvedPath
                                opacity_image = mu.PIL_to_array(Image.open(opacity_texture_path))
                                opacity_texture = mu.create_texture(opacity_image, None, "linear")
                                gs.logger.info(f"Loading Opacity texture: {opacity_texture_path}")

                            roughness_texture_attr = (
                                shader.GetPrim().GetAttribute("inputs:reflectionroughness_texture").Get()
                            )
                            if roughness_texture_attr:
                                roughness_texture_path = roughness_texture_attr.resolvedPath
                                roughness_image = mu.PIL_to_array(Image.open(roughness_texture_path))
                                roughness_texture = mu.create_texture(roughness_image, None, "linear")
                                gs.logger.info(f"Loading Roughness texture: {roughness_texture_path}")

                            metallic_texture_attr = shader.GetPrim().GetAttribute("inputs:metallic_texture").Get()
                            if metallic_texture_attr:
                                metallic_texture_path = metallic_texture_attr.resolvedPath
                                metallic_image = mu.PIL_to_array(Image.open(metallic_texture_path))
                                metallic_texture = mu.create_texture(metallic_image, None, "linear")
                                gs.logger.info(f"Loading Metallic texture: {metallic_texture_path}")

                        # update surface texture
                        mesh_surface.update_texture(
                            color_texture=color_texture,
                            normal_texture=normal_texture_path,
                            opacity_texture=opacity_texture,
                            roughness_texture=roughness_texture,
                            metallic_texture=metallic_texture,
                        )

                    else:  # load from .mdl description
                        source = shader.GetImplementationSource()
                        # handle the mdl shader source from asset
                        if source == UsdShade.Tokens.sourceAsset:
                            mdl_asset_attr = shader.GetPrim().GetAttribute("info:mdl:sourceAsset")
                            mdl_sub_identifier_attr = shader.GetPrim().GetAttribute(
                                "info:mdl:sourceAsset:subIdentifier"
                            )

                            if mdl_asset_attr and mdl_sub_identifier_attr:
                                mdl_asset_path = mdl_asset_attr.Get().resolvedPath  # absolute path
                                mdl_material_name = mdl_sub_identifier_attr.Get()
                                diffuse_texture_path, normal_texture_path, diffuse_tint, orm_texture_path = (
                                    extract_texture_paths_from_mdl(mdl_asset_path)
                                )

                                if diffuse_tint:
                                    diffuse_tint = np.array(list(diffuse_tint) + [1.0], dtype=float)  # RGB + A
                                else:
                                    diffuse_tint = np.ones(4, dtype=float)

                                color_texture = None
                                normal_texture = None
                                opacity_texture = None
                                roughness_texture = None
                                metallic_texture = None

                                if diffuse_texture_path:
                                    diffuse_texture_path = os.path.normpath(
                                        os.path.join(os.path.dirname(mdl_asset_path), diffuse_texture_path)
                                    )
                                    diffuse_image = mu.PIL_to_array(Image.open(diffuse_texture_path))
                                    color_texture = mu.create_texture(diffuse_image, diffuse_tint, "srgb")
                                    gs.logger.info(f"Get Diffuse texture: {diffuse_texture_path}")

                                if normal_texture_path:
                                    normal_texture_path = os.path.normpath(
                                        os.path.join(os.path.dirname(mdl_asset_path), normal_texture_path)
                                    )
                                    normal_image = mu.PIL_to_array(Image.open(normal_texture_path))
                                    normal_texture = mu.create_texture(normal_image, None, "linear")
                                    gs.logger.info(f"Get Normal texture: {normal_texture_path}")

                                if orm_texture_path:
                                    orm_texture_path = os.path.normpath(
                                        os.path.join(os.path.dirname(mdl_asset_path), orm_texture_path)
                                    )
                                    orm_image = mu.PIL_to_array(Image.open(orm_texture_path))
                                    # Split ORM image into separate textures
                                    opacity_texture = mu.create_texture(orm_image[:, :, 0], None, "linear")
                                    roughness_texture = mu.create_texture(orm_image[:, :, 1], None, "linear")
                                    metallic_texture = mu.create_texture(orm_image[:, :, 2], None, "linear")
                                    gs.logger.info(f"Get ORM texture: {orm_texture_path}")

                                mesh_surface.update_texture(
                                    color_texture=color_texture,
                                    normal_texture=normal_texture,
                                    opacity_texture=opacity_texture,
                                    roughness_texture=roughness_texture,
                                    metallic_texture=metallic_texture,
                                )

            meshes.append(
                gs.Mesh.from_attrs(
                    verts=verts,
                    faces=faces,
                    normals=normals,
                    surface=mesh_surface,
                    uvs=uvs,
                    scale=scale,
                )
            )

    del stage
    return meshes