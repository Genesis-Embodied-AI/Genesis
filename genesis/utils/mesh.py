import hashlib
import os
import pickle as pkl
from io import BytesIO
from urllib import request

import coacd
import igl
import numpy as np
import pygltflib
import pyvista as pv
import tetgen
from PIL import Image

import genesis as gs
from genesis.ext import trimesh

from . import geom as gu
from .misc import (
    get_assets_dir,
    get_cvx_cache_dir,
    get_gsd_cache_dir,
    get_ptc_cache_dir,
    get_remesh_cache_dir,
    get_src_dir,
    get_tet_cache_dir,
)


def get_asset_path(file):
    return os.path.join(get_src_dir(), "assets", file)


def get_gsd_path(verts, faces, sdf_cell_size, sdf_min_res, sdf_max_res):
    hashkey = get_hashkey(
        verts.tobytes(),
        faces.tobytes(),
        str(sdf_cell_size).encode(),
        str(sdf_min_res).encode(),
        str(sdf_max_res).encode(),
    )
    return os.path.join(get_gsd_cache_dir(), f"{hashkey}.gsd")


def get_cvx_path(verts, faces, coacd_options):
    hashkey = get_hashkey(verts.tobytes(), faces.tobytes(), str(coacd_options.__dict__).encode())
    return os.path.join(get_cvx_cache_dir(), f"{hashkey}.cvx")


def get_ptc_path(verts, faces, p_size, sampler):
    hashkey = get_hashkey(verts.tobytes(), faces.tobytes(), str(p_size).encode(), sampler.encode())
    return os.path.join(get_ptc_cache_dir(), f"{hashkey}.ptc")


def get_tet_path(verts, faces, tet_cfg):
    hashkey = get_hashkey(verts.tobytes(), faces.tobytes(), str(tet_cfg).encode())
    return os.path.join(get_tet_cache_dir(), f"{hashkey}.tet")


def get_remesh_path(verts, faces, edge_len_abs, edge_len_ratio, fix):
    hashkey = get_hashkey(
        verts.tobytes(), faces.tobytes(), str(edge_len_abs).encode(), str(edge_len_ratio).encode(), str(fix).encode()
    )
    return os.path.join(get_remesh_cache_dir(), f"{hashkey}.rm")


def get_hashkey(*args):
    hasher = hashlib.sha256()
    for arg in args:
        hasher.update(arg)
    hasher.update(gs.__version__.encode())
    return hasher.hexdigest()


def load_mesh(file):
    if isinstance(file, str):
        return trimesh.load(file, force="mesh", skip_texture=True)
    else:
        return file


def normalize_mesh(mesh):
    """
    Normalize mesh to [-0.5, 0.5].
    """
    scale = (mesh.vertices.max(0) - mesh.vertices.min(0)).max()
    center = (mesh.vertices.max(0) + mesh.vertices.min(0)) / 2.0

    normalized_mesh = mesh.copy()
    normalized_mesh.vertices -= center
    normalized_mesh.vertices /= scale
    return normalized_mesh


def scale_mesh(mesh, scale):
    scale = np.array(scale)
    return trimesh.Trimesh(
        vertices=mesh.vertices * scale,
        faces=mesh.faces,
    )


def cleanup_mesh(mesh):
    """
    Retain only mesh's vertices, faces, and normals.
    """
    return trimesh.Trimesh(
        vertices=mesh.vertices,
        faces=mesh.faces,
        vertex_normals=mesh.vertex_normals,
        face_normals=mesh.face_normals,
    )


def compute_sdf_data(mesh, res):
    """
    Convert mesh to sdf voxels and a transformation matrix from mesh frame to voxel frame.
    """
    voxels_radius = 0.6
    x = np.linspace(-voxels_radius, voxels_radius, res)
    y = np.linspace(-voxels_radius, voxels_radius, res)
    z = np.linspace(-voxels_radius, voxels_radius, res)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    query_points = np.stack([X, Y, Z], axis=-1).reshape((-1, 3))

    voxels = igl.signed_distance(query_points, mesh.vertices, mesh.faces)[0]
    voxels = voxels.reshape([res, res, res])

    T_mesh_to_sdf = np.eye(4)
    T_mesh_to_sdf[:3, :3] *= (res - 1) / (voxels_radius * 2)
    T_mesh_to_sdf[:3, 3] = (res - 1) / 2

    sdf_data = {
        "voxels": voxels,
        "T_mesh_to_sdf": T_mesh_to_sdf,
    }
    return sdf_data


def voxelize_mesh(mesh, res):
    return mesh.voxelized(pitch=1.0 / res).fill()


def surface_uvs_to_trimesh_visual(surface, uvs=None, n_verts=None):
    texture = surface.get_rgba()

    if isinstance(texture, gs.textures.ImageTexture):
        if uvs is not None:
            uvs = uvs.copy()
            uvs[:, 1] = 1.0 - uvs[:, 1]
            assert texture.image_array.dtype == np.uint8
            visual = trimesh.visual.TextureVisuals(
                uv=uvs,
                material=trimesh.visual.material.SimpleMaterial(
                    image=Image.fromarray(texture.image_array), diffuse=(1.0, 1.0, 1.0, 1.0)
                ),
            )
        else:
            # fall back to color texture
            visual = trimesh.visual.ColorVisuals(vertex_colors=np.tile(texture.mean_color(), [n_verts, 1]))

    elif isinstance(texture, gs.textures.ColorTexture):
        if n_verts is None:
            gs.raise_exception("n_verts is required for color texture.")
        visual = trimesh.visual.ColorVisuals(vertex_colors=np.tile(np.array(texture.color), [n_verts, 1]))

    else:
        gs.raise_exception("Cannot get texture when generating trimesh visual.")

    return visual


def convex_decompose(mesh, morph):
    if morph.decimate:
        if mesh.vertices.shape[0] > 3:
            mesh = mesh.simplify_quadric_decimation(morph.decimate_face_num)

    # compute file name via hashing for caching
    cvx_path = get_cvx_path(mesh.vertices, mesh.faces, morph.coacd_options)

    # loading pre-computed cache if available
    is_cached_loaded = False
    if os.path.exists(cvx_path):
        gs.logger.debug("Convex decomposition file (.cvx) found in cache.")
        try:
            with open(cvx_path, "rb") as file:
                mesh_parts = pkl.load(file)
            is_cached_loaded = True
        except (EOFError, pkl.UnpicklingError):
            gs.logger.info("Ignoring corrupted cache.")

    if not is_cached_loaded:
        with gs.logger.timer("Running convex decomposition."):
            mesh = coacd.Mesh(mesh.vertices, mesh.faces)
            args = morph.coacd_options
            result = coacd.run_coacd(
                mesh,
                threshold=args.threshold,
                max_convex_hull=args.max_convex_hull,
                preprocess_mode=args.preprocess_mode,
                preprocess_resolution=args.preprocess_resolution,
                resolution=args.resolution,
                mcts_nodes=args.mcts_nodes,
                mcts_iterations=args.mcts_iterations,
                mcts_max_depth=args.mcts_max_depth,
                pca=args.pca,
                merge=args.merge,
                decimate=args.decimate,
                max_ch_vertex=args.max_ch_vertex,
                extrude=args.extrude,
                extrude_margin=args.extrude_margin,
                apx_mode=args.apx_mode,
                seed=args.seed,
            )
            mesh_parts = []
            for vs, fs in result:
                mesh_parts.append(trimesh.Trimesh(vs, fs))

            os.makedirs(os.path.dirname(cvx_path), exist_ok=True)
            with open(cvx_path, "wb") as file:
                pkl.dump(mesh_parts, file)

    return mesh_parts


def parse_visual_and_col_mesh(morph, surface):
    """
    Returns a list of meshes, each will be stored in as a `RigidGeom`.
    We parse all the submeshes in the obj file.
    If group_by_material=True, we group them based on their associated materials. This will dramatically speed up parsing, since we only need to load texture images on a group basis.
    """
    vms = gs.Mesh.from_morph_surface(morph, surface)

    # compute collision mesh
    ms = list()

    if not morph.collision:
        return vms, ms

    if morph.merge_submeshes_for_collision:
        tmeshes = []
        for vm in vms:
            tmeshes.append(vm.trimesh)
        tmesh = trimesh.util.concatenate(tmeshes)

        if morph.convexify or tmesh.is_convex or not morph.decompose_nonconvex:
            ms.append(
                gs.Mesh.from_trimesh(
                    mesh=tmesh,
                    convexify=morph.convexify,
                    decimate=morph.decimate,
                    decimate_face_num=morph.decimate_face_num,
                    surface=gs.surfaces.Collision(),
                )
            )
        else:
            tmeshes = convex_decompose(tmesh, morph)
            for tmesh in tmeshes:
                ms.append(
                    gs.Mesh.from_trimesh(
                        mesh=tmesh,
                        convexify=True,  # just to make sure
                        decimate=morph.decimate,
                        surface=gs.surfaces.Collision(),
                    )
                )

    else:
        for vm in vms:
            if morph.convexify or vm.trimesh.is_convex or not morph.decompose_nonconvex:
                ms.append(
                    gs.Mesh.from_trimesh(
                        mesh=vm.trimesh,
                        convexify=morph.convexify,
                        decimate=morph.decimate,
                        decimate_face_num=morph.decimate_face_num,
                        surface=gs.surfaces.Collision(),
                    )
                )
            else:
                tmeshes = convex_decompose(vm.trimesh, morph)
                for tmesh in tmeshes:
                    ms.append(
                        gs.Mesh.from_trimesh(
                            mesh=tmesh,
                            convexify=True,  # just to make sure
                            decimate=morph.decimate,
                            decimate_face_num=morph.decimate_face_num,
                            surface=gs.surfaces.Collision(),
                        )
                    )

    return vms, ms


def parse_mesh_trimesh(path, group_by_material, scale, surface):
    meshes = []
    for _, mesh in trimesh.load(path, force="scene", group_material=group_by_material, process=False).geometry.items():
        meshes.append(gs.Mesh.from_trimesh(mesh=mesh, scale=scale, surface=surface))
    return meshes


def trimesh_to_mesh(mesh, scale, surface):
    return gs.Mesh.from_trimesh(mesh=mesh, scale=scale, surface=surface)


ctype_to_numpy = {
    5120: (1, np.int8),  # BYTE
    5121: (1, np.uint8),  # UNSIGNED_BYTE
    5122: (2, np.int16),  # SHORT
    5123: (2, np.uint16),  # UNSIGNED_SHORT
    5124: (4, np.int32),  # INT
    5125: (4, np.uint32),  # UNSIGNED_INT
    5126: (4, np.float32),  # FLOAT
}

type_to_count = {
    "SCALAR": (1, []),
    "VEC2": (2, [2]),
    "VEC3": (3, [3]),
    "VEC4": (4, [4]),
    "MAT2": (4, [2, 2]),
    "MAT3": (9, [3, 3]),
    "MAT4": (16, [4, 4]),
}


def parse_mesh_glb(path, group_by_material, scale, surface):
    glb = pygltflib.GLTF2().load(path)
    assert glb is not None

    def parse_tree(node_index):
        node = glb.nodes[node_index]
        if node.matrix is not None:
            matrix = np.array(node.matrix, dtype=float).reshape((4, 4))
        else:
            matrix = np.identity(4, dtype=float)
            if node.translation is not None:
                translation = np.array(node.translation, dtype=float)
                translation_matrix = np.identity(4, dtype=float)
                translation_matrix[3, :3] = translation
                matrix = translation_matrix @ matrix
            if node.rotation is not None:
                rotation = np.array(node.rotation, dtype=float)  # xyzw
                rotation_matrix = np.identity(4, dtype=float)
                rotation = [rotation[3], rotation[0], rotation[1], rotation[2]]
                rotation_matrix[:3, :3] = trimesh.transformations.quaternion_matrix(rotation)[:3, :3].T
                matrix = rotation_matrix @ matrix
            if node.scale is not None:
                scale = np.array(node.scale, dtype=float)
                scale_matrix = np.diag(np.append(scale, 1))
                matrix = scale_matrix @ matrix
        mesh_list = list()
        if node.mesh is not None:
            mesh_list.append([node.mesh, np.identity(4, dtype=float)])
        for sub_node_index in node.children:
            sub_mesh_list = parse_tree(sub_node_index)
            mesh_list.extend(sub_mesh_list)
        for i in range(len(mesh_list)):
            mesh_list[i][1] = mesh_list[i][1] @ matrix
        return mesh_list

    def get_bufferview_data(buffer_view):
        buffer = glb.buffers[buffer_view.buffer]
        return glb.get_data_from_buffer_uri(buffer.uri)

    def get_data_from_accessor(accessor_index):
        accessor = glb.accessors[accessor_index]
        buffer_view = glb.bufferViews[accessor.bufferView]
        buffer_data = get_bufferview_data(buffer_view)

        data_type, data_ctype, count = accessor.type, accessor.componentType, accessor.count
        dtype = ctype_to_numpy[data_ctype][1]
        itemsize = np.dtype(dtype).itemsize
        buffer_byte_offset = (buffer_view.byteOffset or 0) + (accessor.byteOffset or 0)
        num_components = type_to_count[data_type][0]

        byte_stride = buffer_view.byteStride if buffer_view.byteStride else num_components * itemsize
        # Extract data considering byteStride
        if byte_stride == num_components * itemsize:
            # Data is tightly packed
            byte_length = count * num_components * itemsize
            data = buffer_data[buffer_byte_offset : buffer_byte_offset + byte_length]
            array = np.frombuffer(data, dtype=dtype)
            if num_components > 1:
                array = array.reshape((count, num_components))
        else:
            # Data is interleaved
            array = np.zeros((count, num_components), dtype=dtype)
            for i in range(count):
                start = buffer_byte_offset + i * byte_stride
                end = start + num_components * itemsize
                data_slice = buffer_data[start:end]
                array[i] = np.frombuffer(data_slice, dtype=dtype, count=num_components)

        return array.reshape([count] + type_to_count[data_type][1])

    glb.convert_images(pygltflib.ImageFormat.DATAURI)

    scene = glb.scenes[glb.scene]
    mesh_list = list()
    for node_index in scene.nodes:
        root_mesh_list = parse_tree(node_index)
        mesh_list.extend(root_mesh_list)

    temp_infos = dict()
    for i in range(len(mesh_list)):
        mesh = glb.meshes[mesh_list[i][0]]
        matrix = mesh_list[i][1]
        for primitive in mesh.primitives:
            if group_by_material:
                group_idx = primitive.material
            else:
                group_idx = i

            uvs0, uvs1 = None, None
            if "KHR_draco_mesh_compression" in primitive.extensions:
                import DracoPy

                KHR_index = primitive.extensions["KHR_draco_mesh_compression"]["bufferView"]
                mesh_buffer_view = glb.bufferViews[KHR_index]
                mesh_data = get_bufferview_data(mesh_buffer_view)
                mesh = DracoPy.decode(
                    mesh_data[mesh_buffer_view.byteOffset : mesh_buffer_view.byteOffset + mesh_buffer_view.byteLength]
                )
                points = mesh.points
                triangles = mesh.faces
                normals = mesh.normals if len(mesh.normals) > 0 else None
                uvs0 = mesh.tex_coord if len(mesh.tex_coord) > 0 else None

            else:
                # "primitive.attributes" records accessor indices in "glb.accessors", like:
                #      Attributes(POSITION=2, NORMAL=1, TANGENT=None, TEXCOORD_0=None, TEXCOORD_1=None,
                #                 COLOR_0=None, JOINTS_0=None, WEIGHTS_0=None)
                # parse vertices

                points = get_data_from_accessor(primitive.attributes.POSITION).astype(float)

                if primitive.indices is None:
                    indices = np.arange(points.shape[0], dtype=np.uint32)
                else:
                    indices = get_data_from_accessor(primitive.indices).astype(np.int32)

                mode = primitive.mode if primitive.mode is not None else 4

                if mode == 4:  # TRIANGLES
                    triangles = indices.reshape(-1, 3)
                elif mode == 5:  # TRIANGLE_STRIP
                    triangles = []
                    for i in range(len(indices) - 2):
                        if i % 2 == 0:
                            triangles.append([indices[i], indices[i + 1], indices[i + 2]])
                        else:
                            triangles.append([indices[i], indices[i + 2], indices[i + 1]])
                    triangles = np.array(triangles, dtype=np.uint32)
                elif mode == 6:  # TRIANGLE_FAN
                    triangles = []
                    for i in range(1, len(indices) - 1):
                        triangles.append([indices[0], indices[i], indices[i + 1]])
                    triangles = np.array(triangles, dtype=np.uint32)
                else:
                    gs.logger.warning(f"Primitive mode {mode} not supported.")
                    continue  # Skip unsupported modes

                # parse normals
                if primitive.attributes.NORMAL:
                    normals = get_data_from_accessor(primitive.attributes.NORMAL).astype(float)
                else:
                    normals = None

                # parse uvs
                if primitive.attributes.TEXCOORD_0:
                    uvs0 = get_data_from_accessor(primitive.attributes.TEXCOORD_0).astype(float)
                if primitive.attributes.TEXCOORD_1:
                    uvs1 = get_data_from_accessor(primitive.attributes.TEXCOORD_1).astype(float)

            if normals is None:
                normals = trimesh.Trimesh(points, triangles, process=False).vertex_normals
            points, normals = apply_transform(matrix, points, normals)

            if group_idx not in temp_infos.keys():
                temp_infos[group_idx] = {
                    "mat_index": primitive.material,
                    "points": [points],
                    "triangles": [triangles],
                    "normals": [normals],
                    "uvs0": [uvs0],
                    "uvs1": [uvs1],
                    "n_points": len(points),
                }

            else:
                triangles += temp_infos[group_idx]["n_points"]
                temp_infos[group_idx]["points"].append(points)
                temp_infos[group_idx]["triangles"].append(triangles)
                temp_infos[group_idx]["normals"].append(normals)
                temp_infos[group_idx]["uvs0"].append(uvs0)
                temp_infos[group_idx]["uvs1"].append(uvs1)
                temp_infos[group_idx]["n_points"] += len(points)

    meshes = list()
    for group_idx in temp_infos.keys():
        # parse images
        color_texture = None
        opacity_texture = None
        roughness_texture = None
        metallic_texture = None
        normal_texture = None
        emissive_texture = None

        alpha_cutoff = None
        double_sided = None
        ior = None
        uvs_used = 0

        if temp_infos[group_idx]["mat_index"] is not None:
            material = glb.materials[temp_infos[group_idx]["mat_index"]]
            double_sided = material.doubleSided

            # parse normal map
            if material.normalTexture is not None:
                texture = glb.textures[material.normalTexture.index]
                uvs_used = material.normalTexture.texCoord
                image_index = texture.source
                image = Image.open(uri_to_PIL(glb.images[image_index].uri))
                normal_texture = create_texture(np.array(image), None, "linear")

            # TODO: Parse occlusion
            if material.occlusionTexture is not None:
                texture = glb.textures[material.normalTexture.index]
                uvs_used = material.normalTexture.texCoord
                image_index = texture.source
                image = Image.open(uri_to_PIL(glb.images[image_index].uri))
                occlusion_texture = create_texture(np.array(image), None, "linear")

            # parse alpha mode
            if material.alphaMode == "OPAQUE":
                alpha_cutoff = 0.0
            elif material.alphaMode == "MASK":
                alpha_cutoff = material.alphaCutoff
            else:
                alpha_cutoff = None

            # parse pbr roughness and metallic
            if material.pbrMetallicRoughness is not None:
                pbr_texture = material.pbrMetallicRoughness

                # parse metallic and roughness
                roughness_image = None
                metallic_image = None
                if pbr_texture.metallicRoughnessTexture is not None:
                    texture = glb.textures[pbr_texture.metallicRoughnessTexture.index]
                    uvs_used = pbr_texture.metallicRoughnessTexture.texCoord
                    image_index = texture.source
                    image = Image.open(uri_to_PIL(glb.images[image_index].uri))
                    bands = image.split()
                    if len(bands) == 1:
                        roughness_image = np.array(bands[0])
                    else:
                        roughness_image = np.array(bands[1])  # G for roughness
                        metallic_image = np.array(bands[2])  # B for metallic
                        # metallic_image = np.array(bands[0])     # R for metallic????

                metallic_factor = None
                if pbr_texture.metallicFactor is not None:
                    metallic_factor = (pbr_texture.metallicFactor,)

                roughness_factor = None
                if pbr_texture.roughnessFactor is not None:
                    roughness_factor = (pbr_texture.roughnessFactor,)

                metallic_texture = create_texture(metallic_image, metallic_factor, "linear")
                roughness_texture = create_texture(roughness_image, roughness_factor, "linear")

                # Check if material has a base color texture
                color_image = None
                if pbr_texture.baseColorTexture is not None:
                    texture = glb.textures[pbr_texture.baseColorTexture.index]
                    uvs_used = pbr_texture.baseColorTexture.texCoord
                    image_index = texture.source
                    image = Image.open(uri_to_PIL(glb.images[image_index].uri))
                    color_image = np.array(image.convert("RGBA"))

                # parse color
                color_factor = None
                if pbr_texture.baseColorFactor is not None:
                    color_factor = np.array(pbr_texture.baseColorFactor, dtype=float)

                color_texture = create_texture(color_image, color_factor, "srgb")

            elif "KHR_materials_pbrSpecularGlossiness" in material.extensions:
                extension_material = material.extensions["KHR_materials_pbrSpecularGlossiness"]
                color_image = None
                if "diffuseTexture" in extension_material:
                    texture = extension_material["diffuseTexture"]
                    uvs_used = texture["texCoord"]
                    image = Image.open(uri_to_PIL(glb.images[texture["index"]].uri))
                    color_image = np.array(image.convert("RGBA"))

                color_factor = None
                if "diffuseFactor" in extension_material:
                    color_factor = np.array(extension_material["diffuseFactor"], dtype=float)

                color_texture = create_texture(color_image, color_factor, "srgb")

            if color_texture is not None:
                opacity_texture = color_texture.check_dim(3)
                if opacity_texture is not None:
                    opacity_texture.apply_cutoff(alpha_cutoff)

            # TODO: Parse them!
            if "KHR_materials_specular" in material.extensions:
                extension_material = material.extensions["KHR_materials_specular"]
                if "specularColorFactor" in extension_material:
                    specular_color = np.array(extension_material["specularColorFactor"], dtype=float)

            if "KHR_materials_transmission" in material.extensions:
                extension_material = material.extensions["KHR_materials_transmission"]
                specular_transmission = extension_material["transmissionFactor"]  # e.g. 1

            if "KHR_materials_ior" in material.extensions:
                extension_material = material.extensions["KHR_materials_ior"]
                ior = extension_material["ior"]  # e.g. 1.4500000476837158

            if "KHR_materials_unlit" in material.extensions:
                # No unlit material implemented in renderers. Use emissive texture.
                if color_texture is not None:
                    emissive_texture = color_texture
                    color_texture = None
            else:
                # parse emissive
                emissive_image = None
                if material.emissiveTexture is not None:
                    texture = glb.textures[material.emissiveTexture.index]
                    uvs_used = material.emissiveTexture.texCoord
                    image_index = texture.source
                    image = Image.open(uri_to_PIL(glb.images[image_index].uri))
                    if image.mode != "RGB":
                        image = image.convert("RGB")
                    emissive_image = np.array(image)

                emissive_factor = None
                if material.emissiveFactor is not None:
                    emissive_factor = np.array(material.emissiveFactor, dtype=float)

                if emissive_factor is not None and np.any(emissive_factor > 0.0):
                    emissive_texture = create_texture(emissive_image, emissive_factor, "srgb")

        # repair uv
        group_uvs = temp_infos[group_idx]["uvs1"] if uvs_used == 1 else temp_infos[group_idx]["uvs0"]
        group_points = temp_infos[group_idx]["points"]
        member_count = len(group_points)
        group_uv_exist = False

        for i in range(member_count):
            if group_uvs[i] is not None:
                group_uv_exist = True

        if group_uv_exist:
            for i in range(member_count):
                num_points = group_points[i].shape[0]
                if group_uvs[i] is None:
                    group_uvs[i] = np.zeros((num_points, 2), dtype=float)
            uvs = np.concatenate(group_uvs)
        else:
            uvs = None

        # build other group properties
        verts = np.concatenate(temp_infos[group_idx]["points"])
        normals = np.concatenate(temp_infos[group_idx]["normals"])
        faces = np.concatenate(temp_infos[group_idx]["triangles"])

        group_surface = surface.copy()
        group_surface.update_texture(
            color_texture=color_texture,
            opacity_texture=opacity_texture,
            roughness_texture=roughness_texture,
            metallic_texture=metallic_texture,
            normal_texture=normal_texture,
            emissive_texture=emissive_texture,
            ior=ior,
            double_sided=double_sided,
        )

        meshes.append(
            gs.Mesh.from_attrs(
                verts=verts,
                faces=faces,
                normals=normals,
                surface=group_surface,
                uvs=uvs,
                scale=scale,
            )
        )

    return meshes


def PIL_to_array(image):
    return np.array(image)


def uri_to_PIL(data_uri):
    with request.urlopen(data_uri) as response:
        data = response.read()
    return BytesIO(data)


def tonemapped(image):
    exposure = 0.5
    return (np.clip(np.power(image / 255 * np.power(2, exposure), 1 / 2.2), 0, 1) * 255).astype(np.uint8)


def create_texture(image, factor, encoding):
    if image is not None:
        return gs.textures.ImageTexture(image_array=image, image_color=factor, encoding=encoding)
    elif factor is not None:
        return gs.textures.ColorTexture(color=factor)
    else:
        return None


def apply_transform(matrix, positions, normals=None):
    n = positions.shape[0]
    transformed_positions = (np.hstack([positions, np.ones((n, 1))]) @ matrix)[:, :3]
    if normals is not None:
        transformed_normals = (np.hstack([normals, np.zeros((n, 1))]) @ matrix)[:, :3]
    else:
        transformed_normals = None
    return transformed_positions, transformed_normals


def create_frame(
    origin_radius=0.012, axis_radius=0.005, axis_length=1.0, head_radius=0.01, head_length=0.03, sections=12
):
    origin = create_sphere(radius=origin_radius, subdivisions=2)

    x = create_arrow(
        length=axis_length,
        radius=axis_radius,
        l_ratio=head_length / axis_length,
        r_ratio=head_radius / axis_radius,
        body_color=(0.7, 0.0, 0.0, 1.0),
        head_color=(0.7, 0.7, 0.7, 1.0),
        sections=sections,
    )
    y = create_arrow(
        length=axis_length,
        radius=axis_radius,
        l_ratio=head_length / axis_length,
        r_ratio=head_radius / axis_radius,
        body_color=(0.0, 0.7, 0.0, 1.0),
        head_color=(0.7, 0.7, 0.7, 1.0),
        sections=sections,
    )
    z = create_arrow(
        length=axis_length,
        radius=axis_radius,
        l_ratio=head_length / axis_length,
        r_ratio=head_radius / axis_radius,
        body_color=(0.0, 0.0, 0.7, 1.0),
        head_color=(0.7, 0.7, 0.7, 1.0),
        sections=sections,
    )

    x.vertices = gu.transform_by_R(x.vertices, gu.euler_to_R((0, 90, 0)))
    y.vertices = gu.transform_by_R(y.vertices, gu.euler_to_R((-90, 0, 0)))

    return trimesh.util.concatenate([origin, x, y, z])


def create_arrow(
    length=1.0,
    radius=0.02,
    l_ratio=0.25,
    r_ratio=1.5,
    body_color=(1.0, 1.0, 1.0, 1.0),
    head_color=(1.0, 1.0, 1.0, 1.0),
    sections=12,
):
    r_head = radius * r_ratio
    r_body = radius

    l_head = length * l_ratio
    l_body = length - l_head

    offset_body = np.array([0, 0, l_body / 2])
    offset_head = np.array([0, 0, l_body])

    body = trimesh.creation.cylinder(r_body, l_body, sections=sections)
    body.vertices += offset_body
    body.visual = trimesh.visual.ColorVisuals(vertex_colors=np.tile(body_color, [len(body.vertices), 1]))
    head = trimesh.creation.cone(r_head, l_head, sections=sections)
    head.vertices += offset_head
    head.visual = trimesh.visual.ColorVisuals(vertex_colors=np.tile(head_color, [len(head.vertices), 1]).astype(float))
    return trimesh.util.concatenate([body, head])


def create_line(start, end, radius=0.002, color=(1.0, 1.0, 1.0, 1.0), sections=12):
    start = np.array(start)
    end = np.array(end)
    length = np.linalg.norm(end - start)
    mesh = trimesh.creation.cylinder(radius, length, sectioins=sections)  # alonge z-axis
    mesh.vertices[:, -1] += length / 2.0
    mesh.vertices = gu.transform_by_T(mesh.vertices, gu.trans_R_to_T(start, gu.z_to_R(end - start)))
    mesh.visual = trimesh.visual.ColorVisuals(vertex_colors=np.tile(color, [len(mesh.vertices), 1]).astype(float))
    return mesh


def create_camera_frustum(camera, color):
    # camera
    camera_mesh = trimesh.load(os.path.join(get_src_dir(), "assets", "meshes", "camera/camera.obj"))

    # frustum
    near_half_height = camera.near * np.tan(np.deg2rad(camera.fov / 2))
    near_half_width = near_half_height * camera.aspect_ratio
    far_half_height = camera.far * np.tan(np.deg2rad(camera.fov / 2))
    far_half_width = far_half_height * camera.aspect_ratio

    # Define the vertices of the frustum
    vertices = np.array(
        [
            [0, 0, 0],  # apex
            [-near_half_width, -near_half_height, -camera.near],  # near bottom left
            [near_half_width, -near_half_height, -camera.near],  # near bottom right
            [near_half_width, near_half_height, -camera.near],  # near top right
            [-near_half_width, near_half_height, -camera.near],  # near top left
            [-far_half_width, -far_half_height, -camera.far],  # far bottom left
            [far_half_width, -far_half_height, -camera.far],  # far bottom right
            [far_half_width, far_half_height, -camera.far],  # far top right
            [-far_half_width, far_half_height, -camera.far],  # far top left
        ]
    )

    # Define the faces of the frustum
    faces = np.array(
        [
            # # near face
            # [1, 2, 3, 4],
            # # far face
            # [5, 6, 7, 8],
            # side face
            [2, 1, 5, 6],
            [3, 2, 6, 7],
            [4, 3, 7, 8],
            [1, 4, 8, 5],
        ]
    )

    # Create the frustum mesh
    frustum_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    frustum_mesh.visual = trimesh.visual.ColorVisuals(
        vertex_colors=np.tile(color, [len(frustum_mesh.vertices), 1]).astype(float)
    )
    return trimesh.util.concatenate([camera_mesh, frustum_mesh])


def create_box(extents=None, color=(1.0, 1.0, 1.0, 1.0), bounds=None, wireframe=False, wireframe_radius=0.002):
    if wireframe:
        if bounds is not None:
            bounds = np.array(bounds)
            extents = bounds[1] - bounds[0]
            pos = bounds.mean(axis=0)
        elif extents is not None:
            extents = np.array(extents)
            pos = np.zeros(3)
        else:
            gs.raise_exception("Neither `extents` nor `bounds` is provided.")

        vertices = np.array(
            [
                [-0.5, -0.5, -0.5],
                [0.5, -0.5, -0.5],
                [0.5, 0.5, -0.5],
                [-0.5, 0.5, -0.5],
                [-0.5, -0.5, 0.5],
                [0.5, -0.5, 0.5],
                [0.5, 0.5, 0.5],
                [-0.5, 0.5, 0.5],
            ]
        )
        vertices = vertices * extents + pos

        # Define edges connecting the vertices
        edges = [
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 0),  # Bottom face
            (4, 5),
            (5, 6),
            (6, 7),
            (7, 4),  # Top face
            (0, 4),
            (1, 5),
            (2, 6),
            (3, 7),  # Vertical edges
        ]
        mesh_vertices = []
        mesh_faces = []
        n_verts = 0
        for edge in edges:
            edge_mesh = create_line(vertices[edge[0]], vertices[edge[1]], wireframe_radius)
            mesh_vertices.append(edge_mesh.vertices)
            mesh_faces.append(edge_mesh.faces + n_verts)
            n_verts += len(edge_mesh.vertices)
        for vertex in vertices:
            vertex_mesh = create_sphere(radius=wireframe_radius)
            mesh_vertices.append(vertex_mesh.vertices + vertex)
            mesh_faces.append(vertex_mesh.faces + n_verts)
            n_verts += len(vertex_mesh.vertices)
        mesh_vertices = np.concatenate(mesh_vertices)
        mesh_faces = np.concatenate(mesh_faces)
        mesh = trimesh.Trimesh(mesh_vertices, mesh_faces)
    else:
        mesh = trimesh.creation.box(extents=extents, bounds=bounds)

    mesh.visual = trimesh.visual.ColorVisuals(vertex_colors=np.tile(color, [len(mesh.vertices), 1]).astype(float))
    return mesh


def create_sphere(radius, subdivisions=3, color=(1.0, 1.0, 1.0, 1.0)):
    mesh = trimesh.creation.icosphere(radius=radius, subdivisions=subdivisions)
    mesh.visual = trimesh.visual.ColorVisuals(vertex_colors=np.tile(color, [len(mesh.vertices), 1]).astype(float))
    return mesh


def create_tets_mesh(n_tets=1, halfsize=1.0, quats=None, randomize_halfsize=True):
    """
    Create artistic tet-based mesh for rendering particles as tets.
    """
    # create tet-based particles given positions
    vert_per_tet = 12
    face_per_tet = 20
    if quats is None:
        quats = np.tile(gu.random_quaternion(n_tets), [1, vert_per_tet]).reshape(-1, 4)

    if randomize_halfsize:
        halfsize = (
            np.tile(np.random.uniform(0.3, 1.9, size=(n_tets, 1)), [1, vert_per_tet * 3]).reshape(-1, 3) * halfsize
        )
        halfsize = (
            np.tile(np.random.uniform(0.3, 1.9, size=(n_tets * 4, 1)), [1, vert_per_tet // 4 * 3]).reshape(-1, 3)
            * halfsize
        )
        # halfsize = np.random.uniform(0.2, 1.9, size=(n_tets * vert_per_tet, 3)) * halfsize

    vertices = (
        np.tile(
            np.array(
                [
                    [0.91835, 0.836701, 0.91835],
                    [0.91835, 0.91835, 0.836701],
                    [0.836701, 0.91835, 0.91835],
                    [-0.836701, 0.91835, -0.91835],
                    [-0.91835, 0.836701, -0.91835],
                    [-0.91835, 0.91835, -0.836701],
                    [-0.836701, -0.91835, 0.91835],
                    [-0.91835, -0.836701, 0.91835],
                    [-0.91835, -0.91835, 0.836701],
                    [0.91835, -0.836701, -0.91835],
                    [0.91835, -0.91835, -0.836701],
                    [0.836701, -0.91835, -0.91835],
                ]
            ),
            [n_tets, 1],
        )
        * halfsize
    )
    vertices = gu.transform_by_quat(vertices, quats)

    faces = np.tile(
        np.array(
            [
                [0, 6, 10],
                [11, 8, 4],
                [2, 5, 7],
                [0, 1, 2],
                [3, 4, 5],
                [6, 7, 8],
                [9, 10, 11],
                [0, 10, 9],
                [9, 1, 0],
                [1, 3, 5],
                [5, 2, 1],
                [2, 7, 6],
                [6, 0, 2],
                [4, 8, 7],
                [7, 5, 4],
                [8, 11, 10],
                [10, 6, 8],
                [3, 9, 11],
                [11, 4, 3],
                [1, 9, 3],
            ]
        ),
        [n_tets, 1],
    )
    faces_offset = np.tile(np.arange(0, n_tets).reshape(-1, 1) * vert_per_tet, [1, face_per_tet * 3]).reshape(
        n_tets * face_per_tet, 3
    )
    faces += faces_offset

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    return mesh


def transform_tets_mesh_verts(vertices, positions, zs=None):
    vert_per_tet = 12
    assert len(vertices) == len(positions) * vert_per_tet
    if zs is not None:
        assert len(zs) == len(positions)
        vertices = gu.transform_by_R(vertices, np.tile(gu.z_to_R(zs), [1, vert_per_tet, 1]).reshape(-1, 3, 3))
    return vertices + np.array(np.tile(positions, [1, vert_per_tet]).reshape(-1, 3))


def create_cylinder(radius, height, sections=None, color=(1.0, 1.0, 1.0, 1.0)):
    mesh = trimesh.creation.cylinder(radius=radius, height=height, sections=sections)
    mesh.visual = trimesh.visual.ColorVisuals(vertex_colors=np.tile(color, [len(mesh.vertices), 1]).astype(float))
    return mesh


def create_plane(size=1000, color=None, normal=(0, 0, 1)):
    thickness = 1e-2  # for safety
    mesh = trimesh.creation.box(extents=[size, size, thickness])
    mesh.vertices[:, 2] -= thickness / 2
    mesh.vertices = gu.transform_by_R(mesh.vertices, gu.z_to_R(normal))
    if color is None:  # use checkerboard texture
        mesh.visual = trimesh.visual.TextureVisuals(
            uv=np.array(
                [
                    [0, 0],
                    [0, 0],
                    [0, size],
                    [0, size],
                    [size, 0],
                    [size, 0],
                    [size, size],
                    [size, size],
                ],
                dtype=float,
            ),
            material=trimesh.visual.material.SimpleMaterial(
                image=Image.open(os.path.join(get_assets_dir(), "textures/checker.png")),
            ),
        )
    else:
        mesh.visual = trimesh.visual.ColorVisuals(vertex_colors=np.tile(color, [len(mesh.vertices), 1]).astype(float))
    return mesh


def tetrahedralize_mesh(mesh, tet_cfg):
    pv_obj = pv.PolyData(
        mesh.vertices, np.concatenate([np.full((mesh.faces.shape[0], 1), mesh.faces.shape[1]), mesh.faces], axis=1)
    )
    tet = tetgen.TetGen(pv_obj)
    verts, elems = tet.tetrahedralize(**tet_cfg)
    # visualize_tet(tet, pv_obj, show_surface=False, plot_cell_qual=False)
    return verts, elems


def visualize_tet(tet, pv_data, show_surface=True, plot_cell_qual=False):
    grid = tet.grid
    if show_surface:
        grid.plot(show_edges=True)
    else:
        # get cell centroids
        cells = grid.cells.reshape(-1, 5)[:, 1:]
        cell_center = grid.points[cells].mean(1)

        # extract cells below the 0 xy plane
        mask = cell_center[:, 2] < 0
        cell_ind = mask.nonzero()[0]
        subgrid = grid.extract_cells(cell_ind)

        # advanced plotting
        if plot_cell_qual:
            cell_qual = subgrid.compute_cell_quality()["CellQuality"]
            subgrid.plot(
                scalars=cell_qual, stitle="Quality", cmap="bwr", clim=[0, 1], flip_scalars=True, show_edges=True
            )
        else:
            plotter = pv.Plotter()
            plotter.add_mesh(subgrid, "lightgrey", lighting=True, show_edges=True)
            plotter.add_mesh(pv_data, "r", "wireframe")
            plotter.add_legend([[" Input Mesh ", "r"], [" Tessellated Mesh ", "black"]])
            plotter.show()
