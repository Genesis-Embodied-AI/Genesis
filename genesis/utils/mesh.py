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

class MeshInfo:
    def __init__(self, surface):
        self.surface = surface
        self.verts = list()
        self.faces = list()
        self.normals = list()
        self.uvs = list()
        self.n_points = 0
        self.n_members = 0
        self.uvs_exist = False
    
    def append(self, verts, faces, normals, uvs):
        faces += self.n_points
        self.verts.append(verts)
        self.faces.append(faces)
        self.normals.append(normals)
        self.uvs.append(uvs)
        self.n_points += verts.shape[0]
        self.n_members += 1
        if uvs is not None:
            self.uvs_exist = True
    
    def export_mesh(self, scale):
        if self.uvs_exist:
            for i in range(self.n_members):
                if self.uvs[i] is None:
                    self.uvs[i] = np.zeros((self.verts[i].shape[0], 2), dtype=float)
            uvs = np.concatenate(self.uvs, axis=0)
        else:
            uvs = None

        verts = np.concatenate(self.verts, axis=0)
        faces = np.concatenate(self.faces, axis=0)
        normals = np.concatenate(self.normals, axis=0)
        
        return gs.Mesh.from_attrs(
            verts=verts,
            faces=faces,
            normals=normals,
            surface=self.surface,
            uvs=uvs,
            scale=scale,
        )
    
class MeshInfoGroup:
    def __init__(self):
        self.infos = dict()

    def append(self, name, verts, faces, normals, uvs, surface):
        if name not in self.infos:
            self.infos[name] = MeshInfo(surface)
        self.infos[name].append(verts, faces, normals, uvs)
    
    def export_meshes(self, scale):
        return [self.infos[name].export_mesh(scale) for name in self.infos]
    

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

        num_vertices = len(tmesh.vertices)
        if num_vertices > 5000:
            gs.logger.warning(
                f"Mesh '{morph.file}' contains many vertices ({num_vertices}). Consider setting "
                "'morph.decimate=True' to speed up collision detection."
            )
        if not tmesh.is_convex and not (morph.convexify or morph.decompose_nonconvex):
            gs.logger.warning(
                f"Mesh '{morph.file}' is non-convex. Consider setting 'morph.decompose_nonconvex=True' "
                "or 'morph.convexify=True' to speed up collision detection."
            )

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


def adjust_alpha_cutoff(alpha_cutoff, alpha_mode):
    if alpha_mode == 0:     # OPAQUE
        return 0.0
    elif alpha_mode == 1:   # MASK
        return alpha_cutoff
    else:                   # BLEND
        return None

def PIL_to_array(image):
    return np.array(image)

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


def create_plane(size=1e3, color=None, normal=(0, 0, 1)):
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
