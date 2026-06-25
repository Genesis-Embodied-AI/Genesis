import io
import os
import platform
from contextlib import nullcontext

import xml.etree.ElementTree as ET
import numpy as np
import pygltflib
import pytest
import trimesh
from PIL import Image

import genesis as gs
import genesis.utils.geom as gu
import genesis.utils.gltf as gltf_utils
import genesis.utils.mesh as mu

from .utils import assert_allclose, assert_equal, get_hf_dataset


def extract_mesh(gs_mesh):
    """Extract vertices, normals, uvs, and faces from a gs.Mesh object."""
    vertices = gs_mesh.trimesh.vertices
    normals = gs_mesh.trimesh.vertex_normals
    uvs = gs_mesh.trimesh.visual.uv
    faces = gs_mesh.trimesh.faces

    indices = np.lexsort(
        [
            uvs[:, 1],
            uvs[:, 0],
            normals[:, 2],
            normals[:, 1],
            normals[:, 0],
            vertices[:, 2],
            vertices[:, 1],
            vertices[:, 0],
        ]
    )

    vertices = vertices[indices]
    normals = normals[indices]
    uvs = uvs[indices]
    invdices = np.argsort(indices)
    faces = invdices[faces]
    return vertices, faces, normals, uvs


def check_gs_meshes(gs_mesh1, gs_mesh2, mesh_name, vertices_tol, normals_tol):
    """Check if two gs.Mesh objects are equal."""
    vertices1, faces1, normals1, uvs1 = extract_mesh(gs_mesh1)
    vertices2, faces2, normals2, uvs2 = extract_mesh(gs_mesh2)

    assert_allclose(vertices1, vertices2, atol=vertices_tol, err_msg=f"Vertices match failed in mesh {mesh_name}.")
    assert_equal(faces1, faces2, err_msg=f"Faces match failed in mesh {mesh_name}.")
    assert_allclose(normals1, normals2, atol=normals_tol, err_msg=f"Normals match failed in mesh {mesh_name}.")
    assert_allclose(uvs1, uvs2, rtol=gs.EPS, err_msg=f"UVs match failed in mesh {mesh_name}.")


def check_gs_tm_meshes(gs_mesh, tm_mesh, mesh_name, vertices_tol, normals_tol):
    """Check if a gs.Mesh object and a trimesh.Trimesh object are equal."""
    assert_allclose(
        tm_mesh.vertices,
        gs_mesh.trimesh.vertices,
        tol=vertices_tol,
        err_msg=f"Vertices match failed in mesh {mesh_name}.",
    )
    assert_equal(
        tm_mesh.faces,
        gs_mesh.trimesh.faces,
        err_msg=f"Faces match failed in mesh {mesh_name}.",
    )
    assert_allclose(
        tm_mesh.vertex_normals,
        gs_mesh.trimesh.vertex_normals,
        tol=normals_tol,
        err_msg=f"Normals match failed in mesh {mesh_name}.",
    )
    if not isinstance(tm_mesh.visual, trimesh.visual.color.ColorVisuals):
        assert_allclose(
            tm_mesh.visual.uv,
            gs_mesh.trimesh.visual.uv,
            rtol=gs.EPS,
            err_msg=f"UVs match failed in mesh {mesh_name}.",
        )


def check_gs_tm_textures(gs_texture, tm_color, tm_image, default_value, dim, material_name, texture_name):
    """Check if a gs.Texture object and a trimesh.Texture object are equal."""
    if isinstance(gs_texture, gs.textures.ColorTexture):
        tm_color = tm_color or (default_value,) * dim
        assert_allclose(
            tm_color,
            gs_texture.color,
            rtol=gs.EPS,
            err_msg=f"Color mismatch for material {material_name} in {texture_name}.",
        )
    elif isinstance(gs_texture, gs.textures.ImageTexture):
        tm_color = tm_color or (1.0,) * dim
        assert_allclose(
            tm_color,
            gs_texture.image_color,
            rtol=gs.EPS,
            err_msg=f"Color mismatch for material {material_name} in {texture_name}.",
        )
        assert_equal(
            tm_image,
            gs_texture.image_array,
            err_msg=f"Texture mismatch for material {material_name} in {texture_name}.",
        )


def check_gs_textures(gs_texture1, gs_texture2, default_value, material_name, texture_name):
    """Check if two gs.Texture objects are equal."""
    if gs_texture1 is None:
        gs_texture1, gs_texture2 = gs_texture2, gs_texture1
    if gs_texture1 is not None:
        gs_texture1 = gs_texture1.check_simplify()
    if gs_texture2 is not None:
        gs_texture2 = gs_texture2.check_simplify()

    if isinstance(gs_texture1, gs.textures.ColorTexture):
        gs_color2 = (default_value,) * len(gs_texture1.color) if gs_texture2 is None else gs_texture2.color
        assert_allclose(
            gs_texture1.color,
            gs_color2,
            rtol=gs.EPS,
            err_msg=f"Color mismatch for material {material_name} in {texture_name}.",
        )
    elif isinstance(gs_texture1, gs.textures.ImageTexture):
        assert isinstance(gs_texture2, gs.textures.ImageTexture)
        assert_allclose(
            gs_texture1.image_color,
            gs_texture2.image_color,
            rtol=gs.EPS,
            err_msg=f"Color mismatch for material {material_name} in {texture_name}.",
        )
        assert_equal(
            gs_texture1.image_array,
            gs_texture2.image_array,
            err_msg=f"Texture mismatch for material {material_name} in {texture_name}.",
        )
    else:
        assert gs_texture1 is None and gs_texture2 is None, (
            f"Both textures should be None for material {material_name} in {texture_name}."
        )


def check_gs_surfaces(gs_surface1, gs_surface2, material_name):
    """Check if two gs.Surface objects are equal."""
    check_gs_textures(gs_surface1.texture, gs_surface2.texture, 1.0, material_name, "color")
    check_gs_textures(gs_surface1.opacity_texture, gs_surface2.opacity_texture, 1.0, material_name, "opacity")
    check_gs_textures(gs_surface1.roughness_texture, gs_surface2.roughness_texture, 1.0, material_name, "roughness")
    check_gs_textures(gs_surface1.metallic_texture, gs_surface2.metallic_texture, 0.0, material_name, "metallic")
    check_gs_textures(gs_surface1.normal_texture, gs_surface2.normal_texture, 0.0, material_name, "normal")
    check_gs_textures(gs_surface1.emissive_texture, gs_surface2.emissive_texture, 0.0, material_name, "emissive")


# ==================== Scale Tests ====================


@pytest.mark.required
@pytest.mark.parametrize("scale", [(0.5, 2.0, 8.0), (2.0, 2.0, 2.0)])
@pytest.mark.parametrize("mesh_file", ["meshes/camera/camera.glb", "meshes/axis.obj"])
def test_morph_scale(scale, mesh_file, tmp_path):
    urdf_path = tmp_path / "model.urdf"
    urdf_path.write_text(
        f"""<robot name="cannon">
              <link name="base">
                <visual>
                  <geometry><mesh filename="{mu.get_asset_path(mesh_file)}"/></geometry>
                </visual>
              </link>
            </robot>
         """
    )

    scene = gs.Scene(show_viewer=False)
    obj_orig = scene.add_entity(
        morph=gs.morphs.Mesh(
            file=mesh_file,
            file_meshes_are_zup=False,
            pos=(0, 0, 1.0),
            scale=1.0,
            convexify=False,
            fixed=True,
        ),
        surface=gs.surfaces.Default(
            color=(1.0, 0.0, 0.0, 1.0),
        ),
    )
    for vgeom in obj_orig.vgeoms:
        mesh_orig = vgeom.vmesh.trimesh
        mesh_orig.apply_transform(mu.Y_UP_TRANSFORM)
        mesh_orig.apply_scale(scale)

    obj_scaled = scene.add_entity(
        morph=gs.morphs.Mesh(
            file=mesh_file,
            file_meshes_are_zup=True,
            pos=(0, 0, 1.0),
            scale=scale,
            convexify=False,
            fixed=True,
        ),
        surface=gs.surfaces.Default(
            color=(0.0, 1.0, 0.0, 1.0),
        ),
    )
    assert obj_orig.n_vgeoms == obj_scaled.n_vgeoms

    is_isotropic = np.unique(scale).size == 1
    with nullcontext() if is_isotropic else pytest.raises(gs.GenesisException):
        robot_scaled = scene.add_entity(
            gs.morphs.URDF(
                file=urdf_path,
                file_meshes_are_zup=True,
                pos=(0, 0, 1.0),
                scale=scale,
                convexify=False,
                fixed=True,
            ),
            surface=gs.surfaces.Default(
                color=(0.0, 0.0, 1.0, 1.0),
            ),
        )
        assert robot_scaled.n_vgeoms == obj_scaled.n_vgeoms

    for i_vg in range(obj_orig.n_vgeoms):
        vgeom_orig = obj_orig.vgeoms[i_vg]
        mesh_orig = vgeom_orig.vmesh.trimesh.copy()
        w_pos_orig, w_quat_orig = gu.transform_pos_quat_by_trans_quat(
            vgeom_orig.init_pos, vgeom_orig.init_quat, obj_orig.base_link.pos, obj_orig.base_link.quat
        )
        mesh_orig.apply_transform(gu.trans_quat_to_T(w_pos_orig, w_quat_orig))

        vgeom_scaled = obj_scaled.vgeoms[i_vg]
        mesh_scaled = vgeom_scaled.vmesh.trimesh.copy()
        w_pos_scaled, w_quat_scaled = gu.transform_pos_quat_by_trans_quat(
            vgeom_scaled.init_pos, vgeom_scaled.init_quat, obj_scaled.base_link.pos, obj_scaled.base_link.quat
        )
        mesh_scaled.apply_transform(gu.trans_quat_to_T(w_pos_scaled, w_quat_scaled))
        assert_allclose(mesh_orig.vertices, mesh_scaled.vertices, tol=gs.EPS)

        if is_isotropic:
            vgeom_robot = robot_scaled.vgeoms[i_vg]
            mesh_robot_scaled = vgeom_robot.vmesh.trimesh.copy()
            w_pos_robot, w_quat_robot = gu.transform_pos_quat_by_trans_quat(
                vgeom_robot.init_pos, vgeom_robot.init_quat, robot_scaled.base_link.pos, robot_scaled.base_link.quat
            )
            mesh_robot_scaled.apply_transform(gu.trans_quat_to_T(w_pos_robot, w_quat_robot))
            assert_allclose(mesh_robot_scaled.vertices, mesh_scaled.vertices, tol=gs.EPS)


@pytest.mark.required
@pytest.mark.parametrize("mesh_file", ["glb/combined_transform.glb", "yup_zup_coverage/cannon_y_-z.stl"])
def test_urdf_scale(mesh_file, tmp_path, show_viewer):
    SCALE_FACTOR = 2.0

    asset_path = get_hf_dataset(pattern=mesh_file)

    urdf_path = tmp_path / "model.urdf"
    urdf_path.write_text(
        f"""<robot name="shoe">
              <link name="base">
                <visual>
                  <geometry><mesh filename="{os.path.join(asset_path, mesh_file)}"/></geometry>
                </visual>
              </link>
            </robot>
         """
    )

    scene = gs.Scene(
        show_viewer=show_viewer,
        show_FPS=False,
    )
    obj_1 = scene.add_entity(
        gs.morphs.URDF(
            file=urdf_path,
            convexify=False,
            fixed=True,
        ),
    )
    mesh_1 = obj_1.vgeoms[0].vmesh.trimesh
    obj_2 = scene.add_entity(
        gs.morphs.URDF(
            file=urdf_path,
            scale=SCALE_FACTOR,
            convexify=False,
            fixed=True,
        ),
    )
    mesh_2 = obj_2.vgeoms[0].vmesh.trimesh

    assert_allclose(SCALE_FACTOR * mesh_1.extents, mesh_2.extents, tol=gs.EPS)


# ==================== Y-Up Coordinate Tests ====================


@pytest.mark.required
def test_mesh_yup(show_viewer):
    scene = gs.Scene(show_viewer=show_viewer)

    asset_path = get_hf_dataset(pattern="yup_zup_coverage/*")

    glb_y = scene.add_entity(
        morph=gs.morphs.Mesh(
            file=f"{asset_path}/yup_zup_coverage/cannon_y.glb",
            convexify=False,
            fixed=True,
            file_meshes_are_zup=False,
        ),
    )
    glb_z = scene.add_entity(
        morph=gs.morphs.Mesh(
            file=f"{asset_path}/yup_zup_coverage/cannon_z.glb",
            convexify=False,
            fixed=True,
            file_meshes_are_zup=True,
        ),
    )
    stl_y = scene.add_entity(
        morph=gs.morphs.Mesh(
            file=f"{asset_path}/yup_zup_coverage/cannon_y_-z.stl",
            convexify=False,
            fixed=True,
            file_meshes_are_zup=False,
        ),
    )
    stl_z = scene.add_entity(
        morph=gs.morphs.Mesh(
            file=f"{asset_path}/yup_zup_coverage/cannon_z_y.stl",
            convexify=False,
            fixed=True,
        ),
    )
    obj_y = scene.add_entity(
        morph=gs.morphs.Mesh(
            file=f"{asset_path}/yup_zup_coverage/cannon_y_-z.obj",
            convexify=False,
            fixed=True,
            file_meshes_are_zup=False,
        ),
    )
    obj_z = scene.add_entity(
        morph=gs.morphs.Mesh(
            file=f"{asset_path}/yup_zup_coverage/cannon_z_y.obj",
            convexify=False,
            fixed=True,
        ),
    )

    if show_viewer:
        scene.build()

    assert not glb_y.vgeoms[0].vmesh.metadata["imported_as_zup"]
    assert not glb_z.vgeoms[0].vmesh.metadata["imported_as_zup"]
    assert not stl_y.vgeoms[0].vmesh.metadata["imported_as_zup"]
    assert stl_z.vgeoms[0].vmesh.metadata["imported_as_zup"]
    assert not obj_y.vgeoms[0].vmesh.metadata["imported_as_zup"]
    assert obj_z.vgeoms[0].vmesh.metadata["imported_as_zup"]

    bounding_boxes = []
    for entity in (glb_y, glb_z, stl_y, stl_z, obj_y, obj_z):
        tmeshes = []
        for vgeom in entity.vgeoms:
            tmesh = vgeom.vmesh.trimesh.copy()
            w_pos, w_quat = gu.transform_pos_quat_by_trans_quat(
                vgeom.init_pos, vgeom.init_quat, vgeom.link.pos, vgeom.link.quat
            )
            tmesh.apply_transform(gu.trans_quat_to_T(w_pos, w_quat))
            tmeshes.append(tmesh)
        combined = trimesh.util.concatenate(tmeshes)
        assert_allclose(combined.center_mass, (-0.012, -0.142, 0.397), tol=0.002)
        bounding_boxes.append(combined.bounding_box.bounds)
    # FIXME: The STL files are actually different from the glTF...
    # assert_allclose(np.diff(bounding_boxes, axis=0), 0.0, tol=0.001)


@pytest.mark.required
@pytest.mark.parametrize(
    "mesh_file, file_meshes_are_zup",
    [("yup_zup_coverage/cannon_z.glb", True), ("yup_zup_coverage/cannon_y_-z.stl", False)],
)
def test_urdf_yup(mesh_file, file_meshes_are_zup, tmp_path, show_viewer):
    asset_path = get_hf_dataset(pattern=mesh_file)
    urdf_path = tmp_path / "model.urdf"
    urdf_path.write_text(
        f"""<robot name="cannon">
              <link name="base">
                <visual>
                  <geometry><mesh filename="{os.path.join(asset_path, mesh_file)}"/></geometry>
                </visual>
              </link>
            </robot>
         """
    )

    scene = gs.Scene(show_viewer=show_viewer)
    robot = scene.add_entity(
        gs.morphs.URDF(
            file=urdf_path,
            convexify=False,
            fixed=True,
            file_meshes_are_zup=file_meshes_are_zup,
        ),
    )

    if show_viewer:
        scene.build()

    tmeshes = []
    for vgeom in robot.vgeoms:
        tmesh = vgeom.vmesh.trimesh.copy()
        tmesh.apply_transform(gu.trans_quat_to_T(vgeom.link.pos, vgeom.link.quat))
        tmeshes.append(tmesh)
    combined = trimesh.util.concatenate(tmeshes)
    assert_allclose(combined.center_mass, (-0.012, -0.142, 0.397), tol=0.002)


# ==================== Geometry Parsing Tests ====================


@pytest.mark.required
@pytest.mark.parametrize("precision", ["32"])
@pytest.mark.parametrize("glb_file", ["glb/combined_srt.glb", "glb/combined_transform.glb"])
def test_glb_parse_geometry(glb_file, tol):
    """Test glb mesh geometry parsing."""
    asset_path = get_hf_dataset(pattern=glb_file)
    glb_file = os.path.join(asset_path, glb_file)
    gs_meshes = gltf_utils.parse_mesh_glb(
        glb_file,
        group_by_material=False,
        scale=None,
        is_mesh_zup=True,
        surface=gs.surfaces.Default(),
    )

    tm_scene = trimesh.load(glb_file, process=False)
    tm_meshes = {}
    for node_name in tm_scene.graph.nodes_geometry:
        transform, geometry_name = tm_scene.graph[node_name]
        ts_mesh = tm_scene.geometry[geometry_name].copy(include_cache=True)
        ts_mesh = ts_mesh.apply_transform(transform)
        tm_meshes[geometry_name] = ts_mesh
    assert len(tm_meshes) == len(gs_meshes)

    for gs_mesh in gs_meshes:
        mesh_name = gs_mesh.metadata["name"]
        tm_mesh = tm_meshes[mesh_name]
        check_gs_tm_meshes(gs_mesh, tm_mesh, mesh_name, tol, tol)


@pytest.mark.required
@pytest.mark.parametrize("glb_file", ["glb/tycoon_draco_no_normal.glb", "glb/tycoon_with_normal_draco.glb"])
def test_glb_draco_missing_normals_texcoord(glb_file):
    # Normals and tex_coord are not always present in GLB files, typically for Draco-compressed ones.
    asset_path = get_hf_dataset(pattern=glb_file)
    glb_path = os.path.join(asset_path, glb_file)
    gs_meshes = gltf_utils.parse_mesh_glb(
        glb_path,
        group_by_material=False,
        scale=None,
        is_mesh_zup=True,
        surface=gs.surfaces.Default(),
    )

    assert len(gs_meshes) > 0, "Expected at least one mesh"
    for gs_mesh in gs_meshes:
        verts = gs_mesh.trimesh.vertices
        faces = gs_mesh.trimesh.faces
        assert verts.shape[0] > 0, "Mesh has no vertices"
        assert verts.shape[1] == 3, "Vertices should be 3D"
        assert faces.shape[0] > 0, "Mesh has no faces"
        assert faces.shape[1] == 3, "Faces should be triangles"


@pytest.mark.required
def test_urdf_mesh_processing(tmp_path, show_viewer):
    stl_file = "1707/base_link.stl"
    asset_path = get_hf_dataset(pattern=stl_file)
    stl_path = os.path.join(asset_path, stl_file)

    urdf_path = tmp_path / "model.urdf"
    urdf_path.write_text(
        f"""<robot name="shoe">
              <link name="base">
                <visual>
                  <geometry><mesh filename="{stl_path}"/></geometry>
                </visual>
              </link>
            </robot>
         """
    )

    scene = gs.Scene(
        show_viewer=show_viewer,
        show_FPS=False,
    )
    obj = scene.add_entity(
        gs.morphs.Mesh(
            file=stl_path,
        ),
    )
    robot = scene.add_entity(
        gs.morphs.URDF(
            file=urdf_path,
        ),
    )

    tmesh_obj_col = obj.geoms[0].mesh.trimesh
    tmesh_obj_vis = obj.vgeoms[0].vmesh.trimesh
    tmesh_robot_vis = robot.vgeoms[0].vmesh.trimesh

    assert len(tmesh_obj_col.vertices) != len(tmesh_obj_vis.vertices)
    assert len(tmesh_obj_vis.vertices) == len(tmesh_robot_vis.vertices)
    assert len(tmesh_obj_vis.faces) == len(tmesh_robot_vis.faces)

    tmesh = trimesh.Trimesh(vertices=tmesh_obj_vis.vertices, faces=tmesh_obj_vis.faces, process=True)
    assert len(tmesh.vertices) != len(tmesh_obj_vis.vertices)


# ==================== Material/Texture Parsing Tests ====================


@pytest.mark.required
@pytest.mark.parametrize("glb_file", ["glb/chopper.glb"])
def test_glb_parse_material(glb_file):
    """Test glb mesh geometry parsing."""
    asset_path = get_hf_dataset(pattern=glb_file)
    glb_file = os.path.join(asset_path, glb_file)
    gs_meshes = gltf_utils.parse_mesh_glb(
        glb_file,
        group_by_material=True,
        scale=None,
        is_mesh_zup=True,
        surface=gs.surfaces.Default(),
    )

    tm_scene = trimesh.load(glb_file, process=False)
    tm_materials = {}
    for geometry_name in tm_scene.geometry:
        ts_mesh = tm_scene.geometry[geometry_name]
        ts_material = ts_mesh.visual.material.copy()
        tm_materials[ts_material.name] = ts_material

    assert len(tm_materials) == len(gs_meshes)
    for gs_mesh in gs_meshes:
        material_name = gs_mesh.metadata["name"]
        tm_material = tm_materials[material_name]
        gs_material = gs_mesh.surface

        assert isinstance(tm_material, trimesh.visual.material.PBRMaterial)
        check_gs_tm_textures(
            gs_material.texture,
            tm_material.baseColorFactor,
            np.array(tm_material.baseColorTexture),
            1.0,
            3,
            material_name,
            "color",
        )

        if tm_material.metallicRoughnessTexture is not None:
            tm_mr_image = np.array(tm_material.metallicRoughnessTexture)
            tm_roughness_image = tm_mr_image[:, :, 1]
            tm_metallic_image = tm_mr_image[:, :, 2]
        else:
            tm_roughness_image, tm_metallic_image = None, None
        check_gs_tm_textures(
            gs_material.roughness_texture,
            tm_material.roughnessFactor,
            tm_roughness_image,
            1.0,
            1,
            material_name,
            "roughness",
        )
        check_gs_tm_textures(
            gs_material.metallic_texture,
            tm_material.metallicFactor,
            tm_metallic_image,
            0.0,
            1,
            material_name,
            "metallic",
        )

        if tm_material.emissiveFactor is None and tm_material.emissiveFactor is None:
            assert gs_material.emissive_texture is None
        else:
            check_gs_tm_textures(
                gs_material.emissive_texture,
                tm_material.emissiveFactor,
                np.array(tm_material.emissiveTexture),
                0.0,
                3,
                material_name,
                "emissive",
            )


@pytest.mark.required
def test_glb_shared_texture_not_duplicated(tmp_path):
    from genesis.vis.batch_renderer import GenesisGeomRetriever

    n_submeshes = 16
    texture_size = 64

    yy, xx = np.mgrid[0:texture_size, 0:texture_size]
    checker = (((xx // 8) % 2) ^ ((yy // 8) % 2)).astype(np.uint8) * 255
    texture = Image.fromarray(np.stack([checker, checker, checker], axis=-1))
    mesh = trimesh.Trimesh(
        vertices=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        faces=[[0, 1, 2]],
        visual=trimesh.visual.TextureVisuals(
            uv=[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
            material=trimesh.visual.material.PBRMaterial(baseColorTexture=texture, doubleSided=True),
        ),
        process=False,
    )

    # Many nodes referencing a single shared geometry/material - the case that used to blow up host memory.
    tm_scene = trimesh.Scene()
    for i in range(n_submeshes):
        transform = np.eye(4)
        transform[:3, 3] = (2.0 * i, 0.0, 0.0)
        tm_scene.add_geometry(mesh, node_name=f"instance_{i:02d}", geom_name="shared_mesh", transform=transform)
    glb_path = tmp_path / "shared_texture.glb"
    tm_scene.export(glb_path)

    scene = gs.Scene(show_viewer=False)
    entity = scene.add_entity(
        gs.morphs.Mesh(
            file=str(glb_path),
            collision=False,
            fixed=True,
            file_meshes_are_zup=False,
        ),
    )

    # Submeshes are kept separate (group_by_material defaults to False to preserve baked convex decompositions), but
    # they share one Surface whose resolved RGBA texture is memoized, so a single image array backs all of them rather
    # than one full-resolution copy per submesh.
    vgeoms = entity.vgeoms
    assert len(vgeoms) == n_submeshes
    assert len({id(geom.surface) for geom in vgeoms}) == 1
    # Hold every array alive before counting so that ids cannot be recycled by the garbage collector.
    rgba_arrays = [geom.surface.get_rgba().image_array for geom in vgeoms]
    assert len({id(array) for array in rgba_arrays}) == 1

    scene.build()
    retriever = GenesisGeomRetriever(scene.sim.rigid_solver, seg_level="entity")
    retriever.build()
    static_args = retriever.retrieve_rigid_meshes_static()
    assert len(static_args["tex_widths"]) == 1


@pytest.mark.required
def test_glb_multi_primitive_distinct_materials(tmp_path):
    # A single glTF mesh node may hold several primitives with distinct materials/textures. With the default
    # group_by_material=False, each primitive becomes its own visual mesh so all its textures render (merging them
    # under the first primitive's material would silently drop the others), but the node is one physical body, so
    # its primitives are merged into a single collision geom. Separately colliding pieces are authored as nodes.
    texture_size = 16
    prim_colors = np.array([[220, 30, 30], [30, 50, 220]], dtype=np.uint8)
    n_prims = len(prim_colors)

    # One axis-aligned box per primitive plus one solid-color PNG per primitive, packed into a single glTF mesh
    # (one node) with one primitive per material. The boxes share a face so their union is convex, so merging them
    # per node yields a single convex collision geom.
    box = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    parts = []
    accessors = []
    for i in range(n_prims):
        positions = (box.vertices + np.array([0.5 + i, 0.0, 0.0])).astype(np.float32)
        uvs = np.zeros((len(positions), 2), dtype=np.float32)
        indices = box.faces.astype(np.uint32)
        parts.append((positions.tobytes(), 34962))
        parts.append((uvs.tobytes(), 34962))
        parts.append((indices.tobytes(), 34963))
        accessors.append(
            pygltflib.Accessor(
                bufferView=3 * i,
                componentType=5126,
                count=len(positions),
                type="VEC3",
                min=positions.min(axis=0).tolist(),
                max=positions.max(axis=0).tolist(),
            )
        )
        accessors.append(pygltflib.Accessor(bufferView=3 * i + 1, componentType=5126, count=len(uvs), type="VEC2"))
        accessors.append(
            pygltflib.Accessor(bufferView=3 * i + 2, componentType=5125, count=indices.size, type="SCALAR")
        )
    for color in prim_colors:
        image = Image.fromarray(np.broadcast_to(color, (texture_size, texture_size, 3)).copy())
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        parts.append((buffer.getvalue(), None))

    blob = b""
    buffer_views = []
    for data, target in parts:
        blob += b"\x00" * ((4 - len(blob) % 4) % 4)
        buffer_views.append(pygltflib.BufferView(buffer=0, byteOffset=len(blob), byteLength=len(data), target=target))
        blob += data

    gltf = pygltflib.GLTF2(
        scenes=[pygltflib.Scene(nodes=[0])],
        nodes=[pygltflib.Node(mesh=0)],
        meshes=[
            pygltflib.Mesh(
                primitives=[
                    pygltflib.Primitive(
                        attributes=pygltflib.Attributes(POSITION=3 * i, TEXCOORD_0=3 * i + 1),
                        indices=3 * i + 2,
                        material=i,
                    )
                    for i in range(n_prims)
                ]
            )
        ],
        materials=[
            pygltflib.Material(
                pbrMetallicRoughness=pygltflib.PbrMetallicRoughness(
                    baseColorTexture=pygltflib.TextureInfo(index=i, texCoord=0)
                )
            )
            for i in range(n_prims)
        ],
        textures=[pygltflib.Texture(source=i) for i in range(n_prims)],
        images=[pygltflib.Image(bufferView=3 * n_prims + i, mimeType="image/png") for i in range(n_prims)],
        accessors=accessors,
        bufferViews=buffer_views,
        buffers=[pygltflib.Buffer(byteLength=len(blob))],
    )
    gltf.set_binary_blob(blob)
    glb_path = tmp_path / "multi_primitive.glb"
    gltf.save_binary(str(glb_path))

    gs_meshes = gltf_utils.parse_mesh_glb(
        str(glb_path),
        group_by_material=False,
        scale=None,
        is_mesh_zup=False,
        surface=gs.surfaces.Default(),
    )

    # One visual mesh per material rather than a single merged mesh, each keeping its own distinct base-color texture.
    assert len(gs_meshes) == n_prims
    dominant_channels = []
    for gs_mesh in gs_meshes:
        texture = gs_mesh.surface.texture
        assert isinstance(texture, gs.textures.ImageTexture)
        dominant_channels.append(int(np.argmax(texture.image_array[..., :3].mean(axis=(0, 1)))))
    assert sorted(dominant_channels) == [0, 2]

    scene = gs.Scene(show_viewer=False)
    entity = scene.add_entity(
        gs.morphs.Mesh(
            file=str(glb_path),
            file_meshes_are_zup=False,
        ),
    )
    scene.build()
    # Textures render per material (one visual geom each), but the node is a single physical body, so its
    # primitives are merged into one collision geom rather than split by material.
    assert len(entity.vgeoms) == n_prims
    assert len(entity.geoms) == 1


@pytest.fixture
def material_mjcf(tmp_path):
    """Generate an MJCF model with materials and geom-level colors."""
    mjcf = ET.Element("mujoco", model="materials")
    default = ET.SubElement(mjcf, "default")
    ET.SubElement(default, "joint", armature="0.0")

    # Define materials with different properties (at top level, not in default)
    asset = ET.SubElement(mjcf, "asset")
    ET.SubElement(
        asset,
        "material",
        name="red_material",
        rgba="1.0 0.0 0.0 0.6",
        specular="0.5",
        shininess="0.3",
    )

    worldbody = ET.SubElement(mjcf, "worldbody")
    floor = ET.SubElement(worldbody, "body", name="/worldbody/floor")
    ET.SubElement(floor, "geom", type="plane", pos="0. 0. 0.", size="40. 40. 40.")

    # Box with red material (material-level rgba)
    box1 = ET.SubElement(worldbody, "body", name="/worldbody/box1", pos="-0.3 0. 0.3")
    ET.SubElement(
        box1,
        "geom",
        type="box",
        size="0.2 0.2 0.2",
        pos="0. 0. 0.",
        material="red_material",
        contype="0",
        conaffinity="0",
    )
    ET.SubElement(box1, "joint", name="/worldbody/box1_joint", type="free")

    # Box with geom-level rgba (no material, tests geom-level color)
    box2 = ET.SubElement(worldbody, "body", name="/worldbody/box2", pos="0.0 0. 0.6")
    ET.SubElement(
        box2,
        "geom",
        type="box",
        size="0.2 0.2 0.2",
        pos="0. 0. 0.",
        rgba="0.0 1.0 0.0 1.0",
        contype="0",
        conaffinity="0",
    )
    ET.SubElement(box2, "joint", name="/worldbody/box2_joint", type="free")

    # Write to temporary file
    xml_tree = ET.ElementTree(mjcf)
    file_path = str(tmp_path / "material_mjcf.xml")
    xml_tree.write(file_path, encoding="utf-8", xml_declaration=True)
    return file_path


@pytest.mark.parametrize("precision", ["32"])
def test_mjcf_parse_material(material_mjcf, tol):
    """Test that MJCF materials and geom colors are correctly parsed."""
    scene = gs.Scene()
    entity = scene.add_entity(
        gs.morphs.MJCF(
            file=material_mjcf,
            scale=1.0,
            convexify=False,
        ),
        material=gs.materials.Rigid(rho=1000.0),
    )
    scene.build()

    # Find boxes by their names
    box1_vgeom = None
    box2_vgeom = None
    for link in entity.links:
        if link.name == "/worldbody/box1":
            box1_vgeom = link.vgeoms[0]
        elif link.name == "/worldbody/box2":
            box2_vgeom = link.vgeoms[0]
    assert box1_vgeom is not None, "box1 not found"
    assert box2_vgeom is not None, "box2 not found"

    # Check red material (box1) - material-level rgba
    box1_surface = box1_vgeom.vmesh.surface
    box1_roughness = mu.glossiness_to_roughness(0.3 * 128.0)
    check_gs_textures(
        box1_surface.diffuse_texture, gs.textures.ColorTexture(color=(1.0, 0.0, 0.0)), 1.0, "box1", "color"
    )
    check_gs_textures(box1_surface.roughness_texture, None, box1_roughness, "box1", "roughness")
    check_gs_textures(box1_surface.opacity_texture, None, 0.6, "box1", "opacity")

    box2_surface = box2_vgeom.vmesh.surface
    check_gs_textures(
        box2_surface.diffuse_texture, gs.textures.ColorTexture(color=(0.0, 1.0, 0.0)), 1.0, "box2", "color"
    )
    check_gs_textures(box2_surface.opacity_texture, None, 1.0, "box2", "opacity")


@pytest.fixture
def normals_mjcf(tmp_path):
    # MuJoCo packs vertices and normals in separately-addressed blocks. The first mesh has a different number of
    # vertices than normals (a flat-shaded bipyramid: shared positions, per-face normals), which shifts the second
    # mesh's normal block away from its vertex block. The second mesh (a smooth icosphere, whose normals are exactly
    # radial) is the one whose normals must be read at the correct offset and routed through the per-face normal
    # indices, so it is the comparison target.
    n_sides = 24
    radius, half_height = 0.3, 0.4
    angles = 2.0 * np.pi * np.arange(n_sides) / n_sides
    ring = np.stack([radius * np.cos(angles), radius * np.sin(angles), np.zeros(n_sides)], axis=1)
    bipyr_verts = np.vstack([ring, (0.0, 0.0, half_height), (0.0, 0.0, -half_height)])
    top_idx, bot_idx = n_sides, n_sides + 1
    bipyr_faces = []
    for k in range(n_sides):
        a, b = k, (k + 1) % n_sides
        bipyr_faces.append((a, b, top_idx))
        bipyr_faces.append((b, a, bot_idx))
    bipyr_normals = []
    for a, b, c in bipyr_faces:
        normal = np.cross(bipyr_verts[b] - bipyr_verts[a], bipyr_verts[c] - bipyr_verts[a])
        bipyr_normals.append(normal / np.linalg.norm(normal))
    obj_lines = [f"v {v[0]} {v[1]} {v[2]}" for v in bipyr_verts]
    obj_lines += [f"vn {n[0]} {n[1]} {n[2]}" for n in bipyr_normals]
    for i, (a, b, c) in enumerate(bipyr_faces):
        obj_lines.append(f"f {a + 1}//{i + 1} {b + 1}//{i + 1} {c + 1}//{i + 1}")
    bipyr_path = tmp_path / "bipyr.obj"
    bipyr_path.write_text("\n".join(obj_lines) + "\n")

    ico_path = tmp_path / "ico.obj"
    trimesh.creation.icosphere(radius=0.3, subdivisions=3).export(str(ico_path), include_normals=True)

    mjcf = ET.Element("mujoco", model="normals")
    asset = ET.SubElement(mjcf, "asset")
    ET.SubElement(asset, "mesh", name="bipyr", file=str(bipyr_path))
    ET.SubElement(asset, "mesh", name="ico", file=str(ico_path))
    worldbody = ET.SubElement(mjcf, "worldbody")
    body = ET.SubElement(worldbody, "body", name="/worldbody/obj")
    ET.SubElement(body, "freejoint")
    ET.SubElement(body, "geom", type="mesh", mesh="bipyr", contype="0", conaffinity="0")
    ET.SubElement(body, "geom", type="mesh", mesh="ico", contype="0", conaffinity="0")

    file_path = str(tmp_path / "normals_mjcf.xml")
    ET.ElementTree(mjcf).write(file_path, encoding="utf-8", xml_declaration=True)
    return file_path, str(ico_path)


def test_mjcf_parse_mesh_normals(normals_mjcf):
    mjcf_path, ico_path = normals_mjcf

    scene = gs.Scene()
    entity = scene.add_entity(
        gs.morphs.MJCF(
            file=mjcf_path,
            convexify=False,
        ),
    )
    scene.build()

    # The icosphere is the second geom, so its normal block starts at a different offset than its vertex block.
    ico_vgeom = entity.links[0].vgeoms[1]
    parsed = ico_vgeom.vmesh.trimesh
    raw = trimesh.load(ico_path, process=False)

    # MuJoCo reorders and splits vertices, so sort both meshes into a canonical order before comparing.
    parsed_order = np.lexsort((parsed.vertices[:, 2], parsed.vertices[:, 1], parsed.vertices[:, 0]))
    raw_order = np.lexsort((raw.vertices[:, 2], raw.vertices[:, 1], raw.vertices[:, 0]))

    assert_allclose(parsed.vertices[parsed_order], raw.vertices[raw_order], atol=1e-4)
    assert_allclose(parsed.vertex_normals[parsed_order], raw.vertex_normals[raw_order], atol=1e-3)


@pytest.mark.required
def test_2_channels_luminance_alpha_textures(show_viewer):
    scene = gs.Scene(
        show_viewer=show_viewer,
        show_FPS=False,
    )
    asset_path = get_hf_dataset(pattern="fridge/*")
    fridge = scene.add_entity(
        gs.morphs.URDF(
            file=f"{asset_path}/fridge/fridge.urdf",
            fixed=True,
        )
    )


@pytest.mark.required
def test_plane_texture_path_preservation(show_viewer):
    """Test that plane primitives preserve texture paths in metadata."""
    scene = gs.Scene(show_viewer=show_viewer, show_FPS=False)
    plane = scene.add_entity(gs.morphs.Plane())

    # The texture path should be stored in metadata
    assert plane.vgeoms[0].vmesh.metadata["texture_path"] == "textures/checker.png"


@pytest.mark.required
@pytest.mark.parametrize(
    "n_channels, float_type",
    [
        (1, np.float32),  # grayscale → H×W
        (2, np.float64),  # L+A       → H×W×2
    ],
)
def test_urdf_with_float_texture_glb(tmp_path, show_viewer, n_channels, float_type):
    vertices = np.array(
        [[-0.5, -0.5, 0.0], [0.5, -0.5, 0.0], [0.5, 0.5, 0.0], [-0.5, 0.5, 0.0]],
        dtype=np.float32,
    )
    faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.uint32)

    mesh = trimesh.Trimesh(vertices, faces, process=False)

    H = W = 16
    if n_channels == 1:
        img = np.random.rand(H, W).astype(float_type)
    else:
        img = np.random.rand(H, W, n_channels).astype(float_type)

    mesh.visual = trimesh.visual.texture.TextureVisuals(
        uv=np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32),
        material=trimesh.visual.material.SimpleMaterial(image=img),
    )

    glb_path = tmp_path / f"tex_{n_channels}c.glb"
    urdf_path = tmp_path / f"tex_{n_channels}c.urdf"
    trimesh.Scene([mesh]).export(glb_path)
    urdf_path.write_text(
        f"""<robot name="tex{n_channels}c">
              <link name="base">
                <visual>
                  <geometry><mesh filename="{glb_path}"/></geometry>
                </visual>
              </link>
            </robot>
         """
    )

    scene = gs.Scene(show_viewer=show_viewer, show_FPS=False)
    robot = scene.add_entity(
        gs.morphs.URDF(
            file=urdf_path,
        ),
    )


# ==================== Surface Reconstruction Tests ====================


@pytest.mark.required
def test_splashsurf_surface_reconstruction(show_viewer):
    scene = gs.Scene(
        show_viewer=show_viewer,
    )
    scene.add_entity(
        material=gs.materials.SPH.Liquid(),
        morph=gs.morphs.Box(
            pos=(0.15, 0.15, 0.22),
            size=(0.25, 0.25, 0.4),
        ),
        surface=gs.surfaces.Default(
            color=(0.2, 0.6, 1.0, 1.0),
            vis_mode="recon",
        ),
    )
    cam = scene.add_camera(
        pos=(1.3, 1.3, 0.8),
        lookat=(0.0, 0.0, 0.2),
        GUI=show_viewer,
    )
    scene.build()
    cam.render(rgb=True, depth=False, segmentation=False, colorize_seg=False, normal=False)


# ==================== Mesh Processing/Caching Tests ====================


# FIXME: This test is taking too much time on some platform (~1200s)
# @pytest.mark.required
def test_convex_decompose_cache(monkeypatch):
    # Check if the convex decomposition cache is correctly tracked regardless of the scale

    # Monkeypatch the get_cvx_path function to track the cache path
    seen_paths = []
    real_get_cvx_path = mu.get_cvx_path

    def wrapped_get_cvx_path(verts, faces, opts):
        path = real_get_cvx_path(verts, faces, opts)
        seen_paths.append(path)
        return path

    monkeypatch.setattr(mu, "get_cvx_path", wrapped_get_cvx_path)

    # Monkeypatch the convex_decompose function to track the convex decomposition result
    seen_results = []
    real_convex_decompose = mu.convex_decompose

    def wrapped_convex_decompose(mesh, opts):
        result = real_convex_decompose(mesh, opts)
        seen_results.append(result)
        return result

    monkeypatch.setattr(mu, "convex_decompose", wrapped_convex_decompose)

    # First scene building to create the cache
    scene = gs.Scene(
        show_viewer=False,
    )
    first_scale = 2.0
    scene.add_entity(
        morph=gs.morphs.Mesh(
            file="meshes/duck.obj",
            scale=first_scale,
            pos=(0, 0, 1.0),
            quat=(0, 0, 0, 1),
        ),
    )
    scene.build()

    # Second scene building, duck with different scale, translation, and rotation
    scene = gs.Scene(
        show_viewer=False,
    )
    second_scale = 4.0
    scene.add_entity(
        morph=gs.morphs.Mesh(
            file="meshes/duck.obj",
            scale=second_scale,
            pos=(1.0, 0, 1.0),
            quat=(1, 0, 0, 0),
        ),
    )
    scene.build()

    assert len(seen_paths) == 2
    assert len(seen_results) == 2

    # scaled mesh should have the same cache path as the original mesh
    cached_path = seen_paths[0]
    scaled_path = seen_paths[-1]
    assert cached_path == scaled_path

    # check if the scaled parts match the scaled version of the original parts
    cached_parts = seen_results[0]
    scaled_parts = seen_results[-1]
    assert len(scaled_parts) == len(cached_parts)
    for scaled_part, cached_part in zip(scaled_parts, cached_parts):
        assert_allclose(scaled_part.vertices, cached_part.vertices * (second_scale / first_scale), rtol=1e-6)
        assert_equal(scaled_part.faces, cached_part.faces)
