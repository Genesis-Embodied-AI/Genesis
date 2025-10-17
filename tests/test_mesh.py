import os
import sys

import numpy as np
import pytest
import trimesh

import genesis as gs
import genesis.utils.gltf as gltf_utils
import genesis.utils.usda as usda_utils
import genesis.utils.mesh as mesh_utils

from .utils import assert_allclose, assert_array_equal, get_hf_dataset


VERTICES_TOL = 1e-05  # Transformation loses a little precision in vertices
NORMALS_TOL = 1e-02  # Conversion from .usd to .glb loses a little precision in normals
USD_COLOR_TOL = 1e-07  # Parsing from .usd loses a little precision in color


def check_gs_meshes(gs_mesh1, gs_mesh2, mesh_name):
    """Check if two gs.Mesh objects are equal."""

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

    vertices1, faces1, normals1, uvs1 = extract_mesh(gs_mesh1)
    vertices2, faces2, normals2, uvs2 = extract_mesh(gs_mesh2)

    assert_allclose(vertices1, vertices2, atol=VERTICES_TOL, err_msg=f"Vertices match failed in mesh {mesh_name}.")
    assert_array_equal(faces1, faces2, err_msg=f"Faces match failed in mesh {mesh_name}.")
    assert_allclose(normals1, normals2, atol=NORMALS_TOL, err_msg=f"Normals match failed in mesh {mesh_name}.")
    assert_allclose(uvs1, uvs2, rtol=gs.EPS, err_msg=f"UVs match failed in mesh {mesh_name}.")


def check_gs_tm_meshes(gs_mesh, tm_mesh, mesh_name):
    """Check if a gs.Mesh object and a trimesh.Trimesh object are equal."""
    assert_allclose(
        tm_mesh.vertices,
        gs_mesh.trimesh.vertices,
        atol=VERTICES_TOL,
        err_msg=f"Vertices match failed in mesh {mesh_name}.",
    )
    assert_array_equal(
        tm_mesh.faces,
        gs_mesh.trimesh.faces,
        err_msg=f"Faces match failed in mesh {mesh_name}.",
    )
    assert_allclose(
        tm_mesh.vertex_normals,
        gs_mesh.trimesh.vertex_normals,
        atol=NORMALS_TOL,
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
        assert_array_equal(
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
        assert_array_equal(
            gs_texture1.image_array,
            gs_texture2.image_array,
            err_msg=f"Texture mismatch for material {material_name} in {texture_name}.",
        )
    else:
        assert (
            gs_texture1 is None and gs_texture2 is None
        ), f"Both textures should be None for material {material_name} in {texture_name}."


@pytest.mark.required
@pytest.mark.parametrize("glb_file", ["glb/combined_srt.glb", "glb/combined_transform.glb"])
def test_glb_parse_geometry(glb_file):
    """Test glb mesh geometry parsing."""
    asset_path = get_hf_dataset(pattern=glb_file)
    glb_file = os.path.join(asset_path, glb_file)
    gs_meshes = gltf_utils.parse_mesh_glb(
        glb_file,
        group_by_material=False,
        scale=1.0,
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
        check_gs_tm_meshes(gs_mesh, tm_mesh, mesh_name)


@pytest.mark.required
@pytest.mark.parametrize("glb_file", ["glb/chopper.glb"])
def test_glb_parse_material(glb_file):
    """Test glb mesh geometry parsing."""
    asset_path = get_hf_dataset(pattern=glb_file)
    glb_file = os.path.join(asset_path, glb_file)
    gs_meshes = gltf_utils.parse_mesh_glb(
        glb_file,
        group_by_material=True,
        scale=1.0,
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
            gs_material.get_texture(),
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
@pytest.mark.parametrize("usd_filename", ["usd/sneaker_airforce", "usd/RoughnessTest"])
def test_usd_parse(usd_filename):
    asset_path = get_hf_dataset(pattern=f"{usd_filename}.glb")
    glb_file = os.path.join(asset_path, f"{usd_filename}.glb")
    asset_path = get_hf_dataset(pattern=f"{usd_filename}.usdz")
    usd_file = os.path.join(asset_path, f"{usd_filename}.usdz")

    gs_glb_meshes = gltf_utils.parse_mesh_glb(
        glb_file,
        group_by_material=True,
        scale=1.0,
        surface=gs.surfaces.Default(),
    )
    gs_usd_meshes = usda_utils.parse_mesh_usd(
        usd_file,
        group_by_material=True,
        scale=1.0,
        surface=gs.surfaces.Default(),
    )

    assert len(gs_glb_meshes) == len(gs_usd_meshes)
    gs_glb_mesh_dict = {}
    for gs_glb_mesh in gs_glb_meshes:
        gs_glb_mesh_dict[gs_glb_mesh.metadata["name"]] = gs_glb_mesh
    for gs_usd_mesh in gs_usd_meshes:
        mesh_name = gs_usd_mesh.metadata["name"].split("/")[-1]
        gs_glb_mesh = gs_glb_mesh_dict[mesh_name]
        check_gs_meshes(gs_glb_mesh, gs_usd_mesh, mesh_name)

        gs_glb_material = gs_glb_mesh.surface
        gs_usd_material = gs_usd_mesh.surface
        material_name = gs_glb_mesh.metadata["name"]
        check_gs_textures(gs_glb_material.get_texture(), gs_usd_material.get_texture(), 1.0, material_name, "color")
        check_gs_textures(
            gs_glb_material.opacity_texture, gs_usd_material.opacity_texture, 1.0, material_name, "opacity"
        )
        check_gs_textures(
            gs_glb_material.roughness_texture, gs_usd_material.roughness_texture, 1.0, material_name, "roughness"
        )
        check_gs_textures(
            gs_glb_material.metallic_texture, gs_usd_material.metallic_texture, 0.0, material_name, "metallic"
        )
        check_gs_textures(gs_glb_material.normal_texture, gs_usd_material.normal_texture, 0.0, material_name, "normal")
        check_gs_textures(
            gs_glb_material.emissive_texture, gs_usd_material.emissive_texture, 0.0, material_name, "emissive"
        )


@pytest.mark.required
@pytest.mark.parametrize("usd_file", ["usd/nodegraph.usda"])
def test_usd_parse_nodegraph(usd_file):
    asset_path = get_hf_dataset(pattern=usd_file)
    usd_file = os.path.join(asset_path, usd_file)
    gs_usd_meshes = usda_utils.parse_mesh_usd(
        usd_file,
        group_by_material=True,
        scale=1.0,
        surface=gs.surfaces.Default(),
    )
    texture0 = gs_usd_meshes[0].surface.diffuse_texture
    texture1 = gs_usd_meshes[1].surface.diffuse_texture
    assert isinstance(texture0, gs.textures.ColorTexture)
    assert isinstance(texture1, gs.textures.ColorTexture)
    assert_allclose(texture0.color, (0.8, 0.2, 0.2), rtol=USD_COLOR_TOL)
    assert_allclose(texture1.color, (0.2, 0.6, 0.9), rtol=USD_COLOR_TOL)


@pytest.mark.required
@pytest.mark.skipif(
    sys.version_info[:2] != (3, 10) or sys.platform not in ("linux", "win32"),
    reason="omniverse-kit used by USD Baking cannot be correctly installed on this platform now.",
)
@pytest.mark.parametrize(
    "usd_file", ["usd/WoodenCrate/WoodenCrate_D1_1002.usda", "usd/franka_mocap_teleop/table_scene.usd"]
)
@pytest.mark.parametrize("backend", [gs.cuda])
def test_usd_bake(usd_file, show_viewer):
    asset_path = get_hf_dataset(pattern=os.path.join(os.path.dirname(usd_file), "*"), local_dir_use_symlinks=False)
    usd_file = os.path.join(asset_path, usd_file)
    gs_usd_meshes = usda_utils.parse_mesh_usd(
        usd_file, group_by_material=True, scale=1.0, surface=gs.surfaces.Default(), bake_cache=False
    )
    for gs_usd_mesh in gs_usd_meshes:
        require_bake = gs_usd_mesh.metadata["require_bake"]
        bake_success = gs_usd_mesh.metadata["bake_success"]
        assert not require_bake or (require_bake and bake_success)

    scene = gs.Scene(
        show_viewer=show_viewer,
        show_FPS=False,
    )
    robot = scene.add_entity(
        gs.morphs.Mesh(
            file=usd_file,
        ),
    )


@pytest.mark.required
def test_urdf_with_existing_glb(tmp_path, show_viewer):
    glb_file = "usd/sneaker_airforce.glb"
    asset_path = get_hf_dataset(pattern=glb_file)

    urdf_path = tmp_path / "model.urdf"
    urdf_path.write_text(
        f"""<robot name="shoe">
              <link name="base">
                <visual>
                  <geometry><mesh filename="{os.path.join(asset_path, glb_file)}"/></geometry>
                </visual>
              </link>
            </robot>
         """
    )

    scene = gs.Scene(
        show_viewer=show_viewer,
        show_FPS=False,
    )
    robot = scene.add_entity(
        gs.morphs.URDF(
            file=urdf_path,
        ),
    )


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
def test_splashsurf_surface_reconstruction(show_viewer):
    scene = gs.Scene(
        show_viewer=show_viewer,
    )
    water = scene.add_entity(
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


@pytest.mark.required
def test_convex_decompose_cache(monkeypatch):
    # Check if the convex decomposition cache is correctly tracked regardless of the scale

    # Monkeypatch the get_cvx_path function to track the cache path
    seen_paths = []
    real_get_cvx_path = mesh_utils.get_cvx_path

    def wrapped_get_cvx_path(verts, faces, opts):
        path = real_get_cvx_path(verts, faces, opts)
        seen_paths.append(path)
        return path

    monkeypatch.setattr(mesh_utils, "get_cvx_path", wrapped_get_cvx_path)

    # Monkeypatch the convex_decompose function to track the convex decomposition result
    seen_results = []
    real_convex_decompose = mesh_utils.convex_decompose

    def wrapped_convex_decompose(mesh, opts):
        result = real_convex_decompose(mesh, opts)
        seen_results.append(result)
        return result

    monkeypatch.setattr(mesh_utils, "convex_decompose", wrapped_convex_decompose)

    # First scene building to create the cache
    scene = gs.Scene(
        show_viewer=False,
    )
    first_scale = 2.0
    duck = scene.add_entity(
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
    duck = scene.add_entity(
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
        assert_array_equal(scaled_part.faces, cached_part.faces)
