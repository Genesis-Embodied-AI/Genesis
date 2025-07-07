import os
from pathlib import Path
import numpy as np
import pytest
import trimesh

import genesis as gs
import genesis.utils.gltf as gltf_utils
import genesis.utils.mesh as mu
import genesis.utils.usda as usda_utils

from .utils import assert_allclose, assert_array_equal, get_hf_assets

VERTICES_TOL = 1e-05  # Transformation loses a little precision in vertices
NORMALS_TOL = 1e-02  # Conversion from .usd to .glb loses a little precision in normals


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
@pytest.mark.parametrize("glb_file", ["tests/combined_srt.glb", "tests/combined_transform.glb"])
def test_glb_parse_geometry(glb_file):
    """Test glb mesh geometry parsing."""
    glb_file = os.path.join(mu.get_assets_dir(), glb_file)
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
@pytest.mark.parametrize("glb_file", ["tests/chopper.glb"])
def test_glb_parse_material(glb_file):
    """Test glb mesh geometry parsing."""
    glb_file = os.path.join(mu.get_assets_dir(), glb_file)
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
    asset_path = get_hf_assets(pattern=f"{usd_filename}.glb")
    glb_file = os.path.join(asset_path, f"{usd_filename}.glb")
    asset_path = get_hf_assets(pattern=f"{usd_filename}.usdz")
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
def test_urdf_with_existing_glb(tmp_path, show_viewer):
    assets = Path(gs.utils.get_assets_dir())
    glb_path = assets / "usd" / "sneaker_airforce.glb"
    urdf_path = tmp_path / "model.urdf"
    urdf_path.write_text(
        f"""<robot name="shoe">
              <link name="base">
                <visual>
                  <geometry><mesh filename="{glb_path}"/></geometry>
                </visual>
              </link>
            </robot>
         """
    )
    scene = gs.Scene(
        show_viewer=show_viewer,
        show_FPS=False,
    )
    scene.build()
    scene.step()


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
    scene.build()
    scene.step()
