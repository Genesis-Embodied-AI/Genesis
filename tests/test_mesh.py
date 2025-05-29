import pytest
import trimesh
import numpy as np
import os

import genesis as gs
import genesis.utils.mesh as mu
import genesis.utils.gltf as gltf_utils
import genesis.utils.usda as usda_utils


def check_gs_meshes(gs_mesh1, gs_mesh2, mesh_name):
    """Check if two gs.Mesh objects are equal."""
    np.testing.assert_allclose(
        gs_mesh1.trimesh.vertices,
        gs_mesh2.trimesh.vertices,
        rtol=0,
        atol=1e-06,
        err_msg=f"Vertices match failed in mesh {mesh_name}.",
    )
    np.testing.assert_array_equal(
        gs_mesh1.trimesh.faces,
        gs_mesh2.trimesh.faces,
        err_msg=f"Faces match failed in mesh {mesh_name}.",
    )
    np.testing.assert_allclose(
        gs_mesh1.trimesh.vertex_normals,
        gs_mesh2.trimesh.vertex_normals,
        rtol=0,
        atol=1e-06,
        err_msg=f"Normals match failed in mesh {mesh_name}.",
    )
    np.testing.assert_allclose(
        gs_mesh1.trimesh.visual.uv,
        gs_mesh2.trimesh.visual.uv,
        err_msg=f"UVs match failed in mesh {mesh_name}.",
    )


def check_gs_tm_meshes(gs_mesh, tm_mesh, mesh_name):
    """Check if a gs.Mesh object and a trimesh.Trimesh object are equal."""
    np.testing.assert_allclose(
        tm_mesh.vertices,
        gs_mesh.trimesh.vertices,
        rtol=0,
        atol=1e-06,
        err_msg=f"Vertices match failed in mesh {mesh_name}.",
    )
    np.testing.assert_array_equal(
        tm_mesh.faces,
        gs_mesh.trimesh.faces,
        err_msg=f"Faces match failed in mesh {mesh_name}.",
    )
    np.testing.assert_allclose(
        tm_mesh.vertex_normals,
        gs_mesh.trimesh.vertex_normals,
        rtol=0,
        atol=1e-06,
        err_msg=f"Normals match failed in mesh {mesh_name}.",
    )
    if not isinstance(tm_mesh.visual, trimesh.visual.color.ColorVisuals):
        np.testing.assert_allclose(
            tm_mesh.visual.uv,
            gs_mesh.trimesh.visual.uv,
            err_msg=f"UVs match failed in mesh {mesh_name}.",
        )


def check_gs_tm_textures(gs_texture, tm_color, tm_image, dim, material_name, texture_name):
    """Check if a gs.Texture object and a trimesh.Texture object are equal."""
    gs_color = gs_texture.color if isinstance(gs_texture, gs.textures.ColorTexture) else gs_texture.image_color
    tm_color = tm_color or np.ones(dim)
    np.testing.assert_allclose(
        tm_color,
        gs_color,
        rtol=0,
        atol=1e-06,
        err_msg=f"Color mismatch for material {material_name} in {texture_name}.",
    )

    if tm_image is not None:
        np.testing.assert_array_equal(
            tm_image,
            gs_texture.image_array,
            err_msg=f"Texture mismatch for material {material_name} in {texture_name}.",
        )


def check_gs_textures(gs_texture1, gs_texture2, material_name, texture_name):
    """Check if two gs.Texture objects are equal."""
    if isinstance(gs_texture1, gs.textures.ColorTexture):
        assert isinstance(gs_texture2, gs.textures.ColorTexture)
        np.testing.assert_allclose(
            gs_texture1.color,
            gs_texture2.color,
            rtol=0,
            atol=1e-06,
            err_msg=f"Color mismatch for material {material_name} in {texture_name}.",
        )
    elif isinstance(gs_texture1, gs.textures.ImageTexture):
        assert isinstance(gs_texture2, gs.textures.ImageTexture)
        np.testing.assert_allclose(
            gs_texture1.image_color,
            gs_texture2.image_color,
            rtol=0,
            atol=1e-06,
            err_msg=f"Color mismatch for material {material_name} in {texture_name}.",
        )
        np.testing.assert_array_equal(
            gs_texture1.image_array,
            gs_texture2.image_array,
            err_msg=f"Texture mismatch for material {material_name} in {texture_name}.",
        )
    else:
        assert (
            gs_texture1 is None and gs_texture2 is None
        ), f"Both textures should be None for material {material_name} in {texture_name}."


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
            1,
            material_name,
            "roughness",
        )
        check_gs_tm_textures(
            gs_material.metallic_texture,
            tm_material.metallicFactor,
            tm_metallic_image,
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
                3,
                material_name,
                "emissive",
            )


@pytest.mark.parametrize("usd_filename", ["tests/sneaker_airforce"])
def test_usd_parse(usd_filename):
    glb_file = os.path.join(mu.get_assets_dir(), f"{usd_filename}.glb")
    usd_file = os.path.join(mu.get_assets_dir(), f"{usd_filename}.usdz")
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
    for gs_glb_mesh, gs_usd_mesh in zip(gs_glb_meshes, gs_usd_meshes):
        mesh_name = gs_glb_mesh.metadata["name"]
        check_gs_meshes(gs_glb_mesh, gs_usd_mesh, mesh_name)

        gs_glb_material = gs_glb_mesh.surface
        gs_usd_material = gs_usd_mesh.surface
        material_name = gs_glb_mesh.metadata["name"]
        check_gs_textures(gs_glb_material.get_texture(), gs_usd_material.get_texture(), material_name, "color")
        check_gs_textures(gs_glb_material.opacity_texture, gs_usd_material.opacity_texture, material_name, "opacity")
        check_gs_textures(
            gs_glb_material.roughness_texture, gs_usd_material.roughness_texture, material_name, "roughness"
        )
        check_gs_textures(gs_glb_material.metallic_texture, gs_usd_material.metallic_texture, material_name, "metallic")
        check_gs_textures(gs_glb_material.normal_texture, gs_usd_material.normal_texture, material_name, "normal")
        check_gs_textures(gs_glb_material.emissive_texture, gs_usd_material.emissive_texture, material_name, "emissive")
