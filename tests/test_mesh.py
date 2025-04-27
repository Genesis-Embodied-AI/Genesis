import pytest
import trimesh
import numpy as np
import os

import genesis as gs
import genesis.utils.mesh as mu
import genesis.utils.gltf as gltf_utils


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
            err_msg=f"Faces match failed mesh {mesh_name}.",
        )
        np.testing.assert_allclose(
            tm_mesh.vertex_normals,
            gs_mesh.trimesh.vertex_normals,
            rtol=0,
            atol=1e-06,
            err_msg=f"Normals match failed mesh {mesh_name}.",
        )
        if not isinstance(tm_mesh.visual, trimesh.visual.color.ColorVisuals):
            np.testing.assert_allclose(
                tm_mesh.visual.uv,
                gs_mesh.trimesh.visual.uv,
                err_msg=f"UVs match failed mesh {mesh_name}.",
            )


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

    def check_texture(tm_color, tm_texture, gs_texture, dim, material_name, texture_name):
        gs_color = gs_texture.color if isinstance(gs_texture, gs.textures.ColorTexture) else gs_texture.image_color
        tm_color = tm_color or np.ones(dim)
        np.testing.assert_allclose(
            tm_color,
            gs_color,
            rtol=0,
            atol=1e-06,
            err_msg=f"Color mismatch for material {material_name} in {texture_name}.",
        )

        if tm_texture is not None:
            np.testing.assert_array_equal(
                tm_texture,
                gs_texture.image_array,
                err_msg=f"Texture mismatch for material {material_name} in {texture_name}.",
            )

    for gs_mesh in gs_meshes:
        material_name = gs_mesh.metadata["name"]
        tm_material = tm_materials[material_name]
        gs_material = gs_mesh.surface

        assert isinstance(tm_material, trimesh.visual.material.PBRMaterial)
        check_texture(
            tm_material.baseColorFactor,
            np.array(tm_material.baseColorTexture),
            gs_material.get_texture(),
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
        check_texture(
            tm_material.roughnessFactor,
            tm_roughness_image,
            gs_material.roughness_texture,
            1,
            material_name,
            "roughness",
        )
        check_texture(
            tm_material.metallicFactor,
            tm_metallic_image,
            gs_material.metallic_texture,
            1,
            material_name,
            "metallic",
        )

        if tm_material.emissiveFactor is None and tm_material.emissiveFactor is None:
            assert gs_material.emissive_texture is None
        else:
            check_texture(
                tm_material.emissiveFactor,
                np.array(tm_material.emissiveTexture),
                gs_material.emissive_texture,
                3,
                material_name,
                "emissive",
            )
