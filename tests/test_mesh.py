import pytest
import trimesh
import numpy as np
import os

import genesis as gs
import genesis.utils.mesh as mu
import genesis.utils.gltf as gltf_utils


@pytest.mark.parametrize("glb_file", ["tests/combined_srt.glb", "tests/combined_transform.glb"])
def test_glb_parse(glb_file):
    """Test glb mesh parsing."""
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
            gs_mesh.trimesh.vertices,
            tm_mesh.vertices,
            rtol=0,
            atol=1e-06,
            err_msg=f"Vertices match failed in mesh {mesh_name}.",
        )
        np.testing.assert_array_equal(
            gs_mesh.trimesh.faces,
            tm_mesh.faces,
            err_msg=f"Faces match failed mesh {mesh_name}.",
        )
        np.testing.assert_allclose(
            gs_mesh.trimesh.vertex_normals,
            tm_mesh.vertex_normals,
            rtol=0,
            atol=1e-06,
            err_msg=f"Normals match failed mesh {mesh_name}.",
        )
        if not isinstance(tm_mesh.visual, trimesh.visual.color.ColorVisuals):
            np.testing.assert_allclose(
                gs_mesh.trimesh.visual.uv,
                tm_mesh.visual.uv,
                err_msg=f"UVs match failed mesh {mesh_name}.",
            )
