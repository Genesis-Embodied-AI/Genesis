import os

import numpy as np
import pytest
import trimesh

import genesis as gs
import genesis.utils.point_cloud as pc


def test_furthest_point_sample_shape_and_determinism():
    pts = np.random.default_rng(1).random((50, 3))
    out1 = pc.furthest_point_sample(pts, 10, seed=42)
    out2 = pc.furthest_point_sample(pts, 10, seed=42)
    assert out1.shape == (10, 3)
    assert out1.dtype == gs.np_float
    np.testing.assert_array_equal(out1, out2)


def test_furthest_point_sample_no_seed_first_index_zero():
    points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)
    out = pc.furthest_point_sample(points, 3, seed=None)
    assert np.allclose(out[0], points[0])


def test_sample_mesh_point_cloud_shape_determinism():
    mesh = trimesh.creation.box((1.0, 1.0, 1.0))
    points_a = pc.sample_mesh_point_cloud(mesh.vertices, mesh.faces, 16, n_candidates=400, seed=7, use_cache=False)
    points_b = pc.sample_mesh_point_cloud(mesh.vertices, mesh.faces, 16, n_candidates=400, seed=7, use_cache=False)
    assert points_a.shape == (16, 3)
    assert points_a.dtype == gs.np_float
    np.testing.assert_array_equal(points_a, points_b)


def test_sample_mesh_point_cloud_normals_box_axis_aligned_unit():
    mesh = trimesh.creation.box((1.0, 1.0, 1.0))
    pts, nrm = pc.sample_mesh_point_cloud(
        mesh.vertices, mesh.faces, 32, n_candidates=800, seed=0, use_cache=False, return_normals=True
    )
    assert np.allclose(np.linalg.norm(nrm, axis=1), 1.0, atol=1e-5)
    abs_n = np.abs(nrm)
    np.testing.assert_allclose(abs_n.max(axis=1), 1.0, atol=1e-4)
    np.testing.assert_array_less(abs_n.sum(axis=1) - abs_n.max(axis=1), np.full(len(nrm), 1e-3))


def test_sample_mesh_point_cloud_normals_cache():
    mesh = trimesh.creation.box((1.0, 1.0, 1.0))
    kwargs = dict(
        verts=mesh.vertices,
        faces=mesh.faces,
        n_points=5,
        n_candidates=10,
        return_normals=True,
        seed=7,
    )
    path_n = pc.get_fps_pc_path(**kwargs)
    if os.path.exists(path_n):
        os.remove(path_n)
    points_and_normals = pc.sample_mesh_point_cloud(**kwargs, use_cache=True)
    assert os.path.exists(path_n)
    points_and_normals_cached = pc.sample_mesh_point_cloud(**kwargs, use_cache=True)
    np.testing.assert_array_equal(points_and_normals[0], points_and_normals_cached[0])
    np.testing.assert_array_equal(points_and_normals[1], points_and_normals_cached[1])

    kwargs["return_normals"] = False
    if os.path.exists(pc.get_fps_pc_path(**kwargs)):
        os.remove(pc.get_fps_pc_path(**kwargs))
    points = pc.sample_mesh_point_cloud(**kwargs, use_cache=True)
    assert os.path.exists(pc.get_fps_pc_path(**kwargs))
    points_cached = pc.sample_mesh_point_cloud(**kwargs, use_cache=True)
    np.testing.assert_array_equal(points, points_cached)


def test_sample_mesh_point_cloud_on_surface():
    mesh = trimesh.creation.box((1.0, 1.0, 1.0))
    points = pc.sample_mesh_point_cloud(mesh.vertices, mesh.faces, 32, n_candidates=800, seed=0, use_cache=False)
    _, dist, _ = mesh.nearest.on_surface(points)
    assert np.all(dist < 1e-5)


def test_sample_mesh_point_cloud_min_separation():
    mesh = trimesh.creation.box((1.0, 1.0, 1.0))
    points = pc.sample_mesh_point_cloud(mesh.vertices, mesh.faces, 8, n_candidates=2000, seed=0, use_cache=False)
    d = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    assert d.min() > 0.2


def test_gs_sample_mesh_point_cloud():
    mesh = trimesh.creation.box((0.2, 0.4, 0.6))
    gmesh = gs.Mesh.from_trimesh(mesh)
    points = gmesh.sample_point_cloud(10, n_candidates=300, seed=67, use_cache=False)
    assert points.shape == (10, 3)
    _, dist, _ = mesh.nearest.on_surface(points)
    assert np.all(dist < 1e-5)

    pts, nrm = gmesh.sample_point_cloud(10, n_candidates=300, seed=67, use_cache=False, return_normals=True)
    assert pts.shape == (10, 3) and nrm.shape == (10, 3)
    _, dist2, _ = mesh.nearest.on_surface(pts)
    assert np.all(dist2 < 1e-5)
    assert np.allclose(np.linalg.norm(nrm, axis=1), 1.0, atol=1e-5)


def test_furthest_point_sample_invalid():
    with pytest.raises(gs.GenesisException):
        pc.furthest_point_sample(np.zeros((5, 3)), 10, seed=None)
