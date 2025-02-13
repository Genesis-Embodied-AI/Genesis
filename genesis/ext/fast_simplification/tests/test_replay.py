import numpy as np
import pytest

import fast_simplification

try:
    import pyvista as pv

    has_vtk = True
except ModuleNotFoundError:
    has_vtk = False
skip_no_vtk = pytest.mark.skipif(not has_vtk, reason="Requires VTK")


@pytest.fixture
def mesh():
    return pv.Sphere()


def test_collapses_trivial():
    # arrays from:
    # mesh = pv.Plane(i_resolution=2, j_resolution=2).triangulate()
    points = [
        [0.5, -0.5, 0.0],
        [0.0, -0.5, 0.0],
        [-0.5, -0.5, 0.0],
        [0.5, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [-0.5, 0.0, 0.0],
        [0.5, 0.5, 0.0],
        [0.0, 0.5, 0.0],
        [-0.5, 0.5, 0.0],
    ]

    faces = [
        [0, 1, 3],
        [4, 3, 1],
        [1, 2, 4],
        [5, 4, 2],
        [3, 4, 6],
        [7, 6, 4],
        [4, 5, 7],
        [8, 7, 5],
    ]

    with pytest.raises(ValueError, match="You must specify"):
        fast_simplification.simplify(points, faces)

    points_out, faces_out, collapses = fast_simplification.simplify(points, faces, 0.5, return_collapses=True)

    (
        replay_points,
        replay_faces,
        indice_mapping,
    ) = fast_simplification.replay_simplification(points, faces, collapses)
    assert np.allclose(points_out, replay_points)
    assert np.allclose(faces_out, replay_faces)


@skip_no_vtk
def test_collapses_sphere(mesh):
    points = mesh.points
    faces = mesh.faces.reshape(-1, 4)[:, 1:]
    reduction = 0.5

    points_out, faces_out, collapses = fast_simplification.simplify(points, faces, reduction, return_collapses=True)

    (
        replay_points,
        replay_faces,
        indice_mapping,
    ) = fast_simplification.replay_simplification(points, faces, collapses)
    assert np.allclose(points_out, replay_points)
    assert np.allclose(faces_out, replay_faces)


try:
    from pyvista import examples

    @pytest.fixture
    def louis():
        return examples.download_louis_louvre()

    @pytest.fixture
    def human():
        return examples.download_human()

    has_examples = True
except:
    has_examples = False
skip_no_examples = pytest.mark.skipif(not has_examples, reason="Requires pyvista.examples")


@skip_no_examples
@skip_no_vtk
def test_collapses_louis(louis):
    points = louis.points
    faces = louis.faces.reshape(-1, 4)[:, 1:]
    reduction = 0.9

    points_out, faces_out, collapses = fast_simplification.simplify(points, faces, reduction, return_collapses=True)

    (
        replay_points,
        replay_faces,
        indice_mapping,
    ) = fast_simplification.replay_simplification(points, faces, collapses)
    assert np.allclose(points_out, replay_points)
    assert np.allclose(faces_out, replay_faces)


@skip_no_examples
@skip_no_vtk
def test_human(human):
    points = human.points
    faces = human.faces.reshape(-1, 4)[:, 1:]
    reduction = 0.9

    points_out, faces_out, collapses = fast_simplification.simplify(points, faces, reduction, return_collapses=True)

    (
        replay_points,
        replay_faces,
        indice_mapping,
    ) = fast_simplification.replay_simplification(points, faces, collapses)
    assert np.allclose(points_out, replay_points)
    assert np.allclose(faces_out, replay_faces)
