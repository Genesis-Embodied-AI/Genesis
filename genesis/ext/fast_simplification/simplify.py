"""Simplification library."""

import numpy as np

from . import _simplify
from .utils import ascontiguous


def _check_args(target_reduction, target_count, n_faces):
    """Check arguments."""
    if target_reduction and target_count:
        raise ValueError("You may specify ``target_reduction`` or ``target_count``, but not" " both")
    if target_reduction is None and target_count is None:
        raise ValueError("You must specify ``target_reduction`` or ``target_count``")

    if target_reduction is not None:
        if target_reduction > 1 or target_reduction < 0:
            raise ValueError("``target_reduction`` must be between 0 and 1")
        target_count = (1 - target_reduction) * n_faces

    if target_count < 0:
        raise ValueError("``target_count`` must be greater than 0")
    if target_count > n_faces:
        raise ValueError(f"``target_count`` must be less than the number of faces {n_faces}")
    return int(target_count)


@ascontiguous
def simplify(
    points,
    triangles,
    target_reduction=None,
    target_count=None,
    agg=7,
    verbose=False,
    return_collapses=False,
    lossless=False,
):
    """Simplify a triangular mesh.

    Parameters
    ----------
    points : sequence[float | double]
        A ``(n, 3)`` array of points. May be a ``numpy.ndarray`` or a
        sequence of points. Internally converted to double precision.
    triangles : sequence
        A ``(n, 3)`` array of triangle indices. May be a
        ``numpy.ndarray`` or a list of triangle indices.
    target_reduction : float, optional
        Fraction of the original mesh to remove.  If set to ``0.9``,
        this function will try to reduce the data set to 10% of its
        original size and will remove 90% of the input triangles. Use
        this parameter or ``target_count``.
    target_count : int, optional
        Target number of triangles to reduce mesh to.  This may be
        used in place of ``target_reduction``, but both cannot be set.
    agg : int, optional
        Controls how aggressively to decimate the mesh.  A value of 10
        will result in a fast decimation at the expense of mesh
        quality and shape.  A value of 0 will attempt to preserve the
        original mesh geometry at the expense of time.  Setting a low
        value may result in being unable to reach the
        ``target_reduction`` or ``target_count``.
    verbose : bool, optional
        Enable verbose output when simplifying the mesh.
    return_collapses : bool, optional
        If True, return the history of collapses as a
        ``(n_collapses, 2)`` array of indices.
        ``collapses[i] = [i0, i1]`` means that durint the i-th
        collapse, the vertex ``i1`` was collapsed into the vertex
        ``i0``.

    Returns
    -------
    np.ndarray
        Points array.
    np.ndarray
        Triangles array.
    np.ndarray (optional)
        Collapses array.

    Examples
    --------
    This basic example demonstrates how to decimate a simple planar
    mesh composed by 8 triangles.

    >>> import fast_simplification
    >>> points = [
    ...     [0.5, -0.5, 0.0],
    ...     [0.0, -0.5, 0.0],
    ...     [-0.5, -0.5, 0.0],
    ...     [0.5, 0.0, 0.0],
    ...     [0.0, 0.0, 0.0],
    ...     [-0.5, 0.0, 0.0],
    ...     [0.5, 0.5, 0.0],
    ...     [0.0, 0.5, 0.0],
    ...     [-0.5, 0.5, 0.0],
    ... ]
    >>> faces = [
    ...     [0, 1, 3],
    ...     [4, 3, 1],
    ...     [1, 2, 4],
    ...     [5, 4, 2],
    ...     [3, 4, 6],
    ...     [7, 6, 4],
    ...     [4, 5, 7],
    ...     [8, 7, 5],
    ... ]
    >>> points_out, faces_out = fast_simplification.simplify(points, faces, 0.5)

    """

    points = np.asarray(points, dtype=np.float64)
    if not isinstance(triangles, np.ndarray):
        triangles = np.array(triangles, dtype=np.int32)

    if points.ndim != 2:
        raise ValueError("``points`` array must be 2 dimensional")
    if points.shape[1] != 3:
        raise ValueError(f"Expected ``points`` array to be (n, 3), not {points.shape}")

    if triangles.ndim != 2:
        raise ValueError("``triangles`` array must be 2 dimensional")
    if triangles.shape[1] != 3:
        raise ValueError(f"Expected ``triangles`` array to be (n, 3), not {triangles.shape}")

    n_faces = triangles.shape[0]

    triangles = np.ascontiguousarray(triangles)

    if triangles.dtype == np.int32:
        load = _simplify.load_int32
    elif triangles.dtype == np.int64:
        load = _simplify.load_int64
    else:
        load = _simplify.load_int32
        triangles = triangles.astype(np.int32)

    load(
        points.shape[0],
        n_faces,
        points,
        triangles,
    )

    if lossless:
        _simplify.simplify_lossless(verbose)
    else:
        target_count = _check_args(target_reduction, target_count, n_faces)
        _simplify.simplify(target_count, agg, verbose)
    points = _simplify.return_points()
    faces = _simplify.return_faces_int32_no_padding().reshape(-1, 3)

    if return_collapses:
        return points, faces, _simplify.return_collapses()
    return points, faces


def simplify_mesh(mesh, target_reduction=None, target_count=None, agg=7, verbose=False):
    """Simplify a pyvista mesh.

    Parameters
    ----------
    mesh : pyvista.PolyData
        PyVista mesh.
    target_reduction : float
        Fraction of the original mesh to remove.  If set to ``0.9``,
        this function will try to reduce the data set to 10% of its
        original size and will remove 90% of the input triangles. Use
        this parameter or ``target_count``.
    target_count : int, optional
        Target number of triangles to reduce mesh to.  This may be
        used in place of ``target_reduction``, but both cannot be set.
    agg : int, optional
        Controls how aggressively to decimate the mesh.  A value of 10
        will result in a fast decimation at the expense of mesh
        quality and shape.  A value of 0 will attempt to preserve the
        original mesh geometry at the expense of time.  Setting a low
        value may result in being unable to reach the
        ``target_reduction`` or ``target_count``.
    verbose : bool, optional
        Enable verbose output when simplifying the mesh.

    Returns
    -------
    pyvista.PolyData
        Simplified mesh. The field data of the mesh will contain a
        field named ``fast_simplification_collapses`` that contains
        the history of collapses as a ``(n_collapses, 2)`` array of
        indices. ``collapses[i] = [i0, i1]`` means that during the
        i-th collapse, the vertex ``i1`` was collapsed into the vertex
        ``i0``.

    """
    try:
        import pyvista as pv
    except ImportError:
        raise ImportError("Please install pyvista to use this feature with:\n" "pip install pyvista")

    n_faces = mesh.n_cells
    _simplify.load_from_vtk(
        mesh.n_points,
        mesh.points.astype(np.float64, order="C", copy=False),
        mesh.faces.astype(np.int32, order="C", copy=False),
        n_faces,
    )

    target_count = _check_args(target_reduction, target_count, n_faces)
    _simplify.simplify(target_count, agg, verbose)

    # return the correct datatype of the faces
    if pv._get_vtk_id_type() == np.int32:
        faces = _simplify.return_faces_int32()
    else:
        faces = _simplify.return_faces_int64()

    # construct mesh
    mesh = pv.PolyData(_simplify.return_points(), faces, deep=False)
    mesh.field_data["fast_simplification_collapses"] = _simplify.return_collapses()

    return mesh
