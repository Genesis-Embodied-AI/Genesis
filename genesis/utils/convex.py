import itertools
import logging
import numpy as np
from scipy import spatial
import trimesh


_BOX_CORNERS = list(itertools.product((-1, 1), (-1, 1), (-1, 1)))

# pyformat: disable
# The faces of a triangulated box, i.e. the indices in _BOX_CORNERS of the
# vertices of the 12 triangles (two triangles for each side of the box).
_TRIANGULATED_BOX_FACES = [
    0,
    4,
    1,
    4,
    5,
    1,  # left
    0,
    2,
    4,
    2,
    6,
    4,  # bottom
    6,
    5,
    4,
    6,
    7,
    5,  # front
    2,
    3,
    6,
    3,
    7,
    6,  # right
    1,
    5,
    3,
    5,
    7,
    3,  # top
    0,
    1,
    2,
    1,
    3,
    2,  # back
]
# Rectangular box faces using a counter-clockwise winding order convention.
_BOX_FACES = [
    0,
    4,
    5,
    1,  # left
    0,
    2,
    6,
    4,  # bottom
    6,
    7,
    5,
    4,  # front
    2,
    3,
    7,
    6,  # right
    1,
    5,
    7,
    3,  # top
    0,
    1,
    3,
    2,  # back
]
# pyformat: enable

_MAX_HULL_FACE_VERTICES = 20
_CONVEX_CACHE: Dict[Tuple[int, int], Convex] = {}


def get_face_norm(vert: jp.ndarray, face: jp.ndarray) -> jp.ndarray:
    """Calculates face normals given vertices and face indexes."""
    assert len(vert.shape) == 2 and len(face.shape) == 2, (
        f"vert and face should have dim of 2, got {len(vert.shape)} and " f"{len(face.shape)}"
    )
    face_vert = jp.take(vert, face, axis=0)
    # use CCW winding order convention
    edge0 = face_vert[:, 1, :] - face_vert[:, 0, :]
    edge1 = face_vert[:, -1, :] - face_vert[:, 0, :]
    face_norm = jp.cross(edge0, edge1)
    face_norm = face_norm / jp.linalg.norm(face_norm, axis=1).reshape((-1, 1))
    return face_norm


def get_unique_edges(vert: np.ndarray, face: np.ndarray) -> np.ndarray:
    """Returns unique edges.

    Args:
      vert: (n_vert, 3) vertices
      face: (n_face, n_vert) face index array

    Returns:
      edges: 2-tuples of vertex indexes for each edge
    """
    return np.array([[0, 1]])

    r_face = np.roll(face, 1, axis=1)
    edges = np.concatenate(np.array([face, r_face]).T)

    # do a first pass to remove duplicates
    edges.sort(axis=1)
    edges = np.unique(edges, axis=0)
    edges = edges[edges[:, 0] != edges[:, 1]]  # get rid of edges from padded face

    # get normalized edge directions
    edge_vert = vert.take(edges, axis=0)
    edge_dir = edge_vert[:, 0] - edge_vert[:, 1]
    norms = np.sqrt(np.sum(edge_dir**2, axis=1))
    edge_dir = edge_dir / norms.reshape((-1, 1))

    # get the first unique edge for all pairwise comparisons
    diff1 = edge_dir[:, None, :] - edge_dir[None, :, :]
    diff2 = edge_dir[:, None, :] + edge_dir[None, :, :]
    matches = (np.linalg.norm(diff1, axis=-1) < 1e-6) | (np.linalg.norm(diff2, axis=-1) < 1e-6)
    matches = np.tril(matches).sum(axis=-1)
    unique_edge_idx = np.where(matches == 1)[0]

    return edges[unique_edge_idx]


def _box(b: Box, triangulated=True) -> Tuple[np.ndarray, np.ndarray]:
    """Creates face and vert from a box geometry."""
    assert b.halfsize.shape == (3,), f"Box halfsize should have shape (3,), got {b.halfsize.shape}"
    box_corners = jp.array(_BOX_CORNERS)
    vert = box_corners * b.halfsize.reshape(-1, 3)
    box_faces = _TRIANGULATED_BOX_FACES if triangulated else _BOX_FACES
    face_dim = 3 if triangulated else 4
    face = jp.array([box_faces]).reshape(-1, face_dim)
    return vert, face


def box_tri(b: Box) -> Mesh:
    """Creates a triangulated mesh from a box geometry."""
    vert, face = _box(b, triangulated=True)
    return Mesh(  # pytype: disable=wrong-arg-types  # jax-ndarray
        vert=vert,
        face=face,
        link_idx=b.link_idx,
        transform=b.transform,
        friction=b.friction,
        elasticity=b.elasticity,
        solver_params=b.solver_params,
        rgba=b.rgba,
    )


def _box_hull(b: Box) -> Convex:
    """Creates a mesh for a box with rectangular faces."""
    vert, face = _box(b, triangulated=False)
    return Convex(  # pytype: disable=wrong-arg-types  # jax-ndarray
        vert=vert,
        face=face,
        link_idx=b.link_idx,
        transform=b.transform,
        friction=b.friction,
        elasticity=b.elasticity,
        solver_params=b.solver_params,
        unique_edge=get_unique_edges(vert, face),
        rgba=b.rgba,
    )


def convex_hull_2d(points: np.ndarray, normal: np.ndarray) -> np.ndarray:
    """Calculates the hull face for a set of points on a plane."""
    # project points onto the closest axis plane
    best_axis = np.abs(np.eye(3).dot(normal)).argmax()
    axis = np.eye(3)[best_axis]
    d = points.dot(axis).reshape((-1, 1))
    axis_points = points - d * axis
    axis_points = axis_points[:, list({0, 1, 2} - {best_axis})]

    # get the polygon hull face, and make the points ccw wrt the face normal
    # TODO: consider sorting unique edges by their angle to get the hull
    c = spatial.ConvexHull(axis_points)
    order_ = np.where(axis.dot(normal) > 0, 1, -1)
    order_ *= np.where(best_axis == 1, -1, 1)
    hull_point_idx = c.vertices[::order_]
    assert (axis_points - c.points).sum() == 0

    return hull_point_idx


def _merge_coplanar(tm: trimesh.Trimesh) -> np.ndarray:
    """Merges coplanar facets."""
    if not tm.facets:
        return tm.faces.copy()  # no facets, return faces
    if not tm.faces.shape[0]:
        raise ValueError("Mesh has no faces.")

    # Get faces.
    face_idx = set(range(tm.faces.shape[0])) - set(np.concatenate(tm.facets))
    face_idx = np.array(list(face_idx))
    faces = tm.faces[face_idx] if face_idx.shape[0] > 0 else np.array([])

    # Get facets.
    facets = []
    for i, facet in enumerate(tm.facets):
        point_idx = np.unique(tm.faces[facet])
        points = tm.vertices[point_idx]
        normal = tm.facets_normal[i]

        # convert triangulated facet to a polygon
        hull_point_idx = convex_hull_2d(points, normal)
        face = point_idx[hull_point_idx]

        # resize faces that exceed max polygon vertices
        every = face.shape[0] // _MAX_HULL_FACE_VERTICES + 1
        face = face[::every]
        facets.append(face)

    # Pad facets so that they can be stacked.
    max_len = max(f.shape[0] for f in facets) if facets else faces.shape[1]
    assert max_len <= _MAX_HULL_FACE_VERTICES
    for i, f in enumerate(facets):
        if f.shape[0] < max_len:
            f = np.pad(f, (0, max_len - f.shape[0]), "edge")
        facets[i] = f

    if not faces.shape[0]:
        assert facets
        return np.array(facets)  # no faces, return facets

    # Merge faces and facets.
    faces = np.pad(faces, ((0, 0), (0, max_len - faces.shape[1])), "edge")
    return np.concatenate([faces, facets])


def _convex_hull(m: Mesh) -> Convex:
    """Creates a convex hull from a mesh."""
    tm = trimesh.Trimesh(vertices=m.vert, faces=m.face)
    tm_convex = trimesh.convex.convex_hull(tm)
    vert = tm_convex.vertices.copy()
    face = _merge_coplanar(tm_convex)
    return Convex(  # pytype: disable=wrong-arg-types  # jax-ndarray
        vert=vert,
        face=face,
        link_idx=m.link_idx,
        transform=m.transform,
        friction=m.friction,
        elasticity=m.elasticity,
        solver_params=m.solver_params,
        unique_edge=get_unique_edges(vert, face),
        rgba=m.rgba,
    )


def convex_hull(obj: Union[Box, Mesh]) -> Convex:
    """Creates a convex hull from a box or mesh."""
    if isinstance(obj, Box):
        return _box_hull(obj)
    key = (
        hash(obj.vert.data.tobytes()),
        hash(obj.face.data.tobytes()),
    )  # pytype: disable=attribute-error  # jax-ndarray
    if key not in _CONVEX_CACHE:
        logging.info("Converting mesh %s into convex hull.", key)
        _CONVEX_CACHE[key] = _convex_hull(obj)
    convex = _CONVEX_CACHE[key]
    convex = convex.replace(
        link_idx=obj.link_idx,
        transform=obj.transform,
        friction=obj.friction,
        elasticity=obj.elasticity,
        solver_params=obj.solver_params,
        rgba=obj.rgba,
    )
    return convex
