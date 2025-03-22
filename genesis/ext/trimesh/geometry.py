import numpy as np

from . import util
from .constants import log

try:
    import scipy.sparse
except BaseException as E:
    from . import exceptions

    # raise E again if anyone tries to use sparse
    scipy = exceptions.ExceptionWrapper(E)


def plane_transform(origin, normal):
    """
    Given the origin and normal of a plane find the transform
    that will move that plane to be coplanar with the XY plane.

    Parameters
    ----------
    origin : (3,) float
        Point that lies on the plane
    normal : (3,) float
        Vector that points along normal of plane

    Returns
    ---------
    transform: (4,4) float
        Transformation matrix to move points onto XY plane
    """
    transform = align_vectors(normal, [0, 0, 1])
    if origin is not None:
        transform[:3, 3] = -np.dot(transform, np.append(origin, 1))[:3]
    return transform


def align_vectors(a, b, return_angle=False):
    """
    Find the rotation matrix that transforms one 3D vector
    to another.

    Parameters
    ------------
    a : (3,) float
      Unit vector
    b : (3,) float
      Unit vector
    return_angle : bool
      Return the angle between vectors or not

    Returns
    -------------
    matrix : (4, 4) float
      Homogeneous transform to rotate from `a` to `b`
    angle : float
      If `return_angle` angle in radians between `a` and `b`

    """
    a = np.array(a, dtype=np.float64)
    b = np.array(b, dtype=np.float64)
    if a.shape != (3,) or b.shape != (3,):
        raise ValueError("vectors must be (3,)!")

    # find the SVD of the two vectors
    au = np.linalg.svd(a.reshape((-1, 1)))[0]
    bu = np.linalg.svd(b.reshape((-1, 1)))[0]

    if np.linalg.det(au) < 0:
        au[:, -1] *= -1.0
    if np.linalg.det(bu) < 0:
        bu[:, -1] *= -1.0

    # put rotation into homogeneous transformation
    matrix = np.eye(4)
    matrix[:3, :3] = bu.dot(au.T)

    if return_angle:
        # projection of a onto b
        # first row of SVD result is normalized source vector
        dot = np.dot(au[0], bu[0])
        # clip to avoid floating point error
        angle = np.arccos(np.clip(dot, -1.0, 1.0))
        if dot < -1e-5:
            angle += np.pi
        return matrix, angle

    return matrix


def faces_to_edges(faces, return_index=False):
    """
    Given a list of faces (n,3), return a list of edges (n*3,2)

    Parameters
    -----------
    faces : (n, 3) int
      Vertex indices representing faces

    Returns
    -----------
    edges : (n*3, 2) int
      Vertex indices representing edges
    """
    faces = np.asanyarray(faces)

    # each face has three edges
    edges = faces[:, [0, 1, 1, 2, 2, 0]].reshape((-1, 2))

    if return_index:
        # edges are in order of faces due to reshape
        face_index = np.tile(np.arange(len(faces)), (3, 1)).T.reshape(-1)
        return edges, face_index
    return edges


def vector_angle(pairs):
    """
    Find the angles between pairs of unit vectors.

    Parameters
    ----------
    pairs : (n, 2, 3) float
      Unit vector pairs

    Returns
    ----------
    angles : (n,) float
      Angles between vectors in radians
    """
    pairs = np.asanyarray(pairs, dtype=np.float64)
    if len(pairs) == 0:
        return np.array([])
    elif util.is_shape(pairs, (2, 3)):
        pairs = pairs.reshape((-1, 2, 3))
    elif not util.is_shape(pairs, (-1, 2, (2, 3))):
        raise ValueError("pairs must be (n,2,(2|3))!")

    # do the dot product between vectors
    dots = util.diagonal_dot(pairs[:, 0], pairs[:, 1])
    # clip for floating point error
    dots = np.clip(dots, -1.0, 1.0)
    # do cos and remove arbitrary sign
    angles = np.abs(np.arccos(dots))

    return angles


def triangulate_quads(quads, dtype=np.int64):
    """
    Given an array of quad faces return them as triangle faces,
    also handles pure triangles and mixed triangles and quads.

    Parameters
    -----------
    quads: (n, 4) int
      Vertex indices of quad faces.

    Returns
    -----------
    faces : (m, 3) int
      Vertex indices of triangular faces.c
    """

    if len(quads) == 0:
        return np.zeros(0, dtype=dtype)

    try:
        # this will fail in newer versions of numpy
        # if there are mixed quads and tris
        quads = np.array(quads, dtype=dtype)

        if len(quads.shape) == 2 and quads.shape[1] == 3:
            # if they are just triangles return immediately
            return quads.astype(dtype)

        if len(quads.shape) == 2 and quads.shape[1] == 4:
            # if they are just quads stack and return
            return np.vstack((quads[:, [0, 1, 2]], quads[:, [2, 3, 0]]), dtype=dtype)
    except ValueError:
        # new numpy raises an error for sequences
        pass

    # we made it here so we have mixed tris/quads/polygons
    # filter into the three cases
    tri = np.array([i for i in quads if len(i) == 3])
    quad = np.array([i for i in quads if len(i) == 4])
    # triangulate arbitrary polygons as triangle fans
    # this isn't guaranteed to be sane if the polygons
    # aren't convex but that would require a real maniac
    poly = [[[f[0], f[i + 1], f[i + 2]] for i in range(len(f) - 2)] for f in quads if len(f) > 4]

    if len(quad) == 0 and len(poly) == 0:
        return tri.astype(dtype)
    if len(poly) > 0:
        poly = np.vstack(poly)
    if len(quad) > 0:
        quad = np.vstack((quad[:, [0, 1, 2]], quad[:, [2, 3, 0]]))
    # combine triangulated quads with triangles
    return util.vstack_empty([tri, quad, poly]).astype(dtype)


def vertex_face_indices(vertex_count, faces, faces_sparse):
    """
    Find vertex face indices from the faces array of vertices

    Parameters
    -----------
    vertex_count : int
      The number of vertices faces refer to
    faces : (n, 3) int
      List of vertex indices
    faces_sparse : scipy.sparse.COO
      Sparse matrix

    Returns
    -----------
    vertex_faces : (vertex_count, ) int
      Face indices for every vertex
      Array padded with -1 in each row for all vertices with fewer
      face indices than the max number of face indices.
    """
    # Create 2D array with row for each vertex and
    # length of max number of faces for a vertex
    try:
        counts = np.bincount(faces.flatten(), minlength=vertex_count)
    except TypeError:
        # casting failed on 32 bit Windows
        log.warning("casting failed, falling back!")
        # fall back to np.unique (usually ~35x slower than bincount)
        counts = np.unique(faces.flatten(), return_counts=True)[1]
    assert len(counts) == vertex_count
    assert faces.max() < vertex_count

    # start cumulative sum at zero and clip off the last value
    starts = np.append(0, np.cumsum(counts)[:-1])
    # pack incrementing array into final shape
    pack = np.arange(counts.max()) + starts[:, None]
    # pad each row with -1 to pad to the max length
    padded = -(pack >= (starts + counts)[:, None]).astype(np.int64)

    try:
        # do most of the work with a sparse dot product
        identity = scipy.sparse.identity(len(faces), dtype=int)
        sorted_faces = faces_sparse.dot(identity).nonzero()[1]
        # this will fail if any face was degenerate
        # TODO
        # figure out how to filter out degenerate faces from sparse
        # result if sorted_faces.size != faces.size
        padded[padded == 0] = sorted_faces
    except BaseException:
        # fall back to a slow loop
        log.warning("vertex_faces falling back to slow loop! " + "mesh probably has degenerate faces", exc_info=True)
        sort = np.zeros(faces.size, dtype=np.int64)
        flat = faces.flatten()
        for v in range(vertex_count):
            # assign the data in order
            sort[starts[v] : starts[v] + counts[v]] = (np.where(flat == v)[0] // 3)[::-1]
        padded[padded == 0] = sort
    return padded


def mean_vertex_normals(vertex_count, faces, face_normals, sparse=None, **kwargs):
    """
    Find vertex normals from the mean of the faces that contain
    that vertex.

    Parameters
    -----------
    vertex_count : int
      The number of vertices faces refer to
    faces : (n, 3) int
      List of vertex indices
    face_normals : (n, 3) float
      Normal vector for each face

    Returns
    -----------
    vertex_normals : (vertex_count, 3) float
      Normals for every vertex
      Vertices unreferenced by faces will be zero.
    """

    def summed_sparse():
        # use a sparse matrix of which face contains each vertex to
        # figure out the summed normal at each vertex
        # allow cached sparse matrix to be passed
        if sparse is None:
            matrix = index_sparse(vertex_count, faces)
        else:
            matrix = sparse
        summed = matrix.dot(face_normals)
        return summed

    def summed_loop():
        # loop through every face, in tests was ~50x slower than
        # doing this with a sparse matrix
        summed = np.zeros((vertex_count, 3))
        for face, normal in zip(faces, face_normals):
            summed[face] += normal
        return summed

    try:
        summed = summed_sparse()
    except BaseException:
        log.warning("unable to use sparse matrix, falling back!", exc_info=True)
        summed = summed_loop()

    # invalid normals will be returned as zero
    vertex_normals = util.unitize(summed)

    return vertex_normals


def weighted_vertex_normals(vertex_count, faces, face_normals, face_angles, use_loop=False):
    """
    Compute vertex normals from the faces that contain that vertex.
    The contibution of a face's normal to a vertex normal is the
    ratio of the corner-angle in which the vertex is, with respect
    to the sum of all corner-angles surrounding the vertex.

    Grit Thuerrner & Charles A. Wuethrich (1998)
    Computing Vertex Normals from Polygonal Facets,
    Journal of Graphics Tools, 3:1, 43-46

    Parameters
    -----------
    vertex_count : int
      The number of vertices faces refer to
    faces : (n, 3) int
      List of vertex indices
    face_normals : (n, 3) float
      Normal vector for each face
    face_angles : (n, 3) float
      Angles at each vertex in the face

    Returns
    -----------
    vertex_normals : (vertex_count, 3) float
      Normals for every vertex
      Vertices unreferenced by faces will be zero.
    """

    def summed_sparse():
        # use a sparse matrix of which face contains each vertex to
        # figure out the summed normal at each vertex
        # allow cached sparse matrix to be passed
        # fill the matrix with vertex-corner angles as weights
        matrix = index_sparse(vertex_count, faces, data=face_angles.ravel())
        return matrix.dot(face_normals)

    def summed_loop():
        summed = np.zeros((vertex_count, 3), np.float64)
        for vertex_idx in np.arange(vertex_count):
            # loop over all vertices
            # compute normal contributions from surrounding faces
            # obviously slower than with the sparse matrix
            face_idxs, inface_idxs = np.where(faces == vertex_idx)
            surrounding_angles = face_angles[face_idxs, inface_idxs]
            summed[vertex_idx] = np.dot(surrounding_angles / surrounding_angles.sum(), face_normals[face_idxs])

        return summed

    # normals should be unit vectors
    face_ok = (face_normals**2).sum(axis=1) > 0.5
    # don't consider faces with invalid normals
    faces = faces[face_ok]
    face_normals = face_normals[face_ok]
    face_angles = face_angles[face_ok]

    if not use_loop:
        try:
            return util.unitize(summed_sparse())
        except BaseException:
            log.warning("unable to use sparse matrix, falling back!", exc_info=True)
    # we either crashed or were asked to loop
    return util.unitize(summed_loop())


def index_sparse(columns, indices, data=None, dtype=None):
    """
        Return a sparse matrix for which vertices are contained in which faces.
        A data vector can be passed which is then used instead of booleans

        Parameters
        ------------
        columns : int
          Number of columns, usually number of vertices
        indices : (m, d) int
          Usually mesh.faces

        Returns
        ---------
        sparse: scipy.sparse.coo_matrix of shape (columns, len(faces))
                dtype is boolean

        Examples
         ----------
        In [1]: sparse = faces_sparse(len(mesh.vertices), mesh.faces)

        In [2]: sparse.shape
        Out[2]: (12, 20)

        In [3]: mesh.faces.shape
        Out[3]: (20, 3)
    co
        In [4]: mesh.vertices.shape
        Out[4]: (12, 3)

        In [5]: dense = sparse.toarray().astype(int)

        In [6]: dense
        Out[6]:
        array([[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
               [0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0],
               [0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1],
               [1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0],
               [0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0],
               [0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1],
               [0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1]])

        In [7]: dense.sum(axis=0)
        Out[7]: array([3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3])
    """
    indices = np.asanyarray(indices)
    columns = int(columns)

    # flattened list
    row = indices.reshape(-1)
    col = np.tile(np.arange(len(indices)).reshape((-1, 1)), (1, indices.shape[1])).reshape(-1)

    shape = (columns, len(indices))
    if data is None:
        data = np.ones(len(col), dtype=bool)
    elif len(data) != len(col):
        raise ValueError("data incorrect length")

    if dtype is not None:
        data = data.astype(dtype)

    # assemble into sparse matrix
    matrix = scipy.sparse.coo_matrix((data, (row, col)), shape=shape, dtype=data.dtype)

    return matrix
