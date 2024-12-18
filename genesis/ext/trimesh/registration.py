"""
registration.py
---------------

Functions for registering (aligning) point clouds with meshes.
"""

import numpy as np


from . import util
from . import bounds
from . import transformations

from .points import PointCloud, plane_fit
from .geometry import weighted_vertex_normals
from .triangles import normals, angles, cross
from .transformations import transform_points

try:
    from scipy.spatial import cKDTree
    import scipy.sparse as sparse
except BaseException as E:
    # wrapping just ImportError fails in some cases
    # will raise the error when someone tries to use KDtree
    from . import exceptions

    cKDTree = exceptions.ExceptionWrapper(E)
    sparse = exceptions.ExceptionWrapper(E)


def mesh_other(mesh, other, samples=500, scale=False, icp_first=10, icp_final=50, **kwargs):
    """
    Align a mesh with another mesh or a PointCloud using
    the principal axes of inertia as a starting point which
    is refined by iterative closest point.

    Parameters
    ------------
    mesh : trimesh.Trimesh object
      Mesh to align with other
    other : trimesh.Trimesh or (n, 3) float
      Mesh or points in space
    samples : int
      Number of samples from mesh surface to align
    scale : bool
      Allow scaling in transform
    icp_first : int
      How many ICP iterations for the 9 possible
      combinations of sign flippage
    icp_final : int
      How many ICP iterations for the closest
      candidate from the wider search
    kwargs : dict
      Passed through to `icp`, which passes through to `procrustes`

    Returns
    -----------
    mesh_to_other : (4, 4) float
      Transform to align mesh to the other object
    cost : float
      Average squared distance per point
    """

    def key_points(m, count):
        """
        Return a combination of mesh vertices and surface samples
        with vertices chosen by likelihood to be important
        to registration.
        """
        if len(m.vertices) < (count / 2):
            return np.vstack((m.vertices, m.sample(count - len(m.vertices))))
        else:
            return m.sample(count)

    if not util.is_instance_named(mesh, "Trimesh"):
        raise ValueError("mesh must be Trimesh object!")

    inverse = True
    search = mesh
    # if both are meshes use the smaller one for searching
    if util.is_instance_named(other, "Trimesh"):
        if len(mesh.vertices) > len(other.vertices):
            # do the expensive tree construction on the
            # smaller mesh and query the others points
            search = other
            inverse = False
            points = key_points(m=mesh, count=samples)
            points_mesh = mesh
        else:
            points_mesh = other
            points = key_points(m=other, count=samples)

        if points_mesh.is_volume:
            points_PIT = points_mesh.principal_inertia_transform
        else:
            points_PIT = points_mesh.bounding_box_oriented.principal_inertia_transform

    elif util.is_shape(other, (-1, 3)):
        # case where other is just points
        points = other
        points_PIT = bounds.oriented_bounds(points)[0]
    else:
        raise ValueError("other must be mesh or (n, 3) points!")

    # get the transform that aligns the search mesh principal
    # axes of inertia with the XYZ axis at the origin
    if search.is_volume:
        search_PIT = search.principal_inertia_transform
    else:
        search_PIT = search.bounding_box_oriented.principal_inertia_transform

    # transform that moves the principal axes of inertia
    # of the search mesh to be aligned with the best- guess
    # principal axes of the points
    search_to_points = np.dot(np.linalg.inv(points_PIT), search_PIT)

    # permutations of cube rotations
    # the principal inertia transform has arbitrary sign
    # along the 3 major axis so try all combinations of
    # 180 degree rotations with a quick first ICP pass
    cubes = np.array(
        [
            np.eye(4) * np.append(diag, 1)
            for diag in [
                [1, 1, 1],
                [1, 1, -1],
                [1, -1, 1],
                [-1, 1, 1],
                [-1, -1, 1],
                [-1, 1, -1],
                [1, -1, -1],
                [-1, -1, -1],
            ]
        ]
    )

    # loop through permutations and run iterative closest point
    costs = np.ones(len(cubes)) * np.inf
    transforms = [None] * len(cubes)
    centroid = search.centroid

    for i, flip in enumerate(cubes):
        # transform from points to search mesh
        # flipped around the centroid of search
        a_to_b = np.dot(transformations.transform_around(flip, centroid), np.linalg.inv(search_to_points))

        # run first pass ICP
        matrix, junk, cost = icp(a=points, b=search, initial=a_to_b, max_iterations=int(icp_first), scale=scale)

        # save transform and costs from ICP
        transforms[i] = matrix
        costs[i] = cost

    # run a final ICP refinement step
    matrix, junk, cost = icp(
        a=points, b=search, initial=transforms[np.argmin(costs)], max_iterations=int(icp_final), scale=scale
    )

    # convert to per- point distance average
    cost /= len(points)

    # we picked the smaller mesh to construct the tree
    # on so we may have calculated a transform backwards
    # to save computation, so just invert matrix here
    if inverse:
        mesh_to_other = np.linalg.inv(matrix)
    else:
        mesh_to_other = matrix

    return mesh_to_other, cost


def procrustes(a, b, weights=None, reflection=True, translation=True, scale=True, return_cost=True):
    """
    Perform Procrustes' analysis subject to constraints. Finds the
    transformation T mapping a to b which minimizes the square sum
    distances between Ta and b, also called the cost. Optionally
    specify different weights for the points in a to minimize the
    weighted square sum distances between Ta and b. This can
    improve transformation robustness on noisy data if the points'
    probability distribution is known.

    Parameters
    ----------
    a : (n,3) float
      List of points in space
    b : (n,3) float
      List of points in space
    weights : (n,) float
      List of floats representing how much weight is assigned to each point of a
    reflection : bool
      If the transformation is allowed reflections
    translation : bool
      If the transformation is allowed translations
    scale : bool
      If the transformation is allowed scaling
    return_cost : bool
      Whether to return the cost and transformed a as well

    Returns
    ----------
    matrix : (4,4) float
      The transformation matrix sending a to b
    transformed : (n,3) float
      The image of a under the transformation
    cost : float
      The cost of the transformation
    """

    a = np.asanyarray(a, dtype=np.float64)
    b = np.asanyarray(b, dtype=np.float64)
    if not util.is_shape(a, (-1, 3)) or not util.is_shape(b, (-1, 3)):
        raise ValueError("points must be (n,3)!")
    if len(a) != len(b):
        raise ValueError("a and b must contain same number of points!")
    if weights is not None:
        w = np.asanyarray(weights, dtype=np.float64)
        if len(w) != len(a):
            raise ValueError("weights must have same length as a and b!")
        w_norm = (w / w.sum()).reshape((-1, 1))

    # Remove translation component
    if translation:
        # acenter is a weighted average of the individual points.
        if weights is None:
            acenter = a.mean(axis=0)
        else:
            acenter = (a * w_norm).sum(axis=0)
        bcenter = b.mean(axis=0)
    else:
        acenter = np.zeros(a.shape[1])
        bcenter = np.zeros(b.shape[1])

    # Remove scale component
    if scale:
        if weights is None:
            ascale = np.sqrt(((a - acenter) ** 2).sum() / len(a))
            # ascale is the square root of weighted average of the
            # squared difference
            # between each point and acenter.
        else:
            ascale = np.sqrt((((a - acenter) ** 2) * w_norm).sum())

        bscale = np.sqrt(((b - bcenter) ** 2).sum() / len(b))
    else:
        ascale = 1
        bscale = 1

    # Use SVD to find optimal orthogonal matrix R
    # constrained to det(R) = 1 if necessary.
    # w_mat is multiplied with the centered and scaled a, such that the points
    # can be weighted differently.

    if weights is None:
        target = np.dot(((b - bcenter) / bscale).T, ((a - acenter) / ascale))
    else:
        target = np.dot(((b - bcenter) / bscale).T, ((a - acenter) / ascale) * w.reshape((-1, 1)))

    u, s, vh = np.linalg.svd(target)

    if reflection:
        R = np.dot(u, vh)
    else:
        # no reflection allowed, so determinant must be 1.0
        R = np.dot(np.dot(u, np.diag([1, 1, np.linalg.det(np.dot(u, vh))])), vh)

    # Compute our 4D transformation matrix encoding
    # a -> (R @ (a - acenter)/ascale) * bscale + bcenter
    #    = (bscale/ascale)R @ a + (bcenter - (bscale/ascale)R @ acenter)
    translation = bcenter - (bscale / ascale) * np.dot(R, acenter)
    matrix = np.hstack((bscale / ascale * R, translation.reshape(-1, 1)))
    matrix = np.vstack((matrix, np.array([0.0] * (a.shape[1]) + [1.0]).reshape(1, -1)))

    if return_cost:
        transformed = transform_points(a, matrix)
        cost = ((b - transformed) ** 2).mean()
        return matrix, transformed, cost
    else:
        return matrix


def icp(a, b, initial=None, threshold=1e-5, max_iterations=20, **kwargs):
    """
    Apply the iterative closest point algorithm to align a point cloud with
    another point cloud or mesh. Will only produce reasonable results if the
    initial transformation is roughly correct. Initial transformation can be
    found by applying Procrustes' analysis to a suitable set of landmark
    points (often picked manually).

    Parameters
    ----------
    a : (n,3) float
      List of points in space.
    b : (m,3) float or Trimesh
      List of points in space or mesh.
    initial : (4,4) float
      Initial transformation.
    threshold : float
      Stop when change in cost is less than threshold
    max_iterations : int
      Maximum number of iterations
    kwargs : dict
      Args to pass to procrustes

    Returns
    ----------
    matrix : (4,4) float
      The transformation matrix sending a to b
    transformed : (n,3) float
      The image of a under the transformation
    cost : float
      The cost of the transformation
    """

    a = np.asanyarray(a, dtype=np.float64)
    if not util.is_shape(a, (-1, 3)):
        raise ValueError("points must be (n,3)!")

    if initial is None:
        initial = np.eye(4)

    is_mesh = util.is_instance_named(b, "Trimesh")
    if not is_mesh:
        b = np.asanyarray(b, dtype=np.float64)
        if not util.is_shape(b, (-1, 3)):
            raise ValueError("points must be (n,3)!")
        btree = cKDTree(b)

    # transform a under initial_transformation
    a = transform_points(a, initial)
    total_matrix = initial

    # start with infinite cost
    old_cost = np.inf

    # avoid looping forever by capping iterations
    for _ in range(max_iterations):
        # Closest point in b to each point in a
        if is_mesh:
            closest, distance, faces = b.nearest.on_surface(a)
        else:
            distances, ix = btree.query(a, 1)
            closest = b[ix]

        # align a with closest points
        matrix, transformed, cost = procrustes(a=a, b=closest, **kwargs)

        # update a with our new transformed points
        a = transformed
        total_matrix = np.dot(matrix, total_matrix)

        if old_cost - cost < threshold:
            break
        else:
            old_cost = cost

    return total_matrix, transformed, cost


def _normalize_by_source(source_mesh, target_geometry, target_positions):
    # Utility function to put the source mesh in [-1, 1]^3 and transform
    # target geometry accordingly
    if not util.is_instance_named(target_geometry, "Trimesh") and not isinstance(target_geometry, PointCloud):
        vertices = np.asanyarray(target_geometry)
        target_geometry = PointCloud(vertices)
    centroid, scale = source_mesh.centroid, source_mesh.scale
    source_mesh.vertices = (source_mesh.vertices - centroid[None, :]) / scale
    # Dont forget to also transform the target positions
    target_geometry.vertices = (target_geometry.vertices - centroid[None, :]) / scale
    if target_positions is not None:
        target_positions = (target_positions - centroid[None, :]) / scale
    return target_geometry, target_positions, centroid, scale


def _denormalize_by_source(source_mesh, target_geometry, target_positions, result, centroid, scale):
    # Utility function to transform source mesh from
    # [-1, 1]^3 to its original transform
    # and transform target geometry accordingly
    source_mesh.vertices = scale * source_mesh.vertices + centroid[None, :]
    target_geometry.vertices = scale * target_geometry.vertices + centroid[None, :]
    if target_positions is not None:
        target_positions = scale * target_positions + centroid[None, :]
    if isinstance(result, list):
        result = [scale * x + centroid[None, :] for x in result]
    else:
        result = scale * result + centroid[None, :]
    return result


def nricp_amberg(
    source_mesh,
    target_geometry,
    source_landmarks=None,
    target_positions=None,
    steps=None,
    eps=0.0001,
    gamma=1,
    distance_threshold=0.1,
    return_records=False,
    use_faces=True,
    use_vertex_normals=True,
    neighbors_count=8,
):
    """
    Non Rigid Iterative Closest Points

    Implementation of "Amberg et al. 2007: Optimal Step
    Nonrigid ICP Algorithms for Surface Registration."
    Allows to register non-rigidly a mesh on another or
    on a point cloud. The core algorithm is explained
    at the end of page 3 of the paper.

    Comparison between nricp_amberg and nricp_sumner:
    * nricp_amberg fits to the target mesh in less steps
    * nricp_amberg can generate sharp edges
      * only vertices and their neighbors are considered
    * nricp_sumner tend to preserve more the original shape
    * nricp_sumner parameters are easier to tune
    * nricp_sumner solves for triangle positions whereas
      nricp_amberg solves for vertex transforms
    * nricp_sumner is less optimized when wn > 0

    Parameters
    ----------
    source_mesh : Trimesh
        Source mesh containing both vertices and faces.
    target_geometry : Trimesh or PointCloud or (n, 3) float
        Target geometry. It can contain no faces or be a PointCloud.
    source_landmarks : (n,) int or ((n,) int, (n, 3) float)
        n landmarks on the the source mesh.
        Represented as vertex indices (n,) int.
        It can also be represented as a tuple of triangle
        indices and barycentric coordinates ((n,) int, (n, 3) float,).
    target_positions : (n, 3) float
        Target positions assigned to source landmarks
    steps : Core parameters of the algorithm
        Iterable of iterables (ws, wl, wn, max_iter,).
        ws is smoothness term, wl weights landmark importance, wn normal importance
        and max_iter is the maximum number of iterations per step.
    eps : float
        If the error decrease if inferior to this value, the current step ends.
    gamma : float
        Weight the translation part against the rotational/skew part.
        Recommended value : 1.
    distance_threshold : float
        Distance threshold to account for a vertex match or not.
    return_records : bool
        If True, also returns all the intermediate results. It can help debugging
        and tune the parameters to match a specific case.
    use_faces : bool
        If True and if target geometry has faces, use proximity.closest_point to find
        matching points. Else use scipy's cKDTree object.
    use_vertex_normals : bool
        If True and if target geometry has faces, interpolate the normals of the target
        geometry matching points.
        Else use face normals or estimated normals if target geometry has no faces.
    neighbors_count : int
        number of neighbors used for normal estimation. Only used if target geometry has
        no faces or if use_faces is False.

    Returns
    ----------
    result : (n, 3) float or List[(n, 3) float]
        The vertices positions of source_mesh such that it is registered non-rigidly
        onto the target geometry.
        If return_records is True, it returns the list of the vertex positions at each
        iteration.
    """

    def _solve_system(M_kron_G, D, vertices_weight, nearest, ws, nE, nV, Dl, Ul, wl):
        # Solve for Eq. 12
        U = nearest * vertices_weight[:, None]
        use_landmarks = Dl is not None and Ul is not None
        A_stack = [ws * M_kron_G, D.multiply(vertices_weight[:, None])]
        B_shape = (4 * nE + nV, 3)
        if use_landmarks:
            A_stack.append(wl * Dl)
            B_shape = (4 * nE + nV + Ul.shape[0], 3)
        A = sparse.csr_matrix(sparse.vstack(A_stack))
        B = sparse.lil_matrix(B_shape, dtype=np.float32)
        B[4 * nE : (4 * nE + nV), :] = U
        if use_landmarks:
            B[4 * nE + nV : (4 * nE + nV + Ul.shape[0]), :] = Ul * wl
        X = sparse.linalg.spsolve(A.T * A, A.T * B).toarray()
        return X

    def _node_arc_incidence(mesh, do_weight):
        # Computes node-arc incidence matrix of mesh (Eq.10)
        nV = mesh.edges.max() + 1
        nE = len(mesh.edges)
        rows = np.repeat(np.arange(nE), 2)
        cols = mesh.edges.flatten()
        data = np.ones(2 * nE, np.float32)
        data[1::2] = -1
        if do_weight:
            edge_lengths = np.linalg.norm(mesh.vertices[mesh.edges[:, 0]] - mesh.vertices[mesh.edges[:, 1]], axis=-1)
            data *= np.repeat(1 / edge_lengths, 2)
        return sparse.coo_matrix((data, (rows, cols)), shape=(nE, nV))

    def _create_D(vertex_3d_data):
        # Create Data matrix (Eq. 8)
        nV = len(vertex_3d_data)
        rows = np.repeat(np.arange(nV), 4)
        cols = np.arange(4 * nV)
        data = np.concatenate((vertex_3d_data, np.ones((nV, 1))), axis=-1).flatten()
        return sparse.csr_matrix((data, (rows, cols)), shape=(nV, 4 * nV))

    def _create_X(nV):
        # Create Unknowns Matrix (Eq. 1)
        X_ = np.concatenate((np.eye(3), np.array([[0, 0, 0]])), axis=0)
        return np.tile(X_, (nV, 1))

    def _create_Dl_Ul(D, source_mesh, source_landmarks, target_positions):
        # Create landmark terms (Eq. 11)
        Dl, Ul = None, None

        if source_landmarks is None or target_positions is None:
            # If no landmarks are provided, return None for both
            return Dl, Ul

        if isinstance(source_landmarks, tuple):
            source_tids, source_barys = source_landmarks
            source_tri_vids = source_mesh.faces[source_tids]
            # u * x1, v * x2 and w * x3 combined
            Dl = D[source_tri_vids.flatten(), :]
            Dl.data *= source_barys.flatten().repeat(np.diff(Dl.indptr))
            x0 = source_mesh.vertices[source_tri_vids[:, 0]]
            x1 = source_mesh.vertices[source_tri_vids[:, 1]]
            x2 = source_mesh.vertices[source_tri_vids[:, 2]]
            Ul0 = target_positions - x1 * source_barys[:, 1, None] - x2 * source_barys[:, 2, None]
            Ul1 = target_positions - x0 * source_barys[:, 0, None] - x2 * source_barys[:, 2, None]
            Ul2 = target_positions - x0 * source_barys[:, 0, None] - x1 * source_barys[:, 1, None]
            Ul = np.zeros((Ul0.shape[0] * 3, 3))
            Ul[0::3] = Ul0  # y - v * x2 + w * x3
            Ul[1::3] = Ul1  # y - u * x1 + w * x3
            Ul[2::3] = Ul2  # y - u * x1 + v * x2
        else:
            Dl = D[source_landmarks, :]
            Ul = target_positions
        return Dl, Ul

    target_geometry, target_positions, centroid, scale = _normalize_by_source(
        source_mesh, target_geometry, target_positions
    )

    # Number of edges and vertices in source mesh
    nE = len(source_mesh.edges)
    nV = len(source_mesh.vertices)

    # Initialize transformed vertices
    transformed_vertices = source_mesh.vertices.copy()
    # Node-arc incidence (M in Eq. 10)
    M = _node_arc_incidence(source_mesh, True)
    # G (Eq. 10)
    G = np.diag([1, 1, 1, gamma])
    # M kronecker G (Eq. 10)
    M_kron_G = sparse.kron(M, G)
    # D (Eq. 8)
    D = _create_D(source_mesh.vertices)
    # D but for normal computation from the transformations X
    DN = _create_D(source_mesh.vertex_normals)
    # Unknowns 4x3 transformations X (Eq. 1)
    X = _create_X(nV)
    # Landmark related terms (Eq. 11)
    Dl, Ul = _create_Dl_Ul(D, source_mesh, source_landmarks, target_positions)

    # Parameters of the algorithm (Eq. 6)
    # order : Alpha, Beta, normal weighting, and max iteration for step
    if steps is None:
        steps = [
            [0.01, 10, 0.5, 10],
            [0.02, 5, 0.5, 10],
            [0.03, 2.5, 0.5, 10],
            [0.01, 0, 0.0, 10],
        ]
    if return_records:
        records = [transformed_vertices]

    # Main loop
    for ws, wl, wn, max_iter in steps:

        # If normals are estimated from points and if there are less
        # than 3 points per query, avoid normal estimation
        if not use_faces and neighbors_count < 3:
            wn = 0

        last_error = np.finfo(np.float32).max
        error = np.finfo(np.float16).max
        cpt_iter = 0

        # Current step iterations loop
        while last_error - error > eps and (max_iter is None or cpt_iter < max_iter):

            qres = _from_mesh(
                target_geometry,
                transformed_vertices,
                from_vertices_only=not use_faces,
                return_normals=wn > 0,
                return_interpolated_normals=wn > 0 and use_vertex_normals,
                neighbors_count=neighbors_count,
            )

            # Data weighting
            vertices_weight = np.ones(nV)
            vertices_weight[qres["distances"] > distance_threshold] = 0

            if wn > 0 and "normals" in qres:
                target_normals = qres["normals"]
                if use_vertex_normals and "interpolated_normals" in qres:
                    target_normals = qres["interpolated_normals"]
                # Normal weighting = multiplying weights by cosines^wn
                source_normals = DN * X
                dot = util.diagonal_dot(source_normals, target_normals)
                # Normal orientation is only known for meshes as target
                dot = np.clip(dot, 0, 1) if use_faces else np.abs(dot)
                vertices_weight = vertices_weight * dot**wn

            # Actual system solve
            X = _solve_system(M_kron_G, D, vertices_weight, qres["nearest"], ws, nE, nV, Dl, Ul, wl)
            transformed_vertices = D * X
            last_error = error
            error_vec = np.linalg.norm(qres["nearest"] - transformed_vertices, axis=-1)
            error = (error_vec * vertices_weight).mean()
            if return_records:
                records.append(transformed_vertices)
            cpt_iter += 1

    if return_records:
        result = records
    else:
        result = transformed_vertices

    result = _denormalize_by_source(source_mesh, target_geometry, target_positions, result, centroid, scale)
    return result


def _from_mesh(
    mesh,
    input_points,
    from_vertices_only=False,
    return_barycentric_coordinates=False,
    return_normals=False,
    return_interpolated_normals=False,
    neighbors_count=10,
    **kwargs,
):
    """
    Find the the closest points and associated attributes from a Trimesh.

    Parameters
    -----------
    mesh : Trimesh
        Trimesh from which the query is performed
    input_points : (m, 3) float
        Input query points
    from_vertices_only : bool
        If True, consider only the vertices and not the faces
    return_barycentric_coordinates : bool
        If True, return the barycentric coordinates
    return_normals : bool
        If True, compute the normals at each closest point
    return_interpolated_normals : bool
        If True, return the interpolated normal at each closest point
    neighbors_count : int
        The number of closest neighbors to query
    kwargs : dict
        Dict to accept other key word arguments (not used)
    Returns
    ----------
    qres : Dict
      Dictionary containing :
       - nearest points (m, 3) with key 'nearest'
       - distances to nearest point (m,) with key 'distances'
       - support triangle indices of the nearest points (m,) with key 'tids'
       - [optional] normals at nearest points (m,3) with key 'normals'
       - [optional] barycentric coordinates in support triangles (m,3) with key
         'barycentric_coordinates'
       - [optional] interpolated normals (m,3) with key 'interpolated_normals'
    """
    input_points = np.asanyarray(input_points)
    neighbors_count = min(neighbors_count, len(mesh.vertices))

    if from_vertices_only or len(mesh.faces) == 0:
        # Consider only the vertices
        return _from_points(
            mesh.vertices, input_points, mesh.kdtree, return_normals=return_normals, neighbors_count=neighbors_count
        )
    # Else if we consider faces, use proximity.closest_point
    qres = {}
    from .proximity import closest_point
    from .triangles import points_to_barycentric

    qres["nearest"], qres["distances"], qres["tids"] = closest_point(mesh, input_points)

    if return_normals:
        qres["normals"] = mesh.face_normals[qres["tids"]]
    if return_barycentric_coordinates or return_interpolated_normals:
        qres["barycentric_coordinates"] = points_to_barycentric(
            mesh.vertices[mesh.faces[qres["tids"]]], qres["nearest"]
        )

        if return_interpolated_normals:
            # Interpolation from barycentric coordinates
            qres["interpolated_normals"] = np.einsum(
                "ij,ijk->ik", qres["barycentric_coordinates"], mesh.vertex_normals[mesh.faces[qres["tids"]]]
            )
    return qres


def _from_points(target_points, input_points, kdtree=None, return_normals=False, neighbors_count=10, **kwargs):
    """
    Find the the closest points and associated attributes
    from a set of 3D points.

    Parameters
    -----------
    target_points : (n, 3) float
      Points from which the query is performed
    input_points : (m, 3) float
      Input query points
    kdtree : scipy.cKDTree
      KDTree used for query. Computed if not provided
    return_normals : bool
      If True, compute the normals at each nearest point
    neighbors_count : int
      The number of closest neighbors to query
    kwargs : dict
      Dict to accept other key word arguments (not used)

    Returns
    ----------
    qres : Dict
      Dictionary containing :
       - nearest points (m, 3) with key 'nearest'
       - distances to nearest point (m,) with key 'distances'
       - vertex indices of the nearest points (m,) with key 'vertex_indices'
       - [optional] normals at nearest points (m,3) with key 'normals'
    """
    # Empty result
    target_points = np.asanyarray(target_points)
    input_points = np.asanyarray(input_points)
    neighbors_count = min(neighbors_count, len(target_points))
    qres = {}
    if kdtree is None:
        kdtree = cKDTree(target_points)

    if return_normals:
        assert neighbors_count >= 3
        distances, indices = kdtree.query(input_points, k=neighbors_count)
        nearest = target_points[indices, :]
        qres["normals"] = plane_fit(nearest)[1]
        qres["nearest"] = nearest[:, 0]
        qres["distances"] = distances[:, 0]
        qres["vertex_indices"] = indices[:, 0]
    else:
        qres["distances"], qres["vertex_indices"] = kdtree.query(input_points)
        qres["nearest"] = target_points[qres["vertex_indices"], :]

    return qres


def nricp_sumner(
    source_mesh,
    target_geometry,
    source_landmarks=None,
    target_positions=None,
    steps=None,
    distance_threshold=0.1,
    return_records=False,
    use_faces=True,
    use_vertex_normals=True,
    neighbors_count=8,
    face_pairs_type="vertex",
):
    """
    Non Rigid Iterative Closest Points

    Implementation of the correspondence computation part of
    "Sumner and Popovic 2004: Deformation Transfer for Triangle Meshes"
    Allows to register non-rigidly a mesh on another geometry.

    Comparison between nricp_amberg and nricp_sumner:
    * nricp_amberg fits to the target mesh in less steps
    * nricp_amberg can generate sharp edges (only vertices and their
        neighbors are considered)
    * nricp_sumner tend to preserve more the original shape
    * nricp_sumner parameters are easier to tune
    * nricp_sumner solves for triangle positions whereas nricp_amberg solves for
        vertex transforms
    * nricp_sumner is less optimized when wn > 0

    Parameters
    ----------
    source_mesh : Trimesh
        Source mesh containing both vertices and faces.
    target_geometry : Trimesh or PointCloud or (n, 3) float
        Target geometry. It can contain no faces or be a PointCloud.
    source_landmarks : (n,) int or ((n,) int, (n, 3) float)
        n landmarks on the the source mesh.
        Represented as vertex indices (n,) int.
        It can also be represented as a tuple of triangle indices and barycentric
        coordinates ((n,) int, (n, 3) float,).
    target_positions : (n, 3) float
        Target positions assigned to source landmarks
    steps : Core parameters of the algorithm
        Iterable of iterables (wc, wi, ws, wl, wn).
        wc is the correspondence term (strength of fitting), wi is the identity term
        (recommended value : 0.001), ws is smoothness term, wl weights the landmark
        importance and wn the normal importance.
    distance_threshold : float
        Distance threshold to account for a vertex match or not.
    return_records : bool
        If True, also returns all the intermediate results. It can help debugging
        and tune the parameters to match a specific case.
    use_faces : bool
        If True and if target geometry has faces, use proximity.closest_point to find
        matching points. Else use scipy's cKDTree object.
    use_vertex_normals : bool
        If True and if target geometry has faces, interpolate the normals of the target
        geometry matching points.
        Else use face normals or estimated normals if target geometry has no faces.
    neighbors_count : int
        number of neighbors used for normal estimation. Only used if target geometry has
        no faces or if use_faces is False.
    face_pairs_type : str 'vertex' or 'edge'
        Method to determine face pairs used in the smoothness cost. 'vertex' yields
        smoother results.


    Returns
    ----------
    result : (n, 3) float or List[(n, 3) float]
        The vertices positions of source_mesh such that it is registered non-rigidly
        onto the target geometry.
        If return_records is True, it returns the list of the vertex positions at each
        iteration.
    """

    def _construct_transform_matrix(faces, Vinv, size):
        # Utility function for constructing the per-frame transforms
        _construct_transform_matrix._row = np.array([0, 1, 2] * 4)
        nV = len(Vinv)
        rows = np.tile(_construct_transform_matrix._row, nV) + 3 * np.repeat(np.arange(nV), 12)
        cols = np.repeat(faces.flat, 3)
        minus_inv_sum = -Vinv.sum(axis=1)
        Vinv_flat = Vinv.reshape(nV, 9)
        data = np.concatenate((minus_inv_sum, Vinv_flat), axis=-1).flatten()
        return sparse.coo_matrix((data, (rows, cols)), shape=(3 * nV, size), dtype=float)

    def _build_tetrahedrons(mesh):
        # UUtility function for constructing the frames
        v4_vec = mesh.face_normals
        v1 = mesh.triangles[:, 0]
        v2 = mesh.triangles[:, 1]
        v3 = mesh.triangles[:, 2]
        v4 = v1 + v4_vec
        vertices = np.concatenate((mesh.vertices, v4))
        nV, nT = len(mesh.vertices), len(mesh.faces)
        v4_indices = np.arange(nV, nV + nT)[:, None]
        tetrahedrons = np.concatenate((mesh.faces, v4_indices), axis=-1)
        frames = np.concatenate(((v2 - v1)[..., None], (v3 - v1)[..., None], v4_vec[..., None]), axis=-1)
        return vertices, tetrahedrons, frames

    def _construct_identity_cost(vtet, tet, Vinv):
        # Utility function for constructing the identity cost
        AEi = _construct_transform_matrix(
            tet,
            Vinv,
            len(vtet),
        ).tocsr()
        Bi = np.tile(np.identity(3, dtype=float), (len(tet), 1))
        return AEi, Bi

    def _construct_smoothness_cost(vtet, tet, Vinv, face_pairs):
        # Utility function for constructing the smoothness (stiffness) cost
        AEs_r = _construct_transform_matrix(tet[face_pairs[:, 0]], Vinv[face_pairs[:, 0]], len(vtet)).tocsr()
        AEs_l = _construct_transform_matrix(tet[face_pairs[:, 1]], Vinv[face_pairs[:, 1]], len(vtet)).tocsr()
        AEs = (AEs_r - AEs_l).tocsc()
        AEs.eliminate_zeros()
        Bs = np.zeros((len(face_pairs) * 3, 3))
        return AEs, Bs

    def _construct_landmark_cost(vtet, source_mesh, source_landmarks):
        # Utility function for constructing the landmark cost
        if source_landmarks is None:
            return None, np.ones(len(source_mesh.vertices), dtype=bool)
        if isinstance(source_landmarks, tuple):
            # If the input source landmarks are in barycentric form
            source_landmarks_tids, source_landmarks_barys = source_landmarks
            source_landmarks_vids = source_mesh.faces[source_landmarks_tids]
            nL, nVT = len(source_landmarks_tids), len(vtet)

            rows = np.repeat(np.arange(nL), 3)
            cols = source_landmarks_vids.flat
            data = source_landmarks_barys.flat

            AEl = sparse.coo_matrix((data, (rows, cols)), shape=(nL, nVT))
            marker_vids = source_landmarks_vids[source_landmarks_barys > np.finfo(np.float16).eps]
            non_markers_mask = np.ones(len(source_mesh.vertices), dtype=bool)
            non_markers_mask[marker_vids] = False
        else:
            # Else if they are in vertex index form
            nL, nVT = len(source_landmarks), len(vtet)
            rows = np.arange(nL)
            cols = source_landmarks.flat
            data = np.ones(nL)
            AEl = sparse.coo_matrix((data, (rows, cols)), shape=(nL, nVT))
            non_markers_mask = np.ones(len(source_mesh.vertices), dtype=bool)
            non_markers_mask[source_landmarks] = False
        return AEl, non_markers_mask

    def _construct_correspondence_cost(points, non_markers_mask, size):
        # Utility function for constructing the correspondence cost
        AEc = sparse.identity(size, dtype=float, format="csc")[: len(non_markers_mask)]
        AEc = AEc[non_markers_mask]
        Bc = points[non_markers_mask]
        return AEc, Bc

    def _compute_vertex_normals(vertices, faces):
        # Utility function for computing source vertex normals
        mesh_triangles = vertices[faces]
        mesh_triangles_cross = cross(mesh_triangles)
        mesh_face_normals = normals(triangles=mesh_triangles, crosses=mesh_triangles_cross)[0]
        mesh_face_angles = angles(mesh_triangles)
        mesh_normals = weighted_vertex_normals(
            vertex_count=nV, faces=faces, face_normals=mesh_face_normals, face_angles=mesh_face_angles
        )
        return mesh_normals

    # First, normalize the source and target to [-1, 1]^3
    (target_geometry, target_positions, centroid, scale) = _normalize_by_source(
        source_mesh, target_geometry, target_positions
    )
    nV = len(source_mesh.vertices)
    use_landmarks = source_landmarks is not None and target_positions is not None

    if steps is None:
        steps = [
            # [wc, wi, ws, wl, wn],
            [1, 0.001, 1.0, 1000, 0],
            [1, 0.001, 1.0, 1000, 0],
            [10, 0.001, 1.0, 1000, 0],
            [100, 0.001, 1.0, 1000, 0],
        ]

    source_vtet, source_tet, V = _build_tetrahedrons(source_mesh)
    Vinv = np.linalg.inv(V)

    # List of (n, 2) faces index which share a vertex
    if face_pairs_type == "vertex":
        face_pairs = source_mesh.face_neighborhood
    else:
        face_pairs = source_mesh.face_adjacency

    # Construct the cost matrices
    # Identity cost (Eq. 12)
    AEi, Bi = _construct_identity_cost(source_vtet, source_tet, Vinv)
    # Smoothness cost (Eq. 11)
    AEs, Bs = _construct_smoothness_cost(source_vtet, source_tet, Vinv, face_pairs)
    # Landmark cost (Eq. 13)
    AEl, non_markers_mask = _construct_landmark_cost(source_vtet, source_mesh, source_landmarks)

    transformed_vertices = source_vtet.copy()
    if return_records:
        records = [transformed_vertices[:nV]]

    # Main loop
    for i, (wc, wi, ws, wl, wn) in enumerate(steps):

        Astack = [AEi * wi, AEs * ws]
        Bstack = [Bi * wi, Bs * ws]

        if use_landmarks:
            Astack.append(AEl * wl)
            Bstack.append(target_positions * wl)

        if (i > 0 or not use_landmarks) and wc > 0:
            # Query the nearest points
            qres = _from_mesh(
                target_geometry,
                transformed_vertices[:nV],
                from_vertices_only=not use_faces,
                return_normals=wn > 0,
                return_interpolated_normals=(use_vertex_normals and wn > 0),
                neighbors_count=neighbors_count,
            )

            # Correspondence cost (Eq. 13)
            AEc, Bc = _construct_correspondence_cost(qres["nearest"], non_markers_mask, len(source_vtet))
            vertices_weight = np.ones(nV)
            vertices_weight[qres["distances"] > distance_threshold] = 0
            if wn > 0 or "normals" in qres:
                target_normals = qres["normals"]
                if use_vertex_normals and "interpolated_normals" in qres:
                    target_normals = qres["interpolated_normals"]
                # Normal weighting : multiplying weights by cosines^wn
                source_normals = _compute_vertex_normals(transformed_vertices, source_mesh.faces)
                dot = util.diagonal_dot(source_normals, target_normals)
                # Normal orientation is only known for meshes as target
                dot = np.clip(dot, 0, 1) if use_faces else np.abs(dot)
                vertices_weight = vertices_weight * dot**wn

            # Account for vertices' weight
            AEc.data *= vertices_weight[non_markers_mask][AEc.indices]
            Bc *= vertices_weight[non_markers_mask, None]

            Astack.append(AEc * wc)
            Bstack.append(Bc * wc)

        # Now solve Eq. 14 ...
        A = sparse.vstack(Astack, format="csc")
        A.eliminate_zeros()
        b = np.concatenate(Bstack)

        LU = sparse.linalg.splu((A.T * A).tocsc())
        transformed_vertices = LU.solve(A.T * b)
        # done !

        if return_records:
            records.append(transformed_vertices[:nV])

    if return_records:
        result = records
    else:
        result = transformed_vertices[:nV]

    result = _denormalize_by_source(source_mesh, target_geometry, target_positions, result, centroid, scale)
    return result
