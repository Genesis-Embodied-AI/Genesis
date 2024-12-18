"""
poses.py
-----------

Find stable orientations of meshes.
"""

import numpy as np

from .triangles import points_to_barycentric

try:
    import networkx as nx
except BaseException as E:
    # create a dummy module which will raise the ImportError
    # or other exception only when someone tries to use networkx
    from .exceptions import ExceptionWrapper

    nx = ExceptionWrapper(E)


def compute_stable_poses(mesh, center_mass=None, sigma=0.0, n_samples=1, threshold=0.0):
    """
    Computes stable orientations of a mesh and their quasi-static probabilities.

    This method samples the location of the center of mass from a multivariate
    gaussian with the mean at the center of mass, and a covariance
    equal to and identity matrix times sigma, over n_samples.

    For each sample, it computes the stable resting poses of the mesh on a
    a planar workspace and evaluates the probabilities of landing in
    each pose if the object is dropped onto the table randomly.

    This method returns the 4x4 homogeneous transform matrices that place
    the shape against the planar surface with the z-axis pointing upwards
    and a list of the probabilities for each pose.

    The transforms and probabilties that are returned are sorted, with the
    most probable pose first.

    Parameters
    ----------
    mesh : trimesh.Trimesh
      The target mesh
    com : (3,) float
      Rhe object center of mass. If None, this method
      assumes uniform density and watertightness and
      computes a center of mass explicitly
    sigma : float
      Rhe covariance for the multivariate gaussian used
      to sample center of mass locations
    n_samples : int
      The number of samples of the center of mass location
    threshold : float
      The probability value at which to threshold
      returned stable poses

    Returns
    -------
    transforms : (n, 4, 4) float
      The homogeneous matrices that transform the
      object to rest in a stable pose, with the
      new z-axis pointing upwards from the table
      and the object just touching the table.
    probs : (n,) float
      Probability in (0, 1) for each pose
    """

    # save convex hull mesh to avoid a cache check
    cvh = mesh.convex_hull

    if center_mass is None:
        center_mass = mesh.center_mass

    # Sample center of mass, rejecting points outside of conv hull
    sample_coms = []
    while len(sample_coms) < n_samples:
        remaining = n_samples - len(sample_coms)
        coms = np.random.multivariate_normal(center_mass, sigma * np.eye(3), remaining)
        for c in coms:
            dots = np.einsum("ij,ij->i", c - cvh.triangles_center, cvh.face_normals)
            if np.all(dots < 0):
                sample_coms.append(c)

    norms_to_probs = {}  # Map from normal to probabilities

    # For each sample, compute the stable poses
    for sample_com in sample_coms:

        # Create toppling digraph
        dg = _create_topple_graph(cvh, sample_com)

        # Propagate probabilities to sink nodes with a breadth-first traversal
        nodes = [n for n in dg.nodes() if dg.in_degree(n) == 0]
        n_iters = 0
        while len(nodes) > 0 and n_iters <= len(mesh.faces):
            new_nodes = []
            for node in nodes:
                if dg.out_degree(node) == 0:
                    continue
                successor = next(iter(dg.successors(node)))
                dg.nodes[successor]["prob"] += dg.nodes[node]["prob"]
                dg.nodes[node]["prob"] = 0.0
                new_nodes.append(successor)
            nodes = new_nodes
            n_iters += 1

        # Collect stable poses
        for node in dg.nodes():
            if dg.nodes[node]["prob"] > 0.0:
                normal = cvh.face_normals[node]
                prob = dg.nodes[node]["prob"]
                key = tuple(np.around(normal, decimals=3))
                if key in norms_to_probs:
                    norms_to_probs[key]["prob"] += 1.0 / n_samples * prob
                else:
                    norms_to_probs[key] = {"prob": 1.0 / n_samples * prob, "normal": normal}

    transforms = []
    probs = []

    # Filter stable poses
    for key in norms_to_probs:
        prob = norms_to_probs[key]["prob"]
        if prob > threshold:
            tf = np.eye(4)

            # Compute a rotation matrix for this stable pose
            z = -1.0 * norms_to_probs[key]["normal"]
            x = np.array([-z[1], z[0], 0])
            if np.linalg.norm(x) == 0.0:
                x = np.array([1, 0, 0])
            else:
                x = x / np.linalg.norm(x)
            y = np.cross(z, x)
            y = y / np.linalg.norm(y)
            tf[:3, :3] = np.array([x, y, z])

            # Compute the necessary translation for this stable pose
            m = cvh.copy()
            m.apply_transform(tf)
            z = -m.bounds[0][2]
            tf[:3, 3] = np.array([0, 0, z])

            transforms.append(tf)
            probs.append(prob)

    # Sort the results
    transforms = np.array(transforms)
    probs = np.array(probs)
    inds = np.argsort(-probs)

    return transforms[inds], probs[inds]


def _orient3dfast(plane, pd):
    """
    Performs a fast 3D orientation test.

    Parameters
    ----------
    plane: (3,3) float, three points in space that define a plane
    pd:    (3,)  float, a single point

    Returns
    -------
    result: float, if greater than zero then pd is above the plane through
                   the given three points, if less than zero then pd is below
                   the given plane, and if equal to zero then pd is on the
                   given plane.
    """
    pa, pb, pc = plane
    adx = pa[0] - pd[0]
    bdx = pb[0] - pd[0]
    cdx = pc[0] - pd[0]
    ady = pa[1] - pd[1]
    bdy = pb[1] - pd[1]
    cdy = pc[1] - pd[1]
    adz = pa[2] - pd[2]
    bdz = pb[2] - pd[2]
    cdz = pc[2] - pd[2]

    return adx * (bdy * cdz - bdz * cdy) + bdx * (cdy * adz - cdz * ady) + cdx * (ady * bdz - adz * bdy)


def _compute_static_prob(tri, com):
    """
    For an object with the given center of mass, compute
    the probability that the given tri would be the first to hit the
    ground if the object were dropped with a pose chosen uniformly at random.

    Parameters
    ----------
    tri: (3,3) float, the vertices of a triangle
    cm:  (3,) float, the center of mass of the object

    Returns
    -------
    prob: float, the probability in [0,1] for the given triangle
    """
    sv = [(v - com) / np.linalg.norm(v - com) for v in tri]

    # Use L'Huilier's Formula to compute spherical area
    a = np.arccos(min(1, max(-1, np.dot(sv[0], sv[1]))))
    b = np.arccos(min(1, max(-1, np.dot(sv[1], sv[2]))))
    c = np.arccos(min(1, max(-1, np.dot(sv[2], sv[0]))))
    s = (a + b + c) / 2.0

    # Prevents weirdness with arctan
    try:
        return (
            1.0
            / np.pi
            * np.arctan(np.sqrt(np.tan(s / 2) * np.tan((s - a) / 2) * np.tan((s - b) / 2) * np.tan((s - c) / 2)))
        )
    except BaseException:
        s = s + 1e-8
        return (
            1.0
            / np.pi
            * np.arctan(np.sqrt(np.tan(s / 2) * np.tan((s - a) / 2) * np.tan((s - b) / 2) * np.tan((s - c) / 2)))
        )


def _create_topple_graph(cvh_mesh, com):
    """
    Constructs a toppling digraph for the given convex hull mesh and
    center of mass.

    Each node n_i in the digraph corresponds to a face f_i of the mesh and is
    labelled with the probability that the mesh will land on f_i if dropped
    randomly. Not all faces are stable, and node n_i has a directed edge to
    node n_j if the object will quasi-statically topple from f_i to f_j if it
    lands on f_i initially.

    This computation is described in detail in
    http://goldberg.berkeley.edu/pubs/eps.pdf.

    Parameters
    ----------
    cvh_mesh : trimesh.Trimesh
      Rhe convex hull of the target shape
    com : (3,) float
      The 3D location of the target shape's center of mass

    Returns
    -------
    graph : networkx.DiGraph
      Graph representing static probabilities and toppling
      order for the convex hull
    """
    adj_graph = nx.Graph()
    topple_graph = nx.DiGraph()

    # Create face adjacency graph
    face_pairs = cvh_mesh.face_adjacency
    edges = cvh_mesh.face_adjacency_edges

    graph_edges = []
    for fp, e in zip(face_pairs, edges):
        verts = cvh_mesh.vertices[e]
        graph_edges.append([fp[0], fp[1], {"verts": verts}])

    adj_graph.add_edges_from(graph_edges)

    # Compute static probabilities of landing on each face
    for i, tri in enumerate(cvh_mesh.triangles):
        prob = _compute_static_prob(tri, com)
        topple_graph.add_node(i, prob=prob)

    # Compute COM projections onto planes of each triangle in cvh_mesh
    proj_dists = np.einsum("ij,ij->i", cvh_mesh.face_normals, com - cvh_mesh.triangles[:, 0])
    proj_coms = com - np.einsum("i,ij->ij", proj_dists, cvh_mesh.face_normals)
    barys = points_to_barycentric(cvh_mesh.triangles, proj_coms)
    unstable_face_indices = np.where(np.any(barys < 0, axis=1))[0]

    # For each unstable face, compute the face it topples to
    for fi in unstable_face_indices:
        proj_com = proj_coms[fi]
        centroid = cvh_mesh.triangles_center[fi]
        norm = cvh_mesh.face_normals[fi]

        for tfi in adj_graph[fi]:
            v1, v2 = adj_graph[fi][tfi]["verts"]
            if np.dot(np.cross(v1 - centroid, v2 - centroid), norm) < 0:
                tmp = v2
                v2 = v1
                v1 = tmp
            plane1 = [centroid, v1, v1 + norm]
            plane2 = [centroid, v2 + norm, v2]
            if _orient3dfast(plane1, proj_com) >= 0 and _orient3dfast(plane2, proj_com) >= 0:
                break

        topple_graph.add_edge(fi, tfi)

    return topple_graph
