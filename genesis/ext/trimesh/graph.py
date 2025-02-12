"""
graph.py
-------------

Deal with graph operations. Primarily deal with graphs in (n, 2)
edge list form, and abstract the backend graph library being used.

Currently uses networkx or scipy.sparse.csgraph backend.
"""

import numpy as np
import collections

from . import util
from . import grouping
from . import exceptions

from .constants import log, tol
from .geometry import faces_to_edges

try:
    from scipy.sparse import csgraph, coo_matrix
except BaseException as E:
    # re-raise exception when used
    csgraph = exceptions.ExceptionWrapper(E)
    coo_matrix = exceptions.ExceptionWrapper(E)

try:
    import networkx as nx
except BaseException as E:
    # create a dummy module which will raise the ImportError
    # or other exception only when someone tries to use networkx
    nx = exceptions.ExceptionWrapper(E)


def face_adjacency(faces=None, mesh=None, return_edges=False):
    """
    Returns an (n, 2) list of face indices.
    Each pair of faces in the list shares an edge, making them adjacent.


    Parameters
    -----------
    faces : (n, 3) int, or None
        Vertex indices representing triangles
    mesh : Trimesh object
        If passed will used cached edges
        instead of generating from faces
    return_edges : bool
        Return the edges shared by adjacent faces

    Returns
    ----------
    adjacency : (m, 2) int
        Indexes of faces that are adjacent
    edges: (m, 2) int
        Only returned if return_edges is True
        Indexes of vertices which make up the
        edges shared by the adjacent faces

    Examples
    ----------
    This is useful for lots of things such as finding
    face- connected components:
    ```python
    >>> graph = nx.Graph()
    >>> graph.add_edges_from(mesh.face_adjacency)
    >>> groups = nx.connected_components(graph_connected)
    ```
    """

    if mesh is None:
        # first generate the list of edges for the current faces
        # also return the index for which face the edge is from
        edges, edges_face = faces_to_edges(faces, return_index=True)
        # make sure edge rows are sorted
        edges.sort(axis=1)
    else:
        # if passed a mesh, used the cached values
        edges = mesh.edges_sorted
        edges_face = mesh.edges_face

    # this will return the indices for duplicate edges
    # every edge appears twice in a well constructed mesh
    # so for every row in edge_idx:
    # edges[edge_idx[*][0]] == edges[edge_idx[*][1]]
    # in this call to group rows we discard edges which
    # don't occur twice
    edge_groups = grouping.group_rows(edges, require_count=2)

    if len(edge_groups) == 0:
        log.debug("No adjacent faces detected! Did you merge vertices?")

    # the pairs of all adjacent faces
    # so for every row in face_idx, self.faces[face_idx[*][0]] and
    # self.faces[face_idx[*][1]] will share an edge
    adjacency = edges_face[edge_groups]

    # degenerate faces may appear in adjacency as the same value
    nondegenerate = adjacency[:, 0] != adjacency[:, 1]
    adjacency = adjacency[nondegenerate]

    # sort pairs in-place so we can search for indexes with ordered pairs
    adjacency.sort(axis=1)

    if return_edges:
        adjacency_edges = edges[edge_groups[:, 0][nondegenerate]]
        assert len(adjacency_edges) == len(adjacency)
        return adjacency, adjacency_edges
    return adjacency


def face_neighborhood(mesh):
    """
    Find faces that share a vertex i.e. 'neighbors' faces.
    Relies on the fact that an adjacency matrix at a power p
    contains the number of paths of length p connecting two nodes.
    Here we take the bipartite graph from mesh.faces_sparse to the power 2.
    The non-zeros are the faces connected by one vertex.

    Returns
    ----------
    neighborhood : (n, 2) int
        Pairs of faces which share a vertex
    """

    VT = mesh.faces_sparse
    TT = VT.T * VT
    TT.setdiag(0)
    TT.eliminate_zeros()
    TT = TT.tocoo()
    neighborhood = np.concatenate((TT.row[:, None], TT.col[:, None]), axis=-1)
    return neighborhood


def face_adjacency_unshared(mesh):
    """
    Return the vertex index of the two vertices not in the shared
    edge between two adjacent faces

    Parameters
    ----------
    mesh : Trimesh object
      Input mesh

    Returns
    -----------
    vid_unshared : (len(mesh.face_adjacency), 2) int
      Indexes of mesh.vertices
      for degenerate faces without exactly
      one unshared vertex per face it will be -1
    """

    # the non- shared vertex index is the same shape
    # as face_adjacency holding vertex indices vs face indices
    vid_unshared = np.zeros_like(mesh.face_adjacency, dtype=np.int64) - 1
    # get the shared edges between adjacent faces
    edges = mesh.face_adjacency_edges

    # loop through the two columns of face adjacency
    for i, fid in enumerate(mesh.face_adjacency.T):
        # faces from the current column of face adjacency
        faces = mesh.faces[fid]
        # should have one True per row of (3,)
        # index of vertex not included in shared edge
        unshared = np.logical_not(
            np.logical_or(faces == edges[:, 0].reshape((-1, 1)), faces == edges[:, 1].reshape((-1, 1)))
        )
        # each row should have exactly one uncontained verted
        row_ok = unshared.sum(axis=1) == 1
        # any degenerate row should be ignored
        unshared[~row_ok, :] = False
        # set the
        vid_unshared[row_ok, i] = faces[unshared]

    return vid_unshared


def face_adjacency_radius(mesh):
    """
    Compute an approximate radius between adjacent faces.

    Parameters
    --------------
    mesh : trimesh.Trimesh

    Returns
    -------------
    radii : (len(self.face_adjacency),) float
      Approximate radius between faces
      Parallel faces will have a value of np.inf
    span :  (len(self.face_adjacency),) float
      Perpendicular projection distance of two
      unshared vertices onto the shared edge
    """

    # solve for the radius of the adjacent faces
    #       distance
    # R = ---------------
    #     2 * sin(theta)
    nonzero = mesh.face_adjacency_angles > np.radians(0.01)
    denominator = np.abs(2.0 * np.sin(mesh.face_adjacency_angles[nonzero]))

    # consider the distance between the non- shared vertices of the
    # face adjacency pair as the key distance
    point_pairs = mesh.vertices[mesh.face_adjacency_unshared]
    vectors = np.diff(point_pairs, axis=1).reshape((-1, 3))

    # the vertex indices of the shared edge for the adjacency pairx
    edges = mesh.face_adjacency_edges
    # unit vector along shared the edge
    edges_vec = util.unitize(np.diff(mesh.vertices[edges], axis=1).reshape((-1, 3)))

    # the vector of the perpendicular projection to the shared edge
    perp = np.subtract(vectors, (util.diagonal_dot(vectors, edges_vec).reshape((-1, 1)) * edges_vec))
    # the length of the perpendicular projection
    span = util.row_norm(perp)

    # complete the values for non- infinite radii
    radii = np.ones(len(mesh.face_adjacency)) * np.inf
    radii[nonzero] = span[nonzero] / denominator

    return radii, span


def vertex_adjacency_graph(mesh):
    """
    Returns a networkx graph representing the vertices and
    their connections in the mesh.

    Parameters
    ----------
    mesh : Trimesh object

    Returns
    ---------
    graph : networkx.Graph
        Graph representing vertices and edges between
        them where vertices are nodes and edges are edges

    Examples
    ----------
    This is useful for getting nearby vertices for a given vertex,
    potentially for some simple smoothing techniques.
    >>> graph = mesh.vertex_adjacency_graph
    >>> graph.neighbors(0)
    > [1, 3, 4]
    """
    g = nx.Graph()
    g.add_edges_from(mesh.edges_unique)
    return g


def shared_edges(faces_a, faces_b):
    """
    Given two sets of faces, find the edges which are in both sets.

    Parameters
    ---------
    faces_a : (n, 3) int
      Array of faces
    faces_b : (m, 3) int
      Array of faces

    Returns
    ---------
    shared : (p, 2) int
      Edges shared between faces
    """
    e_a = np.sort(faces_to_edges(faces_a), axis=1)
    e_b = np.sort(faces_to_edges(faces_b), axis=1)
    shared = grouping.boolean_rows(e_a, e_b, operation=np.intersect1d)
    return shared


def facets(mesh, engine=None):
    """
    Find the list of parallel adjacent faces.

    Parameters
    -----------
    mesh :  trimesh.Trimesh
    engine : str
       Which graph engine to use:
       ('scipy', 'networkx')

    Returns
    ---------
    facets : sequence of (n,) int
        Groups of face indexes of
        parallel adjacent faces.
    """
    # what is the radius of a circle that passes through the perpendicular
    # projection of the vector between the two non- shared vertices
    # onto the shared edge, with the face normal from the two adjacent faces
    radii = mesh.face_adjacency_radius
    # what is the span perpendicular to the shared edge
    span = mesh.face_adjacency_span
    # a very arbitrary formula for declaring two adjacent faces
    # parallel in a way that is hopefully (and anecdotally) robust
    # to numeric error
    # a common failure mode is two faces that are very narrow with a slight
    # angle between them, so here we divide by the perpendicular span
    # to penalize very narrow faces, and then square it just for fun
    parallel = np.ones(len(radii), dtype=bool)
    # if span is zero we know faces are small/parallel
    nonzero = np.abs(span) > tol.zero
    # faces with a radii/span ratio larger than a threshold pass
    parallel[nonzero] = (radii[nonzero] / span[nonzero]) ** 2 > tol.facet_threshold

    # run connected components on the parallel faces to group them
    components = connected_components(
        mesh.face_adjacency[parallel], nodes=np.arange(len(mesh.faces)), min_len=2, engine=engine
    )

    return components


def split(mesh, only_watertight=True, adjacency=None, engine=None, **kwargs):
    """
    Split a mesh into multiple meshes from face
    connectivity.

    If only_watertight is true it will only return
    watertight meshes and will attempt to repair
    single triangle or quad holes.

    Parameters
    ----------
    mesh : trimesh.Trimesh
    only_watertight: bool
      Only return watertight components
    adjacency : (n, 2) int
      Face adjacency to override full mesh
    engine : str or None
      Which graph engine to use

    Returns
    ----------
    meshes : (m,) trimesh.Trimesh
      Results of splitting
    """
    if adjacency is None:
        adjacency = mesh.face_adjacency

    # if only watertight the shortest thing we can split has 3 triangles
    if only_watertight:
        min_len = 4
    else:
        min_len = 1

    components = connected_components(edges=adjacency, nodes=np.arange(len(mesh.faces)), min_len=min_len, engine=engine)
    meshes = mesh.submesh(components, only_watertight=only_watertight, **kwargs)
    return meshes


def connected_components(edges, min_len=1, nodes=None, engine=None):
    """
    Find groups of connected nodes from an edge list.

    Parameters
    -----------
    edges : (n, 2) int
      Edges between nodes
    nodes : (m, ) int or None
      List of nodes that exist
    min_len : int
      Minimum length of a component group to return
    engine :  str or None
      Which graph engine to use (None for automatic):
      (None, 'networkx', 'scipy')


    Returns
    -----------
    components : (n,) sequence of (*,) int
      Nodes which are connected
    """

    def components_networkx():
        """
        Find connected components using networkx
        """
        graph = nx.from_edgelist(edges)
        # make sure every face has a node, so single triangles
        # aren't discarded (as they aren't adjacent to anything)
        if min_len <= 1:
            graph.add_nodes_from(nodes)
        return [list(i) for i in nx.connected_components(graph)]

    def components_csgraph():
        """
        Find connected components using scipy.sparse.csgraph
        """
        # label each node
        labels = connected_component_labels(edges, node_count=node_count)

        # we have to remove results that contain nodes outside
        # of the specified node set and reindex
        contained = np.zeros(node_count, dtype=bool)
        contained[nodes] = True
        index = np.arange(node_count, dtype=np.int64)[contained]
        components = grouping.group(labels[contained], min_len=min_len)
        return [index[c] for c in components]

        return components

    # check input edges
    edges = np.asanyarray(edges, dtype=np.int64)
    # if no nodes were specified just use unique
    if nodes is None:
        nodes = np.unique(edges)

    # exit early if we have no nodes
    if len(nodes) == 0:
        return []
    elif len(edges) == 0:
        if min_len <= 1:
            return np.reshape(nodes, (-1, 1)).tolist()
        else:
            return []

    if not util.is_shape(edges, (-1, 2)):
        raise ValueError("edges must be (n, 2)!")

    # find the maximum index referenced in either nodes or edges
    counts = [0]
    if len(edges) > 0:
        counts.append(edges.max())
    if len(nodes) > 0:
        counts.append(nodes.max())
    node_count = np.max(counts) + 1

    # remove edges that don't have both nodes in the node set
    mask = np.zeros(node_count, dtype=bool)
    mask[nodes] = True
    edges_ok = mask[edges].all(axis=1)
    edges = edges[edges_ok]

    # networkx is pure python and is usually 5-10x slower than scipy
    engines = collections.OrderedDict((("scipy", components_csgraph), ("networkx", components_networkx)))

    # if a graph engine has explicitly been requested use it
    if engine in engines:
        return engines[engine]()

    # otherwise, go through our ordered list of graph engines
    # until we get to one that has actually been installed
    for function in engines.values():
        try:
            return function()
        # will be raised if the library didn't import correctly above
        except BaseException:
            continue
    raise ImportError("no graph engines available!")


def connected_component_labels(edges, node_count=None):
    """
    Label graph nodes from an edge list, using scipy.sparse.csgraph

    Parameters
    -----------
    edges : (n, 2) int
       Edges of a graph
    node_count : int, or None
        The largest node in the graph.

    Returns
    ----------
    labels : (node_count,) int
        Component labels for each node
    """
    matrix = edges_to_coo(edges, node_count)
    body_count, labels = csgraph.connected_components(matrix, directed=False)

    if node_count is not None:
        assert len(labels) == node_count

    return labels


def split_traversal(traversal, edges, edges_hash=None):
    """
    Given a traversal as a list of nodes, split the traversal
    if a sequential index pair is not in the given edges.

    Parameters
    --------------
    edges : (n, 2) int
       Graph edge indexes
    traversal : (m,) int
       Traversal through edges
    edge_hash : (n,)
       Edges sorted on axis=1 and
       passed to grouping.hashable_rows

    Returns
    ---------------
    split : sequence of (p,) int
    """
    traversal = np.asanyarray(traversal, dtype=np.int64)

    # hash edge rows for contains checks
    if edges_hash is None:
        edges_hash = grouping.hashable_rows(np.sort(edges, axis=1))

    # turn the (n,) traversal into (n-1, 2) edges
    trav_edge = np.column_stack((traversal[:-1], traversal[1:]))
    # hash each edge so we can compare to edge set
    trav_hash = grouping.hashable_rows(np.sort(trav_edge, axis=1))
    # check if each edge is contained in edge set
    contained = np.isin(trav_hash, edges_hash)

    # exit early if every edge of traversal exists
    if contained.all():
        # just reshape one traversal
        split = [traversal]
    else:
        # find contiguous groups of contained edges
        blocks = grouping.blocks(contained, min_len=1, only_nonzero=True)

        # turn edges back in to sequence of traversals
        split = [np.append(trav_edge[b][:, 0], trav_edge[b[-1]][1]) for b in blocks]

    # close traversals if necessary
    for i, t in enumerate(split):
        # make sure elements of sequence are numpy arrays
        split[i] = np.asanyarray(split[i], dtype=np.int64)
        # don't close if its a single edge
        if len(t) <= 2:
            continue
        # make sure it's not already closed
        edge = np.sort([t[0], t[-1]])
        if np.ptp(edge) == 0:
            continue
        close = grouping.hashable_rows(edge.reshape((1, 2)))[0]
        # if we need the edge add it
        if close in edges_hash:
            split[i] = np.append(t, t[0]).astype(np.int64)

    return split


def fill_traversals(traversals, edges, edges_hash=None):
    """
    Convert a traversal of a list of edges into a sequence of
    traversals where every pair of consecutive node indexes
    is an edge in a passed edge list

    Parameters
    -------------
    traversals : sequence of (m,) int
       Node indexes of traversals of a graph
    edges : (n, 2) int
       Pairs of connected node indexes
    edges_hash : None, or (n,) int
       Edges sorted along axis 1 then hashed
       using grouping.hashable_rows

    Returns
    --------------
    splits : sequence of (p,) int
       Node indexes of connected traversals
    """
    # make sure edges are correct type
    edges = np.asanyarray(edges, dtype=np.int64)
    # make sure edges are sorted
    edges.sort(axis=1)

    # if there are no traversals just return edges
    if len(traversals) == 0:
        return edges.copy()

    # hash edges for contains checks
    if edges_hash is None:
        edges_hash = grouping.hashable_rows(edges)

    splits = []
    for nodes in traversals:
        # split traversals to remove edges
        # that don't actually exist
        splits.extend(split_traversal(traversal=nodes, edges=edges, edges_hash=edges_hash))
    # turn the split traversals back into (n, 2) edges
    included = util.vstack_empty([np.column_stack((i[:-1], i[1:])) for i in splits])
    if len(included) > 0:
        # sort included edges in place
        included.sort(axis=1)
        # make sure any edges not included in split traversals
        # are just added as a length 2 traversal
        splits.extend(grouping.boolean_rows(edges, included, operation=np.setdiff1d))
    else:
        # no edges were included, so our filled traversal
        # is just the original edges copied over
        splits = edges.copy()

    return splits


def traversals(edges, mode="bfs"):
    """
    Given an edge list generate a sequence of ordered depth
    first search traversals using scipy.csgraph routines.

    Parameters
    ------------
    edges : (n, 2) int
      Undirected edges of a graph
    mode :  str
      Traversal type, 'bfs' or 'dfs'

    Returns
    -----------
    traversals : (m,) sequence of (p,) int
      Ordered DFS or BFS traversals of the graph.
    """
    edges = np.array(edges, dtype=np.int64)
    if len(edges) == 0:
        return []
    elif not util.is_shape(edges, (-1, 2)):
        raise ValueError("edges are not (n, 2)!")

    # pick the traversal method
    mode = str(mode).lower().strip()
    if mode == "bfs":
        func = csgraph.breadth_first_order
    elif mode == "dfs":
        func = csgraph.depth_first_order
    else:
        raise ValueError("traversal mode must be either dfs or bfs")

    # make sure edges are sorted so we can query
    # an ordered pair later
    edges.sort(axis=1)
    # set of nodes to make sure we get every node
    nodes = set(edges.reshape(-1))
    # coo_matrix for csgraph routines
    graph = edges_to_coo(edges)

    # we're going to make a sequence of traversals
    traversals = []

    while len(nodes) > 0:
        # starting at any node
        start = nodes.pop()
        # get an (n,) ordered traversal
        ordered = func(graph, i_start=start, return_predecessors=False, directed=False).astype(np.int64)

        traversals.append(ordered)
        # remove the nodes we've consumed
        nodes.difference_update(ordered)

    return traversals


def edges_to_coo(edges, count=None, data=None):
    """
    Given an edge list, return a boolean scipy.sparse.coo_matrix
    representing the edges in matrix form.

    Parameters
    ------------
    edges : (n, 2) int
      Edges of a graph
    count : int
      The total number of nodes in the graph
      if None: count = edges.max() + 1
    data : (n,) any
      Assign data to each edge, if None will
      be bool True for each specified edge

    Returns
    ------------
    matrix: (count, count) scipy.sparse.coo_matrix
      Sparse COO
    """
    edges = np.asanyarray(edges, dtype=np.int64)
    if not (len(edges) == 0 or util.is_shape(edges, (-1, 2))):
        raise ValueError("edges must be (n, 2)!")

    # if count isn't specified just set it to largest
    # value referenced in edges
    if count is None:
        count = edges.max() + 1
    count = int(count)

    # if no data is specified set every specified edge
    # to True
    if data is None:
        data = np.ones(len(edges), dtype=bool)

    matrix = coo_matrix((data, edges.T), dtype=data.dtype, shape=(count, count))
    return matrix


def neighbors(edges, max_index=None, directed=False):
    """
    Find the neighbors for each node in an edgelist graph.

    TODO : re-write this with sparse matrix operations

    Parameters
    ------------
    edges : (n, 2) int
      Connected nodes
    directed : bool
      If True, only connect edges in one direction

    Returns
    ---------
    neighbors : sequence
      Vertex index corresponds to set of other vertex indices
    """
    neighbors = collections.defaultdict(set)
    if directed:
        [neighbors[edge[0]].add(edge[1]) for edge in edges]
    else:
        [(neighbors[edge[0]].add(edge[1]), neighbors[edge[1]].add(edge[0])) for edge in edges]

    if max_index is None:
        max_index = edges.max() + 1
    array = [list(neighbors[i]) for i in range(max_index)]

    return array


def smoothed(mesh, angle=None, facet_minarea=10):
    """
    Return a non- watertight version of the mesh which
    will render nicely with smooth shading by
    disconnecting faces at sharp angles to each other.

    Parameters
    -----------
    mesh : trimesh.Trimesh
      Source geometry
    angle : float or None
      Angle in radians face pairs with angles
      smaller than this will appear smoothed
    facet_minarea : float or None
      Minimum area fraction to consider
      IE for `facets_minarea=25` only facets larger
      than `mesh.area / 25` will be considered.

    Returns
    ---------
    smooth : trimesh.Trimesh
      Geometry with disconnected face patches
    """
    if angle is None:
        angle = np.radians(20)

    # if the mesh has no adjacent faces return a copy
    if len(mesh.face_adjacency) == 0:
        return mesh.copy()

    # face pairs below angle threshold
    angle_ok = mesh.face_adjacency_angles < angle
    # subset of face adjacency
    adjacency = mesh.face_adjacency[angle_ok]

    # coplanar groups of faces
    facets = []
    nodes = None
    # collect coplanar regions for smoothing
    if facet_minarea is not None:
        areas = mesh.area_faces
        min_area = mesh.area / facet_minarea
        try:
            # we can survive not knowing facets
            # exclude facets with few faces
            facets = [f for f in mesh.facets if areas[f].sum() > min_area]
            if len(facets) > 0:
                # mask for removing adjacency pairs where
                # one of the faces is contained in a facet
                mask = np.ones(len(mesh.faces), dtype=bool)
                mask[np.hstack(facets)] = False
                # apply the mask to adjacency
                adjacency = adjacency[mask[adjacency].all(axis=1)]
                # nodes are no longer every faces
                nodes = np.unique(adjacency)
        except BaseException:
            log.warning("failed to calculate facets", exc_info=True)
    # run connected components on facet adjacency
    components = connected_components(adjacency, min_len=2, nodes=nodes)

    # add back coplanar groups if any exist
    if len(facets) > 0:
        components.extend(facets)

    if len(components) == 0:
        # if no components for some reason
        # just return a copy of the original mesh
        return mesh.copy()

    # add back any faces that were missed
    unique = np.unique(np.hstack(components))
    if len(unique) != len(mesh.faces):
        # things like single loose faces
        # or groups below facet_minlen
        broke = np.setdiff1d(np.arange(len(mesh.faces)), unique)
        components.extend(broke.reshape((-1, 1)))

    # get a submesh as a single appended Trimesh
    smooth = mesh.submesh(components, only_watertight=False, append=True)
    # store face indices from original mesh
    smooth.metadata["original_components"] = components
    # smoothed should have exactly the same number of faces
    if len(smooth.faces) != len(mesh.faces):
        log.warning("face count in smooth wrong!")
    return smooth


def is_watertight(edges, edges_sorted=None):
    """
    Parameters
    -----------
    edges : (n, 2) int
      List of vertex indices
    edges_sorted : (n, 2) int
      Pass vertex indices sorted on axis 1 as a speedup

    Returns
    ---------
    watertight : boolean
      Whether every edge is shared by an even
      number of faces
    winding : boolean
      Whether every shared edge is reversed
    """
    # passing edges_sorted is a speedup only
    if edges_sorted is None:
        edges_sorted = np.sort(edges, axis=1)

    # group sorted edges
    groups = grouping.group_rows(edges_sorted, require_count=2)
    watertight = bool((len(groups) * 2) == len(edges))

    # are opposing edges reversed
    opposing = edges[groups].reshape((-1, 4))[:, 1:3].T
    # wrap the weird numpy bool
    winding = bool(np.equal(*opposing).all())

    return watertight, winding


def graph_to_svg(graph):
    """
    Turn a networkx graph into an SVG string
    using graphviz `dot`.

    Parameters
    ----------
    graph: networkx graph

    Returns
    ---------
    svg: string, pictoral layout in SVG format
    """

    import tempfile
    import subprocess

    with tempfile.NamedTemporaryFile() as dot_file:
        nx.drawing.nx_agraph.write_dot(graph, dot_file.name)
        svg = subprocess.check_output(["dot", dot_file.name, "-Tsvg"])
    return svg


def multigraph_paths(G, source, cutoff=None):
    """
    For a networkx MultiDiGraph, find all paths from a source node
    to leaf nodes. This function returns edge instance numbers
    in addition to nodes, unlike networkx.all_simple_paths.

    Parameters
    ---------------
    G : networkx.MultiDiGraph
      Graph to evaluate
    source : hashable
      Node to start traversal at
    cutoff : int
      Number of nodes to visit
      If None will visit all nodes

    Returns
    ----------
    traversals : (n,) list of [(node, edge instance index), ] paths
      Traversals of the multigraph
    """
    if cutoff is None:
        cutoff = (len(G.edges()) * len(G.nodes())) + 1

    # the path starts at the node specified
    current = [(source, 0)]
    # traversals we need to go back and do
    queue = []
    # completed paths
    traversals = []

    for _ in range(cutoff):
        # paths are stored as (node, instance) so
        # get the node of the last place visited
        current_node = current[-1][0]
        # get all the children of the current node
        child = G[current_node]

        if len(child) == 0:
            # we have no children, so we are at the end of this path
            # save the path as a completed traversal
            traversals.append(current)
            # if there is nothing on the queue, we are done
            if len(queue) == 0:
                break
            # otherwise continue traversing with the next path
            # on the queue
            current = queue.pop()
        else:
            # oh no, we have multiple edges from current -> child
            start = True
            # iterate through child nodes and edge instances
            for node in child.keys():
                for instance in child[node].keys():
                    if start:
                        # if this is the first edge, keep it on the
                        # current traversal and save the others for later
                        current.append((node, instance))
                        start = False
                    else:
                        # this child has multiple instances
                        # so we will need to traverse them multiple times
                        # we appended a node to current, so only take the
                        # first n-1 visits
                        queue.append(current[:-1] + [(node, instance)])
    return traversals


def multigraph_collect(G, traversal, attrib=None):
    """
    Given a MultiDiGraph traversal, collect attributes along it.

    Parameters
    -------------
    G:          networkx.MultiDiGraph
    traversal:  (n) list of (node, instance) tuples
    attrib:     dict key, name to collect. If None, will return all

    Returns
    -------------
    collected: (len(traversal) - 1) list of attributes
    """

    collected = []
    for u, v in util.pairwise(traversal):
        attribs = G[u[0]][v[0]][v[1]]
        if attrib is None:
            collected.append(attribs)
        else:
            collected.append(attribs[attrib])
    return collected
