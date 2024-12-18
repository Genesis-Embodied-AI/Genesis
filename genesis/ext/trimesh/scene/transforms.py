import numpy as np
import collections

from copy import deepcopy

from .. import util
from .. import caching

from ..transformations import rotation_matrix, quaternion_matrix, fix_rigid
from ..caching import hash_fast

# we compare to identity a lot
_identity = np.eye(4)
_identity.flags["WRITEABLE"] = False


class SceneGraph(object):
    """
    Hold data about positions and instances of geometry
    in a scene. This includes a forest (i.e. multi-root tree)
    of transforms and information on which node is the base
    frame, and which geometries are affiliated with which
    nodes.
    """

    def __init__(self, base_frame="world", repair_rigid=1e-5):
        """
        Create a scene graph, holding homogeneous transformation
        matrices and instance information about geometry.

        Parameters
        -----------
        base_frame : any
          The root node transforms will be positioned from.
        repair_rigid : None or float
          If a float will attempt to repair rotation matrices
          where `M @ M.T` differs from an identity matrix by
          more than floating point zero but less than this value.
          This can happen in a deep tree with a lot of matrix
          multiplies.
        """
        # a graph structure, subclass of networkx DiGraph
        self.transforms = EnforcedForest()
        # hashable, the base or root frame
        self.base_frame = base_frame
        # if passed as a float try to repair rigid transforms
        # that have accumulated floating point error
        self.repair_rigid = repair_rigid
        # cache transformation matrices keyed with tuples
        self._cache = caching.Cache(self.__hash__)

    def update(self, frame_to, frame_from=None, **kwargs):
        """
        Update a transform in the tree.

        Parameters
        ------------
        frame_from : hashable object
          Usually a string (eg 'world').
          If left as None it will be set to self.base_frame
        frame_to :  hashable object
          Usually a string (eg 'mesh_0')
        matrix : (4,4) float
          Homogeneous transformation matrix
        quaternion :  (4,) float
          Quaternion ordered [w, x, y, z]
        axis : (3,) float
          Axis of rotation
        angle :  float
          Angle of rotation, in radians
        translation : (3,) float
          Distance to translate
        geometry : hashable
          Geometry object name, e.g. 'mesh_0'
        metadata: dictionary
          Optional metadata attached to the new frame
          (exports to glTF node 'extras').
        """
        # if no frame specified, use base frame
        if frame_from is None:
            frame_from = self.base_frame

        # pass through
        attr = {k: v for k, v in kwargs.items() if k in {"geometry", "metadata"}}
        # convert various kwargs to a single matrix
        attr["matrix"] = kwargs_to_matrix(**kwargs)

        # add the edges for the transforms
        # wi ll return if it changed anything
        self.transforms.add_edge(frame_from, frame_to, **attr)

        # set the node attribute with the geometry information
        if "geometry" in kwargs:
            self.transforms.node_data[frame_to]["geometry"] = kwargs["geometry"]

    def get(self, frame_to, frame_from=None):
        """
        Get the transform from one frame to another.

        Parameters
        ------------
        frame_to : hashable
          Node name, usually a string (eg 'mesh_0')
        frame_from : hashable
          Node name, usually a string (eg 'world').
          If None it will be set to self.base_frame

        Returns
        ----------
        transform : (4, 4) float
          Homogeneous transformation matrix

        Raises
        -----------
        ValueError
          If the frames aren't connected.
        """

        # use base frame if not specified
        if frame_from is None:
            frame_from = self.base_frame

        # look up transform to see if we have it already
        key = (frame_from, frame_to)
        if key in self._cache:
            return self._cache[key]

        # get the geometry at the final node if any
        geometry = self.transforms.node_data[frame_to].get("geometry")

        # get a local reference to edge data
        data = self.transforms.edge_data

        if frame_from == frame_to:
            # if we're going from ourself return identity
            matrix = _identity
        elif key in data:
            # if the path is just an edge return early
            matrix = data[key]["matrix"]
        else:
            # we have a 3+ node path
            # get the path from the forest always going from
            # parent -> child -> child
            path = self.transforms.shortest_path(frame_from, frame_to)
            # the path should always start with `frame_from`
            assert path[0] == frame_from
            # and end with the `frame_to` node
            assert path[-1] == frame_to

            # loop through pairs of the path
            matrices = []
            for u, v in zip(path[:-1], path[1:]):
                forward = data.get((u, v))
                if forward is not None:
                    if "matrix" in forward:
                        # append the matrix from u to v
                        matrices.append(forward["matrix"])
                    continue
                # since forwards didn't exist backward must
                # exist otherwise this is a disconnected path
                # and we should raise an error anyway
                backward = data[(v, u)]
                if "matrix" in backward:
                    # append the inverted backwards matrix
                    matrices.append(np.linalg.inv(backward["matrix"]))
            # filter out any identity matrices
            matrices = [m for m in matrices if np.abs((m - _identity)).max() > 1e-8]
            if len(matrices) == 0:
                matrix = _identity
            elif len(matrices) == 1:
                matrix = matrices[0]
            else:
                # multiply matrices into single transform
                matrix = util.multi_dot(matrices)

        # if instructed to repair rigid transforms do it here
        if self.repair_rigid is not None:
            matrix = fix_rigid(matrix, max_deviance=self.repair_rigid)

        # matrix being edited in-place leads to subtle bugs
        matrix.flags["WRITEABLE"] = False

        # store the result
        self._cache[key] = (matrix, geometry)

        return matrix, geometry

    def __hash__(self):
        return self.transforms.__hash__()

    def copy(self):
        """
        Return a copy of the current TransformForest.

        Returns
        ------------
        copied : TransformForest
          Copy of current object.
        """
        # create a copy without transferring cache
        copied = SceneGraph()
        copied.base_frame = deepcopy(self.base_frame)
        copied.transforms = deepcopy(self.transforms)
        return copied

    def to_flattened(self):
        """
        Export the current transform graph with all
        transforms baked into world->instance.

        Returns
        ---------
        flat : dict
          Keyed {node : {transform, geometry}
        """
        flat = {}
        base_frame = self.base_frame
        for node in self.nodes:
            if node == base_frame:
                continue
            # get the matrix and geometry name
            matrix, geometry = self.get(frame_to=node, frame_from=base_frame)
            # store matrix as list rather than numpy array
            flat[node] = {"transform": matrix.tolist(), "geometry": geometry}

        return flat

    def to_gltf(self, scene, mesh_index=None):
        """
        Export a transforms as the 'nodes' section of the
        GLTF header dict.

        Parameters
        ------------
        scene : trimesh.Scene
          Scene with geometry.
        mesh_index : dict or None
          Mapping { key in scene.geometry : int }

        Returns
        --------
        gltf : dict
          With 'nodes' referencing a list of dicts
        """

        if mesh_index is None:
            # geometry is an OrderedDict
            # map mesh name to index: {geometry key : index}
            mesh_index = {name: i for i, name in enumerate(scene.geometry.keys())}

        # get graph information into local scope before loop
        graph = self.transforms
        # get the stored node data
        node_data = graph.node_data
        edge_data = graph.edge_data
        base_frame = self.base_frame

        # list of dict, in gltf format
        # start with base frame as first node index
        result = [{"name": base_frame}]
        # {node name : node index in gltf}
        lookup = {base_frame: 0}

        # collect the nodes in order
        for node in node_data.keys():
            if node == base_frame:
                continue
            # assign the index to the node-name lookup
            lookup[node] = len(result)
            # populate a result at the correct index
            result.append({"name": node})

        # get generated properties outside of loop
        # does the scene have a defined camera to export
        has_camera = scene.has_camera
        children = graph.children

        extensions_used = set()

        # then iterate through to collect data
        for info in result:
            # name of the scene node
            node = info["name"]

            # get the original node names for children
            childs = children.get(node, [])
            if len(childs) > 0:
                info["children"] = [lookup[k] for k in childs]

            # if we have a mesh store by index
            if "geometry" in node_data[node]:
                mesh_key = node_data[node]["geometry"]
                if mesh_key in mesh_index:
                    info["mesh"] = mesh_index[mesh_key]
            # check to see if we have camera node
            if has_camera and node == scene.camera.name:
                info["camera"] = 0

            if node != base_frame:
                parent = graph.parents[node]
                node_edge = edge_data[(parent, node)]

                # get the matrix from this edge
                matrix = node_edge["matrix"]
                # only include if it's not an identify matrix
                if not util.allclose(matrix, _identity):
                    info["matrix"] = matrix.T.reshape(-1).tolist()

                # if an extra was stored on this edge
                extras = node_edge.get("metadata")
                if extras:
                    extras = extras.copy()

                    # if extensionss were stored on this edge
                    extensions = extras.pop("gltf_extensions", None)
                    if isinstance(extensions, dict):
                        info["extensions"] = extensions
                        extensions_used = extensions_used.union(set(extensions.keys()))

                    # convert any numpy arrays to lists
                    extras.update({k: v.tolist() for k, v in extras.items() if hasattr(v, "tolist")})
                    info["extras"] = extras

        gltf = {"nodes": result}
        if len(extensions_used) > 0:
            gltf["extensionsUsed"] = list(extensions_used)
        return gltf

    def to_edgelist(self):
        """
        Export the current transforms as a list of
        edge tuples, with each tuple having the format:
        (node_a, node_b, {metadata})

        Returns
        ---------
        edgelist : (n,) list
          Of edge tuples
        """
        # save local reference to node_data
        nodes = self.transforms.node_data
        # save cleaned edges
        export = []
        # loop through (node, node, edge attributes)
        for edge, attr in self.transforms.edge_data.items():
            # node indexes from edge
            a, b = edge
            # geometry is a node property but save it to the
            # edge so we don't need two dictionaries
            b_attr = nodes[b]
            # make sure we're not stomping on original
            attr_new = attr.copy()
            # apply node geometry to edge attributes
            if "geometry" in b_attr:
                attr_new["geometry"] = b_attr["geometry"]
            # convert any numpy arrays to regular lists
            attr_new.update({k: v.tolist() for k, v in attr_new.items() if hasattr(v, "tolist")})
            export.append([a, b, attr_new])
        return export

    def from_edgelist(self, edges, strict=True):
        """
        Load transform data from an edge list into the current
        scene graph.

        Parameters
        -------------
        edgelist : (n,) tuples
          Keyed (node_a, node_b, {key: value})
        strict : bool
          If True raise a ValueError when a
          malformed edge is passed in a tuple.
        """

        # loop through each edge
        for edge in edges:
            # edge contains attributes
            if len(edge) == 3:
                self.update(edge[1], edge[0], **edge[2])
            # edge just contains nodes
            elif len(edge) == 2:
                self.update(edge[1], edge[0])
            # edge is broken
            elif strict:
                raise ValueError("edge incorrect shape: %s", str(edge))

    def to_networkx(self):
        """
        Return a `networkx` copy of this graph.

        Returns
        ----------
        graph : networkx.DiGraph
          Directed graph.
        """
        import networkx

        return networkx.from_edgelist(self.to_edgelist(), create_using=networkx.DiGraph)

    def show(self, **kwargs):
        """
        Plot the scene graph using `networkx.draw_networkx`
        which uses matplotlib to display the graph.

        Parameters
        -----------
        kwargs : dict
          Passed to `networkx.draw_networkx`
        """
        import networkx
        import matplotlib.pyplot as plt

        # default kwargs will only be set if not
        # passed explicitly to the show command
        defaults = {"with_labels": True}
        kwargs.update(**{k: v for k, v in defaults.items() if k not in kwargs})
        networkx.draw_networkx(G=self.to_networkx(), **kwargs)

        plt.show()

    def load(self, edgelist):
        """
        Load transform data from an edge list into the current
        scene graph.

        Parameters
        -------------
        edgelist : (n,) tuples
          Structured (node_a, node_b, {key: value})
        """
        self.from_edgelist(edgelist, strict=True)

    @caching.cache_decorator
    def nodes(self):
        """
        A list of every node in the graph.

        Returns
        -------------
        nodes : (n,) array
          All node names.
        """
        return self.transforms.nodes

    @caching.cache_decorator
    def nodes_geometry(self):
        """
        The nodes in the scene graph with geometry attached.

        Returns
        ------------
        nodes_geometry : (m,) array
          Node names which have geometry associated
        """
        return [n for n, attr in self.transforms.node_data.items() if "geometry" in attr]

    @caching.cache_decorator
    def geometry_nodes(self):
        """
        Which nodes have this geometry? Inverse
        of `nodes_geometry`.

        Returns
        ------------
        geometry_nodes : dict
          Keyed {geometry_name : node name}
        """
        res = collections.defaultdict(list)
        for node, attr in self.transforms.node_data.items():
            if "geometry" in attr:
                res[attr["geometry"]].append(node)
        return res

    def remove_geometries(self, geometries):
        """
        Remove the reference for specified geometries
        from nodes without deleting the node.

        Parameters
        ------------
        geometries : list or str
          Name of scene.geometry to dereference.
        """
        # make sure we have a set of geometries to remove
        if util.is_string(geometries):
            geometries = [geometries]
        geometries = set(geometries)

        # remove the geometry reference from the node without deleting nodes
        # this lets us keep our cached paths, and will not screw up children
        for attrib in self.transforms.node_data.values():
            if "geometry" in attrib and attrib["geometry"] in geometries:
                attrib.pop("geometry")

        # it would be safer to just run _cache.clear
        # but the only property using the geometry should be
        # nodes_geometry: if this becomes not true change this to clear!
        self._cache.cache.pop("nodes_geometry", None)

    def __contains__(self, key):
        return key in self.transforms.node_data

    def __getitem__(self, key):
        return self.get(key)

    def __setitem__(self, key, value):
        value = np.asanyarray(value)
        if value.shape != (4, 4):
            raise ValueError("Matrix must be specified!")
        return self.update(key, matrix=value)

    def clear(self):
        self.transforms = EnforcedForest()
        self._cache.clear()


class EnforcedForest(object):
    """
    A simple forest graph data structure: every node
    is allowed to have exactly one parent. This makes
    traversal and implementation much simpler than a
    full graph data type; by storing only one parent
    reference, it enforces the structure for "free."
    """

    def __init__(self):
        # since every node can have only one parent
        # this data structure transparently enforces
        # the forest data structure without checks
        # a dict {child : parent}
        self.parents = {}

        # store data for a particular edge keyed by tuple
        # {(u, v) : data }
        self.edge_data = collections.defaultdict(dict)
        # {u: data}
        self.node_data = collections.defaultdict(dict)

        # if multiple calls are made for the same path
        # but the connectivity hasn't changed return cached
        self._cache = {}

    def add_edge(self, u, v, **kwargs):
        """
        Add an edge to the forest cleanly.

        Parameters
        -----------
        u : any
          Hashable node key.
        v : any
          Hashable node key.
        kwargs : dict
           Stored as (u, v) edge data.

        Returns
        --------
        changed : bool
          Return if this operation changed anything.
        """
        self._hash = None

        # topology has changed so clear cache
        if (u, v) not in self.edge_data:
            self._cache = {}
        else:
            # check to see if matrix and geometry are identical
            edge = self.edge_data[(u, v)]
            if util.allclose(kwargs.get("matrix", _identity), edge.get("matrix", _identity), 1e-8) and (
                edge.get("geometry") == kwargs.get("geometry")
            ):
                return False

        # store a parent reference for traversal
        self.parents[v] = u
        # store kwargs for edge data keyed with tuple
        self.edge_data[(u, v)] = kwargs
        # set empty node data
        self.node_data[u].update({})
        if "geometry" in kwargs:
            self.node_data[v].update({"geometry": kwargs["geometry"]})
        else:
            self.node_data[v].update({})

        return True

    def remove_node(self, u):
        """
        Remove a node from the forest.

        Parameters
        -----------
        u : any
          Hashable node key.

        Returns
        --------
        changed : bool
          Return if this operation changed anything.
        """
        # check if node is part of forest
        if u not in self.node_data:
            return False

        # topology will change so clear cache
        self._cache = {}
        self._hash = None

        # delete all children's references and parent reference
        children = [child for (child, parent) in self.parents.items() if parent == u]
        for c in children:
            del self.parents[c]
        if u in self.parents:
            del self.parents[u]

        # delete edge data
        edges = [(a, b) for (a, b) in self.edge_data if a == u or b == u]
        for e in edges:
            del self.edge_data[e]

        # delete node data
        del self.node_data[u]

        return True

    def shortest_path(self, u, v):
        """
        Find the shortest path between `u` and `v`, returning
        a path where the first element is always `u` and the
        last element is always `v`, disregarding edge direction.

        Parameters
        -----------
        u : any
          Hashable node key.
        v : any
          Hashable node key.

        Returns
        -----------
        path : (n,)
          Path between `u` and `v`
        """
        # see if we've already computed this path
        if u == v:
            #  the path between itself is an edge case
            return []
        elif (u, v) in self._cache:
            # return the same path for either direction
            return self._cache[(u, v)]
        elif (v, u) in self._cache:
            return self._cache[(v, u)][::-1]

        # local reference to parent dict for performance
        parents = self.parents
        # store both forward and backwards traversal
        forward = [u]
        backward = [v]

        # cap iteration to number of total nodes
        for _ in range(len(parents) + 1):
            # store the parent both forwards and backwards
            f = parents.get(forward[-1])
            b = parents.get(backward[-1])
            forward.append(f)
            backward.append(b)

            if f == v:
                self._cache[(u, v)] = forward
                return forward
            elif b == u:
                # return reversed path
                backward = backward[::-1]
                self._cache[(u, v)] = backward
                return backward
            elif (b in forward) or (f is None and b is None):
                # we have a either a common node between both
                # traversal directions or we have consumed the whole
                # tree in both directions so try to find the common node
                common = set(backward).intersection(forward).difference({None})
                if len(common) == 0:
                    raise ValueError("No path from {}->{}!".format(u, v))
                elif len(common) > 1:
                    # get the first occurring common element in "forward"
                    link = next(f for f in forward if f in common)
                    assert link in common
                else:
                    # take the only common element
                    link = next(iter(common))

                # combine the forward and backwards traversals
                a = forward[: forward.index(link) + 1]
                b = backward[: backward.index(link)]
                path = a + b[::-1]

                # verify we didn't screw up the order
                assert path[0] == u
                assert path[-1] == v

                self._cache[(u, v)] = path

                return path

        raise ValueError("Iteration limit exceeded!")

    @property
    def nodes(self):
        """
        Get a set of every node.

        Returns
        -----------
        nodes : set
          Every node currently stored.
        """
        return self.node_data.keys()

    @property
    def children(self):
        """
        Get the children of each node.

        Returns
        ----------
        children : dict
          Keyed {node : [child, child, ...]}
        """
        if "children" in self._cache:
            return self._cache["children"]
        child = collections.defaultdict(list)
        # append children to parent references
        # skip self-references to avoid a node loop
        [child[v].append(u) for u, v in self.parents.items() if u != v]

        # cache and return as a vanilla dict
        self._cache["children"] = dict(child)
        return self._cache["children"]

    def successors(self, node):
        """
        Get all nodes that are successors to specified node,
        including the specified node.

        Parameters
        -------------
        node : any
          Hashable key for a node.

        Returns
        ------------
        successors : set
          Nodes that succeed specified node.
        """
        # get mapping of {parent : child}
        children = self.children
        # if node doesn't exist return early
        if node not in children:
            return set([node])

        # children we need to collect
        queue = [node]
        # start collecting values with children of source
        collected = set(queue)

        # cap maximum iterations
        for _ in range(len(self.node_data) + 1):
            if len(queue) == 0:
                # no more nodes to visit so we're done
                return collected
            # add the children of this node to be processed
            childs = children.get(queue.pop())
            if childs is not None:
                queue.extend(childs)
                collected.update(childs)
        return collected

    def __hash__(self):
        """
        Actually hash all of the data, but use a "dirty" mechanism
        in functions that modify the data, which MUST
        # all invalidate the hash by setting `self._hash = None`

        This was optimized a bit, and is evaluating on an
        older laptop on a scene with 77 nodes and 76 edges
        10,000 times in 0.7s which seems fast enough.
        """
        # see if there is an available hash value
        # if you are seeing cache bugs this is the thing
        # to try eliminating because it is very likely that
        # someone somewhere is modifying the data without
        # setting `self._hash = None`
        hashed = getattr(self, "_hash", None)
        if hashed is not None:
            return hashed

        hashed = hash_fast(
            (
                "".join(str(hash(k)) + v.get("geometry", "") for k, v in self.edge_data.items())
                + "".join(str(k) + v.get("geometry", "") for k, v in self.node_data.items())
            ).encode("utf-8")
            + b"".join(v["matrix"].tobytes() for v in self.edge_data.values() if "matrix" in v)
        )
        self._hash = hashed
        return hashed


def kwargs_to_matrix(matrix=None, quaternion=None, translation=None, axis=None, angle=None, **kwargs):
    """
    Take multiple keyword arguments and parse them
    into a homogeneous transformation matrix.

    Returns
    ---------
    matrix : (4, 4) float
      Homogeneous transformation matrix.
    """
    if matrix is not None:
        # a matrix takes immediate precedence over other options
        return np.array(matrix, dtype=np.float64)
    elif quaternion is not None:
        matrix = quaternion_matrix(quaternion)
    elif axis is not None and angle is not None:
        matrix = rotation_matrix(angle, axis)
    else:
        matrix = np.eye(4)

    if translation is not None:
        # translation can be used in conjunction with any
        # of the methods specifying transforms
        matrix[:3, 3] += translation

    return matrix
