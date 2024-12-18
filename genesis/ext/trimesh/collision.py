import collections

import numpy as np

try:
    # pip install python-fcl
    import fcl
except BaseException:
    fcl = None


class ContactData(object):
    """
    Data structure for holding information about a collision contact.
    """

    def __init__(self, names, contact):
        """
        Initialize a ContactData.

        Parameters
        ----------
        names : list of str
          The names of the two objects in order.
        contact : fcl.Contact
          The contact in question.
        """
        self.names = set(names)
        self._inds = {names[0]: contact.b1, names[1]: contact.b2}
        self._normal = contact.normal
        self._point = contact.pos
        self._depth = contact.penetration_depth

    @property
    def normal(self):
        """
        The 3D intersection normal for this contact.

        Returns
        -------
        normal : (3,) float
          The intersection normal.
        """
        return self._normal

    @property
    def point(self):
        """
        The 3D point of intersection for this contact.

        Returns
        -------
        point : (3,) float
          The intersection point.
        """
        return self._point

    @property
    def depth(self):
        """
        The penetration depth of the 3D point of intersection for this contact.

        Returns
        -------
        depth : float
          The penetration depth.
        """
        return self._depth

    def index(self, name):
        """
        Returns the index of the face in contact for the mesh with
        the given name.

        Parameters
        ----------
        name : str
          The name of the target object.

        Returns
        -------
        index : int
          The index of the face in collision
        """
        return self._inds[name]


class DistanceData(object):
    """
    Data structure for holding information about a distance query.
    """

    def __init__(self, names, result):
        """
        Initialize a DistanceData.

        Parameters
        ----------
        names : list of str
          The names of the two objects in order.
        contact : fcl.DistanceResult
          The distance query result.
        """
        self.names = set(names)
        self._inds = {names[0]: result.b1, names[1]: result.b2}
        self._points = {names[0]: result.nearest_points[0], names[1]: result.nearest_points[1]}
        self._distance = result.min_distance

    @property
    def distance(self):
        """
        Returns the distance between the two objects.

        Returns
        -------
        distance : float
          The euclidean distance between the objects.
        """
        return self._distance

    def index(self, name):
        """
        Returns the index of the closest face for the mesh with
        the given name.

        Parameters
        ----------
        name : str
          The name of the target object.

        Returns
        -------
        index : int
          The index of the face in collisoin.
        """
        return self._inds[name]

    def point(self, name):
        """
        The 3D point of closest distance on the mesh with the given name.

        Parameters
        ----------
        name : str
          The name of the target object.

        Returns
        -------
        point : (3,) float
          The closest point.
        """
        return self._points[name]


class CollisionManager(object):
    """
    A mesh-mesh collision manager.
    """

    def __init__(self):
        """
        Initialize a mesh-mesh collision manager.
        """
        if fcl is None:
            raise ValueError("No FCL Available! Please install the python-fcl library")
        # {name: {geom:, obj}}
        self._objs = {}
        # {id(bvh) : str, name}
        # unpopulated values will return None
        self._names = collections.defaultdict(lambda: None)

        self._manager = fcl.DynamicAABBTreeCollisionManager()
        self._manager.setup()

    def add_object(self, name, mesh, transform=None):
        """
        Add an object to the collision manager.

        If an object with the given name is already in the manager,
        replace it.

        Parameters
        ----------
        name : str
          An identifier for the object
        mesh : Trimesh object
          The geometry of the collision object
        transform : (4,4) float
          Homogeneous transform matrix for the object
        """

        # if no transform passed, assume identity transform
        if transform is None:
            transform = np.eye(4)
        transform = np.asanyarray(transform, dtype=np.float32)
        if transform.shape != (4, 4):
            raise ValueError("transform must be (4,4)!")

        # create BVH/Convex
        geom = self._get_fcl_obj(mesh)

        # create the FCL transform from (4,4) matrix
        t = fcl.Transform(transform[:3, :3], transform[:3, 3])
        o = fcl.CollisionObject(geom, t)

        # Add collision object to set
        if name in self._objs:
            self._manager.unregisterObject(self._objs[name])
        self._objs[name] = {"obj": o, "geom": geom}
        # store the name of the geometry
        self._names[id(geom)] = name

        self._manager.registerObject(o)
        self._manager.update()
        return o

    def remove_object(self, name):
        """
        Delete an object from the collision manager.

        Parameters
        ----------
        name : str
          The identifier for the object
        """
        if name in self._objs:
            self._manager.unregisterObject(self._objs[name]["obj"])
            self._manager.update(self._objs[name]["obj"])
            # remove objects from _objs
            geom_id = gs.UID(self._objs.pop(name)["geom"])
            # remove names
            self._names.pop(geom_id)
        else:
            raise ValueError("{} not in collision manager!".format(name))

    def set_transform(self, name, transform):
        """
        Set the transform for one of the manager's objects.
        This replaces the prior transform.

        Parameters
        ----------
        name : str
          An identifier for the object already in the manager
        transform : (4,4) float
          A new homogeneous transform matrix for the object
        """
        if name in self._objs:
            o = self._objs[name]["obj"]
            o.setRotation(transform[:3, :3])
            o.setTranslation(transform[:3, 3])
            self._manager.update(o)
        else:
            raise ValueError("{} not in collision manager!".format(name))

    def in_collision_single(self, mesh, transform=None, return_names=False, return_data=False):
        """
        Check a single object for collisions against all objects in the
        manager.

        Parameters
        ----------
        mesh : Trimesh object
          The geometry of the collision object
        transform : (4,4) float
          Homogeneous transform matrix
        return_names : bool
          If true, a set is returned containing the names
          of all objects in collision with the object
        return_data :  bool
          If true, a list of ContactData is returned as well

        Returns
        ------------
        is_collision : bool
          True if a collision occurs and False otherwise
        names : set of str
          [OPTIONAL] The set of names of objects that collided with the
          provided one
        contacts : list of ContactData
          [OPTIONAL] All contacts detected
        """
        if transform is None:
            transform = np.eye(4)

        # create BVH/Convex
        geom = self._get_fcl_obj(mesh)

        # create the FCL transform from (4,4) matrix
        t = fcl.Transform(transform[:3, :3], transform[:3, 3])
        o = fcl.CollisionObject(geom, t)

        # Collide with manager's objects
        cdata = fcl.CollisionData()
        if return_names or return_data:
            cdata = fcl.CollisionData(request=fcl.CollisionRequest(num_max_contacts=100000, enable_contact=True))

        self._manager.collide(o, cdata, fcl.defaultCollisionCallback)
        result = cdata.result.is_collision

        # If we want to return the objects that were collision, collect them.
        objs_in_collision = set()
        contact_data = []
        if return_names or return_data:
            for contact in cdata.result.contacts:
                cg = contact.o1
                if cg == geom:
                    cg = contact.o2
                name = self._extract_name(cg)

                names = (name, "__external")
                if cg == contact.o2:
                    names = tuple(reversed(names))

                if return_names:
                    objs_in_collision.add(name)
                if return_data:
                    contact_data.append(ContactData(names, contact))

        if return_names and return_data:
            return result, objs_in_collision, contact_data
        elif return_names:
            return result, objs_in_collision
        elif return_data:
            return result, contact_data
        else:
            return result

    def in_collision_internal(self, return_names=False, return_data=False):
        """
        Check if any pair of objects in the manager collide with one another.

        Parameters
        ----------
        return_names : bool
          If true, a set is returned containing the names
          of all pairs of objects in collision.
        return_data :  bool
          If true, a list of ContactData is returned as well

        Returns
        -------
        is_collision : bool
          True if a collision occurred between any pair of objects
          and False otherwise
        names : set of 2-tup
          The set of pairwise collisions. Each tuple
          contains two names in alphabetical order indicating
          that the two corresponding objects are in collision.
        contacts : list of ContactData
          All contacts detected
        """
        cdata = fcl.CollisionData()
        if return_names or return_data:
            cdata = fcl.CollisionData(request=fcl.CollisionRequest(num_max_contacts=100000, enable_contact=True))

        self._manager.collide(cdata, fcl.defaultCollisionCallback)

        result = cdata.result.is_collision

        objs_in_collision = set()
        contact_data = []
        if return_names or return_data:
            for contact in cdata.result.contacts:
                names = (self._extract_name(contact.o1), self._extract_name(contact.o2))

                if return_names:
                    objs_in_collision.add(tuple(sorted(names)))
                if return_data:
                    contact_data.append(ContactData(names, contact))

        if return_names and return_data:
            return result, objs_in_collision, contact_data
        elif return_names:
            return result, objs_in_collision
        elif return_data:
            return result, contact_data
        else:
            return result

    def in_collision_other(self, other_manager, return_names=False, return_data=False):
        """
        Check if any object from this manager collides with any object
        from another manager.

        Parameters
        -------------------
        other_manager : CollisionManager
          Another collision manager object
        return_names : bool
          If true, a set is returned containing the names
          of all pairs of objects in collision.
        return_data : bool
          If true, a list of ContactData is returned as well

        Returns
        -------------
        is_collision : bool
          True if a collision occurred between any pair of objects
          and False otherwise
        names : set of 2-tup
          The set of pairwise collisions. Each tuple
          contains two names (first from this manager,
          second from the other_manager) indicating
          that the two corresponding objects are in collision.
        contacts : list of ContactData
          All contacts detected
        """
        cdata = fcl.CollisionData()
        if return_names or return_data:
            cdata = fcl.CollisionData(request=fcl.CollisionRequest(num_max_contacts=100000, enable_contact=True))
        self._manager.collide(other_manager._manager, cdata, fcl.defaultCollisionCallback)
        result = cdata.result.is_collision

        objs_in_collision = set()
        contact_data = []
        if return_names or return_data:
            for contact in cdata.result.contacts:
                reverse = False
                names = (self._extract_name(contact.o1), other_manager._extract_name(contact.o2))
                if names[0] is None:
                    names = (self._extract_name(contact.o2), other_manager._extract_name(contact.o1))
                    reverse = True

                if return_names:
                    objs_in_collision.add(names)
                if return_data:
                    if reverse:
                        names = tuple(reversed(names))
                    contact_data.append(ContactData(names, contact))

        if return_names and return_data:
            return result, objs_in_collision, contact_data
        elif return_names:
            return result, objs_in_collision
        elif return_data:
            return result, contact_data
        else:
            return result

    def min_distance_single(self, mesh, transform=None, return_name=False, return_data=False):
        """
        Get the minimum distance between a single object and any
        object in the manager.

        Parameters
        ---------------
        mesh : Trimesh object
          The geometry of the collision object
        transform : (4,4) float
          Homogeneous transform matrix for the object
        return_names : bool
          If true, return name of the closest object
        return_data : bool
          If true, a DistanceData object is returned as well

        Returns
        -------------
        distance : float
          Min distance between mesh and any object in the manager
        name : str
          The name of the object in the manager that was closest
        data : DistanceData
          Extra data about the distance query
        """
        if transform is None:
            transform = np.eye(4)

        # create BVH/Convex
        geom = self._get_fcl_obj(mesh)

        # create the FCL transform from (4,4) matrix
        t = fcl.Transform(transform[:3, :3], transform[:3, 3])
        o = fcl.CollisionObject(geom, t)

        # Collide with manager's objects
        ddata = fcl.DistanceData(fcl.DistanceRequest(enable_signed_distance=True))
        if return_data:
            ddata = fcl.DistanceData(
                fcl.DistanceRequest(enable_nearest_points=True, enable_signed_distance=True), fcl.DistanceResult()
            )

        self._manager.distance(o, ddata, fcl.defaultDistanceCallback)

        distance = ddata.result.min_distance

        # If we want to return the objects that were collision, collect them.
        name, data = None, None
        if return_name or return_data:
            cg = ddata.result.o1
            if cg == geom:
                cg = ddata.result.o2

            name = self._extract_name(cg)

            names = (name, "__external")
            if cg == ddata.result.o2:
                names = tuple(reversed(names))
            data = DistanceData(names, ddata.result)

        if return_name and return_data:
            return distance, name, data
        elif return_name:
            return distance, name
        elif return_data:
            return distance, data
        else:
            return distance

    def min_distance_internal(self, return_names=False, return_data=False):
        """
        Get the minimum distance between any pair of objects in the manager.

        Parameters
        -------------
        return_names : bool
          If true, a 2-tuple is returned containing the names
          of the closest objects.
        return_data : bool
          If true, a DistanceData object is returned as well

        Returns
        -----------
        distance : float
          Min distance between any two managed objects
        names : (2,) str
          The names of the closest objects
        data : DistanceData
          Extra data about the distance query
        """
        ddata = fcl.DistanceData(fcl.DistanceRequest(enable_signed_distance=True))
        if return_data:
            ddata = fcl.DistanceData(
                fcl.DistanceRequest(
                    enable_nearest_points=True,
                    enable_signed_distance=True,
                ),
                fcl.DistanceResult(),
            )

        self._manager.distance(ddata, fcl.defaultDistanceCallback)

        distance = ddata.result.min_distance

        names, data = None, None
        if return_names or return_data:
            names = (self._extract_name(ddata.result.o1), self._extract_name(ddata.result.o2))
            data = DistanceData(names, ddata.result)
            names = tuple(sorted(names))

        if return_names and return_data:
            return distance, names, data
        elif return_names:
            return distance, names
        elif return_data:
            return distance, data
        else:
            return distance

    def min_distance_other(self, other_manager, return_names=False, return_data=False):
        """
        Get the minimum distance between any pair of objects,
        one in each manager.

        Parameters
        ----------
        other_manager : CollisionManager
          Another collision manager object
        return_names : bool
          If true, a 2-tuple is returned containing
          the names of the closest objects.
        return_data : bool
          If true, a DistanceData object is returned as well

        Returns
        -----------
        distance : float
          The min distance between a pair of objects,
          one from each manager.
        names : 2-tup of str
          A 2-tuple containing two names (first from this manager,
          second from the other_manager) indicating
          the two closest objects.
        data : DistanceData
          Extra data about the distance query
        """
        ddata = fcl.DistanceData(fcl.DistanceRequest(enable_signed_distance=True))
        if return_data:
            ddata = fcl.DistanceData(
                fcl.DistanceRequest(
                    enable_nearest_points=True,
                    enable_signed_distance=True,
                ),
                fcl.DistanceResult(),
            )

        self._manager.distance(other_manager._manager, ddata, fcl.defaultDistanceCallback)

        distance = ddata.result.min_distance

        names, data = None, None
        if return_names or return_data:
            reverse = False
            names = (self._extract_name(ddata.result.o1), other_manager._extract_name(ddata.result.o2))
            if names[0] is None:
                reverse = True
                names = (self._extract_name(ddata.result.o2), other_manager._extract_name(ddata.result.o1))

            dnames = tuple(names)
            if reverse:
                dnames = tuple(reversed(dnames))
            data = DistanceData(dnames, ddata.result)

        if return_names and return_data:
            return distance, names, data
        elif return_names:
            return distance, names
        elif return_data:
            return distance, data
        else:
            return distance

    def _get_fcl_obj(self, mesh):
        """
        Get a BVH or Convex for a mesh.

        Parameters
        -------------
        mesh : Trimesh
          Mesh to create BVH/Convex for

        Returns
        --------------
        obj : fcl.BVHModel or fcl.Convex
          BVH/Convex object of source mesh
        """

        if mesh.is_convex:
            obj = mesh_to_convex(mesh)
        else:
            obj = mesh_to_BVH(mesh)
        return obj

    def _extract_name(self, geom):
        """
        Retrieve the name of an object from the manager by its
        CollisionObject, or return None if not found.

        Parameters
        -----------
        geom : CollisionObject or BVHModel
          Input model

        Returns
        ------------
        names : hashable
          Name of input geometry
        """
        return self._names[id(geom)]


def mesh_to_BVH(mesh):
    """
    Create a BVHModel object from a Trimesh object

    Parameters
    -----------
    mesh : Trimesh
      Input geometry

    Returns
    ------------
    bvh : fcl.BVHModel
      BVH of input geometry
    """
    bvh = fcl.BVHModel()
    bvh.beginModel(num_tris_=len(mesh.faces), num_vertices_=len(mesh.vertices))
    bvh.addSubModel(verts=mesh.vertices, triangles=mesh.faces)
    bvh.endModel()
    return bvh


def mesh_to_convex(mesh):
    """
    Create a Convex object from a Trimesh object

    Parameters
    -----------
    mesh : Trimesh
      Input geometry

    Returns
    ------------
    convex : fcl.Convex
      Convex of input geometry
    """
    fs = np.concatenate((3 * np.ones((len(mesh.faces), 1), dtype=np.int64), mesh.faces), axis=1)
    return fcl.Convex(mesh.vertices, len(fs), fs.flatten())


def scene_to_collision(scene):
    """
    Create collision objects from a trimesh.Scene object.

    Parameters
    ------------
    scene : trimesh.Scene
      Scene to create collision objects for

    Returns
    ------------
    manager : CollisionManager
      CollisionManager for objects in scene
    objects: {node name: CollisionObject}
      Collision objects for nodes in scene
    """
    manager = CollisionManager()
    objects = {}
    for node in scene.graph.nodes_geometry:
        T, geometry = scene.graph[node]
        objects[node] = manager.add_object(name=node, mesh=scene.geometry[geometry], transform=T)
    return manager, objects
