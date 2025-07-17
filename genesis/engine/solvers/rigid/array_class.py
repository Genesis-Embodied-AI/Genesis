from dataclasses import dataclass
from typing import Callable

import taichi as ti

import genesis as gs
import numpy as np

# we will use struct for DofsState and DofsInfo after Hugh adds array_struct feature to taichi
DofsState = ti.template()
DofsInfo = ti.template()
GeomsState = ti.template()
GeomsInfo = ti.template()
GeomsInitAABB = ti.template()  # TODO: move to rigid global info
LinksState = ti.template()
LinksInfo = ti.template()
JointsInfo = ti.template()
JointsState = ti.template()
VertsState = ti.template()
VertsInfo = ti.template()
EdgesInfo = ti.template()
FacesInfo = ti.template()
VVertsInfo = ti.template()
VFacesInfo = ti.template()
VGeomsInfo = ti.template()
VGeomsState = ti.template()
EntitiesState = ti.template()
EntitiesInfo = ti.template()
EqualitiesInfo = ti.template()


@ti.data_oriented
class RigidGlobalInfo:
    def __init__(self, solver, n_dofs: int, n_entities: int, n_geoms: int, _B: int, f_batch: Callable):
        self.n_awake_dofs = ti.field(dtype=gs.ti_int, shape=f_batch())
        self.awake_dofs = ti.field(dtype=gs.ti_int, shape=f_batch(n_dofs))

        self.qpos0 = ti.field(dtype=gs.ti_float, shape=solver._batch_shape(solver.n_qs_))
        self.qpos = ti.field(dtype=gs.ti_float, shape=solver._batch_shape(solver.n_qs_))

        # self.links_T = ti.Matrix.field(n=4, m=4, dtype=gs.ti_float, shape=solver.n_links)


# =========================================== Constraint ===========================================


@ti.data_oriented
class ConstraintState:
    """
    Class to store the mutable constraint data, all of which type is [ti.fields].
    """

    def __init__(self, solver):
        f_batch = solver._batch_shape
        self.n_constraints = ti.field(dtype=gs.ti_int, shape=f_batch())


# =========================================== Collider ===========================================


@ti.data_oriented
class ColliderState:
    """
    Class to store the MUTABLE collider data, all of which type is [ti.fields] (later we will support NDArrays).
    """

    def __init__(self, solver, n_possible_pairs, collider_static_config):
        """
        Parameters:
        ----------
        n_possible_pairs: int
            Maximum number of possible collision pairs based on geom configurations. For instance, when adjacent
            collision is disabled, adjacent geoms are not considered in counting possible pairs.
        n_vert_neighbors: int
            Size of the vertex neighbors array.
        """
        _B = solver._B
        f_batch = solver._batch_shape
        n_geoms = solver.n_geoms_
        max_collision_pairs = min(solver._max_collision_pairs, n_possible_pairs)
        max_collision_pairs_broad = max_collision_pairs * collider_static_config.max_collision_pairs_broad_k
        max_contact_pairs = max_collision_pairs * collider_static_config.n_contacts_per_pair
        use_hibernation = solver._static_rigid_sim_config.use_hibernation
        box_box_detection = solver._static_rigid_sim_config.box_box_detection

        ############## broad phase SAP ##############
        # This buffer stores the AABBs along the search axis of all geoms
        struct_sort_buffer = ti.types.struct(value=gs.ti_float, i_g=gs.ti_int, is_max=gs.ti_int)
        self.sort_buffer = struct_sort_buffer.field(shape=f_batch(2 * n_geoms), layout=ti.Layout.SOA)

        # This buffer stores indexes of active geoms during SAP search
        if use_hibernation:
            self.active_buffer_awake = ti.field(dtype=gs.ti_int, shape=f_batch(n_geoms))
            self.active_buffer_hib = ti.field(dtype=gs.ti_int, shape=f_batch(n_geoms))
        self.active_buffer = ti.field(dtype=gs.ti_int, shape=f_batch(n_geoms))

        # Whether or not this is the first time to run the broad phase for each batch
        self.first_time = ti.field(gs.ti_int, shape=_B)

        # Final results of the broad phase
        self.n_broad_pairs = ti.field(dtype=gs.ti_int, shape=_B)
        self.broad_collision_pairs = ti.Vector.field(
            2, dtype=gs.ti_int, shape=f_batch(max(1, max_collision_pairs_broad))
        )

        ############## narrow phase ##############
        struct_contact_data = ti.types.struct(
            geom_a=gs.ti_int,
            geom_b=gs.ti_int,
            penetration=gs.ti_float,
            normal=gs.ti_vec3,
            pos=gs.ti_vec3,
            friction=gs.ti_float,
            sol_params=gs.ti_vec7,
            force=gs.ti_vec3,
            link_a=gs.ti_int,
            link_b=gs.ti_int,
        )
        self.contact_data = struct_contact_data.field(
            shape=f_batch(max(1, max_contact_pairs)),
            layout=ti.Layout.SOA,
        )
        # total number of contacts, including hibernated contacts
        self.n_contacts = ti.field(gs.ti_int, shape=_B)
        self.n_contacts_hibernated = ti.field(gs.ti_int, shape=_B)

        # contact caching for warmstart collision detection
        struct_contact_cache = ti.types.struct(
            # i_va_ws=gs.ti_int,
            # penetration=gs.ti_float,
            normal=gs.ti_vec3,
        )
        self.contact_cache = struct_contact_cache.field(shape=f_batch((n_geoms, n_geoms)), layout=ti.Layout.SOA)

        ########## Box-box contact detection ##########
        if box_box_detection:
            # With the existing Box-Box collision detection algorithm, it is not clear where the contact points are
            # located depending of the pose and size of each box. In practice, up to 11 contact points have been
            # observed. The theoretical worst case scenario would be 2 cubes roughly the same size and same center,
            # with transform RPY = (45, 45, 45), resulting in 3 contact points per faces for a total of 16 points.
            self.box_depth = ti.field(dtype=gs.ti_float, shape=f_batch(collider_static_config.box_MAXCONPAIR))
            self.box_points = ti.field(gs.ti_vec3, shape=f_batch(collider_static_config.box_MAXCONPAIR))
            self.box_pts = ti.field(gs.ti_vec3, shape=f_batch(6))
            self.box_lines = ti.field(gs.ti_vec6, shape=f_batch(4))
            self.box_linesu = ti.field(gs.ti_vec6, shape=f_batch(4))
            self.box_axi = ti.field(gs.ti_vec3, shape=f_batch(3))
            self.box_ppts2 = ti.field(dtype=gs.ti_float, shape=f_batch((4, 2)))
            self.box_pu = ti.field(gs.ti_vec3, shape=f_batch(4))

        ########## Terrain contact detection ##########
        if collider_static_config.has_terrain:
            # for faster compilation
            self.xyz_max_min = ti.field(dtype=gs.ti_float, shape=f_batch(6))
            self.prism = ti.field(dtype=gs.ti_vec3, shape=f_batch(6))


@ti.data_oriented
class ColliderInfo:
    """
    Class to store the IMMUTABLE collider data, all of which type is [ti.fields] (later we will support NDArrays).
    """

    def __init__(self, solver, n_vert_neighbors, collider_static_config):
        """
        Parameters:
        ----------
        n_vert_neighbors: int
            Size of the vertex neighbors array.
        """
        n_geoms = solver.n_geoms_
        n_verts = solver.n_verts_

        ############## vertex connectivity ##############
        self.vert_neighbors = ti.field(dtype=gs.ti_int, shape=max(1, n_vert_neighbors))
        self.vert_neighbor_start = ti.field(dtype=gs.ti_int, shape=n_verts)
        self.vert_n_neighbors = ti.field(dtype=gs.ti_int, shape=n_verts)

        ############## broad phase SAP ##############
        # Stores the validity of the collision pairs
        self.collision_pair_validity = ti.field(dtype=gs.ti_int, shape=(n_geoms, n_geoms))

        # Number of possible pairs of collision, store them in a field to avoid recompilation
        self._max_possible_pairs = ti.field(dtype=gs.ti_int, shape=())
        self._max_collision_pairs = ti.field(dtype=gs.ti_int, shape=())
        self._max_contact_pairs = ti.field(dtype=gs.ti_int, shape=())
        self._max_collision_pairs_broad = ti.field(dtype=gs.ti_int, shape=())

        ########## Terrain contact detection ##########
        if collider_static_config.has_terrain:
            links_idx = solver.geoms_info.link_idx.to_numpy()[solver.geoms_info.type.to_numpy() == gs.GEOM_TYPE.TERRAIN]
            entity = solver._entities[solver.links_info.entity_idx.to_numpy()[links_idx[0]]]

            self.terrain_hf = ti.field(dtype=gs.ti_float, shape=entity.terrain_hf.shape)
            self.terrain_rc = ti.field(dtype=gs.ti_int, shape=2)
            self.terrain_scale = ti.field(dtype=gs.ti_float, shape=2)
            self.terrain_xyz_maxmin = ti.field(dtype=gs.ti_float, shape=6)


# =========================================== MPR ===========================================
@ti.data_oriented
class MPRState:
    def __init__(self, f_batch):
        struct_support = ti.types.struct(
            v1=gs.ti_vec3,
            v2=gs.ti_vec3,
            v=gs.ti_vec3,
        )
        self.simplex_support = struct_support.field(
            shape=f_batch(4),
            layout=ti.Layout.SOA,
        )
        self.simplex_size = ti.field(gs.ti_int, shape=f_batch())


# =========================================== GJK ===========================================
@ti.data_oriented
class GJKState:
    def __init__(self, solver, static_rigid_sim_config, gjk_static_config):
        _B = solver._B
        polytope_max_faces = gjk_static_config.polytope_max_faces
        max_contacts_per_pair = gjk_static_config.max_contacts_per_pair
        max_contact_polygon_verts = gjk_static_config.max_contact_polygon_verts

        # Cache to store the previous support points for support mesh function.
        self.support_mesh_prev_vertex_id = ti.field(dtype=gs.ti_int, shape=(_B, 2))

        ### GJK simplex
        struct_simplex_vertex = ti.types.struct(
            # Support points on the two objects
            obj1=gs.ti_vec3,
            obj2=gs.ti_vec3,
            # Support point IDs on the two objects
            id1=gs.ti_int,
            id2=gs.ti_int,
            # Vertex on Minkowski difference
            mink=gs.ti_vec3,
        )
        struct_simplex = ti.types.struct(
            # Number of vertices in the simplex
            nverts=gs.ti_int,
            # Distance from the origin to the simplex
            dist=gs.ti_float,
        )
        struct_simplex_buffer = ti.types.struct(
            # Normals of the simplex faces
            normal=gs.ti_vec3,
            # Signed distances of the simplex faces from the origin
            sdist=gs.ti_float,
        )
        self.simplex_vertex = struct_simplex_vertex.field(shape=(_B, 4))
        self.simplex_buffer = struct_simplex_buffer.field(shape=(_B, 4))
        self.simplex = struct_simplex.field(shape=(_B,))

        # Only when we enable MuJoCo compatibility, we use the simplex vertex and buffer for intersection checks.
        if static_rigid_sim_config.enable_mujoco_compatibility:
            self.simplex_vertex_intersect = struct_simplex_vertex.field(shape=(_B, 4))
            self.simplex_buffer_intersect = struct_simplex_buffer.field(shape=(_B, 4))
            self.nsimplex = ti.field(dtype=gs.ti_int, shape=(_B,))

        # In safe GJK, if the initial simplex is degenerate and the geometries are discrete, we go through vertices
        # on the Minkowski difference to find a vertex that would make a valid simplex. To prevent iterating through
        # the same vertices again during initial simplex construction, we keep the vertex ID of the last vertex that
        # we searched, so that we can start searching from the next vertex.
        self.last_searched_simplex_vertex_id = ti.field(dtype=gs.ti_int, shape=(_B,))

        ### EPA polytope
        struct_polytope_vertex = struct_simplex_vertex
        struct_polytope_face = ti.types.struct(
            # Indices of the vertices forming the face on the polytope
            verts_idx=gs.ti_ivec3,
            # Indices of adjacent faces, one for each edge: [v1,v2], [v2,v3], [v3,v1]
            adj_idx=gs.ti_ivec3,
            # Projection of the origin onto the face, can be used as face normal
            normal=gs.ti_vec3,
            # Square of 2-norm of the normal vector, negative means deleted face
            dist2=gs.ti_float,
            # Index of the face in the polytope map, -1 for not in the map, -2 for deleted
            map_idx=gs.ti_int,
        )
        # Horizon is used for representing the faces to delete when the polytope is expanded by inserting a new vertex.
        struct_polytope_horizon_data = ti.types.struct(
            # Indices of faces on horizon
            face_idx=gs.ti_int,
            # Corresponding edge of each face on the horizon
            edge_idx=gs.ti_int,
        )
        struct_polytope = ti.types.struct(
            # Number of vertices in the polytope
            nverts=gs.ti_int,
            # Number of faces in the polytope (it could include deleted faces)
            nfaces=gs.ti_int,
            # Number of faces in the polytope map (only valid faces on polytope)
            nfaces_map=gs.ti_int,
            # Number of edges in the horizon
            horizon_nedges=gs.ti_int,
            # Support point on the Minkowski difference where the horizon is created
            horizon_w=gs.ti_vec3,
        )

        self.polytope = struct_polytope.field(shape=(_B,))
        self.polytope_verts = struct_polytope_vertex.field(shape=(_B, 5 + gjk_static_config.epa_max_iterations))
        self.polytope_faces = struct_polytope_face.field(shape=(_B, polytope_max_faces))
        self.polytope_horizon_data = struct_polytope_horizon_data.field(
            shape=(_B, 6 + gjk_static_config.epa_max_iterations)
        )

        # Face indices that form the polytope. The first [nfaces_map] indices are the faces that form the polytope.
        self.polytope_faces_map = ti.Vector.field(n=polytope_max_faces, dtype=gs.ti_int, shape=(_B,))

        # Stack to use for visiting faces during the horizon construction. The size is (# max faces * 3),
        # because a face has 3 edges.
        self.polytope_horizon_stack = struct_polytope_horizon_data.field(shape=(_B, polytope_max_faces * 3))

        # Data structures for multi-contact detection based on MuJoCo's implementation.
        if gjk_static_config.enable_mujoco_multi_contact:
            struct_contact_face = ti.types.struct(
                # Vertices from the two colliding faces
                vert1=gs.ti_vec3,
                vert2=gs.ti_vec3,
                endverts=gs.ti_vec3,
                # Normals of the two colliding faces
                normal1=gs.ti_vec3,
                normal2=gs.ti_vec3,
                # Face ID of the two colliding faces
                id1=gs.ti_int,
                id2=gs.ti_int,
            )
            # Struct for storing temp. contact normals
            struct_contact_normal = ti.types.struct(
                endverts=gs.ti_vec3,
                # Normal vector of the contact point
                normal=gs.ti_vec3,
                # Face ID
                id=gs.ti_int,
            )
            struct_contact_halfspace = ti.types.struct(
                # Halfspace normal
                normal=gs.ti_vec3,
                # Halfspace distance from the origin
                dist=gs.ti_float,
            )
            self.contact_faces = struct_contact_face.field(shape=(_B, max_contact_polygon_verts))
            self.contact_normals = struct_contact_normal.field(shape=(_B, max_contact_polygon_verts))
            self.contact_halfspaces = struct_contact_halfspace.field(shape=(_B, max_contact_polygon_verts))
            self.contact_clipped_polygons = gs.ti_vec3.field(shape=(_B, 2, max_contact_polygon_verts))

        # Whether or not the MuJoCo's contact manifold detection algorithm was used for the current pair.
        self.multi_contact_flag = ti.field(dtype=gs.ti_int, shape=(_B,))

        ### Final results
        # Witness information
        struct_witness = ti.types.struct(
            # Witness points on the two objects
            point_obj1=gs.ti_vec3,
            point_obj2=gs.ti_vec3,
        )
        self.witness = struct_witness.field(shape=(_B, max_contacts_per_pair))
        self.n_witness = ti.field(dtype=gs.ti_int, shape=(_B,))

        # Contact information, the namings are the same as those from the calling function. Even if they could be
        # redundant, we keep them for easier use from the calling function.
        self.n_contacts = ti.field(dtype=gs.ti_int, shape=(_B,))
        self.contact_pos = gs.ti_vec3.field(shape=(_B, max_contacts_per_pair))
        self.normal = gs.ti_vec3.field(shape=(_B, max_contacts_per_pair))
        self.is_col = ti.field(dtype=gs.ti_int, shape=(_B,))
        self.penetration = ti.field(dtype=gs.ti_float, shape=(_B,))

        # Distance between the two objects.
        # If the objects are separated, the distance is positive.
        # If the objects are intersecting, the distance is negative (depth).
        self.distance = ti.field(dtype=gs.ti_float, shape=(_B,))


# =========================================== SupportField ===========================================
@ti.data_oriented
class SupportFieldInfo:
    """
    Class to store the IMMUTABLE support field data, all of which type is [ti.fields] (later we will support NDArrays).
    """

    def __init__(self, n_geoms, n_support_cells):
        self.support_cell_start = ti.field(dtype=gs.ti_int, shape=n_geoms)
        self.support_v = ti.Vector.field(3, dtype=gs.ti_float, shape=max(1, n_support_cells))
        self.support_vid = ti.field(dtype=gs.ti_int, shape=max(1, n_support_cells))
