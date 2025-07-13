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


# =========================================== Collider ===========================================


@ti.data_oriented
class ColliderState:
    """
    Class to store the mutable collider data, all of which type is [ti.fields].
    """

    def __init__(self, solver, n_possible_pairs, n_vert_neighbors, collider_info):
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
        n_verts = solver.n_verts_
        max_collision_pairs = min(solver._max_collision_pairs, n_possible_pairs)
        max_contact_pairs = max_collision_pairs * collider_info.n_contacts_per_pair
        use_hibernation = solver._static_rigid_sim_config.use_hibernation
        box_box_detection = solver._static_rigid_sim_config.box_box_detection

        ############## vertex connectivity ##############
        self.vert_neighbors = ti.field(dtype=gs.ti_int, shape=max(1, n_vert_neighbors))
        self.vert_neighbor_start = ti.field(dtype=gs.ti_int, shape=n_verts)
        self.vert_n_neighbors = ti.field(dtype=gs.ti_int, shape=n_verts)

        ############## broad phase SAP ##############
        # This buffer stores the AABBs along the search axis of all geoms
        struct_sort_buffer = ti.types.struct(value=gs.ti_float, i_g=gs.ti_int, is_max=gs.ti_int)
        self.sort_buffer = struct_sort_buffer.field(shape=f_batch(2 * n_geoms), layout=ti.Layout.SOA)

        # This buffer stores indexes of active geoms during SAP search
        if use_hibernation:
            self.active_buffer_awake = ti.field(dtype=gs.ti_int, shape=f_batch(n_geoms))
            self.active_buffer_hib = ti.field(dtype=gs.ti_int, shape=f_batch(n_geoms))
        self.active_buffer = ti.field(dtype=gs.ti_int, shape=f_batch(n_geoms))

        # Stores the validity of the collision pairs
        self.collision_pair_validity = ti.field(dtype=gs.ti_int, shape=(n_geoms, n_geoms))

        # Whether or not this is the first time to run the broad phase for each batch
        self.first_time = ti.field(gs.ti_int, shape=_B)

        # Number of possible pairs of collision, store them in a field to avoid recompilation
        self._max_possible_pairs = ti.field(dtype=gs.ti_int, shape=())
        self._max_collision_pairs = ti.field(dtype=gs.ti_int, shape=())
        self._max_contact_pairs = ti.field(dtype=gs.ti_int, shape=())

        # Final results of the broad phase
        self.n_broad_pairs = ti.field(dtype=gs.ti_int, shape=_B)
        self.broad_collision_pairs = ti.Vector.field(2, dtype=gs.ti_int, shape=f_batch(max(1, max_collision_pairs)))

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
            self.box_depth = ti.field(dtype=gs.ti_float, shape=f_batch(collider_info.box_MAXCONPAIR))
            self.box_points = ti.field(gs.ti_vec3, shape=f_batch(collider_info.box_MAXCONPAIR))
            self.box_pts = ti.field(gs.ti_vec3, shape=f_batch(6))
            self.box_lines = ti.field(gs.ti_vec6, shape=f_batch(4))
            self.box_linesu = ti.field(gs.ti_vec6, shape=f_batch(4))
            self.box_axi = ti.field(gs.ti_vec3, shape=f_batch(3))
            self.box_ppts2 = ti.field(dtype=gs.ti_float, shape=f_batch((4, 2)))
            self.box_pu = ti.field(gs.ti_vec3, shape=f_batch(4))

        ########## Terrain contact detection ##########
        if collider_info.has_terrain:
            links_idx = solver.geoms_info.link_idx.to_numpy()[solver.geoms_info.type.to_numpy() == gs.GEOM_TYPE.TERRAIN]
            entity = solver._entities[solver.links_info.entity_idx.to_numpy()[links_idx[0]]]

            self.terrain_hf = ti.field(dtype=gs.ti_float, shape=entity.terrain_hf.shape)
            self.terrain_rc = ti.field(dtype=gs.ti_int, shape=2)
            self.terrain_scale = ti.field(dtype=gs.ti_float, shape=2)
            self.terrain_xyz_maxmin = ti.field(dtype=gs.ti_float, shape=6)

            # for faster compilation
            self.xyz_max_min = ti.field(dtype=gs.ti_float, shape=f_batch(6))
            self.prism = ti.field(dtype=gs.ti_vec3, shape=f_batch(6))
