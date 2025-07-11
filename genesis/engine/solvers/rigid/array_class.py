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
GeomsInitAABB = ti.template()
LinksState = ti.template()
LinksInfo = ti.template()
VertsInfo = ti.template()
EdgesInfo = ti.template()


@ti.data_oriented
class RigidGlobalInfo:
    def __init__(self, n_dofs: int, n_entities: int, n_geoms: int, _B: int, f_batch: Callable):
        self.n_awake_dofs = ti.field(dtype=gs.ti_int, shape=f_batch())
        self.awake_dofs = ti.field(dtype=gs.ti_int, shape=f_batch(n_dofs))

        self.n_geoms = ti.field(dtype=gs.ti_int, shape=())
        self.n_geoms[None] = n_geoms

        self._B = ti.field(dtype=gs.ti_int, shape=())
        self._B[None] = _B


# =========================================== Collider ===========================================


@ti.data_oriented
class ColliderState:
    """
    Class to store the mutable collider data, all of which type is [ti.fields].
    """

    def __init__(self, solver, collider_info):
        _B = solver._B
        f_batch = solver._batch_shape
        n_geoms = solver.n_geoms_
        max_collision_pairs = solver._max_collision_pairs
        use_hibernation = solver._static_rigid_sim_config.use_hibernation

        ############## vertex connectivity ##############
        self._init_verts_connectivity(solver)

        ############## broad phase SAP ##############
        # This buffer stores the AABBs along the search axis of all geoms
        struct_sort_buffer = ti.types.struct(value=gs.ti_float, i_g=gs.ti_int, is_max=gs.ti_int)
        self.sort_buffer = struct_sort_buffer.field(
            shape=f_batch(2 * n_geoms),
            layout=ti.Layout.SOA,
        )

        # This buffer stores indexes of active geoms during SAP search
        if use_hibernation:
            self.active_buffer_awake = ti.field(dtype=gs.ti_int, shape=f_batch(n_geoms))
            self.active_buffer_hib = ti.field(dtype=gs.ti_int, shape=f_batch(n_geoms))
        self.active_buffer = ti.field(dtype=gs.ti_int, shape=f_batch(n_geoms))

        # Stores the validity of the collision pairs
        self.collision_pair_validity = ti.field(dtype=gs.ti_int, shape=(n_geoms, n_geoms))
        n_possible_pairs = self._init_collision_pair_validity(solver)

        # Whether or not this is the first time to run the broad phase for each batch
        self.first_time = ti.field(gs.ti_int, shape=_B)

        # Number of possible pairs of collision, store them in a field to avoid recompilation
        self._max_possible_pairs = ti.field(dtype=gs.ti_int, shape=())
        self._max_collision_pairs = ti.field(dtype=gs.ti_int, shape=())
        self._max_contact_pairs = ti.field(dtype=gs.ti_int, shape=())

        self._max_possible_pairs[None] = n_possible_pairs
        self._max_collision_pairs[None] = min(n_possible_pairs, max_collision_pairs)
        self._max_contact_pairs[None] = self._max_collision_pairs[None] * collider_info.n_contacts_per_pair

        # Final results of the broad phase
        self.n_broad_pairs = ti.field(dtype=gs.ti_int, shape=_B)
        self.broad_collision_pairs = ti.Vector.field(
            2, dtype=gs.ti_int, shape=f_batch(max(1, self._max_collision_pairs[None]))
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
            shape=f_batch(max(1, self._max_contact_pairs[None])),
            layout=ti.Layout.SOA,
        )
        # total number of contacts, including hibernated contacts
        self.n_contacts = ti.field(gs.ti_int, shape=_B)
        self.n_contacts_hibernated = ti.field(gs.ti_int, shape=_B)
        self._contacts_info_cache = {}

        # contact caching for warmstart collision detection
        struct_contact_cache = ti.types.struct(
            # i_va_ws=gs.ti_int,
            # penetration=gs.ti_float,
            normal=gs.ti_vec3,
        )
        self.contact_cache = struct_contact_cache.field(
            shape=f_batch((n_geoms, n_geoms)),
            layout=ti.Layout.SOA,
        )

        # for faster compilation
        if collider_info.has_terrain:
            self.xyz_max_min = ti.field(dtype=gs.ti_float, shape=f_batch(6))
            self.prism = ti.field(dtype=gs.ti_vec3, shape=f_batch(6))

        ########## Box-box contact detection ##########
        if solver._box_box_detection:
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
        ##---------------- box box

    def _init_verts_connectivity(self, solver) -> None:
        """
        Initialize the vertex connectivity fields.
        """
        vert_neighbors = []
        vert_neighbor_start = []
        vert_n_neighbors = []
        offset = 0
        for geom in solver.geoms:
            vert_neighbors.append(geom.vert_neighbors + geom.vert_start)
            vert_neighbor_start.append(geom.vert_neighbor_start + offset)
            vert_n_neighbors.append(geom.vert_n_neighbors)
            offset += len(geom.vert_neighbors)

        if solver.n_verts > 0:
            vert_neighbors = np.concatenate(vert_neighbors, dtype=gs.np_int)
            vert_neighbor_start = np.concatenate(vert_neighbor_start, dtype=gs.np_int)
            vert_n_neighbors = np.concatenate(vert_n_neighbors, dtype=gs.np_int)

        self.vert_neighbors = ti.field(dtype=gs.ti_int, shape=max(1, len(vert_neighbors)))
        self.vert_neighbor_start = ti.field(dtype=gs.ti_int, shape=solver.n_verts_)
        self.vert_n_neighbors = ti.field(dtype=gs.ti_int, shape=solver.n_verts_)

        if solver.n_verts > 0:
            self.vert_neighbors.from_numpy(vert_neighbors)
            self.vert_neighbor_start.from_numpy(vert_neighbor_start)
            self.vert_n_neighbors.from_numpy(vert_n_neighbors)

    def _init_collision_pair_validity(self, solver):
        """
        Initialize the collision pair validity matrix.

        For each pair of geoms, determine if they can collide based on their properties and the solver configuration.
        """
        n_geoms = solver.n_geoms_
        enable_self_collision = solver._static_rigid_sim_config.enable_self_collision
        enable_adjacent_collision = solver._static_rigid_sim_config.enable_adjacent_collision
        batch_links_info = solver._static_rigid_sim_config.batch_links_info

        geoms_link_idx = solver.geoms_info.link_idx.to_numpy()
        geoms_contype = solver.geoms_info.contype.to_numpy()
        geoms_conaffinity = solver.geoms_info.conaffinity.to_numpy()
        links_entity_idx = solver.links_info.entity_idx.to_numpy()
        links_root_idx = solver.links_info.root_idx.to_numpy()
        links_parent_idx = solver.links_info.parent_idx.to_numpy()
        links_is_fixed = solver.links_info.is_fixed.to_numpy()
        if batch_links_info:
            links_entity_idx = links_entity_idx[:, 0]
            links_root_idx = links_root_idx[:, 0]
            links_parent_idx = links_parent_idx[:, 0]
            links_is_fixed = links_is_fixed[:, 0]

        n_possible_pairs = 0
        for i_ga in range(n_geoms):
            for i_gb in range(i_ga + 1, n_geoms):
                i_la = geoms_link_idx[i_ga]
                i_lb = geoms_link_idx[i_gb]

                # geoms in the same link
                if i_la == i_lb:
                    continue

                # self collision
                if links_root_idx[i_la] == links_root_idx[i_lb]:
                    if not enable_self_collision:
                        continue

                    # adjacent links
                    if not enable_adjacent_collision and (
                        links_parent_idx[i_la] == i_lb or links_parent_idx[i_lb] == i_la
                    ):
                        continue

                # contype and conaffinity
                if links_entity_idx[i_la] == links_entity_idx[i_lb] and not (
                    (geoms_contype[i_ga] & geoms_conaffinity[i_gb]) or (geoms_contype[i_gb] & geoms_conaffinity[i_ga])
                ):
                    continue

                # pair of fixed links wrt the world
                if links_is_fixed[i_la] and links_is_fixed[i_lb]:
                    continue

                self.collision_pair_validity[i_ga, i_gb] = 1
                n_possible_pairs += 1

        return n_possible_pairs
