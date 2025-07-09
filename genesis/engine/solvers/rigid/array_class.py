from dataclasses import dataclass
from typing import Callable

import taichi as ti

import genesis as gs

# we will use struct for DofsState and DofsInfo after Hugh adds array_struct feature to taichi
DofsState = ti.template()
DofsInfo = ti.template()
GeomsState = ti.template()
GeomsInfo = ti.template()
LinksState = ti.template()
LinksInfo = ti.template()


@ti.data_oriented
class RigidGlobalInfo:
    def __init__(self, n_dofs: int, n_entities: int, n_geoms: int, f_batch: Callable):
        self.n_awake_dofs = ti.field(dtype=gs.ti_int, shape=f_batch())
        self.awake_dofs = ti.field(dtype=gs.ti_int, shape=f_batch(n_dofs))


# =========================================== Collider ===========================================


@ti.data_oriented
class ColliderGlobalInfo:
    def __init__(self, solver):
        _B = solver._B
        f_batch = solver._batch_shape
        n_geoms = solver.n_geoms_
        max_collision_pairs = solver._max_collision_pairs
        use_hibernation = solver._static_rigid_sim_config.use_hibernation

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
        n_possible_pairs = self.init_collision_pair_validity(solver)

        # Whether or not this is the first time to run the broad phase for each batch
        self.first_time = ti.field(gs.ti_int, shape=_B)

        # Number of possible pairs of collision, store them in a field to avoid recompilation
        self._n_contacts_per_pair = ti.field(dtype=gs.ti_int, shape=())
        self._max_possible_pairs = ti.field(dtype=gs.ti_int, shape=())
        self._max_collision_pairs = ti.field(dtype=gs.ti_int, shape=())
        self._max_contact_pairs = ti.field(dtype=gs.ti_int, shape=())

        self._n_contacts_per_pair[None] = 5     # CONSTANT. CANNOT NOT BE CHANGED.
        self._max_possible_pairs[None] = n_possible_pairs
        self._max_collision_pairs[None] = min(n_possible_pairs, max_collision_pairs)
        self._max_contact_pairs[None] = self._max_collision_pairs[None] * self._n_contacts_per_pair[None]

        # Final results of the broad phase
        self.n_broad_pairs = ti.field(dtype=gs.ti_int, shape=_B)
        self.broad_collision_pairs = ti.Vector.field(
            2, dtype=gs.ti_int, shape=f_batch(max(1, self._max_collision_pairs[None]))
        )


    def init_collision_pair_validity(self, solver):
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