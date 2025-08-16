from typing import TYPE_CHECKING

import numpy as np
import gstaichi as ti

import genesis as gs
import genesis.utils.geom as gu

if TYPE_CHECKING:
    from genesis.engine.solvers.rigid.collider_decomp import Collider
    from genesis.engine.solvers.rigid.rigid_solver_decomp import RigidSolver

INVALID_NEXT_HIBERNATED_ENTITY_IDX = -1


@ti.data_oriented
class ContactIsland:
    def __init__(self, collider: "Collider"):
        self.solver: "RigidSolver" = collider._solver
        self.collider: "Collider" = collider

        struct_agg_list = ti.types.struct(
            curr=gs.ti_int,
            n=gs.ti_int,
            start=gs.ti_int,
        )

        max_contact_pairs = self.collider._collider_info._max_contact_pairs[None]
        max_contact_pairs = max(max_contact_pairs, 1)  # can't create 0-sized fields

        self.ci_edges = ti.field(dtype=gs.ti_int, shape=self.solver._batch_shape((max_contact_pairs, 2)))

        # maps half-edges (half-edges are referenced by entity_edge range) to actual edge index
        # description: half_edge_ref_to_edge_idx
        self.edge_id = ti.field(dtype=gs.ti_int, shape=self.solver._batch_shape((max_contact_pairs * 2)))

        # maps collider_state.contact_data index to island idx
        self.constraint_list = ti.field(dtype=gs.ti_int, shape=self.solver._batch_shape((max_contact_pairs)))

        # analogous to edge_id: maps island's constraint local-index to world's contact index
        self.constraint_id = ti.field(dtype=gs.ti_int, shape=self.solver._batch_shape((max_contact_pairs * 2)))

        # per-entity range of half-edges (indexing into edge_id)
        # description: entity_idx_to_half_edge_ref_range
        self.entity_edge = struct_agg_list.field(
            shape=self.solver._batch_shape(self.solver.n_entities), needs_grad=False, layout=ti.Layout.SOA
        )

        # records number of collision edges per island
        # description: island_idx_to_contact_ref_range
        self.island_col = struct_agg_list.field(
            shape=self.solver._batch_shape(self.solver.n_entities), needs_grad=False, layout=ti.Layout.SOA
        )

        self.island_hibernated = ti.field(dtype=gs.ti_int, shape=self.solver._batch_shape((self.solver.n_entities)))

        # description: island_idx_to_entity_ref_range
        self.island_entity = struct_agg_list.field(
            shape=self.solver._batch_shape(self.solver.n_entities), needs_grad=False, layout=ti.Layout.SOA
        )

        # map per-island entity local-index to world's entity index
        # description: entity_ref_to_entity_idx
        self.entity_id = ti.field(dtype=gs.ti_int, shape=self.solver._batch_shape((self.solver.n_entities)))

        # num all collision edges in world
        self.n_edges = ti.field(dtype=gs.ti_int, shape=self.solver._B)
        self.n_islands = ti.field(dtype=gs.ti_int, shape=self.solver._B)
        self.n_stack = ti.field(dtype=gs.ti_int, shape=self.solver._B)

        # description: entity_idx_to_island_idx
        self.entity_island = ti.field(dtype=gs.ti_int, shape=self.solver._batch_shape(self.solver.n_entities))
        self.stack = ti.field(dtype=gs.ti_int, shape=self.solver._batch_shape(self.solver.n_entities))

        # Used to make islands persist through hibernation:
        self.entity_idx_to_next_entity_idx_in_hibernated_island = ti.field(
            dtype=gs.ti_int, shape=self.solver._batch_shape(self.solver.n_entities)
        )
        self.entity_idx_to_next_entity_idx_in_hibernated_island.fill(INVALID_NEXT_HIBERNATED_ENTITY_IDX)

    @ti.kernel
    def clear(self):
        ti.loop_config(serialize=self.solver._para_level < gs.PARA_LEVEL.ALL)
        for i_e, i_b in ti.ndrange(self.solver.n_entities, self.solver._B):
            self.entity_edge[i_e, i_b].n = 0
            self.island_col[i_e, i_b].n = 0
            self.island_entity[i_e, i_b].n = 0
            self.entity_island[i_e, i_b] = -1

        ti.loop_config(serialize=self.solver._para_level < gs.PARA_LEVEL.ALL)
        for i_b in range(self.solver._B):
            self.n_edges[i_b] = 0
            self.n_islands[i_b] = 0

    @ti.func
    def add_edge(self, link_a, link_b, i_b):
        link_a_maybe_batch = [link_a, i_b] if ti.static(self.solver._options.batch_links_info) else link_a
        link_b_maybe_batch = [link_b, i_b] if ti.static(self.solver._options.batch_links_info) else link_b

        ea = self.solver.links_info.entity_idx[link_a_maybe_batch]
        eb = self.solver.links_info.entity_idx[link_b_maybe_batch]

        # update num edges per entity
        self.entity_edge[ea, i_b].n = self.entity_edge[ea, i_b].n + 1
        self.entity_edge[eb, i_b].n = self.entity_edge[eb, i_b].n + 1

        # fill in collider-info edges with indices to connected entities.
        n_edge = self.n_edges[i_b]
        self.ci_edges[n_edge, 0, i_b] = ea
        self.ci_edges[n_edge, 1, i_b] = eb
        self.n_edges[i_b] = n_edge + 1

    @ti.kernel
    def add_contact_edges_to_islands(self):
        ti.loop_config(serialize=self.solver._para_level < gs.PARA_LEVEL.ALL)
        for i_b in range(self.solver._B):
            for i_col in range(self.collider._collider_state.n_contacts[i_b]):
                # get links indices of the impact
                link_a = self.collider._collider_state.contact_data.link_a[i_col, i_b]
                link_b = self.collider._collider_state.contact_data.link_b[i_col, i_b]
                self.add_edge(link_a, link_b, i_b)

    @ti.kernel
    def add_hiberanted_edges_to_islands(self):
        _B = self.solver._B
        n_entities = self.solver.n_entities
        ti.loop_config(serialize=self.solver._para_level < gs.PARA_LEVEL.ALL)
        for i_b in range(_B):
            for i_e in range(n_entities):
                next_entity_idx = self.entity_idx_to_next_entity_idx_in_hibernated_island[i_e, i_b]
                if next_entity_idx != INVALID_NEXT_HIBERNATED_ENTITY_IDX and next_entity_idx != i_e:
                    any_link_a = self.solver.entities_info.link_start[i_e]
                    any_link_b = self.solver.entities_info.link_start[next_entity_idx]
                    self.add_edge(any_link_a, any_link_b, i_b)

    def construct(self):
        self.clear()
        self.add_contact_edges_to_islands()
        self.add_hiberanted_edges_to_islands()
        self.preprocess_island_and_map_entities_to_edges()
        self.construct_islands()
        self.postprocess_island_and_assign_contact_data()

    @ti.kernel
    def postprocess_island_and_assign_contact_data(self):
        ti.loop_config(serialize=self.solver._para_level < gs.PARA_LEVEL.ALL)
        for i_b in range(self.solver._B):
            for i_col in range(self.collider._collider_state.n_contacts[i_b]):
                # get links indices of the impact
                link_a = self.collider._collider_state.contact_data.link_a[i_col, i_b]
                link_b = self.collider._collider_state.contact_data.link_b[i_col, i_b]
                link_a_maybe_batch = [link_a, i_b] if ti.static(self.solver._options.batch_links_info) else link_a
                link_b_maybe_batch = [link_b, i_b] if ti.static(self.solver._options.batch_links_info) else link_b

                ea = self.solver.links_info.entity_idx[link_a_maybe_batch]
                eb = self.solver.links_info.entity_idx[link_b_maybe_batch]

                island_a = self.entity_island[ea, i_b]
                island_b = self.entity_island[eb, i_b]

                # handle collisions between dynamic and fixed entities (island_idx == -1)
                island = island_a
                if island_a == -1:
                    island = island_b

                self.island_col[island, i_b].n = self.island_col[island, i_b].n + 1
                self.constraint_list[i_col, i_b] = island

            constraint_list_start = 0
            for i in range(self.n_islands[i_b]):
                self.island_col[i, i_b].start = constraint_list_start
                constraint_list_start = constraint_list_start + self.island_col[i, i_b].n
                self.island_col[i, i_b].curr = self.island_col[i, i_b].start

                self.island_hibernated[i, i_b] = 1

            for i_col in range(self.collider._collider_state.n_contacts[i_b]):
                island = self.constraint_list[i_col, i_b]
                self.constraint_id[self.island_col[island, i_b].curr, i_b] = i_col
                self.island_col[island, i_b].curr = self.island_col[island, i_b].curr + 1

            # island_entity
            for i in range(self.solver.n_entities):
                if self.entity_island[i, i_b] >= 0:
                    self.island_entity[self.entity_island[i, i_b], i_b].n = (
                        self.island_entity[self.entity_island[i, i_b], i_b].n + 1
                    )
                    if self.solver.entities_state.hibernated[i, i_b] == 0:
                        self.island_hibernated[self.entity_island[i, i_b], i_b] = 0

            entity_list_start = 0
            for i in range(self.n_islands[i_b]):
                self.island_entity[i, i_b].start = entity_list_start
                self.island_entity[i, i_b].curr = self.island_entity[i, i_b].start
                entity_list_start = entity_list_start + self.island_entity[i, i_b].n

            for i in range(self.solver.n_entities):
                island = self.entity_island[i, i_b]
                if island >= 0:
                    self.entity_id[self.island_entity[island, i_b].curr, i_b] = i
                    self.island_entity[island, i_b].curr = self.island_entity[island, i_b].curr + 1

    @ti.kernel
    def preprocess_island_and_map_entities_to_edges(self):
        ti.loop_config(serialize=self.solver._para_level < gs.PARA_LEVEL.ALL)
        for i_b in range(self.solver._B):
            entity_list_start = 0
            for i in range(self.solver.n_entities):
                self.entity_edge[i, i_b].start = entity_list_start
                self.entity_edge[i, i_b].curr = entity_list_start
                entity_list_start = entity_list_start + self.entity_edge[i, i_b].n

            # process added collider-info edges
            for i in range(self.n_edges[i_b]):
                ea = self.ci_edges[i, 0, i_b]
                eb = self.ci_edges[i, 1, i_b]

                # map entity's half-edge index to edge index.
                self.edge_id[self.entity_edge[ea, i_b].curr, i_b] = i
                self.edge_id[self.entity_edge[eb, i_b].curr, i_b] = i

                self.entity_edge[ea, i_b].curr = self.entity_edge[ea, i_b].curr + 1
                self.entity_edge[eb, i_b].curr = self.entity_edge[eb, i_b].curr + 1

    @ti.kernel
    def construct_islands(self):
        """
        This assigns entities to islands, by setting their entity_island[entity_idx, batch_idx] = island_idx.
        """
        ti.loop_config(serialize=self.solver._para_level < gs.PARA_LEVEL.ALL)
        for i_b in range(self.solver._B):
            for i_v in range(self.solver.n_entities):
                # only create islands for entities with collisions and with dofs
                if self.entity_edge[i_v, i_b].n > 0 and self.solver.entities_info.n_dofs[i_v] > 0:
                    if self.entity_island[i_v, i_b] != -1:
                        continue
                    self.n_stack[i_b] = 0
                    self.stack[self.n_stack[i_b], i_b] = i_v
                    self.n_stack[i_b] = self.n_stack[i_b] + 1
                    self.entity_island[i_v, i_b] = self.n_islands[i_b]
                    # FIXME: Add proper mechanism to detection overflow in Taichi-scope
                    # but raise exception in Python-scope

                    while self.n_stack[i_b] > 0:
                        self.n_stack[i_b] = self.n_stack[i_b] - 1
                        v = self.stack[self.n_stack[i_b], i_b]

                        for i_edge in range(self.entity_edge[v, i_b].n):
                            _id = self.entity_edge[v, i_b].start + i_edge  # half-edge index
                            edge = self.edge_id[_id, i_b]  # edge index
                            next_v = self.ci_edges[edge, 0, i_b]  # other entity index, connected by edge
                            if next_v == v:
                                next_v = self.ci_edges[edge, 1, i_b]

                            if (
                                self.solver.entities_info.n_dofs[next_v] > 0
                                and next_v != v
                                and self.entity_island[next_v, i_b] == -1
                            ):  # 2nd condition must not happen ?
                                self.stack[self.n_stack[i_b], i_b] = next_v
                                self.n_stack[i_b] = self.n_stack[i_b] + 1
                                self.entity_island[next_v, i_b] = self.n_islands[i_b]
                                # FIXME: Add proper mechanism to detection overflow in Taichi-scope
                                # but raise exception in Python-scope

                    self.n_islands[i_b] = self.n_islands[i_b] + 1

        # create single-entity islands for entities without collisions
        if self.solver._enable_joint_limit:
            ti.loop_config(serialize=self.solver._para_level < gs.PARA_LEVEL.ALL)
            for i_b in range(self.solver._B):
                for i_v in range(self.solver.n_entities):
                    if self.solver.entities_info.n_dofs[i_v] > 0 and self.entity_island[i_v, i_b] == -1:
                        self.entity_island[i_v, i_b] = self.n_islands[i_b]
                        self.n_islands[i_b] = self.n_islands[i_b] + 1
