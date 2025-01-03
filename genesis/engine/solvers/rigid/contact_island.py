import numpy as np
import taichi as ti

import genesis as gs
import genesis.utils.geom as gu


@ti.data_oriented
class ContactIsland:
    def __init__(self, collider):
        self.solver = collider._solver
        self.collider = collider

        struct_agg_list = ti.types.struct(
            curr=gs.ti_int,
            n=gs.ti_int,
            start=gs.ti_int,
        )

        self.ci_edges = ti.field(dtype=gs.ti_int, shape=self.solver._batch_shape((self.collider._max_contact_pairs, 2)))

        self.edge_id = ti.field(dtype=gs.ti_int, shape=self.solver._batch_shape((self.collider._max_contact_pairs * 2)))

        self.constraint_list = ti.field(
            dtype=gs.ti_int, shape=self.solver._batch_shape((self.collider._max_contact_pairs))
        )

        self.constraint_id = ti.field(
            dtype=gs.ti_int, shape=self.solver._batch_shape((self.collider._max_contact_pairs * 2))
        )

        self.entity_edge = struct_agg_list.field(
            shape=self.solver._batch_shape(self.solver.n_entities), needs_grad=False, layout=ti.Layout.SOA
        )

        self.island_col = struct_agg_list.field(
            shape=self.solver._batch_shape(self.solver.n_entities), needs_grad=False, layout=ti.Layout.SOA
        )

        self.island_hibernated = ti.field(dtype=gs.ti_int, shape=self.solver._batch_shape((self.solver.n_entities)))

        self.island_entity = struct_agg_list.field(
            shape=self.solver._batch_shape(self.solver.n_entities), needs_grad=False, layout=ti.Layout.SOA
        )

        self.entity_id = ti.field(dtype=gs.ti_int, shape=self.solver._batch_shape((self.solver.n_entities)))

        self.n_edge = ti.field(dtype=gs.ti_int, shape=self.solver._B)
        self.n_island = ti.field(dtype=gs.ti_int, shape=self.solver._B)
        self.n_stack = ti.field(dtype=gs.ti_int, shape=self.solver._B)

        self.entity_island = ti.field(dtype=gs.ti_int, shape=self.solver._batch_shape(self.solver.n_entities))
        self.stack = ti.field(dtype=gs.ti_int, shape=self.solver._batch_shape(self.solver.n_entities))

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
            self.n_edge[i_b] = 0
            self.n_island[i_b] = 0

    @ti.func
    def add_edge(self, link_a, link_b, i_b):
        link_a_maybe_batch = [link_a, i_b] if ti.static(self.solver._options.batch_links_info) else link_a
        link_b_maybe_batch = [link_b, i_b] if ti.static(self.solver._options.batch_links_info) else link_b

        ea = self.solver.links_info[link_a_maybe_batch].entity_idx
        eb = self.solver.links_info[link_b_maybe_batch].entity_idx

        self.entity_edge[ea, i_b].n = self.entity_edge[ea, i_b].n + 1
        self.entity_edge[eb, i_b].n = self.entity_edge[eb, i_b].n + 1

        n_edge = self.n_edge[i_b]
        self.ci_edges[n_edge, 0, i_b] = ea
        self.ci_edges[n_edge, 1, i_b] = eb
        self.n_edge[i_b] = n_edge + 1

    @ti.kernel
    def add_island(self):
        ti.loop_config(serialize=self.solver._para_level < gs.PARA_LEVEL.ALL)
        for i_b in range(self.solver._B):
            for i_col in range(self.collider.n_contacts[i_b]):
                impact = self.collider.contact_data[i_col, i_b]
                self.add_edge(impact.link_a, impact.link_b, i_b)

    def construct(self):
        self.clear()
        self.add_island()
        self.preprocess_island()
        self.construct_island()
        self.postprocess_island()

    @ti.kernel
    def postprocess_island(self):
        ti.loop_config(serialize=self.solver._para_level < gs.PARA_LEVEL.ALL)
        for i_b in range(self.solver._B):
            for i_col in range(self.collider.n_contacts[i_b]):
                impact = self.collider.contact_data[i_col, i_b]
                link_a = impact.link_a
                link_b = impact.link_b
                link_a_maybe_batch = [link_a, i_b] if ti.static(self.solver._options.batch_links_info) else link_a
                link_b_maybe_batch = [link_b, i_b] if ti.static(self.solver._options.batch_links_info) else link_b

                ea = self.solver.links_info[link_a_maybe_batch].entity_idx
                eb = self.solver.links_info[link_b_maybe_batch].entity_idx

                island_a = self.entity_island[ea, i_b]
                island_b = self.entity_island[eb, i_b]

                island = island_a
                if island_a == -1:
                    island = island_b

                self.island_col[island, i_b].n = self.island_col[island, i_b].n + 1
                self.constraint_list[i_col, i_b] = island

            constraint_list_start = 0
            for i in range(self.n_island[i_b]):
                self.island_col[i, i_b].start = constraint_list_start
                constraint_list_start = constraint_list_start + self.island_col[i, i_b].n
                self.island_col[i, i_b].curr = self.island_col[i, i_b].start

                self.island_hibernated[i, i_b] = 1

            for i_col in range(self.collider.n_contacts[i_b]):
                island = self.constraint_list[i_col, i_b]
                self.constraint_id[self.island_col[island, i_b].curr, i_b] = i_col
                self.island_col[island, i_b].curr = self.island_col[island, i_b].curr + 1

            # island_entity
            for i in range(self.solver.n_entities):
                if self.entity_island[i, i_b] >= 0:
                    self.island_entity[self.entity_island[i, i_b], i_b].n = (
                        self.island_entity[self.entity_island[i, i_b], i_b].n + 1
                    )
                    if self.solver.entities_state[i, i_b].hibernated == 0:
                        self.island_hibernated[self.entity_island[i, i_b], i_b] = 0

            entity_list_start = 0
            for i in range(self.n_island[i_b]):
                self.island_entity[i, i_b].start = entity_list_start
                self.island_entity[i, i_b].curr = self.island_entity[i, i_b].start
                entity_list_start = entity_list_start + self.island_entity[i, i_b].n

            for i in range(self.solver.n_entities):
                island = self.entity_island[i, i_b]
                if island >= 0:
                    self.entity_id[self.island_entity[island, i_b].curr, i_b] = i
                    self.island_entity[island, i_b].curr = self.island_entity[island, i_b].curr + 1

    @ti.kernel
    def preprocess_island(self):
        ti.loop_config(serialize=self.solver._para_level < gs.PARA_LEVEL.ALL)
        for i_b in range(self.solver._B):
            entity_list_start = 0
            for i in range(self.solver.n_entities):
                self.entity_edge[i, i_b].start = entity_list_start
                entity_list_start = entity_list_start + self.entity_edge[i, i_b].n
                self.entity_edge[i, i_b].curr = self.entity_edge[i, i_b].start

            for i in range(self.n_edge[i_b]):
                ea = self.ci_edges[i, 0, i_b]
                eb = self.ci_edges[i, 1, i_b]

                self.edge_id[self.entity_edge[ea, i_b].curr, i_b] = i
                self.edge_id[self.entity_edge[eb, i_b].curr, i_b] = i

                self.entity_edge[ea, i_b].curr = self.entity_edge[ea, i_b].curr + 1
                self.entity_edge[eb, i_b].curr = self.entity_edge[eb, i_b].curr + 1

    @ti.kernel
    def construct_island(self):
        ti.loop_config(serialize=self.solver._para_level < gs.PARA_LEVEL.ALL)
        for i_b in range(self.solver._B):
            for i_v in range(self.solver.n_entities):
                if self.entity_edge[i_v, i_b].n > 0 and self.solver.entities_info[i_v].n_dofs > 0:
                    if self.entity_island[i_v, i_b] != -1:
                        continue
                    self.n_stack[i_b] = 0
                    self.stack[self.n_stack[i_b], i_b] = i_v
                    self.n_stack[i_b] = self.n_stack[i_b] + 1

                    while self.n_stack[i_b] > 0:
                        self.n_stack[i_b] = self.n_stack[i_b] - 1
                        v = self.stack[self.n_stack[i_b], i_b]
                        if self.entity_island[v, i_b] != -1:
                            continue
                        self.entity_island[v, i_b] = self.n_island[i_b]

                        for i_edge in range(self.entity_edge[v, i_b].n):
                            _id = self.entity_edge[v, i_b].start + i_edge
                            edge = self.edge_id[_id, i_b]
                            next_v = self.ci_edges[edge, 0, i_b]
                            if next_v == v:
                                next_v = self.ci_edges[edge, 1, i_b]

                            if self.solver.entities_info[next_v].n_dofs > 0 and next_v != v:
                                self.stack[self.n_stack[i_b], i_b] = next_v
                                self.n_stack[i_b] = self.n_stack[i_b] + 1

                    self.n_island[i_b] = self.n_island[i_b] + 1

        if ti.static(self.solver._enable_joint_limit):
            ti.loop_config(serialize=self.solver._para_level < gs.PARA_LEVEL.ALL)
            for i_b in range(self.solver._B):
                for i_v in range(self.solver.n_entities):
                    if self.solver.entities_info[i_v].n_dofs > 0 and self.entity_island[i_v, i_b] == -1:
                        self.entity_island[i_v, i_b] = self.n_island[i_b]
                        self.n_island[i_b] = self.n_island[i_b] + 1
