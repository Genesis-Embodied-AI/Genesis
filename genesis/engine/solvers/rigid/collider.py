import numpy as np
import taichi as ti

import genesis as gs
import genesis.utils.geom as gu


@ti.data_oriented
class Collider:
    def __init__(self, rigid_solver):
        self._solver = rigid_solver
        self._init_verts_connectivity()
        self._init_collision_fields()

        # multi contact perturbation and tolerance
        self._mc_perturbation = 1e-3
        self._mc_tolerance = 5e-3  # 5mm

    def _init_verts_connectivity(self):
        vert_neighbors = []
        vert_neighbor_start = []
        vert_n_neighbors = []
        offset = 0
        for geom in self._solver.geoms:
            vert_neighbors.append(geom.vert_neighbors + geom.vert_start)
            vert_neighbor_start.append(geom.vert_neighbor_start + offset)
            vert_n_neighbors.append(geom.vert_n_neighbors)
            offset += len(geom.vert_neighbors)

        if self._solver.n_verts > 0:
            vert_neighbors = np.concatenate(vert_neighbors, dtype=gs.np_int)
            vert_neighbor_start = np.concatenate(vert_neighbor_start, dtype=gs.np_int)
            vert_n_neighbors = np.concatenate(vert_n_neighbors, dtype=gs.np_int)

        self.vert_neighbors = ti.field(dtype=gs.ti_int, shape=max(1, len(vert_neighbors)))
        self.vert_neighbor_start = ti.field(dtype=gs.ti_int, shape=self._solver.n_verts_)
        self.vert_n_neighbors = ti.field(dtype=gs.ti_int, shape=self._solver.n_verts_)

        if self._solver.n_verts > 0:
            self.vert_neighbors.from_numpy(vert_neighbors)
            self.vert_neighbor_start.from_numpy(vert_neighbor_start)
            self.vert_n_neighbors.from_numpy(vert_n_neighbors)

    def _init_collision_fields(self):
        # compute collision pairs
        # convert to numpy array for faster retrieval
        geoms_link_idx = self._solver.geoms_info.link_idx.to_numpy()
        links_root_idx = self._solver.links_info.root_idx.to_numpy()
        links_parent_idx = self._solver.links_info.parent_idx.to_numpy()
        links_is_fixed = self._solver.links_info.is_fixed.to_numpy()
        n_possible_pairs = 0
        for i in range(self._solver.n_geoms):
            for j in range(i + 1, self._solver.n_geoms):
                i_la = geoms_link_idx[i]
                i_lb = geoms_link_idx[j]

                # geoms in the same link
                if i_la == i_lb:
                    continue

                # self collision
                if not self._solver._enable_self_collision and links_root_idx[i_la] == links_root_idx[i_lb]:
                    continue

                # adjacent links
                if links_parent_idx[i_la] == i_lb or links_parent_idx[i_lb] == i_la:
                    continue

                # pair of fixed base links
                if links_is_fixed[i_la] and links_is_fixed[i_lb]:
                    continue

                n_possible_pairs += 1

        self._n_contacts_per_pair = 4
        self._max_collision_pairs = min(n_possible_pairs, self._solver._max_collision_pairs)
        self._max_contact_pairs = self._max_collision_pairs * self._n_contacts_per_pair

        ############## broad phase SAP ##############
        # This buffer stores the AABBs along the search axis of all geoms
        struct_sort_buffer = ti.types.struct(value=gs.ti_float, i_g=gs.ti_int, is_max=gs.ti_int)
        self.sort_buffer = struct_sort_buffer.field(
            shape=self._solver._batch_shape(2 * self._solver.n_geoms_),
            layout=ti.Layout.SOA,
        )
        # This buffer stores indexes of active geoms during SAP search
        if self._solver._use_hibernation:
            self.active_buffer_awake = ti.field(dtype=gs.ti_int, shape=self._solver._batch_shape(self._solver.n_geoms_))
            self.active_buffer_hib = ti.field(dtype=gs.ti_int, shape=self._solver._batch_shape(self._solver.n_geoms_))
        self.active_buffer = ti.field(dtype=gs.ti_int, shape=self._solver._batch_shape(self._solver.n_geoms_))

        self.n_broad_pairs = ti.field(dtype=gs.ti_int, shape=self._solver._B)
        self.broad_collision_pairs = ti.Vector.field(
            2, dtype=gs.ti_int, shape=self._solver._batch_shape(max(1, self._max_collision_pairs))
        )

        self.first_time = ti.field(gs.ti_int, shape=())

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
            shape=self._solver._batch_shape(max(1, self._max_contact_pairs)),
            layout=ti.Layout.SOA,
        )
        self.n_contacts = ti.field(
            gs.ti_int, shape=self._solver._B
        )  # total number of contacts, including hibernated contacts
        self.n_contacts_hibernated = ti.field(gs.ti_int, shape=self._solver._B)

        # contact caching for warmstart collision detection
        struct_contact_cache = ti.types.struct(
            i_va_0=gs.ti_int,
            penetration=gs.ti_float,
            normal=gs.ti_vec3,
        )
        self.contact_cache = struct_contact_cache.field(
            shape=self._solver._batch_shape((self._solver.n_geoms_, self._solver.n_geoms_)),
            layout=ti.Layout.SOA,
        )

        # for faster compilation
        self._has_nonconvex_nonterrain = np.logical_and(
            self._solver.geoms_info.is_convex.to_numpy() == 0,
            self._solver.geoms_info.type.to_numpy() != gs.GEOM_TYPE.TERRAIN,
        ).any()
        self._has_sphere = (self._solver.geoms_info.type.to_numpy() == gs.GEOM_TYPE.SPHERE).any()
        self._has_terrain = (self._solver.geoms_info.type.to_numpy() == gs.GEOM_TYPE.TERRAIN).any()

        self.reset()

    def reset(self):
        self.first_time.fill(1)
        self.contact_cache.i_va_0.fill(-1)
        self.contact_cache.penetration.fill(0)
        self.contact_cache.normal.fill(0)

    @ti.func
    def clear(self):
        ti.loop_config(serialize=self._solver._para_level < gs.PARA_LEVEL.ALL)
        for i_b in range(self._solver._B):
            # self.n_contacts_hibernated[i_b] = 0

            # if ti.static(self._solver._use_hibernation):
            #     # advect hibernated contacts
            #     for i_c in range(self.n_contacts[i_b]):
            #         i_la = self.contact_data[i_c, i_b].link_a
            #         i_lb = self.contact_data[i_c, i_b].link_b

            #         # pair of hibernated-fixed links -> hibernated contact
            #         # TODO: we should also include hibernated-hibernated links and wake up the whole contact island once a new collision is detected
            #         if (self._solver.links_state[i_la, i_b].hibernated and self._solver.links_info[i_lb].is_fixed) or \
            #             (self._solver.links_state[i_lb, i_b].hibernated and self._solver.links_info[i_la].is_fixed):
            #             i_c_hibernated = self.n_contacts_hibernated[i_b]
            #             if i_c != i_c_hibernated:
            #                 self.contact_data[i_c_hibernated, i_b] = self.contact_data[i_c, i_b]
            #             self.n_contacts_hibernated[i_b] = i_c_hibernated + 1

            # self.n_contacts[i_b] = self.n_contacts_hibernated[i_b]

            self.n_contacts[i_b] = 0

    @ti.func
    def detection(self):
        self._func_update_aabbs()
        self._func_broad_phase()
        self._func_narrow_phase()

    @ti.func
    def _func_point_in_geom_aabb(self, point, i_g, i_b):
        return (point < self._solver.geoms_state[i_g, i_b].aabb_max).all() and (
            point > self._solver.geoms_state[i_g, i_b].aabb_min
        ).all()

    @ti.func
    def _func_is_geom_aabbs_overlap(self, i_ga, i_gb, i_b):
        return not (
            (self._solver.geoms_state[i_ga, i_b].aabb_max <= self._solver.geoms_state[i_gb, i_b].aabb_min).any()
            or (self._solver.geoms_state[i_ga, i_b].aabb_min >= self._solver.geoms_state[i_gb, i_b].aabb_max).any()
        )

    @ti.func
    def _func_find_intersect_midpoint(self, i_ga, i_gb):
        # return the center of the intersecting AABB of AABBs of two geoms
        intersect_lower = ti.max(self._solver.geoms_state[i_ga].aabb_min, self._solver.geoms_state[i_gb].aabb_min)
        intersect_upper = ti.min(self._solver.geoms_state[i_ga].aabb_max, self._solver.geoms_state[i_gb].aabb_max)
        return 0.5 * (intersect_lower + intersect_upper)

    @ti.func
    def _func_contact_sphere_sdf(self, i_ga, i_gb, i_b):
        ga_info = self._solver.geoms_info[i_ga]
        is_col = gs.ti_int(0)
        penetration = gs.ti_float(0.0)
        normal = ti.Vector.zero(gs.ti_float, 3)
        contact_pos = ti.Vector.zero(gs.ti_float, 3)

        sphere_center = self._solver.geoms_state[i_ga, i_b].pos
        sphere_radius = ga_info.data[0]

        center_to_b_dist = self._solver.sdf.sdf_world(sphere_center, i_gb, i_b)
        if center_to_b_dist < sphere_radius:
            is_col = 1
            normal = self._solver.sdf.sdf_normal_world(sphere_center, i_gb, i_b)
            contact_pos = sphere_center - sphere_radius * normal
            penetration = sphere_radius - center_to_b_dist

        return is_col, normal, penetration, contact_pos

    @ti.func
    def _func_contact_vertex_sdf(self, i_ga, i_gb, i_b, ga_quat_rot):
        ga_info = self._solver.geoms_info[i_ga]
        ga_pos = self._solver.geoms_state[i_ga, i_b].pos
        ga_quat = self._solver.geoms_state[i_ga, i_b].quat

        # rotate ga with small purturbation
        ga_quat = gu.ti_transform_quat_by_quat(ga_quat, ga_quat_rot)

        is_col = gs.ti_int(0)
        penetration = gs.ti_float(0.0)
        normal = ti.Vector.zero(gs.ti_float, 3)
        contact_pos = ti.Vector.zero(gs.ti_float, 3)

        for i_v in range(ga_info.vert_start, ga_info.vert_end):
            p = gu.ti_transform_by_trans_quat(self._solver.verts_info[i_v].init_pos, ga_pos, ga_quat)
            if self._func_point_in_geom_aabb(p, i_gb, i_b):
                new_penetration = -self._solver.sdf.sdf_world(p, i_gb, i_b)

                if new_penetration > penetration:
                    is_col = 1
                    normal = self._solver.sdf.sdf_normal_world(p, i_gb, i_b)
                    contact_pos = p
                    penetration = new_penetration

        return is_col, normal, penetration, contact_pos

    @ti.func
    def _func_contact_edge_sdf(self, i_ga, i_gb, i_b):
        ga_info = self._solver.geoms_info[i_ga]
        ga_state = self._solver.geoms_state[i_ga, i_b]

        is_col = gs.ti_int(0)
        penetration = gs.ti_float(0.0)
        normal = ti.Vector.zero(gs.ti_float, 3)
        contact_pos = ti.Vector.zero(gs.ti_float, 3)

        ga_sdf_cell_size = self._solver.sdf.geoms_info[i_ga].sdf_cell_size

        for i_e in range(ga_info.edge_start, ga_info.edge_end):
            cur_length = self._solver.edges_info[i_e].length
            if cur_length > ga_sdf_cell_size:

                i_v0 = self._solver.edges_info[i_e].v0
                i_v1 = self._solver.edges_info[i_e].v1

                p_0 = gu.ti_transform_by_trans_quat(self._solver.verts_info[i_v0].init_pos, ga_state.pos, ga_state.quat)
                p_1 = gu.ti_transform_by_trans_quat(self._solver.verts_info[i_v1].init_pos, ga_state.pos, ga_state.quat)
                vec_01 = gu.ti_normalize(p_1 - p_0)

                sdf_grad_0_b = self._solver.sdf.sdf_grad_world(p_0, i_gb, i_b)
                sdf_grad_1_b = self._solver.sdf.sdf_grad_world(p_1, i_gb, i_b)

                # check if the edge on a is facing towards mesh b (I am not 100% sure about this, subject to removal)
                sdf_grad_0_a = self._solver.sdf.sdf_grad_world(p_0, i_ga, i_b)
                sdf_grad_1_a = self._solver.sdf.sdf_grad_world(p_1, i_ga, i_b)
                normal_edge_0 = sdf_grad_0_a - sdf_grad_0_a.dot(vec_01) * vec_01
                normal_edge_1 = sdf_grad_1_a - sdf_grad_1_a.dot(vec_01) * vec_01

                if normal_edge_0.dot(sdf_grad_0_b) < 0 or normal_edge_1.dot(sdf_grad_1_b) < 0:

                    # check if closest point is between the two points
                    if sdf_grad_0_b.dot(vec_01) < 0 and sdf_grad_1_b.dot(vec_01) > 0:

                        while cur_length > ga_sdf_cell_size:
                            p_mid = 0.5 * (p_0 + p_1)
                            if self._solver.sdf.sdf_grad_world(p_mid, i_gb, i_b).dot(vec_01) < 0:
                                p_0 = p_mid
                            else:
                                p_1 = p_mid
                            cur_length = 0.5 * cur_length

                        p = 0.5 * (p_0 + p_1)
                        new_penetration = -self._solver.sdf.sdf_world(p, i_gb, i_b)

                        if new_penetration > penetration:
                            is_col = 1
                            normal = self._solver.sdf.sdf_normal_world(p, i_gb, i_b)
                            contact_pos = p
                            penetration = new_penetration

        return is_col, normal, penetration, contact_pos

    @ti.func
    def _func_contact_convex_convex(self, i_ga, i_gb, i_b, ga_quat_rot, i_va_ws):
        gb_vert_start = self._solver.geoms_info[i_gb].vert_start
        ga_pos = self._solver.geoms_state[i_ga, i_b].pos
        ga_quat = self._solver.geoms_state[i_ga, i_b].quat
        gb_pos = self._solver.geoms_state[i_gb, i_b].pos
        gb_quat = self._solver.geoms_state[i_gb, i_b].quat

        # rotate ga with small purturbation
        ga_quat = gu.ti_transform_quat_by_quat(ga_quat, ga_quat_rot)

        is_col = gs.ti_int(0)
        penetration = gs.ti_float(0.0)
        normal = ti.Vector.zero(gs.ti_float, 3)
        contact_pos = ti.Vector.zero(gs.ti_float, 3)

        i_va = i_va_ws
        if i_va == -1:
            # start traversing on the vertex graph with a smart initial vertex
            pos_vb = gu.ti_transform_by_trans_quat(self._solver.verts_info[gb_vert_start].init_pos, gb_pos, gb_quat)
            i_va = self._solver.sdf._func_find_closest_vert(pos_vb, i_ga, i_b)
        i_v_closest = i_va
        pos_v_closest = gu.ti_transform_by_trans_quat(self._solver.verts_info[i_v_closest].init_pos, ga_pos, ga_quat)
        sd_v_closest = self._solver.sdf.sdf_world(pos_v_closest, i_gb, i_b)

        while True:
            for i_neighbor_ in range(
                self.vert_neighbor_start[i_va], self.vert_neighbor_start[i_va] + self.vert_n_neighbors[i_va]
            ):
                i_neighbor = self.vert_neighbors[i_neighbor_]
                pos_neighbor = gu.ti_transform_by_trans_quat(
                    self._solver.verts_info[i_neighbor].init_pos, ga_pos, ga_quat
                )
                sd_neighbor = self._solver.sdf.sdf_world(pos_neighbor, i_gb, i_b)
                if (
                    sd_neighbor < sd_v_closest - 1e-5
                ):  # 1e-5 (0.01mm) to avoid endless loop due to numerical instability
                    i_v_closest = i_neighbor
                    sd_v_closest = sd_neighbor
                    pos_v_closest = pos_neighbor

            if i_v_closest == i_va:  # no better neighbor
                break
            else:
                i_va = i_v_closest

        # i_va is the deepest vertex
        pos_a = pos_v_closest
        if sd_v_closest < 0:
            is_col = 1
            normal = self._solver.sdf.sdf_normal_world(pos_a, i_gb, i_b)
            contact_pos = pos_a
            penetration = -sd_v_closest

        else:  # check edge surrounding it
            for i_neighbor_ in range(
                self.vert_neighbor_start[i_va], self.vert_neighbor_start[i_va] + self.vert_n_neighbors[i_va]
            ):
                i_neighbor = self.vert_neighbors[i_neighbor_]

                p_0 = pos_v_closest
                p_1 = gu.ti_transform_by_trans_quat(self._solver.verts_info[i_neighbor].init_pos, ga_pos, ga_quat)
                vec_01 = gu.ti_normalize(p_1 - p_0)

                sdf_grad_0_b = self._solver.sdf.sdf_grad_world(p_0, i_gb, i_b)
                sdf_grad_1_b = self._solver.sdf.sdf_grad_world(p_1, i_gb, i_b)

                # check if the edge on a is facing towards mesh b (I am not 100% sure about this, subject to removal)
                sdf_grad_0_a = self._solver.sdf.sdf_grad_world(p_0, i_ga, i_b)
                sdf_grad_1_a = self._solver.sdf.sdf_grad_world(p_1, i_ga, i_b)
                normal_edge_0 = sdf_grad_0_a - sdf_grad_0_a.dot(vec_01) * vec_01
                normal_edge_1 = sdf_grad_1_a - sdf_grad_1_a.dot(vec_01) * vec_01

                if normal_edge_0.dot(sdf_grad_0_b) < 0 or normal_edge_1.dot(sdf_grad_1_b) < 0:

                    # check if closest point is between the two points
                    if sdf_grad_0_b.dot(vec_01) < 0 and sdf_grad_1_b.dot(vec_01) > 0:

                        cur_length = (p_1 - p_0).norm()
                        ga_sdf_cell_size = self._solver.sdf.geoms_info[i_ga].sdf_cell_size
                        while cur_length > ga_sdf_cell_size:
                            p_mid = 0.5 * (p_0 + p_1)
                            if self._solver.sdf.sdf_grad_world(p_mid, i_gb, i_b).dot(vec_01) < 0:
                                p_0 = p_mid
                            else:
                                p_1 = p_mid

                            cur_length = 0.5 * cur_length

                        p = 0.5 * (p_0 + p_1)

                        new_penetration = -self._solver.sdf.sdf_world(p, i_gb, i_b)

                        if new_penetration > 0:
                            is_col = 1
                            normal = self._solver.sdf.sdf_normal_world(p, i_gb, i_b)
                            contact_pos = p
                            penetration = new_penetration
                            break

        return is_col, normal, penetration, contact_pos, i_va

    @ti.func
    def _func_contact_convex_terrain(self, i_ga, i_gb, i_b, ga_quat_rot, i_va_ws):
        ga_pos = self._solver.geoms_state[i_ga, i_b].pos
        ga_quat = self._solver.geoms_state[i_ga, i_b].quat

        # rotate ga with small purturbation
        ga_quat = gu.ti_transform_quat_by_quat(ga_quat, ga_quat_rot)

        is_col = gs.ti_int(0)
        penetration = gs.ti_float(0.0)
        normal = ti.Vector.zero(gs.ti_float, 3)
        contact_pos = ti.Vector.zero(gs.ti_float, 3)

        i_va = i_va_ws
        if i_va == -1:
            # start traversing on the vertex graph with a smart initial vertex
            # NOTE: we assume the terrain's feature unit is bigger than the mesh.
            pos_vb = ti.Vector([0, 0, -1e6])  # deep enough
            i_va = self._solver.sdf._func_find_closest_vert(pos_vb, i_ga, i_b)
        i_v_closest = i_va
        pos_v_closest = gu.ti_transform_by_trans_quat(self._solver.verts_info[i_v_closest].init_pos, ga_pos, ga_quat)
        sd_v_closest = self._solver.sdf.sdf_world(pos_v_closest, i_gb, i_b)

        while True:
            for i_neighbor_ in range(
                self.vert_neighbor_start[i_va], self.vert_neighbor_start[i_va] + self.vert_n_neighbors[i_va]
            ):
                i_neighbor = self.vert_neighbors[i_neighbor_]
                pos_neighbor = gu.ti_transform_by_trans_quat(
                    self._solver.verts_info[i_neighbor].init_pos, ga_pos, ga_quat
                )
                sd_neighbor = self._solver.sdf.sdf_world(pos_neighbor, i_gb, i_b)
                if (
                    sd_neighbor < sd_v_closest - 1e-5
                ):  # 1e-5 (0.01mm) to avoid endless loop due to numerical instability
                    i_v_closest = i_neighbor
                    sd_v_closest = sd_neighbor
                    pos_v_closest = pos_neighbor

            if i_v_closest == i_va:  # no better neighbor
                break
            else:
                i_va = i_v_closest

        # i_va is the deepest vertex
        pos_a = pos_v_closest
        if sd_v_closest < 0:
            is_col = 1
            normal = self._solver.sdf.sdf_normal_world(pos_a, i_gb, i_b)
            contact_pos = pos_a
            penetration = -sd_v_closest

        # # check edge surrounding it
        # for i_neighbor_ in range(self.vert_neighbor_start[i_va], self.vert_neighbor_start[i_va]+self.vert_n_neighbors[i_va]):
        #     i_neighbor = self.vert_neighbors[i_neighbor_]

        #     p_0 = pos_a
        #     p_1 = gu.ti_transform_by_trans_quat(
        #         self._solver.verts_info[i_neighbor].init_pos,
        #         ga_pos, ga_quat
        #     )
        #     vec_01 = gu.ti_normalize(p_1 - p_0)

        #     sdf_grad_0_b = self._solver.sdf.sdf_grad_world(p_0, i_gb, i_b)
        #     sdf_grad_1_b = self._solver.sdf.sdf_grad_world(p_1, i_gb, i_b)

        #     # check if the edge on a is facing towards mesh b (I am not 100% sure about this, subject to removal)
        #     sdf_grad_0_a = self._solver.sdf.sdf_grad_world(p_0, i_ga, i_b)
        #     sdf_grad_1_a = self._solver.sdf.sdf_grad_world(p_1, i_ga, i_b)
        #     normal_edge_0 = sdf_grad_0_a - sdf_grad_0_a.dot(vec_01) * vec_01
        #     normal_edge_1 = sdf_grad_1_a - sdf_grad_1_a.dot(vec_01) * vec_01

        #     if normal_edge_0.dot(sdf_grad_0_b) < 0 or normal_edge_1.dot(sdf_grad_1_b) < 0:
        #         cur_length = (p_1 - p_0).norm()
        #         ga_sdf_cell_size = self._solver.sdf.geoms_info[i_ga].sdf_cell_size
        #         while cur_length > ga_sdf_cell_size:
        #             p_mid = 0.5 * (p_0 + p_1)
        #             if self._solver.sdf.sdf_grad_world(p_mid, i_gb, i_b).dot(vec_01) < 0:
        #                 p_0 = p_mid
        #             else:
        #                 p_1 = p_mid

        #             cur_length = 0.5 * cur_length

        #         p = 0.5 * (p_0 + p_1)
        #         new_penetration = -self._solver.sdf.sdf_world(p, i_gb, i_b)

        #         if new_penetration > penetration:
        #             is_col = 1
        #             normal = self._solver.sdf.sdf_normal_world(p, i_gb, i_b)
        #             contact_pos = p
        #             penetration = new_penetration

        return is_col, normal, penetration, contact_pos, i_va

    @ti.func
    def _func_update_aabbs(self):
        self._solver._func_update_geom_aabbs()

    @ti.func
    def _func_check_collision_valid(self, i_ga, i_gb, i_b):
        i_la = self._solver.geoms_info[i_ga].link_idx
        i_lb = self._solver.geoms_info[i_gb].link_idx
        is_valid = True

        # geoms in the same link
        if i_la == i_lb:
            is_valid = False

        # self collision
        if (
            ti.static(not self._solver._enable_self_collision)
            and self._solver.links_info[i_la].root_idx == self._solver.links_info[i_lb].root_idx
        ):
            is_valid = False

        # adjacent links
        if self._solver.links_info[i_la].parent_idx == i_lb or self._solver.links_info[i_lb].parent_idx == i_la:
            is_valid = False

        # pair of fixed links
        if self._solver.links_info[i_la].is_fixed and self._solver.links_info[i_lb].is_fixed:
            is_valid = False

        # hibernated <-> fixed links
        if ti.static(self._solver._use_hibernation):
            if (self._solver.links_state[i_la, i_b].hibernated and self._solver.links_info[i_lb].is_fixed) or (
                self._solver.links_state[i_lb, i_b].hibernated and self._solver.links_info[i_la].is_fixed
            ):
                is_valid = False

        return is_valid

    @ti.func
    def _func_broad_phase(self):
        """
        Sweep and Prune (SAP) for broad-phase collision detection.

        This function sorts the geometry axis-aligned bounding boxes (AABBs) along a specified axis and checks for potential collision pairs based on the AABB overlap.
        """

        first_time = self.first_time[None]
        if first_time:
            self.first_time[None] = False

        ti.loop_config(serialize=self._solver._para_level < gs.PARA_LEVEL.ALL)
        for i_b in range(self._solver._B):
            axis = 0

            # copy updated geom aabbs to buffer for sorting
            if first_time:
                for i in range(self._solver.n_geoms):
                    self.sort_buffer[2 * i, i_b].value = self._solver.geoms_state[i, i_b].aabb_min[axis]
                    self.sort_buffer[2 * i, i_b].i_g = i
                    self.sort_buffer[2 * i, i_b].is_max = 0

                    self.sort_buffer[2 * i + 1, i_b].value = self._solver.geoms_state[i, i_b].aabb_max[axis]
                    self.sort_buffer[2 * i + 1, i_b].i_g = i
                    self.sort_buffer[2 * i + 1, i_b].is_max = 1

                    self._solver.geoms_state[i, i_b].min_buffer_idx = 2 * i
                    self._solver.geoms_state[i, i_b].max_buffer_idx = 2 * i + 1

            else:
                # warm start. If `use_hibernation=True`, it's already updated in rigid_solver.
                if ti.static(not self._solver._use_hibernation):
                    for i in range(self._solver.n_geoms * 2):
                        if self.sort_buffer[i, i_b].is_max:
                            self.sort_buffer[i, i_b].value = self._solver.geoms_state[
                                self.sort_buffer[i, i_b].i_g, i_b
                            ].aabb_max[axis]
                        else:
                            self.sort_buffer[i, i_b].value = self._solver.geoms_state[
                                self.sort_buffer[i, i_b].i_g, i_b
                            ].aabb_min[axis]

            # insertion sort, which has complexity near O(n) for nearly sorted array
            for i in range(1, 2 * self._solver.n_geoms):
                key = self.sort_buffer[i, i_b]

                j = i - 1
                while j >= 0 and key.value < self.sort_buffer[j, i_b].value:
                    self.sort_buffer[j + 1, i_b] = self.sort_buffer[j, i_b]

                    if ti.static(self._solver._use_hibernation):
                        if self.sort_buffer[j, i_b].is_max:
                            self._solver.geoms_state[self.sort_buffer[j, i_b].i_g, i_b].max_buffer_idx = j + 1
                        else:
                            self._solver.geoms_state[self.sort_buffer[j, i_b].i_g, i_b].min_buffer_idx = j + 1

                    j -= 1
                self.sort_buffer[j + 1, i_b] = key

                if ti.static(self._solver._use_hibernation):
                    if key.is_max:
                        self._solver.geoms_state[key.i_g, i_b].max_buffer_idx = j + 1
                    else:
                        self._solver.geoms_state[key.i_g, i_b].min_buffer_idx = j + 1

            # sweep over the sorted AABBs to find potential collision pairs
            self.n_broad_pairs[i_b] = 0
            if ti.static(not self._solver._use_hibernation):
                n_active = 0
                for i in range(2 * self._solver.n_geoms):
                    if not self.sort_buffer[i, i_b].is_max:
                        for j in range(n_active):
                            i_ga = self.active_buffer[j, i_b]
                            i_gb = self.sort_buffer[i, i_b].i_g

                            if not self._func_is_geom_aabbs_overlap(i_ga, i_gb, i_b):
                                continue

                            if not self._func_check_collision_valid(i_ga, i_gb, i_b):
                                continue

                            self.broad_collision_pairs[self.n_broad_pairs[i_b], i_b][0] = i_ga
                            self.broad_collision_pairs[self.n_broad_pairs[i_b], i_b][1] = i_gb
                            self.n_broad_pairs[i_b] = self.n_broad_pairs[i_b] + 1

                        self.active_buffer[n_active, i_b] = self.sort_buffer[i, i_b].i_g
                        n_active = n_active + 1
                    else:
                        i_g_to_remove = self.sort_buffer[i, i_b].i_g
                        for j in range(n_active):
                            if self.active_buffer[j, i_b] == i_g_to_remove:
                                if j < n_active - 1:
                                    for k in range(j, n_active - 1):
                                        self.active_buffer[k, i_b] = self.active_buffer[k + 1, i_b]
                                n_active = n_active - 1
                                break
            else:
                if self._solver.n_awake_dofs[i_b] == 0:
                    pass
                else:
                    n_active_awake = 0
                    n_active_hib = 0
                    for i in range(2 * self._solver.n_geoms):
                        is_incoming_geom_hibernated = self._solver.geoms_state[
                            self.sort_buffer[i, i_b].i_g, i_b
                        ].hibernated

                        if not self.sort_buffer[i, i_b].is_max:
                            # both awake and hibernated geom check with active awake geoms
                            for j in range(n_active_awake):
                                i_ga = self.active_buffer_awake[j, i_b]
                                i_gb = self.sort_buffer[i, i_b].i_g

                                if not self._func_is_geom_aabbs_overlap(i_ga, i_gb, i_b):
                                    continue

                                if not self._func_check_collision_valid(i_ga, i_gb, i_b):
                                    continue

                                self.broad_collision_pairs[self.n_broad_pairs[i_b], i_b][0] = i_ga
                                self.broad_collision_pairs[self.n_broad_pairs[i_b], i_b][1] = i_gb
                                self.n_broad_pairs[i_b] = self.n_broad_pairs[i_b] + 1

                            # if incoming geom is awake, also need to check with hibernated geoms
                            if not is_incoming_geom_hibernated:
                                for j in range(n_active_hib):
                                    i_ga = self.active_buffer_hib[j, i_b]
                                    i_gb = self.sort_buffer[i, i_b].i_g

                                    if not self._func_is_geom_aabbs_overlap(i_ga, i_gb, i_b):
                                        continue

                                    if not self._func_check_collision_valid(i_ga, i_gb, i_b):
                                        continue

                                    self.broad_collision_pairs[self.n_broad_pairs[i_b], i_b][0] = i_ga
                                    self.broad_collision_pairs[self.n_broad_pairs[i_b], i_b][1] = i_gb
                                    self.n_broad_pairs[i_b] = self.n_broad_pairs[i_b] + 1

                            if is_incoming_geom_hibernated:
                                self.active_buffer_hib[n_active_hib, i_b] = self.sort_buffer[i, i_b].i_g
                                n_active_hib = n_active_hib + 1
                            else:
                                self.active_buffer_awake[n_active_awake, i_b] = self.sort_buffer[i, i_b].i_g
                                n_active_awake = n_active_awake + 1
                        else:
                            i_g_to_remove = self.sort_buffer[i, i_b].i_g
                            if is_incoming_geom_hibernated:
                                for j in range(n_active_hib):
                                    if self.active_buffer_hib[j, i_b] == i_g_to_remove:
                                        if j < n_active_hib - 1:
                                            for k in range(j, n_active_hib - 1):
                                                self.active_buffer_hib[k, i_b] = self.active_buffer_hib[k + 1, i_b]
                                        n_active_hib = n_active_hib - 1
                                        break
                            else:
                                for j in range(n_active_awake):
                                    if self.active_buffer_awake[j, i_b] == i_g_to_remove:
                                        if j < n_active_awake - 1:
                                            for k in range(j, n_active_awake - 1):
                                                self.active_buffer_awake[k, i_b] = self.active_buffer_awake[k + 1, i_b]
                                        n_active_awake = n_active_awake - 1
                                        break

    @ti.func
    def _func_narrow_phase(self):
        """
        NOTE: for a single non-batched scene with a lot of collisioin pairs, it will be faster if we also parallelize over `self.n_collision_pairs`. However, parallelize over both B and collisioin_pairs (instead of only over B) leads to significantly slow performance for batched scene. We can treat B=0 and B>0 separately, but we will end up with messier code.
        Therefore, for a big non-batched scene, users are encouraged to simply use `gs.cpu` backend.
        Updated NOTE & TODO: For a HUGE scene with numerous bodies, it's also reasonable to run on GPU. Let's save this for later.
        Update2: Now we use n_broad_pairs instead of n_collision_pairs, so we probably need to think about how to handle non-batched large scene better.
        """
        ti.loop_config(serialize=self._solver._para_level < gs.PARA_LEVEL.ALL)
        for i_b in range(self._solver._B):
            for i_pair in range(self.n_broad_pairs[i_b]):
                i_ga = self.broad_collision_pairs[i_pair, i_b][0]
                i_gb = self.broad_collision_pairs[i_pair, i_b][1]

                is_col = gs.ti_int(0)
                penetration = gs.ti_float(0.0)
                normal = ti.Vector.zero(gs.ti_float, 3)
                contact_pos = ti.Vector.zero(gs.ti_float, 3)

                if (
                    self._solver.geoms_info[i_ga].type == gs.GEOM_TYPE.SPHERE
                    or self._solver.geoms_info[i_gb].type == gs.GEOM_TYPE.SPHERE
                ):
                    if ti.static(self._has_sphere):
                        if self._solver.geoms_info[i_gb].type == gs.GEOM_TYPE.SPHERE:
                            i_ga, i_gb = i_gb, i_ga
                        is_col, normal, penetration, contact_pos = self._func_contact_sphere_sdf(i_ga, i_gb, i_b)

                        if is_col:
                            self._func_add_contact(i_ga, i_gb, normal, contact_pos, penetration, i_b)

                elif (
                    self._solver.geoms_info[i_ga].type == gs.GEOM_TYPE.TERRAIN
                    or self._solver.geoms_info[i_gb].type == gs.GEOM_TYPE.TERRAIN
                ):
                    if ti.static(self._has_terrain):
                        if self._solver.geoms_info[i_ga].type == gs.GEOM_TYPE.TERRAIN:
                            i_ga, i_gb = i_gb, i_ga

                        if self._solver.geoms_info[i_ga].is_convex:
                            # initial point
                            is_col_0, normal_0, penetration_0, contact_pos_0, i_va0 = self._func_contact_convex_terrain(
                                i_ga, i_gb, i_b, gu.ti_identity_quat(), self.contact_cache[i_ga, i_gb, i_b].i_va_0
                            )

                            if is_col_0:
                                # perturb geom_a around two orthogonal axes to find multiple contacts
                                axis_0, axis_1 = gu.orthogonals2(normal_0)
                                n_valid = 0
                                for i_rot in range(4):
                                    axis = axis_0
                                    if i_rot == 1:
                                        axis = -axis_0
                                    elif i_rot == 2:
                                        axis = axis_1
                                    elif i_rot == 3:
                                        axis = -axis_1

                                    is_col, normal, penetration, contact_pos, i_va = self._func_contact_convex_terrain(
                                        i_ga, i_gb, i_b, gu.ti_rotvec_to_quat(axis * self._mc_perturbation), i_va0
                                    )

                                    if is_col:
                                        i_col = self.n_contacts[i_b]
                                        valid = True
                                        for j in range(n_valid):
                                            if (
                                                contact_pos - self.contact_data[i_col - j - 1, i_b].pos
                                            ).norm() < self._mc_tolerance:
                                                valid = False
                                                break

                                        if valid:
                                            n_valid += 1
                                            self._func_add_contact(i_ga, i_gb, normal, contact_pos, penetration_0, i_b)

                                # cache deepest vertex
                                self.contact_cache[i_ga, i_gb, i_b].i_va_0 = i_va0
                                break

                            else:
                                self.contact_cache[i_ga, i_gb, i_b].i_va_0 = -1

                        else:
                            # TODO: support nonconvex<->terrain
                            pass

                elif self._solver.geoms_info[i_ga].is_convex and self._solver.geoms_info[i_gb].is_convex:
                    for i in range(2):
                        if i == 1:
                            i_ga, i_gb = i_gb, i_ga
                        # initial point
                        is_col_0, normal_0, penetration_0, contact_pos_0, i_va0 = self._func_contact_convex_convex(
                            i_ga, i_gb, i_b, gu.ti_identity_quat(), self.contact_cache[i_ga, i_gb, i_b].i_va_0
                        )

                        if is_col_0:
                            # perturb geom_a around two orthogonal axes to find multiple contacts
                            axis_0, axis_1 = gu.orthogonals2(normal_0)
                            n_valid = 0
                            for i_rot in range(4):
                                axis = axis_0
                                if i_rot == 1:
                                    axis = -axis_0
                                elif i_rot == 2:
                                    axis = axis_1
                                elif i_rot == 3:
                                    axis = -axis_1

                                is_col, normal, penetration, contact_pos, i_va = self._func_contact_convex_convex(
                                    i_ga, i_gb, i_b, gu.ti_rotvec_to_quat(axis * self._mc_perturbation), i_va0
                                )

                                if is_col:
                                    i_col = self.n_contacts[i_b]
                                    valid = True
                                    for j in range(n_valid):
                                        if (
                                            contact_pos - self.contact_data[i_col - j - 1, i_b].pos
                                        ).norm() < self._mc_tolerance:
                                            valid = False
                                            break

                                    if valid:
                                        n_valid += 1
                                        self._func_add_contact(i_ga, i_gb, normal, contact_pos, penetration_0, i_b)

                            # cache deepest vertex
                            self.contact_cache[i_ga, i_gb, i_b].i_va_0 = i_va0
                            break

                        else:
                            self.contact_cache[i_ga, i_gb, i_b].i_va_0 = -1

                else:
                    if ti.static(self._has_nonconvex_nonterrain):
                        for i in range(2):
                            if i == 1:
                                i_ga, i_gb = i_gb, i_ga
                            # initial point
                            is_col_0, normal_0, penetration_0, contact_pos_0 = self._func_contact_vertex_sdf(
                                i_ga, i_gb, i_b, gu.ti_identity_quat()
                            )

                            penetrated = normal_0.dot(self.contact_cache[i_ga, i_gb, i_b].normal) >= 0
                            if (not is_col_0) or penetrated:
                                self.contact_cache[i_ga, i_gb, i_b].penetration = penetration_0
                                self.contact_cache[i_ga, i_gb, i_b].normal = normal_0

                            if is_col_0:
                                # perturb geom_a around two orthogonal axes to find multiple contacts
                                axis_0, axis_1 = gu.orthogonals2(normal_0)
                                n_valid = 0
                                for i_rot in range(4):
                                    axis = axis_0
                                    if i_rot == 1:
                                        axis = -axis_0
                                    elif i_rot == 2:
                                        axis = axis_1
                                    elif i_rot == 3:
                                        axis = -axis_1

                                    is_col, normal, penetration, contact_pos = self._func_contact_vertex_sdf(
                                        i_ga, i_gb, i_b, gu.ti_rotvec_to_quat(axis * self._mc_perturbation)
                                    )

                                    if penetrated:
                                        normal = self.contact_cache[i_ga, i_gb, i_b].normal
                                        penetration = self.contact_cache[i_ga, i_gb, i_b].penetration
                                    else:
                                        penetration = penetration_0

                                    if is_col:
                                        i_col = self.n_contacts[i_b]
                                        valid = True
                                        for j in range(n_valid):
                                            if (
                                                contact_pos - self.contact_data[i_col - j - 1, i_b].pos
                                            ).norm() < self._mc_tolerance:
                                                valid = False
                                                break

                                        if valid:
                                            n_valid += 1
                                            self._func_add_contact(i_ga, i_gb, normal, contact_pos, penetration, i_b)

                                break

                        if not is_col:  # check edge-edge if vertex-face is not detected
                            is_col, normal, penetration, contact_pos = self._func_contact_edge_sdf(i_ga, i_gb, i_b)

                            if is_col:
                                self._func_add_contact(i_ga, i_gb, normal, contact_pos, penetration, i_b)

    @ti.func
    def _func_add_contact(self, i_ga, i_gb, normal, contact_pos, penetration, i_b):
        i_col = self.n_contacts[i_b]

        ga_info = self._solver.geoms_info[i_ga]
        gb_info = self._solver.geoms_info[i_gb]

        # b to a
        self.contact_data[i_col, i_b].geom_a = i_ga
        self.contact_data[i_col, i_b].geom_b = i_gb
        self.contact_data[i_col, i_b].normal = normal
        self.contact_data[i_col, i_b].pos = contact_pos
        self.contact_data[i_col, i_b].penetration = penetration
        self.contact_data[i_col, i_b].friction = ti.max(ga_info.friction, gb_info.friction)
        self.contact_data[i_col, i_b].sol_params = 0.5 * (ga_info.sol_params + gb_info.sol_params)
        self.contact_data[i_col, i_b].link_a = ga_info.link_idx
        self.contact_data[i_col, i_b].link_b = gb_info.link_idx

        self.n_contacts[i_b] = i_col + 1
