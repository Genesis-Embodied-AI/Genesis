import numpy as np
import taichi as ti

import genesis as gs

# import genesis.engine.bodies.rigid_utils as ru


@ti.func
def stp(v1, v2, v3):
    return v1.dot(v2.cross(v3))


@ti.func
def solve_quadratic(a, b, c):
    res = 0
    x0 = x1 = 0.0
    d = b * b - 4 * a * c
    if d < 0:
        x0 = -b / (2 * a)
        res = 0
    else:
        q = -(b + ti.math.sign(b) * ti.math.sqrt(d)) / 2
        if ti.abs(a) > 1e-12 * ti.abs(q):
            x0 = q / a
            res += 1
        if ti.abs(q) > 1e-12 * ti.abs(c):
            if res == 0:
                x0 = c / q
            elif res == 1:
                x1 = c / q
            res += 1
        if res == 2 and x0 > x1:
            x0, x1 = x1, x0
    return res, x0, x1


@ti.func
def newtons_method(a, b, c, d, x0, init_dir):
    if init_dir != 0:
        y0 = d + x0 * (c + x0 * (b + x0 * a))
        ddy0 = 2 * b + x0 * (6 * a)
        x0 += init_dir * ti.math.sqrt(ti.abs(2 * y0 / ddy0))
    for i in range(100):
        y = d + x0 * (c + x0 * (b + x0 * a))
        dy = c + x0 * (2 * b + x0 * 3 * a)
        if dy == 0:
            break
        x1 = x0 - y / dy
        if ti.abs(x0 - x1) < 1e-6:
            break
        x0 = x1
    return x0


@ti.func
def solve_cubic(a, b, c, d):
    nr = 0
    x0 = x1 = x2 = 1.0
    ncrit, xc0, xc1 = solve_quadratic(3 * a, 2 * b, c)
    if ncrit == 0:
        x0 = newtons_method(a, b, c, d, xc0, 0)
        nr = 1
    elif ncrit == 1:
        nr, x0, x1 = solve_quadratic(3 * a, 2 * b, c)
    else:
        yc0 = d + xc0 * (c + xc0 * (b + xc0 * a))
        yc1 = d + xc1 * (c + xc1 * (b + xc1 * a))
        if yc0 * a >= 0:
            x0 = newtons_method(a, b, c, d, xc0, -1)
            nr += 1
        if yc0 * yc1 <= 0:
            _x = 1.0
            if ti.abs(yc0) < ti.abs(yc1):
                _x = newtons_method(a, b, c, d, xc0, 1)
            else:
                _x = newtons_method(a, b, c, d, xc1, -1)
            if nr == 0:
                x0 = _x
            elif nr == 1:
                x1 = _x
            nr += 1

        if yc1 * a <= 0:
            _x = newtons_method(a, b, c, d, xc1, 1)

            if nr == 0:
                x0 = _x
            elif nr == 1:
                x1 = _x
            elif nr == 2:
                x2 = _x
            nr += 2
    return nr, x0, x1, x2


@ti.func
def signed_vf_distance(x, y0, y1, y2):
    n = (y1 - y0).normalized().cross((y2 - y0).normalized())
    h = np.inf
    w = ti.Vector([0.0, 0.0, 0.0, 0.0])
    if n.norm() < 1e-6:
        pass
    else:
        n = n.normalized()
        h = (x - y0).dot(n)
        b0 = stp(y1 - x, y2 - x, n)
        b1 = stp(y2 - x, y0 - x, n)
        b2 = stp(y0 - x, y1 - x, n)

        w = ti.Vector([1.0, -b0 / (b0 + b1 + b2), -b1 / (b0 + b1 + b2), -b2 / (b0 + b1 + b2)])
    return h, n, w


@ti.func
def signed_ee_distance(x0, x1, y0, y1):
    n = (x1 - x0).normalized().cross((y1 - y0).normalized())
    h = np.inf
    w = ti.Vector([0.0, 0.0, 0.0, 0.0])
    if n.norm() < 1e-6:
        pass
    else:
        n = n.normalized()
        h = (x0 - y0).dot(n)

        a0 = stp(y1 - x1, y0 - x1, n)
        a1 = stp(y0 - x0, y1 - x0, n)

        b0 = stp(x0 - y1, x1 - y1, n)
        b1 = stp(x1 - y0, x0 - y0, n)

        w = ti.Vector([a0 / (a0 + a1), a1 / (a0 + a1), -b0 / (b0 + b1), -b1 / (b0 + b1)])

        w[0] = a0 / (a0 + a1)
        w[1] = a1 / (a0 + a1)
        w[2] = -b0 / (b0 + b1)
        w[3] = -b1 / (b0 + b1)
    return h, n, w


@ti.data_oriented
class AABB:
    # Class for axis aligned bounding box
    def __init__(self, box_data):
        self.box_data = box_data

    @ti.func
    def update_axis(self, idx):
        umax = self.box_data[idx].umax
        umin = self.box_data[idx].umin

        size = umax - umin
        if size[0] < size[1]:
            self.box_data[idx].axis = 1
        if ti.max(size[0], size[1]) < size[2]:
            self.box_data[idx].axis = 2

    @ti.func
    def reset_box(self, idx):
        self.box_data[idx].umax = ti.Vector([-np.inf, -np.inf, -np.inf])
        self.box_data[idx].umin = ti.Vector([np.inf, np.inf, np.inf])

    @ti.func
    def add_vert(self, idx, v):
        self.box_data[idx].umax = ti.max(v, self.box_data[idx].umax)
        self.box_data[idx].umin = ti.min(v, self.box_data[idx].umin)

    @ti.func
    def overlap(self, a, b):
        result = True
        for i in ti.static(range(3)):
            if self.box_data[a].umax[i] < self.box_data[b].umin[i]:
                result = False
            if self.box_data[b].umax[i] < self.box_data[a].umin[i]:
                result = False
        return result

    @ti.func
    def copy_box(self, idx, src):
        self.box_data[idx].umax = self.box_data[src].umax
        self.box_data[idx].umin = self.box_data[src].umin
        self.box_data[idx].axis = self.box_data[src].axis

    @ti.func
    def add_box(self, idx, src):
        self.box_data[idx].umin = ti.min(self.box_data[idx].umin, self.box_data[src].umin)
        self.box_data[idx].umax = ti.max(self.box_data[idx].umax, self.box_data[src].umax)

    @ti.func
    def inside_left(self, idx, v):
        axis = self.box_data[idx].axis
        flag = True
        if axis == 0:
            flag = v[0] < 0.5 * (self.box_data[idx].umax + self.box_data[idx].umin)[0]
        elif axis == 1:
            flag = v[1] < 0.5 * (self.box_data[idx].umax + self.box_data[idx].umin)[1]
        else:
            flag = v[2] < 0.5 * (self.box_data[idx].umax + self.box_data[idx].umin)[2]

        return flag


@ti.data_oriented
class BVHTree:
    # Class for BVHTree
    def __init__(self, rigid_bodies):
        self.solver = rigid_bodies

        # self.verts = verts
        # self.faces = faces
        self.n_f = self.solver.n_faces_max
        self.n_links = self.solver.n_geoms_max

        self.len_cd_que = 10000
        self.len_cd_impact = 100000
        self.len_cd_candidate = 100000
        self.len_constraints = 2000

        box_class = AABB

        struct_box = ti.types.struct(
            umin=gs.ti_vec3,
            umax=gs.ti_vec3,
            axis=gs.ti_int,
        )
        # [0, 2 * self.n_f) nodes, [2 * self.n_f, 3 * self.n_f) triangle boxes, 3 * self.n_f total
        num_box = self.n_f * 3 + self.n_links

        self.box_data = struct_box.field(shape=(num_box), needs_grad=False, layout=ti.Layout.SOA)
        self.init_box_data(num_box)

        struct_node = ti.types.struct(
            parent=gs.ti_int,
            left=gs.ti_int,
            right=gs.ti_int,
            depth=gs.ti_int,
            face=gs.ti_int,
            box=gs.ti_int,
        )
        self.node_data = struct_node.field(shape=(2 * self.n_f), needs_grad=False, layout=ti.Layout.SOA)
        self.init_node_data(self.n_f * 2)

        struct_candidate = ti.types.struct(
            t=gs.ti_int,  # 0: vf, 1: ee
            i0=gs.ti_int,
            i1=gs.ti_int,
            i2=gs.ti_int,
            i3=gs.ti_int,
        )
        self.candidate_data = struct_candidate.field(
            shape=(self.len_cd_candidate), needs_grad=False, layout=ti.Layout.SOA
        )

        struct_narrow_candidate = ti.types.struct(
            v0=gs.ti_int,
            v1=gs.ti_int,
            v2=gs.ti_int,
            v3=gs.ti_int,
            ctype=gs.ti_int,  # 0: vertex face; 1: edge edge
            x0=gs.ti_vec3,
            x1=gs.ti_vec3,
            x2=gs.ti_vec3,
            x3=gs.ti_vec3,
            dx0=gs.ti_vec3,
            dx1=gs.ti_vec3,
            dx2=gs.ti_vec3,
            dx3=gs.ti_vec3,
            # accd
            toi=gs.ti_float,
            is_col=gs.ti_int,  # 0: no collision; 1: collision; -1: unkown
            max_disp_mag=gs.ti_float,
            distance=gs.ti_float,
            # dvd
            penetration=gs.ti_float,
            n=gs.ti_vec3,
            pos=gs.ti_vec3,
            ga=gs.ti_int,
            gb=gs.ti_int,
        )

        self.narrow_candidate = struct_narrow_candidate.field(
            shape=(self.len_cd_candidate), needs_grad=False, layout=ti.Layout.SOA
        )

        struct_contact_aggregate = ti.types.struct(
            ctype=gs.ti_int,
            penetration=gs.ti_float,
            n=gs.ti_vec3,
            pos=gs.ti_vec3,
        )
        self.contact_aggregate = struct_contact_aggregate.field(
            shape=(self.solver.n_geoms_max, self.solver.n_geoms_max), needs_grad=False, layout=ti.Layout.SOA
        )

        struct_impact = ti.types.struct(
            pos=gs.ti_vec3,
            n=gs.ti_vec3,
            penetration=gs.ti_float,
            friction=gs.ti_float,
        )
        self.impact_data = struct_impact.field(shape=(self.len_cd_impact), needs_grad=True, layout=ti.Layout.SOA)

        struct_impact_info = ti.types.struct(
            link_a=gs.ti_int,
            link_b=gs.ti_int,
            is_contact=gs.ti_int,
            i0=gs.ti_int,
            i1=gs.ti_int,
            i2=gs.ti_int,
            i3=gs.ti_int,
        )
        self.impact_info = struct_impact_info.field(shape=(self.len_cd_impact), needs_grad=False, layout=ti.Layout.SOA)

        self.solver_params_contact = [2.0e-02, 1.0e00, 9.0e-01, 9.5e-01, 1.0e-03, 5.0e-01, 2.0e00]

        self.con_jac = ti.field(dtype=gs.ti_float, shape=(self.len_constraints, self.solver.n_dofs))
        self.con_diag = ti.field(dtype=gs.ti_float, shape=(self.len_constraints))
        self.con_aref = ti.field(dtype=gs.ti_float, shape=(self.len_constraints))

        self.constraint_A = ti.field(dtype=gs.ti_float, shape=(self.len_constraints, self.len_constraints))
        self.constraint_b = ti.field(dtype=gs.ti_float, shape=(self.len_constraints))

        self.x = ti.field(dtype=gs.ti_float, shape=(self.len_constraints))
        self.x1 = ti.field(dtype=gs.ti_float, shape=(self.len_constraints))
        self.xgrad = ti.field(dtype=gs.ti_float, shape=(self.len_constraints))

        self.Ax_b = ti.field(dtype=gs.ti_float, shape=(self.len_constraints))

        # 0 candidate, 1 impact, 3 constraints
        self.cd_counter = ti.field(gs.ti_int, shape=(3))

        self._box = box_class(self.box_data)

        self.onode = 0
        self.otri = self.n_f * 2
        self.ototal = self.n_f * 3

        self.tri_centers = ti.Vector.field(3, dtype=gs.ti_float, shape=(self.n_f))

        # my_id, fstart, n, parent_id
        self.que = ti.field(dtype=gs.ti_int, shape=(2 * self.n_f, 4))
        # self.face_buffer[self.n_f] records the current node idx during building tree
        self.face_buffer = ti.field(dtype=gs.ti_int, shape=(self.n_f + self.n_links))

        self.mbox = num_box
        self.mnode = self.n_f * 2
        self.mque = 2 * self.n_f
        self.mbuffer = self.n_f + self.n_links

        self.build_bvh()
        self.process_structure()
        ti.sync()
        print(f"init tree done, n_f = {self.n_f}")
        # self.solver.link_curr_geometry[0][0]

        self.generate_collision_shape_pairs()

    def process_structure(self):
        depth = self.node_data.depth.to_numpy()
        max_depth = depth.max()
        print("max depth", max_depth)
        node_idx = np.arange(len(depth), dtype=gs.np_int)
        self.depth_node_idx = []
        for i_d in range(max_depth + 1):
            nodes = node_idx[depth == i_d]
            self.depth_node_idx.append((len(nodes), nodes))

        nodes_leaf_to_root = []
        for i_d in range(max_depth + 1):
            nodes = node_idx[depth == (max_depth - i_d)]
            nodes_leaf_to_root.append(nodes)

        nodes_leaf_to_root = np.concatenate(nodes_leaf_to_root)
        self.nodes_leaf_to_root = ti.field(dtype=gs.ti_int, shape=nodes_leaf_to_root.shape)
        self.nodes_leaf_to_root.from_numpy(nodes_leaf_to_root)
        print("nodes_leaf_to_root", nodes_leaf_to_root)

        self.refit_kernel()
        # self.refit_tree()

    def refit_tree(self):
        num_layers = len(self.depth_node_idx)
        # print("refit_tree")
        # print("self.node_data", self.node_data.box)
        for i in range(num_layers):
            n, nodes = self.depth_node_idx[num_layers - 1 - i]
            self.refit_depth(n, nodes)

    def refit_kernel(self):
        self.refit_func()

    @ti.kernel
    def refit_func(self):
        ti.loop_config(serialize=True)
        for i in range(self.nodes_leaf_to_root.shape[0]):
            node = self.nodes_leaf_to_root[i]
            box = self.node_data[node].box
            self._box.reset_box(box)
            if self.node_data[node].face != -1:
                f = self.node_data[node].face
                p0 = self.solver.verts_state[self.solver.faces_info[f].verts_idx[0]].pos
                p1 = self.solver.verts_state[self.solver.faces_info[f].verts_idx[1]].pos
                p2 = self.solver.verts_state[self.solver.faces_info[f].verts_idx[2]].pos
                self._box.add_vert(box, p0)
                self._box.add_vert(box, p1)
                self._box.add_vert(box, p2)

                # print("-- leaf", node, f, self._box.box_data[box].umin, self._box.box_data[box].umax)
            else:
                left = self.node_data[node].left
                right = self.node_data[node].right

                left_box = self.node_data[left].box
                right_box = self.node_data[right].box
                self._box.add_box(box, left_box)
                self._box.add_box(box, right_box)

                # print("-- node", node, left, right, self._box.box_data[box].umin, self._box.box_data[box].umax)

    @ti.kernel
    def refit_depth(self, n: int, nodes: ti.types.ndarray()):
        for i in nodes:
            node = nodes[i]
            box = self.node_data[node].box
            self._box.reset_box(box)
            if self.node_data[node].face != -1:
                f = self.node_data[node].face

                p0 = self.solver.verts[self.solver.faces[f].verts_idx[0]].pos
                p1 = self.solver.verts[self.solver.faces[f].verts_idx[1]].pos
                p2 = self.solver.verts[self.solver.faces[f].verts_idx[2]].pos

                self._box.add_vert(box, p0)
                self._box.add_vert(box, p1)
                self._box.add_vert(box, p2)
                # print("-- face", f, node, self.solver.n_faces_max)
            else:
                left = self.node_data[node].left
                right = self.node_data[node].right
                # print("-- structure", node, left, right)

                left_box = self.node_data[left].box
                right_box = self.node_data[right].box
                self._box.add_box(box, left_box)
                self._box.add_box(box, right_box)
                # print("-- box", node, box, self.box_data[box].umax, self.box_data[box].umin)

    @ti.func
    def det3x2(self, b, c, d):
        return (
            b[0] * c[1] * d[2]
            + c[0] * d[1] * b[2]
            + d[0] * b[1] * c[2]
            - d[0] * c[1] * b[2]
            - c[0] * b[1] * d[2]
            - b[0] * d[1] * c[2]
        )

    @ti.func
    def point_inside_tet(self, v0, v1, v2, v3, p):
        a = v0 - p
        b = v1 - p
        c = v2 - p
        d = v3 - p
        detA = self.det3x2(b, c, d)
        detB = self.det3x2(a, c, d)
        detC = self.det3x2(a, b, d)
        detD = self.det3x2(a, b, c)
        ret0 = detA > 0.0 and detB < 0.0 and detC > 0.0 and detD < 0.0
        ret1 = detA < 0.0 and detB > 0.0 and detC < 0.0 and detD > 0.0

        is_col = ret0 or ret1
        return is_col

    @ti.func
    def segment_intersect_triangle(self, e0, e1, A, B, C):
        Norm = (e1 - e0).norm()
        Dir = (e1 - e0) / Norm
        Origin = e0

        E1 = B - A
        E2 = C - A
        N = E1.cross(E2)

        det = -Dir.dot(N)
        invdet = 1.0 / det
        AO = Origin - A
        DAO = AO.cross(Dir)
        u = E2.dot(DAO) * invdet
        v = -E1.dot(DAO) * invdet
        t = AO.dot(N) * invdet

        t = t / Norm

        is_col = det >= 1e-6 and t > 0.0001 and t < 0.9999 and u >= 0.0 and v >= 0.0 and (u + v) <= 1.0
        return is_col, t, u, v

    @ti.func
    def point_plane_distance_normal(self, p, origin, normal):
        point_to_plane = (p - origin).dot(normal)
        d = point_to_plane / normal.norm()
        return d**2

    @ti.func
    def point_plane_distance(self, p, t0, t1, t2):
        return self.point_plane_distance_normal(p, t0, (t1 - t0).cross(t2 - t0))

    @ti.func
    def cd_check_penetration(self, i, ctype):

        v0 = self.narrow_candidate[i].v0
        v1 = self.narrow_candidate[i].v1
        v2 = self.narrow_candidate[i].v2
        v3 = self.narrow_candidate[i].v3
        # min_distance = 0.001

        if ctype == 0:  # vertex face
            p_t1 = self.solver.verts_state[v0].pos
            t0_t1 = self.solver.verts_state[v1].pos
            t1_t1 = self.solver.verts_state[v2].pos
            t2_t1 = self.solver.verts_state[v3].pos

            t_geom1_idx = self.solver.verts_info[v1].geom_idx
            t_center_b = self.solver.verts_state[v1].center_pos

            # print(t0_t1, t1_t1, t2_t1, t_center_b, p_t1)
            is_col = self.point_inside_tet(t0_t1, t1_t1, t2_t1, t_center_b, p_t1)

            if is_col == 1:
                penetration = ti.sqrt(self.point_plane_distance(p_t1, t0_t1, t1_t1, t2_t1))

                if penetration > self.narrow_candidate[i].penetration:

                    self.narrow_candidate[i].ctype = ctype
                    self.narrow_candidate[i].is_col = is_col
                    self.narrow_candidate[i].penetration = penetration

                    n = (t1_t1 - t0_t1).cross(t2_t1 - t0_t1)
                    sign = (n.dot(t0_t1 - t_center_b) > 0) * 2 - 1
                    self.narrow_candidate[i].n = n * sign / n.norm()
                    self.narrow_candidate[i].pos = p_t1

                    self.narrow_candidate[i].ga = self.solver.verts_info[v0].geom_idx
                    self.narrow_candidate[i].gb = self.solver.verts_info[v1].geom_idx

        elif ctype == 1:  # edge edge
            ea0_t1 = self.solver.verts_state[v0].pos
            ea1_t1 = self.solver.verts_state[v1].pos
            eb0_t1 = self.solver.verts_state[v2].pos
            eb1_t1 = self.solver.verts_state[v3].pos

            t_geom2_idx = self.solver.verts_info[v2].geom_idx
            t_center_b = self.solver.verts_state[v2].center_pos

            is_col, t, u, v = self.segment_intersect_triangle(ea0_t1, ea1_t1, eb0_t1, eb1_t1, t_center_b)

            # if ea0_t1[2] == ea1_t1[2] and eb0_t1[2] == eb1_t1[2]:
            #     if ti.abs(ea1_t1[2] - eb0_t1[2]) == 1.0 and ti.min(ea1_t1[2], eb0_t1[2]) == 0.0:
            #         print("!!!!!", i)
            #         print(ea0_t1, ea1_t1, eb0_t1, eb1_t1)
            #         print(t_center_b, t, u, v)
            # if self.narrow_candidate[i].is_col:
            #     print("-----", i)
            #     print(ea0_t1, ea1_t1, eb0_t1, eb1_t1)
            #     print(t_center_b, t)
            #     print(self.point_plane_distance(ea1_t1, eb0_t1, eb1_t1, t_center_b))

            if is_col == 1:

                n = (ea1_t1 - ea0_t1).cross(eb1_t1 - eb0_t1)
                # sign = (n.dot(self.narrow_candidate[i].pos - t_center_b) > 0) * 2 - 1
                sign = (n.dot(eb0_t1 - t_center_b) > 0) * 2 - 1
                n = n * sign / n.norm()
                # TODO: sign here

                pos = ea0_t1 * (1 - t) + ea1_t1 * t
                penetration = -n.dot(pos - eb0_t1)
                # print("edge col!", penetration)

                if penetration > self.narrow_candidate[i].penetration:

                    self.narrow_candidate[i].ctype = ctype
                    self.narrow_candidate[i].is_col = is_col
                    self.narrow_candidate[i].penetration = penetration

                    self.narrow_candidate[i].pos = pos
                    self.narrow_candidate[i].n = n

                    self.narrow_candidate[i].ga = self.solver.verts_info[v0].geom_idx
                    self.narrow_candidate[i].gb = self.solver.verts_info[v2].geom_idx

                # if self.narrow_candidate[i].penetration > 0:
                #     print("===edge", self.narrow_candidate[i].ga, self.narrow_candidate[i].gb,
                #         sign, self.narrow_candidate[i].n, n, self.narrow_candidate[i].pos - t_center_b)
                #     print("points", eb1_t1, eb0_t1, t_center_b)

                # print(t, self.narrow_candidate[i].n,
                #     self.narrow_candidate[i].ga, self.narrow_candidate[i].gb,
                #     self.narrow_candidate[i].penetration)

    @ti.func
    def cd_add_face_and_dcd(self, fa, fb):
        idx = self.cd_counter[0]
        self.cd_counter[0] += 1
        # idx = ti.atomic_add(self.cd_counter[0], 1)
        self.narrow_candidate[idx].is_col = 0
        self.narrow_candidate[idx].penetration = -1

        for j in ti.static(range(3)):
            self.narrow_candidate[idx].v0 = self.solver.faces_info[fa].verts_idx[j % 3]
            self.narrow_candidate[idx].v1 = self.solver.faces_info[fb].verts_idx[0]
            self.narrow_candidate[idx].v2 = self.solver.faces_info[fb].verts_idx[1]
            self.narrow_candidate[idx].v3 = self.solver.faces_info[fb].verts_idx[2]
            self.cd_check_penetration(idx, ctype=0)

        for j in ti.static(range(3)):
            self.narrow_candidate[idx].v0 = self.solver.faces_info[fb].verts_idx[j % 3]
            self.narrow_candidate[idx].v1 = self.solver.faces_info[fa].verts_idx[0]
            self.narrow_candidate[idx].v2 = self.solver.faces_info[fa].verts_idx[1]
            self.narrow_candidate[idx].v3 = self.solver.faces_info[fa].verts_idx[2]

            self.cd_check_penetration(idx, ctype=0)

        # edge edge
        for j1 in ti.static(range(3)):
            for j2 in ti.static(range(3)):
                self.narrow_candidate[idx].v0 = self.solver.faces_info[fa].verts_idx[j1 % 3]
                self.narrow_candidate[idx].v1 = self.solver.faces_info[fa].verts_idx[(j1 + 1) % 3]
                self.narrow_candidate[idx].v2 = self.solver.faces_info[fb].verts_idx[j2 % 3]
                self.narrow_candidate[idx].v3 = self.solver.faces_info[fb].verts_idx[(j2 + 1) % 3]
                self.cd_check_penetration(idx, ctype=1)

    @ti.func
    def cd_add_face(self, fa, fb):
        # !!
        # print("fa, fb", fa, fb)
        for i in ti.static(range(3)):
            self.cd_add_vf(self.solver.faces[fa].verts_idx[i], fb)
            self.cd_add_vf(self.solver.faces[fb].verts_idx[i], fa)

        for i in ti.static(range(3)):
            for j in ti.static(range(3)):
                self.cd_add_ee(fa, fb, i, j)

    @ti.kernel
    def cd_aggregate(self):

        for i in range(self.contact_aggregate.shape[0]):
            for j in range(self.contact_aggregate.shape[1]):
                self.contact_aggregate[i, j].ctype = -1

        for i in range(self.cd_counter[0]):
            if self.narrow_candidate[i].is_col == 1:
                ga = self.narrow_candidate[i].ga
                gb = self.narrow_candidate[i].gb
                ctype = self.narrow_candidate[i].ctype

                n = self.narrow_candidate[i].n
                pos = self.narrow_candidate[i].pos
                penetration = self.narrow_candidate[i].penetration

                if ga == gb:
                    print("warning ga == gb", ga, gb, ctype)
                if ga >= gb:
                    ga, gb = gb, ga
                    n = n * -1

                if self.contact_aggregate[ga, gb].ctype == -1:
                    self.contact_aggregate[ga, gb].ctype = ctype

                    self.contact_aggregate[ga, gb].n = n
                    self.contact_aggregate[ga, gb].pos = pos
                    self.contact_aggregate[ga, gb].penetration = penetration

                elif self.contact_aggregate[ga, gb].ctype == 0:  # vertex face
                    if (
                        self.narrow_candidate[i].ctype == 0
                        and self.contact_aggregate[ga, gb].penetration > self.narrow_candidate[i].penetration
                    ):
                        self.contact_aggregate[ga, gb].n = n
                        self.contact_aggregate[ga, gb].pos = pos
                        self.contact_aggregate[ga, gb].penetration = penetration

                else:  # edge edge
                    if (
                        self.narrow_candidate[i].ctype == 0
                        or self.contact_aggregate[ga, gb].penetration > self.narrow_candidate[i].penetration
                    ):
                        self.contact_aggregate[ga, gb].ctype = ctype
                        self.contact_aggregate[ga, gb].n = n
                        self.contact_aggregate[ga, gb].pos = pos
                        self.contact_aggregate[ga, gb].penetration = penetration

    def collision_detection(self):
        from genesis.utils.tools import create_timer

        timer = create_timer(name="solve_quadratic", level=4, ti_sync=True, skip_first_call=True)
        self.cd_init()
        timer.stamp("cd_init")
        self.refit_func()
        timer.stamp("refit_func")
        self.cd_tree_phase()  # 18 ms
        timer.stamp("cd_tree_phase")
        self.cd_aggregate()
        timer.stamp("cd_aggregate")
        # print("overlapped_face, overlapped_node", self.cd_counter[0], self.cd_counter[1])

        # exit()
        # self.cd_candidate_phase() # not too much
        # print("self.cd_counter 2", self.cd_counter)
        # self.cd_impact_phase() # 1 ms
        # print("self.cd_counter 3", self.cd_counter)

    @ti.kernel
    def compute_constraint_system(self, n_con: int, n_dof: int):
        # TODO TEST
        for jd1 in range(n_dof):
            self.solver.dof_state.qf_smooth[jd1] = 0.1

        for ic1 in range(n_con):
            for ic2 in range(n_con):
                self.constraint_A[ic1, ic2] = 0.0

        for ic1 in range(n_con):
            self.constraint_b[ic1] = 0.0

        for ic1 in range(n_con):
            for ic2 in range(n_con):
                for jd1 in range(n_dof):
                    for jd2 in range(n_dof):
                        self.constraint_A[ic1, ic2] += (
                            self.con_jac[ic1, jd1] * self.solver.mass_mat_inv[jd1, jd2] * self.con_jac[ic2, jd2]
                        )

        for ic1 in range(n_con):
            for jd1 in range(n_dof):
                for jd2 in range(n_dof):
                    self.constraint_b[ic1] += (
                        self.con_jac[ic1, jd1]
                        * self.solver.mass_mat_inv[jd1, jd2]
                        * self.solver.dof_state.qf_smooth[jd2]
                    )

        for ic1 in range(n_con):
            self.constraint_b[ic1] -= self.con_aref[ic1]

        for ic1 in range(n_con):
            self.x[ic1] = 0.0

    @ti.kernel
    def solve_system(self, Gammak: float, n_con: int, n_dof: int):
        for it in ti.static(range(self.n_iters)):
            # x_kplus1 = xk - Gammak*grf(A,xk,b)
            ## A.dot(x) - b
            for ni in range(n_con):
                self.Ax_b[ni] = -self.constraint_b[ni]
            for ni in range(n_con):
                for pi in range(n_con):
                    self.Ax_b[ni] += self.constraint_A[ni, pi] * self.x[pi]
            ## xgrad = A.T.dot(Ax_b)
            for pi in range(n_con):
                self.x1[pi] = self.x[pi]
                self.xgrad[pi] = 0
            for ni in range(n_con):
                for pi in range(n_con):
                    self.xgrad[pi] += self.constraint_A[ni, pi] * self.Ax_b[ni]
            ## x1 = xk - Gammak * xgrad
            for pi in range(n_con):
                self.x1[pi] -= Gammak * self.xgrad[pi]

            # projection
            for pi in range(n_con):
                self.x[pi] = ti.math.sign(self.x1[pi]) * ti.max(0.0, ti.abs(self.x1[pi]) - self.lamda)

    @ti.kernel
    def project_constraint_force(self, n_con: int, n_dof: int):
        for jd1 in range(n_dof):
            self.solver.dof_state.qf_constraint[jd1] = 0.0

        for ic1 in range(n_con):
            for jd1 in range(n_dof):
                self.solver.dof_state.qf_constraint[jd1] += self.con_jac[ic1, jd1] * self.x[ic1]

    @ti.kernel
    def cd_init(self):
        for i in range(self.cd_counter.shape[0]):
            self.cd_counter[i] = 0

    @ti.kernel
    def cd_tree_phase(self):
        for pi in range(self.collision_pairs.shape[0]):
            ga = self.collision_pairs[pi][0]
            gb = self.collision_pairs[pi][1]

            na = 2 * self.solver.geoms_info[ga].face_start
            nb = 2 * self.solver.geoms_info[gb].face_start

            head = tail = 0

            self.cd_node_que[ga, gb, tail, 0] = na
            self.cd_node_que[ga, gb, tail, 1] = nb
            tail += 1

            # print("na, nb", na, nb)

            box = self.node_data[na].box
            # print("box", na, box, self.box_data[box].umax, self.box_data[box].umin)

            box = self.node_data[nb].box
            # print("box", nb, box, self.box_data[box].umax, self.box_data[box].umin)

            while head < tail:
                self.cd_counter[1] += 1
                _h = head % self.len_cd_que
                na = self.cd_node_que[ga, gb, _h, 0]
                nb = self.cd_node_que[ga, gb, _h, 1]
                # print("na, nb", na, nb)

                head += 1
                if not self._box.overlap(self.node_data[na].box, self.node_data[nb].box):
                    continue

                if self.node_data[na].face != -1 and self.node_data[nb].face != -1:
                    # self.cd_add_face(self.node_data[na].face, self.node_data[nb].face)
                    self.cd_add_face_and_dcd(self.node_data[na].face, self.node_data[nb].face)

                elif self.node_data[na].face != -1:
                    self.cd_node_que[ga, gb, tail % self.len_cd_que, 0] = na
                    self.cd_node_que[ga, gb, tail % self.len_cd_que, 1] = self.node_data[nb].left
                    tail += 1
                    self.cd_node_que[ga, gb, tail % self.len_cd_que, 0] = na
                    self.cd_node_que[ga, gb, tail % self.len_cd_que, 1] = self.node_data[nb].right
                    tail += 1
                else:
                    self.cd_node_que[ga, gb, tail % self.len_cd_que, 0] = self.node_data[na].left
                    self.cd_node_que[ga, gb, tail % self.len_cd_que, 1] = nb
                    tail += 1
                    self.cd_node_que[ga, gb, tail % self.len_cd_que, 0] = self.node_data[na].right
                    self.cd_node_que[ga, gb, tail % self.len_cd_que, 1] = nb
                    tail += 1

    def generate_collision_shape_pairs(self):
        pairs = []
        # pairs.append([0, 1])
        for i in range(self.solver.n_geoms_max):
            for j in range(i + 1, self.solver.n_geoms_max):
                # if not self.solver.geoms_info[i].is_col or not self.solver.geoms_info[j].is_col:
                #     continue

                l_i = self.solver.geoms_info[i].link_idx
                l_j = self.solver.geoms_info[j].link_idx

                if self.solver.links_info[l_i].root_idx == self.solver.links_info[l_j].root_idx:
                    continue

                if self.solver.links_info[l_i].parent_idx == l_j or self.solver.links_info[l_j].parent_idx == l_i:
                    continue
                pairs.append([i, j])

        self.collision_pairs = ti.Vector.field(2, dtype=gs.ti_int, shape=(len(pairs)))
        self.collision_pairs.from_numpy(np.array(pairs, dtype=gs.np_int))
        print(f"generate_collision_shape_pairs done, {len(pairs)} pairs")
        print("pairs", pairs)

        # node a, node b
        self.cd_node_que = ti.field(
            dtype=gs.ti_int, shape=(self.solver.n_geoms_max, self.solver.n_geoms_max, self.len_cd_que, 2)
        )

    @ti.kernel
    def build_bvh(self):
        for i in self.solver.geoms_info:
            # if self.solver.geoms_info[i].is_col:
            self.build_one_tree(i)

    @ti.kernel
    def init_box_data(self, nf: int):
        for i in self.box_data:
            self.box_data[i].umax = ti.Vector([-np.inf, -np.inf, -np.inf])
            self.box_data[i].umin = ti.Vector([np.inf, np.inf, np.inf])
            self.box_data[i].axis = 0

    @ti.kernel
    def init_node_data(self, nf: int):
        for i in self.node_data:
            self.node_data[i].parent = -1
            self.node_data[i].left = -1
            self.node_data[i].right = -1
            self.node_data[i].depth = -1
            self.node_data[i].face = -1
            self.node_data[i].box = -1

    @ti.func
    def build_one_tree(self, link):
        offset = 3 * self.solver.geoms_info[link].face_start + link
        ototal = offset + 3 * self.solver.geoms_info[link].face_num
        otri = offset + 2 * self.solver.geoms_info[link].face_num
        onode = offset
        node_offset = 2 * self.solver.geoms_info[link].face_start
        face_offset = self.solver.geoms_info[link].face_start
        # [onode --2 nf-- otri --1 nf-- ototal]

        for i in range(self.solver.geoms_info[link].vert_start, self.solver.geoms_info[link].vert_end):
            self._box.add_vert(ototal, self.solver.verts_state[i].pos)

        self._box.update_axis(ototal)

        buffer_start = self.solver.geoms_info[link].face_start + link
        buffer_end = self.solver.geoms_info[link].face_end + link

        self.face_buffer[buffer_end] = node_offset
        left_idx = buffer_start
        right_idx = buffer_end

        for i in range(self.solver.geoms_info[link].face_start, self.solver.geoms_info[link].face_end):
            p0 = self.solver.verts_state[self.solver.faces_info[i].verts_idx[0]].pos
            p1 = self.solver.verts_state[self.solver.faces_info[i].verts_idx[1]].pos
            p2 = self.solver.verts_state[self.solver.faces_info[i].verts_idx[2]].pos

            tri_box_idx = otri + i - face_offset
            self._box.add_vert(tri_box_idx, p0)
            self._box.add_vert(tri_box_idx, p1)
            self._box.add_vert(tri_box_idx, p2)

            self.tri_centers[i] = 0.5 * (ti.min(p0, p1, p2) + ti.max(p0, p1, p2))
            if self._box.inside_left(ototal, self.tri_centers[i]):
                self.face_buffer[left_idx] = i
                left_idx += 1
            else:
                right_idx -= 1
                self.face_buffer[right_idx] = i

        # non_recursive
        # que: [my, left, n, parent]
        head = tail = 2 * self.solver.geoms_info[link].face_start

        self._box.copy_box(onode, ototal)

        self.node_data[node_offset].depth = 0
        self.node_data[node_offset].box = onode
        if self.solver.geoms_info[link].face_num == 1:
            self.node_data[node_offset].face = self.face_buffer[left_idx]
        else:
            if left_idx == buffer_start or left_idx == buffer_end:
                left_idx = buffer_start + self.solver.geoms_info[link].face_num // 2

        self.face_buffer[buffer_end] += 1
        self.que[tail, 0] = self.face_buffer[buffer_end]
        self.que[tail, 1] = buffer_start
        self.que[tail, 2] = left_idx - buffer_start
        self.que[tail, 3] = node_offset
        self.node_data[node_offset].left = self.face_buffer[buffer_end]
        tail += 1

        self.face_buffer[buffer_end] += 1
        self.que[tail, 0] = self.face_buffer[buffer_end]
        self.que[tail, 1] = left_idx
        self.que[tail, 2] = buffer_end - left_idx
        self.que[tail, 3] = node_offset
        self.node_data[node_offset].right = self.face_buffer[buffer_end]
        tail += 1

        while head < tail:
            my_id = self.que[head, 0]
            fstart = self.que[head, 1]
            n = self.que[head, 2]
            parent_id = self.que[head, 3]
            head += 1

            self.node_data[my_id].parent = parent_id
            self.node_data[my_id].depth = self.node_data[parent_id].depth + 1
            # print("depth", my_id, parent_id, fstart, n, self.node_data[my_id].depth)
            self.node_data[my_id].box = onode + my_id - node_offset

            # if self.node_data[my_id].depth > 3:
            #     continue

            if n == 1:
                self.node_data[my_id].face = self.face_buffer[fstart]
                self._box.copy_box(onode + my_id - node_offset, otri + self.face_buffer[fstart] - face_offset)

            else:
                for j in range(n):
                    self._box.add_box(onode + my_id - node_offset, otri + self.face_buffer[fstart + j] - face_offset)
                self._box.update_axis(onode + my_id - node_offset)

                if n == 2:

                    self.face_buffer[buffer_end] += 1
                    self.que[tail, 0] = self.face_buffer[buffer_end]
                    self.que[tail, 1] = fstart
                    self.que[tail, 2] = 1
                    self.que[tail, 3] = my_id
                    self.node_data[my_id].left = self.face_buffer[buffer_end]
                    tail += 1

                    self.face_buffer[buffer_end] += 1
                    self.que[tail, 0] = self.face_buffer[buffer_end]
                    self.que[tail, 1] = fstart + 1
                    self.que[tail, 2] = 1
                    self.que[tail, 3] = my_id
                    self.node_data[my_id].right = self.face_buffer[buffer_end]
                    tail += 1

                else:
                    left_idx = fstart
                    right_idx = fstart + n - 1
                    for j in range(n):
                        face_id = left_idx
                        # face_id = self.face_buffer[fstart + j]
                        if self._box.inside_left(onode + my_id - node_offset, self.tri_centers[face_id]):
                            left_idx += 1
                        else:
                            self.face_buffer[left_idx], self.face_buffer[right_idx] = (
                                self.face_buffer[right_idx],
                                self.face_buffer[left_idx],
                            )
                            right_idx -= 1

                    if left_idx == fstart or left_idx == fstart + n:
                        left_idx = fstart + n // 2

                    self.face_buffer[buffer_end] += 1
                    self.que[tail, 0] = self.face_buffer[buffer_end]
                    self.que[tail, 1] = fstart
                    self.que[tail, 2] = left_idx - fstart
                    self.que[tail, 3] = my_id
                    self.node_data[my_id].left = self.face_buffer[buffer_end]
                    tail += 1

                    self.face_buffer[buffer_end] += 1
                    self.que[tail, 0] = self.face_buffer[buffer_end]
                    self.que[tail, 1] = left_idx
                    self.que[tail, 2] = fstart + n - left_idx
                    self.que[tail, 3] = my_id
                    self.node_data[my_id].right = self.face_buffer[buffer_end]
                    tail += 1

    # test functions ---------------------

    def get_bvh_face_box_link(self, link):
        print(f"get_bvh_face_box_link {link}")
        print(self.solver.geoms_info[0].face_start)
        a = self.solver.geoms_info[link].face_start
        offset = 3 * self.solver.geoms_info[link].face_start + link
        ototal = offset + 3 * self.solver.geoms_info[link].face_num
        otri = offset + 2 * self.solver.geoms_info[link].face_num
        onode = offset
        face_offset = self.solver.geoms_info[link].face_start

        box_minmax = []
        for i in range(self.solver.geoms_info[link].face_start, self.solver.geoms_info[link].face_end):
            tri_box_idx = otri + i - face_offset
            box_minmax.append([self.box_data[tri_box_idx].umin, self.box_data[tri_box_idx].umax])

        # print(box_minmax)
        return box_minmax

    @staticmethod
    def box_to_mesh(box):
        x0, y0, z0 = box[0]
        x1, y1, z1 = box[1]
        verts = np.array(
            [
                [x1, y0, z1],
                [x1, y0, z0],
                [x1, y1, z0],
                [x1, y1, z1],
                [x0, y0, z1],
                [x0, y0, z0],
                [x0, y1, z0],
                [x0, y1, z1],
            ]
        )
        faces = np.array(
            [
                [4, 0, 3],
                [4, 3, 7],
                [0, 1, 2],
                [0, 2, 3],
                [1, 5, 6],
                [1, 6, 2],
                [5, 4, 7],
                [5, 7, 6],
                [7, 3, 2],
                [7, 2, 6],
                [0, 5, 1],
                [0, 4, 5],
            ]
        )
        return verts, faces

    def boxes_to_mesh(self, box_minmax):
        all_verts = []
        all_faces = []
        num_vert = 0
        for box in box_minmax:
            verts, faces = self.box_to_mesh(box)
            all_verts += [verts]
            all_faces += [faces + num_vert]
            num_vert += 8
        return np.concatenate(all_verts, 0), np.concatenate(all_faces, 0)

    def get_bvh_box_level_link(self, level, link):
        print(f"get_bvh_box_level_link {level} {link}")

        depth_node = []
        node_offset = 2 * self.solver.geoms_info[link].face_start
        que_node = [node_offset]

        for i_level in range(level):
            new_que = []
            num_non_leaf = 0
            num_leaf = 0
            for n in que_node:
                if self.node_data[n].left == -1:
                    num_leaf += 1
                    new_que.append(n)
                else:
                    num_non_leaf += 1
                    new_que.append(self.node_data[n].left)
                    new_que.append(self.node_data[n].right)
            depth_node.append([i_level, num_non_leaf, num_leaf])
            que_node = new_que

        depth_node = np.array(depth_node)
        print("non_leaf_node", depth_node[:, 1].sum())
        print("leaf_node", depth_node[:, 2].max())

        box_id = [self.node_data[i].box for i in que_node]
        box_minmax = [[self.box_data[i].umin, self.box_data[i].umax] for i in box_id]
        print(f"{len(box_minmax)} boxes")
        return box_minmax

    def get_impact_verts(self):
        n_impact = self.cd_counter[1]
        i0 = np.array(self.impact_info.i0.to_numpy()[:n_impact].tolist())
        i1 = np.array(self.impact_info.i1.to_numpy()[:n_impact].tolist())
        i2 = np.array(self.impact_info.i2.to_numpy()[:n_impact].tolist())
        i3 = np.array(self.impact_info.i3.to_numpy()[:n_impact].tolist())

        verts = np.array(self.solver.verts.pos.to_numpy())

        v0 = verts[i0]
        v1 = verts[i1]
        v2 = verts[i2]
        v3 = verts[i3]

        verts = np.concatenate([v0, v1, v2, v3], 0)
        return verts
