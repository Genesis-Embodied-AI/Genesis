import taichi as ti
import torch

import genesis as gs
from genesis.engine.boundaries import FloorBoundary
from genesis.engine.entities.fem_entity import FEMEntity
from genesis.engine.states.solvers import FEMSolverState

from .base_solver import Solver


@ti.data_oriented
class FEMSolver(Solver):
    # ------------------------------------------------------------------------------------
    # --------------------------------- Initialization -----------------------------------
    # ------------------------------------------------------------------------------------

    def __init__(self, scene, sim, options):
        super().__init__(scene, sim, options)

        # options
        self._floor_height = options.floor_height
        self._damping = options.damping

        # use scaled volume for better numerical stability, similar to p_vol_scale in mpm
        self._vol_scale = float(1e4)

        # materials
        self._mats = list()
        self._mats_idx = list()
        self._mats_update_stress = list()

        # boundary
        self.setup_boundary()

    def _batch_shape(self, shape=None, first_dim=False, B=None):
        if B is None:
            B = self._B

        if shape is None:
            return (B,)
        elif isinstance(shape, (list, tuple)):
            return (B,) + shape if first_dim else shape + (B,)
        else:
            return (B, shape) if first_dim else (shape, B)

    def setup_boundary(self):
        self.boundary = FloorBoundary(height=self._floor_height)

    def init_element_fields(self):
        # element state in vertices
        element_state_v = ti.types.struct(
            pos=gs.ti_vec3,  # position
            vel=gs.ti_vec3,  # velocity
        )

        # element state in elements
        element_state_el = ti.types.struct(
            actu=gs.ti_float,  # actuation
        )

        # element state without gradient
        element_state_el_ng = ti.types.struct(
            active=gs.ti_int,
        )

        # element info (properties that remain static through time)
        element_info = ti.types.struct(
            el2v=gs.ti_ivec4,  # vertex index of an element
            mu=gs.ti_float,  # lame parameters (1)
            lam=gs.ti_float,  # lame parameters (2)
            mass_scaled=gs.ti_float,  # scaled element mass. The real mass is mass_scaled / self._vol_scale
            mat_idx=gs.ti_int,  # material model index
            B=gs.ti_mat3,
            # for muscle
            muscle_group=gs.ti_int,
            muscle_direction=gs.ti_vec3,
        )

        # construct field
        self.elements_v = element_state_v.field(
            shape=self._batch_shape((self.sim.substeps_local + 1, self.n_vertices)),
            needs_grad=True,
            layout=ti.Layout.SOA,
        )
        self.elements_el = element_state_el.field(
            shape=self._batch_shape((self.sim.substeps_local + 1, self.n_elements)),
            needs_grad=True,
            layout=ti.Layout.SOA,
        )
        self.elements_el_ng = element_state_el_ng.field(
            shape=self._batch_shape((self.sim.substeps_local + 1, self.n_elements)),
            needs_grad=False,
            layout=ti.Layout.SOA,
        )
        self.elements_i = element_info.field(
            shape=(self.n_elements),
            needs_grad=False,
            layout=ti.Layout.SOA,
        )

    def init_surface_fields(self):
        # NOTE: the number of triangles won't exceed number of tetrahedrons even if bodies are changed
        n_vertices_max = self.n_vertices
        n_surfaces_max = self.n_elements

        # surface info (for coupling)
        surface_state = ti.types.struct(
            tri2v=gs.ti_ivec3,  # vertex index of a triangle
            tri2el=gs.ti_int,  # element index of a triangle
            active=gs.ti_int,
        )

        # for rendering (this is more of a surface)
        surface_state_render_v = ti.types.struct(
            vertices=gs.ti_vec3,
        )

        surface_state_render_f = ti.types.struct(
            indices=gs.ti_int,
        )

        # construct field
        self.surface = surface_state.field(
            shape=(n_surfaces_max),
            needs_grad=False,
            layout=ti.Layout.SOA,
        )

        self.surface_render_v = surface_state_render_v.field(
            shape=self._batch_shape((n_vertices_max)),
            needs_grad=False,
            layout=ti.Layout.SOA,
        )
        self.surface_render_f = surface_state_render_f.field(
            shape=(n_surfaces_max * 3),
            needs_grad=False,
            layout=ti.Layout.SOA,
        )

    def init_ckpt(self):
        self._ckpt = dict()

    def reset_grad(self):
        self.elements_v.grad.fill(0)
        self.elements_el.grad.fill(0)

        for entity in self._entities:
            entity.reset_grad()

    def build(self):
        self.n_envs = self.sim.n_envs
        self._B = self.sim._B
        # elements and bodies
        self._n_elements_max = self.n_elements
        self._n_vertices_max = self.n_vertices
        if self.n_elements_max > 0:
            self.init_element_fields()
            self.init_surface_fields()
            self.init_ckpt()

            for entity in self._entities:
                entity._add_to_solver()

    def add_entity(self, idx, material, morph, surface):
        # add material's update methods if not matching any existing material
        exist = False
        for mat in self._mats:
            if material == mat:
                material._idx = mat._idx
                exist = True
                break
        self._mats.append(material)
        if not exist:
            material._idx = len(self._mats_idx)
            self._mats_idx.append(material._idx)
            self._mats_update_stress.append(material.update_stress)

        # create entity
        entity = FEMEntity(
            scene=self._scene,
            solver=self,
            material=material,
            morph=morph,
            surface=surface,
            idx=idx,
            v_start=self.n_vertices,
            el_start=self.n_elements,
            s_start=self.n_surfaces,
        )

        self._entities.append(entity)
        return entity

    def is_active(self):
        return self.n_elements_max > 0

    # ------------------------------------------------------------------------------------
    # ----------------------------------- simulation -------------------------------------
    # ------------------------------------------------------------------------------------

    @ti.kernel
    def init_pos_and_vel(self, f: ti.i32):
        for i_v, i_b in ti.ndrange(self.n_vertices, self._B):
            self.elements_v[f + 1, i_v, i_b].pos = self.elements_v[f, i_v, i_b].pos
            self.elements_v[f + 1, i_v, i_b].vel = self.elements_v[f, i_v, i_b].vel

    @ti.kernel
    def compute_vel(self, f: ti.i32):
        for i_e, i_b in ti.ndrange(self.n_elements, self._B):
            ia, ib, ic, id = self.elements_i[i_e].el2v
            a = self.elements_v[f, ia, i_b].pos
            b = self.elements_v[f, ib, i_b].pos
            c = self.elements_v[f, ic, i_b].pos
            d = self.elements_v[f, id, i_b].pos
            D = ti.Matrix.cols([a - d, b - d, c - d])

            V_scaled = ti.abs(D.determinant()) / 6.0 * self._vol_scale
            B = self.elements_i[i_e].B
            F = D @ B
            J = F.determinant()

            stress = ti.Matrix.zero(gs.ti_float, 3, 3)
            for mat_idx in ti.static(self._mats_idx):
                if self.elements_i[i_e].mat_idx == mat_idx:
                    stress = self._mats_update_stress[mat_idx](
                        mu=self.elements_i[i_e].mu,
                        lam=self.elements_i[i_e].lam,
                        J=J,
                        F=F,
                        actu=self.elements_el[f, i_e, i_b].actu,
                        m_dir=self.elements_i[i_e].muscle_direction,
                    )

            verts = self.elements_i[i_e].el2v
            mass_scaled = self.elements_i[i_e].mass_scaled
            H_scaled = -V_scaled * stress @ B.transpose()
            dt = self.substep_dt
            for k in ti.static(range(3)):
                force_scaled = ti.Vector([H_scaled[j, k] for j in range(3)])
                dv = dt * force_scaled / mass_scaled
                self.elements_v[f + 1, verts[k], i_b].vel += dv
                self.elements_v[f + 1, verts[3], i_b].vel -= dv

    @ti.kernel
    def apply_uniform_force(self, f: ti.i32):
        for i_v, i_b in ti.ndrange(self.n_vertices, self._B):
            dt = self.substep_dt

            # NOTE: damping should only be applied to velocity from internal force and thus come first here
            #       given the immediate previous function call is compute_internal_vel --> however, shouldn't
            #       be done at dv only and need to wait for all elements updated (cannot be in the compute_internal_vel kernel)
            #       however, this inevitably damp the gravity.
            self.elements_v[f + 1, i_v, i_b].vel *= ti.exp(-dt * self.damping)
            # Add gravity (avoiding damping on gravity)
            self.elements_v[f + 1, i_v, i_b].vel += dt * self._gravity[None]

    @ti.kernel
    def compute_pos(self, f: ti.i32):
        for i_v, i_b in ti.ndrange(self.n_vertices, self._B):
            dt = self.substep_dt
            self.elements_v[f + 1, i_v, i_b].pos += dt * self.elements_v[f + 1, i_v, i_b].vel

    # ------------------------------------------------------------------------------------
    # ------------------------------------ stepping --------------------------------------
    # ------------------------------------------------------------------------------------

    def process_input(self, in_backward=False):
        for entity in self._entities:
            entity.process_input(in_backward=in_backward)

    def process_input_grad(self):
        for entity in self._entities[::-1]:
            entity.process_input_grad()

    def substep_pre_coupling(self, f):
        if self.is_active():
            self.init_pos_and_vel(f)
            self.compute_vel(f)
            self.apply_uniform_force(f)

    def substep_pre_coupling_grad(self, f):
        if self.is_active():
            self.apply_uniform_force.grad(f)
            self.compute_vel.grad(f)
            self.init_pos_and_vel.grad(f)

    def substep_post_coupling(self, f):
        if self.is_active():
            self.compute_pos(f)

    def substep_post_coupling_grad(self, f):
        if self.is_active():
            self.compute_pos.grad(f)

    @ti.kernel
    def copy_frame(self, source: ti.i32, target: ti.i32):
        # Copy pos/vel for all vertices and all batch indices
        for i_v, i_b in ti.ndrange(self.n_vertices_max, self._B):
            self.elements_v[target, i_v, i_b].pos = self.elements_v[source, i_v, i_b].pos
            self.elements_v[target, i_v, i_b].vel = self.elements_v[source, i_v, i_b].vel

        # Copy 'active' for all elements and all batch indices
        for i_e, i_b in ti.ndrange(self.n_elements_max, self._B):
            self.elements_el_ng[target, i_e, i_b].active = self.elements_el_ng[source, i_e, i_b].active

    @ti.kernel
    def copy_grad(self, source: ti.i32, target: ti.i32):
        # Copy gradients for vertices
        for i_v, i_b in ti.ndrange(self.n_vertices_max, self._B):
            self.elements_v.grad[target, i_v, i_b].pos = self.elements_v.grad[source, i_v, i_b].pos
            self.elements_v.grad[target, i_v, i_b].vel = self.elements_v.grad[source, i_v, i_b].vel

        # Copy 'active' for elements
        for i_e, i_b in ti.ndrange(self.n_elements_max, self._B):
            self.elements_el_ng[target, i_e, i_b].active = self.elements_el_ng[source, i_e, i_b].active

    @ti.kernel
    def reset_grad_till_frame(self, f: ti.i32):
        # Zero out v.grad in frame 0..(f-1) for all vertices, all batch indices
        for frame_i, vert_i, i_b in ti.ndrange((0, f), self.n_vertices_max, self._B):
            self.elements_v.grad[frame_i, vert_i, i_b].pos = 0
            self.elements_v.grad[frame_i, vert_i, i_b].vel = 0

        # Zero out elements_el.grad in frame 0..(f-1) for all elements, all batch indices
        for frame_i, elem_i, i_b in ti.ndrange((0, f), self.n_elements_max, self._B):
            self.elements_el.grad[frame_i, elem_i, i_b].actu = 0

    # ------------------------------------------------------------------------------------
    # ----------------------------------- gradient ---------------------------------------
    # ------------------------------------------------------------------------------------

    def collect_output_grads(self):
        for entity in self._entities:
            entity.collect_output_grads()

    def add_grad_from_state(self, state):
        if self.is_active():
            if state.pos.grad is not None:
                state.pos.assert_contiguous()
                self._kernel_add_grad_from_pos(self._sim.cur_substep_local, state.pos.grad)

            if state.vel.grad is not None:
                state.vel.assert_contiguous()
                self._kernel_add_grad_from_vel(self._sim.cur_substep_local, state.vel.grad)

    def save_ckpt(self, ckpt_name):
        if self.is_active():
            if not ckpt_name in self._ckpt:
                self._ckpt[ckpt_name] = dict()
                self._ckpt[ckpt_name]["pos"] = torch.zeros(
                    self._batch_shape((self.n_vertices, 3), first_dim=True), dtype=gs.tc_float
                )
                self._ckpt[ckpt_name]["vel"] = torch.zeros(
                    self._batch_shape((self.n_vertices, 3), first_dim=True), dtype=gs.tc_float
                )
                self._ckpt[ckpt_name]["active"] = torch.zeros(
                    self._batch_shape((self.n_elements,), first_dim=True), dtype=gs.tc_int
                )

            self._kernel_get_state(
                0,
                self._ckpt[ckpt_name]["pos"],
                self._ckpt[ckpt_name]["vel"],
                self._ckpt[ckpt_name]["active"],
            )

            self.copy_frame(self.sim.substeps_local, 0)

    def load_ckpt(self, ckpt_name):
        self.copy_frame(0, self._sim.substeps_local)
        self.copy_grad(0, self._sim.substeps_local)

        if self._sim.requires_grad:
            self.reset_grad_till_frame(self._sim.substeps_local)

            self._kernel_set_state(
                0,
                self._ckpt[ckpt_name]["pos"],
                self._ckpt[ckpt_name]["vel"],
                self._ckpt[ckpt_name]["active"],
            )

            for entity in self._entities:
                entity.load_ckpt(ckpt_name=ckpt_name)

    # ------------------------------------------------------------------------------------
    # --------------------------------------- io -----------------------------------------
    # ------------------------------------------------------------------------------------

    def set_state(self, f, state, envs_idx=None):
        if self.is_active():
            self._kernel_set_state(f, state.pos, state.vel, state.active)

    def get_state(self, f):
        if self.is_active():
            state = FEMSolverState(self._scene)
            self._kernel_get_state(f, state.pos, state.vel, state.active)
        else:
            state = None
        return state

    def get_state_render(self, f):
        self.get_state_render_kernel(f)
        vertices = self.surface_render_v.vertices
        indices = self.surface_render_f.indices

        return vertices, indices

    @ti.kernel
    def _kernel_add_elements(
        self,
        f: ti.i32,
        mat_idx: ti.i32,
        mat_mu: ti.f32,
        mat_lam: ti.f32,
        mat_rho: ti.f32,
        n_surfaces: ti.i32,
        v_start: ti.i32,
        el_start: ti.i32,
        s_start: ti.i32,
        verts: ti.types.ndarray(),
        elems: ti.types.ndarray(),
        tri2v: ti.types.ndarray(),
        tri2el: ti.types.ndarray(),
    ):
        n_verts_local = verts.shape[0]
        for i_v, i_b in ti.ndrange(n_verts_local, self._B):
            i_global = i_v + v_start
            for j in ti.static(range(3)):
                self.elements_v[f, i_global, i_b].pos[j] = verts[i_v, j]
            self.elements_v[f, i_global, i_b].vel = ti.Vector.zero(gs.ti_float, 3)

        n_elems_local = elems.shape[0]
        for i_e in range(n_elems_local):
            i_global = i_e + el_start

            a = self.elements_v[f, elems[i_e, 0] + v_start, 0].pos
            b = self.elements_v[f, elems[i_e, 1] + v_start, 0].pos
            c = self.elements_v[f, elems[i_e, 2] + v_start, 0].pos
            d = self.elements_v[f, elems[i_e, 3] + v_start, 0].pos
            B_inv = ti.Matrix.cols([a - d, b - d, c - d])
            self.elements_i[i_global].B = B_inv.inverse()
            V_scaled = ti.abs(B_inv.determinant()) / 6 * self._vol_scale

            for j in ti.static(range(4)):
                self.elements_i[i_global].el2v[j] = elems[i_e, j] + v_start
            self.elements_i[i_global].mat_idx = mat_idx
            self.elements_i[i_global].mu = mat_mu
            self.elements_i[i_global].lam = mat_lam
            self.elements_i[i_global].mass_scaled = mat_rho * V_scaled
            self.elements_i[i_global].muscle_group = 0
            self.elements_i[i_global].muscle_direction = ti.Vector([0.0, 0.0, 1.0], dt=gs.ti_float)

        for i_e, i_b in ti.ndrange(n_elems_local, self._B):
            i_global = i_e + el_start
            self.elements_el[f, i_global, i_b].actu = 0.0
            self.elements_el_ng[f, i_global, i_b].active = 1

        for i_s, i_b in ti.ndrange(n_surfaces, self._B):
            i_global = i_s + s_start
            for j in ti.static(range(3)):
                self.surface[i_global].tri2v[j] = tri2v[i_s, j] + v_start
            self.surface[i_global].tri2el = tri2el[i_s] + el_start
            self.surface[i_global].active = 1

    @ti.kernel
    def _kernel_set_elements_pos(
        self,
        f: ti.i32,
        element_v_start: ti.i32,
        n_vertices: ti.i32,
        pos: ti.types.ndarray(),
    ):
        for i_v, i_b in ti.ndrange(n_vertices, self._B):
            i_global = i_v + element_v_start
            for k in ti.static(range(3)):
                self.elements_v[f, i_global, i_b].pos[k] = pos[i_b, i_v, k]

    @ti.kernel
    def _kernel_set_elements_pos_grad(
        self,
        f: ti.i32,
        element_v_start: ti.i32,
        n_vertices: ti.i32,
        pos_grad: ti.types.ndarray(),
    ):
        for i_v, i_b in ti.ndrange(n_vertices, self._B):
            i_global = i_v + element_v_start
            for k in ti.static(range(3)):
                self.elements_v.grad[f, i_global, i_b].pos[k] = pos_grad[i_b, i_v, k]

    @ti.kernel
    def _kernel_set_elements_vel(
        self,
        f: ti.i32,
        element_v_start: ti.i32,
        n_vertices: ti.i32,
        vel: ti.types.ndarray(),  # shape [B, n_vertices, 3]
    ):
        for i_v, i_b in ti.ndrange(n_vertices, self._B):
            i_global = i_v + element_v_start
            for k in ti.static(range(3)):
                self.elements_v[f, i_global, i_b].vel[k] = vel[i_b, i_v, k]

    @ti.kernel
    def _kernel_set_elements_vel_grad(
        self,
        f: ti.i32,
        element_v_start: ti.i32,
        n_vertices: ti.i32,
        vel_grad: ti.types.ndarray(),  # shape [B, n_vertices, 3]
    ):
        for i_v, i_b in ti.ndrange(n_vertices, self._B):
            i_global = i_v + element_v_start
            for k in ti.static(range(3)):
                self.elements_v.grad[f, i_global, i_b].vel[k] = vel_grad[i_b, i_v, k]

    @ti.kernel
    def _kernel_set_elements_actu(
        self,
        f: ti.i32,
        element_el_start: ti.i32,
        n_elements: ti.i32,
        n_groups: ti.i32,
        actu: ti.types.ndarray(),  # shape [B, n_elements, n_groups]
    ):
        for i_e, j_g, i_b in ti.ndrange(n_elements, n_groups, self._B):
            i_global = i_e + element_el_start
            if self.elements_i[i_global].muscle_group == j_g:
                self.elements_el[f, i_global, i_b].actu = actu[i_b, j_g]

    @ti.kernel
    def _kernel_set_elements_actu_grad(
        self,
        f: ti.i32,
        element_el_start: ti.i32,
        n_elements: ti.i32,
        actu_grad: ti.types.ndarray(),  # shape [B, n_elements]
    ):
        for i_e, i_b in ti.ndrange(n_elements, self._B):
            i_global = i_e + element_el_start
            self.elements_el.grad[f, i_global, i_b].actu = actu_grad[i_b, i_e]

    @ti.kernel
    def _kernel_set_active(
        self,
        f: ti.i32,
        element_el_start: ti.i32,
        n_elements: ti.i32,
        active: ti.types.ndarray(),  # shape [B, n_elements]
    ):
        for i_e, i_b in ti.ndrange(n_elements, self._B):
            i_global = i_e + element_el_start
            self.elements_el_ng[f, i_global, i_b].active = active[i_b, i_e]

    @ti.kernel
    def _kernel_set_muscle_group(
        self,
        element_el_start: ti.i32,
        n_elements: ti.i32,
        muscle_group: ti.types.ndarray(),
    ):
        for i_e in range(n_elements):
            i_global = i_e + element_el_start
            self.elements_i[i_global].muscle_group = muscle_group[i_e]

    @ti.kernel
    def _kernel_set_muscle_direction(
        self,
        element_el_start: ti.i32,
        n_elements: ti.i32,
        muscle_direction: ti.types.ndarray(),
    ):
        for i_e in range(n_elements):
            i_global = i_e + element_el_start
            for j in ti.static(range(3)):
                self.elements_i[i_global].muscle_direction[j] = muscle_direction[i_e, j]

    @ti.kernel
    def _kernel_get_el2v(
        self,
        element_el_start: ti.i32,
        n_elements: ti.i32,
        el2v: ti.types.ndarray(),
    ):
        for i_e in range(n_elements):
            i_global = i_e + element_el_start
            for j in ti.static(range(4)):
                el2v[i_global, j] = self.elements_i[i_global].el2v[j]

    @ti.kernel
    def _kernel_get_state(
        self,
        f: ti.i32,
        pos: ti.types.ndarray(),  # shape [B, n_vertices, 3]
        vel: ti.types.ndarray(),  # shape [B, n_vertices, 3]
        active: ti.types.ndarray(),  # shape [B, n_elements]
    ):
        for i_v, i_b in ti.ndrange(self.n_vertices, self._B):
            for j in ti.static(range(3)):
                pos[i_b, i_v, j] = self.elements_v[f, i_v, i_b].pos[j]
                vel[i_b, i_v, j] = self.elements_v[f, i_v, i_b].vel[j]

        for i_e, i_b in ti.ndrange(self.n_elements, self._B):
            active[i_b, i_e] = self.elements_el_ng[f, i_e, i_b].active

    @ti.kernel
    def get_state_render_kernel(self, f: ti.i32):
        for i_v, i_b in ti.ndrange(self.n_vertices, self._B):
            for j in ti.static(range(3)):
                self.surface_render_v[i_v, i_b].vertices[j] = ti.cast(self.elements_v[f, i_v, i_b].pos[j], ti.f32)

        for i_s, i_b in ti.ndrange(self.n_surfaces, self._B):
            for j in ti.static(range(3)):
                self.surface_render_f[i_s * 3 + j].indices = ti.cast(self.surface[i_s].tri2v[j], ti.i32)

    @ti.kernel
    def _kernel_set_state(
        self,
        f: ti.i32,
        pos: ti.types.ndarray(),  # shape [B, n_vertices, 3]
        vel: ti.types.ndarray(),  # shape [B, n_vertices, 3]
        active: ti.types.ndarray(),  # shape [B, n_elements]
    ):
        for i_v, i_b in ti.ndrange(self.n_vertices, self._B):
            for j in ti.static(range(3)):
                self.elements_v[f, i_v, i_b].pos[j] = pos[i_b, i_v, j]
                self.elements_v[f, i_v, i_b].vel[j] = vel[i_b, i_v, j]

        for i_e, i_b in ti.ndrange(self.n_elements, self._B):
            self.elements_el_ng[f, i_e, i_b].active = active[i_b, i_e]

    @ti.kernel
    def _kernel_add_grad_from_pos(self, f: ti.i32, pos_grad: ti.types.ndarray()):
        for i_v, i_b in ti.ndrange(self.n_vertices, self._B):
            for j in ti.static(range(3)):
                self.elements_v.grad[f, i_v, i_b].pos[j] += pos_grad[i_b, i_v, j]

    @ti.kernel
    def _kernel_add_grad_from_vel(self, f: ti.i32, vel_grad: ti.types.ndarray()):
        for i_v, i_b in ti.ndrange(self.n_vertices, self._B):
            for j in ti.static(range(3)):
                self.elements_v.grad[f, i_v, i_b].vel[j] += vel_grad[i_b, i_v, j]

    # ------------------------------------------------------------------------------------
    # ----------------------------------- properties -------------------------------------
    # ------------------------------------------------------------------------------------

    @property
    def floor_height(self):
        return self._floor_height

    @property
    def damping(self):
        return self._damping

    @property
    def n_vertices(self):
        return sum([entity.n_vertices for entity in self._entities])

    @property
    def n_elements(self):
        return sum([entity.n_elements for entity in self._entities])

    @property
    def n_surfaces(self):
        return sum([entity.n_surfaces for entity in self.entities])

    @property
    def n_vertices_max(self):
        return self._n_vertices_max

    @property
    def n_elements_max(self):
        return self._n_elements_max

    @property
    def vol_scale(self):
        return self._vol_scale
