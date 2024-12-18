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
            shape=(self.sim.substeps_local + 1, self.n_vertices),
            needs_grad=True,
            layout=ti.Layout.SOA,
        )
        self.elements_el = element_state_el.field(
            shape=(self.sim.substeps_local + 1, self.n_elements),
            needs_grad=True,
            layout=ti.Layout.SOA,
        )
        self.elements_el_ng = element_state_el_ng.field(
            shape=(self.sim.substeps_local + 1, self.n_elements),
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
            shape=(n_vertices_max),
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
        for i in range(self.n_vertices):
            self.elements_v[f + 1, i].pos = self.elements_v[f, i].pos
            self.elements_v[f + 1, i].vel = self.elements_v[f, i].vel

    @ti.kernel
    def compute_vel(self, f: ti.i32):
        for i in range(self.n_elements):
            ia, ib, ic, id = self.elements_i[i].el2v
            a = self.elements_v[f, ia].pos
            b = self.elements_v[f, ib].pos
            c = self.elements_v[f, ic].pos
            d = self.elements_v[f, id].pos
            D = ti.Matrix.cols([a - d, b - d, c - d])

            V_scaled = ti.abs(D.determinant()) / 6.0 * self._vol_scale
            B = self.elements_i[i].B
            F = D @ B
            J = F.determinant()

            stress = ti.Matrix.zero(gs.ti_float, 3, 3)
            for mat_idx in ti.static(self._mats_idx):
                if self.elements_i[i].mat_idx == mat_idx:
                    stress = self._mats_update_stress[mat_idx](
                        mu=self.elements_i[i].mu,
                        lam=self.elements_i[i].lam,
                        J=J,
                        F=F,
                        actu=self.elements_el[f, i].actu,
                        m_dir=self.elements_i[i].muscle_direction,
                    )

            verts = self.elements_i[i].el2v
            mass_scaled = self.elements_i[i].mass_scaled
            H_scaled = -V_scaled * stress @ B.transpose()
            dt = self.substep_dt
            for k in ti.static(range(3)):
                force_scaled = ti.Vector([H_scaled[j, k] for j in range(3)])
                # scaling equivalent to dt * force / (mass_scaled / self._vol_scale) = dt * (force_scaled / self._vol_scale) / (mass_scaled / self._vol_scale)
                dv = dt * force_scaled / mass_scaled
                self.elements_v[f + 1, verts[k]].vel += dv
                self.elements_v[f + 1, verts[3]].vel -= dv

    @ti.kernel
    def apply_uniform_force(self, f: ti.i32):
        for i in range(self.n_vertices):
            dt = self.substep_dt

            # NOTE: damping should only be applied to velocity from internal force and thus come first here
            #       given the immediate previous function call is compute_internal_vel --> however, shouldn't
            #       be done at dv only and need to wait for all elements updated (cannot be in the compute_internal_vel kernel)
            #       however, this inevitably damp the gravity.
            self.elements_v[f + 1, i].vel *= ti.exp(-dt * self.damping)
            self.elements_v[f + 1, i].vel += dt * self._gravity[None]  # avoid applying damping to gravity

    @ti.kernel
    def compute_pos(self, f: ti.i32):
        for i in range(self.n_vertices):
            dt = self.substep_dt
            self.elements_v[f + 1, i].pos += dt * self.elements_v[f + 1, i].vel

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
        for i in range(self.n_vertices_max):
            self.elements_v[target, i].pos = self.elements_v[source, i].pos
            self.elements_v[target, i].vel = self.elements_v[source, i].vel

        for i in range(self.n_elements_max):
            self.elements_el_ng[target, i].active = self.elements_el_ng[source, i].active

    @ti.kernel
    def copy_grad(self, source: ti.i32, target: ti.i32):
        for i in range(self.n_vertices_max):
            self.elements_v.grad[target, i].pos = self.elements_v.grad[source, i].pos
            self.elements_v.grad[target, i].vel = self.elements_v.grad[source, i].vel

        for i in range(self.n_elements_max):
            self.elements_el_ng[target, i].active = self.elements_el_ng[source, i].active

    @ti.kernel
    def reset_grad_till_frame(self, f: ti.i32):
        for i, j in ti.ndrange(f, self.n_vertices_max):
            self.elements_v.grad[i, j].pos = 0
            self.elements_v.grad[i, j].vel = 0

        for i in range(self.n_elements_max):
            self.elements_el.grad[i, i].actu = 0

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
                self._ckpt[ckpt_name]["pos"] = torch.zeros((self.n_vertices, 3), dtype=gs.tc_float)
                self._ckpt[ckpt_name]["vel"] = torch.zeros((self.n_vertices, 3), dtype=gs.tc_float)
                self._ckpt[ckpt_name]["active"] = torch.zeros((self.n_elements,), dtype=gs.tc_int)

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

    def set_state(self, f, state):
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
        for i in range(self.n_vertices):
            i_global = i + v_start
            for j in ti.static(range(3)):
                self.elements_v[f, i_global].pos[j] = verts[i, j]
            self.elements_v[f, i_global].vel = ti.Vector.zero(gs.ti_float, 3)

        for i in range(self.n_elements):
            i_global = i + el_start

            a = self.elements_v[f, elems[i, 0] + v_start].pos
            b = self.elements_v[f, elems[i, 1] + v_start].pos
            c = self.elements_v[f, elems[i, 2] + v_start].pos
            d = self.elements_v[f, elems[i, 3] + v_start].pos
            B_inv = ti.Matrix.cols([a - d, b - d, c - d])
            self.elements_i[i_global].B = B_inv.inverse()
            V_scaled = ti.abs(B_inv.determinant()) / 6 * self._vol_scale

            self.elements_el[f, i_global].actu = 0.0

            self.elements_el_ng[f, i_global].active = 1

            for j in ti.static(range(4)):
                self.elements_i[i_global].el2v[j] = elems[i, j] + v_start
            self.elements_i[i_global].mat_idx = mat_idx
            self.elements_i[i_global].mu = mat_mu
            self.elements_i[i_global].lam = mat_lam
            self.elements_i[i_global].mass_scaled = mat_rho * V_scaled
            self.elements_i[i_global].muscle_group = 0
            self.elements_i[i_global].muscle_direction = ti.Vector([0.0, 0.0, 1.0], dt=gs.ti_float)

        for i in range(n_surfaces):
            i_global = i + s_start
            for j in ti.static(range(3)):
                self.surface[i_global].tri2v[j] = tri2v[i, j] + v_start
            self.surface[i_global].tri2el = tri2el[i] + el_start
            self.surface[i_global].active = 1

    @ti.kernel
    def _kernel_set_elements_pos(
        self,
        f: ti.i32,
        element_v_start: ti.i32,
        n_vertices: ti.i32,
        pos: ti.types.ndarray(),
    ):
        for i in range(n_vertices):
            i_global = i + element_v_start
            for k in ti.static(range(3)):
                self.elements_v[f, i_global].pos[k] = pos[i, k]

    @ti.kernel
    def _kernel_set_elements_pos_grad(
        self,
        f: ti.i32,
        element_v_start: ti.i32,
        n_vertices: ti.i32,
        pos_grad: ti.types.ndarray(),
    ):
        for i in range(n_vertices):
            i_global = i + element_v_start
            for k in ti.static(range(3)):
                self.elements_v.grad[f, i_global].pos[k] = pos_grad[i, k]

    @ti.kernel
    def _kernel_set_elements_vel(
        self,
        f: ti.i32,
        element_v_start: ti.i32,
        n_vertices: ti.i32,
        vel: ti.types.ndarray(),
    ):
        for i in range(n_vertices):
            i_global = i + element_v_start
            for k in ti.static(range(3)):
                self.elements_v[f, i_global].vel[k] = vel[i, k]

    @ti.kernel
    def _kernel_set_elements_vel_grad(
        self,
        f: ti.i32,
        element_v_start: ti.i32,
        n_vertices: ti.i32,
        vel_grad: ti.types.ndarray(),
    ):
        for i in range(n_vertices):
            i_global = i + element_v_start
            for k in ti.static(range(3)):
                self.elements_v.grad[f, i_global].vel[k] = vel_grad[i, k]

    @ti.kernel
    def _kernel_set_elements_actu(
        self,
        f: ti.i32,
        element_el_start: ti.i32,
        n_elements: ti.i32,
        n_groups: ti.i32,
        actu: ti.types.ndarray(),
    ):
        for i in range(n_elements):
            i_global = i + element_el_start
            for j in range(n_groups):
                if self.elements_i[i_global].muscle_group == j:
                    self.elements_el[f, i_global].actu = actu[i, j]

    @ti.kernel
    def _kernel_set_elements_actu_grad(
        self,
        f: ti.i32,
        element_el_start: ti.i32,
        n_elements: ti.i32,
        actu_grad: ti.types.ndarray(),
    ):
        for i in range(n_elements):
            i_global = i + element_el_start
            self.elements_el.grad[f, i_global].actu = actu_grad[i]

    @ti.kernel
    def _kernel_set_active(
        self,
        f: ti.i32,
        element_el_start: ti.i32,
        n_elements: ti.i32,
        active: ti.types.ndarray(),
    ):
        for i in range(n_elements):
            i_global = i + element_el_start
            self.elements_el_ng[f, i_global].active = active

    @ti.kernel
    def _kernel_set_muscle_group(
        self,
        element_el_start: ti.i32,
        n_elements: ti.i32,
        muscle_group: ti.types.ndarray(),
    ):
        for i in range(n_elements):
            i_global = i + element_el_start
            self.elements_i[i_global].muscle_group = muscle_group[i]

    @ti.kernel
    def _kernel_set_muscle_direction(
        self,
        element_el_start: ti.i32,
        n_elements: ti.i32,
        muscle_direction: ti.types.ndarray(),
    ):
        for i in range(n_elements):
            i_global = i + element_el_start
            for j in ti.static(range(3)):
                self.elements_i[i_global].muscle_direction[j] = muscle_direction[i, j]

    @ti.kernel
    def _kernel_get_el2v(
        self,
        element_el_start: ti.i32,
        n_elements: ti.i32,
        el2v: ti.types.ndarray(),
    ):
        for i in range(n_elements):
            i_global = i + element_el_start
            for j in ti.static(range(4)):
                el2v[i_global, j] = self.elements_i[i_global].el2v[j]

    @ti.kernel
    def _kernel_get_state(
        self,
        f: ti.i32,
        pos: ti.types.ndarray(),
        vel: ti.types.ndarray(),
        active: ti.types.ndarray(),
    ):
        for i in range(self.n_vertices):
            for j in ti.static(range(3)):
                pos[i, j] = self.elements_v[f, i].pos[j]
                vel[i, j] = self.elements_v[f, i].vel[j]

        for i in range(self.n_elements):
            active[i] = self.elements_el_ng[f, i].active

    @ti.kernel
    def get_state_render_kernel(self, f: ti.i32):
        for i in range(self.n_vertices):
            for j in ti.static(range(3)):
                self.surface_render_v[i].vertices[j] = ti.cast(self.elements_v[f, i].pos[j], gs.ti_float)

        for i in range(self.n_surfaces):
            for j in ti.static(range(3)):
                self.surface_render_f[i * 3 + j].indices = ti.cast(self.surface[i].tri2v[j], gs.ti_int)

    @ti.kernel
    def _kernel_set_state(
        self,
        f: ti.i32,
        pos: ti.types.ndarray(),
        vel: ti.types.ndarray(),
        active: ti.types.ndarray(),
    ):
        for i in range(self.n_vertices):
            for j in ti.static(range(3)):
                self.elements_v[f, i].pos[j] = pos[i, j]
                self.elements_v[f, i].vel[j] = vel[i, j]

        for i in range(self.n_elements):
            self.elements_el_ng[f, i].active = active[i]

    @ti.kernel
    def _kernel_add_grad_from_pos(self, f: ti.i32, pos_grad: ti.types.ndarray()):
        for i in range(self.n_vertices):
            for j in ti.static(range(3)):
                self.elements_v.grad[f, i].pos[j] += pos_grad[i, j]

    @ti.kernel
    def _kernel_add_grad_from_vel(self, f: ti.i32, vel_grad: ti.types.ndarray()):
        for i in range(self.n_vertices):
            for j in ti.static(range(3)):
                self.elements_v.grad[f, i].vel[j] += vel_grad[i, j]

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
