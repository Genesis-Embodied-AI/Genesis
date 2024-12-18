import numpy as np
import taichi as ti
import torch

import genesis as gs
import genesis.utils.element as eu
import genesis.utils.geom as gu
from genesis.engine.states.cache import QueriedStates
from genesis.engine.states.entities import FEMEntityState
from genesis.utils.misc import to_gs_tensor

from .base_entity import Entity


@ti.data_oriented
class FEMEntity(Entity):
    """
    FEM-based entity.
    """

    def __init__(self, scene, solver, material, morph, surface, idx, v_start=0, el_start=0, s_start=0):
        super().__init__(idx, scene, morph, solver, material, surface)

        self._v_start = v_start  # offset for vertex index of elements
        self._el_start = el_start  # offset for element index
        self._s_start = s_start  # offset for surface triangles
        self._step_global_added = None

        self._surface.update_texture()

        self.sample()

        self.init_tgt_vars()
        self.init_ckpt()

        self._queried_states = QueriedStates()

        self.active = False  # This attribute is only used in forward pass. It should NOT be used during backward pass.

    # ------------------------------------------------------------------------------------
    # ----------------------------------- basic entity ops -------------------------------
    # ------------------------------------------------------------------------------------

    def set_position(self, pos):
        self._assert_active()
        gs.logger.warning("Manally setting element positions. This is not recommended and could break gradient flow.")

        pos = to_gs_tensor(pos)

        if len(pos.shape) == 1:
            assert pos.shape == (3,)
            self._tgt["pos"] = self.init_positions_COM_offset + pos

        elif len(pos.shape) == 2:
            assert pos.shape == (self.n_vertices, 3)
            self._tgt["pos"] = pos

        else:
            gs.raise_exception("Tensor shape not supported.")

    def set_velocity(self, vel):
        self._assert_active()
        gs.logger.warning("Manally setting element velocities. This is not recommended and could break gradient flow.")

        vel = to_gs_tensor(vel)

        if len(vel.shape) == 1:
            assert vel.shape == (3,)
            self._tgt["vel"] = torch.tile(vel, [self.n_vertices, 1])

        elif len(vel.shape) == 2:
            assert vel.shape == (self.n_vertices, 3)
            self._tgt["vel"] = vel

        else:
            gs.raise_exception("Tensor shape not supported.")

    def set_actuation(self, actu):
        self._assert_active()

        actu = to_gs_tensor(actu)

        n_groups = getattr(self.material, "n_groups", 1)

        if len(actu.shape) == 0:
            assert actu.shape == ()
            self._tgt["actu"] = torch.tile(actu, [self.n_elements, n_groups])

        elif len(actu.shape) == 1:
            if actu.shape[0] == n_groups:
                assert self.n_elements != n_groups  # ambiguous
                actu = actu.tile([self.n_elements, 1])
            else:
                assert actu.shape == (self.n_elements,)
                gs.raise_exception("Cannot set per-element actuation")
            self._tgt["actu"] = actu

        else:
            gs.raise_exception("Tensor shape not supported.")

    def set_muscle(self, muscle_group=None, muscle_direction=None):
        self._assert_active()

        if muscle_group is not None:
            n_groups = getattr(self.material, "n_groups", 1)
            max_group_id = muscle_group.max().item()

            muscle_group = to_gs_tensor(muscle_group)

            assert muscle_group.shape == (self.n_elements,)
            assert isinstance(max_group_id, int) and max_group_id < n_groups

            self.set_muscle_group(muscle_group)

        if muscle_direction is not None:
            muscle_direction = to_gs_tensor(muscle_direction)
            assert muscle_direction.shape == (self.n_elements, 3)
            assert torch.allclose(muscle_direction.norm(dim=-1), torch.Tensor([1.0]).to(muscle_direction))

            self.set_muscle_direction(muscle_direction)

    def get_state(self):
        state = FEMEntityState(self, self._sim.cur_step_global)
        self.get_frame(
            self._sim.cur_substep_local,
            state.pos,
            state.vel,
            state.active,
        )

        # we store all queried states to track gradient flow
        self._queried_states.append(state)

        return state

    def deactivate(self):
        gs.logger.info(f"{self.__class__.__name__} <{self.id}> deactivated.")
        self._tgt["act"] = gs.INACTIVE
        self.active = False

    def activate(self):
        gs.logger.info(f"{self.__class__.__name__} <{self.id}> activated.")
        self._tgt["act"] = gs.ACTIVE
        self.active = True

    # ------------------------------------------------------------------------------------
    # ----------------------------------- instantiation ----------------------------------
    # ------------------------------------------------------------------------------------

    def instantiate(self, verts, elems):
        # rotate
        R = gu.quat_to_R(np.array(self.morph.quat))
        verts_COM = verts.mean(0)
        init_positions = (R @ (verts - verts_COM).T).T + verts_COM

        if not init_positions.shape[0] > 0:
            gs.raise_exception(f"Entity has zero vertices.")

        self.init_positions = gs.tensor(init_positions).contiguous()
        self.init_positions_COM_offset = (self.init_positions - gs.tensor(verts_COM)).contiguous()

        self.elems = elems

    def sample(self):
        if isinstance(self.morph, gs.options.morphs.Sphere):
            verts, elems = eu.sphere_to_elements(
                pos=self._morph.pos,
                radius=self._morph.radius,
                tet_cfg=self.tet_cfg,
            )
        elif isinstance(self.morph, gs.options.morphs.Box):
            verts, elems = eu.box_to_elements(
                pos=self._morph.pos,
                size=self._morph.size,
                tet_cfg=self.tet_cfg,
            )
        elif isinstance(self.morph, gs.options.morphs.Cylinder):
            verts, elems = eu.cylinder_to_elements()
        elif isinstance(self.morph, gs.options.morphs.Mesh):
            verts, elems = eu.mesh_to_elements(
                file=self._morph.file,
                pos=self._morph.pos,
                scale=self._morph.scale,
                tet_cfg=self.tet_cfg,
            )
        else:
            gs.raise_exception(f"Unsupported morph: {self.morph}.")

        self.instantiate(verts, elems)

    def _add_to_solver(self, in_backward=False):
        if not in_backward:
            self._step_global_added = self._sim.cur_step_global
            gs.logger.info(
                f"Entity {self.uid} added. class: {self.__class__.__name__}, morph: {self.morph.__class__.__name__}, size: ({self.n_elements}, {self.n_vertices}), material: {self.material}."
            )

        el2tri = np.array(
            [  # follow the order with correct normal
                [[v[0], v[2], v[1]], [v[1], v[2], v[3]], [v[0], v[1], v[3]], [v[0], v[3], v[2]]] for v in self.elems
            ]
        )
        all_tri = el2tri.reshape(-1, 3)
        all_tri_sorted = np.sort(all_tri, axis=1)
        _, unique_idcs, cnt = np.unique(all_tri_sorted, axis=0, return_counts=True, return_index=True)
        unique_tri = all_tri[unique_idcs]
        surface_tri = unique_tri[cnt == 1]
        surface_tri = surface_tri.astype(gs.np_int)
        self._n_surfaces = len(surface_tri)
        self._n_surface_vertices = len(np.unique(surface_tri))

        tri2el = np.repeat(np.arange(self.elems.shape[0])[:, None], 4, axis=-1)
        all_el = tri2el.reshape(
            -1,
        )
        unique_el = all_el[unique_idcs]
        surface_el = unique_el[cnt == 1].astype(gs.np_int)

        self._solver._kernel_add_elements(
            f=self._sim.cur_substep_local,
            mat_idx=self._material.idx,
            mat_mu=self._material.mu,
            mat_lam=self._material.lam,
            mat_rho=self._material.rho,
            n_surfaces=self._n_surfaces,
            v_start=self._v_start,
            el_start=self._el_start,
            s_start=self._s_start,
            verts=self.init_positions,
            elems=self.elems,
            tri2v=surface_tri,
            tri2el=surface_el,
        )
        self.active = True

    # ------------------------------------------------------------------------------------
    # ---------------------------- checkpoint and buffer ---------------------------------
    # ------------------------------------------------------------------------------------

    def init_tgt_keys(self):
        self._tgt_keys = ["vel", "pos", "act", "actu"]

    def init_tgt_vars(self):
        # temp variable to store targets for next step
        self._tgt = dict()
        self._tgt_buffer = dict()
        self.init_tgt_keys()

        for key in self._tgt_keys:
            self._tgt[key] = None
            self._tgt_buffer[key] = list()

    def init_ckpt(self):
        self._ckpt = dict()

    def save_ckpt(self, ckpt_name):
        if not ckpt_name in self._ckpt:
            self._ckpt[ckpt_name] = {
                "_tgt_buffer": dict(),
            }

        for key in self._tgt_keys:
            self._ckpt[ckpt_name]["_tgt_buffer"][key] = list(self._tgt_buffer[key])
            self._tgt_buffer[key].clear()

    def load_ckpt(self, ckpt_name):
        for key in self._tgt_keys:
            self._tgt_buffer[key] = list(self._ckpt[ckpt_name]["_tgt_buffer"][key])

    def reset_grad(self):
        for key in self._tgt_keys:
            self._tgt_buffer[key].clear()
        self._queried_states.clear()

    def process_input(self, in_backward=False):
        if in_backward:
            # use negative index because buffer length might not be full
            index = self._sim.cur_step_local - self._sim.max_steps_local
            for key in self._tgt_keys:
                self._tgt[key] = self._tgt_buffer[key][index]

        else:
            for key in self._tgt_keys:
                self._tgt_buffer[key].append(self._tgt[key])

        # set_pos followed by set_vel, because set_pos resets velocity.
        if self._tgt["pos"] is not None:
            self._tgt["pos"].assert_contiguous()
            self._tgt["pos"].assert_sceneless()
            self.set_pos(self._sim.cur_substep_local, self._tgt["pos"])

        if self._tgt["vel"] is not None:
            self._tgt["vel"].assert_contiguous()
            self._tgt["vel"].assert_sceneless()
            self.set_vel(self._sim.cur_substep_local, self._tgt["vel"])

        if self._tgt["act"] is not None:
            assert self._tgt["act"] in [gs.ACTIVE, gs.INACTIVE]
            self.set_active(self._sim.cur_substep_local, self._tgt["act"])

        if self._tgt["actu"] is not None:
            self._tgt["actu"].assert_contiguous()
            self._tgt["actu"].assert_sceneless()
            self.set_actu(self._sim.cur_substep_local, self._tgt["actu"])

        for key in self._tgt_keys:
            self._tgt[key] = None

    def process_input_grad(self):
        _tgt_actu = self._tgt_buffer["actu"].pop()
        _tgt_vel = self._tgt_buffer["vel"].pop()
        _tgt_pos = self._tgt_buffer["pos"].pop()

        if _tgt_actu is not None and _tgt_actu.requires_grad:
            _tgt_actu._backward_from_ti(self.set_actu_grad, self._sim.cur_substep_local)

        if _tgt_vel is not None and _tgt_vel.requires_grad:
            _tgt_vel._backward_from_ti(self.set_vel_grad, self._sim.cur_substep_local)

        if _tgt_pos is not None and _tgt_pos.requires_grad:
            _tgt_pos._backward_from_ti(self.set_pos_grad, self._sim.cur_substep_local)

        if _tgt_vel is not None or _tgt_pos is not None or _tgt_actu is not None:
            # manually zero the grad since manually setting state breaks gradient flow
            self.clear_grad(self._sim.cur_substep_local)

    def _assert_active(self):
        if not self.active:
            gs.raise_exception(f"{self.__class__.__name__} is inactive. Call `entity.activate()` first.")

    # ------------------------------------------------------------------------------------
    # ---------------------------- interfacing with solver -------------------------------
    # ------------------------------------------------------------------------------------

    def set_pos(self, f, pos):
        self.solver._kernel_set_elements_pos(
            f=f,
            element_v_start=self._v_start,
            n_vertices=self.n_vertices,
            pos=pos,
        )

    def set_pos_grad(self, f, pos_grad):
        self.solver._kernel_set_elements_pos_grad(
            f=f,
            element_v_start=self._v_start,
            n_vertices=self.n_vertices,
            pos_grad=pos_grad,
        )

    def set_vel(self, f, vel):
        self.solver._kernel_set_elements_vel(
            f=f,
            element_v_start=self._v_start,
            n_vertices=self.n_vertices,
            vel=vel,
        )

    def set_vel_grad(self, f, vel_grad):
        self.solver._kernel_set_elements_vel_grad(
            f=f,
            element_v_start=self._v_start,
            n_vertices=self.n_vertices,
            vel_grad=vel_grad,
        )

    def set_actu(self, f, actu):
        self.solver._kernel_set_elements_actu(
            f=f,
            element_el_start=self._el_start,
            n_elements=self.n_elements,
            n_groups=self.material.n_groups,
            actu=actu,
        )

    def set_actu_grad(self, f, actu_grad):
        self.solver._kernel_set_elements_actu(
            f=f,
            element_el_start=self._el_start,
            n_elements=self.n_elements,
            actu_grad=actu_grad,
        )

    def set_active(self, f, active):
        self.solver._kernel_set_active(
            f=f,
            element_el_start=self._el_start,
            n_elements=self.n_elements,
            active=active,
        )

    def set_muscle_group(self, muscle_group):
        self.solver._kernel_set_muscle_group(
            element_el_start=self._el_start,
            n_elements=self.n_elements,
            muscle_group=muscle_group,
        )

    def set_muscle_direction(self, muscle_direction):
        self.solver._kernel_set_muscle_direction(
            element_el_start=self._el_start,
            n_elements=self.n_elements,
            muscle_direction=muscle_direction,
        )

    def get_el2v(self):
        el2v = gs.zeros((self.n_elements, 4), dtype=int, requires_grad=False, scene=self.scene)
        self.solver._kernel_get_el2v(
            element_el_start=self._el_start,
            n_elements=self.n_elements,
            el2v=el2v,
        )

        return el2v

    @ti.kernel
    def get_frame(self, f: ti.i32, pos: ti.types.ndarray(), vel: ti.types.ndarray(), active: ti.types.ndarray()):
        for i in range(self.n_vertices):
            i_global = i + self.v_start
            for j in ti.static(range(3)):
                pos[i, j] = self._solver.elements_v[f, i_global].pos[j]
                vel[i, j] = self._solver.elements_v[f, i_global].vel[j]

        for i in range(self.n_elements):
            i_global = i + self.el_start
            active[i] = self._solver.elements_el_ng[f, i_global].active

    @ti.kernel
    def clear_grad(self, f: ti.i32):
        # TODO: not well-tested
        for i in range(self.n_vertices):
            i_global = i + self.v_start
            self._solver.elements_v.grad[f, i_global].pos = 0
            self._solver.elements_v.grad[f, i_global].vel = 0

        for i in range(self.n_elements):
            i_global = i + self.el_start
            self._solver.elements_el.grad[f, i_global].actu = 0

    # ------------------------------------------------------------------------------------
    # ----------------------------------- properties -------------------------------------
    # ------------------------------------------------------------------------------------

    @property
    def n_vertices(self):
        return len(self.init_positions)

    @property
    def n_elements(self):
        return len(self.elems)

    @property
    def n_surfaces(self):
        return self._n_surfaces

    @property
    def v_start(self):
        return self._v_start

    @property
    def el_start(self):
        return self._el_start

    @property
    def s_start(self):
        return self._s_start

    @property
    def morph(self):
        return self._morph

    @property
    def material(self):
        return self._material

    @property
    def surface(self):
        return self._surface

    @property
    def n_surface_vertices(self):
        return self._n_surface_vertices

    @property
    def tet_cfg(self):
        tet_cfg = dict(
            order=getattr(self.morph, "order", 1),
            mindihedral=getattr(self.morph, "mindihedral", 10),
            minratio=getattr(self.morph, "minratio", 1.1),
            nobisect=getattr(self.morph, "nobisect", True),
            quality=getattr(self.morph, "quality", True),
            verbose=getattr(self.morph, "verbose", 0),
        )
        return tet_cfg
