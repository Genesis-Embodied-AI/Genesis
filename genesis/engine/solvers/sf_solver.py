import numpy as np
import taichi as ti
import torch

import genesis.utils.geom as gu
from genesis.utils.misc import *

from .base_solver import Solver


@ti.data_oriented
class SFSolver(Solver):
    """
    Stable Fluid solver for eulerian-based gaseous simulation.
    """

    def __init__(self, scene, sim, options, res=128, dt=0.03, solver_iters=500, q_dim=3, decay=0.99):
        super().__init__(scene, sim, options)

        if options is None:
            return

        self.n_grid = res
        self.dx = 1 / self.n_grid
        self.res = (res, res, res)
        self.solver_iters = solver_iters
        self.q_dim = q_dim
        self.decay = decay
        self.high_T = 1.0
        self.low_T = 0.0
        self.lower_y = 60
        self.higher_y = 68
        self.lower_y_vis = 60
        self.higher_y_vis = 68
        self.hot_color = ti.Vector([1.0, 0.45, 0.14, 0.8])
        self.cold_color = ti.Vector([0.0, 0.55, 1.0, 0.8])

    def build(self):
        return

        self.mpm_sim = mpm_sim
        self.max_steps_local = mpm_sim.max_steps_local
        self.mpm_grid_ratio = self.n_grid / mpm_sim.n_grid
        self.mpm_v_coeff = 1.0
        self.agent = mpm_sim.agent

        self.setup_fields()
        self.init_fields()
        self.init_ckpt()

    def is_active(self):
        return False

    def init_ckpt(self):
        self.ckpt_ram = dict()

    def setup_fields(self):
        """
        Taichi fields for smoke simulation.
        """
        cell_state = ti.types.struct(
            v=ti.types.vec3_ti,
            v_tmp=ti.types.vec3_ti,
            div=gs.ti_float,
            p=gs.ti_float,
            q=ti.types.vector(self.q_dim, gs.ti_float),
        )

        cell_state_ng = ti.types.struct(
            is_free=gs.ti_int,
        )

        self.grid = cell_state.field(shape=(self.max_steps_local + 1, *self.res), needs_grad=True, layout=ti.Layout.SOA)
        self.grid_ng = cell_state_ng.field(
            shape=(self.max_steps_local + 1, *self.res), needs_grad=False, layout=ti.Layout.SOA
        )

        # swap area for pressure projection solver
        self.p_swap = TexPair(
            cur=ti.field(dtype=gs.ti_float, shape=self.res, needs_grad=True),
            nxt=ti.field(dtype=gs.ti_float, shape=self.res, needs_grad=True),
        )

        self.vis_particles = ti.Vector.field(3, float, shape=np.prod(self.res))
        self.vis_particles_c = ti.Vector.field(4, float, shape=np.prod(self.res))

    @ti.kernel
    def init_fields(self):
        for i, j, k in ti.ndrange(*self.res):
            ind = j * self.n_grid * self.n_grid + i * self.n_grid + k
            self.vis_particles[ind] = (ti.Vector([i, j, k], dt=gs.ti_float) + 0.5) * self.dx
            if self.lower_y < j < self.higher_y:
                self.grid[0, i, j, k].q = ti.Vector([self.high_T])

    def process_input(self, in_backward=False):
        return

    def process_input_grad(self):
        return

    def step(self, s, f):
        return
        self.compute_free_space(s, f)
        self.advect_and_impulse(s, f)
        self.divergence(s)

        # projection
        self.reset_swap_and_grad()
        self.pressure_to_swap(s)
        for i in range(self.solver_iters):
            self.pressure_jacobi(self.p_swap.cur, self.p_swap.nxt, s)
            self.p_swap.swap()
        self.pressure_from_swap(s)
        self.reset_swap_and_grad()

        self.subtract_gradient(s)
        self.colorize(s)

    def substep_pre_coupling(self, f):
        return

    def substep_pre_coupling_grad(self, f):
        return

    def substep_post_coupling(self, f):
        return

    def substep_post_coupling_grad(self, f):
        return

    def _step_grad(self, s, f):
        return
        self.compute_free_space(s, f)

        self.subtract_gradient.grad(s)

        self.reset_swap_and_grad()
        self.pressure_from_swap.grad(s)
        for i in range(self.solver_iters - 1, -1, -1):
            self.p_swap.swap()
            self.p_swap.cur.grad.fill(0)
            self.pressure_jacobi.grad(self.p_swap.cur, self.p_swap.nxt, s)
        self.pressure_to_swap.grad(s)
        self.reset_swap_and_grad()

        self.divergence.grad(s)
        self.advect_and_impulse.grad(s, f)

    def add_grad_from_state(self, state):
        pass

    def reset_swap_and_grad(self):
        self.p_swap.cur.fill(0)
        self.p_swap.nxt.fill(0)
        self.p_swap.cur.grad.fill(0)
        self.p_swap.nxt.grad.fill(0)

    @ti.kernel
    def pressure_jacobi(self, pf: ti.template(), new_pf: ti.template(), s: ti.i32):
        for i, j, k in ti.ndrange(*self.res):
            if self.grid_ng[s, i, j, k].is_free:
                pl = pf[self.compute_location(s, i, j, k, -1, 0, 0)]
                pr = pf[self.compute_location(s, i, j, k, 1, 0, 0)]
                pb = pf[self.compute_location(s, i, j, k, 0, -1, 0)]
                pt = pf[self.compute_location(s, i, j, k, 0, 1, 0)]
                pp = pf[self.compute_location(s, i, j, k, 0, 0, -1)]
                pq = pf[self.compute_location(s, i, j, k, 0, 0, 1)]

                new_pf[i, j, k] = (pl + pr + pb + pt + pp + pq - self.grid[s, i, j, k].div) / 6.0

    @ti.kernel
    def pressure_jacobi_grad(self, pf: ti.template(), new_pf: ti.template(), s: ti.i32):
        for i, j, k in ti.ndrange(*self.res):
            if self.grid_ng[s, i, j, k].is_free:

                self.grid.grad[s, i, j, k].div += -1.0 / 6.0 * new_pf.grad[i, j, k]

                pf.grad[self.compute_location(s, i, j, k, -1, 0, 0)] += 1.0 / 6.0 * new_pf.grad[i, j, k]
                pf.grad[self.compute_location(s, i, j, k, 1, 0, 0)] += 1.0 / 6.0 * new_pf.grad[i, j, k]
                pf.grad[self.compute_location(s, i, j, k, 0, -1, 0)] += 1.0 / 6.0 * new_pf.grad[i, j, k]
                pf.grad[self.compute_location(s, i, j, k, 0, 1, 0)] += 1.0 / 6.0 * new_pf.grad[i, j, k]
                pf.grad[self.compute_location(s, i, j, k, 0, 0, -1)] += 1.0 / 6.0 * new_pf.grad[i, j, k]
                pf.grad[self.compute_location(s, i, j, k, 0, 0, 1)] += 1.0 / 6.0 * new_pf.grad[i, j, k]

    @ti.kernel
    def copy_frame(self, source: ti.i32, target: ti.i32):
        for i, j, k in ti.ndrange(*self.res):
            self.grid[target, i, j, k].v = self.grid[source, i, j, k].v
            self.grid[target, i, j, k].v_tmp = self.grid[source, i, j, k].v_tmp
            self.grid[target, i, j, k].div = self.grid[source, i, j, k].div
            self.grid[target, i, j, k].p = self.grid[source, i, j, k].p
            self.grid[target, i, j, k].q = self.grid[source, i, j, k].q

    @ti.kernel
    def copy_grad(self, source: ti.i32, target: ti.i32):
        for i, j, k in ti.ndrange(*self.res):
            self.grid.grad[target, i, j, k].v = self.grid.grad[source, i, j, k].v
            self.grid.grad[target, i, j, k].v_tmp = self.grid.grad[source, i, j, k].v_tmp
            self.grid.grad[target, i, j, k].div = self.grid.grad[source, i, j, k].div
            self.grid.grad[target, i, j, k].p = self.grid.grad[source, i, j, k].p
            self.grid.grad[target, i, j, k].q = self.grid.grad[source, i, j, k].q

    def reset_grad(self):
        return
        self.grid.grad.fill(0)
        self.p_swap.cur.grad.fill(0)
        self.p_swap.nxt.grad.fill(0)

    @ti.kernel
    def reset_grad_till_frame(self, s: ti.i32):
        for n, i, j, k in ti.ndrange(s, *self.res):
            self.grid.grad[n, i, j, k].fill(0)

    @ti.kernel
    def compute_free_space(self, s: ti.i32, f: ti.i32):
        for i, j, k in ti.ndrange(*self.res):
            self.grid_ng[s, i, j, k].is_free = 0
            if self.lower_y < j < self.higher_y:
                self.grid_ng[s, i, j, k].is_free = 1

            p = ti.Vector([i, j, k], dt=gs.ti_float) + 0.5
            if ti.static(self.mpm_sim.n_statics > 0):
                for static_i in ti.static(range(self.mpm_sim.n_statics)):
                    if self.mpm_sim.statics[static_i].is_collide(p * self.dx):
                        self.grid_ng[s, i, j, k].is_free = 0

    @ti.kernel
    def advect_and_impulse(self, s: ti.i32, f: ti.i32):
        for i, j, k in ti.ndrange(*self.res):
            if self.grid_ng[s, i, j, k].is_free:
                p = ti.Vector([i, j, k], dt=gs.ti_float) + 0.5

                p = self.backtrace(s, self.grid.v, p, self.dt)
                v_f = self.trilerp(s, self.grid.v, p) * 1
                q_f = self.trilerp(s, self.grid.q, p) * 1

                # apply agent impulse
                imp_pos = self.agent.aircon.pos[f] / self.dx
                imp_dir = gu.ti_transform_by_quat(self.agent.aircon.inject_v, self.agent.aircon.quat[f])
                dist = (ti.Vector([i, j, k]) - imp_pos).norm(EPS)
                factor = ti.exp(-dist / self.agent.aircon.r[f])
                momentum = (imp_dir * self.agent.aircon.s[f] * factor) * self.dt

                # compute impulse from mpm particles
                momentum_mpm = ti.Vector([0.0, 0.0, 0.0], dt=gs.ti_float)
                # I_mpm = ti.floor(ti.Vector([i, j, k], dt=gs.ti_float) / self.mpm_grid_ratio, gs.ti_int)
                # if self.mpm_sim.grid[f, I_mpm].mass > EPS:
                #     momentum_mpm = self.mpm_sim.grid[f, I_mpm].v_out * self.mpm_v_coeff

                v_tmp = v_f + momentum + momentum_mpm

                self.grid.v_tmp[s, i, j, k] = v_tmp
                self.grid.q[s + 1, i, j, k] = (1 - factor) * q_f + factor * ti.Vector([self.low_T])

            else:
                self.grid.v_tmp[s, i, j, k] = ti.Vector([0.0, 0.0, 0.0])
                self.grid.q[s + 1, i, j, k] = self.grid.q[s, i, j, k]

    @ti.kernel
    def divergence(self, s: ti.i32):
        for i, j, k in ti.ndrange(*self.res):
            if self.grid_ng[s, i, j, k].is_free:
                vl = self.grid.v_tmp[s, self.compute_location(s, i, j, k, -1, 0, 0)]
                vr = self.grid.v_tmp[s, self.compute_location(s, i, j, k, 1, 0, 0)]
                vb = self.grid.v_tmp[s, self.compute_location(s, i, j, k, 0, -1, 0)]
                vt = self.grid.v_tmp[s, self.compute_location(s, i, j, k, 0, 1, 0)]
                vp = self.grid.v_tmp[s, self.compute_location(s, i, j, k, 0, 0, -1)]
                vq = self.grid.v_tmp[s, self.compute_location(s, i, j, k, 0, 0, 1)]
                vc = self.grid.v_tmp[s, self.compute_location(s, i, j, k, 0, 0, 0)]

                if not self.is_free(s, i, j, k, -1, 0, 0):
                    vl.x = -vc.x
                if not self.is_free(s, i, j, k, 1, 0, 0):
                    vr.x = -vc.x
                if not self.is_free(s, i, j, k, 0, -1, 0):
                    vb.y = -vc.y
                if not self.is_free(s, i, j, k, 0, 1, 0):
                    vt.y = -vc.y
                if not self.is_free(s, i, j, k, 0, 0, -1):
                    vp.z = -vc.z
                if not self.is_free(s, i, j, k, 0, 0, 1):
                    vq.z = -vc.z

                self.grid.div[s, i, j, k] = (vr.x - vl.x + vt.y - vb.y + vq.z - vp.z) * 0.5

    @ti.kernel
    def pressure_to_swap(self, s: ti.i32):
        for i, j, k in ti.ndrange(*self.res):
            if self.grid_ng[s, i, j, k].is_free:
                self.p_swap.cur[i, j, k] = self.grid.p[s, i, j, k]

    @ti.kernel
    def pressure_from_swap(self, s: ti.i32):
        for i, j, k in ti.ndrange(*self.res):
            if self.grid_ng[s, i, j, k].is_free:
                self.grid.p[s + 1, i, j, k] = self.p_swap.cur[i, j, k]

    @ti.kernel
    def subtract_gradient(self, s: ti.i32):
        for i, j, k in ti.ndrange(*self.res):
            if self.grid_ng[s, i, j, k].is_free:

                pl = self.grid.p[s + 1, self.compute_location(s, i, j, k, -1, 0, 0)]
                pr = self.grid.p[s + 1, self.compute_location(s, i, j, k, 1, 0, 0)]
                pb = self.grid.p[s + 1, self.compute_location(s, i, j, k, 0, -1, 0)]
                pt = self.grid.p[s + 1, self.compute_location(s, i, j, k, 0, 1, 0)]
                pp = self.grid.p[s + 1, self.compute_location(s, i, j, k, 0, 0, -1)]
                pq = self.grid.p[s + 1, self.compute_location(s, i, j, k, 0, 0, 1)]

                self.grid.v[s + 1, i, j, k] = self.grid.v_tmp[s, i, j, k] - 0.5 * ti.Vector([pr - pl, pt - pb, pq - pp])
            else:
                self.grid.v[s + 1, i, j, k] = self.grid.v_tmp[s, i, j, k]

    @ti.kernel
    def colorize(self, s: ti.i32):
        for i, j, k in ti.ndrange(*self.res):
            ind = j * self.n_grid * self.n_grid + i * self.n_grid + k
            if self.lower_y_vis < j < self.higher_y_vis:
                color = (
                    self.cold_color * (1 - self.grid.q[s + 1, i, j, k][0])
                    + self.hot_color * self.grid.q[s + 1, i, j, k][0]
                )
                self.vis_particles_c[ind] = color
            else:
                self.vis_particles_c[ind] = ti.Vector([0.0, 0.0, 0.0, 0.0])

    @ti.func
    def compute_location(self, s, u, v, w, du, dv, dw):
        I = ti.Vector([int(u + du), int(v + dv), int(w + dw)])
        I = max(0, min(self.n_grid - 1, I))

        if not self.grid_ng[s, I].is_free:
            I = ti.Vector([int(u), int(v), int(w)])

        return I

    @ti.func
    def is_free(self, s, u, v, w, du, dv, dw):
        flag = 1

        I = ti.Vector([int(u + du), int(v + dv), int(w + dw)])
        if (I < 0).any() or (I > self.n_grid - 1).any():
            flag = 0

        elif not self.grid_ng[s, I].is_free:
            flag = 0

        return flag

    @ti.func
    def trilerp(self, f, qf, p):
        """
        p: position, within (0, 1).
        qf: field for interpolation
        """
        # convert position to grid index
        base_I = ti.floor(p - 0.5, gs.ti_int)
        p_I = p - 0.5

        q = ti.Vector.zero(gs.ti_float, qf.n)
        w_total = 0.0
        for offset in ti.static(ti.grouped(ti.ndrange(2, 2, 2))):
            grid_I = base_I + offset
            w_xyz = 1 - ti.abs(p_I - grid_I)
            w = w_xyz[0] * w_xyz[1] * w_xyz[2]
            grid_I_ = self.compute_location(f, grid_I[0], grid_I[1], grid_I[2], 0, 0, 0)
            q += w * qf[f, grid_I_]
            w_total += w
        # w_total is less then one when at boundary
        q /= w_total
        return q

    # RK3
    @ti.func
    def backtrace(self, f, vf, p, dt):
        """
        vf: velocity field
        """
        v1 = self.trilerp(f, vf, p)
        p1 = p - 0.5 * dt * v1
        v2 = self.trilerp(f, vf, p1)
        p2 = p - 0.75 * dt * v2
        v3 = self.trilerp(f, vf, p2)
        p -= dt * ((2 / 9) * v1 + (1 / 3) * v2 + (4 / 9) * v3)
        return p

    @ti.kernel
    def get_frame(
        self,
        s: ti.i32,
        v_np: ti.types.ndarray(),
        v_tmp_np: ti.types.ndarray(),
        div_np: ti.types.ndarray(),
        p_np: ti.types.ndarray(),
        q_np: ti.types.ndarray(),
    ):
        for i, j, k in ti.ndrange(*self.res):
            for n in ti.static(range(3)):
                v_np[i, j, k, n] = self.grid[s, i, j, k].v[n]
                v_tmp_np[i, j, k, n] = self.grid[s, i, j, k].v_tmp[n]
            for n in ti.static(range(self.q_dim)):
                q_np[i, j, k, n] = self.grid[s, i, j, k].q[n]
            div_np[i, j, k] = self.grid[s, i, j, k].div
            p_np[i, j, k] = self.grid[s, i, j, k].p

    @ti.kernel
    def set_frame(
        self,
        s: ti.i32,
        v_np: ti.types.ndarray(),
        v_tmp_np: ti.types.ndarray(),
        div_np: ti.types.ndarray(),
        p_np: ti.types.ndarray(),
        q_np: ti.types.ndarray(),
    ):
        for i, j, k in ti.ndrange(*self.res):
            for n in ti.static(range(3)):
                self.grid[s, i, j, k].v[n] = v_np[i, j, k, n]
                self.grid[s, i, j, k].v_tmp[n] = v_tmp_np[i, j, k, n]
            for n in ti.static(range(self.q_dim)):
                self.grid[s, i, j, k].q[n] = q_np[i, j, k, n]
            self.grid[s, i, j, k].div = div_np[i, j, k]
            self.grid[s, i, j, k].p = p_np[i, j, k]

    def save_ckpt(self, ckpt_name):
        return
        if not ckpt_name in self.ckpt_ram:
            device = "cpu"
            self.ckpt_ram[ckpt_name] = {
                "v": torch.zeros((*self.res, 3), dtype=float_tc, device=device),
                "v_tmp": torch.zeros((*self.res, 3), dtype=float_tc, device=device),
                "div": torch.zeros((*self.res,), dtype=float_tc, device=device),
                "p": torch.zeros((*self.res,), dtype=float_tc, device=device),
                "q": torch.zeros((*self.res, self.q_dim), dtype=float_tc, device=device),
            }
        self.get_frame(
            0,
            self.ckpt_ram[ckpt_name]["v"],
            self.ckpt_ram[ckpt_name]["v_tmp"],
            self.ckpt_ram[ckpt_name]["div"],
            self.ckpt_ram[ckpt_name]["p"],
            self.ckpt_ram[ckpt_name]["q"],
        )

        self.copy_frame(self.sim.max_steps_local, 0)

    def load_ckpt(self, ckpt_name):
        return
        self.copy_frame(0, self.sim.max_steps_local)
        self.copy_grad(0, self.sim.max_steps_local)
        self.reset_grad_till_frame(self.sim.max_steps_local)

        ckpt = self.ckpt_ram[ckpt_name]
        self.set_frame(0, ckpt["v"], ckpt["v_tmp"], ckpt["div"], ckpt["p"], ckpt["q"])

    def collect_output_grads(self):
        return

    def get_state(self, s):
        return None
        state = {
            "v": np.zeros((*self.res, 3), dtype=float_np),
            "v_tmp": np.zeros((*self.res, 3), dtype=float_np),
            "div": np.zeros((*self.res,), dtype=float_np),
            "p": np.zeros((*self.res,), dtype=float_np),
            "q": np.zeros((*self.res, self.q_dim), dtype=float_np),
        }
        self.get_frame(s, state["v"], state["v_tmp"], state["div"], state["p"], state["q"])
        return state

    def set_state(self, s, state):
        return
        self.set_frame(s, state["v"], state["v_tmp"], state["div"], state["p"], state["q"])


class TexPair:
    def __init__(self, cur, nxt):
        self.cur = cur
        self.nxt = nxt

    def swap(self):
        self.cur, self.nxt = self.nxt, self.cur
