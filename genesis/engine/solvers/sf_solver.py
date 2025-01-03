import math
import numpy as np
import taichi as ti

import genesis as gs
import genesis.utils.geom as gu
from genesis.engine.entities import SFParticleEntity
from genesis.engine.boundaries import CubeBoundary

from .base_solver import Solver


@ti.data_oriented
class SFSolver(Solver):
    """
    Stable Fluid solver for eulerian-based gaseous simulation.
    """

    # ------------------------------------------------------------------------------------
    # --------------------------------- Initialization -----------------------------------
    # ------------------------------------------------------------------------------------

    def __init__(self, scene, sim, options):
        super().__init__(scene, sim, options)

        if options is None:
            return

        self.n_grid = options.res
        self.dx = 1 / self.n_grid
        self.res = (self.n_grid, self.n_grid, self.n_grid)
        self.solver_iters = options.solver_iters
        self.decay = options.decay

        self.t = 0.0
        self.inlet_s = options.inlet_s

        self.jets = []

    def set_jets(self, jets):
        self.jets = jets

    def build(self):
        if self.is_active():
            self.t = 0.0
            self.setup_fields()
            self.init_fields()

    def is_active(self):
        return len(self.jets) > 0

    def setup_fields(self):
        cell_state = ti.types.struct(
            v=gs.ti_vec3,
            v_tmp=gs.ti_vec3,
            div=gs.ti_float,
            p=gs.ti_float,
            q=ti.types.vector(len(self.jets), gs.ti_float),
        )

        self.grid = cell_state.field(shape=self.res, layout=ti.Layout.SOA)

        # swap area for pressure projection solver
        self.p_swap = TexPair(
            cur=ti.field(dtype=gs.ti_float, shape=self.res),
            nxt=ti.field(dtype=gs.ti_float, shape=self.res),
        )

    @ti.kernel
    def init_fields(self):
        for i, j, k in ti.ndrange(*self.res):
            for q in ti.static(range(self.grid.q.n)):
                self.grid.q[i, j, k][q] = 0.0

    def reset_swap(self):
        self.p_swap.cur.fill(0)
        self.p_swap.nxt.fill(0)

    @ti.kernel
    def pressure_jacobi(self, pf: ti.template(), new_pf: ti.template()):
        for i, j, k in ti.ndrange(*self.res):
            pl = pf[self.compute_location(i, j, k, -1, 0, 0)]
            pr = pf[self.compute_location(i, j, k, 1, 0, 0)]
            pb = pf[self.compute_location(i, j, k, 0, -1, 0)]
            pt = pf[self.compute_location(i, j, k, 0, 1, 0)]
            pp = pf[self.compute_location(i, j, k, 0, 0, -1)]
            pq = pf[self.compute_location(i, j, k, 0, 0, 1)]

            new_pf[i, j, k] = (pl + pr + pb + pt + pp + pq - self.grid[i, j, k].div) / 6.0

    @ti.kernel
    def advect_and_impulse(self, f: ti.i32, t: ti.f32):
        for i, j, k in ti.ndrange(*self.res):
            p = ti.Vector([i, j, k], dt=gs.ti_float) + 0.5
            p = self.backtrace(self.grid.v, p, self.dt)
            v_tmp = self.trilerp(self.grid.v, p)

            for q in ti.static(range(self.grid.q.n)):
                q_f = self.trilerp_scalar(self.grid.q, p, q)

                imp_dir = self.jets[q].get_tan_dir(t)
                factor = self.jets[q].get_factor(i, j, k, self.dx, t)
                momentum = (imp_dir * self.inlet_s * factor) * self.dt

                v_tmp += momentum

                self.grid.q[i, j, k][q] = (1 - factor) * q_f + factor
                # self.grid.q[i, j, k][q] *= self.decay
                self.grid.q[i, j, k][q] -= self.decay * self.dt
                self.grid.q[i, j, k][q] = max(0.0, self.grid.q[i, j, k][q])

            self.grid.v_tmp[i, j, k] = v_tmp

    @ti.kernel
    def divergence(self):
        for i, j, k in ti.ndrange(*self.res):
            vl = self.grid.v_tmp[self.compute_location(i, j, k, -1, 0, 0)]
            vr = self.grid.v_tmp[self.compute_location(i, j, k, 1, 0, 0)]
            vb = self.grid.v_tmp[self.compute_location(i, j, k, 0, -1, 0)]
            vt = self.grid.v_tmp[self.compute_location(i, j, k, 0, 1, 0)]
            vp = self.grid.v_tmp[self.compute_location(i, j, k, 0, 0, -1)]
            vq = self.grid.v_tmp[self.compute_location(i, j, k, 0, 0, 1)]
            vc = self.grid.v_tmp[self.compute_location(i, j, k, 0, 0, 0)]

            if not self.is_free(i, j, k, -1, 0, 0):
                vl.x = -vc.x
            if not self.is_free(i, j, k, 1, 0, 0):
                vr.x = -vc.x
            if not self.is_free(i, j, k, 0, -1, 0):
                vb.y = -vc.y
            if not self.is_free(i, j, k, 0, 1, 0):
                vt.y = -vc.y
            if not self.is_free(i, j, k, 0, 0, -1):
                vp.z = -vc.z
            if not self.is_free(i, j, k, 0, 0, 1):
                vq.z = -vc.z

            self.grid.div[i, j, k] = (vr.x - vl.x + vt.y - vb.y + vq.z - vp.z) * 0.5

    @ti.kernel
    def pressure_to_swap(self):
        for i, j, k in ti.ndrange(*self.res):
            self.p_swap.cur[i, j, k] = self.grid.p[i, j, k]

    @ti.kernel
    def pressure_from_swap(self):
        for i, j, k in ti.ndrange(*self.res):
            self.grid.p[i, j, k] = self.p_swap.cur[i, j, k]

    @ti.kernel
    def subtract_gradient(self):
        for i, j, k in ti.ndrange(*self.res):
            pl = self.grid.p[self.compute_location(i, j, k, -1, 0, 0)]
            pr = self.grid.p[self.compute_location(i, j, k, 1, 0, 0)]
            pb = self.grid.p[self.compute_location(i, j, k, 0, -1, 0)]
            pt = self.grid.p[self.compute_location(i, j, k, 0, 1, 0)]
            pp = self.grid.p[self.compute_location(i, j, k, 0, 0, -1)]
            pq = self.grid.p[self.compute_location(i, j, k, 0, 0, 1)]

            self.grid.v[i, j, k] = self.grid.v_tmp[i, j, k] - 0.5 * ti.Vector([pr - pl, pt - pb, pq - pp])

    @ti.func
    def compute_location(self, u, v, w, du, dv, dw):
        I = ti.Vector([int(u + du), int(v + dv), int(w + dw)])
        I = max(0, min(self.n_grid - 1, I))
        return I

    @ti.func
    def is_free(self, u, v, w, du, dv, dw):
        flag = 1

        I = ti.Vector([int(u + du), int(v + dv), int(w + dw)])
        if (I < 0).any() or (I > self.n_grid - 1).any():
            flag = 0

        return flag

    @ti.func
    def trilerp_scalar(self, qf, p, qf_idx):
        """
        p: position, within (0, 1).
        qf: field for interpolation
        """
        # convert position to grid index
        base_I = ti.floor(p - 0.5, gs.ti_int)
        p_I = p - 0.5

        q = 0.0
        w_total = 0.0
        for offset in ti.static(ti.grouped(ti.ndrange(2, 2, 2))):
            grid_I = base_I + offset
            w_xyz = 1 - ti.abs(p_I - grid_I)
            w = w_xyz[0] * w_xyz[1] * w_xyz[2]
            grid_I_ = self.compute_location(grid_I[0], grid_I[1], grid_I[2], 0, 0, 0)
            q += w * qf[grid_I_][qf_idx]
            w_total += w
        # w_total is less then one when at boundary
        q /= w_total
        return q

    @ti.func
    def trilerp(self, qf, p):
        """
        p: position, within (0, 1).
        qf: field for interpolation
        """
        # convert position to grid index
        base_I = ti.floor(p - 0.5, gs.ti_int)
        p_I = p - 0.5

        q = ti.Vector([0.0, 0.0, 0.0], dt=gs.ti_float)
        w_total = 0.0
        for offset in ti.static(ti.grouped(ti.ndrange(2, 2, 2))):
            grid_I = base_I + offset
            w_xyz = 1 - ti.abs(p_I - grid_I)
            w = w_xyz[0] * w_xyz[1] * w_xyz[2]
            grid_I_ = self.compute_location(grid_I[0], grid_I[1], grid_I[2], 0, 0, 0)
            q += w * qf[grid_I_]
            w_total += w
        # w_total is less then one when at boundary
        q /= w_total
        return q

    # RK3
    @ti.func
    def backtrace(self, vf, p, dt):
        """
        vf: velocity field
        """
        v1 = self.trilerp(vf, p)
        p1 = p - 0.5 * dt * v1
        v2 = self.trilerp(vf, p1)
        p2 = p - 0.75 * dt * v2
        v3 = self.trilerp(vf, p2)
        p -= dt * ((2 / 9) * v1 + (1 / 3) * v2 + (4 / 9) * v3)
        return p

    # ------------------------------------------------------------------------------------
    # ------------------------------------ stepping --------------------------------------
    # ------------------------------------------------------------------------------------

    def process_input(self, in_backward):
        return None

    def substep_pre_coupling(self, f):
        self.advect_and_impulse(f, self.t)
        self.divergence()

        # projection
        self.reset_swap()
        self.pressure_to_swap()
        for _ in range(self.solver_iters):
            self.pressure_jacobi(self.p_swap.cur, self.p_swap.nxt)
            self.p_swap.swap()
        self.pressure_from_swap()
        self.reset_swap()

        self.subtract_gradient()
        self.t += self.dt

    def substep_post_coupling(self, f):
        return

    def reset_grad(self):
        return None

    # ------------------------------------------------------------------------------------
    # --------------------------------------- io -----------------------------------------
    # ------------------------------------------------------------------------------------

    def get_state(self, f):
        return None

    def set_state(self, f, state):
        return None

    # ------------------------------------------------------------------------------------
    # ------------------------------------ gradient --------------------------------------
    # ------------------------------------------------------------------------------------
    def collect_output_grads(self):
        """
        Collect gradients from downstream queried states.
        """
        pass

    def add_grad_from_state(self, state):
        pass

    def save_ckpt(self, ckpt_name):
        pass

    def load_ckpt(self, ckpt_name):
        pass


class TexPair:
    def __init__(self, cur, nxt):
        self.cur = cur
        self.nxt = nxt

    def swap(self):
        self.cur, self.nxt = self.nxt, self.cur
