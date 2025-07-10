from typing import TYPE_CHECKING

import numpy as np
import taichi as ti

import genesis as gs
from genesis.options.solvers import CouplerOptions, SAPCouplerOptions
from genesis.repr_base import RBC

if TYPE_CHECKING:
    from genesis.engine.simulator import Simulator


@ti.data_oriented
class Coupler(RBC):
    """
    This class handles all the coupling between different solvers.
    """

    # ------------------------------------------------------------------------------------
    # --------------------------------- Initialization -----------------------------------
    # ------------------------------------------------------------------------------------

    def __init__(
        self,
        simulator: "Simulator",
        options: "CouplerOptions",
    ) -> None:
        self.sim = simulator
        self.options = options

        self.tool_solver = self.sim.tool_solver
        self.rigid_solver = self.sim.rigid_solver
        self.avatar_solver = self.sim.avatar_solver
        self.mpm_solver = self.sim.mpm_solver
        self.sph_solver = self.sim.sph_solver
        self.pbd_solver = self.sim.pbd_solver
        self.fem_solver = self.sim.fem_solver
        self.sf_solver = self.sim.sf_solver

    def build(self) -> None:
        self._rigid_mpm = self.rigid_solver.is_active() and self.mpm_solver.is_active() and self.options.rigid_mpm
        self._rigid_sph = self.rigid_solver.is_active() and self.sph_solver.is_active() and self.options.rigid_sph
        self._rigid_pbd = self.rigid_solver.is_active() and self.pbd_solver.is_active() and self.options.rigid_pbd
        self._rigid_fem = self.rigid_solver.is_active() and self.fem_solver.is_active() and self.options.rigid_fem
        self._mpm_sph = self.mpm_solver.is_active() and self.sph_solver.is_active() and self.options.mpm_sph
        self._mpm_pbd = self.mpm_solver.is_active() and self.pbd_solver.is_active() and self.options.mpm_pbd
        self._fem_mpm = self.fem_solver.is_active() and self.mpm_solver.is_active() and self.options.fem_mpm
        self._fem_sph = self.fem_solver.is_active() and self.sph_solver.is_active() and self.options.fem_sph

        if self._rigid_mpm and self.mpm_solver.enable_CPIC:
            # this field stores the geom index of the thin shell rigid object (if any) that separates particle and its surrounding grid cell
            self.cpic_flag = ti.field(gs.ti_int, shape=(self.mpm_solver.n_particles, 3, 3, 3, self.mpm_solver._B))
            self.mpm_rigid_normal = ti.Vector.field(
                3,
                dtype=gs.ti_float,
                shape=(self.mpm_solver.n_particles, self.rigid_solver.n_geoms_, self.mpm_solver._B),
            )

        if self._rigid_sph:
            self.sph_rigid_normal = ti.Vector.field(
                3,
                dtype=gs.ti_float,
                shape=(self.sph_solver.n_particles, self.rigid_solver.n_geoms_, self.sph_solver._B),
            )
            self.sph_rigid_normal_reordered = ti.Vector.field(
                3,
                dtype=gs.ti_float,
                shape=(self.sph_solver.n_particles, self.rigid_solver.n_geoms_, self.sph_solver._B),
            )

        if self._rigid_pbd:
            self.pbd_rigid_normal_reordered = ti.Vector.field(
                3, dtype=gs.ti_float, shape=(self.pbd_solver.n_particles, self.pbd_solver._B, self.rigid_solver.n_geoms)
            )

        if self._mpm_sph:
            self.mpm_sph_stencil_size = int(np.floor(self.mpm_solver.dx / self.sph_solver.hash_grid_cell_size) + 2)

        if self._mpm_pbd:
            self.mpm_pbd_stencil_size = int(np.floor(self.mpm_solver.dx / self.pbd_solver.hash_grid_cell_size) + 2)

        ## DEBUG
        self._dx = 1 / 1024
        self._stencil_size = int(np.floor(self._dx / self.sph_solver.hash_grid_cell_size) + 2)

        self.reset(envs_idx=self.sim.scene._envs_idx)

    def reset(self, envs_idx=None) -> None:
        if self._rigid_mpm and self.mpm_solver.enable_CPIC:
            if envs_idx is None:
                self.mpm_rigid_normal.fill(0)
            else:
                self._kernel_reset_mpm(envs_idx)

        if self._rigid_sph:
            if envs_idx is None:
                self.sph_rigid_normal.fill(0)
            else:
                self._kernel_reset_sph(envs_idx)

    @ti.kernel
    def _kernel_reset_mpm(self, envs_idx: ti.types.ndarray()):
        for i_p, i_g, i_b_ in ti.ndrange(self.mpm_solver.n_particles, self.rigid_solver.n_geoms, envs_idx.shape[0]):
            self.mpm_rigid_normal[i_p, i_g, envs_idx[i_b_]] = 0.0

    @ti.kernel
    def _kernel_reset_sph(self, envs_idx: ti.types.ndarray()):
        for i_p, i_g, i_b_ in ti.ndrange(self.sph_solver.n_particles, self.rigid_solver.n_geoms, envs_idx.shape[0]):
            self.sph_rigid_normal[i_p, i_g, envs_idx[i_b_]] = 0.0

    @ti.func
    def _func_collide_with_rigid(self, f, pos_world, vel, mass, i_b):
        for i_g in range(self.rigid_solver.n_geoms):
            if self.rigid_solver.geoms_info[i_g].needs_coup:
                vel = self._func_collide_with_rigid_geom(pos_world, vel, mass, i_g, i_b)
        return vel

    @ti.func
    def _func_collide_with_rigid_geom(self, pos_world, vel, mass, geom_idx, batch_idx):
        g_info = self.rigid_solver.geoms_info[geom_idx]
        signed_dist = self.rigid_solver.sdf.sdf_world(pos_world, geom_idx, batch_idx)

        # bigger coup_softness implies that the coupling influence extends further away from the object.
        influence = ti.min(ti.exp(-signed_dist / max(1e-10, g_info.coup_softness)), 1)

        if influence > 0.1:
            normal_rigid = self.rigid_solver.sdf.sdf_normal_world(pos_world, geom_idx, batch_idx)
            vel = self._func_collide_in_rigid_geom(pos_world, vel, mass, normal_rigid, influence, geom_idx, batch_idx)

        return vel

    @ti.func
    def _func_collide_with_rigid_geom_robust(self, pos_world, vel, mass, normal_prev, geom_idx, batch_idx):
        """
        Similar to _func_collide_with_rigid_geom, but additionally handles potential side flip due to penetration.
        """
        g_info = self.rigid_solver.geoms_info[geom_idx]
        signed_dist = self.rigid_solver.sdf.sdf_world(pos_world, geom_idx, batch_idx)
        normal_rigid = self.rigid_solver.sdf.sdf_normal_world(pos_world, geom_idx, batch_idx)

        # bigger coup_softness implies that the coupling influence extends further away from the object.
        influence = ti.min(ti.exp(-signed_dist / max(1e-10, g_info.coup_softness)), 1)

        # if normal_rigid.dot(normal_prev) < 0: # side flip due to penetration
        #     influence = 1.0
        #     normal_rigid = normal_prev
        if influence > 0.1:
            vel = self._func_collide_in_rigid_geom(pos_world, vel, mass, normal_rigid, influence, geom_idx, batch_idx)

        # attraction force
        # if 0.001 < signed_dist < 0.01:
        #     vel = vel - normal_rigid * 0.1 * signed_dist

        return vel, normal_rigid

    @ti.func
    def _func_collide_in_rigid_geom(self, pos_world, vel, mass, normal_rigid, influence, geom_idx, batch_idx):
        """
        Resolves collision when a particle is already in collision with a rigid object.
        This function assumes known normal_rigid and influence.
        """
        g_info = self.rigid_solver.geoms_info[geom_idx]
        vel_rigid = self.rigid_solver._func_vel_at_point(pos_world, g_info.link_idx, batch_idx)

        # v w.r.t rigid
        rvel = vel - vel_rigid
        rvel_normal_magnitude = rvel.dot(normal_rigid)  # negative if inward

        if rvel_normal_magnitude < 0:  # colliding
            #################### rigid -> particle ####################
            # tangential component
            rvel_tan = rvel - rvel_normal_magnitude * normal_rigid
            rvel_tan_norm = rvel_tan.norm(gs.EPS)

            # tangential component after friction
            rvel_tan = (
                rvel_tan / rvel_tan_norm * ti.max(0, rvel_tan_norm + rvel_normal_magnitude * g_info.coup_friction)
            )

            # normal component after collision
            rvel_normal = -normal_rigid * rvel_normal_magnitude * g_info.coup_restitution

            # normal + tangential component
            rvel_new = rvel_tan + rvel_normal

            # apply influence
            vel_old = vel
            vel = vel_rigid + rvel_new * influence + rvel * (1 - influence)

            #################### particle -> rigid ####################
            # Compute delta momentum and apply to rigid body.
            delta_mv = mass * (vel - vel_old)
            force = -delta_mv / self.rigid_solver.substep_dt
            self.rigid_solver._func_apply_external_force(pos_world, force, g_info.link_idx, batch_idx)

        return vel

    @ti.func
    def _func_mpm_tool(self, f, pos_world, vel, i_b):
        for entity in ti.static(self.tool_solver.entities):
            if ti.static(entity.material.collision):
                vel = entity.collide(f, pos_world, vel, i_b)
        return vel

    @ti.kernel
    def mpm_grid_op(self, f: ti.i32, t: ti.f32):
        """
        This combines mpm's grid_op with coupling operations.
        If we decouple grid_op with coupling with different solvers, we need to run grid-level operations for each coupling pair, which is inefficient.
        """
        for ii, jj, kk, i_b in ti.ndrange(*self.mpm_solver.grid_res, self.mpm_solver._B):
            I = (ii, jj, kk)
            if self.mpm_solver.grid[f, I, i_b].mass > gs.EPS:
                #################### MPM grid op ####################
                # Momentum to velocity
                vel_mpm = (1 / self.mpm_solver.grid[f, I, i_b].mass) * self.mpm_solver.grid[f, I, i_b].vel_in

                # gravity
                vel_mpm += self.mpm_solver.substep_dt * self.mpm_solver._gravity[i_b]

                pos = (I + self.mpm_solver.grid_offset) * self.mpm_solver.dx
                mass_mpm = self.mpm_solver.grid[f, I, i_b].mass / self.mpm_solver._p_vol_scale

                # external force fields
                for i_ff in ti.static(range(len(self.mpm_solver._ffs))):
                    vel_mpm += self.mpm_solver._ffs[i_ff].get_acc(pos, vel_mpm, t, -1) * self.mpm_solver.substep_dt

                #################### MPM <-> Tool ####################
                if ti.static(self.tool_solver.is_active()):
                    vel_mpm = self._func_mpm_tool(f, pos, vel_mpm, i_b)

                #################### MPM <-> Rigid ####################
                if ti.static(self._rigid_mpm):
                    vel_mpm = self._func_collide_with_rigid(f, pos, vel_mpm, mass_mpm, i_b)

                #################### MPM <-> SPH ####################
                if ti.static(self._mpm_sph):
                    # using the lower corner of MPM cell to find the corresponding SPH base cell
                    base = self.sph_solver.sh.pos_to_grid(pos - 0.5 * self.mpm_solver.dx)

                    # ---------- SPH -> MPM ----------
                    sph_vel = ti.Vector([0.0, 0.0, 0.0])
                    colliding_particles = 0
                    for offset in ti.grouped(
                        ti.ndrange(self.mpm_sph_stencil_size, self.mpm_sph_stencil_size, self.mpm_sph_stencil_size)
                    ):
                        slot_idx = self.sph_solver.sh.grid_to_slot(base + offset)
                        for i in range(
                            self.sph_solver.sh.slot_start[slot_idx, i_b],
                            self.sph_solver.sh.slot_start[slot_idx, i_b] + self.sph_solver.sh.slot_size[slot_idx, i_b],
                        ):
                            if (
                                ti.abs(pos - self.sph_solver.particles_reordered.pos[i, i_b]).max()
                                < self.mpm_solver.dx * 0.5
                            ):
                                sph_vel += self.sph_solver.particles_reordered.vel[i, i_b]
                                colliding_particles += 1
                    if colliding_particles > 0:
                        vel_old = vel_mpm
                        vel_mpm = sph_vel / colliding_particles

                        # ---------- MPM -> SPH ----------
                        delta_mv = mass_mpm * (vel_mpm - vel_old)

                        for offset in ti.grouped(
                            ti.ndrange(self.mpm_sph_stencil_size, self.mpm_sph_stencil_size, self.mpm_sph_stencil_size)
                        ):
                            slot_idx = self.sph_solver.sh.grid_to_slot(base + offset)
                            for i in range(
                                self.sph_solver.sh.slot_start[slot_idx, i_b],
                                self.sph_solver.sh.slot_start[slot_idx, i_b]
                                + self.sph_solver.sh.slot_size[slot_idx, i_b],
                            ):
                                if (
                                    ti.abs(pos - self.sph_solver.particles_reordered.pos[i, i_b]).max()
                                    < self.mpm_solver.dx * 0.5
                                ):
                                    self.sph_solver.particles_reordered[i, i_b].vel = (
                                        self.sph_solver.particles_reordered[i, i_b].vel
                                        - delta_mv / self.sph_solver.particles_info_reordered[i, i_b].mass
                                    )

                #################### MPM <-> PBD ####################
                if ti.static(self._mpm_pbd):
                    # using the lower corner of MPM cell to find the corresponding PBD base cell
                    base = self.pbd_solver.sh.pos_to_grid(pos - 0.5 * self.mpm_solver.dx)

                    # ---------- PBD -> MPM ----------
                    pbd_vel = ti.Vector([0.0, 0.0, 0.0])
                    colliding_particles = 0
                    for offset in ti.grouped(
                        ti.ndrange(self.mpm_pbd_stencil_size, self.mpm_pbd_stencil_size, self.mpm_pbd_stencil_size)
                    ):
                        slot_idx = self.pbd_solver.sh.grid_to_slot(base + offset)
                        for i in range(
                            self.pbd_solver.sh.slot_start[slot_idx, i_b],
                            self.pbd_solver.sh.slot_start[slot_idx, i_b] + self.pbd_solver.sh.slot_size[slot_idx, i_b],
                        ):
                            if (
                                ti.abs(pos - self.pbd_solver.particles_reordered.pos[i, i_b]).max()
                                < self.mpm_solver.dx * 0.5
                            ):
                                pbd_vel += self.pbd_solver.particles_reordered.vel[i, i_b]
                                colliding_particles += 1
                    if colliding_particles > 0:
                        vel_old = vel_mpm
                        vel_mpm = pbd_vel / colliding_particles

                        # ---------- MPM -> PBD ----------
                        delta_mv = mass_mpm * (vel_mpm - vel_old)

                        for offset in ti.grouped(
                            ti.ndrange(self.mpm_pbd_stencil_size, self.mpm_pbd_stencil_size, self.mpm_pbd_stencil_size)
                        ):
                            slot_idx = self.pbd_solver.sh.grid_to_slot(base + offset)
                            for i in range(
                                self.pbd_solver.sh.slot_start[slot_idx, i_b],
                                self.pbd_solver.sh.slot_start[slot_idx, i_b]
                                + self.pbd_solver.sh.slot_size[slot_idx, i_b],
                            ):
                                if (
                                    ti.abs(pos - self.pbd_solver.particles_reordered.pos[i, i_b]).max()
                                    < self.mpm_solver.dx * 0.5
                                ):
                                    if self.pbd_solver.particles_reordered[i, i_b].free:
                                        self.pbd_solver.particles_reordered[i, i_b].vel = (
                                            self.pbd_solver.particles_reordered[i, i_b].vel
                                            - delta_mv / self.pbd_solver.particles_info_reordered[i, i_b].mass
                                        )

                #################### MPM boundary ####################
                _, self.mpm_solver.grid[f, I, i_b].vel_out = self.mpm_solver.boundary.impose_pos_vel(pos, vel_mpm)

    @ti.kernel
    def mpm_surface_to_particle(self, f: ti.i32):
        for i_p, i_b in ti.ndrange(self.mpm_solver.n_particles, self.mpm_solver._B):
            if self.mpm_solver.particles_ng[f, i_p, i_b].active:
                for i_g in range(self.rigid_solver.n_geoms):
                    if self.rigid_solver.geoms_info[i_g].needs_coup:
                        sdf_normal = self.rigid_solver.sdf.sdf_normal_world(
                            self.mpm_solver.particles[f, i_p, i_b].pos, i_g, i_b
                        )
                        # we only update the normal if the particle does not the object
                        if sdf_normal.dot(self.mpm_rigid_normal[i_p, i_g, i_b]) >= 0:
                            self.mpm_rigid_normal[i_p, i_g, i_b] = sdf_normal

    @ti.kernel
    def fem_surface_force(self, f: ti.i32):
        # TODO: all collisions are on vertices instead of surface and edge
        for i_s, i_b in ti.ndrange(self.fem_solver.n_surfaces, self.fem_solver._B):
            if self.fem_solver.surface[i_s].active:
                dt = self.fem_solver.substep_dt
                iel = self.fem_solver.surface[i_s].tri2el
                mass = self.fem_solver.elements_i[iel].mass_scaled / self.fem_solver.vol_scale

                p1 = self.fem_solver.elements_v[f, self.fem_solver.surface[i_s].tri2v[0], i_b].pos
                p2 = self.fem_solver.elements_v[f, self.fem_solver.surface[i_s].tri2v[1], i_b].pos
                p3 = self.fem_solver.elements_v[f, self.fem_solver.surface[i_s].tri2v[2], i_b].pos
                u = p2 - p1
                v = p3 - p1
                surface_normal = ti.math.cross(u, v)
                surface_normal = surface_normal / surface_normal.norm(gs.EPS)

                # FEM <-> Rigid
                if ti.static(self._rigid_fem):
                    # NOTE: collision only on surface vertices
                    for j in ti.static(range(3)):
                        iv = self.fem_solver.surface[i_s].tri2v[j]
                        vel_fem_sv = self._func_collide_with_rigid(
                            f,
                            self.fem_solver.elements_v[f, iv, i_b].pos,
                            self.fem_solver.elements_v[f + 1, iv, i_b].vel,
                            mass / 3.0,  # assume element mass uniformly distributed to vertices
                            i_b,
                        )
                        self.fem_solver.elements_v[f + 1, iv, i_b].vel = vel_fem_sv

                # FEM <-> MPM (interact with MPM grid instead of particles)
                # NOTE: not doing this in mpm_grid_op otherwise we need to search for fem surface for each particles
                #       however, this function is called after mpm boundary conditions.
                if ti.static(self._fem_mpm):
                    for j in ti.static(range(3)):
                        iv = self.fem_solver.surface[i_s].tri2v[j]
                        pos = self.fem_solver.elements_v[f, iv, i_b].pos
                        vel_fem_sv = self.fem_solver.elements_v[f + 1, iv, i_b].vel
                        mass_fem_sv = mass / 4.0  # assume element mass uniformly distributed

                        # follow MPM p2g scheme
                        vel_mpm = ti.Vector([0.0, 0.0, 0.0])
                        mass_mpm = 0.0
                        mpm_base = ti.floor(pos * self.mpm_solver.inv_dx - 0.5).cast(gs.ti_int)
                        mpm_fx = pos * self.mpm_solver.inv_dx - mpm_base.cast(gs.ti_float)
                        mpm_w = [0.5 * (1.5 - mpm_fx) ** 2, 0.75 - (mpm_fx - 1.0) ** 2, 0.5 * (mpm_fx - 0.5) ** 2]
                        new_vel_fem_sv = vel_fem_sv
                        for mpm_offset in ti.static(ti.grouped(self.mpm_solver.stencil_range())):
                            mpm_grid_I = mpm_base - self.mpm_solver.grid_offset + mpm_offset
                            mpm_grid_mass = self.mpm_solver.grid[f, mpm_grid_I, i_b].mass / self.mpm_solver.p_vol_scale

                            mpm_weight = ti.cast(1.0, gs.ti_float)
                            for d in ti.static(range(3)):
                                mpm_weight *= mpm_w[mpm_offset[d]][d]

                            # FEM -> MPM
                            mpm_grid_pos = (mpm_grid_I + self.mpm_solver.grid_offset) * self.mpm_solver.dx
                            signed_dist = (mpm_grid_pos - pos).dot(surface_normal)
                            if signed_dist <= self.mpm_solver.dx:  # NOTE: use dx as minimal unit for collision
                                vel_mpm_at_cell = mpm_weight * self.mpm_solver.grid[f, mpm_grid_I, i_b].vel_out
                                mass_mpm_at_cell = mpm_weight * mpm_grid_mass

                                vel_mpm += vel_mpm_at_cell
                                mass_mpm += mass_mpm_at_cell

                                if mass_mpm_at_cell > gs.EPS:
                                    delta_mpm_vel_at_cell_unmul = (
                                        vel_fem_sv * mpm_weight - self.mpm_solver.grid[f, mpm_grid_I, i_b].vel_out
                                    )
                                    mass_mul_at_cell = (
                                        mpm_grid_mass / mass_fem_sv
                                    )  # NOTE: use un-reweighted mass instead of mass_mpm_at_cell
                                    delta_mpm_vel_at_cell = delta_mpm_vel_at_cell_unmul * mass_mul_at_cell
                                    self.mpm_solver.grid[f, mpm_grid_I, i_b].vel_out += delta_mpm_vel_at_cell

                                    new_vel_fem_sv -= delta_mpm_vel_at_cell * mass_mpm_at_cell / mass_fem_sv

                        # MPM -> FEM
                        if mass_mpm > gs.EPS:
                            # delta_mv = (vel_mpm - vel_fem_sv) * mass_mpm
                            # delta_vel_fem_sv = delta_mv / mass_fem_sv
                            # self.fem_solver.elements_v[f + 1, iv].vel += delta_vel_fem_sv
                            self.fem_solver.elements_v[f + 1, iv, i_b].vel = new_vel_fem_sv

                # FEM <-> SPH TODO: this doesn't work well
                if ti.static(self._fem_sph):
                    for j in ti.static(range(3)):
                        iv = self.fem_solver.surface[i_s].tri2v[j]
                        pos = self.fem_solver.elements_v[f, iv, i_b].pos
                        vel_fem_sv = self.fem_solver.elements_v[f + 1, iv, i_b].vel
                        mass_fem_sv = mass / 4.0

                        dx = self.sph_solver.hash_grid_cell_size  # self._dx
                        stencil_size = 2  # self._stencil_size

                        base = self.sph_solver.sh.pos_to_grid(pos - 0.5 * dx)

                        # ---------- SPH -> FEM ----------
                        sph_vel = ti.Vector([0.0, 0.0, 0.0])
                        colliding_particles = 0
                        for offset in ti.grouped(ti.ndrange(stencil_size, stencil_size, stencil_size)):
                            slot_idx = self.sph_solver.sh.grid_to_slot(base + offset)
                            for k in range(
                                self.sph_solver.sh.slot_start[slot_idx, i_b],
                                self.sph_solver.sh.slot_start[slot_idx, i_b]
                                + self.sph_solver.sh.slot_size[slot_idx, i_b],
                            ):
                                if ti.abs(pos - self.sph_solver.particles_reordered.pos[k, i_b]).max() < dx * 0.5:
                                    sph_vel += self.sph_solver.particles_reordered.vel[k, i_b]
                                    colliding_particles += 1

                        if colliding_particles > 0:
                            vel_old = vel_fem_sv
                            vel_fem_sv_unprojected = sph_vel / colliding_particles
                            vel_fem_sv = (
                                vel_fem_sv_unprojected.dot(surface_normal) * surface_normal
                            )  # exclude tangential velocity

                            # ---------- FEM -> SPH ----------
                            delta_mv = mass_fem_sv * (vel_fem_sv - vel_old)

                            for offset in ti.grouped(ti.ndrange(stencil_size, stencil_size, stencil_size)):
                                slot_idx = self.sph_solver.sh.grid_to_slot(base + offset)
                                for k in range(
                                    self.sph_solver.sh.slot_start[slot_idx, i_b],
                                    self.sph_solver.sh.slot_start[slot_idx, i_b]
                                    + self.sph_solver.sh.slot_size[slot_idx, i_b],
                                ):
                                    if ti.abs(pos - self.sph_solver.particles_reordered.pos[k, i_b]).max() < dx * 0.5:
                                        self.sph_solver.particles_reordered[k, i_b].vel = (
                                            self.sph_solver.particles_reordered[k, i_b].vel
                                            - delta_mv / self.sph_solver.particles_info_reordered[k, i_b].mass
                                        )

                            self.fem_solver.elements_v[f + 1, iv, i_b].vel = vel_fem_sv

                # boundary condition
                for j in ti.static(range(3)):
                    iv = self.fem_solver.surface[i_s].tri2v[j]
                    _, self.fem_solver.elements_v[f + 1, iv, i_b].vel = self.fem_solver.boundary.impose_pos_vel(
                        self.fem_solver.elements_v[f, iv, i_b].pos, self.fem_solver.elements_v[f + 1, iv, i_b].vel
                    )

    def fem_hydroelastic(self, f: ti.i32):
        # Floor contact

        # collision detection
        self.fem_solver.floor_hydroelastic_detection(f)

    @ti.kernel
    def sph_rigid(self, f: ti.i32):
        for i_p, i_b in ti.ndrange(self.sph_solver._n_particles, self.sph_solver._B):
            if self.sph_solver.particles_ng_reordered[i_p, i_b].active:
                for i_g in range(self.rigid_solver.n_geoms):
                    if self.rigid_solver.geoms_info[i_g].needs_coup:
                        (
                            self.sph_solver.particles_reordered[i_p, i_b].vel,
                            self.sph_rigid_normal_reordered[i_p, i_g, i_b],
                        ) = self._func_collide_with_rigid_geom_robust(
                            self.sph_solver.particles_reordered[i_p, i_b].pos,
                            self.sph_solver.particles_reordered[i_p, i_b].vel,
                            self.sph_solver.particles_info_reordered[i_p, i_b].mass,
                            self.sph_rigid_normal_reordered[i_p, i_g, i_b],
                            i_g,
                            i_b,
                        )

    @ti.kernel
    def pbd_rigid(self, f: ti.i32):
        for i_p, i_b in ti.ndrange(self.pbd_solver._n_particles, self.sph_solver._B):
            if self.pbd_solver.particles_ng_reordered[i_p, i_b].active:
                # NOTE: Couldn't figure out a good way to handle collision with non-free particle. Such collision is not phsically plausible anyway.
                for i_g in range(self.rigid_solver.n_geoms):
                    if self.rigid_solver.geoms_info[i_g].needs_coup:
                        (
                            self.pbd_solver.particles_reordered[i_p, i_b].pos,
                            self.pbd_solver.particles_reordered[i_p, i_b].vel,
                            self.pbd_rigid_normal_reordered[i_p, i_b, i_g],
                        ) = self._func_pbd_collide_with_rigid_geom(
                            i_p,
                            self.pbd_solver.particles_reordered[i_p, i_b].pos,
                            self.pbd_solver.particles_reordered[i_p, i_b].vel,
                            self.pbd_solver.particles_info_reordered[i_p, i_b].mass,
                            self.pbd_rigid_normal_reordered[i_p, i_b, i_g],
                            i_g,
                            i_b,
                        )

    @ti.func
    def _func_pbd_collide_with_rigid_geom(self, i, pos_world, vel, mass, normal_prev, geom_idx, batch_idx):
        """
        Resolves collision when a particle is already in collision with a rigid object.
        This function assumes known normal_rigid and influence.
        """
        g_info = self.rigid_solver.geoms_info[geom_idx]
        signed_dist = self.rigid_solver.sdf.sdf_world(pos_world, geom_idx, batch_idx)
        vel_rigid = self.rigid_solver._func_vel_at_point(pos_world, g_info.link_idx, batch_idx)
        normal_rigid = self.rigid_solver.sdf.sdf_normal_world(pos_world, geom_idx, batch_idx)
        new_pos = pos_world
        if signed_dist < self.pbd_solver.particle_size / 2:  # skip non-penetration particles

            rvel = vel - vel_rigid
            rvel_normal_magnitude = rvel.dot(normal_rigid)  # negative if inward
            rvel_tan = rvel - rvel_normal_magnitude * normal_rigid
            rvel_tan_norm = rvel_tan.norm(gs.EPS)

            #################### rigid -> particle ####################
            stiffness = 1.0  # value in [0, 1]
            friction = 0.15
            energy_loss = 0.0  # value in [0, 1]
            new_pos = pos_world + stiffness * normal_rigid * (self.pbd_solver.particle_size / 2 - signed_dist)
            v_norm = (new_pos - self.pbd_solver.particles_reordered[i, batch_idx].ipos) / self.pbd_solver._substep_dt

            delta_normal_magnitude = (v_norm - vel).dot(normal_rigid)

            delta_v_norm = delta_normal_magnitude * normal_rigid
            vel = v_norm

            #################### particle -> rigid ####################
            delta_mv = mass * delta_v_norm
            force = (-delta_mv / self.rigid_solver._substep_dt) * (1 - energy_loss)

            self.rigid_solver._func_apply_external_force(pos_world, force, g_info.link_idx, batch_idx)

        return new_pos, vel, normal_rigid

    def preprocess(self, f):
        # preprocess for MPM CPIC
        if self.mpm_solver.is_active() and self.mpm_solver.enable_CPIC:
            self.mpm_surface_to_particle(f)

    def couple(self, f):
        # MPM <-> all others
        if self.mpm_solver.is_active():
            self.mpm_grid_op(f, self.sim.cur_t)

        # SPH <-> Rigid
        if self._rigid_sph:
            self.sph_rigid(f)

        # PBD <-> Rigid
        if self._rigid_pbd:
            self.pbd_rigid(f)

        if self.fem_solver.is_active():
            self.fem_surface_force(f)

    def couple_grad(self, f):
        if self.mpm_solver.is_active():
            self.mpm_grid_op.grad(f, self.sim.cur_t)

        if self.fem_solver.is_active():
            self.fem_surface_force.grad(f)

    @property
    def active_solvers(self):
        """All the active solvers managed by the scene's simulator."""
        return self.sim.active_solvers


@ti.func
def tet_barycentric(p, tet_vertices):
    """
    Compute the barycentric coordinates of point p with respect to the tetrahedron defined by tet_vertices.
    tet_vertices is a matrix of shape (3, 4) where each column is a vertex of the tetrahedron.
    """
    v0 = tet_vertices[:, 0]
    v1 = tet_vertices[:, 1]
    v2 = tet_vertices[:, 2]
    v3 = tet_vertices[:, 3]

    # Compute the vectors from the vertices to the point p
    v1_p = p - v1
    v2_p = p - v2
    v3_p = p - v3

    # Compute the volumes of the tetrahedra formed by the point and the vertices
    vol_tet = ti.math.dot(v1 - v0, ti.math.cross(v2 - v0, v3 - v0))

    # Compute the barycentric coordinates
    b0 = -ti.math.dot(v1_p, ti.math.cross(v2 - v1, v3 - v1)) / vol_tet
    b1 = ti.math.dot(v2_p, ti.math.cross(v3 - v2, v0 - v2)) / vol_tet
    b2 = -ti.math.dot(v3_p, ti.math.cross(v0 - v3, v1 - v3)) / vol_tet
    b3 = 1.0 - b0 - b1 - b2

    return ti.Vector([b0, b1, b2, b3])


@ti.data_oriented
class SAPCoupler(RBC):
    """
    This class handles all the coupling between different solvers using the
    Semi-Analytic Primal (SAP) contact solver used in Drake.

    Note
    ----
    Paper reference: https://arxiv.org/abs/2110.10107
    Drake reference: https://drake.mit.edu/release_notes/v1.5.0.html
    """

    # ------------------------------------------------------------------------------------
    # --------------------------------- Initialization -----------------------------------
    # ------------------------------------------------------------------------------------

    def __init__(
        self,
        simulator: "Simulator",
        options: "SAPCouplerOptions",
    ) -> None:
        self.sim = simulator
        self.options = options
        self.rigid_solver = self.sim.rigid_solver
        self.fem_solver = self.sim.fem_solver
        self._n_sap_iterations = options.n_sap_iterations
        self._sap_threshold = options.sap_threshold
        self._n_pcg_iterations = options.n_pcg_iterations
        self._pcg_threshold = options.pcg_threshold
        self._n_linesearch_iterations = options.n_linesearch_iterations
        self._linesearch_c = options.linesearch_c
        self._linesearch_tau = options.linesearch_tau
        self.default_deformable_g = 1.0e8  # default deformable geometry size

    def build(self) -> None:
        self._B = self.sim._B
        self._rigid_fem = self.rigid_solver.is_active() and self.fem_solver.is_active() and self.options.rigid_fem

        if self.fem_solver.is_active():
            if self.fem_solver._use_implicit_solver is False:
                gs.raise_exception(
                    "SAPCoupler requires FEM to use implicit solver. "
                    "Please set `use_implicit_solver=True` in FEM options."
                )
            self.init_fem_fields()

        self.init_sap_fields()
        self.init_pcg_fields()
        self.init_linesearch_fields()

    def reset(self, envs_idx=None) -> None:
        pass

    def init_fem_fields(self):
        fem_solver = self.fem_solver
        self.fem_pressure = ti.field(gs.ti_float, shape=(fem_solver.n_vertices))
        fem_pressure_np = np.concatenate([fem_entity.pressure_field_np for fem_entity in fem_solver.entities])
        self.fem_pressure.from_numpy(fem_pressure_np)
        self.fem_pressure_gradient = ti.field(gs.ti_vec3, shape=(fem_solver._B, fem_solver.n_elements))
        self.fem_floor_contact_pair_type = ti.types.struct(
            active=gs.ti_int,  # whether the contact pair is active
            batch_idx=gs.ti_int,  # batch index
            geom_idx=gs.ti_int,  # index of the FEM element
            intersection_code=gs.ti_int,  # intersection code for the element
            distance=gs.ti_vec4,  # distance vector for the element
            k=gs.ti_float,  # contact stiffness
            phi0=gs.ti_float,  # initial signed distance
            fn0=gs.ti_float,  # initial normal force magnitude
            taud=gs.ti_float,  # dissipation time scale
            barycentric=gs.ti_vec4,  # barycentric coordinates of the contact point
            Rn=gs.ti_float,  # Regularitaion for normal
            Rt=gs.ti_float,  # Regularitaion for tangential
            vn_hat=gs.ti_float,  # Stablization for normal velocity
            mu=gs.ti_float,  # friction coefficient
            mu_hat=gs.ti_float,  # friction coefficient regularized
            mu_factor=gs.ti_float,  # friction coefficient factor, 1/(1+mu_tilde**2)
            energy=gs.ti_float,  # energy
            G=gs.ti_mat3,  # Hessian matrix
        )
        self.max_fem_floor_contact_pairs = fem_solver.n_surfaces * fem_solver._B
        self.n_fem_floor_contact_pairs = ti.field(gs.ti_int, shape=())
        self.fem_floor_contact_pairs = self.fem_floor_contact_pair_type.field(shape=(self.max_fem_floor_contact_pairs,))

        # Lookup table for marching tetrahedra edges
        kMarchingTetsEdgeTable_np = np.array(
            [
                [-1, -1, -1, -1],
                [0, 3, 2, -1],
                [0, 1, 4, -1],
                [4, 3, 2, 1],
                [1, 2, 5, -1],
                [0, 3, 5, 1],
                [0, 2, 5, 4],
                [3, 5, 4, -1],
                [3, 4, 5, -1],
                [4, 5, 2, 0],
                [1, 5, 3, 0],
                [1, 5, 2, -1],
                [1, 2, 3, 4],
                [0, 4, 1, -1],
                [0, 2, 3, -1],
                [-1, -1, -1, -1],
            ]
        )
        self.kMarchingTetsEdgeTable = ti.field(gs.ti_ivec4, shape=kMarchingTetsEdgeTable_np.shape[0])
        self.kMarchingTetsEdgeTable.from_numpy(kMarchingTetsEdgeTable_np)

        kTetEdges_np = np.array([[0, 1], [1, 2], [2, 0], [0, 3], [1, 3], [2, 3]])
        self.kTetEdges = ti.field(gs.ti_ivec2, shape=kTetEdges_np.shape[0])
        self.kTetEdges.from_numpy(kTetEdges_np)

    def init_sap_fields(self):
        self.batch_active = ti.field(
            dtype=ti.u1,
            shape=self.sim._B,
            needs_grad=False,
        )
        self.v = ti.field(gs.ti_vec3, shape=(self.fem_solver._B, self.fem_solver.n_vertices))
        self.v_diff = ti.field(gs.ti_vec3, shape=(self.fem_solver._B, self.fem_solver.n_vertices))
        self.gradient = ti.field(gs.ti_vec3, shape=(self.fem_solver._B, self.fem_solver.n_vertices))

    def init_pcg_fields(self):
        self.batch_pcg_active = ti.field(
            dtype=ti.u1,
            shape=self.sim._B,
            needs_grad=False,
        )

        pcg_state = ti.types.struct(
            rTr=gs.ti_float,
            rTz=gs.ti_float,
            rTr_new=gs.ti_float,
            rTz_new=gs.ti_float,
            pTAp=gs.ti_float,
            alpha=gs.ti_float,
            beta=gs.ti_float,
        )

        self.pcg_state = pcg_state.field(
            shape=self.sim._B,
            needs_grad=False,
            layout=ti.Layout.SOA,
        )

        pcg_state_v = ti.types.struct(
            diag3x3=gs.ti_mat3,  # diagonal 3-by-3 block of the hessian
            prec=gs.ti_mat3,  # preconditioner
            x=gs.ti_vec3,  # solution vector
            r=gs.ti_vec3,  # residual vector
            z=gs.ti_vec3,  # preconditioned residual vector
            p=gs.ti_vec3,  # search direction vector
            Ap=gs.ti_vec3,  # matrix-vector product
        )

        self.pcg_state_v = pcg_state_v.field(
            shape=(self.sim._B, self.fem_solver.n_vertices),
            needs_grad=False,
            layout=ti.Layout.SOA,
        )

    def init_linesearch_fields(self):
        self.batch_linesearch_active = ti.field(
            dtype=ti.u1,
            shape=self.sim._B,
            needs_grad=False,
        )

        linesearch_state = ti.types.struct(
            prev_energy=gs.ti_float,
            energy=gs.ti_float,
            step_size=gs.ti_float,
            m=gs.ti_float,
        )

        self.linesearch_state = linesearch_state.field(
            shape=self.sim._B,
            needs_grad=False,
            layout=ti.Layout.SOA,
        )

        linesearch_state_v = ti.types.struct(
            x_prev=gs.ti_vec3,  # solution vector
        )

        self.linesearch_state_v = linesearch_state_v.field(
            shape=(self.sim._B, self.fem_solver.n_vertices),
            needs_grad=False,
            layout=ti.Layout.SOA,
        )

    def preprocess(self, f):
        pass

    def couple(self, f):
        self.has_contact = False
        if self.fem_solver.is_active():
            self.fem_compute_pressure_gradient(f)
            self.fem_floor_detection(f)
            self.has_fem_floor_contact = self.n_fem_floor_contact_pairs[None] > 0
            self.has_contact = self.has_contact or self.has_fem_floor_contact

        if self.has_contact:
            self.sap_solve(f)
            self.update_vel(f)

    def couple_grad(self, f):
        gs.raise_exception("couple_grad is not available for HydroelasticCoupler. Please use Coupler instead.")

    @ti.kernel
    def update_vel(self, f: ti.i32):
        fem_solver = self.fem_solver
        for i_b, i_v in ti.ndrange(fem_solver._B, fem_solver.n_vertices):
            self.fem_solver.elements_v[f + 1, i_v, i_b].vel = self.v[i_b, i_v]

    @ti.kernel
    def fem_compute_pressure_gradient(self, f: ti.i32):
        fem_solver = self.fem_solver
        for i_b, i_e in ti.ndrange(fem_solver._B, fem_solver.n_elements):
            grad = ti.static(self.fem_pressure_gradient)
            grad[i_b, i_e].fill(0.0)

            for i in ti.static(range(4)):
                i_v0 = fem_solver.elements_i[i_e].el2v[i]
                i_v1 = fem_solver.elements_i[i_e].el2v[(i + 1) % 4]
                i_v2 = fem_solver.elements_i[i_e].el2v[(i + 2) % 4]
                i_V3 = fem_solver.elements_i[i_e].el2v[(i + 3) % 4]
                pos_v0 = fem_solver.elements_v[f + 1, i_v0, i_b].pos
                pos_v1 = fem_solver.elements_v[f + 1, i_v1, i_b].pos
                pos_v2 = fem_solver.elements_v[f + 1, i_v2, i_b].pos
                pos_v3 = fem_solver.elements_v[f + 1, i_V3, i_b].pos

                e10 = pos_v0 - pos_v1
                e12 = pos_v2 - pos_v1
                e13 = pos_v3 - pos_v1

                area_vector = e12.cross(e13)  # area vector of the triangle formed by v1, v2, v3
                signed_volume = area_vector.dot(e10)  # signed volume of the tetrahedron formed by v0, v1, v2, v3
                if ti.abs(signed_volume) > gs.EPS:
                    grad_i = area_vector / signed_volume
                    grad[i_b, i_e] += grad_i * self.fem_pressure[i_v0]

    @ti.kernel
    def fem_floor_detection(self, f: ti.i32):
        fem_solver = self.fem_solver

        # Compute contact pairs
        self.n_fem_floor_contact_pairs[None] = 0
        # TODO Check surface element only instead of all elements
        for i_b, i_e in ti.ndrange(fem_solver._B, fem_solver.n_elements):
            intersection_code = ti.int32(0)
            distance = ti.Vector([0.0, 0.0, 0.0, 0.0])
            for i in ti.static(range(4)):
                i_v = fem_solver.elements_i[i_e].el2v[i]
                pos_v = fem_solver.elements_v[f + 1, i_v, i_b].pos
                distance[i] = pos_v.z - fem_solver.floor_height
                if distance[i] > 0:
                    intersection_code |= 1 << i

            # check if the element intersect with the floor
            if intersection_code != 0 and intersection_code != 15:
                pair_idx = ti.atomic_add(self.n_fem_floor_contact_pairs[None], 1)
                if pair_idx < self.max_fem_floor_contact_pairs:
                    self.fem_floor_contact_pairs[pair_idx].batch_idx = i_b
                    self.fem_floor_contact_pairs[pair_idx].geom_idx = i_e
                    self.fem_floor_contact_pairs[pair_idx].intersection_code = intersection_code
                    self.fem_floor_contact_pairs[pair_idx].distance = distance

        # Compute data for each contact pair
        for i_c in range(self.n_fem_floor_contact_pairs[None]):
            pair = self.fem_floor_contact_pairs[i_c]
            self.fem_floor_contact_pairs[i_c].active = 1  # mark the contact pair as active
            i_b = pair.batch_idx
            i_e = pair.geom_idx
            intersection_code = pair.intersection_code
            distance = pair.distance
            intersected_edges = ti.static(self.kMarchingTetsEdgeTable)[intersection_code]
            tet_vertices = ti.Matrix.zero(gs.ti_float, 3, 4)  # 4 vertices
            tet_pressures = ti.Vector.zero(gs.ti_float, 4)  # pressures at the vertices

            for i in ti.static(range(4)):
                i_v = fem_solver.elements_i[i_e].el2v[i]
                tet_vertices[:, i] = fem_solver.elements_v[f + 1, i_v, i_b].pos
                tet_pressures[i] = self.fem_pressure[i_v]

            polygon_vertices = ti.Matrix.zero(gs.ti_float, 3, 4)  # 3 or 4 vertices
            total_area = gs.EPS  # avoid division by zero
            total_area_weighted_centroid = ti.Vector([0.0, 0.0, 0.0])
            for i in range(4):
                if intersected_edges[i] >= 0:
                    edge = ti.static(self.kTetEdges)[intersected_edges[i]]
                    pos_v0 = tet_vertices[:, edge[0]]
                    pos_v1 = tet_vertices[:, edge[1]]
                    d_v0 = distance[edge[0]]
                    d_v1 = distance[edge[1]]
                    t = d_v0 / (d_v0 - d_v1)
                    polygon_vertices[:, i] = pos_v0 + t * (pos_v1 - pos_v0)

                    # Compute tirangle area and centroid
                    if i >= 2:
                        e1 = polygon_vertices[:, i - 1] - polygon_vertices[:, 0]
                        e2 = polygon_vertices[:, i] - polygon_vertices[:, 0]
                        area = 0.5 * e1.cross(e2).norm()
                        total_area += area
                        total_area_weighted_centroid += (
                            area * (polygon_vertices[:, 0] + polygon_vertices[:, i - 1] + polygon_vertices[:, i]) / 3.0
                        )

            centroid = total_area_weighted_centroid / total_area

            # Compute barycentric coordinates
            barycentric = tet_barycentric(centroid, tet_vertices)
            pressure = (
                barycentric[0] * tet_pressures[0]
                + barycentric[1] * tet_pressures[1]
                + barycentric[2] * tet_pressures[2]
                + barycentric[3] * tet_pressures[3]
            )
            self.fem_floor_contact_pairs[i_c].barycentric = barycentric

            rigid_g = self.fem_pressure_gradient[i_b, i_e].z
            # TODO A better way to handle corner cases where pressure and pressure gradient are ill defined
            if total_area < gs.EPS or rigid_g < gs.EPS:
                self.fem_floor_contact_pairs[i_c].active = 0
                continue
            g = self.default_deformable_g * rigid_g / (self.default_deformable_g + rigid_g)  # harmonic average
            rigid_k = total_area * g
            rigid_phi0 = -pressure / g
            rigid_fn0 = total_area * pressure
            # TODO custom dissipation
            self.fem_floor_contact_pairs[i_c].k = rigid_k  # contact stiffness
            self.fem_floor_contact_pairs[i_c].phi0 = rigid_phi0
            self.fem_floor_contact_pairs[i_c].fn0 = rigid_fn0
            self.fem_floor_contact_pairs[i_c].taud = self.sim._substep_dt * 1.0 / ti.math.pi

    def sap_solve(self, f):
        self.init_sap_solve(f)
        for i in range(self._n_sap_iterations):
            # init gradient and preconditioner
            self.compute_non_contact_gradient_diag(f, i)

            # compute contact hessian and gradient
            self.compute_contact_gradient_hessian_diag_prec(f)

            # solve for the vertex velocity
            self.pcg_solve()

            # line search
            self.linesearch(f)
            # TODO add convergence check

    def init_sap_solve(self, f: ti.i32):
        self.init_v(f)
        self.batch_active.fill(1)
        if self.has_fem_floor_contact:
            self.compute_fem_floor_regularization(f)

    @ti.kernel
    def init_v(self, f: ti.i32):
        fem_solver = self.fem_solver
        for i_b, i_v in ti.ndrange(fem_solver._B, fem_solver.n_vertices):
            self.v[i_b, i_v] = fem_solver.elements_v[f + 1, i_v, i_b].vel

    @ti.kernel
    def compute_fem_floor_regularization(self, f: ti.i32):
        pairs = ti.static(self.fem_floor_contact_pairs)
        time_step = self.sim._substep_dt
        dt2_inv = 1.0 / (time_step**2)
        fem_solver = self.fem_solver

        for i_c in range(self.n_fem_floor_contact_pairs[None]):
            if pairs[i_c].active == 0:
                continue
            i_b = pairs[i_c].batch_idx
            i_e = pairs[i_c].geom_idx
            W = ti.Matrix.zero(gs.ti_float, 3, 3)
            # W = sum (JA^-1J^T)
            # With floor, J is Identity times the barycentric coordinates
            for i in ti.static(range(4)):
                i_v = fem_solver.elements_i[i_e].el2v[i]
                W += pairs[i_c].barycentric[i] ** 2 * dt2_inv * fem_solver.pcg_state_v[i_b, i_v].prec
            w_rms = W.norm() / 3.0
            beta = ti.static(1.0)
            beta_factor = ti.static(beta**2 / (4.0 * ti.math.pi**2))
            k = pairs[i_c].k
            taud = pairs[i_c].taud
            Rn = max(beta_factor * w_rms, 1.0 / (time_step * k * (time_step + taud)))
            sigma = ti.static(1.0e-3)
            Rt = sigma * w_rms
            vn_hat = -pairs[i_c].phi0 / (time_step + taud)

            pairs[i_c].Rn = Rn
            pairs[i_c].Rt = Rt
            pairs[i_c].vn_hat = vn_hat
            pairs[i_c].mu = fem_solver.elements_i[i_e].friction_mu
            pairs[i_c].mu_hat = pairs[i_c].mu * Rt / Rn
            pairs[i_c].mu_factor = 1.0 / (1.0 + pairs[i_c].mu * pairs[i_c].mu_hat)

    @ti.kernel
    def compute_non_contact_gradient_diag(self, f: ti.i32, iter: int):
        fem_solver = self.fem_solver
        dt2 = fem_solver._substep_dt**2
        damping_alpha_dt = fem_solver._damping_alpha * fem_solver._substep_dt
        damping_alpha_factor = damping_alpha_dt + 1.0
        damping_beta_over_dt = fem_solver._damping_beta / fem_solver._substep_dt
        damping_beta_factor = damping_beta_over_dt + 1.0

        for i_b, i_v in ti.ndrange(fem_solver._B, fem_solver.n_vertices):
            self.gradient[i_b, i_v].fill(0.0)
            # was using position now using velocity, need to multiply dt^2
            self.pcg_state_v[i_b, i_v].diag3x3 = fem_solver.pcg_state_v[i_b, i_v].diag3x3 * dt2
            self.v_diff[i_b, i_v] = self.v[i_b, i_v] - fem_solver.elements_v[f + 1, i_v, i_b].vel

        # No need to do this for iter=0 because v=v* and A(v-v*) = 0
        if iter > 0:
            for i_b, i_v in ti.ndrange(fem_solver._B, fem_solver.n_vertices):
                self.gradient[i_b, i_v] = (
                    fem_solver.elements_v_info[i_v].mass_over_dt2 * self.v_diff[i_b, i_v] * dt2 * damping_alpha_factor
                )

            for i_b, i_e in ti.ndrange(fem_solver._B, fem_solver.n_elements):
                V_dt2 = fem_solver.elements_i[i_e].V * dt2
                B = fem_solver.elements_i[i_e].B
                s = -B[0, :] - B[1, :] - B[2, :]  # s is the negative sum of B rows
                p9 = ti.Vector([0.0] * 9, dt=gs.ti_float)
                i_v0, i_v1, i_v2, i_v3 = fem_solver.elements_i[i_e].el2v

                for i in ti.static(range(3)):
                    p9[i * 3 : i * 3 + 3] = (
                        B[0, i] * self.v_diff[i_b, i_v0]
                        + B[1, i] * self.v_diff[i_b, i_v1]
                        + B[2, i] * self.v_diff[i_b, i_v2]
                        + s[i] * self.v_diff[i_b, i_v3]
                    )

                new_p9 = ti.Vector([0.0] * 9, dt=gs.ti_float)

                for i in ti.static(range(3)):
                    new_p9[i * 3 : i * 3 + 3] = (
                        fem_solver.elements_el_hessian[i_b, i, 0, i_e] @ p9[0:3]
                        + fem_solver.elements_el_hessian[i_b, i, 1, i_e] @ p9[3:6]
                        + fem_solver.elements_el_hessian[i_b, i, 2, i_e] @ p9[6:9]
                    )

                # atomic
                self.gradient[i_b, i_v0] += (
                    (B[0, 0] * new_p9[0:3] + B[0, 1] * new_p9[3:6] + B[0, 2] * new_p9[6:9])
                    * V_dt2
                    * damping_beta_factor
                )
                self.gradient[i_b, i_v1] += (
                    (B[1, 0] * new_p9[0:3] + B[1, 1] * new_p9[3:6] + B[1, 2] * new_p9[6:9])
                    * V_dt2
                    * damping_beta_factor
                )
                self.gradient[i_b, i_v2] += (
                    (B[2, 0] * new_p9[0:3] + B[2, 1] * new_p9[3:6] + B[2, 2] * new_p9[6:9])
                    * V_dt2
                    * damping_beta_factor
                )
                self.gradient[i_b, i_v3] += (
                    (s[0] * new_p9[0:3] + s[1] * new_p9[3:6] + s[2] * new_p9[6:9]) * V_dt2 * damping_beta_factor
                )

    @ti.kernel
    def compute_contact_gradient_hessian_diag_prec(self, f: ti.i32):
        pairs = ti.static(self.fem_floor_contact_pairs)
        fem_solver = self.fem_solver

        for i_c in range(self.n_fem_floor_contact_pairs[None]):
            if pairs[i_c].active == 0:
                continue
            i_b = pairs[i_c].batch_idx
            i_e = pairs[i_c].geom_idx
            vc = ti.Vector([0.0, 0.0, 0.0])
            # With floor, the contact frame is the same as the world frame
            for i in ti.static(range(4)):
                i_v = fem_solver.elements_i[i_e].el2v[i]
                vc += pairs[i_c].barycentric[i] * self.v[i_b, i_v]
            y = ti.Vector([0.0, 0.0, pairs[i_c].vn_hat]) - vc
            y[0] /= pairs[i_c].Rt
            y[1] /= pairs[i_c].Rt
            y[2] /= pairs[i_c].Rn
            yr = y[:2].norm(gs.EPS)
            yn = y[2]

            t_hat = y[:2] / yr
            contact_mode = self.compute_contact_mode(pairs[i_c].mu, pairs[i_c].mu_hat, yr, yn)
            gamma = ti.Vector.zero(gs.ti_float, 3)
            pairs[i_c].G.fill(0.0)
            if contact_mode == 0:  # Sticking
                gamma = y
                pairs[i_c].G[0, 0] = 1.0 / pairs[i_c].Rt
                pairs[i_c].G[1, 1] = 1.0 / pairs[i_c].Rt
                pairs[i_c].G[2, 2] = 1.0 / pairs[i_c].Rn
            elif contact_mode == 1:  # Sliding
                gn = (yn + pairs[i_c].mu_hat * yr) * pairs[i_c].mu_factor
                gt = pairs[i_c].mu * gn * t_hat
                gamma = ti.Vector([gt[0], gt[1], gn])
                P = t_hat.outer_product(t_hat)
                Pperp = ti.Matrix.identity(gs.ti_float, 2) - P
                dgt_dyt = pairs[i_c].mu * (gn / yr * Pperp + pairs[i_c].mu_hat * pairs[i_c].mu_factor * P)
                dgt_dyn = pairs[i_c].mu * pairs[i_c].mu_factor * t_hat
                dgn_dyt = pairs[i_c].mu_hat * pairs[i_c].mu_factor * t_hat
                dgn_dyn = pairs[i_c].mu_factor

                pairs[i_c].G[:2, :2] = dgt_dyt
                pairs[i_c].G[:2, 2] = dgt_dyn
                pairs[i_c].G[2, :2] = dgn_dyt
                pairs[i_c].G[2, 2] = dgn_dyn

                pairs[i_c].G[:, :2] *= 1.0 / pairs[i_c].Rt
                pairs[i_c].G[:, 2] *= 1.0 / pairs[i_c].Rn

            else:  # No contact
                pass

            R_gamma = gamma
            R_gamma[0] *= pairs[i_c].Rt
            R_gamma[1] *= pairs[i_c].Rt
            R_gamma[2] *= pairs[i_c].Rn
            pairs[i_c].energy = 0.5 * gamma.dot(R_gamma)
            for i in ti.static(range(4)):
                i_v = fem_solver.elements_i[i_e].el2v[i]
                self.gradient[i_b, i_v] -= pairs[i_c].barycentric[i] * gamma
                self.pcg_state_v[i_b, i_v].diag3x3 += pairs[i_c].barycentric[i] ** 2 * pairs[i_c].G

        for i_b, i_v in ti.ndrange(fem_solver._B, fem_solver.n_vertices):
            if not self.batch_active[i_b]:
                continue
            self.pcg_state_v[i_b, i_v].prec = self.pcg_state_v[i_b, i_v].diag3x3.inverse()

    @ti.func
    def compute_contact_energy(self, f: ti.i32):
        pairs = ti.static(self.fem_floor_contact_pairs)
        fem_solver = self.fem_solver

        for i_c in range(self.n_fem_floor_contact_pairs[None]):
            if pairs[i_c].active == 0:
                continue
            i_b = pairs[i_c].batch_idx
            if not self.batch_linesearch_active[i_b]:
                continue
            i_e = pairs[i_c].geom_idx
            vc = ti.Vector([0.0, 0.0, 0.0])
            # With floor, the contact frame is the same as the world frame
            for i in ti.static(range(4)):
                i_v = fem_solver.elements_i[i_e].el2v[i]
                vc += pairs[i_c].barycentric[i] * self.v[i_b, i_v]
            y = ti.Vector([0.0, 0.0, pairs[i_c].vn_hat]) - vc
            y[0] /= pairs[i_c].Rt
            y[1] /= pairs[i_c].Rt
            y[2] /= pairs[i_c].Rn
            yr = y[:2].norm(gs.EPS)
            yn = y[2]

            t_hat = y[:2] / yr
            contact_mode = self.compute_contact_mode(pairs[i_c].mu, pairs[i_c].mu_hat, yr, yn)
            gamma = ti.Vector.zero(gs.ti_float, 3)
            if contact_mode == 0:  # Sticking
                gamma = y
            elif contact_mode == 1:  # Sliding
                gn = (yn + pairs[i_c].mu_hat * yr) * pairs[i_c].mu_factor
                gt = pairs[i_c].mu * gn * t_hat
                gamma = ti.Vector([gt[0], gt[1], gn])
            else:  # No contact
                pass

            R_gamma = gamma
            R_gamma[0] *= pairs[i_c].Rt
            R_gamma[1] *= pairs[i_c].Rt
            R_gamma[2] *= pairs[i_c].Rn
            pairs[i_c].energy = 0.5 * gamma.dot(R_gamma)

    @ti.func
    def compute_contact_mode(self, mu, mu_hat, yr, yn):
        """
        Compute the contact mode based on the friction coefficients and the relative velocities.
        Returns:
            0: Sticking
            1: Sliding
            2: No contact
        """
        result = 2  # No contact
        if yr <= mu * yn:
            result = 0  # Sticking
        elif -mu_hat * yr < yn and yn < yr / mu:
            result = 1  # Sliding
        return result

    @ti.func
    def compute_Ap(self):
        fem_solver = self.fem_solver
        dt2 = fem_solver._substep_dt**2
        damping_alpha_dt = fem_solver._damping_alpha * fem_solver._substep_dt
        damping_alpha_factor = damping_alpha_dt + 1.0
        damping_beta_over_dt = fem_solver._damping_beta / fem_solver._substep_dt
        damping_beta_factor = damping_beta_over_dt + 1.0

        for i_b, i_v in ti.ndrange(fem_solver._B, fem_solver.n_vertices):
            if not self.batch_pcg_active[i_b]:
                continue
            self.pcg_state_v[i_b, i_v].Ap = (
                fem_solver.elements_v_info[i_v].mass_over_dt2
                * self.pcg_state_v[i_b, i_v].p
                * dt2
                * damping_alpha_factor
            )

        for i_b, i_e in ti.ndrange(fem_solver._B, fem_solver.n_elements):
            if not self.batch_pcg_active[i_b]:
                continue
            V_dt2 = fem_solver.elements_i[i_e].V * dt2
            B = fem_solver.elements_i[i_e].B
            s = -B[0, :] - B[1, :] - B[2, :]  # s is the negative sum of B rows
            p9 = ti.Vector([0.0] * 9, dt=gs.ti_float)
            i_v0, i_v1, i_v2, i_v3 = fem_solver.elements_i[i_e].el2v

            for i in ti.static(range(3)):
                p9[i * 3 : i * 3 + 3] = (
                    B[0, i] * self.pcg_state_v[i_b, i_v0].p
                    + B[1, i] * self.pcg_state_v[i_b, i_v1].p
                    + B[2, i] * self.pcg_state_v[i_b, i_v2].p
                    + s[i] * self.pcg_state_v[i_b, i_v3].p
                )

            new_p9 = ti.Vector([0.0] * 9, dt=gs.ti_float)

            for i in ti.static(range(3)):
                new_p9[i * 3 : i * 3 + 3] = (
                    fem_solver.elements_el_hessian[i_b, i, 0, i_e] @ p9[0:3]
                    + fem_solver.elements_el_hessian[i_b, i, 1, i_e] @ p9[3:6]
                    + fem_solver.elements_el_hessian[i_b, i, 2, i_e] @ p9[6:9]
                )

            # atomic
            self.pcg_state_v[i_b, i_v0].Ap += (
                (B[0, 0] * new_p9[0:3] + B[0, 1] * new_p9[3:6] + B[0, 2] * new_p9[6:9]) * V_dt2 * damping_beta_factor
            )
            self.pcg_state_v[i_b, i_v1].Ap += (
                (B[1, 0] * new_p9[0:3] + B[1, 1] * new_p9[3:6] + B[1, 2] * new_p9[6:9]) * V_dt2 * damping_beta_factor
            )
            self.pcg_state_v[i_b, i_v2].Ap += (
                (B[2, 0] * new_p9[0:3] + B[2, 1] * new_p9[3:6] + B[2, 2] * new_p9[6:9]) * V_dt2 * damping_beta_factor
            )
            self.pcg_state_v[i_b, i_v3].Ap += (
                (s[0] * new_p9[0:3] + s[1] * new_p9[3:6] + s[2] * new_p9[6:9]) * V_dt2 * damping_beta_factor
            )

        pairs = ti.static(self.fem_floor_contact_pairs)
        for i_c in range(self.n_fem_floor_contact_pairs[None]):
            if pairs[i_c].active == 0:
                continue
            i_b = pairs[i_c].batch_idx
            i_e = pairs[i_c].geom_idx

            x = ti.Vector.zero(gs.ti_float, 3)
            for i in ti.static(range(4)):
                i_v = fem_solver.elements_i[i_e].el2v[i]
                x += pairs[i_c].barycentric[i] * self.pcg_state_v[i_b, i_v].p

            x = pairs[i_c].G @ x

            for i in ti.static(range(4)):
                i_v = fem_solver.elements_i[i_e].el2v[i]
                self.pcg_state_v[i_b, i_v].Ap += pairs[i_c].barycentric[i] * x

    @ti.kernel
    def init_pcg_solve(self):
        fem_solver = self.fem_solver
        for i_b in ti.ndrange(self._B):
            self.batch_pcg_active[i_b] = self.batch_active[i_b]
            if not self.batch_pcg_active[i_b]:
                continue
            self.pcg_state[i_b].rTr = 0.0
            self.pcg_state[i_b].rTz = 0.0
        for i_b, i_v in ti.ndrange(self._B, fem_solver.n_vertices):
            if not self.batch_pcg_active[i_b]:
                continue
            self.pcg_state_v[i_b, i_v].x = 0
            self.pcg_state_v[i_b, i_v].r = -self.gradient[i_b, i_v]
            self.pcg_state_v[i_b, i_v].z = self.pcg_state_v[i_b, i_v].prec @ self.pcg_state_v[i_b, i_v].r
            self.pcg_state_v[i_b, i_v].p = self.pcg_state_v[i_b, i_v].z
            ti.atomic_add(self.pcg_state[i_b].rTr, self.pcg_state_v[i_b, i_v].r.dot(self.pcg_state_v[i_b, i_v].r))
            ti.atomic_add(self.pcg_state[i_b].rTz, self.pcg_state_v[i_b, i_v].r.dot(self.pcg_state_v[i_b, i_v].z))
        for i_b in ti.ndrange(self._B):
            if not self.batch_pcg_active[i_b]:
                continue
            self.batch_pcg_active[i_b] = self.pcg_state[i_b].rTr > self._pcg_threshold

    @ti.kernel
    def one_pcg_iter(self):
        self.compute_Ap()

        fem_solver = self.fem_solver

        # compute pTAp
        for i_b in ti.ndrange(self._B):
            if not self.batch_pcg_active[i_b]:
                continue
            self.pcg_state[i_b].pTAp = 0.0
        for i_b, i_v in ti.ndrange(self._B, fem_solver.n_vertices):
            if not self.batch_pcg_active[i_b]:
                continue
            ti.atomic_add(self.pcg_state[i_b].pTAp, self.pcg_state_v[i_b, i_v].p.dot(self.pcg_state_v[i_b, i_v].Ap))

        # compute alpha and update x, r, z, rTr, rTz
        for i_b in ti.ndrange(self._B):
            if not self.batch_pcg_active[i_b]:
                continue
            self.pcg_state[i_b].alpha = self.pcg_state[i_b].rTz / self.pcg_state[i_b].pTAp
            self.pcg_state[i_b].rTr_new = 0.0
            self.pcg_state[i_b].rTz_new = 0.0
        for i_b, i_v in ti.ndrange(self._B, fem_solver.n_vertices):
            if not self.batch_pcg_active[i_b]:
                continue
            self.pcg_state_v[i_b, i_v].x += self.pcg_state[i_b].alpha * self.pcg_state_v[i_b, i_v].p
            self.pcg_state_v[i_b, i_v].r -= self.pcg_state[i_b].alpha * self.pcg_state_v[i_b, i_v].Ap
            self.pcg_state_v[i_b, i_v].z = self.pcg_state_v[i_b, i_v].prec @ self.pcg_state_v[i_b, i_v].r
            ti.atomic_add(self.pcg_state[i_b].rTr_new, self.pcg_state_v[i_b, i_v].r.dot(self.pcg_state_v[i_b, i_v].r))
            ti.atomic_add(self.pcg_state[i_b].rTz_new, self.pcg_state_v[i_b, i_v].r.dot(self.pcg_state_v[i_b, i_v].z))

        # check convergence
        for i_b in ti.ndrange(self._B):
            if not self.batch_pcg_active[i_b]:
                continue
            self.batch_pcg_active[i_b] = self.pcg_state[i_b].rTr_new > self._pcg_threshold

        # update beta, rTr, rTz
        for i_b in ti.ndrange(self._B):
            if not self.batch_pcg_active[i_b]:
                continue
            self.pcg_state[i_b].beta = self.pcg_state[i_b].rTr_new / self.pcg_state[i_b].rTr
            self.pcg_state[i_b].rTr = self.pcg_state[i_b].rTr_new
            self.pcg_state[i_b].rTz = self.pcg_state[i_b].rTz_new

        # update p
        for i_b, i_v in ti.ndrange(self._B, fem_solver.n_vertices):
            if not self.batch_pcg_active[i_b]:
                continue
            self.pcg_state_v[i_b, i_v].p = (
                self.pcg_state_v[i_b, i_v].z + self.pcg_state[i_b].beta * self.pcg_state_v[i_b, i_v].p
            )

    def pcg_solve(self):
        self.init_pcg_solve()
        for i in range(self._n_pcg_iterations):
            self.one_pcg_iter()

    @ti.func
    def compute_total_energy(self, f, energy):
        fem_solver = self.fem_solver
        dt2 = fem_solver._substep_dt**2
        damping_alpha_dt = fem_solver._damping_alpha * fem_solver._substep_dt
        damping_alpha_factor = damping_alpha_dt + 1.0
        damping_beta_over_dt = fem_solver._damping_beta / fem_solver._substep_dt
        damping_beta_factor = damping_beta_over_dt + 1.0

        for i_b in ti.ndrange(self._B):
            if not self.batch_linesearch_active[i_b]:
                continue
            energy[i_b] = 0.0

        # Inertia
        for i_b, i_v in ti.ndrange(self._B, fem_solver.n_vertices):
            if not self.batch_linesearch_active[i_b]:
                continue
            self.v_diff[i_b, i_v] = self.v[i_b, i_v] - fem_solver.elements_v[f + 1, i_v, i_b].vel
            energy[i_b] += (
                0.5
                * fem_solver.elements_v_info[i_v].mass_over_dt2
                * self.v_diff[i_b, i_v].dot(self.v_diff[i_b, i_v])
                * dt2
                * damping_alpha_factor
            )

        # Elastic
        for i_b, i_e in ti.ndrange(self._B, fem_solver.n_elements):
            if not self.batch_linesearch_active[i_b]:
                continue

            V_dt2 = fem_solver.elements_i[i_e].V * dt2
            B = fem_solver.elements_i[i_e].B
            s = -B[0, :] - B[1, :] - B[2, :]  # s is the negative sum of B rows
            p9 = ti.Vector.zero(gs.ti_float, 9)
            i_v0, i_v1, i_v2, i_v3 = fem_solver.elements_i[i_e].el2v

            for i in ti.static(range(3)):
                p9[i * 3 : i * 3 + 3] = (
                    B[0, i] * self.v_diff[i_b, i_v0]
                    + B[1, i] * self.v_diff[i_b, i_v1]
                    + B[2, i] * self.v_diff[i_b, i_v2]
                    + s[i] * self.v_diff[i_b, i_v3]
                )

            H9_p9 = ti.Vector.zero(gs.ti_float, 9)

            for i in ti.static(range(3)):
                H9_p9[i * 3 : i * 3 + 3] = (
                    fem_solver.elements_el_hessian[i_b, i, 0, i_e] @ p9[0:3]
                    + fem_solver.elements_el_hessian[i_b, i, 1, i_e] @ p9[3:6]
                    + fem_solver.elements_el_hessian[i_b, i, 2, i_e] @ p9[6:9]
                )

            energy[i_b] += 0.5 * V_dt2 * p9.dot(H9_p9) * damping_beta_factor

        # Contact
        self.compute_contact_energy(f)
        for i_c in range(self.n_fem_floor_contact_pairs[None]):
            pair = self.fem_floor_contact_pairs[i_c]
            i_b = pair.batch_idx
            if not self.batch_linesearch_active[i_b] or pair.active == 0:
                continue
            energy[i_b] += pair.energy

    @ti.kernel
    def init_linesearch(self, f: ti.i32):
        fem_solver = self.fem_solver
        dt = ti.static(self.sim._substep_dt)
        dt2 = dt**2
        for i_b in ti.ndrange(self._B):
            self.batch_linesearch_active[i_b] = self.batch_active[i_b]
            if not self.batch_linesearch_active[i_b]:
                continue
            self.linesearch_state[i_b].step_size = 1.0 / ti.static(self._linesearch_tau)
            self.linesearch_state[i_b].m = 0.0

        # x_prev, m
        for i_b, i_v in ti.ndrange(self._B, fem_solver.n_vertices):
            if not self.batch_linesearch_active[i_b]:
                continue
            self.linesearch_state_v[i_b, i_v].x_prev = self.v[i_b, i_v]
            self.linesearch_state[i_b].m += self.pcg_state_v[i_b, i_v].x.dot(self.gradient[i_b, i_v])

        self.compute_total_energy(f, self.linesearch_state.prev_energy)

    @ti.kernel
    def one_linesearch_iter(self, f: ti.i32):
        fem_solver = self.fem_solver

        # update vel
        for i_b, i_v in ti.ndrange(self._B, fem_solver.n_vertices):
            if not self.batch_linesearch_active[i_b]:
                continue
            self.v[i_b, i_v] = (
                self.linesearch_state_v[i_b, i_v].x_prev
                + self.linesearch_state[i_b].step_size * self.pcg_state_v[i_b, i_v].x
            )

        self.compute_total_energy(f, self.linesearch_state.energy)

        # check condition
        for i_b in ti.ndrange(self._B):
            if not self.batch_linesearch_active[i_b]:
                continue
            self.batch_linesearch_active[i_b] = (
                self.linesearch_state[i_b].energy
                > self.linesearch_state[i_b].prev_energy
                + self._linesearch_c * self.linesearch_state[i_b].step_size * self.linesearch_state[i_b].m
            )
            if not self.batch_linesearch_active[i_b]:
                continue
            self.linesearch_state[i_b].step_size *= self._linesearch_tau

    def linesearch(self, f: ti.i32):
        """
        Note
        ------
        https://en.wikipedia.org/wiki/Backtracking_line_search#Algorithm
        """
        self.init_linesearch(f)
        for i in range(self._n_linesearch_iterations):
            self.one_linesearch_iter(f)

    @property
    def active_solvers(self):
        """All the active solvers managed by the scene's simulator."""
        return self.sim.active_solvers
