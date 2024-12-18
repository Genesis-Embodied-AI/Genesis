import numpy as np
import taichi as ti

import genesis as gs
from genesis.repr_base import RBC


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
        simulator,
        options,
    ):
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

    def build(self):
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
            self.cpic_flag = ti.field(gs.ti_int, shape=(self.mpm_solver.n_particles, 3, 3, 3))
            self.mpm_rigid_normal = ti.Vector.field(
                3, dtype=gs.ti_float, shape=(self.mpm_solver.n_particles, self.rigid_solver.n_geoms_)
            )

        if self._rigid_sph:
            self.sph_rigid_normal = ti.Vector.field(
                3, dtype=gs.ti_float, shape=(self.sph_solver.n_particles, self.rigid_solver.n_geoms_)
            )
            self.sph_rigid_normal_reordered = ti.Vector.field(
                3, dtype=gs.ti_float, shape=(self.sph_solver.n_particles, self.rigid_solver.n_geoms_)
            )

        if self._rigid_pbd:
            self.pbd_rigid_normal = ti.Vector.field(
                3, dtype=gs.ti_float, shape=(self.pbd_solver.n_particles, self.rigid_solver.n_geoms)
            )
            self.pbd_rigid_normal_reordered = ti.Vector.field(
                3, dtype=gs.ti_float, shape=(self.pbd_solver.n_particles, self.rigid_solver.n_geoms)
            )

        if self._mpm_sph:
            self.mpm_sph_stencil_size = int(np.floor(self.mpm_solver.dx / self.sph_solver.hash_grid_cell_size) + 2)

        if self._mpm_pbd:
            self.mpm_pbd_stencil_size = int(np.floor(self.mpm_solver.dx / self.pbd_solver.hash_grid_cell_size) + 2)

        ## DEBUG
        self._dx = 1 / 1024
        self._stencil_size = int(np.floor(self._dx / self.sph_solver.hash_grid_cell_size) + 2)

        self.reset()

    def reset(self):
        if self._rigid_mpm and self.mpm_solver.enable_CPIC:
            self.mpm_rigid_normal.fill(0)

        if self._rigid_sph:
            self.sph_rigid_normal.fill(0)

    @ti.func
    def _func_collide_with_rigid(self, f, pos_world, vel, mass):
        for i_g in range(self.rigid_solver.n_geoms):
            if self.rigid_solver.geoms_info[i_g].needs_coup:
                i_b = 0
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
    def _func_mpm_tool(self, f, pos_world, vel):
        for entity in ti.static(self.tool_solver.entities):
            if ti.static(entity.material.collision):
                vel = entity.collide(f, pos_world, vel)
        return vel

    @ti.kernel
    def mpm_grid_op(self, f: ti.i32, t: ti.f32):
        """
        This combines mpm's grid_op with coupling operations.
        If we decouple grid_op with coupling with different solvers, we need to run grid-level operations for each coupling pair, which is inefficient.
        """
        for I in ti.grouped(ti.ndrange(*self.mpm_solver.grid_res)):
            if self.mpm_solver.grid[f, I].mass > gs.EPS:
                #################### MPM grid op ####################
                # Momentum to velocity
                vel_mpm = (1 / self.mpm_solver.grid[f, I].mass) * self.mpm_solver.grid[f, I].vel_in

                # gravity
                vel_mpm += self.mpm_solver.substep_dt * self.mpm_solver._gravity[None]

                pos = (I + self.mpm_solver.grid_offset) * self.mpm_solver.dx
                mass_mpm = self.mpm_solver.grid[f, I].mass / self.mpm_solver._p_vol_scale

                # external force fields
                for i_ff in ti.static(range(len(self.mpm_solver._ffs))):
                    vel_mpm += self.mpm_solver._ffs[i_ff].get_acc(pos, vel_mpm, t) * self.mpm_solver.substep_dt

                #################### MPM <-> Tool ####################
                if ti.static(self.tool_solver.is_active()):
                    vel_mpm = self._func_mpm_tool(f, pos, vel_mpm)

                #################### MPM <-> Rigid ####################
                if ti.static(self._rigid_mpm):
                    vel_mpm = self._func_collide_with_rigid(
                        f,
                        pos,
                        vel_mpm,
                        mass_mpm,
                    )

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
                            self.sph_solver.sh.slot_start[slot_idx],
                            self.sph_solver.sh.slot_start[slot_idx] + self.sph_solver.sh.slot_size[slot_idx],
                        ):
                            if (
                                ti.abs(pos - self.sph_solver.particles_reordered.pos[i]).max()
                                < self.mpm_solver.dx * 0.5
                            ):
                                sph_vel += self.sph_solver.particles_reordered.vel[i]
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
                                self.sph_solver.sh.slot_start[slot_idx],
                                self.sph_solver.sh.slot_start[slot_idx] + self.sph_solver.sh.slot_size[slot_idx],
                            ):
                                if (
                                    ti.abs(pos - self.sph_solver.particles_reordered.pos[i]).max()
                                    < self.mpm_solver.dx * 0.5
                                ):
                                    self.sph_solver.particles_reordered[i].vel = (
                                        self.sph_solver.particles_reordered[i].vel
                                        - delta_mv / self.sph_solver.particles_info_reordered[i].mass
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
                            self.pbd_solver.sh.slot_start[slot_idx],
                            self.pbd_solver.sh.slot_start[slot_idx] + self.pbd_solver.sh.slot_size[slot_idx],
                        ):
                            if (
                                ti.abs(pos - self.pbd_solver.particles_reordered.pos[i]).max()
                                < self.mpm_solver.dx * 0.5
                            ):
                                pbd_vel += self.pbd_solver.particles_reordered.vel[i]
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
                                self.pbd_solver.sh.slot_start[slot_idx],
                                self.pbd_solver.sh.slot_start[slot_idx] + self.pbd_solver.sh.slot_size[slot_idx],
                            ):
                                if (
                                    ti.abs(pos - self.pbd_solver.particles_reordered.pos[i]).max()
                                    < self.mpm_solver.dx * 0.5
                                ):
                                    if self.pbd_solver.particles_reordered[i].free:
                                        self.pbd_solver.particles_reordered[i].vel = (
                                            self.pbd_solver.particles_reordered[i].vel
                                            - delta_mv / self.pbd_solver.particles_info_reordered[i].mass
                                        )

                #################### MPM boundary ####################
                _, self.mpm_solver.grid[f, I].vel_out = self.mpm_solver.boundary.impose_pos_vel(pos, vel_mpm)

    @ti.kernel
    def mpm_surface_to_particle(self, f: ti.i32):
        for i in range(self.mpm_solver.n_particles):
            if self.mpm_solver.particles_ng[f, i].active:
                for i_g in range(self.rigid_solver.n_geoms):
                    if self.rigid_solver.geoms_info[i_g].needs_coup:
                        sdf_normal = self.rigid_solver.sdf.sdf_normal_world(self.mpm_solver.particles[f, i].pos, i_g, 0)
                        # we only update the normal if the particle does not the object
                        if sdf_normal.dot(self.mpm_rigid_normal[i, i_g]) >= 0:
                            self.mpm_rigid_normal[i, i_g] = sdf_normal

    @ti.kernel
    def fem_surface_force(self, f: ti.i32):
        # TODO: all collisions are on vertices instead of surface and edge
        for i in range(self.fem_solver.n_surfaces):
            if self.fem_solver.surface[i].active:
                dt = self.fem_solver.substep_dt
                iel = self.fem_solver.surface[i].tri2el
                mass = self.fem_solver.elements_i[iel].mass_scaled / self.fem_solver.vol_scale

                p1 = self.fem_solver.elements_v[f, self.fem_solver.surface[i].tri2v[0]].pos
                p2 = self.fem_solver.elements_v[f, self.fem_solver.surface[i].tri2v[1]].pos
                p3 = self.fem_solver.elements_v[f, self.fem_solver.surface[i].tri2v[2]].pos
                u = p2 - p1
                v = p3 - p1
                surface_normal = ti.math.cross(u, v)
                surface_normal = surface_normal / surface_normal.norm(gs.EPS)

                # FEM <-> Rigid
                if ti.static(self._rigid_fem):
                    # NOTE: collision only on surface vertices
                    for j in ti.static(range(3)):
                        iv = self.fem_solver.surface[i].tri2v[j]
                        vel_fem_sv = self._func_collide_with_rigid(
                            f,
                            self.fem_solver.elements_v[f, iv].pos,
                            self.fem_solver.elements_v[f + 1, iv].vel,
                            mass / 3.0,  # assume element mass uniformly distributed to vertices
                        )
                        self.fem_solver.elements_v[f + 1, iv].vel = vel_fem_sv

                # FEM <-> MPM (interact with MPM grid instead of particles)
                # NOTE: not doing this in mpm_grid_op otherwise we need to search for fem surface for each particles
                #       however, this function is called after mpm boundary conditions.
                if ti.static(self._fem_mpm):
                    for j in ti.static(range(3)):
                        iv = self.fem_solver.surface[i].tri2v[j]
                        pos = self.fem_solver.elements_v[f, iv].pos
                        vel_fem_sv = self.fem_solver.elements_v[f + 1, iv].vel
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
                            mpm_grid_mass = self.mpm_solver.grid[f, mpm_grid_I].mass / self.mpm_solver.p_vol_scale

                            mpm_weight = ti.cast(1.0, gs.ti_float)
                            for d in ti.static(range(3)):
                                mpm_weight *= mpm_w[mpm_offset[d]][d]

                            # FEM -> MPM
                            mpm_grid_pos = (mpm_grid_I + self.mpm_solver.grid_offset) * self.mpm_solver.dx
                            signed_dist = (mpm_grid_pos - pos).dot(surface_normal)
                            if signed_dist <= self.mpm_solver.dx:  # NOTE: use dx as minimal unit for collision
                                vel_mpm_at_cell = mpm_weight * self.mpm_solver.grid[f, mpm_grid_I].vel_out
                                mass_mpm_at_cell = mpm_weight * mpm_grid_mass

                                vel_mpm += vel_mpm_at_cell
                                mass_mpm += mass_mpm_at_cell

                                if mass_mpm_at_cell > gs.EPS:
                                    delta_mpm_vel_at_cell_unmul = (
                                        vel_fem_sv * mpm_weight - self.mpm_solver.grid[f, mpm_grid_I].vel_out
                                    )
                                    mass_mul_at_cell = (
                                        mpm_grid_mass / mass_fem_sv
                                    )  # NOTE: use un-reweighted mass instead of mass_mpm_at_cell
                                    delta_mpm_vel_at_cell = delta_mpm_vel_at_cell_unmul * mass_mul_at_cell
                                    self.mpm_solver.grid[f, mpm_grid_I].vel_out += delta_mpm_vel_at_cell

                                    new_vel_fem_sv -= delta_mpm_vel_at_cell * mass_mpm_at_cell / mass_fem_sv

                        # MPM -> FEM
                        if mass_mpm > gs.EPS:
                            # delta_mv = (vel_mpm - vel_fem_sv) * mass_mpm
                            # delta_vel_fem_sv = delta_mv / mass_fem_sv
                            # self.fem_solver.elements_v[f + 1, iv].vel += delta_vel_fem_sv
                            self.fem_solver.elements_v[f + 1, iv].vel = new_vel_fem_sv

                # FEM <-> SPH TODO: this doesn't work well
                if ti.static(self._fem_sph):
                    for j in ti.static(range(3)):
                        iv = self.fem_solver.surface[i].tri2v[j]
                        pos = self.fem_solver.elements_v[f, iv].pos
                        vel_fem_sv = self.fem_solver.elements_v[f + 1, iv].vel
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
                                self.sph_solver.sh.slot_start[slot_idx],
                                self.sph_solver.sh.slot_start[slot_idx] + self.sph_solver.sh.slot_size[slot_idx],
                            ):
                                if ti.abs(pos - self.sph_solver.particles_reordered.pos[k]).max() < dx * 0.5:
                                    sph_vel += self.sph_solver.particles_reordered.vel[k]
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
                                    self.sph_solver.sh.slot_start[slot_idx],
                                    self.sph_solver.sh.slot_start[slot_idx] + self.sph_solver.sh.slot_size[slot_idx],
                                ):
                                    if ti.abs(pos - self.sph_solver.particles_reordered.pos[k]).max() < dx * 0.5:
                                        self.sph_solver.particles_reordered[k].vel = (
                                            self.sph_solver.particles_reordered[k].vel
                                            - delta_mv / self.sph_solver.particles_info_reordered[k].mass
                                        )

                            self.fem_solver.elements_v[f + 1, iv].vel = vel_fem_sv

                # boundary condition
                for j in ti.static(range(3)):
                    iv = self.fem_solver.surface[i].tri2v[j]
                    _, self.fem_solver.elements_v[f + 1, iv].vel = self.fem_solver.boundary.impose_pos_vel(
                        self.fem_solver.elements_v[f, iv].pos, self.fem_solver.elements_v[f + 1, iv].vel
                    )

    @ti.kernel
    def sph_rigid(self, f: ti.i32):
        for i in range(self.sph_solver._n_particles):
            if self.sph_solver.particles_ng_reordered[i].active:

                for i_g in range(self.rigid_solver.n_geoms):
                    if self.rigid_solver.geoms_info[i_g].needs_coup:
                        i_b = 0
                        self.sph_solver.particles_reordered[i].vel, self.sph_rigid_normal_reordered[i, i_g] = (
                            self._func_collide_with_rigid_geom_robust(
                                self.sph_solver.particles_reordered[i].pos,
                                self.sph_solver.particles_reordered[i].vel,
                                self.sph_solver.particles_info_reordered[i].mass,
                                self.sph_rigid_normal_reordered[i, i_g],
                                i_g,
                                i_b,
                            )
                        )

    @ti.kernel
    def pbd_rigid(self, f: ti.i32):
        for i in range(self.pbd_solver._n_particles):
            if self.pbd_solver.particles_ng_reordered[i].active:
                # NOTE: Couldn't figure out a good way to handle collision with non-free particle. Such collision is not phsically plausible anyway.

                for i_g in range(self.rigid_solver.n_geoms):
                    if self.rigid_solver.geoms_info[i_g].needs_coup:
                        i_b = 0
                        (
                            self.pbd_solver.particles_reordered[i].pos,
                            self.pbd_solver.particles_reordered[i].vel,
                            self.pbd_rigid_normal_reordered[i, i_g],
                        ) = self._func_pbd_collide_with_rigid_geom(
                            i,
                            self.pbd_solver.particles_reordered[i].pos,
                            self.pbd_solver.particles_reordered[i].vel,
                            self.pbd_solver.particles_info_reordered[i].mass,
                            self.pbd_rigid_normal_reordered[i, i_g],
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
            v_norm = (new_pos - self.pbd_solver.particles_reordered[i].ipos) / self.pbd_solver._substep_dt

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
