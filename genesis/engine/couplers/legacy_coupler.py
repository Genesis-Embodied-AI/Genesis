from typing import TYPE_CHECKING

import numpy as np
import gstaichi as ti

import genesis as gs
import genesis.utils.sdf_decomp as sdf_decomp

from genesis.options.solvers import LegacyCouplerOptions
from genesis.repr_base import RBC
from genesis.utils.array_class import LinksState
from genesis.utils.geom import ti_inv_transform_by_trans_quat, ti_transform_by_trans_quat

if TYPE_CHECKING:
    from genesis.engine.simulator import Simulator

CLAMPED_INV_DT = 50.0


@ti.data_oriented
class LegacyCoupler(RBC):
    """
    This class handles all the coupling between different solvers. LegacyCoupler will be deprecated in the future.
    """

    # ------------------------------------------------------------------------------------
    # --------------------------------- Initialization -----------------------------------
    # ------------------------------------------------------------------------------------

    def __init__(
        self,
        simulator: "Simulator",
        options: "LegacyCouplerOptions",
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
        self._rigid_mpm = self.rigid_solver.is_active and self.mpm_solver.is_active and self.options.rigid_mpm
        self._rigid_sph = self.rigid_solver.is_active and self.sph_solver.is_active and self.options.rigid_sph
        self._rigid_pbd = self.rigid_solver.is_active and self.pbd_solver.is_active and self.options.rigid_pbd
        self._rigid_fem = self.rigid_solver.is_active and self.fem_solver.is_active and self.options.rigid_fem
        self._mpm_sph = self.mpm_solver.is_active and self.sph_solver.is_active and self.options.mpm_sph
        self._mpm_pbd = self.mpm_solver.is_active and self.pbd_solver.is_active and self.options.mpm_pbd
        self._fem_mpm = self.fem_solver.is_active and self.mpm_solver.is_active and self.options.fem_mpm
        self._fem_sph = self.fem_solver.is_active and self.sph_solver.is_active and self.options.fem_sph

        if gs.use_ndarray and (self._rigid_mpm or self._rigid_sph or self._rigid_pbd or self._rigid_fem):
            gs.raise_exception(
                "LegacyCoupler does not support Gstaichi dynamic array type in case of coupling Rigid vs MPM, SPH, PBD "
                "or FEM. Please enable performance mode at init, gs.init(..., performance_mode=True)."
            )

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

            struct_particle_attach_info = ti.types.struct(
                link_idx=gs.ti_int,
                local_pos=gs.ti_vec3,
            )
            pbd_batch_shape = self.pbd_solver._batch_shape(self.pbd_solver._n_particles)

            self.particle_attach_info = struct_particle_attach_info.field(shape=pbd_batch_shape, layout=ti.Layout.SOA)
            self.particle_attach_info.link_idx.fill(-1)
            self.particle_attach_info.local_pos.fill(gs.ti_vec3(0.0, 0.0, 0.0))

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
            if self.rigid_solver.geoms_info.needs_coup[i_g]:
                vel = self._func_collide_with_rigid_geom(pos_world, vel, mass, i_g, i_b)
        return vel

    @ti.func
    def _func_collide_with_rigid_geom(self, pos_world, vel, mass, geom_idx, batch_idx):
        signed_dist = sdf_decomp.sdf_func_world(
            geoms_state=self.rigid_solver.geoms_state,
            geoms_info=self.rigid_solver.geoms_info,
            sdf_info=self.rigid_solver.sdf._sdf_info,
            pos_world=pos_world,
            geom_idx=geom_idx,
            batch_idx=batch_idx,
        )

        # bigger coup_softness implies that the coupling influence extends further away from the object.
        influence = ti.min(ti.exp(-signed_dist / max(1e-10, self.rigid_solver.geoms_info.coup_softness[geom_idx])), 1)

        if influence > 0.1:
            normal_rigid = sdf_decomp.sdf_func_normal_world(
                geoms_state=self.rigid_solver.geoms_state,
                geoms_info=self.rigid_solver.geoms_info,
                collider_static_config=self.rigid_solver.collider._collider_static_config,
                sdf_info=self.rigid_solver.sdf._sdf_info,
                pos_world=pos_world,
                geom_idx=geom_idx,
                batch_idx=batch_idx,
            )
            vel = self._func_collide_in_rigid_geom(pos_world, vel, mass, normal_rigid, influence, geom_idx, batch_idx)

        return vel

    @ti.func
    def _func_collide_with_rigid_geom_robust(self, pos_world, vel, mass, normal_prev, geom_idx, batch_idx):
        """
        Similar to _func_collide_with_rigid_geom, but additionally handles potential side flip due to penetration.
        """
        signed_dist = sdf_decomp.sdf_func_world(
            geoms_state=self.rigid_solver.geoms_state,
            geoms_info=self.rigid_solver.geoms_info,
            sdf_info=self.rigid_solver.sdf._sdf_info,
            pos_world=pos_world,
            geom_idx=geom_idx,
            batch_idx=batch_idx,
        )
        normal_rigid = sdf_decomp.sdf_func_normal_world(
            geoms_state=self.rigid_solver.geoms_state,
            geoms_info=self.rigid_solver.geoms_info,
            collider_static_config=self.rigid_solver.collider._collider_static_config,
            sdf_info=self.rigid_solver.sdf._sdf_info,
            pos_world=pos_world,
            geom_idx=geom_idx,
            batch_idx=batch_idx,
        )

        # bigger coup_softness implies that the coupling influence extends further away from the object.
        influence = ti.min(ti.exp(-signed_dist / max(1e-10, self.rigid_solver.geoms_info.coup_softness[geom_idx])), 1)

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
        vel_rigid = self.rigid_solver._func_vel_at_point(
            pos_world=pos_world,
            link_idx=self.rigid_solver.geoms_info.link_idx[geom_idx],
            i_b=batch_idx,
            links_state=self.rigid_solver.links_state,
        )

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
                rvel_tan
                / rvel_tan_norm
                * ti.max(
                    0, rvel_tan_norm + rvel_normal_magnitude * self.rigid_solver.geoms_info.coup_friction[geom_idx]
                )
            )

            # normal component after collision
            rvel_normal = (
                -normal_rigid * rvel_normal_magnitude * self.rigid_solver.geoms_info.coup_restitution[geom_idx]
            )

            # normal + tangential component
            rvel_new = rvel_tan + rvel_normal

            # apply influence
            vel_old = vel
            vel = vel_rigid + rvel_new * influence + rvel * (1 - influence)

            #################### particle -> rigid ####################
            # Compute delta momentum and apply to rigid body.
            delta_mv = mass * (vel - vel_old)
            force = -delta_mv / self.rigid_solver.substep_dt
            self.rigid_solver._func_apply_external_force(
                pos_world,
                force,
                self.rigid_solver.geoms_info.link_idx[geom_idx],
                batch_idx,
                self.rigid_solver.links_state,
            )

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
                mass_mpm = self.mpm_solver.grid[f, I, i_b].mass / self.mpm_solver._particle_volume_scale

                # external force fields
                for i_ff in ti.static(range(len(self.mpm_solver._ffs))):
                    vel_mpm += self.mpm_solver._ffs[i_ff].get_acc(pos, vel_mpm, t, -1) * self.mpm_solver.substep_dt

                #################### MPM <-> Tool ####################
                if ti.static(self.tool_solver.is_active):
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
                    if self.rigid_solver.geoms_info.needs_coup[i_g]:
                        sdf_normal = sdf_decomp.sdf_func_normal_world(
                            geoms_state=self.rigid_solver.geoms_state,
                            geoms_info=self.rigid_solver.geoms_info,
                            collider_static_config=self.rigid_solver.collider._collider_static_config,
                            sdf_info=self.rigid_solver.sdf._sdf_info,
                            pos_world=self.mpm_solver.particles[f, i_p, i_b].pos,
                            geom_idx=i_g,
                            batch_idx=i_b,
                        )
                        # we only update the normal if the particle does not the object
                        if sdf_normal.dot(self.mpm_rigid_normal[i_p, i_g, i_b]) >= 0:
                            self.mpm_rigid_normal[i_p, i_g, i_b] = sdf_normal

    def fem_rigid_link_constraints(self):
        if self.fem_solver._constraints_initialized and self.rigid_solver.is_active:
            links_pos = self.rigid_solver.links_state.pos
            links_quat = self.rigid_solver.links_state.quat
            self.fem_solver._kernel_update_linked_vertex_constraints(links_pos, links_quat)

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
                            mpm_grid_mass = (
                                self.mpm_solver.grid[f, mpm_grid_I, i_b].mass / self.mpm_solver.particle_volume_scale
                            )

                            mpm_weight = gs.ti_float(1.0)
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
                    if self.rigid_solver.geoms_info.needs_coup[i_g]:
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
    def kernel_pbd_rigid_collide(self):
        for i_p, i_b in ti.ndrange(self.pbd_solver._n_particles, self.sph_solver._B):
            if self.pbd_solver.particles_ng_reordered[i_p, i_b].active:
                # NOTE: Couldn't figure out a good way to handle collision with non-free particle. Such collision is not phsically plausible anyway.
                for i_g in range(self.rigid_solver.n_geoms):
                    if self.rigid_solver.geoms_info.needs_coup[i_g]:
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

    @ti.kernel
    def kernel_attach_pbd_to_rigid_link(
        self,
        particles_idx: ti.types.ndarray(),
        envs_idx: ti.types.ndarray(),
        link_idx: ti.i32,
        links_state: LinksState,
    ) -> None:
        """
        Sets listed particles in listed environments to be animated by the link.

        Current position of the particle, relatively to the link, is stored and preserved.
        """
        pdb = self.pbd_solver

        for i_p_, i_b_ in ti.ndrange(particles_idx.shape[1], envs_idx.shape[0]):
            i_p = particles_idx[i_b_, i_p_]
            i_b = envs_idx[i_b_]
            link_pos = links_state.pos[link_idx, i_b]
            link_quat = links_state.quat[link_idx, i_b]

            # compute local offset from link to the particle
            world_pos = pdb.particles[i_p, i_b].pos
            local_pos = ti_inv_transform_by_trans_quat(world_pos, link_pos, link_quat)

            # set particle to be animated (not free) and store animation info
            pdb.particles[i_p, i_b].free = False
            self.particle_attach_info[i_p, i_b].link_idx = link_idx
            self.particle_attach_info[i_p, i_b].local_pos = local_pos

    @ti.kernel
    def kernel_pbd_rigid_clear_animate_particles_by_link(
        self,
        particles_idx: ti.types.ndarray(),
        envs_idx: ti.types.ndarray(),
    ) -> None:
        """Detach listed particles from links, and simulate them freely."""
        pdb = self.pbd_solver
        for i_p_, i_b_ in ti.ndrange(particles_idx.shape[1], envs_idx.shape[0]):
            i_p = particles_idx[i_b_, i_p_]
            i_b = envs_idx[i_b_]
            pdb.particles[i_p, i_b].free = True
            self.particle_attach_info[i_p, i_b].link_idx = -1
            self.particle_attach_info[i_p, i_b].local_pos = ti.math.vec3([0.0, 0.0, 0.0])

    @ti.kernel
    def kernel_pbd_rigid_solve_animate_particles_by_link(self, clamped_inv_dt: ti.f32, links_state: LinksState):
        """
        Itearates all particles and environments, and sets corrective velocity for all animated particle.

        Computes target position and velocity from the attachment/reference link and local offset position.

        Note, that this step shoudl be done after rigid solver update, and before PDB solver update.
        Currently, this is done after both rigid and PBD solver updates, hence the corrective velocity
        is off by a frame.

        Note, it's adviced to clamp inv_dt to avoid large jerks and instability. 1/0.02 might be a good max value.
        """
        pdb = self.pbd_solver
        for i_p, i_env in ti.ndrange(pdb._n_particles, pdb._B):
            if self.particle_attach_info[i_p, i_env].link_idx >= 0:

                # read link state
                link_idx = self.particle_attach_info[i_p, i_env].link_idx
                link_pos = links_state.pos[link_idx, i_env]
                link_quat = links_state.quat[link_idx, i_env]

                link_lin_vel = links_state.cd_vel[link_idx, i_env]
                link_ang_vel = links_state.cd_ang[link_idx, i_env]
                link_com_in_world = links_state.root_COM[link_idx, i_env] + links_state.i_pos[link_idx, i_env]

                # calculate target pos and vel of the particle
                local_pos = self.particle_attach_info[i_p, i_env].local_pos
                target_world_pos = ti_transform_by_trans_quat(local_pos, link_pos, link_quat)

                world_arm = target_world_pos - link_com_in_world
                target_world_vel = link_lin_vel + link_ang_vel.cross(world_arm)

                # compute and apply corrective velocity
                i_rp = pdb.particles_ng[i_p, i_env].reordered_idx
                particle_pos = pdb.particles_reordered[i_rp, i_env].pos
                pos_correction = target_world_pos - particle_pos
                corrective_vel = pos_correction * clamped_inv_dt
                old_vel = pdb.particles_reordered[i_rp, i_env].vel
                pdb.particles_reordered[i_rp, i_env].vel = corrective_vel + target_world_vel

    @ti.func
    def _func_pbd_collide_with_rigid_geom(self, i, pos_world, vel, mass, normal_prev, geom_idx, batch_idx):
        """
        Resolves collision when a particle is already in collision with a rigid object.
        This function assumes known normal_rigid and influence.
        """
        signed_dist = sdf_decomp.sdf_func_world(
            geoms_state=self.rigid_solver.geoms_state,
            geoms_info=self.rigid_solver.geoms_info,
            sdf_info=self.rigid_solver.sdf._sdf_info,
            pos_world=pos_world,
            geom_idx=geom_idx,
            batch_idx=batch_idx,
        )
        vel_rigid = self.rigid_solver._func_vel_at_point(
            pos_world=pos_world,
            link_idx=self.rigid_solver.geoms_info.link_idx[geom_idx],
            i_b=batch_idx,
            links_state=self.rigid_solver.links_state,
        )
        contact_normal = sdf_decomp.sdf_func_normal_world(
            geoms_state=self.rigid_solver.geoms_state,
            geoms_info=self.rigid_solver.geoms_info,
            collider_static_config=self.rigid_solver.collider._collider_static_config,
            sdf_info=self.rigid_solver.sdf._sdf_info,
            pos_world=pos_world,
            geom_idx=geom_idx,
            batch_idx=batch_idx,
        )
        new_pos = pos_world
        new_vel = vel
        if signed_dist < self.pbd_solver.particle_size / 2:  # skip non-penetration particles
            stiffness = 1.0  # value in [0, 1]

            # we don't consider friction for now
            # friction = 0.15
            # rvel = vel - vel_rigid
            # rvel_normal_magnitude = rvel.dot(contact_normal)  # negative if inward
            # rvel_tan = rvel - rvel_normal_magnitude * contact_normal
            # rvel_tan_norm = rvel_tan.norm(gs.EPS)

            #################### rigid -> particle ####################

            energy_loss = 0.0  # value in [0, 1]
            new_pos = pos_world + stiffness * contact_normal * (self.pbd_solver.particle_size / 2 - signed_dist)
            prev_pos = self.pbd_solver.particles_reordered[i, batch_idx].ipos
            new_vel = (new_pos - prev_pos) / self.pbd_solver._substep_dt

            #################### particle -> rigid ####################
            delta_mv = mass * (new_vel - vel)
            force = (-delta_mv / self.rigid_solver._substep_dt) * (1 - energy_loss)

            self.rigid_solver._func_apply_external_force(
                pos_world,
                force,
                self.rigid_solver.geoms_info.link_idx[geom_idx],
                batch_idx,
                self.rigid_solver.links_state,
            )

        return new_pos, new_vel, contact_normal

    def preprocess(self, f):
        # preprocess for MPM CPIC
        if self._rigid_mpm and self.mpm_solver.enable_CPIC:
            self.mpm_surface_to_particle(f)

    def couple(self, f):
        # MPM <-> all others
        if self.mpm_solver.is_active:
            self.mpm_grid_op(f, self.sim.cur_t)

        # SPH <-> Rigid
        if self._rigid_sph:
            self.sph_rigid(f)

        # PBD <-> Rigid
        if self._rigid_pbd:
            self.kernel_pbd_rigid_collide()

            # 1-way: animate particles by links
            full_step_inv_dt = 1.0 / self.pbd_solver._dt
            clamped_inv_dt = min(full_step_inv_dt, CLAMPED_INV_DT)
            self.kernel_pbd_rigid_solve_animate_particles_by_link(clamped_inv_dt, self.rigid_solver.links_state)

        if self.fem_solver.is_active:
            self.fem_surface_force(f)
            self.fem_rigid_link_constraints()

    def couple_grad(self, f):
        if self.mpm_solver.is_active:
            self.mpm_grid_op.grad(f, self.sim.cur_t)

        if self.fem_solver.is_active:
            self.fem_surface_force.grad(f)

    @property
    def active_solvers(self):
        """All the active solvers managed by the scene's simulator."""
        return self.sim.active_solvers
