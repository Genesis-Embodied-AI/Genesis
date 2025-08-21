import sys
from typing import TYPE_CHECKING
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import torch

import gstaichi as ti

import genesis as gs
import genesis.utils.geom as gu
from genesis.styles import colors, formats
from genesis.utils.misc import ti_field_to_torch
import genesis.utils.array_class as array_class
import genesis.engine.solvers.rigid.gjk_decomp as gjk
import genesis.engine.solvers.rigid.mpr_decomp as mpr
import genesis.utils.sdf_decomp as sdf
import genesis.engine.solvers.rigid.support_field_decomp as support_field
import genesis.engine.solvers.rigid.rigid_solver_decomp as rigid_solver

from .mpr_decomp import MPR
from .gjk_decomp import GJK
from ....utils.sdf_decomp import SDF
from .support_field_decomp import SupportField

from enum import IntEnum

if TYPE_CHECKING:
    from genesis.engine.solvers.rigid.rigid_solver_decomp import RigidSolver


class CCD_ALGORITHM_CODE(IntEnum):
    # Our MPR (with SDF)
    MPR = 0
    # MuJoCo MPR
    MJ_MPR = 1
    # Our GJK
    GJK = 2
    # MuJoCo GJK
    MJ_GJK = 3


@ti.data_oriented
class Collider:

    @ti.data_oriented
    class ColliderStaticConfig:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

    def __init__(self, rigid_solver: "RigidSolver"):
        self._solver = rigid_solver

        self._init_static_config()
        self._init_collision_fields()

        # FIXME: MPR is necessary because it is used for terrain collision detection
        self._mpr = MPR(rigid_solver)
        self._gjk = GJK(rigid_solver)
        self._sdf = SDF(rigid_solver)

        # Support field used for mpr and gjk. Rather than having separate support fields for each algorithm, keep only
        # one copy here to save memory and maintain cleaner code.
        self._support_field = SupportField(rigid_solver)

    def _init_static_config(self) -> None:
        # Identify the convex collision detection (ccd) algorithm
        if self._solver._options.use_gjk_collision:
            if self._solver._enable_mujoco_compatibility:
                ccd_algorithm = CCD_ALGORITHM_CODE.MJ_GJK
            else:
                ccd_algorithm = CCD_ALGORITHM_CODE.GJK
        else:
            if self._solver._enable_mujoco_compatibility:
                ccd_algorithm = CCD_ALGORITHM_CODE.MJ_MPR
            else:
                ccd_algorithm = CCD_ALGORITHM_CODE.MPR

        # Initialize the static config
        self._collider_static_config = Collider.ColliderStaticConfig(
            has_nonconvex_nonterrain=np.logical_and(
                self._solver.geoms_info.is_convex.to_numpy() == 0,
                self._solver.geoms_info.type.to_numpy() != gs.GEOM_TYPE.TERRAIN,
            ).any(),
            has_terrain=(self._solver.geoms_info.type.to_numpy() == gs.GEOM_TYPE.TERRAIN).any(),
            # multi contact perturbation and tolerance
            mc_perturbation=1e-3 if self._solver._enable_mujoco_compatibility else 1e-2,
            mc_tolerance=1e-3 if self._solver._enable_mujoco_compatibility else 1e-2,
            mpr_to_sdf_overlap_ratio=0.4,
            # multiplier k for the maximum number of contact pairs for the broad phase
            max_collision_pairs_broad_k=20,
            # maximum number of contact pairs per collision pair
            n_contacts_per_pair=5,
            # maximum number of contact points for box-box collision detection
            box_MAXCONPAIR=16,
            # ccd algorithm
            ccd_algorithm=ccd_algorithm,
        )

    def _init_collision_fields(self) -> None:
        # Pre-compute fields, as they are needed to initialize the collider state and info.
        n_possible_pairs, collision_pair_validity = self._compute_collision_pair_validity()
        vert_neighbors, vert_neighbor_start, vert_n_neighbors = self._compute_verts_connectivity()
        n_vert_neighbors = len(vert_neighbors)

        # Initialize [state], which stores every MUTABLE collision data.
        self._collider_state = array_class.get_collider_state(
            self._solver, n_possible_pairs, self._collider_static_config
        )
        # array_class.ColliderState(self._solver, n_possible_pairs, self._collider_static_config)

        # Initialize [info], which stores every IMMUTABLE collision data.
        # self._collider_info = array_class.ColliderInfo(self._solver, n_vert_neighbors, self._collider_static_config)
        self._collider_info = array_class.get_collider_info(
            self._solver, n_vert_neighbors, self._collider_static_config
        )
        self._init_collision_pair_validity(collision_pair_validity)
        self._init_verts_connectivity(vert_neighbors, vert_neighbor_start, vert_n_neighbors)
        self._init_max_contact_pairs(n_possible_pairs)
        self._init_terrain_state()

        # [contacts_info_cache] is not used in Taichi kernels, so keep it outside of the collider state / info.
        self._contacts_info_cache = {}

        # FIXME: 'ti.static_print' cannot be used as it will be printed systematically, completely ignoring guard
        # condition, while 'print' is slowing down the kernel even if every called in practice...
        self._warn_msg_max_collision_pairs = (
            f"{colors.YELLOW}[Genesis] [00:00:00] [WARNING] Ignoring contact pair to avoid exceeding max "
            f"({self._collider_info._max_contact_pairs[None]}). Please increase the value of RigidSolver's option "
            f"'max_collision_pairs'.{formats.RESET}"
        )

        self.reset()

    def _compute_collision_pair_validity(self):
        """
        Compute the collision pair validity matrix.

        For each pair of geoms, determine if they can collide based on their properties and the solver configuration.
        """
        solver = self._solver
        n_geoms = solver.n_geoms_
        enable_self_collision = solver._static_rigid_sim_config.enable_self_collision
        enable_adjacent_collision = solver._static_rigid_sim_config.enable_adjacent_collision
        batch_links_info = solver._static_rigid_sim_config.batch_links_info

        geoms_link_idx = solver.geoms_info.link_idx.to_numpy()
        geoms_contype = solver.geoms_info.contype.to_numpy()
        geoms_conaffinity = solver.geoms_info.conaffinity.to_numpy()
        links_entity_idx = solver.links_info.entity_idx.to_numpy()
        links_root_idx = solver.links_info.root_idx.to_numpy()
        links_parent_idx = solver.links_info.parent_idx.to_numpy()
        links_is_fixed = solver.links_info.is_fixed.to_numpy()
        if batch_links_info:
            links_entity_idx = links_entity_idx[:, 0]
            links_root_idx = links_root_idx[:, 0]
            links_parent_idx = links_parent_idx[:, 0]
            links_is_fixed = links_is_fixed[:, 0]
        entities_is_local_collision_mask = solver.entities_info.is_local_collision_mask.to_numpy()

        n_possible_pairs = 0
        collision_pair_validity = np.zeros((n_geoms, n_geoms), dtype=gs.np_int)
        for i_ga in range(n_geoms):
            for i_gb in range(i_ga + 1, n_geoms):
                i_la = geoms_link_idx[i_ga]
                i_lb = geoms_link_idx[i_gb]
                i_ea = links_entity_idx[i_la]
                i_eb = links_entity_idx[i_lb]

                # geoms in the same link
                if i_la == i_lb:
                    continue

                # self collision
                if links_root_idx[i_la] == links_root_idx[i_lb]:
                    if not enable_self_collision:
                        continue

                    # adjacent links
                    if not enable_adjacent_collision and (
                        links_parent_idx[i_la] == i_lb or links_parent_idx[i_lb] == i_la
                    ):
                        continue

                # contype and conaffinity
                if (
                    (i_ea == i_eb)
                    or not (entities_is_local_collision_mask[i_ea] or entities_is_local_collision_mask[i_eb])
                ) and not (
                    (geoms_contype[i_ga] & geoms_conaffinity[i_gb]) or (geoms_contype[i_gb] & geoms_conaffinity[i_ga])
                ):
                    continue

                # pair of fixed links wrt the world
                if links_is_fixed[i_la] and links_is_fixed[i_lb]:
                    continue

                collision_pair_validity[i_ga, i_gb] = 1
                n_possible_pairs += 1

        return n_possible_pairs, collision_pair_validity

    def _compute_verts_connectivity(self):
        """
        Compute the vertex connectivity.
        """
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

        return vert_neighbors, vert_neighbor_start, vert_n_neighbors

    def _init_collision_pair_validity(self, collision_pair_validity):
        self._collider_info.collision_pair_validity.from_numpy(collision_pair_validity)

    def _init_verts_connectivity(self, vert_neighbors, vert_neighbor_start, vert_n_neighbors):
        if self._solver.n_verts > 0:
            self._collider_info.vert_neighbors.from_numpy(vert_neighbors)
            self._collider_info.vert_neighbor_start.from_numpy(vert_neighbor_start)
            self._collider_info.vert_n_neighbors.from_numpy(vert_n_neighbors)

    def _init_max_contact_pairs(self, n_possible_pairs):
        max_collision_pairs = min(self._solver._max_collision_pairs, n_possible_pairs)
        max_contact_pairs = max_collision_pairs * self._collider_static_config.n_contacts_per_pair
        max_contact_pairs_broad = max_collision_pairs * self._collider_static_config.max_collision_pairs_broad_k

        self._collider_info._max_possible_pairs[None] = n_possible_pairs
        self._collider_info._max_collision_pairs[None] = max_collision_pairs
        self._collider_info._max_collision_pairs_broad[None] = max_contact_pairs_broad

        self._collider_info._max_contact_pairs[None] = max_contact_pairs

    def _init_terrain_state(self):
        if self._collider_static_config.has_terrain:
            solver = self._solver
            links_idx = solver.geoms_info.link_idx.to_numpy()[solver.geoms_info.type.to_numpy() == gs.GEOM_TYPE.TERRAIN]
            entity = solver._entities[solver.links_info.entity_idx.to_numpy()[links_idx[0]]]

            scale = entity.terrain_scale.astype(gs.np_float)
            rc = np.array(entity.terrain_hf.shape, dtype=gs.np_int)
            hf = entity.terrain_hf.astype(gs.np_float) * scale[1]
            xyz_maxmin = np.array(
                [rc[0] * scale[0], rc[1] * scale[0], hf.max(), 0, 0, hf.min() - 1.0],
                dtype=gs.np_float,
            )

            self._collider_info.terrain_hf.from_numpy(hf)
            self._collider_info.terrain_rc.from_numpy(rc)
            self._collider_info.terrain_scale.from_numpy(scale)
            self._collider_info.terrain_xyz_maxmin.from_numpy(xyz_maxmin)

    def reset(self, envs_idx: npt.NDArray[np.int32] | None = None) -> None:
        if envs_idx is None:
            envs_idx = self._solver._scene._envs_idx
        collider_kernel_reset(
            envs_idx,
            self._solver._static_rigid_sim_config,
            self._collider_state,
        )
        self._contacts_info_cache = {}

    def clear(self, envs_idx=None):
        if envs_idx is None:
            envs_idx = self._solver._scene._envs_idx
        kernel_collider_clear(
            envs_idx,
            self._solver.links_state,
            self._solver.links_info,
            self._solver._static_rigid_sim_config,
            self._collider_state,
        )

    def detection(self) -> None:
        # from genesis.utils.tools import create_timer

        self._contacts_info_cache = {}
        # timer = create_timer(name="69477ab0-5e75-47cb-a4a5-d4eebd9336ca", level=3, ti_sync=True, skip_first_call=True)
        rigid_solver.kernel_update_geom_aabbs(
            self._solver.geoms_state,
            self._solver.geoms_init_AABB,
            self._solver._static_rigid_sim_config,
        )
        # timer.stamp("func_update_aabbs")
        func_broad_phase(
            self._solver.links_state,
            self._solver.links_info,
            self._solver.geoms_state,
            self._solver.geoms_info,
            self._solver._rigid_global_info,
            self._solver._static_rigid_sim_config,
            self._collider_state,
            self._collider_info,
        )
        # timer.stamp("func_broad_phase")
        func_narrow_phase_convex_vs_convex(
            self._solver.links_state,
            self._solver.links_info,
            self._solver.geoms_state,
            self._solver.geoms_info,
            self._solver.geoms_init_AABB,
            self._solver.verts_info,
            self._solver.faces_info,
            self._solver._rigid_global_info,
            self._solver._static_rigid_sim_config,
            self._collider_state,
            self._collider_info,
            self._collider_static_config,
            self._mpr._mpr_state,
            self._mpr._mpr_static_config,
            self._gjk._gjk_state,
            self._gjk._gjk_static_config,
            self._sdf._sdf_info,
            self._support_field._support_field_info,
            self._support_field._support_field_static_config,
        )
        func_narrow_phase_convex_specializations(
            self._solver.geoms_state,
            self._solver.geoms_info,
            self._solver.geoms_init_AABB,
            self._solver.verts_info,
            self._solver._rigid_global_info,
            self._solver._static_rigid_sim_config,
            self._collider_state,
            self._collider_info,
            self._collider_static_config,
        )
        # timer.stamp("func_narrow_phase")
        if self._collider_static_config.has_terrain:
            func_narrow_phase_any_vs_terrain(
                self._solver.geoms_state,
                self._solver.geoms_info,
                self._solver.geoms_init_AABB,
                self._solver._rigid_global_info,
                self._solver._static_rigid_sim_config,
                self._collider_state,
                self._collider_info,
                self._collider_static_config,
                self._mpr._mpr_state,
                self._mpr._mpr_static_config,
                self._support_field._support_field_info,
                self._support_field._support_field_static_config,
            )
            # timer.stamp("func_narrow_phase_any_vs_terrain")
        if self._collider_static_config.has_nonconvex_nonterrain:
            func_narrow_phase_nonconvex_vs_nonterrain(
                self._solver.links_state,
                self._solver.links_info,
                self._solver.geoms_state,
                self._solver.geoms_info,
                self._solver.geoms_init_AABB,
                self._solver.verts_info,
                self._solver.edges_info,
                self._solver._rigid_global_info,
                self._solver._static_rigid_sim_config,
                self._collider_state,
                self._collider_info,
                self._collider_static_config,
                self._sdf._sdf_info,
            )
            # timer.stamp("func_narrow_phase_nonconvex_vs_nonterrain")

    def get_contacts(self, as_tensor: bool = True, to_torch: bool = True, keep_batch_dim: bool = False):
        # Early return if already pre-computed
        contacts_info = self._contacts_info_cache.get((as_tensor, to_torch))
        if contacts_info is not None:
            return contacts_info.copy()

        # Find out how much dynamic memory must be allocated
        n_contacts = tuple(self._collider_state.n_contacts.to_numpy())
        n_envs = len(n_contacts)
        n_contacts_max = max(n_contacts)
        if as_tensor:
            out_size = n_contacts_max * n_envs
        else:
            *n_contacts_starts, out_size = np.cumsum(n_contacts)

        # Allocate output buffer
        if to_torch:
            iout = torch.full((out_size, 4), -1, dtype=gs.tc_int, device=gs.device)
            fout = torch.zeros((out_size, 10), dtype=gs.tc_float, device=gs.device)
        else:
            iout = np.full((out_size, 4), -1, dtype=gs.np_int)
            fout = np.zeros((out_size, 10), dtype=gs.np_float)

        # Copy contact data
        if n_contacts_max > 0:
            collider_kernel_get_contacts(
                as_tensor,
                iout,
                fout,
                self._solver._rigid_global_info,
                self._solver._static_rigid_sim_config,
                self._collider_state,
                self._collider_info,
            )

        # Build structured view (no copy)
        if as_tensor:
            if self._solver.n_envs > 0:
                iout = iout.reshape((n_envs, n_contacts_max, 4))
                fout = fout.reshape((n_envs, n_contacts_max, 10))
            if keep_batch_dim and self._solver.n_envs == 0:
                iout = iout.reshape((1, n_contacts_max, 4))
                fout = fout.reshape((1, n_contacts_max, 10))
            iout_chunks = (iout[..., 0], iout[..., 1], iout[..., 2], iout[..., 3])
            fout_chunks = (fout[..., 0], fout[..., 1:4], fout[..., 4:7], fout[..., 7:])
            values = (*iout_chunks, *fout_chunks)
        else:
            # Split smallest dimension first, then largest dimension
            if self._solver.n_envs == 0:
                iout_chunks = (iout[..., 0], iout[..., 1], iout[..., 2], iout[..., 3])
                fout_chunks = (fout[..., 0], fout[..., 1:4], fout[..., 4:7], fout[..., 7:])
                values = (*iout_chunks, *fout_chunks)
            elif n_contacts_max >= n_envs:
                if to_torch:
                    iout_chunks = torch.split(iout, n_contacts)
                    fout_chunks = torch.split(fout, n_contacts)
                else:
                    iout_chunks = np.split(iout, n_contacts_starts)
                    fout_chunks = np.split(fout, n_contacts_starts)
                iout_chunks = ((out[..., 0], out[..., 1], out[..., 2], out[..., 3]) for out in iout_chunks)
                fout_chunks = ((out[..., 0], out[..., 1:4], out[..., 4:7], out[..., 7:]) for out in fout_chunks)
                values = (*zip(*iout_chunks), *zip(*fout_chunks))
            else:
                iout_chunks = (iout[..., 0], iout[..., 1], iout[..., 2], iout[..., 3])
                fout_chunks = (fout[..., 0], fout[..., 1:4], fout[..., 4:7], fout[..., 7:])
                if self._solver.n_envs == 1:
                    values = [(value,) for value in (*iout_chunks, *fout_chunks)]
                else:
                    if to_torch:
                        iout_chunks = (torch.split(out, n_contacts) for out in iout_chunks)
                        fout_chunks = (torch.split(out, n_contacts) for out in fout_chunks)
                    else:
                        iout_chunks = (np.split(out, n_contacts_starts) for out in iout_chunks)
                        fout_chunks = (np.split(out, n_contacts_starts) for out in fout_chunks)
                    values = (*iout_chunks, *fout_chunks)

        contacts_info = dict(
            zip(("link_a", "link_b", "geom_a", "geom_b", "penetration", "position", "normal", "force"), values)
        )

        # Cache contact information before returning
        self._contacts_info_cache[(as_tensor, to_torch)] = contacts_info

        return contacts_info.copy()


@ti.func
def rotaxis(vecin, i0, i1, i2, f0, f1, f2):
    vecres = ti.Vector([0.0, 0.0, 0.0], dt=gs.ti_float)
    vecres[0] = vecin[i0] * f0
    vecres[1] = vecin[i1] * f1
    vecres[2] = vecin[i2] * f2
    return vecres


@ti.func
def rotmatx(matin, i0, i1, i2, f0, f1, f2):
    matres = ti.Matrix.zero(gs.ti_float, 3, 3)
    matres[0, :] = matin[i0, :] * f0
    matres[1, :] = matin[i1, :] * f1
    matres[2, :] = matin[i2, :] * f2
    return matres


@ti.kernel
def collider_kernel_reset(
    envs_idx: ti.types.ndarray(),
    static_rigid_sim_config: ti.template(),
    collider_state: array_class.ColliderState,
):
    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
    for i_b_ in range(envs_idx.shape[0]):
        i_b = envs_idx[i_b_]
        collider_state.first_time[i_b] = 1
        n_geoms = collider_state.active_buffer.shape[0]
        for i_ga in range(n_geoms):
            for i_gb in range(n_geoms):
                # self.contact_cache[i_ga, i_gb, i_b].i_va_ws = -1
                # self.contact_cache[i_ga, i_gb, i_b].penetration = 0.0
                collider_state.contact_cache.normal[i_ga, i_gb, i_b] = ti.Vector.zero(gs.ti_float, 3)


# only used with hibernation ??
@ti.kernel
def kernel_collider_clear(
    envs_idx: ti.types.ndarray(),
    links_state: array_class.LinksState,
    links_info: array_class.LinksInfo,
    static_rigid_sim_config: ti.template(),
    collider_state: array_class.ColliderState,
):
    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
    for i_b_ in range(envs_idx.shape[0]):
        i_b = envs_idx[i_b_]

        if ti.static(static_rigid_sim_config.use_hibernation):
            collider_state.n_contacts_hibernated[i_b] = 0

            # advect hibernated contacts
            for i_c in range(collider_state.n_contacts[i_b]):
                i_la = collider_state.contact_data.link_a[i_c, i_b]
                i_lb = collider_state.contact_data.link_b[i_c, i_b]

                I_la = [i_la, i_b] if ti.static(static_rigid_sim_config.batch_links_info) else i_la
                I_lb = [i_lb, i_b] if ti.static(static_rigid_sim_config.batch_links_info) else i_lb

                # pair of hibernated-fixed links -> hibernated contact
                # TODO: we should also include hibernated-hibernated links and wake up the whole contact island
                # once a new collision is detected
                if (links_state.hibernated[i_la, i_b] and links_info.is_fixed[I_lb]) or (
                    links_state.hibernated[i_lb, i_b] and links_info.is_fixed[I_la]
                ):
                    i_c_hibernated = collider_state.n_contacts_hibernated[i_b]
                    if i_c != i_c_hibernated:
                        # Copying all fields of class StructContactData:
                        # fmt: off
                        collider_state.contact_data.geom_a[i_c_hibernated, i_b] = collider_state.contact_data.geom_a[i_c, i_b]
                        collider_state.contact_data.geom_b[i_c_hibernated, i_b] = collider_state.contact_data.geom_b[i_c, i_b]
                        collider_state.contact_data.penetration[i_c_hibernated, i_b] = collider_state.contact_data.penetration[i_c, i_b]
                        collider_state.contact_data.normal[i_c_hibernated, i_b] = collider_state.contact_data.normal[i_c, i_b]
                        collider_state.contact_data.pos[i_c_hibernated, i_b] = collider_state.contact_data.pos[i_c, i_b]
                        collider_state.contact_data.friction[i_c_hibernated, i_b] = collider_state.contact_data.friction[i_c, i_b]
                        collider_state.contact_data.sol_params[i_c_hibernated, i_b] = collider_state.contact_data.sol_params[i_c, i_b]
                        collider_state.contact_data.force[i_c_hibernated, i_b] = collider_state.contact_data.force[i_c, i_b]
                        collider_state.contact_data.link_a[i_c_hibernated, i_b] = collider_state.contact_data.link_a[i_c, i_b]
                        collider_state.contact_data.link_b[i_c_hibernated, i_b] = collider_state.contact_data.link_b[i_c, i_b]
                        # fmt: on

                    collider_state.n_contacts_hibernated[i_b] = i_c_hibernated + 1

            collider_state.n_contacts[i_b] = collider_state.n_contacts_hibernated[i_b]
        else:
            collider_state.n_contacts[i_b] = 0


@ti.kernel
def collider_kernel_get_contacts(
    is_padded: ti.template(),
    iout: ti.types.ndarray(),
    fout: ti.types.ndarray(),
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: ti.template(),
    collider_state: array_class.ColliderState,
    collider_info: array_class.ColliderInfo,
):
    _B = collider_state.active_buffer.shape[1]
    n_contacts_max = gs.ti_int(0)

    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
    for i_b in range(_B):
        n_contacts = collider_state.n_contacts[i_b]
        if n_contacts > n_contacts_max:
            n_contacts_max = n_contacts

    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
    for i_b in range(_B):
        i_c_start = gs.ti_int(0)
        if ti.static(is_padded):
            i_c_start = i_b * n_contacts_max
        else:
            for j_b in range(i_b):
                i_c_start = i_c_start + collider_state.n_contacts[j_b]

        for i_c_ in range(collider_state.n_contacts[i_b]):
            i_c = i_c_start + i_c_

            iout[i_c, 0] = collider_state.contact_data.link_a[i_c_, i_b]
            iout[i_c, 1] = collider_state.contact_data.link_b[i_c_, i_b]
            iout[i_c, 2] = collider_state.contact_data.geom_a[i_c_, i_b]
            iout[i_c, 3] = collider_state.contact_data.geom_b[i_c_, i_b]
            fout[i_c, 0] = collider_state.contact_data.penetration[i_c_, i_b]
            for j in ti.static(range(3)):
                fout[i_c, 1 + j] = collider_state.contact_data.pos[i_c_, i_b][j]
                fout[i_c, 4 + j] = collider_state.contact_data.normal[i_c_, i_b][j]
                fout[i_c, 7 + j] = collider_state.contact_data.force[i_c_, i_b][j]


@ti.func
def func_point_in_geom_aabb(
    i_g,
    i_b,
    geoms_state: array_class.GeomsState,
    geoms_info: array_class.GeomsInfo,
    point: ti.types.vector(3),
):
    return (point < geoms_state.aabb_max[i_g, i_b]).all() and (point > geoms_state.aabb_min[i_g, i_b]).all()


@ti.func
def func_is_geom_aabbs_overlap(
    i_ga,
    i_gb,
    i_b,
    geoms_state: array_class.GeomsState,
    geoms_info: array_class.GeomsInfo,
):
    return not (
        (geoms_state.aabb_max[i_ga, i_b] <= geoms_state.aabb_min[i_gb, i_b]).any()
        or (geoms_state.aabb_min[i_ga, i_b] >= geoms_state.aabb_max[i_gb, i_b]).any()
    )


@ti.func
def func_find_intersect_midpoint(
    i_ga,
    i_gb,
    i_b,
    geoms_state: array_class.GeomsState,
    geoms_info: array_class.GeomsInfo,
):
    # return the center of the intersecting AABB of AABBs of two geoms
    intersect_lower = ti.max(geoms_state.aabb_min[i_ga, i_b], geoms_state.aabb_min[i_gb, i_b])
    intersect_upper = ti.min(geoms_state.aabb_max[i_ga, i_b], geoms_state.aabb_max[i_gb, i_b])
    return 0.5 * (intersect_lower + intersect_upper)


@ti.func
def func_contact_sphere_sdf(
    i_ga,
    i_gb,
    i_b,
    geoms_state: array_class.GeomsState,
    geoms_info: array_class.GeomsInfo,
    collider_static_config: ti.template(),
    sdf_info: array_class.SDFInfo,
):
    is_col = False
    penetration = gs.ti_float(0.0)
    normal = ti.Vector.zero(gs.ti_float, 3)
    contact_pos = ti.Vector.zero(gs.ti_float, 3)

    sphere_center = geoms_state.pos[i_ga, i_b]
    sphere_radius = geoms_info.data[i_ga][0]

    center_to_b_dist = sdf.sdf_func_world(geoms_state, geoms_info, sdf_info, sphere_center, i_gb, i_b)
    if center_to_b_dist < sphere_radius:
        is_col = True
        normal = sdf.sdf_func_normal_world(
            geoms_state, geoms_info, collider_static_config, sdf_info, sphere_center, i_gb, i_b
        )
        penetration = sphere_radius - center_to_b_dist
        contact_pos = sphere_center - (sphere_radius - 0.5 * penetration) * normal

    return is_col, normal, penetration, contact_pos


@ti.func
def func_contact_vertex_sdf(
    i_ga,
    i_gb,
    i_b,
    geoms_state: array_class.GeomsState,
    geoms_info: array_class.GeomsInfo,
    verts_info: array_class.VertsInfo,
    collider_static_config: ti.template(),
    sdf_info: array_class.SDFInfo,
):
    ga_pos = geoms_state.pos[i_ga, i_b]
    ga_quat = geoms_state.quat[i_ga, i_b]

    is_col = False
    penetration = gs.ti_float(0.0)
    normal = ti.Vector.zero(gs.ti_float, 3)
    contact_pos = ti.Vector.zero(gs.ti_float, 3)

    for i_v in range(geoms_info.vert_start[i_ga], geoms_info.vert_end[i_ga]):
        vertex_pos = gu.ti_transform_by_trans_quat(verts_info.init_pos[i_v], ga_pos, ga_quat)
        if func_point_in_geom_aabb(i_gb, i_b, geoms_state, geoms_info, vertex_pos):
            new_penetration = -sdf.sdf_func_world(geoms_state, geoms_info, sdf_info, vertex_pos, i_gb, i_b)
            if new_penetration > penetration:
                is_col = True
                contact_pos = vertex_pos
                penetration = new_penetration

    # Compute contact normal only once, and only in case of contact
    if is_col:
        normal = sdf.sdf_func_normal_world(
            geoms_state, geoms_info, collider_static_config, sdf_info, contact_pos, i_gb, i_b
        )

    # The contact point must be offsetted by half the penetration depth
    contact_pos += 0.5 * penetration * normal

    return is_col, normal, penetration, contact_pos


@ti.func
def func_contact_edge_sdf(
    i_ga,
    i_gb,
    i_b,
    geoms_state: array_class.GeomsState,
    geoms_info: array_class.GeomsInfo,
    verts_info: array_class.VertsInfo,
    edges_info: array_class.EdgesInfo,
    collider_static_config: ti.template(),
    sdf_info: array_class.SDFInfo,
):
    is_col = False
    penetration = gs.ti_float(0.0)
    normal = ti.Vector.zero(gs.ti_float, 3)
    contact_pos = ti.Vector.zero(gs.ti_float, 3)

    ga_sdf_cell_size = sdf_info.geoms_info.sdf_cell_size[i_ga]

    for i_e in range(geoms_info.edge_start[i_ga], geoms_info.edge_end[i_ga]):
        cur_length = edges_info.length[i_e]
        if cur_length > ga_sdf_cell_size:

            i_v0 = edges_info.v0[i_e]
            i_v1 = edges_info.v1[i_e]

            p_0 = gu.ti_transform_by_trans_quat(
                verts_info.init_pos[i_v0], geoms_state.pos[i_ga, i_b], geoms_state.quat[i_ga, i_b]
            )
            p_1 = gu.ti_transform_by_trans_quat(
                verts_info.init_pos[i_v1], geoms_state.pos[i_ga, i_b], geoms_state.quat[i_ga, i_b]
            )
            vec_01 = gu.ti_normalize(p_1 - p_0)

            sdf_grad_0_b = sdf.sdf_func_grad_world(
                geoms_state, geoms_info, collider_static_config, sdf_info, p_0, i_gb, i_b
            )
            sdf_grad_1_b = sdf.sdf_func_grad_world(
                geoms_state, geoms_info, collider_static_config, sdf_info, p_1, i_gb, i_b
            )

            # check if the edge on a is facing towards mesh b (I am not 100% sure about this, subject to removal)
            sdf_grad_0_a = sdf.sdf_func_grad_world(
                geoms_state, geoms_info, collider_static_config, sdf_info, p_0, i_ga, i_b
            )
            sdf_grad_1_a = sdf.sdf_func_grad_world(
                geoms_state, geoms_info, collider_static_config, sdf_info, p_1, i_ga, i_b
            )
            normal_edge_0 = sdf_grad_0_a - sdf_grad_0_a.dot(vec_01) * vec_01
            normal_edge_1 = sdf_grad_1_a - sdf_grad_1_a.dot(vec_01) * vec_01

            if normal_edge_0.dot(sdf_grad_0_b) < 0 or normal_edge_1.dot(sdf_grad_1_b) < 0:

                # check if closest point is between the two points
                if sdf_grad_0_b.dot(vec_01) < 0 and sdf_grad_1_b.dot(vec_01) > 0:

                    while cur_length > ga_sdf_cell_size:
                        p_mid = 0.5 * (p_0 + p_1)
                        if (
                            sdf.sdf_func_grad_world(
                                geoms_state, geoms_info, collider_static_config, sdf_info, p_mid, i_gb, i_b
                            ).dot(vec_01)
                            < 0
                        ):
                            p_0 = p_mid
                        else:
                            p_1 = p_mid
                        cur_length = 0.5 * cur_length

                    p = 0.5 * (p_0 + p_1)
                    new_penetration = -sdf.sdf_func_world(geoms_state, geoms_info, sdf_info, p, i_gb, i_b)

                    if new_penetration > penetration:
                        is_col = True
                        normal = sdf.sdf_func_normal_world(
                            geoms_state, geoms_info, collider_static_config, sdf_info, p, i_gb, i_b
                        )
                        contact_pos = p
                        penetration = new_penetration

    # The contact point must be offsetted by half the penetration depth, for consistency with MPR
    contact_pos += 0.5 * penetration * normal

    return is_col, normal, penetration, contact_pos


@ti.func
def func_contact_convex_convex_sdf(
    i_ga,
    i_gb,
    i_b,
    i_va_ws,
    geoms_state: array_class.GeomsState,
    geoms_info: array_class.GeomsInfo,
    verts_info: array_class.VertsInfo,
    collider_info: array_class.ColliderInfo,
    collider_static_config: ti.template(),
    sdf_info: array_class.SDFInfo,
):
    gb_vert_start = geoms_info.vert_start[i_gb]
    ga_pos = geoms_state.pos[i_ga, i_b]
    ga_quat = geoms_state.quat[i_ga, i_b]
    gb_pos = geoms_state.pos[i_gb, i_b]
    gb_quat = geoms_state.quat[i_gb, i_b]

    is_col = False
    penetration = gs.ti_float(0.0)
    normal = ti.Vector.zero(gs.ti_float, 3)
    contact_pos = ti.Vector.zero(gs.ti_float, 3)

    i_va = i_va_ws
    if i_va == -1:
        # start traversing on the vertex graph with a smart initial vertex
        pos_vb = gu.ti_transform_by_trans_quat(verts_info.init_pos[gb_vert_start], gb_pos, gb_quat)
        i_va = sdf.sdf_func_find_closest_vert(geoms_state, geoms_info, sdf_info, pos_vb, i_ga, i_b)
    i_v_closest = i_va
    pos_v_closest = gu.ti_transform_by_trans_quat(verts_info.init_pos[i_v_closest], ga_pos, ga_quat)
    sd_v_closest = sdf.sdf_func_world(geoms_state, geoms_info, sdf_info, pos_v_closest, i_gb, i_b)

    while True:
        for i_neighbor_ in range(
            collider_info.vert_neighbor_start[i_va],
            collider_info.vert_neighbor_start[i_va] + collider_info.vert_n_neighbors[i_va],
        ):
            i_neighbor = collider_info.vert_neighbors[i_neighbor_]
            pos_neighbor = gu.ti_transform_by_trans_quat(verts_info.init_pos[i_neighbor], ga_pos, ga_quat)
            sd_neighbor = sdf.sdf_func_world(geoms_state, geoms_info, sdf_info, pos_neighbor, i_gb, i_b)
            if sd_neighbor < sd_v_closest - 1e-5:  # 1e-5 (0.01mm) to avoid endless loop due to numerical instability
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
        is_col = True
        normal = sdf.sdf_func_normal_world(geoms_state, geoms_info, collider_static_config, sdf_info, pos_a, i_gb, i_b)
        contact_pos = pos_a
        penetration = -sd_v_closest

    else:  # check edge surrounding it
        for i_neighbor_ in range(
            collider_info.vert_neighbor_start[i_va],
            collider_info.vert_neighbor_start[i_va] + collider_info.vert_n_neighbors[i_va],
        ):
            i_neighbor = collider_info.vert_neighbors[i_neighbor_]

            p_0 = pos_v_closest
            p_1 = gu.ti_transform_by_trans_quat(verts_info.init_pos[i_neighbor], ga_pos, ga_quat)
            vec_01 = gu.ti_normalize(p_1 - p_0)

            sdf_grad_0_b = sdf.sdf_func_grad_world(
                geoms_state, geoms_info, collider_static_config, sdf_info, p_0, i_gb, i_b
            )
            sdf_grad_1_b = sdf.sdf_func_grad_world(
                geoms_state, geoms_info, collider_static_config, sdf_info, p_1, i_gb, i_b
            )

            # check if the edge on a is facing towards mesh b (I am not 100% sure about this, subject to removal)
            sdf_grad_0_a = sdf.sdf_func_grad_world(
                geoms_state, geoms_info, collider_static_config, sdf_info, p_0, i_ga, i_b
            )
            sdf_grad_1_a = sdf.sdf_func_grad_world(
                geoms_state, geoms_info, collider_static_config, sdf_info, p_1, i_ga, i_b
            )
            normal_edge_0 = sdf_grad_0_a - sdf_grad_0_a.dot(vec_01) * vec_01
            normal_edge_1 = sdf_grad_1_a - sdf_grad_1_a.dot(vec_01) * vec_01

            if normal_edge_0.dot(sdf_grad_0_b) < 0 or normal_edge_1.dot(sdf_grad_1_b) < 0:
                # check if closest point is between the two points
                if sdf_grad_0_b.dot(vec_01) < 0 and sdf_grad_1_b.dot(vec_01) > 0:
                    cur_length = (p_1 - p_0).norm()
                    ga_sdf_cell_size = sdf_info.geoms_info.sdf_cell_size[i_ga]
                    while cur_length > ga_sdf_cell_size:
                        p_mid = 0.5 * (p_0 + p_1)
                        if (
                            sdf.sdf_func_grad_world(
                                geoms_state, geoms_info, collider_static_config, sdf_info, p_mid, i_gb, i_b
                            ).dot(vec_01)
                            < 0
                        ):
                            p_0 = p_mid
                        else:
                            p_1 = p_mid

                        cur_length = 0.5 * cur_length

                    p = 0.5 * (p_0 + p_1)

                    new_penetration = -sdf.sdf_func_world(geoms_state, geoms_info, sdf_info, p, i_gb, i_b)

                    if new_penetration > 0:
                        is_col = True
                        normal = sdf.sdf_func_normal_world(
                            geoms_state, geoms_info, collider_static_config, sdf_info, p, i_gb, i_b
                        )
                        contact_pos = p
                        penetration = new_penetration
                        break

    return is_col, normal, penetration, contact_pos, i_va


@ti.func
def func_contact_mpr_terrain(
    i_ga,
    i_gb,
    i_b,
    geoms_state: array_class.GeomsState,
    geoms_info: array_class.GeomsInfo,
    geoms_init_AABB: array_class.GeomsInitAABB,
    static_rigid_sim_config: ti.template(),
    collider_state: array_class.ColliderState,
    collider_info: array_class.ColliderInfo,
    collider_static_config: ti.template(),
    mpr_state: array_class.MPRState,
    mpr_static_config: ti.template(),
    support_field_info: array_class.SupportFieldInfo,
    support_field_static_config: ti.template(),
):
    ga_pos, ga_quat = geoms_state.pos[i_ga, i_b], geoms_state.quat[i_ga, i_b]
    gb_pos, gb_quat = geoms_state.pos[i_gb, i_b], geoms_state.quat[i_gb, i_b]
    margin = gs.ti_float(0.0)

    is_return = False
    tolerance = func_compute_tolerance(i_ga, i_gb, i_b, geoms_info, geoms_init_AABB, collider_static_config)
    # pos = self._solver.geoms_state[i_ga, i_b].pos - self._solver.geoms_state[i_gb, i_b].pos
    # for i in range(3):
    #     if self._solver.terrain_xyz_maxmin[i] < pos[i] - r2 - margin or \
    #         self._solver.terrain_xyz_maxmin[i+3] > pos[i] + r2 + margin:
    #         is_return = True

    if not is_return:
        geoms_state.pos[i_ga, i_b], geoms_state.quat[i_ga, i_b] = gu.ti_transform_pos_quat_by_trans_quat(
            ga_pos - geoms_state.pos[i_gb, i_b],
            ga_quat,
            ti.Vector.zero(gs.ti_float, 3),
            gu.ti_inv_quat(geoms_state.quat[i_gb, i_b]),
        )

        for i_axis, i_m in ti.ndrange(3, 2):
            direction = ti.Vector.zero(gs.ti_float, 3)
            if i_m == 0:
                direction[i_axis] = 1.0
            else:
                direction[i_axis] = -1.0
            v1 = mpr.support_driver(
                geoms_state,
                geoms_info,
                collider_state,
                collider_info,
                collider_static_config,
                support_field_info,
                support_field_static_config,
                direction,
                i_ga,
                i_b,
            )
            collider_state.xyz_max_min[3 * i_m + i_axis, i_b] = v1[i_axis]

        for i in ti.static(range(3)):
            collider_state.prism[i, i_b][2] = collider_info.terrain_xyz_maxmin[5]

            if (
                collider_info.terrain_xyz_maxmin[i] < collider_state.xyz_max_min[i + 3, i_b] - margin
                or collider_info.terrain_xyz_maxmin[i + 3] > collider_state.xyz_max_min[i, i_b] + margin
            ):
                is_return = True

        if not is_return:
            sh = collider_info.terrain_scale[0]
            r_min = gs.ti_int(ti.floor((collider_state.xyz_max_min[3, i_b] - collider_info.terrain_xyz_maxmin[3]) / sh))
            r_max = gs.ti_int(ti.ceil((collider_state.xyz_max_min[0, i_b] - collider_info.terrain_xyz_maxmin[3]) / sh))
            c_min = gs.ti_int(ti.floor((collider_state.xyz_max_min[4, i_b] - collider_info.terrain_xyz_maxmin[4]) / sh))
            c_max = gs.ti_int(ti.ceil((collider_state.xyz_max_min[1, i_b] - collider_info.terrain_xyz_maxmin[4]) / sh))

            r_min = ti.max(0, r_min)
            c_min = ti.max(0, c_min)
            r_max = ti.min(collider_info.terrain_rc[0] - 1, r_max)
            c_max = ti.min(collider_info.terrain_rc[1] - 1, c_max)

            n_con = 0
            for r in range(r_min, r_max):
                nvert = 0
                for c in range(c_min, c_max + 1):
                    for i in range(2):
                        if n_con < ti.static(collider_static_config.n_contacts_per_pair):
                            nvert = nvert + 1
                            func_add_prism_vert(
                                sh * (r + i) + collider_info.terrain_xyz_maxmin[3],
                                sh * c + collider_info.terrain_xyz_maxmin[4],
                                collider_info.terrain_hf[r + i, c] + margin,
                                i_b,
                                collider_state,
                            )
                            if nvert > 2 and (
                                collider_state.prism[3, i_b][2] >= collider_state.xyz_max_min[5, i_b]
                                or collider_state.prism[4, i_b][2] >= collider_state.xyz_max_min[5, i_b]
                                or collider_state.prism[5, i_b][2] >= collider_state.xyz_max_min[5, i_b]
                            ):
                                center_a = gu.ti_transform_by_trans_quat(geoms_info.center[i_ga], ga_pos, ga_quat)
                                center_b = ti.Vector.zero(gs.ti_float, 3)
                                for i_p in ti.static(range(6)):
                                    center_b = center_b + collider_state.prism[i_p, i_b]
                                center_b = center_b / 6.0

                                geoms_state.pos[i_gb, i_b] = ti.Vector.zero(gs.ti_float, 3)
                                geoms_state.quat[i_gb, i_b] = gu.ti_identity_quat()

                                is_col, normal, penetration, contact_pos = mpr.func_mpr_contact_from_centers(
                                    geoms_state,
                                    geoms_info,
                                    static_rigid_sim_config,
                                    collider_state,
                                    collider_info,
                                    collider_static_config,
                                    mpr_state,
                                    mpr_static_config,
                                    support_field_info,
                                    support_field_static_config,
                                    i_ga,
                                    i_gb,
                                    i_b,
                                    center_a,
                                    center_b,
                                )
                                if is_col:
                                    normal = gu.ti_transform_by_quat(normal, gb_quat)
                                    contact_pos = gu.ti_transform_by_quat(contact_pos, gb_quat)
                                    contact_pos = contact_pos + gb_pos

                                    valid = True
                                    i_c = collider_state.n_contacts[i_b]
                                    for j in range(n_con):
                                        if (
                                            contact_pos - collider_state.contact_data.pos[i_c - j - 1, i_b]
                                        ).norm() < tolerance:
                                            valid = False
                                            break

                                    if valid:
                                        func_add_contact(
                                            i_ga,
                                            i_gb,
                                            normal,
                                            contact_pos,
                                            penetration,
                                            i_b,
                                            geoms_state,
                                            geoms_info,
                                            collider_state,
                                            collider_info,
                                        )
                                        n_con = n_con + 1

    geoms_state.pos[i_ga, i_b], geoms_state.quat[i_ga, i_b] = ga_pos, ga_quat
    geoms_state.pos[i_gb, i_b], geoms_state.quat[i_gb, i_b] = gb_pos, gb_quat


@ti.func
def func_add_prism_vert(
    x,
    y,
    z,
    i_b,
    collider_state: array_class.ColliderState,
):
    collider_state.prism[0, i_b] = collider_state.prism[1, i_b]
    collider_state.prism[1, i_b] = collider_state.prism[2, i_b]
    collider_state.prism[3, i_b] = collider_state.prism[4, i_b]
    collider_state.prism[4, i_b] = collider_state.prism[5, i_b]

    collider_state.prism[2, i_b][0] = x
    collider_state.prism[5, i_b][0] = x
    collider_state.prism[2, i_b][1] = y
    collider_state.prism[5, i_b][1] = y
    collider_state.prism[5, i_b][2] = z


@ti.func
def func_check_collision_valid(
    i_ga,
    i_gb,
    i_b,
    links_state: array_class.LinksState,
    links_info: array_class.LinksInfo,
    geoms_info: array_class.GeomsInfo,
    static_rigid_sim_config: ti.template(),
    collider_info: array_class.ColliderInfo,
):
    is_valid = collider_info.collision_pair_validity[i_ga, i_gb]

    # hibernated <-> fixed links
    if ti.static(static_rigid_sim_config.use_hibernation):
        i_la = geoms_info.link_idx[i_ga]
        i_lb = geoms_info.link_idx[i_gb]
        I_la = [i_la, i_b] if ti.static(static_rigid_sim_config.batch_links_info) else i_la
        I_lb = [i_lb, i_b] if ti.static(static_rigid_sim_config.batch_links_info) else i_lb

        if (links_state.hibernated[i_la, i_b] and links_info.is_fixed[I_lb]) or (
            links_state.hibernated[i_lb, i_b] and links_info.is_fixed[I_la]
        ):
            is_valid = False

    return is_valid


@ti.kernel
def func_broad_phase(
    links_state: array_class.LinksState,
    links_info: array_class.LinksInfo,
    geoms_state: array_class.GeomsState,
    geoms_info: array_class.GeomsInfo,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: ti.template(),
    # we will use ColliderBroadPhaseBuffer as typing after Hugh adds array_struct feature to gstaichi
    collider_state: array_class.ColliderState,
    collider_info: array_class.ColliderInfo,
):
    """
    Sweep and Prune (SAP) for broad-phase collision detection.

    This function sorts the geometry axis-aligned bounding boxes (AABBs) along a specified axis and checks for
    potential collision pairs based on the AABB overlap.
    """
    _B = collider_state.active_buffer.shape[1]
    n_geoms = collider_state.active_buffer.shape[0]

    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
    for i_b in range(_B):
        axis = 0

        # copy updated geom aabbs to buffer for sorting
        if collider_state.first_time[i_b]:
            for i in range(n_geoms):
                collider_state.sort_buffer.value[2 * i, i_b] = geoms_state.aabb_min[i, i_b][axis]
                collider_state.sort_buffer.i_g[2 * i, i_b] = i
                collider_state.sort_buffer.is_max[2 * i, i_b] = 0

                collider_state.sort_buffer.value[2 * i + 1, i_b] = geoms_state.aabb_max[i, i_b][axis]
                collider_state.sort_buffer.i_g[2 * i + 1, i_b] = i
                collider_state.sort_buffer.is_max[2 * i + 1, i_b] = 1

                geoms_state.min_buffer_idx[i, i_b] = 2 * i
                geoms_state.max_buffer_idx[i, i_b] = 2 * i + 1

            collider_state.first_time[i_b] = False

        else:
            # warm start. If `use_hibernation=True`, it's already updated in rigid_solver.
            if ti.static(not static_rigid_sim_config.use_hibernation):
                for i in range(n_geoms * 2):
                    if collider_state.sort_buffer.is_max[i, i_b]:
                        collider_state.sort_buffer.value[i, i_b] = geoms_state.aabb_max[
                            collider_state.sort_buffer.i_g[i, i_b], i_b
                        ][axis]
                    else:
                        collider_state.sort_buffer.value[i, i_b] = geoms_state.aabb_min[
                            collider_state.sort_buffer.i_g[i, i_b], i_b
                        ][axis]

        # insertion sort, which has complexity near O(n) for nearly sorted array
        for i in range(1, 2 * n_geoms):
            key_value = collider_state.sort_buffer.value[i, i_b]
            key_is_max = collider_state.sort_buffer.is_max[i, i_b]
            key_i_g = collider_state.sort_buffer.i_g[i, i_b]

            j = i - 1
            while j >= 0 and key_value < collider_state.sort_buffer.value[j, i_b]:
                collider_state.sort_buffer.value[j + 1, i_b] = collider_state.sort_buffer.value[j, i_b]
                collider_state.sort_buffer.is_max[j + 1, i_b] = collider_state.sort_buffer.is_max[j, i_b]
                collider_state.sort_buffer.i_g[j + 1, i_b] = collider_state.sort_buffer.i_g[j, i_b]

                if ti.static(static_rigid_sim_config.use_hibernation):
                    if collider_state.sort_buffer.is_max[j, i_b]:
                        geoms_state.max_buffer_idx[collider_state.sort_buffer.i_g[j, i_b], i_b] = j + 1
                    else:
                        geoms_state.min_buffer_idx[collider_state.sort_buffer.i_g[j, i_b], i_b] = j + 1

                j -= 1
            collider_state.sort_buffer.value[j + 1, i_b] = key_value
            collider_state.sort_buffer.is_max[j + 1, i_b] = key_is_max
            collider_state.sort_buffer.i_g[j + 1, i_b] = key_i_g

            if ti.static(static_rigid_sim_config.use_hibernation):
                if key_is_max:
                    geoms_state.max_buffer_idx[key_i_g, i_b] = j + 1
                else:
                    geoms_state.min_buffer_idx[key_i_g, i_b] = j + 1

        # sweep over the sorted AABBs to find potential collision pairs
        collider_state.n_broad_pairs[i_b] = 0
        if ti.static(not static_rigid_sim_config.use_hibernation):
            n_active = 0
            for i in range(2 * n_geoms):
                if not collider_state.sort_buffer.is_max[i, i_b]:
                    for j in range(n_active):
                        i_ga = collider_state.active_buffer[j, i_b]
                        i_gb = collider_state.sort_buffer.i_g[i, i_b]
                        if i_ga > i_gb:
                            i_ga, i_gb = i_gb, i_ga

                        if not func_check_collision_valid(
                            i_ga,
                            i_gb,
                            i_b,
                            links_state,
                            links_info,
                            geoms_info,
                            static_rigid_sim_config,
                            collider_info,
                        ):
                            continue

                        if not func_is_geom_aabbs_overlap(i_ga, i_gb, i_b, geoms_state, geoms_info):
                            # Clear collision normal cache if not in contact
                            if ti.static(not static_rigid_sim_config.enable_mujoco_compatibility):
                                # self.contact_cache[i_ga, i_gb, i_b].i_va_ws = -1
                                collider_state.contact_cache.normal[i_ga, i_gb, i_b] = ti.Vector.zero(gs.ti_float, 3)
                            continue

                        i_p = collider_state.n_broad_pairs[i_b]
                        if i_p == collider_info._max_collision_pairs_broad[None]:
                            # print(self._warn_msg_max_collision_pairs_broad)
                            break
                        collider_state.broad_collision_pairs[i_p, i_b][0] = i_ga
                        collider_state.broad_collision_pairs[i_p, i_b][1] = i_gb
                        collider_state.n_broad_pairs[i_b] += 1

                    collider_state.active_buffer[n_active, i_b] = collider_state.sort_buffer.i_g[i, i_b]
                    n_active = n_active + 1
                else:
                    i_g_to_remove = collider_state.sort_buffer.i_g[i, i_b]
                    for j in range(n_active):
                        if collider_state.active_buffer[j, i_b] == i_g_to_remove:
                            if j < n_active - 1:
                                for k in range(j, n_active - 1):
                                    collider_state.active_buffer[k, i_b] = collider_state.active_buffer[k + 1, i_b]
                            n_active = n_active - 1
                            break
        else:
            if rigid_global_info.n_awake_dofs[i_b] > 0:
                n_active_awake = 0
                n_active_hib = 0
                for i in range(2 * n_geoms):
                    is_incoming_geom_hibernated = geoms_state.hibernated[collider_state.sort_buffer.i_g[i, i_b], i_b]

                    if not collider_state.sort_buffer.is_max[i, i_b]:
                        # both awake and hibernated geom check with active awake geoms
                        for j in range(n_active_awake):
                            i_ga = collider_state.active_buffer_awake[j, i_b]
                            i_gb = collider_state.sort_buffer.i_g[i, i_b]
                            if i_ga > i_gb:
                                i_ga, i_gb = i_gb, i_ga

                            if not func_check_collision_valid(
                                i_ga,
                                i_gb,
                                i_b,
                                links_state,
                                links_info,
                                geoms_info,
                                static_rigid_sim_config,
                                collider_info,
                            ):
                                continue

                            if not func_is_geom_aabbs_overlap(i_ga, i_gb, i_b, geoms_state, geoms_info):
                                # Clear collision normal cache if not in contact
                                if ti.static(not static_rigid_sim_config.enable_mujoco_compatibility):
                                    # self.contact_cache[i_ga, i_gb, i_b].i_va_ws = -1
                                    collider_state.contact_cache.normal[i_ga, i_gb, i_b] = ti.Vector.zero(
                                        gs.ti_float, 3
                                    )
                                continue

                            collider_state.broad_collision_pairs[collider_state.n_broad_pairs[i_b], i_b][0] = i_ga
                            collider_state.broad_collision_pairs[collider_state.n_broad_pairs[i_b], i_b][1] = i_gb
                            collider_state.n_broad_pairs[i_b] = collider_state.n_broad_pairs[i_b] + 1

                        # if incoming geom is awake, also need to check with hibernated geoms
                        if not is_incoming_geom_hibernated:
                            for j in range(n_active_hib):
                                i_ga = collider_state.active_buffer_hib[j, i_b]
                                i_gb = collider_state.sort_buffer.i_g[i, i_b]
                                if i_ga > i_gb:
                                    i_ga, i_gb = i_gb, i_ga

                                if not func_check_collision_valid(
                                    i_ga,
                                    i_gb,
                                    i_b,
                                    links_state,
                                    links_info,
                                    geoms_info,
                                    static_rigid_sim_config,
                                    collider_info,
                                ):
                                    continue

                                if not func_is_geom_aabbs_overlap(i_ga, i_gb, i_b, geoms_state, geoms_info):
                                    # Clear collision normal cache if not in contact
                                    # self.contact_cache[i_ga, i_gb, i_b].i_va_ws = -1
                                    collider_state.contact_cache.normal[i_ga, i_gb, i_b] = ti.Vector.zero(
                                        gs.ti_float, 3
                                    )
                                    continue

                                collider_state.broad_collision_pairs[collider_state.n_broad_pairs[i_b], i_b][0] = i_ga
                                collider_state.broad_collision_pairs[collider_state.n_broad_pairs[i_b], i_b][1] = i_gb
                                collider_state.n_broad_pairs[i_b] = collider_state.n_broad_pairs[i_b] + 1

                        if is_incoming_geom_hibernated:
                            collider_state.active_buffer_hib[n_active_hib, i_b] = collider_state.sort_buffer.i_g[i, i_b]
                            n_active_hib = n_active_hib + 1
                        else:
                            collider_state.active_buffer_awake[n_active_awake, i_b] = collider_state.sort_buffer.i_g[
                                i, i_b
                            ]
                            n_active_awake = n_active_awake + 1
                    else:
                        i_g_to_remove = collider_state.sort_buffer.i_g[i, i_b]
                        if is_incoming_geom_hibernated:
                            for j in range(n_active_hib):
                                if collider_state.active_buffer_hib[j, i_b] == i_g_to_remove:
                                    if j < n_active_hib - 1:
                                        for k in range(j, n_active_hib - 1):
                                            collider_state.active_buffer_hib[k, i_b] = collider_state.active_buffer_hib[
                                                k + 1, i_b
                                            ]
                                    n_active_hib = n_active_hib - 1
                                    break
                        else:
                            for j in range(n_active_awake):
                                if collider_state.active_buffer_awake[j, i_b] == i_g_to_remove:
                                    if j < n_active_awake - 1:
                                        for k in range(j, n_active_awake - 1):
                                            collider_state.active_buffer_awake[k, i_b] = (
                                                collider_state.active_buffer_awake[k + 1, i_b]
                                            )
                                    n_active_awake = n_active_awake - 1
                                    break


@ti.kernel
def func_narrow_phase_convex_vs_convex(
    links_state: array_class.LinksState,
    links_info: array_class.LinksInfo,
    geoms_state: array_class.GeomsState,
    geoms_info: array_class.GeomsInfo,
    geoms_init_AABB: array_class.GeomsInitAABB,
    verts_info: array_class.VertsInfo,
    faces_info: array_class.FacesInfo,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: ti.template(),
    collider_state: array_class.ColliderState,
    collider_info: array_class.ColliderInfo,
    collider_static_config: ti.template(),
    mpr_state: array_class.MPRState,
    mpr_static_config: ti.template(),
    gjk_state: array_class.GJKState,
    gjk_static_config: ti.template(),
    sdf_info: array_class.SDFInfo,
    support_field_info: array_class.SupportFieldInfo,
    support_field_static_config: ti.template(),
):
    """
    NOTE: for a single non-batched scene with a lot of collisioin pairs, it will be faster if we also parallelize over `self.n_collision_pairs`.
    However, parallelize over both B and collision_pairs (instead of only over B) leads to significantly slow performance for batched scene.
    We can treat B=0 and B>0 separately, but we will end up with messier code.
    Therefore, for a big non-batched scene, users are encouraged to simply use `gs.cpu` backend.
    Updated NOTE & TODO: For a HUGE scene with numerous bodies, it's also reasonable to run on GPU. Let's save this for later.
    Update2: Now we use n_broad_pairs instead of n_collision_pairs, so we probably need to think about how to handle non-batched large scene better.
    """
    _B = collider_state.active_buffer.shape[1]

    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
    for i_b in range(_B):
        for i_pair in range(collider_state.n_broad_pairs[i_b]):
            i_ga = collider_state.broad_collision_pairs[i_pair, i_b][0]
            i_gb = collider_state.broad_collision_pairs[i_pair, i_b][1]

            if geoms_info.type[i_ga] > geoms_info.type[i_gb]:
                i_ga, i_gb = i_gb, i_ga

            if (
                geoms_info.is_convex[i_ga]
                and geoms_info.is_convex[i_gb]
                and not geoms_info.type[i_gb] == gs.GEOM_TYPE.TERRAIN
                and not (
                    static_rigid_sim_config.box_box_detection
                    and geoms_info.type[i_ga] == gs.GEOM_TYPE.BOX
                    and geoms_info.type[i_gb] == gs.GEOM_TYPE.BOX
                )
            ):
                if ti.static(sys.platform == "darwin"):
                    func_convex_convex_contact(
                        i_ga=i_ga,
                        i_gb=i_gb,
                        i_b=i_b,
                        links_state=links_state,
                        links_info=links_info,
                        geoms_state=geoms_state,
                        geoms_info=geoms_info,
                        geoms_init_AABB=geoms_init_AABB,
                        verts_info=verts_info,
                        faces_info=faces_info,
                        static_rigid_sim_config=static_rigid_sim_config,
                        collider_state=collider_state,
                        collider_info=collider_info,
                        collider_static_config=collider_static_config,
                        mpr_state=mpr_state,
                        mpr_static_config=mpr_static_config,
                        gjk_state=gjk_state,
                        gjk_static_config=gjk_static_config,
                        sdf_info=sdf_info,
                        support_field_info=support_field_info,
                        support_field_static_config=support_field_static_config,
                    )
                else:
                    if not (geoms_info.type[i_ga] == gs.GEOM_TYPE.PLANE and geoms_info.type[i_gb] == gs.GEOM_TYPE.BOX):
                        func_convex_convex_contact(
                            i_ga=i_ga,
                            i_gb=i_gb,
                            i_b=i_b,
                            links_state=links_state,
                            links_info=links_info,
                            geoms_state=geoms_state,
                            geoms_info=geoms_info,
                            geoms_init_AABB=geoms_init_AABB,
                            verts_info=verts_info,
                            faces_info=faces_info,
                            static_rigid_sim_config=static_rigid_sim_config,
                            collider_state=collider_state,
                            collider_info=collider_info,
                            collider_static_config=collider_static_config,
                            mpr_state=mpr_state,
                            mpr_static_config=mpr_static_config,
                            gjk_state=gjk_state,
                            gjk_static_config=gjk_static_config,
                            sdf_info=sdf_info,
                            support_field_info=support_field_info,
                            support_field_static_config=support_field_static_config,
                        )


@ti.kernel
def func_narrow_phase_convex_specializations(
    geoms_state: array_class.GeomsState,
    geoms_info: array_class.GeomsInfo,
    geoms_init_AABB: array_class.GeomsInitAABB,
    verts_info: array_class.VertsInfo,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: ti.template(),
    collider_state: array_class.ColliderState,
    collider_info: array_class.ColliderInfo,
    collider_static_config: ti.template(),
):
    _B = collider_state.active_buffer.shape[1]
    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
    for i_b in range(_B):
        for i_pair in range(collider_state.n_broad_pairs[i_b]):
            i_ga = collider_state.broad_collision_pairs[i_pair, i_b][0]
            i_gb = collider_state.broad_collision_pairs[i_pair, i_b][1]

            if geoms_info.type[i_ga] > geoms_info.type[i_gb]:
                i_ga, i_gb = i_gb, i_ga

            if ti.static(sys.platform != "darwin"):
                if geoms_info.type[i_ga] == gs.GEOM_TYPE.PLANE and geoms_info.type[i_gb] == gs.GEOM_TYPE.BOX:
                    func_plane_box_contact(
                        i_ga,
                        i_gb,
                        i_b,
                        geoms_state,
                        geoms_info,
                        geoms_init_AABB,
                        verts_info,
                        static_rigid_sim_config,
                        collider_state,
                        collider_info,
                        collider_static_config,
                    )

            if ti.static(static_rigid_sim_config.box_box_detection):
                if geoms_info.type[i_ga] == gs.GEOM_TYPE.BOX and geoms_info.type[i_gb] == gs.GEOM_TYPE.BOX:
                    func_box_box_contact(
                        i_ga,
                        i_gb,
                        i_b,
                        geoms_state,
                        geoms_info,
                        collider_state,
                        collider_info,
                        collider_static_config,
                    )


@ti.kernel
def func_narrow_phase_any_vs_terrain(
    geoms_state: array_class.GeomsState,
    geoms_info: array_class.GeomsInfo,
    geoms_init_AABB: array_class.GeomsInitAABB,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: ti.template(),
    collider_state: array_class.ColliderState,
    collider_info: array_class.ColliderInfo,
    collider_static_config: ti.template(),
    mpr_state: array_class.MPRState,
    mpr_static_config: ti.template(),
    support_field_info: array_class.SupportFieldInfo,
    support_field_static_config: ti.template(),
):
    """
    NOTE: for a single non-batched scene with a lot of collisioin pairs, it will be faster if we also parallelize over `self.n_collision_pairs`. However, parallelize over both B and collisioin_pairs (instead of only over B) leads to significantly slow performance for batched scene. We can treat B=0 and B>0 separately, but we will end up with messier code.
    Therefore, for a big non-batched scene, users are encouraged to simply use `gs.cpu` backend.
    Updated NOTE & TODO: For a HUGE scene with numerous bodies, it's also reasonable to run on GPU. Let's save this for later.
    Update2: Now we use n_broad_pairs instead of n_collision_pairs, so we probably need to think about how to handle non-batched large scene better.
    """
    _B = collider_state.active_buffer.shape[1]
    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
    for i_b in range(_B):
        for i_pair in range(collider_state.n_broad_pairs[i_b]):
            i_ga = collider_state.broad_collision_pairs[i_pair, i_b][0]
            i_gb = collider_state.broad_collision_pairs[i_pair, i_b][1]

            if ti.static(collider_static_config.has_terrain):
                if geoms_info.type[i_ga] == gs.GEOM_TYPE.TERRAIN:
                    i_ga, i_gb = i_gb, i_ga

                if geoms_info.type[i_gb] == gs.GEOM_TYPE.TERRAIN:
                    func_contact_mpr_terrain(
                        i_ga,
                        i_gb,
                        i_b,
                        geoms_state,
                        geoms_info,
                        geoms_init_AABB,
                        static_rigid_sim_config,
                        collider_state,
                        collider_info,
                        collider_static_config,
                        mpr_state,
                        mpr_static_config,
                        support_field_info,
                        support_field_static_config,
                    )


@ti.kernel
def func_narrow_phase_nonconvex_vs_nonterrain(
    links_state: array_class.LinksState,
    links_info: array_class.LinksInfo,
    geoms_state: array_class.GeomsState,
    geoms_info: array_class.GeomsInfo,
    geoms_init_AABB: array_class.GeomsInitAABB,
    verts_info: array_class.VertsInfo,
    edges_info: array_class.EdgesInfo,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: ti.template(),
    collider_state: array_class.ColliderState,
    collider_info: array_class.ColliderInfo,
    collider_static_config: ti.template(),
    sdf_info: array_class.SDFInfo,
):
    """
    NOTE: for a single non-batched scene with a lot of collisioin pairs, it will be faster if we also parallelize over `self.n_collision_pairs`. However, parallelize over both B and collisioin_pairs (instead of only over B) leads to significantly slow performance for batched scene. We can treat B=0 and B>0 separately, but we will end up with messier code.
    Therefore, for a big non-batched scene, users are encouraged to simply use `gs.cpu` backend.
    Updated NOTE & TODO: For a HUGE scene with numerous bodies, it's also reasonable to run on GPU. Let's save this for later.
    Update2: Now we use n_broad_pairs instead of n_collision_pairs, so we probably need to think about how to handle non-batched large scene better.
    """
    _B = collider_state.active_buffer.shape[1]
    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
    for i_b in range(_B):
        for i_pair in range(collider_state.n_broad_pairs[i_b]):
            i_ga = collider_state.broad_collision_pairs[i_pair, i_b][0]
            i_gb = collider_state.broad_collision_pairs[i_pair, i_b][1]

            if ti.static(collider_static_config.has_nonconvex_nonterrain):
                if (
                    not (geoms_info.is_convex[i_ga] and geoms_info.is_convex[i_gb])
                    and geoms_info.type[i_gb] != gs.GEOM_TYPE.TERRAIN
                ):
                    is_col = False
                    tolerance = func_compute_tolerance(
                        i_ga, i_gb, i_b, geoms_info, geoms_init_AABB, collider_static_config
                    )
                    for i in range(2):
                        if i == 1:
                            i_ga, i_gb = i_gb, i_ga

                        # initial point
                        is_col_i = False
                        normal_i = ti.Vector.zero(gs.ti_float, 3)
                        contact_pos_i = ti.Vector.zero(gs.ti_float, 3)
                        if not is_col:
                            is_col_i, normal_i, penetration_i, contact_pos_i = func_contact_vertex_sdf(
                                i_ga,
                                i_gb,
                                i_b,
                                geoms_state,
                                geoms_info,
                                verts_info,
                                collider_static_config,
                                sdf_info,
                            )
                            if is_col_i:
                                func_add_contact(
                                    i_ga,
                                    i_gb,
                                    normal_i,
                                    contact_pos_i,
                                    penetration_i,
                                    i_b,
                                    geoms_state,
                                    geoms_info,
                                    collider_state,
                                    collider_info,
                                )

                        if ti.static(static_rigid_sim_config.enable_multi_contact):
                            if not is_col and is_col_i:
                                ga_pos, ga_quat = geoms_state.pos[i_ga, i_b], geoms_state.quat[i_ga, i_b]
                                gb_pos, gb_quat = geoms_state.pos[i_gb, i_b], geoms_state.quat[i_gb, i_b]

                                # Perturb geom_a around two orthogonal axes to find multiple contacts
                                axis_0, axis_1 = func_contact_orthogonals(
                                    i_ga,
                                    i_gb,
                                    normal_i,
                                    i_b,
                                    links_state,
                                    links_info,
                                    geoms_state,
                                    geoms_info,
                                    geoms_init_AABB,
                                    static_rigid_sim_config,
                                )

                                n_con = 1
                                for i_rot in range(1, 5):
                                    axis = (2 * (i_rot % 2) - 1) * axis_0 + (1 - 2 * ((i_rot // 2) % 2)) * axis_1

                                    qrot = gu.ti_rotvec_to_quat(collider_static_config.mc_perturbation * axis)
                                    func_rotate_frame(i_ga, contact_pos_i, qrot, i_b, geoms_state, geoms_info)
                                    func_rotate_frame(
                                        i_gb, contact_pos_i, gu.ti_inv_quat(qrot), i_b, geoms_state, geoms_info
                                    )

                                    is_col, normal, penetration, contact_pos = func_contact_vertex_sdf(
                                        i_ga,
                                        i_gb,
                                        i_b,
                                        geoms_state,
                                        geoms_info,
                                        verts_info,
                                        collider_static_config,
                                        sdf_info,
                                    )

                                    if is_col:
                                        if ti.static(not static_rigid_sim_config.enable_mujoco_compatibility):
                                            # 1. Project the contact point on both geometries
                                            # 2. Revert the effect of small rotation
                                            # 3. Update contact point
                                            contact_point_a = (
                                                gu.ti_transform_by_quat(
                                                    (contact_pos - 0.5 * penetration * normal) - contact_pos_i,
                                                    gu.ti_inv_quat(qrot),
                                                )
                                                + contact_pos_i
                                            )
                                            contact_point_b = (
                                                gu.ti_transform_by_quat(
                                                    (contact_pos + 0.5 * penetration * normal) - contact_pos_i,
                                                    qrot,
                                                )
                                                + contact_pos_i
                                            )
                                            contact_pos = 0.5 * (contact_point_a + contact_point_b)

                                            # First-order correction of the normal direction
                                            twist_rotvec = ti.math.clamp(
                                                normal.cross(normal_i),
                                                -collider_static_config.mc_perturbation,
                                                collider_static_config.mc_perturbation,
                                            )
                                            normal += twist_rotvec.cross(normal)

                                            # Make sure that the penetration is still positive
                                            penetration = normal.dot(contact_point_b - contact_point_a)

                                        # Discard contact point is repeated
                                        repeated = False
                                        for i_c in range(n_con):
                                            if not repeated:
                                                idx_prev = collider_state.n_contacts[i_b] - 1 - i_c
                                                prev_contact = collider_state.contact_data.pos[idx_prev, i_b]
                                                if (contact_pos - prev_contact).norm() < tolerance:
                                                    repeated = True

                                        if not repeated:
                                            if penetration > -tolerance:
                                                penetration = ti.max(penetration, 0.0)
                                                func_add_contact(
                                                    i_ga,
                                                    i_gb,
                                                    normal,
                                                    contact_pos,
                                                    penetration,
                                                    i_b,
                                                    geoms_state,
                                                    geoms_info,
                                                    collider_state,
                                                    collider_info,
                                                )
                                                n_con += 1

                                    geoms_state.pos[i_ga, i_b], geoms_state.quat[i_ga, i_b] = ga_pos, ga_quat
                                    geoms_state.pos[i_gb, i_b], geoms_state.quat[i_gb, i_b] = gb_pos, gb_quat

                    if not is_col:  # check edge-edge if vertex-face is not detected
                        is_col, normal, penetration, contact_pos = func_contact_edge_sdf(
                            i_ga,
                            i_gb,
                            i_b,
                            geoms_state,
                            geoms_info,
                            verts_info,
                            edges_info,
                            collider_static_config,
                            sdf_info,
                        )
                        if is_col:
                            func_add_contact(
                                i_ga,
                                i_gb,
                                normal,
                                contact_pos,
                                penetration,
                                i_b,
                                geoms_state,
                                geoms_info,
                                collider_state,
                                collider_info,
                            )


@ti.func
def func_plane_box_contact(
    i_ga,
    i_gb,
    i_b,
    geoms_state: array_class.GeomsState,
    geoms_info: array_class.GeomsInfo,
    geoms_init_AABB: array_class.GeomsInitAABB,
    verts_info: array_class.VertsInfo,
    static_rigid_sim_config: ti.template(),
    collider_state: array_class.ColliderState,
    collider_info: array_class.ColliderInfo,
    collider_static_config: ti.template(),
):
    ga_pos, ga_quat = geoms_state.pos[i_ga, i_b], geoms_state.quat[i_ga, i_b]
    gb_pos, gb_quat = geoms_state.pos[i_gb, i_b], geoms_state.quat[i_gb, i_b]

    plane_dir = ti.Vector(
        [geoms_info.data[i_ga][0], geoms_info.data[i_ga][1], geoms_info.data[i_ga][2]], dt=gs.ti_float
    )
    plane_dir = gu.ti_transform_by_quat(plane_dir, ga_quat)
    normal = -plane_dir.normalized()

    v1, _ = support_field._func_support_box(geoms_state, geoms_info, normal, i_gb, i_b)
    penetration = normal.dot(v1 - ga_pos)

    if penetration > 0.0:
        contact_pos = v1 - 0.5 * penetration * normal
        func_add_contact(
            i_ga,
            i_gb,
            normal,
            contact_pos,
            penetration,
            i_b,
            geoms_state,
            geoms_info,
            collider_state,
            collider_info,
        )

        if ti.static(static_rigid_sim_config.enable_multi_contact):
            n_con = 1
            contact_pos_0 = contact_pos
            tolerance = func_compute_tolerance(i_ga, i_gb, i_b, geoms_info, geoms_init_AABB, collider_static_config)
            for i_v in range(geoms_info.vert_start[i_gb], geoms_info.vert_end[i_gb]):
                if n_con < ti.static(collider_static_config.n_contacts_per_pair):
                    pos_corner = gu.ti_transform_by_trans_quat(verts_info.init_pos[i_v], gb_pos, gb_quat)
                    penetration = normal.dot(pos_corner - ga_pos)
                    if penetration > 0.0:
                        contact_pos = pos_corner - 0.5 * penetration * normal
                        if (contact_pos - contact_pos_0).norm() > tolerance:
                            func_add_contact(
                                i_ga,
                                i_gb,
                                normal,
                                contact_pos,
                                penetration,
                                i_b,
                                geoms_state,
                                geoms_info,
                                collider_state,
                                collider_info,
                            )
                            n_con = n_con + 1


@ti.func
def func_add_contact(
    i_ga,
    i_gb,
    normal: ti.types.vector(3),
    contact_pos: ti.types.vector(3),
    penetration,
    i_b,
    geoms_state: array_class.GeomsState,
    geoms_info: array_class.GeomsInfo,
    collider_state: array_class.ColliderState,
    collider_info: array_class.ColliderInfo,
):
    i_c = collider_state.n_contacts[i_b]
    if i_c == collider_info._max_contact_pairs[None]:
        # FIXME: 'ti.static_print' cannot be used as it will be printed systematically, completely ignoring guard
        # condition, while 'print' is slowing down the kernel even if every called in practice...
        # print(self._warn_msg_max_collision_pairs)
        pass
    else:
        friction_a = geoms_info.friction[i_ga] * geoms_state.friction_ratio[i_ga, i_b]
        friction_b = geoms_info.friction[i_gb] * geoms_state.friction_ratio[i_gb, i_b]

        # b to a
        collider_state.contact_data.geom_a[i_c, i_b] = i_ga
        collider_state.contact_data.geom_b[i_c, i_b] = i_gb
        collider_state.contact_data.normal[i_c, i_b] = normal
        collider_state.contact_data.pos[i_c, i_b] = contact_pos
        collider_state.contact_data.penetration[i_c, i_b] = penetration
        collider_state.contact_data.friction[i_c, i_b] = ti.max(ti.max(friction_a, friction_b), 1e-2)
        collider_state.contact_data.sol_params[i_c, i_b] = 0.5 * (
            geoms_info.sol_params[i_ga] + geoms_info.sol_params[i_gb]
        )
        collider_state.contact_data.link_a[i_c, i_b] = geoms_info.link_idx[i_ga]
        collider_state.contact_data.link_b[i_c, i_b] = geoms_info.link_idx[i_gb]

        collider_state.n_contacts[i_b] = i_c + 1


@ti.func
def func_compute_tolerance(
    i_ga,
    i_gb,
    i_b,
    geoms_info: array_class.GeomsInfo,
    geoms_init_AABB: array_class.GeomsInitAABB,
    collider_static_config: ti.template(),
):
    # Note that the original world-aligned bounding box is used to computed the absolute tolerance from the
    # relative one. This way, it is a constant that does not depends on the orientation of the geometry, which
    # makes sense since the scale of the geometries is an intrinsic property and not something that is supposed
    # to change dynamically.
    aabb_size_b = (geoms_init_AABB[i_gb, 7] - geoms_init_AABB[i_gb, 0]).norm()
    aabb_size = aabb_size_b
    if geoms_info.type[i_ga] != gs.GEOM_TYPE.PLANE:
        aabb_size_a = (geoms_init_AABB[i_ga, 7] - geoms_init_AABB[i_ga, 0]).norm()
        aabb_size = ti.min(aabb_size_a, aabb_size_b)

    return 0.5 * collider_static_config.mc_tolerance * aabb_size


@ti.func
def func_contact_orthogonals(
    i_ga,
    i_gb,
    normal: ti.types.vector(3),
    i_b,
    links_state: array_class.LinksState,
    links_info: array_class.LinksInfo,
    geoms_state: array_class.GeomsState,
    geoms_info: array_class.GeomsInfo,
    geoms_init_AABB: array_class.GeomsInitAABB,
    static_rigid_sim_config: ti.template(),
):
    axis_0 = ti.Vector.zero(gs.ti_float, 3)
    axis_1 = ti.Vector.zero(gs.ti_float, 3)

    if ti.static(static_rigid_sim_config.enable_mujoco_compatibility):
        # Choose between world axes Y or Z to avoid colinearity issue
        if ti.abs(normal[1]) < 0.5:
            axis_0[1] = 1.0
        else:
            axis_0[2] = 1.0

        # Project axis on orthogonal plane to contact normal
        axis_0 = (axis_0 - normal.dot(axis_0) * normal).normalized()

        # Perturb with some noise so that they do not align with world axes to avoid denegerated cases
        axis_1 = (normal.cross(axis_0) + 0.1 * axis_0).normalized()
        axis_0 = axis_1.cross(normal)
    else:
        # The reference geometry is the one that will have the largest impact on the position of
        # the contact point. Basically, the smallest one between the two, which can be approximated
        # by the volume of their respective bounding box.
        i_g = i_gb
        if geoms_info.type[i_ga] != gs.GEOM_TYPE.PLANE:
            size_ga = geoms_init_AABB[i_ga, 7]
            volume_ga = size_ga[0] * size_ga[1] * size_ga[2]
            size_gb = geoms_init_AABB[i_gb, 7]
            volume_gb = size_gb[0] * size_gb[1] * size_gb[2]
            i_g = i_ga if volume_ga < volume_gb else i_gb

        # Compute orthogonal basis mixing principal inertia axes of geometry with contact normal
        i_l = geoms_info.link_idx[i_g]
        rot = gu.ti_quat_to_R(links_state.i_quat[i_l, i_b])
        axis_idx = gs.ti_int(0)
        axis_angle_max = gs.ti_float(0.0)
        for i in ti.static(range(3)):
            axis_angle = ti.abs(rot[:, i].dot(normal))
            if axis_angle > axis_angle_max:
                axis_angle_max = axis_angle
                axis_idx = i
        axis_idx = (axis_idx + 1) % 3
        axis_0 = rot[:, axis_idx]
        axis_0 = (axis_0 - normal.dot(axis_0) * normal).normalized()
        axis_1 = normal.cross(axis_0)

    return axis_0, axis_1


@ti.func
def func_convex_convex_contact(
    i_ga,
    i_gb,
    i_b,
    links_state: array_class.LinksState,
    links_info: array_class.LinksInfo,
    geoms_state: array_class.GeomsState,
    geoms_info: array_class.GeomsInfo,
    geoms_init_AABB: array_class.GeomsInitAABB,
    verts_info: array_class.VertsInfo,
    faces_info: array_class.FacesInfo,
    static_rigid_sim_config: ti.template(),
    collider_state: array_class.ColliderState,
    collider_info: array_class.ColliderInfo,
    collider_static_config: ti.template(),
    mpr_state: array_class.MPRState,
    mpr_static_config: ti.template(),
    gjk_state: array_class.GJKState,
    gjk_static_config: ti.template(),
    sdf_info: array_class.SDFInfo,
    support_field_info: array_class.SupportFieldInfo,
    support_field_static_config: ti.template(),
):
    if geoms_info.type[i_ga] == gs.GEOM_TYPE.PLANE and geoms_info.type[i_gb] == gs.GEOM_TYPE.BOX:
        if ti.static(sys.platform == "darwin"):
            func_plane_box_contact(
                i_ga=i_ga,
                i_gb=i_gb,
                i_b=i_b,
                geoms_state=geoms_state,
                geoms_info=geoms_info,
                geoms_init_AABB=geoms_init_AABB,
                verts_info=verts_info,
                static_rigid_sim_config=static_rigid_sim_config,
                collider_state=collider_state,
                collider_info=collider_info,
                collider_static_config=collider_static_config,
            )
    else:
        # Disabling multi-contact for pairs of decomposed geoms would speed up simulation but may cause physical
        # instabilities in the few cases where multiple contact points are actually need. Increasing the tolerance
        # criteria to get rid of redundant contact points seems to be a better option.
        multi_contact = (
            static_rigid_sim_config.enable_multi_contact
            # and not (self._solver.geoms_info[i_ga].is_decomposed and self._solver.geoms_info[i_gb].is_decomposed)
            and geoms_info.type[i_ga] != gs.GEOM_TYPE.SPHERE
            and geoms_info.type[i_ga] != gs.GEOM_TYPE.ELLIPSOID
            and geoms_info.type[i_gb] != gs.GEOM_TYPE.SPHERE
            and geoms_info.type[i_gb] != gs.GEOM_TYPE.ELLIPSOID
        )

        tolerance = func_compute_tolerance(i_ga, i_gb, i_b, geoms_info, geoms_init_AABB, collider_static_config)

        # Backup state before local perturbation
        ga_pos, ga_quat = geoms_state.pos[i_ga, i_b], geoms_state.quat[i_ga, i_b]
        gb_pos, gb_quat = geoms_state.pos[i_gb, i_b], geoms_state.quat[i_gb, i_b]

        # Pre-allocate some buffers
        is_col_0 = False
        penetration_0 = gs.ti_float(0.0)
        normal_0 = ti.Vector.zero(gs.ti_float, 3)
        contact_pos_0 = ti.Vector.zero(gs.ti_float, 3)

        is_col = False
        penetration = gs.ti_float(0.0)
        normal = ti.Vector.zero(gs.ti_float, 3)
        contact_pos = ti.Vector.zero(gs.ti_float, 3)

        n_con = gs.ti_int(0)
        axis_0 = ti.Vector.zero(gs.ti_float, 3)
        axis_1 = ti.Vector.zero(gs.ti_float, 3)
        qrot = ti.Vector.zero(gs.ti_float, 4)

        for i_detection in range(5):
            if multi_contact and is_col_0:
                # Perturbation axis must not be aligned with the principal axes of inertia the geometry,
                # otherwise it would be more sensitive to ill-conditionning.
                axis = (2 * (i_detection % 2) - 1) * axis_0 + (1 - 2 * ((i_detection // 2) % 2)) * axis_1
                qrot = gu.ti_rotvec_to_quat(collider_static_config.mc_perturbation * axis)
                func_rotate_frame(i_ga, contact_pos_0, qrot, i_b, geoms_state, geoms_info)
                func_rotate_frame(i_gb, contact_pos_0, gu.ti_inv_quat(qrot), i_b, geoms_state, geoms_info)

            if (multi_contact and is_col_0) or (i_detection == 0):
                try_sdf = False
                if geoms_info.type[i_ga] == gs.GEOM_TYPE.PLANE:
                    plane_dir = ti.Vector(
                        [geoms_info.data[i_ga][0], geoms_info.data[i_ga][1], geoms_info.data[i_ga][2]],
                        dt=gs.ti_float,
                    )
                    plane_dir = gu.ti_transform_by_quat(plane_dir, geoms_state.quat[i_ga, i_b])
                    normal = -plane_dir.normalized()

                    v1 = mpr.support_driver(
                        geoms_state,
                        geoms_info,
                        collider_state,
                        collider_info,
                        collider_static_config,
                        support_field_info,
                        support_field_static_config,
                        normal,
                        i_gb,
                        i_b,
                    )
                    penetration = normal.dot(v1 - geoms_state.pos[i_ga, i_b])
                    contact_pos = v1 - 0.5 * penetration * normal
                    is_col = penetration > 0
                else:
                    ### MPR, MJ_MPR
                    if ti.static(
                        collider_static_config.ccd_algorithm in (CCD_ALGORITHM_CODE.MPR, CCD_ALGORITHM_CODE.MJ_MPR)
                    ):
                        # Try using MPR before anything else
                        is_mpr_updated = False
                        is_mpr_guess_direction_available = True
                        normal_ws = collider_state.contact_cache.normal[i_ga, i_gb, i_b]
                        for i_mpr in range(2):
                            if i_mpr == 1:
                                # Try without warm-start if no contact was detected using it.
                                # When penetration depth is very shallow, MPR may wrongly classify two geometries as not in
                                # contact while they actually are. This helps to improve contact persistence without increasing
                                # much the overall computational cost since the fallback should not be triggered very often.
                                is_mpr_guess_direction_available = (ti.abs(normal_ws) > gs.EPS).any()
                                if (i_detection == 0) and not is_col and is_mpr_guess_direction_available:
                                    normal_ws = ti.Vector.zero(gs.ti_float, 3)
                                    is_mpr_updated = False

                            if not is_mpr_updated:
                                is_col, normal, penetration, contact_pos = mpr.func_mpr_contact(
                                    geoms_state,
                                    geoms_info,
                                    geoms_init_AABB,
                                    static_rigid_sim_config,
                                    collider_state,
                                    collider_info,
                                    collider_static_config,
                                    mpr_state,
                                    mpr_static_config,
                                    support_field_info,
                                    support_field_static_config,
                                    i_ga,
                                    i_gb,
                                    i_b,
                                    normal_ws,
                                )
                                is_mpr_updated = True

                        # Fallback on SDF if collision is detected by MPR but no collision direction was cached but the
                        # initial penetration is already quite large, because the contact information provided by MPR
                        # may be unreliable in such a case.
                        # Here it is assumed that generic SDF is much slower than MPR, so it is faster in average
                        # to first make sure that the geometries are truly colliding and only after to run SDF if
                        # necessary. This would probably not be the case anymore if it was possible to rely on
                        # specialized SDF implementation for convex-convex collision detection in the first place.
                        if is_col and penetration > tolerance and not is_mpr_guess_direction_available:
                            # Note that SDF may detect different collision points depending on geometry ordering.
                            # Because of this, it is necessary to run it twice and take the contact information
                            # associated with the point of deepest penetration.
                            try_sdf = True

                    ### GJK, MJ_GJK
                    elif ti.static(
                        collider_static_config.ccd_algorithm in (CCD_ALGORITHM_CODE.GJK, CCD_ALGORITHM_CODE.MJ_GJK)
                    ):
                        gjk.func_gjk_contact(
                            geoms_state,
                            geoms_info,
                            verts_info,
                            faces_info,
                            static_rigid_sim_config,
                            collider_state,
                            collider_static_config,
                            gjk_state,
                            gjk_static_config,
                            support_field_info,
                            support_field_static_config,
                            i_ga,
                            i_gb,
                            i_b,
                        )

                        is_col = gjk_state.is_col[i_b] == 1
                        penetration = gjk_state.penetration[i_b]
                        n_contacts = gjk_state.n_contacts[i_b]

                        if is_col:
                            if gjk_state.multi_contact_flag[i_b]:
                                # Used MuJoCo's multi-contact algorithm to find multiple contact points. Therefore,
                                # add the discovered contact points and stop multi-contact search.
                                for i_c in range(n_contacts):
                                    # Ignore contact points if the number of contacts exceeds the limit.
                                    if i_c < ti.static(collider_static_config.n_contacts_per_pair):
                                        contact_pos = gjk_state.contact_pos[i_b, i_c]
                                        normal = gjk_state.normal[i_b, i_c]
                                        func_add_contact(
                                            i_ga,
                                            i_gb,
                                            normal,
                                            contact_pos,
                                            penetration,
                                            i_b,
                                            geoms_state,
                                            geoms_info,
                                            collider_state,
                                            collider_info,
                                        )

                                break
                            else:
                                contact_pos = gjk_state.contact_pos[i_b, 0]
                                normal = gjk_state.normal[i_b, 0]

                if ti.static(collider_static_config.ccd_algorithm == CCD_ALGORITHM_CODE.MPR):
                    if try_sdf:
                        is_col_a = False
                        is_col_b = False
                        normal_a = ti.Vector.zero(gs.ti_float, 3)
                        normal_b = ti.Vector.zero(gs.ti_float, 3)
                        penetration_b = gs.ti_float(0.0)
                        penetration_a = gs.ti_float(0.0)
                        contact_pos_a = ti.Vector.zero(gs.ti_float, 3)
                        contact_pos_b = ti.Vector.zero(gs.ti_float, 3)
                        for i_sdf in range(2):
                            # FIXME: It is impossible to rely on `_func_contact_convex_convex_sdf` to get the contact
                            # information because the compilation times skyrockets from 42s for `_func_contact_vertex_sdf`
                            # to 2min51s on Apple Silicon M4 Max, which is not acceptable.
                            # is_col_i, normal_i, penetration_i, contact_pos_i, i_va = (
                            #     self._func_contact_convex_convex_sdf(
                            #         i_ga if i_sdf == 0 else i_gb,
                            #         i_gb if i_sdf == 0 else i_ga,
                            #         i_b,
                            #         self.contact_cache[i_ga, i_gb, i_b].i_va_ws,
                            #     )
                            # )
                            # self.contact_cache[i_ga, i_gb, i_b].i_va_ws = i_va
                            is_col_i, normal_i, penetration_i, contact_pos_i = func_contact_vertex_sdf(
                                i_ga if i_sdf == 0 else i_gb,
                                i_gb if i_sdf == 0 else i_ga,
                                i_b,
                                geoms_state,
                                geoms_info,
                                verts_info,
                                collider_static_config,
                                sdf_info,
                            )
                            if i_sdf == 0:
                                is_col_a = is_col_i
                                normal_a = normal_i
                                penetration_a = penetration_i
                                contact_pos_a = contact_pos_i
                            else:
                                is_col_b = is_col_i
                                normal_b = -normal_i
                                penetration_b = penetration_i
                                contact_pos_b = contact_pos_i

                        # MPR cannot handle collision detection for fully enclosed geometries. Falling back to SDF.
                        # Note that SDF does not take into account to direction of interest. As such, it cannot be
                        # used reliably for anything else than the point of deepest penetration.
                        prefer_sdf = (
                            collider_static_config.mc_tolerance * penetration
                            >= collider_static_config.mpr_to_sdf_overlap_ratio * tolerance
                        )

                        if is_col_a and (
                            not is_col_b or penetration_a >= max(penetration_b, (not prefer_sdf) * penetration)
                        ):
                            is_col = is_col_a
                            normal = normal_a
                            penetration = penetration_a
                            contact_pos = contact_pos_a
                        elif is_col_b and (
                            not is_col_a or penetration_b > max(penetration_a, (not prefer_sdf) * penetration)
                        ):
                            is_col = is_col_b
                            normal = normal_b
                            penetration = penetration_b
                            contact_pos = contact_pos_b

            if i_detection == 0:
                is_col_0, normal_0, penetration_0, contact_pos_0 = is_col, normal, penetration, contact_pos
                if is_col_0:
                    func_add_contact(
                        i_ga,
                        i_gb,
                        normal_0,
                        contact_pos_0,
                        penetration_0,
                        i_b,
                        geoms_state,
                        geoms_info,
                        collider_state,
                        collider_info,
                    )
                    if multi_contact:
                        # perturb geom_a around two orthogonal axes to find multiple contacts
                        axis_0, axis_1 = func_contact_orthogonals(
                            i_ga,
                            i_gb,
                            normal,
                            i_b,
                            links_state,
                            links_info,
                            geoms_state,
                            geoms_info,
                            geoms_init_AABB,
                            static_rigid_sim_config,
                        )
                        n_con = 1

                    if ti.static(
                        collider_static_config.ccd_algorithm in (CCD_ALGORITHM_CODE.MPR, CCD_ALGORITHM_CODE.GJK)
                    ):
                        collider_state.contact_cache.normal[i_ga, i_gb, i_b] = normal
                else:
                    # Clear collision normal cache if not in contact
                    # self.contact_cache[i_ga, i_gb, i_b].i_va_ws = -1
                    collider_state.contact_cache.normal[i_ga, i_gb, i_b] = ti.Vector.zero(gs.ti_float, 3)

            elif multi_contact and is_col_0 > 0 and is_col > 0:
                if ti.static(collider_static_config.ccd_algorithm in (CCD_ALGORITHM_CODE.MPR, CCD_ALGORITHM_CODE.GJK)):
                    # 1. Project the contact point on both geometries
                    # 2. Revert the effect of small rotation
                    # 3. Update contact point
                    contact_point_a = (
                        gu.ti_transform_by_quat(
                            (contact_pos - 0.5 * penetration * normal) - contact_pos_0,
                            gu.ti_inv_quat(qrot),
                        )
                        + contact_pos_0
                    )
                    contact_point_b = (
                        gu.ti_transform_by_quat(
                            (contact_pos + 0.5 * penetration * normal) - contact_pos_0,
                            qrot,
                        )
                        + contact_pos_0
                    )
                    contact_pos = 0.5 * (contact_point_a + contact_point_b)

                    # First-order correction of the normal direction.
                    # The way the contact normal gets twisted by applying perturbation of geometry poses is
                    # unpredictable as it depends on the final portal discovered by MPR. Alternatively, let compute
                    # the mininal rotation that makes the corrected twisted normal as closed as possible to the
                    # original one, up to the scale of the perturbation, then apply first-order Taylor expension of
                    # Rodrigues' rotation formula.
                    twist_rotvec = ti.math.clamp(
                        normal.cross(normal_0),
                        -collider_static_config.mc_perturbation,
                        collider_static_config.mc_perturbation,
                    )
                    normal += twist_rotvec.cross(normal)

                    # Make sure that the penetration is still positive before adding contact point.
                    # Note that adding some negative tolerance improves physical stability by encouraging persistent
                    # contact points and thefore more continuous contact forces, without changing the mean-field
                    # dynamics since zero-penetration contact points should not induce any force.
                    penetration = normal.dot(contact_point_b - contact_point_a)

                elif ti.static(collider_static_config.ccd_algorithm == CCD_ALGORITHM_CODE.MJ_GJK):
                    # Only change penetration to the initial one, because the normal vector could change abruptly
                    # under MuJoCo's GJK-EPA.
                    penetration = penetration_0

                # Discard contact point is repeated
                repeated = False
                for i_c in range(n_con):
                    if not repeated:
                        idx_prev = collider_state.n_contacts[i_b] - 1 - i_c
                        prev_contact = collider_state.contact_data.pos[idx_prev, i_b]
                        if (contact_pos - prev_contact).norm() < tolerance:
                            repeated = True

                if not repeated:
                    if penetration > -tolerance:
                        penetration = ti.max(penetration, 0.0)
                        func_add_contact(
                            i_ga,
                            i_gb,
                            normal,
                            contact_pos,
                            penetration,
                            i_b,
                            geoms_state,
                            geoms_info,
                            collider_state,
                            collider_info,
                        )
                        n_con = n_con + 1

            geoms_state.pos[i_ga, i_b] = ga_pos
            geoms_state.quat[i_ga, i_b] = ga_quat
            geoms_state.pos[i_gb, i_b] = gb_pos
            geoms_state.quat[i_gb, i_b] = gb_quat


@ti.func
def func_rotate_frame(
    i_g,
    contact_pos: ti.types.vector(3),
    qrot: ti.types.vector(4),
    i_b,
    geoms_state: array_class.GeomsState,
    geoms_info: array_class.GeomsInfo,
):
    geoms_state.quat[i_g, i_b] = gu.ti_transform_quat_by_quat(geoms_state.quat[i_g, i_b], qrot)

    rel = contact_pos - geoms_state.pos[i_g, i_b]
    vec = gu.ti_transform_by_quat(rel, qrot)
    vec = vec - rel
    geoms_state.pos[i_g, i_b] = geoms_state.pos[i_g, i_b] - vec


@ti.func
def func_box_box_contact(
    i_ga,
    i_gb,
    i_b,
    geoms_state: array_class.GeomsState,
    geoms_info: array_class.GeomsInfo,
    collider_state: array_class.ColliderState,
    collider_info: array_class.ColliderInfo,
    collider_static_config: ti.template(),
):
    """
    Use Mujoco's box-box contact detection algorithm for more stable collision detection.

    The compilation and running time of this function is longer than the MPR-based contact detection.

    Algorithm is from

    https://github.com/google-deepmind/mujoco/blob/main/src/engine/engine_collision_box.c
    """
    n = 0
    code = -1
    margin = gs.ti_float(0.0)
    is_return = False
    cle1, cle2 = 0, 0
    in_ = 0
    tmp2 = ti.Vector.zero(gs.ti_float, 3)
    margin2 = margin * margin
    rotmore = ti.Matrix.zero(gs.ti_float, 3, 3)

    ga_pos = geoms_state.pos[i_ga, i_b]
    gb_pos = geoms_state.pos[i_gb, i_b]
    ga_quat = geoms_state.quat[i_ga, i_b]
    gb_quat = geoms_state.quat[i_gb, i_b]

    size1 = (
        ti.Vector([geoms_info.data[i_ga][0], geoms_info.data[i_ga][1], geoms_info.data[i_ga][2]], dt=gs.ti_float) / 2
    )
    size2 = (
        ti.Vector([geoms_info.data[i_gb][0], geoms_info.data[i_gb][1], geoms_info.data[i_gb][2]], dt=gs.ti_float) / 2
    )

    pos1, pos2 = ga_pos, gb_pos
    mat1, mat2 = gu.ti_quat_to_R(ga_quat), gu.ti_quat_to_R(gb_quat)

    tmp1 = pos2 - pos1
    pos21 = mat1.transpose() @ tmp1

    tmp1 = pos1 - pos2
    pos12 = mat2.transpose() @ tmp1

    rot = mat1.transpose() @ mat2
    rott = rot.transpose()

    rotabs = ti.abs(rot)
    rottabs = ti.abs(rott)

    plen2 = rotabs @ size2
    plen1 = rotabs.transpose() @ size1
    penetration = margin
    for i in ti.static(range(3)):
        penetration = penetration + size1[i] * 3 + size2[i] * 3
    for i in ti.static(range(3)):
        c1 = -ti.abs(pos21[i]) + size1[i] + plen2[i]
        c2 = -ti.abs(pos12[i]) + size2[i] + plen1[i]

        if (c1 < -margin) or (c2 < -margin):
            is_return = True

        if c1 < penetration:
            penetration = c1
            code = i + 3 * (pos21[i] < 0) + 0

        if c2 < penetration:
            penetration = c2
            code = i + 3 * (pos12[i] < 0) + 6
    clnorm = ti.Vector([0.0, 0.0, 0.0], dt=gs.ti_float)
    for i, j in ti.static(ti.ndrange(3, 3)):
        rj0 = rott[j, 0]
        rj1 = rott[j, 1]
        rj2 = rott[j, 2]
        if i == 0:
            tmp2 = ti.Vector([0.0, -rj2, +rj1], dt=gs.ti_float)
        elif i == 1:
            tmp2 = ti.Vector([+rj2, 0.0, -rj0], dt=gs.ti_float)
        else:
            tmp2 = ti.Vector([-rj1, +rj0, 0.0], dt=gs.ti_float)

        c1 = tmp2.norm()
        tmp2 = tmp2 / c1
        if c1 >= gs.EPS:
            c2 = pos21.dot(tmp2)

            c3 = gs.ti_float(0.0)

            for k in ti.static(range(3)):
                if k != i:
                    c3 = c3 + size1[k] * ti.abs(tmp2[k])

            for k in ti.static(range(3)):
                if k != j:
                    m = i
                    n = 3 - k - j
                    if k - j > 3:
                        m = m - 1
                        n = n + 3
                    c3 = c3 + size2[k] * rotabs[m, n] / c1

                    3 * i + 3 - k - j

            c3 = c3 - ti.abs(c2)

            if c3 < -margin:
                is_return = True

            if c3 < penetration * (1.0 - 1e-12):
                penetration = c3
                cle1 = 0
                for k in ti.static(range(3)):
                    if (k != i) and ((tmp2[k] > 0) ^ (c2 < 0)):
                        cle1 = cle1 + (1 << k)

                cle2 = 0
                for k in ti.static(range(3)):
                    if k != j:
                        m = i
                        n = 3 - k - j
                        if k - j > 3:
                            m = m - 1
                            n = n + 3
                        if (rot[m, n] > 0) ^ (c2 < 0) ^ (ti.raw_mod(k - j + 3, 3) == 1):
                            cle2 = cle2 + (1 << k)

                code = 12 + i * 3 + j
                clnorm = tmp2
                in_ = c2 < 0
    if code == -1:
        is_return = True

    if not is_return:
        if code < 12:
            q1 = code % 6
            q2 = code // 6

            if q1 == 0:
                rotmore[0, 2] = -1
                rotmore[1, 1] = +1
                rotmore[2, 0] = +1
            elif q1 == 1:
                rotmore[0, 0] = +1
                rotmore[1, 2] = -1
                rotmore[2, 1] = +1
            elif q1 == 2:
                rotmore[0, 0] = +1
                rotmore[1, 1] = +1
                rotmore[2, 2] = +1
            elif q1 == 3:
                rotmore[0, 2] = +1
                rotmore[1, 1] = +1
                rotmore[2, 0] = -1
            elif q1 == 4:
                rotmore[0, 0] = +1
                rotmore[1, 2] = +1
                rotmore[2, 1] = -1
            elif q1 == 5:
                rotmore[0, 0] = -1
                rotmore[1, 1] = +1
                rotmore[2, 2] = -1

            i0 = 0
            i1 = 1
            i2 = 2
            f0 = f1 = f2 = 1
            if q1 == 0:
                i0 = 2
                f0 = -1
                i2 = 0
            elif q1 == 1:
                i1 = 2
                f1 = -1
                i2 = 1
            elif q1 == 3:
                i0 = 2
                i2 = 0
                f2 = -1
            elif q1 == 4:
                i1 = 2
                i2 = 1
                f2 = -1
            elif q1 == 5:
                f0 = -1
                f2 = -1

            r = ti.Matrix.zero(gs.ti_float, 3, 3)
            p = ti.Vector.zero(gs.ti_float, 3)
            s = ti.Vector.zero(gs.ti_float, 3)
            if q2:
                r = rotmore @ rot.transpose()
                p = rotaxis(pos12, i0, i1, i2, f0, f1, f2)
                tmp1 = rotaxis(size2, i0, i1, i2, f0, f1, f2)
                s = size1
            else:
                r = rotmatx(rot, i0, i1, i2, f0, f1, f2)
                p = rotaxis(pos21, i0, i1, i2, f0, f1, f2)
                tmp1 = rotaxis(size1, i0, i1, i2, f0, f1, f2)
                s = size2

            rt = r.transpose()
            ss = ti.abs(tmp1)
            lx = ss[0]
            ly = ss[1]
            hz = ss[2]
            p[2] = p[2] - hz
            lp = p

            clcorner = 0

            for i in ti.static(range(3)):
                if r[2, i] < 0:
                    clcorner = clcorner + (1 << i)

            for i in ti.static(range(3)):
                lp = lp + rt[i, :] * s[i] * (1 if (clcorner & (1 << i)) else -1)

            m, k = 0, 0
            collider_state.box_pts[m, i_b] = lp
            m = m + 1

            for i in ti.static(range(3)):
                if ti.abs(r[2, i]) < 0.5:
                    collider_state.box_pts[m, i_b] = rt[i, :] * s[i] * (-2 if (clcorner & (1 << i)) else 2)
                    m = m + 1

            collider_state.box_pts[3, i_b] = collider_state.box_pts[0, i_b] + collider_state.box_pts[1, i_b]
            collider_state.box_pts[4, i_b] = collider_state.box_pts[0, i_b] + collider_state.box_pts[2, i_b]
            collider_state.box_pts[5, i_b] = collider_state.box_pts[3, i_b] + collider_state.box_pts[2, i_b]

            if m > 1:
                collider_state.box_lines[k, i_b][0:3] = collider_state.box_pts[0, i_b]
                collider_state.box_lines[k, i_b][3:6] = collider_state.box_pts[1, i_b]
                k = k + 1

            if m > 2:
                collider_state.box_lines[k, i_b][0:3] = collider_state.box_pts[0, i_b]
                collider_state.box_lines[k, i_b][3:6] = collider_state.box_pts[2, i_b]
                k = k + 1

                collider_state.box_lines[k, i_b][0:3] = collider_state.box_pts[3, i_b]
                collider_state.box_lines[k, i_b][3:6] = collider_state.box_pts[2, i_b]
                k = k + 1

                collider_state.box_lines[k, i_b][0:3] = collider_state.box_pts[4, i_b]
                collider_state.box_lines[k, i_b][3:6] = collider_state.box_pts[1, i_b]
                k = k + 1

            for i in range(k):
                for q in ti.static(range(2)):
                    a = collider_state.box_lines[i, i_b][0 + q]
                    b = collider_state.box_lines[i, i_b][3 + q]
                    c = collider_state.box_lines[i, i_b][1 - q]
                    d = collider_state.box_lines[i, i_b][4 - q]
                    if ti.abs(b) > gs.EPS:
                        for _j in ti.static(range(2)):
                            j = 2 * _j - 1
                            l = ss[q] * j
                            c1 = (l - a) / b
                            if 0 <= c1 and c1 <= 1:
                                c2 = c + d * c1
                                if ti.abs(c2) <= ss[1 - q]:
                                    collider_state.box_points[n, i_b] = (
                                        collider_state.box_lines[i, i_b][0:3]
                                        + collider_state.box_lines[i, i_b][3:6] * c1
                                    )
                                    n = n + 1
            a = collider_state.box_pts[1, i_b][0]
            b = collider_state.box_pts[2, i_b][0]
            c = collider_state.box_pts[1, i_b][1]
            d = collider_state.box_pts[2, i_b][1]
            c1 = a * d - b * c

            if m > 2:
                for i in ti.static(range(4)):
                    llx = lx if (i // 2) else -lx
                    lly = ly if (i % 2) else -ly

                    x = llx - collider_state.box_pts[0, i_b][0]
                    y = lly - collider_state.box_pts[0, i_b][1]

                    u = (x * d - y * b) / c1
                    v = (y * a - x * c) / c1

                    if 0 < u and u < 1 and 0 < v and v < 1:
                        collider_state.box_points[n, i_b] = ti.Vector(
                            [
                                llx,
                                lly,
                                collider_state.box_pts[0, i_b][2]
                                + u * collider_state.box_pts[1, i_b][2]
                                + v * collider_state.box_pts[2, i_b][2],
                            ]
                        )
                        n = n + 1

            for i in range(1 << (m - 1)):
                tmp1 = collider_state.box_pts[0 if i == 0 else i + 2, i_b]
                if not (i and (tmp1[0] <= -lx or tmp1[0] >= lx or tmp1[1] <= -ly or tmp1[1] >= ly)):
                    collider_state.box_points[n, i_b] = tmp1
                    n = n + 1
            m = n
            n = 0

            for i in range(m):
                if collider_state.box_points[i, i_b][2] <= margin:
                    collider_state.box_points[n, i_b] = collider_state.box_points[i, i_b]
                    collider_state.box_depth[n, i_b] = collider_state.box_points[n, i_b][2]
                    collider_state.box_points[n, i_b][2] = collider_state.box_points[n, i_b][2] * 0.5
                    n = n + 1
            r = (mat2 if q2 else mat1) @ rotmore.transpose()
            p = pos2 if q2 else pos1
            tmp2 = ti.Vector(
                [(-1 if q2 else 1) * r[0, 2], (-1 if q2 else 1) * r[1, 2], (-1 if q2 else 1) * r[2, 2]],
                dt=gs.ti_float,
            )
            normal_0 = tmp2
            for i in range(n):
                dist = collider_state.box_points[i, i_b][2]
                collider_state.box_points[i, i_b][2] = collider_state.box_points[i, i_b][2] + hz
                tmp2 = r @ collider_state.box_points[i, i_b]
                contact_pos = tmp2 + p
                func_add_contact(
                    i_ga,
                    i_gb,
                    -normal_0,
                    contact_pos,
                    -dist,
                    i_b,
                    geoms_state,
                    geoms_info,
                    collider_state,
                    collider_info,
                )

        else:
            code = code - 12

            q1 = code // 3
            q2 = code % 3

            ax1, ax2 = 0, 0
            pax1, pax2 = 0, 0

            if q2 == 0:
                ax1, ax2 = 1, 2
            elif q2 == 1:
                ax1, ax2 = 0, 2
            elif q2 == 2:
                ax1, ax2 = 1, 0

            if q1 == 0:
                pax1, pax2 = 1, 2
            elif q1 == 1:
                pax1, pax2 = 0, 2
            elif q1 == 2:
                pax1, pax2 = 1, 0
            if rotabs[q1, ax1] < rotabs[q1, ax2]:
                ax1 = ax2
                ax2 = 3 - q2 - ax1

            if rottabs[q2, pax1] < rottabs[q2, pax2]:
                pax1 = pax2
                pax2 = 3 - q1 - pax1

            clface = 0
            if cle1 & (1 << pax2):
                clface = pax2
            else:
                clface = pax2 + 3

            rotmore.fill(0.0)
            if clface == 0:
                rotmore[0, 2], rotmore[1, 1], rotmore[2, 0] = -1, +1, +1
            elif clface == 1:
                rotmore[0, 0], rotmore[1, 2], rotmore[2, 1] = +1, -1, +1
            elif clface == 2:
                rotmore[0, 0], rotmore[1, 1], rotmore[2, 2] = +1, +1, +1
            elif clface == 3:
                rotmore[0, 2], rotmore[1, 1], rotmore[2, 0] = +1, +1, -1
            elif clface == 4:
                rotmore[0, 0], rotmore[1, 2], rotmore[2, 1] = +1, +1, -1
            elif clface == 5:
                rotmore[0, 0], rotmore[1, 1], rotmore[2, 2] = -1, +1, -1

            i0, i1, i2 = 0, 1, 2
            f0, f1, f2 = 1, 1, 1

            if clface == 0:
                i0, i2, f0 = 2, 0, -1
            elif clface == 1:
                i1, i2, f1 = 2, 1, -1
            elif clface == 3:
                i0, i2, f2 = 2, 0, -1
            elif clface == 4:
                i1, i2, f2 = 2, 1, -1
            elif clface == 5:
                f0, f2 = -1, -1

            p = rotaxis(pos21, i0, i1, i2, f0, f1, f2)
            rnorm = rotaxis(clnorm, i0, i1, i2, f0, f1, f2)
            r = rotmatx(rot, i0, i1, i2, f0, f1, f2)

            # TODO
            tmp1 = rotmore.transpose() @ size1

            s = ti.abs(tmp1)
            rt = r.transpose()

            lx, ly, hz = s[0], s[1], s[2]
            p[2] = p[2] - hz

            n = 0
            collider_state.box_points[n, i_b] = p

            collider_state.box_points[n, i_b] = collider_state.box_points[n, i_b] + rt[ax1, :] * size2[ax1] * (
                1 if (cle2 & (1 << ax1)) else -1
            )
            collider_state.box_points[n, i_b] = collider_state.box_points[n, i_b] + rt[ax2, :] * size2[ax2] * (
                1 if (cle2 & (1 << ax2)) else -1
            )

            collider_state.box_points[n + 1, i_b] = collider_state.box_points[n, i_b]
            collider_state.box_points[n, i_b] = collider_state.box_points[n, i_b] + rt[q2, :] * size2[q2]

            n = 1
            collider_state.box_points[n, i_b] = collider_state.box_points[n, i_b] - rt[q2, :] * size2[q2]

            n = 2
            collider_state.box_points[n, i_b] = p
            collider_state.box_points[n, i_b] = collider_state.box_points[n, i_b] + rt[ax1, :] * size2[ax1] * (
                -1 if (cle2 & (1 << ax1)) else 1
            )
            collider_state.box_points[n, i_b] = collider_state.box_points[n, i_b] + rt[ax2, :] * size2[ax2] * (
                1 if (cle2 & (1 << ax2)) else -1
            )

            collider_state.box_points[n + 1, i_b] = collider_state.box_points[n, i_b]
            collider_state.box_points[n, i_b] = collider_state.box_points[n, i_b] + rt[q2, :] * size2[q2]

            n = 3
            collider_state.box_points[n, i_b] = collider_state.box_points[n, i_b] - rt[q2, :] * size2[q2]

            n = 4
            collider_state.box_axi[0, i_b] = collider_state.box_points[0, i_b]
            collider_state.box_axi[1, i_b] = collider_state.box_points[1, i_b] - collider_state.box_points[0, i_b]
            collider_state.box_axi[2, i_b] = collider_state.box_points[2, i_b] - collider_state.box_points[0, i_b]

            if ti.abs(rnorm[2]) < gs.EPS:
                is_return = True
            if not is_return:
                innorm = (1 / rnorm[2]) * (-1 if in_ else 1)

                for i in ti.static(range(4)):
                    c1 = -collider_state.box_points[i, i_b][2] / rnorm[2]
                    collider_state.box_pu[i, i_b] = collider_state.box_points[i, i_b]
                    collider_state.box_points[i, i_b] = collider_state.box_points[i, i_b] + c1 * rnorm

                    collider_state.box_ppts2[i, 0, i_b] = collider_state.box_points[i, i_b][0]
                    collider_state.box_ppts2[i, 1, i_b] = collider_state.box_points[i, i_b][1]
                collider_state.box_pts[0, i_b] = collider_state.box_points[0, i_b]
                collider_state.box_pts[1, i_b] = collider_state.box_points[1, i_b] - collider_state.box_points[0, i_b]
                collider_state.box_pts[2, i_b] = collider_state.box_points[2, i_b] - collider_state.box_points[0, i_b]

                m = 3
                k = 0
                n = 0

                if m > 1:
                    collider_state.box_lines[k, i_b][0:3] = collider_state.box_pts[0, i_b]
                    collider_state.box_lines[k, i_b][3:6] = collider_state.box_pts[1, i_b]
                    collider_state.box_linesu[k, i_b][0:3] = collider_state.box_axi[0, i_b]
                    collider_state.box_linesu[k, i_b][3:6] = collider_state.box_axi[1, i_b]
                    k = k + 1

                if m > 2:
                    collider_state.box_lines[k, i_b][0:3] = collider_state.box_pts[0, i_b]
                    collider_state.box_lines[k, i_b][3:6] = collider_state.box_pts[2, i_b]
                    collider_state.box_linesu[k, i_b][0:3] = collider_state.box_axi[0, i_b]
                    collider_state.box_linesu[k, i_b][3:6] = collider_state.box_axi[2, i_b]
                    k = k + 1

                    collider_state.box_lines[k, i_b][0:3] = (
                        collider_state.box_pts[0, i_b] + collider_state.box_pts[1, i_b]
                    )
                    collider_state.box_lines[k, i_b][3:6] = collider_state.box_pts[2, i_b]
                    collider_state.box_linesu[k, i_b][0:3] = (
                        collider_state.box_axi[0, i_b] + collider_state.box_axi[1, i_b]
                    )
                    collider_state.box_linesu[k, i_b][3:6] = collider_state.box_axi[2, i_b]
                    k = k + 1

                    collider_state.box_lines[k, i_b][0:3] = (
                        collider_state.box_pts[0, i_b] + collider_state.box_pts[2, i_b]
                    )
                    collider_state.box_lines[k, i_b][3:6] = collider_state.box_pts[1, i_b]
                    collider_state.box_linesu[k, i_b][0:3] = (
                        collider_state.box_axi[0, i_b] + collider_state.box_axi[2, i_b]
                    )
                    collider_state.box_linesu[k, i_b][3:6] = collider_state.box_axi[1, i_b]
                    k = k + 1

                for i in range(k):
                    for q in ti.static(range(2)):
                        a = collider_state.box_lines[i, i_b][q]
                        b = collider_state.box_lines[i, i_b][q + 3]
                        c = collider_state.box_lines[i, i_b][1 - q]
                        d = collider_state.box_lines[i, i_b][4 - q]

                        if ti.abs(b) > gs.EPS:
                            for _j in ti.static(range(2)):
                                j = 2 * _j - 1
                                if n < collider_static_config.box_MAXCONPAIR:
                                    l = s[q] * j
                                    c1 = (l - a) / b
                                    if 0 <= c1 and c1 <= 1:
                                        c2 = c + d * c1
                                        if (ti.abs(c2) <= s[1 - q]) and (
                                            (
                                                collider_state.box_linesu[i, i_b][2]
                                                + collider_state.box_linesu[i, i_b][5] * c1
                                            )
                                            * innorm
                                            <= margin
                                        ):
                                            collider_state.box_points[n, i_b] = (
                                                collider_state.box_linesu[i, i_b][0:3] * 0.5
                                                + c1 * 0.5 * collider_state.box_linesu[i, i_b][3:6]
                                            )
                                            collider_state.box_points[n, i_b][q] = (
                                                collider_state.box_points[n, i_b][q] + 0.5 * l
                                            )
                                            collider_state.box_points[n, i_b][1 - q] = (
                                                collider_state.box_points[n, i_b][1 - q] + 0.5 * c2
                                            )
                                            collider_state.box_depth[n, i_b] = (
                                                collider_state.box_points[n, i_b][2] * innorm * 2
                                            )
                                            n = n + 1

                nl = n
                a = collider_state.box_pts[1, i_b][0]
                b = collider_state.box_pts[2, i_b][0]
                c = collider_state.box_pts[1, i_b][1]
                d = collider_state.box_pts[2, i_b][1]
                c1 = a * d - b * c

                for i in range(4):
                    if n < collider_static_config.box_MAXCONPAIR:
                        llx = lx if (i // 2) else -lx
                        lly = ly if (i % 2) else -ly

                        x = llx - collider_state.box_pts[0, i_b][0]
                        y = lly - collider_state.box_pts[0, i_b][1]

                        u = (x * d - y * b) / c1
                        v = (y * a - x * c) / c1

                        if nl == 0:
                            if (u < 0 or u > 1) and (v < 0 or v > 1):
                                continue
                        elif u < 0 or u > 1 or v < 0 or v > 1:
                            continue

                        u = ti.math.clamp(u, 0, 1)
                        v = ti.math.clamp(v, 0, 1)
                        tmp1 = (
                            collider_state.box_pu[0, i_b] * (1 - u - v)
                            + collider_state.box_pu[1, i_b] * u
                            + collider_state.box_pu[2, i_b] * v
                        )
                        collider_state.box_points[n, i_b][0] = llx
                        collider_state.box_points[n, i_b][1] = lly
                        collider_state.box_points[n, i_b][2] = 0

                        tmp2 = collider_state.box_points[n, i_b] - tmp1

                        c2 = tmp2.dot(tmp2)

                        if not (tmp1[2] > 0 and c2 > margin2):
                            collider_state.box_points[n, i_b] = collider_state.box_points[n, i_b] + tmp1
                            collider_state.box_points[n, i_b] = collider_state.box_points[n, i_b] * 0.5

                            collider_state.box_depth[n, i_b] = ti.sqrt(c2) * (-1 if tmp1[2] < 0 else 1)
                            n = n + 1

                nf = n

                for i in range(4):
                    if n < collider_static_config.box_MAXCONPAIR:
                        x, y = collider_state.box_ppts2[i, 0, i_b], collider_state.box_ppts2[i, 1, i_b]

                        if nl == 0:
                            if (nf != 0) and (x < -lx or x > lx) and (y < -ly or y > ly):
                                continue
                        elif x < -lx or x > lx or y < -ly or y > ly:
                            continue

                        c1 = 0
                        for j in ti.static(range(2)):
                            if collider_state.box_ppts2[i, j, i_b] < -s[j]:
                                c1 = c1 + (collider_state.box_ppts2[i, j, i_b] + s[j]) ** 2
                            elif collider_state.box_ppts2[i, j, i_b] > s[j]:
                                c1 = c1 + (collider_state.box_ppts2[i, j, i_b] - s[j]) ** 2

                        c1 = c1 + (collider_state.box_pu[i, i_b][2] * innorm) ** 2

                        if collider_state.box_pu[i, i_b][2] > 0 and c1 > margin2:
                            continue

                        tmp1 = ti.Vector(
                            [
                                collider_state.box_ppts2[i, 0, i_b] * 0.5,
                                collider_state.box_ppts2[i, 1, i_b] * 0.5,
                                0,
                            ],
                            dt=gs.ti_float,
                        )

                        for j in ti.static(range(2)):
                            if collider_state.box_ppts2[i, j, i_b] < -s[j]:
                                tmp1[j] = -s[j] * 0.5
                            elif collider_state.box_ppts2[i, j, i_b] > s[j]:
                                tmp1[j] = s[j] * 0.5

                        tmp1 = tmp1 + collider_state.box_pu[i, i_b] * 0.5
                        collider_state.box_points[n, i_b] = tmp1

                        collider_state.box_depth[n, i_b] = ti.sqrt(c1) * (
                            -1 if collider_state.box_pu[i, i_b][2] < 0 else 1
                        )
                        n = n + 1

                r = mat1 @ rotmore.transpose()

                tmp1 = r @ rnorm
                normal_0 = tmp1 * (-1 if in_ else 1)

                for i in range(n):
                    dist = collider_state.box_depth[i, i_b]
                    collider_state.box_points[i, i_b][2] = collider_state.box_points[i, i_b][2] + hz
                    tmp2 = r @ collider_state.box_points[i, i_b]
                    contact_pos = tmp2 + pos1
                    func_add_contact(
                        i_ga,
                        i_gb,
                        -normal_0,
                        contact_pos,
                        -dist,
                        i_b,
                        geoms_state,
                        geoms_info,
                        collider_state,
                        collider_info,
                    )
