"""
Collider module for rigid body collision detection.

This module provides collision detection functionality for the rigid body solver,
including broad-phase (sweep-and-prune), narrow-phase (convex-convex, SDF-based,
terrain), and contact management.
"""

import math
from typing import TYPE_CHECKING

import numpy as np
import torch
import trimesh

import genesis as gs

import genesis.utils.array_class as array_class
import genesis.engine.solvers.rigid.rigid_solver as rigid_solver
from genesis.engine.materials.rigid import Rigid
from genesis.utils.misc import assign_indexed_tensor, tensor_to_array, qd_to_torch, qd_to_numpy, indices_to_mask
from genesis.utils.sdf import SDF

from . import mpr
from . import gjk
from . import support_field

# Import and re-export from submodules for backward compatibility
from .broadphase import (
    func_find_intersect_midpoint,
    func_check_collision_valid,
    func_collision_clear,
    func_broad_phase,
    _func_broad_phase_sap,
    _func_broad_phase_all_vs_all,
)

from .contact import (
    collider_kernel_reset,
    kernel_collider_clear,
    kernel_masked_collider_clear,
    collider_kernel_get_contacts,
    func_add_contact,
    func_set_contact,
    func_add_diff_contact_input,
    func_compute_tolerance,
    func_contact_orthogonals,
    func_rotate_frame,
    func_set_upstream_grad,
    func_clamp_and_sort_contacts,
)
from . import narrowphase
from .narrowphase import (
    CCD_ALGORITHM_CODE,
    func_contact_sphere_sdf,
    func_contact_vertex_sdf,
    func_contact_edge_sdf,
    func_contact_convex_convex_sdf,
    func_contact_mpr_terrain,
    func_add_prism_vert,
    func_plane_box_contact,
    func_convex_convex_contact,
    func_box_box_contact,
    func_narrow_phase_diff_convex_vs_convex,
    func_narrow_phase_convex_specializations,
    func_narrow_phase_any_vs_terrain,
    func_narrow_phase_nonconvex_vs_nonterrain,
)

if TYPE_CHECKING:
    from genesis.engine.solvers.rigid.rigid_solver import RigidSolver


IS_OLD_TORCH = tuple(map(int, torch.__version__.split(".")[:2])) < (2, 8)


NEUTRAL_COLLISION_RES_ABS = 0.01
NEUTRAL_COLLISION_RES_REL = 0.05


class Collider:
    def __init__(self, rigid_solver: "RigidSolver"):
        self._solver = rigid_solver

        self._mc_perturbation = 1e-3 if self._solver._enable_mujoco_compatibility else 1e-2
        self._mc_tolerance = 1e-3 if self._solver._enable_mujoco_compatibility else 1e-2
        self._mpr_to_gjk_overlap_ratio = 0.25
        self._box_MAXCONPAIR = 16
        self._diff_pos_tolerance = 1e-2
        self._diff_normal_tolerance = 1e-2

        self._init_static_config()
        self._use_split_narrowphase = (
            self._collider_static_config.has_non_box_plane_convex_convex and gs.backend != gs.cpu
        )
        self._init_collision_fields()

        self._sdf = SDF(rigid_solver)
        self._mpr = mpr.MPR(rigid_solver)
        self._gjk = gjk.GJK(rigid_solver)
        self._support_field = support_field.SupportField(rigid_solver)

        if self._collider_static_config.has_nonconvex_nonterrain:
            self._sdf.activate()
        if self._collider_static_config.has_non_box_plane_convex_convex:
            self._gjk.activate()
        if self._collider_static_config.has_terrain or self._collider_static_config.has_non_box_plane_convex_convex:
            self._support_field.activate()

        if self._use_split_narrowphase:
            self._init_multicontact_gjk_state()

        if gs.use_zerocopy:
            self._contact_data: dict[str, torch.Tensor] = {}
            for key, name in (
                ("link_a", "link_a"),
                ("link_b", "link_b"),
                ("geom_a", "geom_a"),
                ("geom_b", "geom_b"),
                ("penetration", "penetration"),
                ("position", "pos"),
                ("normal", "normal"),
                ("force", "force"),
            ):
                self._contact_data[key] = qd_to_torch(
                    getattr(self._collider_state.contact_data, name), transpose=True, copy=False
                )

        # Make sure that the initial state is clean
        self.clear()

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

        n_contacts_per_pair = 20 if self._solver._static_rigid_sim_config.requires_grad else 5
        if (
            self._solver._options.box_box_detection
            and sum(geom.type == gs.GEOM_TYPE.BOX for geom in self._solver.geoms) > 1
        ):
            n_contacts_per_pair = max(n_contacts_per_pair, self._box_MAXCONPAIR)

        # Compute collision pairs and algorithm flags in a single pass
        (
            self._n_possible_pairs,
            self._collision_pair_idx,
            self._valid_collision_pairs,
            has_terrain,
            has_non_box_plane_convex_convex,
            has_convex_specialization,
            has_nonconvex_nonterrain,
        ) = self._compute_collision_pair_idx()

        # Initialize the static config, which stores every data that are compile-time constants.
        # Note that updating any of them will trigger recompilation.
        self._collider_static_config = array_class.StructColliderStaticConfig(
            has_terrain=has_terrain,
            has_non_box_plane_convex_convex=has_non_box_plane_convex_convex,
            has_convex_specialization=has_convex_specialization,
            has_nonconvex_nonterrain=has_nonconvex_nonterrain,
            n_contacts_per_pair=n_contacts_per_pair,
            ccd_algorithm=ccd_algorithm,
        )

    def _init_collision_fields(self) -> None:
        # Pre-compute fields, as they are needed to initialize the collider state and info.
        vert_neighbors, vert_neighbor_start, vert_n_neighbors = self._compute_verts_connectivity()
        n_vert_neighbors = len(vert_neighbors)
        n_valid_pairs = len(self._valid_collision_pairs)

        # Initialize [info], which stores every data that must be considered mutable from Quadrants's perspective,
        # i.e. unknown at compile time, but IMMUTABLE from Genesis scene's perspective after build.
        self._collider_info = array_class.get_collider_info(
            self._solver,
            n_vert_neighbors,
            n_valid_pairs,
            self._collider_static_config,
            mc_perturbation=self._mc_perturbation,
            mc_tolerance=self._mc_tolerance,
            mpr_to_gjk_overlap_ratio=self._mpr_to_gjk_overlap_ratio,
            diff_pos_tolerance=self._diff_pos_tolerance,
            diff_normal_tolerance=self._diff_normal_tolerance,
        )
        self._init_collision_pair_idx(self._collision_pair_idx)
        self._init_valid_pairs()
        self._init_verts_connectivity(vert_neighbors, vert_neighbor_start, vert_n_neighbors)
        self._init_max_contact_pairs(self._n_possible_pairs)
        self._init_terrain_state()

        # Initialize [state], which stores every data that are may be updated at every single simulation step
        n_possible_pairs_ = max(self._n_possible_pairs, 1)
        self._collider_state = array_class.get_collider_state(
            self._solver,
            self._solver._static_rigid_sim_config,
            n_possible_pairs_,
            self._solver._options.multiplier_collision_broad_phase,
            self._collider_info,
            self._collider_static_config,
        )

        # 'contact_data_cache' is not used in Quadrants kernels, so keep it outside of the collider state / info
        self._contact_data_cache: dict[tuple[bool, bool], dict[str, torch.Tensor | tuple[torch.Tensor]]] = {}

        # Contact0 & multicontact scratch states only needed when split narrowphase is active.
        # FIXME: Quadrants should expose a unified API to query GPU core count across all backends.
        # Falling back to upper bound for backends where torch.cuda is unavailable (e.g., CPU-only torch). Benchmarks
        # on RTX 6000 Blackwell (Genesis-Embodied-AI/Genesis#2616) showed that switching from hardcoded 40000 threads
        # to hardware-derived 21760 had marginal performance impact, so it should be fine.
        if self._use_split_narrowphase:
            if torch.cuda.is_available():
                gpu_props = torch.cuda.get_device_properties(torch.cuda.current_device())
                # NVIDIA: 128 CUDA cores per SM. AMD/ROCm: 64 stream processors per CU.
                cores_per_unit = 64 if torch.version.hip else 128
                gpu_cores = gpu_props.multi_processor_count * cores_per_unit
            elif gs.backend == gs.metal:
                # Upper-bound estimate for Apple Silicon: 40 GPU cores, each GPU core having 128 ALUs
                cores_per_unit = 128
                gpu_cores = 5120
            else:
                # Using AMD GPU as a baseline. AMD MI350X has 256 SM (so-called Compute Units) with 64 cores each.
                # See: https://www.amd.com/en/products/accelerators/instinct/mi350/mi350x.html
                # For comparison, RTX6000 Blackwell boasts 188 SMs, compared to 170 SMs for RTX5090 with 128 cores each.
                cores_per_unit = 64
                gpu_cores = 16384
            self._contact0_n_chunks = max(1, math.ceil(gpu_cores / self._solver._B))
            self._contact0_grid_size = self._solver._B * self._contact0_n_chunks
            self._contact0_mpr_state = array_class.get_mpr_state(self._contact0_grid_size)
            self._contact0_gjk_state = array_class.get_gjk_state_contact_only(self._contact0_grid_size)

            if self._collider_static_config.ccd_algorithm in (CCD_ALGORITHM_CODE.GJK, CCD_ALGORITHM_CODE.MJ_GJK):
                self._multicontact_n_gjk_threads = gpu_cores
            else:
                # Heuristic to distribute the workflow between GJK and MPR
                self._multicontact_n_gjk_threads = math.ceil((gpu_cores // 32) / 64) * 64
            self._multicontact_n_total_threads = gpu_cores
            self._multicontact_max_items_per_thread = cores_per_unit
            self._multicontact_mpr_state = array_class.get_mpr_state(self._multicontact_n_total_threads)

    def _init_multicontact_gjk_state(self):
        """Kernel 2 GJK state. Must be called after self._gjk is initialized."""
        self._multicontact_gjk_state = array_class.get_gjk_state(
            self._multicontact_n_gjk_threads,
            self._solver._static_rigid_sim_config,
            self._gjk._gjk_info,
            True,
            self._solver._static_rigid_sim_config.requires_grad,
        )

    def _compute_collision_pair_idx(self):
        """
        Compute flat indices of all valid collision pairs, plus algorithm flags.

        For each pair of geoms, determine if they can collide based on their properties and the solver configuration.
        Pairs that are already colliding at the initial configuration (qpos0) are filtered out with a warning.

        Returns (n_possible_pairs, collision_pair_idx, pair_flags) where pair_flags is a dict of booleans
        for has_terrain, has_non_box_plane_convex_convex, has_convex_specialization, has_nonconvex_nonterrain.
        """
        # Links whose contact is handled by an external solver (e.g. IPC) — exclude from GJK collision.
        # Only applies when the IPC coupler is active. Mirrors the link filtering logic in
        # IPCCoupler._add_rigid_geoms_to_ipc: for two_way_soft_constraint with a link filter,
        # only the filtered links are in IPC; for all other coupling modes, all links are in IPC.
        from genesis.engine.couplers import IPCCoupler

        n_geoms = self._solver.n_geoms
        geoms = self._solver.geoms

        if n_geoms == 0:
            empty_pairs = np.empty((0, 2), dtype=gs.np_int)
            return 0, np.full((0, 0), fill_value=-1, dtype=gs.np_int), empty_pairs, False, False, False, False

        # Links delegated to IPC coupler (skip pair only when BOTH are IPC-handled)
        ipc_delegated_link_idxs = set()
        ipc_only_link_idxs = set()
        if isinstance(self._solver.sim.coupler, IPCCoupler):
            for entity in self._solver._entities:
                if not entity.material.needs_coup:
                    continue
                mode = entity.material.coup_type
                if mode is None:
                    continue
                if mode == "ipc_only":
                    ipc_only_link_idxs.update(l.idx for l in entity.links)
                link_filter_names = entity.material.coup_links
                if mode == "two_way_soft_constraint" and link_filter_names is not None:
                    for name in link_filter_names:
                        ipc_delegated_link_idxs.add(entity.get_link(name=name).idx)
                else:
                    ipc_delegated_link_idxs.update(l.idx for l in entity.links)

        # Pre-compute per-geom properties into numpy arrays for vectorized filtering
        geom_link_idx = np.array([g.link.idx for g in geoms], dtype=np.int32)
        geom_root_idx = np.array([g.link.root_idx for g in geoms], dtype=np.int32)
        geom_is_fixed = np.array([g.link.is_fixed for g in geoms], dtype=bool)
        geom_entity_id = np.array([id(g.entity) for g in geoms], dtype=np.int64)
        geom_contype = np.array([g.contype for g in geoms], dtype=np.int64)
        geom_conaffinity = np.array([g.conaffinity for g in geoms], dtype=np.int64)
        geom_local_mask = np.array([g.entity.is_local_collision_mask for g in geoms], dtype=bool)
        geom_is_ipc_only = np.array([g.link.idx in ipc_only_link_idxs for g in geoms], dtype=bool)
        geom_is_ipc_deleg = np.array([g.link.idx in ipc_delegated_link_idxs for g in geoms], dtype=bool)
        geom_type = np.array([g.type for g in geoms], dtype=np.int32)
        geom_is_convex = np.array([g.is_convex for g in geoms], dtype=bool)

        # Build weld pairs set for O(1) lookup (use sorted tuple keys)
        weld_pairs = set()
        for eq in self._solver.equalities:
            if eq.type == gs.EQUALITY_TYPE.WELD:
                a, b = eq.eq_obj1id, eq.eq_obj2id
                weld_pairs.add((min(a, b), max(a, b)))

        # --- Vectorized filtering: build upper-triangular valid-pair mask ---
        row, col = np.triu_indices(n_geoms, k=1)

        link_a = geom_link_idx[row]
        link_b = geom_link_idx[col]

        # geoms in the same link
        valid = link_a != link_b

        # Skip all pairs involving ipc_only links
        valid &= ~geom_is_ipc_only[row]
        valid &= ~geom_is_ipc_only[col]

        # Skip pairs where both links are delegated to IPC
        valid &= ~(geom_is_ipc_deleg[row] & geom_is_ipc_deleg[col])

        # pair of fixed links wrt the world
        valid &= ~(geom_is_fixed[row] & geom_is_fixed[col])

        # contype and conaffinity
        same_entity = geom_entity_id[row] == geom_entity_id[col]
        has_local_mask = geom_local_mask[row] | geom_local_mask[col]
        con_match = (geom_contype[row] & geom_conaffinity[col]) | (geom_contype[col] & geom_conaffinity[row])
        con_skip = (same_entity | ~has_local_mask) & (con_match == 0)
        valid &= ~con_skip

        # self collision (same root) — disabled
        same_root = geom_root_idx[row] == geom_root_idx[col]
        if not self._solver._enable_self_collision:
            valid &= ~same_root

        # Weld constraint filtering
        if weld_pairs:
            link_min = np.minimum(link_a, link_b)
            link_max = np.maximum(link_a, link_b)
            is_weld = np.array([(link_min[i], link_max[i]) in weld_pairs for i in range(len(row))], dtype=bool)
            valid &= ~is_weld

        # --- Self-collision: adjacent and neutral overlap checks (Python loop, only same-root pairs) ---
        # These checks only apply when self_collision is enabled and the pair passed all vectorized filters
        self_colliding_pairs: list[tuple[int, int]] = []

        needs_self_check = self._solver._enable_self_collision and np.any(valid & same_root)
        needs_neutral_check = needs_self_check and not self._solver._enable_neutral_collision

        # Lazily compute geom vertices only for geoms that need neutral overlap checks
        geoms_verts: dict[int, np.ndarray] = {}
        if needs_neutral_check:
            self_root_indices = np.where(valid & same_root)[0]
            self_root_geom_idxs = np.unique(np.concatenate([row[self_root_indices], col[self_root_indices]]))
            # Compute vertices only for geoms involved in self-collision pairs,
            # shrunk by 0.1% to avoid false positive when detecting self-collision
            for gi in self_root_geom_idxs:
                verts = tensor_to_array(geoms[gi].get_verts())
                verts = verts.reshape((-1, *verts.shape[-2:]))
                centroid = verts.mean(axis=1, keepdims=True)
                verts = centroid + (1.0 - 1e-3) * (verts - centroid)
                geoms_verts[gi] = verts

        if needs_self_check:
            self_root_indices = np.where(valid & same_root)[0]
            for idx in self_root_indices:
                i_ga, i_gb = row[idx], col[idx]
                link_ga = geoms[i_ga].link
                link_gb = geoms[i_gb].link

                # adjacent links
                # FIXME: Links should be considered adjacent if connected by only fixed joints.
                if not self._solver._enable_adjacent_collision:
                    is_adjacent = False
                    link_a_, link_b_ = (link_ga, link_gb) if link_ga.idx < link_gb.idx else (link_gb, link_ga)
                    while link_b_.parent_idx != -1:
                        if link_b_.parent_idx == link_a_.idx:
                            is_adjacent = True
                            break
                        if not all(joint.type is gs.JOINT_TYPE.FIXED for joint in link_b_.joints):
                            break
                        link_b_ = self._solver.links[link_b_.parent_idx]
                    if is_adjacent:
                        valid[idx] = False
                        continue

                # active in neutral configuration (qpos0)
                if needs_neutral_check:
                    verts_a = geoms_verts[i_ga][0]
                    mesh_a = trimesh.Trimesh(vertices=verts_a, faces=geoms[i_ga].init_faces, process=False)
                    verts_b = geoms_verts[i_gb][0]
                    mesh_b = trimesh.Trimesh(vertices=verts_b, faces=geoms[i_gb].init_faces, process=False)
                    bounds_a, bounds_b = mesh_a.bounds, mesh_b.bounds
                    if not ((bounds_a[1] < bounds_b[0]).any() or (bounds_b[1] < bounds_a[0]).any()):
                        voxels_a = mesh_a.voxelized(
                            pitch=min(NEUTRAL_COLLISION_RES_ABS, NEUTRAL_COLLISION_RES_REL * max(mesh_a.extents))
                        )
                        voxels_b = mesh_b.voxelized(
                            pitch=min(NEUTRAL_COLLISION_RES_ABS, NEUTRAL_COLLISION_RES_REL * max(mesh_b.extents))
                        )
                        coords_a = voxels_a.indices_to_points(np.argwhere(voxels_a.matrix))
                        coords_b = voxels_b.indices_to_points(np.argwhere(voxels_b.matrix))
                        if voxels_a.is_filled(coords_b).any() or voxels_b.is_filled(coords_a).any():
                            self_colliding_pairs.append((i_ga, i_gb))
                            valid[idx] = False
                            continue

        # Emit warning for self-collision pairs
        if self_colliding_pairs:
            pairs = ", ".join((f"({i_ga}, {i_gb})") for i_ga, i_gb in self_colliding_pairs)
            gs.logger.warning(
                f"Filtered out geometry pairs causing self-collision for the neutral configuration (qpos0): {pairs}. "
                "Consider tuning Morph option 'decompose_robot_error_threshold' or specify dedicated collision meshes. "
                "This behavior can be disabled by setting Morph option 'enable_neutral_collision=True'."
            )

        # --- Build collision_pair_idx, valid pairs list, and count ---
        valid_indices = np.where(valid)[0]
        n_possible_pairs = len(valid_indices)
        collision_pair_idx = np.full((n_geoms, n_geoms), fill_value=-1, dtype=gs.np_int)
        collision_pair_idx[row[valid_indices], col[valid_indices]] = np.arange(n_possible_pairs, dtype=gs.np_int)

        valid_collision_pairs = np.stack([row[valid_indices], col[valid_indices]], axis=1).astype(gs.np_int)

        # --- Compute algorithm flags from valid pairs ---
        valid_type_a = geom_type[row[valid_indices]]
        valid_type_b = geom_type[col[valid_indices]]
        valid_convex_a = geom_is_convex[row[valid_indices]]
        valid_convex_b = geom_is_convex[col[valid_indices]]

        has_any_vs_terrain = bool(
            np.any((valid_type_a == gs.GEOM_TYPE.TERRAIN) | (valid_type_b == gs.GEOM_TYPE.TERRAIN))
        )
        both_convex = valid_convex_a & valid_convex_b
        if np.any(both_convex):
            is_box_a = valid_type_a == gs.GEOM_TYPE.BOX
            is_box_b = valid_type_b == gs.GEOM_TYPE.BOX
            is_plane_a = valid_type_a == gs.GEOM_TYPE.PLANE
            is_plane_b = valid_type_b == gs.GEOM_TYPE.PLANE
            is_plane_box = (is_plane_a & is_box_b) | (is_box_a & is_plane_b)
            specialized = is_plane_box
            if self._solver._options.box_box_detection:
                specialized = specialized | (is_box_a & is_box_b)
            has_non_box_plane_convex_convex = bool(np.any(both_convex & ~specialized))
        else:
            has_non_box_plane_convex_convex = False

        if self._solver._options.box_box_detection:
            spec_types = [gs.GEOM_TYPE.TERRAIN, gs.GEOM_TYPE.BOX]
            has_convex_specialization = bool(
                np.any(np.isin(valid_type_a, spec_types) | np.isin(valid_type_b, spec_types))
            )
        else:
            has_convex_specialization = bool(
                np.any(
                    ((valid_type_a == gs.GEOM_TYPE.BOX) & (valid_type_b == gs.GEOM_TYPE.PLANE))
                    | ((valid_type_a == gs.GEOM_TYPE.PLANE) & (valid_type_b == gs.GEOM_TYPE.BOX))
                )
            )

        has_nonconvex_vs_nonterrain = bool(
            np.any(
                ~(valid_convex_a & valid_convex_b)
                & (valid_type_a != gs.GEOM_TYPE.TERRAIN)
                & (valid_type_b != gs.GEOM_TYPE.TERRAIN)
            )
        )

        return (
            n_possible_pairs,
            collision_pair_idx,
            valid_collision_pairs,
            has_any_vs_terrain,
            has_non_box_plane_convex_convex,
            has_convex_specialization,
            has_nonconvex_vs_nonterrain,
        )

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
            offset = offset + len(geom.vert_neighbors)

        if self._solver.n_verts > 0:
            vert_neighbors = np.concatenate(vert_neighbors, dtype=gs.np_int)
            vert_neighbor_start = np.concatenate(vert_neighbor_start, dtype=gs.np_int)
            vert_n_neighbors = np.concatenate(vert_n_neighbors, dtype=gs.np_int)

        return vert_neighbors, vert_neighbor_start, vert_n_neighbors

    def _init_collision_pair_idx(self, collision_pair_idx):
        if self._n_possible_pairs == 0:
            self._collider_info.collision_pair_idx.fill(-1)
            return
        self._collider_info.collision_pair_idx.from_numpy(collision_pair_idx)

    def _init_valid_pairs(self):
        if len(self._valid_collision_pairs) > 0:
            self._collider_info.valid_collision_pairs.from_numpy(self._valid_collision_pairs)

    def _init_verts_connectivity(self, vert_neighbors, vert_neighbor_start, vert_n_neighbors):
        if self._solver.n_verts > 0:
            self._collider_info.vert_neighbors.from_numpy(vert_neighbors)
            self._collider_info.vert_neighbor_start.from_numpy(vert_neighbor_start)
            self._collider_info.vert_n_neighbors.from_numpy(vert_n_neighbors)

    def _init_max_contact_pairs(self, n_possible_pairs):
        max_collision_pairs = min(self._solver.max_collision_pairs, n_possible_pairs)
        max_contact_pairs = max_collision_pairs * self._collider_static_config.n_contacts_per_pair
        max_contact_pairs_broad = max_collision_pairs * self._solver._options.multiplier_collision_broad_phase

        self._collider_info.max_possible_pairs[None] = n_possible_pairs
        self._collider_info.max_collision_pairs[None] = max_collision_pairs
        self._collider_info.max_collision_pairs_broad[None] = max_contact_pairs_broad
        self._collider_info.max_contact_pairs[None] = max_contact_pairs

    def _init_terrain_state(self):
        if self._collider_static_config.has_terrain:
            solver = self._solver
            links_idx = solver.geoms_info.link_idx.to_numpy()[solver.geoms_info.type.to_numpy() == gs.GEOM_TYPE.TERRAIN]
            entity_idx = solver.links_info.entity_idx.to_numpy()[links_idx[0]]
            if isinstance(entity_idx, np.ndarray):
                entity_idx = entity_idx[0]
            entity = solver._entities[entity_idx]

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

    def reset(self, envs_idx=None, *, cache_only: bool = True) -> None:
        self._contact_data_cache.clear()
        if gs.use_zerocopy:
            envs_idx = slice(None) if envs_idx is None else envs_idx
            if not cache_only:
                first_time = qd_to_torch(self._collider_state.first_time, copy=False)
                if isinstance(envs_idx, torch.Tensor) and envs_idx.dtype == torch.bool:
                    first_time.masked_fill_(envs_idx, True)
                else:
                    first_time[envs_idx] = True

            normal = qd_to_torch(self._collider_state.contact_cache.normal, copy=False)
            if isinstance(envs_idx, torch.Tensor) and (not IS_OLD_TORCH or envs_idx.dtype == torch.bool):
                if envs_idx.dtype == torch.bool:
                    normal.masked_fill_(envs_idx[None, :, None], 0.0)
                else:
                    normal.scatter_(1, envs_idx[None, :, None].expand((normal.shape[0], -1, 3)), 0.0)
            elif envs_idx is None:
                normal.zero_()
            else:
                normal[:, envs_idx] = 0.0
            if gs.backend == gs.metal:
                torch.mps.synchronize()
            return

        envs_idx = self._solver._scene._sanitize_envs_idx(envs_idx)
        collider_kernel_reset(envs_idx, self._solver._static_rigid_sim_config, self._collider_state, cache_only)

    def clear(self, envs_idx=None):
        self.reset(envs_idx, cache_only=False)

        if (
            gs.use_zerocopy
            and not self._solver._use_hibernation
            and (not isinstance(envs_idx, torch.Tensor) or (not IS_OLD_TORCH or envs_idx.dtype == torch.bool))
        ):
            n_contacts = qd_to_torch(self._collider_state.n_contacts, copy=False)
            link_a = qd_to_torch(self._collider_state.contact_data.link_a, copy=False)
            link_b = qd_to_torch(self._collider_state.contact_data.link_b, copy=False)
            geom_a = qd_to_torch(self._collider_state.contact_data.geom_a, copy=False)
            geom_b = qd_to_torch(self._collider_state.contact_data.geom_b, copy=False)
            penetration = qd_to_torch(self._collider_state.contact_data.penetration, copy=False)
            pos = qd_to_torch(self._collider_state.contact_data.pos, copy=False)
            normal = qd_to_torch(self._collider_state.contact_data.normal, copy=False)
            force = qd_to_torch(self._collider_state.contact_data.force, copy=False)
            if isinstance(envs_idx, torch.Tensor) and envs_idx.dtype == torch.bool:
                n_contacts.masked_fill_(envs_idx, 0)
                link_a.masked_fill_(envs_idx[None, :], -1)
                link_b.masked_fill_(envs_idx[None, :], -1)
                geom_a.masked_fill_(envs_idx[None, :], -1)
                geom_b.masked_fill_(envs_idx[None, :], -1)
                penetration.masked_fill_(envs_idx[None, :], 0.0)
                pos.masked_fill_(envs_idx[None, :, None], 0.0)
                normal.masked_fill_(envs_idx[None, :, None], 0.0)
                force.masked_fill_(envs_idx[None, :, None], 0.0)
            elif isinstance(envs_idx, torch.Tensor):
                n_contacts.scatter_(0, envs_idx, 0)
                link_a.scatter_(1, envs_idx[None, :].expand(link_a.shape[0], -1), -1)
                link_b.scatter_(1, envs_idx[None, :].expand(link_b.shape[0], -1), -1)
                geom_a.scatter_(1, envs_idx[None, :].expand(geom_a.shape[0], -1), -1)
                geom_b.scatter_(1, envs_idx[None, :].expand(geom_b.shape[0], -1), -1)
                penetration.scatter_(1, envs_idx[None, :].expand(link_a.shape[0], -1), 0.0)
                pos.scatter_(1, envs_idx[None, :, None].expand(link_a.shape[0], -1, 3), 0.0)
                normal.scatter_(1, envs_idx[None, :, None].expand(link_a.shape[0], -1, 3), 0.0)
                force.scatter_(1, envs_idx[None, :, None].expand(link_a.shape[0], -1, 3), 0.0)
            else:
                env_mask = indices_to_mask(envs_idx)
                n_contacts[env_mask] = 0
                link_a[:, envs_idx] = -1
                link_b[:, envs_idx] = -1
                geom_a[:, envs_idx] = -1
                geom_b[:, envs_idx] = -1
                penetration[:, envs_idx] = 0.0
                pos[:, envs_idx] = 0.0
                normal[:, envs_idx] = 0.0
                force[:, envs_idx] = 0.0
            if gs.backend == gs.metal:
                torch.mps.synchronize()
            return

        if not isinstance(envs_idx, torch.Tensor):
            envs_idx = self._solver._scene._sanitize_envs_idx(envs_idx)
        if isinstance(envs_idx, torch.Tensor) and envs_idx.dtype == torch.bool:
            fn = kernel_masked_collider_clear
        else:
            fn = kernel_collider_clear
        fn(
            envs_idx,
            self._solver.links_state,
            self._solver.links_info,
            self._solver._static_rigid_sim_config,
            self._collider_state,
        )

    def _call_multicontact(self):
        narrowphase._func_narrowphase_multicontact_mixed(
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
            self._multicontact_mpr_state,
            self._mpr._mpr_info,
            self._multicontact_gjk_state,
            self._gjk._gjk_info,
            self._gjk._gjk_static_config,
            self._support_field._support_field_info,
            self._multicontact_gjk_state.diff_contact_input,
            self._solver._errno,
            self._multicontact_n_gjk_threads,
            self._multicontact_n_total_threads,
            self._multicontact_max_items_per_thread,
        )

    def detection(self) -> None:
        rigid_solver.kernel_update_geom_aabbs(
            self._solver.geoms_state,
            self._solver.geoms_init_AABB,
            self._solver._static_rigid_sim_config,
        )

        if self._n_possible_pairs == 0:
            return

        self._contact_data_cache.clear()
        func_broad_phase(
            self._solver.links_state,
            self._solver.links_info,
            self._solver.geoms_state,
            self._solver.geoms_info,
            self._solver._rigid_global_info,
            self._solver._static_rigid_sim_config,
            self._solver.constraint_solver.constraint_state,
            self._collider_state,
            self._solver.equalities_info,
            self._collider_info,
            self._solver._errno,
        )
        if self._use_split_narrowphase:
            narrowphase._func_reset_narrowphase_work_queues(
                self._collider_state,
            )
            narrowphase._func_narrowphase_contact0(
                self._solver.geoms_state,
                self._solver.geoms_info,
                self._solver.geoms_init_AABB,
                self._solver.verts_info,
                self._solver._rigid_global_info,
                self._solver._static_rigid_sim_config,
                self._collider_state,
                self._collider_info,
                self._collider_static_config,
                self._contact0_mpr_state,
                self._mpr._mpr_info,
                self._contact0_gjk_state,
                self._gjk._gjk_info,
                self._support_field._support_field_info,
                self._solver._errno,
                self._solver._B,
                self._contact0_n_chunks,
            )
            self._call_multicontact()
            narrowphase._func_prepare_gjk_rerun(self._collider_state)
            self._call_multicontact()
        elif self._collider_static_config.has_non_box_plane_convex_convex:
            narrowphase.func_narrow_phase_convex_vs_convex(
                self._solver.links_state,
                self._solver.links_info,
                self._solver.geoms_state,
                self._solver.geoms_info,
                self._solver.geoms_init_AABB,
                self._solver.verts_info,
                self._solver.faces_info,
                self._solver.edges_info,
                self._solver._rigid_global_info,
                self._solver._static_rigid_sim_config,
                self._collider_state,
                self._collider_info,
                self._collider_static_config,
                self._mpr._mpr_state,
                self._mpr._mpr_info,
                self._gjk._gjk_state,
                self._gjk._gjk_info,
                self._gjk._gjk_static_config,
                self._sdf._sdf_info,
                self._support_field._support_field_info,
                self._gjk._gjk_state.diff_contact_input,
                self._solver._errno,
            )
        if self._collider_static_config.has_convex_specialization:
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
                self._solver._errno,
            )
        if self._collider_static_config.has_terrain:
            func_narrow_phase_any_vs_terrain(
                self._solver.geoms_state,
                self._solver.geoms_info,
                self._solver.geoms_init_AABB,
                self._solver._static_rigid_sim_config,
                self._collider_state,
                self._collider_info,
                self._collider_static_config,
                self._mpr._mpr_state,
                self._mpr._mpr_info,
                self._support_field._support_field_info,
                self._solver._errno,
            )
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
                self._solver._errno,
            )

        if self._use_split_narrowphase:
            func_clamp_and_sort_contacts(
                self._collider_state,
                self._collider_info,
                self._solver._static_rigid_sim_config,
            )

    def get_contacts(self, as_tensor: bool = True, to_torch: bool = True, keep_batch_dim: bool = False):
        # Early return if already pre-computed
        contact_data = self._contact_data_cache.setdefault((as_tensor, to_torch), {})
        if contact_data:
            return contact_data.copy()

        n_envs = self._solver.n_envs
        if gs.use_zerocopy:
            n_contacts = qd_to_torch(self._collider_state.n_contacts, copy=False)
            if as_tensor or n_envs == 0:
                n_contacts_max = (n_contacts if n_envs == 0 else n_contacts.max()).item()

            for key, data in self._contact_data.items():
                if n_envs == 0:
                    data = data[0, :n_contacts_max] if not keep_batch_dim else data[:, :n_contacts_max]
                elif as_tensor:
                    data = data[:, :n_contacts_max]

                if to_torch:
                    if gs.backend == gs.cpu:
                        data = data.clone()
                else:
                    data = tensor_to_array(data)

                if n_envs > 0 and not as_tensor:
                    if keep_batch_dim:
                        data = tuple([data[i : i + 1, :j] for i, j in enumerate(n_contacts.tolist())])
                    else:
                        data = tuple([data[i, :j] for i, j in enumerate(n_contacts.tolist())])

                contact_data[key] = data

            return contact_data.copy()

        # Find out how much dynamic memory must be allocated
        n_contacts = qd_to_numpy(self._collider_state.n_contacts)
        n_contacts_max = n_contacts.max().item()
        if as_tensor:
            out_size = n_contacts_max * max(n_envs, 1)
        else:
            *n_contacts_starts, out_size = np.cumsum(n_contacts)
        n_contacts = n_contacts.tolist()

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
                as_tensor, iout, fout, self._solver._static_rigid_sim_config, self._collider_state
            )

        # Build structured view (no copy)
        if as_tensor:
            if n_envs > 0:
                iout = iout.reshape((n_envs, n_contacts_max, 4))
                fout = fout.reshape((n_envs, n_contacts_max, 10))
            if keep_batch_dim and n_envs == 0:
                iout = iout.reshape((1, n_contacts_max, 4))
                fout = fout.reshape((1, n_contacts_max, 10))
            iout_chunks = (iout[..., 0], iout[..., 1], iout[..., 2], iout[..., 3])
            fout_chunks = (fout[..., 0], fout[..., 1:4], fout[..., 4:7], fout[..., 7:])
            values = (*iout_chunks, *fout_chunks)
        else:
            # Split smallest dimension first, then largest dimension
            if n_envs == 0:
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
                if n_envs == 1:
                    values = [(value,) for value in (*iout_chunks, *fout_chunks)]
                else:
                    if to_torch:
                        iout_chunks = (torch.split(out, n_contacts) for out in iout_chunks)
                        fout_chunks = (torch.split(out, n_contacts) for out in fout_chunks)
                    else:
                        iout_chunks = (np.split(out, n_contacts_starts) for out in iout_chunks)
                        fout_chunks = (np.split(out, n_contacts_starts) for out in fout_chunks)
                    values = (*iout_chunks, *fout_chunks)

        # Store contact information in cache
        contact_data.update(
            zip(("link_a", "link_b", "geom_a", "geom_b", "penetration", "position", "normal", "force"), values)
        )

        return contact_data.copy()

    def backward(self, dL_dposition, dL_dnormal, dL_dpenetration):
        func_set_upstream_grad(dL_dposition, dL_dnormal, dL_dpenetration, self._collider_state)

        # Compute gradient
        func_narrow_phase_diff_convex_vs_convex.grad(
            self._solver.geoms_state,
            self._solver.geoms_info,
            self._solver._static_rigid_sim_config,
            self._collider_state,
            self._collider_info,
            self._gjk._gjk_info,
            self._collider_state.diff_contact_input,
        )


from genesis.utils.deprecated_module_wrapper import create_virtual_deprecated_module

create_virtual_deprecated_module(__name__, "genesis.engine.solvers.rigid.collider_decomp")
