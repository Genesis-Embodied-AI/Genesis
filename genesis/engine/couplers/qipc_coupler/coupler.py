from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

import numpy as np
import quadrants as qd
import torch

import genesis as gs
import genesis.utils.array_class as array_class
import genesis.utils.geom as gu
from genesis.repr_base import RBC
from genesis.utils.misc import qd_to_torch

from .utils import rotate_inertia_to_link_frame


def _rodrigues(axis: np.ndarray, theta: float) -> np.ndarray:
    """Rodrigues rotation: sign-preserving axis-angle to 3x3 matrix.

    Unlike gu.axis_angle_to_R which derives sin from sqrt(1-cos^2) (losing sign),
    this computes sin(theta) directly so negative angles rotate correctly.
    """
    axis = axis / np.linalg.norm(axis)
    c = np.cos(theta)
    s = np.sin(theta)
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]], dtype=np.float64)
    return np.eye(3, dtype=np.float64) * c + (1 - c) * np.outer(axis, axis) + s * K


if TYPE_CHECKING:
    from genesis.engine.simulator import Simulator
    from genesis.options.solvers import QIPCCouplerOptions


# ---------------------------------------------------------------------------
# Quadrants kernels — run entirely on GPU, no host roundtrip
# ---------------------------------------------------------------------------


@qd.func
def _func_mat3_to_quat(
    r00: gs.qd_float, r01: gs.qd_float, r02: gs.qd_float,
    r10: gs.qd_float, r11: gs.qd_float, r12: gs.qd_float,
    r20: gs.qd_float, r21: gs.qd_float, r22: gs.qd_float,
):
    """3x3 matrix -> quaternion (w,x,y,z) via Shepperd's method. No SVD needed for high-kappa ABD."""
    trace = r00 + r11 + r22
    w = 0.0
    x = 0.0
    y = 0.0
    z = 0.0
    if trace > 0.0:
        s = 2.0 * qd.sqrt(trace + 1.0)
        w = 0.25 * s
        x = (r21 - r12) / s
        y = (r02 - r20) / s
        z = (r10 - r01) / s
    elif r00 > r11 and r00 > r22:
        s = 2.0 * qd.sqrt(1.0 + r00 - r11 - r22)
        w = (r21 - r12) / s
        x = 0.25 * s
        y = (r01 + r10) / s
        z = (r02 + r20) / s
    elif r11 > r22:
        s = 2.0 * qd.sqrt(1.0 + r11 - r00 - r22)
        w = (r02 - r20) / s
        x = (r01 + r10) / s
        y = 0.25 * s
        z = (r12 + r21) / s
    else:
        s = 2.0 * qd.sqrt(1.0 + r22 - r00 - r11)
        w = (r10 - r01) / s
        x = (r02 + r20) / s
        y = (r12 + r21) / s
        z = 0.25 * s

    norm = qd.sqrt(w * w + x * x + y * y + z * z)
    return w / norm, x / norm, y / norm, z / norm


@qd.func
def _func_q12_to_T(t, R):
    """Build 4x4 transform from translation vector and 3x3 rotation matrix."""
    T = qd.Matrix.identity(gs.qd_float, 4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


@qd.func
def _func_R_to_quat(R):
    """3x3 rotation matrix to quaternion (w,x,y,z) via Shepperd's method."""
    return _func_mat3_to_quat(
        R[0, 0], R[0, 1], R[0, 2],
        R[1, 0], R[1, 1], R[1, 2],
        R[2, 0], R[2, 1], R[2, 2],
    )


@qd.kernel(fastcache=True)
def _kernel_qipc_writeback(
    abd_q: qd.types.ndarray(),
    body_indices: qd.types.ndarray(),
    link_indices: qd.types.ndarray(),
    rel_transforms: qd.types.ndarray(),
    dofs_pos: qd.types.ndarray(),
    dof_indices: qd.types.ndarray(),
    links_state: array_class.LinksState,
    dofs_state: array_class.DofsState,
):
    """Write QIPC ABD body poses and joint theta back to Genesis link/dof state.

    Fixed-joint merging maps multiple Genesis links to one ABD body. Each link
    carries a fixed relative transform (identity for the representative, a
    constant offset for merged members). The world pose of each link is
    T_body @ T_relative, where T_body comes from the ABD q of the merged body.
    """
    n_links = link_indices.shape[0]
    n_dofs = dof_indices.shape[0]

    for i in range(n_links):
        i_b = body_indices[i]
        link_idx = link_indices[i]

        T_body = _func_q12_to_T(
            qd.Vector([abd_q[i_b, 0], abd_q[i_b, 1], abd_q[i_b, 2]]),
            qd.Matrix([
                [abd_q[i_b, 3], abd_q[i_b, 4], abd_q[i_b, 5]],
                [abd_q[i_b, 6], abd_q[i_b, 7], abd_q[i_b, 8]],
                [abd_q[i_b, 9], abd_q[i_b, 10], abd_q[i_b, 11]],
            ]),
        )

        T_rel = _func_q12_to_T(
            qd.Vector([rel_transforms[i, 0], rel_transforms[i, 1], rel_transforms[i, 2]]),
            qd.Matrix([
                [rel_transforms[i, 3], rel_transforms[i, 4], rel_transforms[i, 5]],
                [rel_transforms[i, 6], rel_transforms[i, 7], rel_transforms[i, 8]],
                [rel_transforms[i, 9], rel_transforms[i, 10], rel_transforms[i, 11]],
            ]),
        )

        T_link = T_body @ T_rel

        links_state.pos[link_idx, 0] = T_link[:3, 3]
        w, x, y, z = _func_R_to_quat(T_link[:3, :3])
        links_state.quat[link_idx, 0][0] = w
        links_state.quat[link_idx, 0][1] = x
        links_state.quat[link_idx, 0][2] = y
        links_state.quat[link_idx, 0][3] = z

    for i in range(n_dofs):
        idx = dof_indices[i]
        dofs_state.pos[idx, 0] = dofs_pos[i]
        dofs_state.vel[idx, 0] = 0.0


# ---------------------------------------------------------------------------
# QIPCCoupler
# ---------------------------------------------------------------------------


class QIPCCoupler(RBC):
    """
    QIPC coupler: uses cuda-graph-qipc as the sole physics engine for rigid/FEM entities.

    Genesis provides all scene data (link meshes, mass/inertia, joint topology);
    QIPC receives structured data and runs the physics. No asset files are loaded by QIPC.
    """

    def __init__(self, simulator: Simulator, options: QIPCCouplerOptions):
        self._sim = simulator
        self._options = options

    @property
    def sim(self) -> Simulator:
        return self._sim

    @property
    def options(self) -> QIPCCouplerOptions:
        return self._options

    @staticmethod
    def _get_entity_config(entity):
        """Read per-entity QIPC config from material, with hardcoded defaults."""
        mat = entity.material
        return {
            "abd_kappa": getattr(mat, "qipc_abd_kappa", None) or 1e8,
            "kappa_pivot": getattr(mat, "qipc_kappa_pivot", None) or 1e8,
            "kappa_axis": getattr(mat, "qipc_kappa_axis", None) or 1e8,
            "default_kp": getattr(mat, "qipc_default_kp", None) or 100.0,
            "default_kv": getattr(mat, "qipc_default_kv", None) or 10.0,
            "home_qpos": getattr(mat, "qipc_home_qpos", None),
        }

    def build(self):
        from qipc import Scene as QIPCScene, trimesh
        from qipc.constitution import AffineBodyConstitution
        from qipc.scene.joint_collection import JointCollection

        assert self._sim.n_envs <= 1, "QIPCCoupler: n_envs > 1 not supported"

        entity = self._sim.rigid_solver.entities[0]
        self._entity = entity
        self._entity_config = self._get_entity_config(entity)

        self._scene = QIPCScene(
            dt=self._sim.dt,
            gravity=tuple(self._sim._gravity),
            **{"contact/enable": self._options.contact_enable},
        )

        abd = AffineBodyConstitution()

        T_world = self._compute_initial_transforms()
        self._T_world = T_world

        merge_groups, link_to_rep = self._build_merge_groups(entity)

        self._link_to_rep = link_to_rep
        self._merge_groups = merge_groups

        self._link_slots = {}
        self._group_slots = {}

        for rep, members in merge_groups:
            slot = self._create_merged_body(
                entity, rep, members, T_world, abd, trimesh,
            )
            if slot is None:
                continue
            self._group_slots[rep] = slot
            for m in members:
                self._link_slots[m] = slot

        # --- Joint creation ---
        per_joint_jcs = []
        self._genesis_dof_order = []

        home_qpos = self._entity_config["home_qpos"]
        init_qpos = home_qpos if home_qpos is not None else entity.init_qpos

        for joint in entity.joints:
            if joint.type == gs.JOINT_TYPE.FIXED:
                continue

            child_link = joint.link
            parent_local = child_link.parent_idx - entity.link_start
            child_local = child_link.idx_local

            parent_rep = link_to_rep[parent_local]
            child_rep = link_to_rep[child_local]

            if parent_rep == child_rep:
                continue

            if parent_rep not in self._group_slots or child_rep not in self._group_slots:
                gs.logger.warning(
                    f"QIPCCoupler: skipping joint '{joint.name}' — "
                    f"parent or child body not created."
                )
                continue

            if joint.type == gs.JOINT_TYPE.REVOLUTE:
                jtype = "revolute"
                axis_local = joint.dofs_motion_ang[0]
                extra_kwargs = {"kappa_pivot": self._entity_config["kappa_pivot"]}
            elif joint.type == gs.JOINT_TYPE.PRISMATIC:
                jtype = "prismatic"
                axis_local = joint.dofs_motion_vel[0]
                extra_kwargs = {"kappa_lateral": self._entity_config["kappa_pivot"]}
            else:
                gs.logger.warning(
                    f"QIPCCoupler: skipping unsupported joint type "
                    f"{joint.type} for joint '{joint.name}'."
                )
                continue

            T_parent_rep = T_world[parent_rep]
            T_child_rep = T_world[child_rep]
            T_parent_link = T_world[parent_local]
            T_child_link = T_world[child_local]

            R_parent_rep_inv = T_parent_rep[:3, :3].T
            R_child_rep_inv = T_child_rep[:3, :3].T

            # Joint pivot in world frame
            T_joint_world = T_parent_link @ self._make_link_to_child_T(child_link)
            anchor_world = T_joint_world[:3, 3]

            # anchor in each rep's local frame
            anchor_left = R_parent_rep_inv @ (anchor_world - T_parent_rep[:3, 3])
            anchor_right = R_child_rep_inv @ (anchor_world - T_child_rep[:3, 3])

            # axis in world frame
            R_child_in_parent = gu.quat_to_R(np.array(child_link.quat, dtype=np.float64))
            axis_world = T_parent_link[:3, :3] @ R_child_in_parent @ np.array(axis_local, dtype=np.float64)
            axis_world = axis_world / np.linalg.norm(axis_world)

            # axis in each rep's local frame
            axis_left = R_parent_rep_inv @ axis_world
            axis_right = R_child_rep_inv @ axis_world

            kp, kv = self._resolve_joint_gains(joint)

            dof_local = joint.dofs_idx_local[0]
            jc = self._scene.add_joint(
                joint.name,
                type=jtype,
                left=self._group_slots[parent_rep],
                right=self._group_slots[child_rep],
                anchor_left=anchor_left.tolist(),
                anchor_right=anchor_right.tolist(),
                axis_left=axis_left.tolist(),
                axis_right=axis_right.tolist(),
                kappa_axis=self._entity_config["kappa_axis"],
                enable_controller=True,
                kp=kp,
                kv=kv,
                theta_lower=float(joint.dofs_limit[0, 0]),
                theta_upper=float(joint.dofs_limit[0, 1]),
                init_theta=float(init_qpos[dof_local]),
                **extra_kwargs,
            )
            per_joint_jcs.append(jc)
            self._genesis_dof_order.append(joint.dofs_idx_local[0])

        self._jc = JointCollection.merge(per_joint_jcs) if per_joint_jcs else None

        self._genesis_dof_order = torch.tensor(
            self._genesis_dof_order, dtype=torch.int64, device=gs.device
        )

        self._scene.init()

        # --- Build body index mapping (after init) ---
        self._link_body_indices = {}
        for link in entity.links:
            rep = link_to_rep[link.idx_local]
            if rep not in self._group_slots:
                continue
            slot = self._group_slots[rep]
            body_offset = int(slot.geometry.meta["abd_body_offset"].cpu()[0])
            self._link_body_indices[link.idx_local] = body_offset

        # --- Pre-compute GPU index tensors ---
        active_links = [link for link in entity.links if link.idx_local in self._link_body_indices]

        link_idx_list = [entity.link_start + link.idx_local for link in active_links]
        self._link_indices_t = torch.tensor(link_idx_list, dtype=torch.int32, device=gs.device)
        self._body_indices_t = torch.tensor(
            [self._link_body_indices[link.idx_local] for link in active_links],
            dtype=torch.int64, device=gs.device,
        )

        # Per-link relative transforms for writeback (12-DOF: [t_rel(3), R_rel(9)])
        rel_data = np.zeros((len(active_links), 12), dtype=np.float64)
        for i, link in enumerate(active_links):
            rep = link_to_rep[link.idx_local]
            if link.idx_local == rep:
                rel_data[i, 3] = 1.0   # identity rotation
                rel_data[i, 7] = 1.0
                rel_data[i, 11] = 1.0
            else:
                T_rep = T_world[rep]
                T_member = T_world[link.idx_local]
                T_rel = np.linalg.inv(T_rep) @ T_member
                rel_data[i, 0:3] = T_rel[:3, 3]
                rel_data[i, 3:12] = T_rel[:3, :3].flatten()
        self._rel_transforms_t = torch.tensor(rel_data, dtype=torch.float64, device=gs.device)

        n_dofs = entity.n_dofs
        dof_idx_list = []
        for joint in entity.joints:
            if joint.type == gs.JOINT_TYPE.FIXED:
                continue
            for d in range(joint.n_dofs):
                dof_idx_list.append(entity.dof_start + joint.dofs_idx_local[d])
        self._dof_indices_t = torch.tensor(dof_idx_list, dtype=torch.int32, device=gs.device)

        self._wb_dofs_pos = torch.zeros(n_dofs, dtype=gs.tc_float, device=gs.device)

        self._debug_viewer = None
        if self._options.debug_viewer:
            self._debug_viewer = self._scene.viewer
            self._debug_viewer.up_axis = "z"

        self._substep_count = 0
        self._substeps_per_step = self._sim._substeps
        self._skip_first_step = True

        self._writeback_state()

    def reset(self, envs_idx=None):
        pass

    def preprocess(self, f):
        """Read ctrl_pos from Genesis and forward to QIPC (absolute frame)."""
        if self._jc is None:
            return
        ctrl_pos_view = qd_to_torch(self._sim.rigid_solver.dofs_state.ctrl_pos)
        entity_ctrl = ctrl_pos_view[self._entity.dof_start:self._entity.dof_end, 0].to(torch.float64)
        targets = entity_ctrl[self._genesis_dof_order]
        self._jc.control_dofs_position(targets)

    def couple(self, f):
        self._substep_count += 1
        if self._substep_count < self._substeps_per_step:
            return
        self._substep_count = 0

        if self._skip_first_step:
            self._skip_first_step = False
            return

        self._scene.step()

        gs.logger.debug(
            f"[QIPC] step_ms={self._scene.solver.step_ms:.2f} "
            f"newton={self._scene.solver.newton_iters} "
            f"pcg_max={self._scene.solver.max_pcg_iters} "
            f"ls_max={self._scene.solver.max_ls_iters}"
        )

        self._writeback_state()

        if self._debug_viewer is not None:
            self._debug_viewer.show()

    def couple_grad(self, f):
        pass

    # -------------------------------------------------------------------------
    # State writeback
    # -------------------------------------------------------------------------

    def _writeback_state(self):
        """Write QIPC state -> Genesis buffers (absolute frame, no offset needed)."""
        abd_q = self._scene.affine_body.q

        if self._jc is not None:
            theta = self._jc.get_dofs_position()
            self._wb_dofs_pos[self._genesis_dof_order] = theta.to(gs.tc_float)

        _kernel_qipc_writeback(
            abd_q=abd_q,
            body_indices=self._body_indices_t,
            link_indices=self._link_indices_t,
            rel_transforms=self._rel_transforms_t,
            dofs_pos=self._wb_dofs_pos,
            dof_indices=self._dof_indices_t,
            links_state=self._sim.rigid_solver.links_state,
            dofs_state=self._sim.rigid_solver.dofs_state,
        )

    # -------------------------------------------------------------------------
    # Build-time helpers
    # -------------------------------------------------------------------------

    @staticmethod
    def _build_merge_groups(entity):
        """Group links connected by fixed joints.

        Returns (groups, link_to_rep) where groups is a list of (rep, members)
        tuples and link_to_rep maps each idx_local to its group representative.
        """
        fixed_adj: dict[int, list[int]] = defaultdict(list)

        for link in entity.links:
            for joint in link.joints:
                if joint.type == gs.JOINT_TYPE.FIXED:
                    parent_local = link.parent_idx - entity.link_start
                    fixed_adj[link.idx_local].append(parent_local)
                    fixed_adj[parent_local].append(link.idx_local)

        # Also treat jointless non-root links as fixed (MJCF bodies without <joint>)
        for link in entity.links:
            if link.parent_idx >= 0 and len(link.joints) == 0:
                parent_local = link.parent_idx - entity.link_start
                fixed_adj[link.idx_local].append(parent_local)
                fixed_adj[parent_local].append(link.idx_local)

        # Compute link depth via BFS from root
        depth: dict[int, int] = {}
        for link in entity.links:
            if link.parent_idx < 0:
                depth[link.idx_local] = 0
        bfs_queue = [idx for idx in depth]
        while bfs_queue:
            current = bfs_queue.pop(0)
            for link in entity.links:
                parent_local = link.parent_idx - entity.link_start
                if parent_local == current and link.idx_local not in depth:
                    depth[link.idx_local] = depth[current] + 1
                    bfs_queue.append(link.idx_local)

        # BFS through fixed adjacency to build groups
        visited: set[int] = set()
        groups: list[tuple[int, list[int]]] = []

        for link in entity.links:
            if link.idx_local in visited:
                continue
            members: list[int] = []
            queue = [link.idx_local]
            while queue:
                n = queue.pop(0)
                if n in visited:
                    continue
                visited.add(n)
                members.append(n)
                for neighbor in fixed_adj.get(n, []):
                    if neighbor not in visited:
                        queue.append(neighbor)

            rep = min(members, key=lambda x: depth.get(x, 999))
            groups.append((rep, members))

        link_to_rep = {}
        for rep, members in groups:
            for m in members:
                link_to_rep[m] = rep

        return groups, link_to_rep

    def _create_merged_body(self, entity, rep, members, T_world, abd, trimesh_factory):
        """Create a single ABD body from a merge group."""
        T_rep = T_world[rep]
        T_rep_inv = np.linalg.inv(T_rep)
        R_rep = T_rep[:3, :3]
        t_rep = T_rep[:3, 3]

        all_verts = []
        all_faces = []
        vert_offset = 0

        link_by_idx = {link.idx_local: link for link in entity.links}

        for m_idx in members:
            link = link_by_idx[m_idx]
            if not hasattr(link, "geoms") or len(link.geoms) == 0:
                continue

            T_member = T_world[m_idx]

            for geom in link.geoms:
                v = geom.init_verts.copy().astype(np.float64)
                R_geom = gu.quat_to_R(np.array(geom.init_quat, dtype=np.float64))
                v = (R_geom @ v.T).T + geom.init_pos

                # Transform from link-local to world, then to rep-local
                T_geom_world = T_member
                T_geom_in_rep = T_rep_inv @ T_geom_world
                v_h = np.hstack([v, np.ones((len(v), 1))])
                v_rep = (T_geom_in_rep @ v_h.T).T[:, :3]

                all_verts.append(v_rep)
                all_faces.append(geom.init_faces.copy() + vert_offset)
                vert_offset += len(v_rep)

        total_mass, com_world, I_world = self._merge_inertials(
            entity, members, T_world,
        )
        com_local = R_rep.T @ (com_world - t_rep)
        I_local = R_rep.T @ I_world @ R_rep

        is_fixed = any(link_by_idx[m].is_fixed for m in members)

        if all_verts:
            merged_verts = np.concatenate(all_verts, axis=0)
            merged_faces = np.concatenate(all_faces, axis=0)

            geo = trimesh_factory(merged_verts, merged_faces)
            geo.instances.resize(1)
            geo.transforms = T_rep.reshape(1, 4, 4)

            if total_mass > 0:
                vol = self._compute_merged_volume(entity, members)
                abd.apply_to(
                    geo,
                    kappa=self._entity_config["abd_kappa"],
                    mass=total_mass,
                    center_of_mass=com_local,
                    inertia=I_local,
                    volume=vol,
                    is_fixed=is_fixed,
                )
            else:
                abd.apply_to(
                    geo,
                    kappa=self._entity_config["abd_kappa"],
                    mass_density=1e3,
                    is_fixed=is_fixed,
                )

            rep_link = link_by_idx[rep]
            slot = self._scene.geometries.create(rep_link.name, geo)
            return slot
        elif total_mass > 0:
            vol = total_mass / 1e3
            geo = abd.create_proxy(
                kappa=self._entity_config["abd_kappa"],
                mass=total_mass,
                center_of_mass=com_local,
                inertia=I_local,
                volume=vol,
            )
            if is_fixed:
                geo.instances["is_fixed"] = np.array([1], dtype=np.int32)
            geo.instances["transform"] = T_rep.reshape(1, 4, 4)

            rep_link = link_by_idx[rep]
            slot = self._scene.geometries.create(rep_link.name, geo)
            return slot

        return None

    @staticmethod
    def _merge_inertials(entity, members, T_world):
        """Combine inertials of multiple links using parallel axis theorem.

        Returns (total_mass, com_world, inertia_world_at_com).
        """
        link_by_idx = {link.idx_local: link for link in entity.links}

        total_mass = 0.0
        weighted_com = np.zeros(3, dtype=np.float64)
        entries = []

        for m_idx in members:
            link = link_by_idx[m_idx]
            if link.inertial_mass is None or link.inertial_mass <= 0:
                continue

            m = float(link.inertial_mass)
            T_link = T_world[m_idx]

            inertial_pos = np.array(link.inertial_pos, dtype=np.float64)
            com_link_h = np.array([*inertial_pos, 1.0], dtype=np.float64)
            com_world = (T_link @ com_link_h)[:3]

            R_link = T_link[:3, :3]
            R_inertial = gu.quat_to_R(np.array(link.inertial_quat, dtype=np.float64))
            R_world = R_link @ R_inertial
            I_principal = np.array(link.inertial_i, dtype=np.float64)
            I_world = R_world @ I_principal @ R_world.T

            entries.append((m, com_world, I_world))
            total_mass += m
            weighted_com += m * com_world

        if total_mass <= 0:
            return 0.0, np.zeros(3), np.zeros((3, 3))

        com = weighted_com / total_mass

        I_combined = np.zeros((3, 3), dtype=np.float64)
        for m, com_i, I_i in entries:
            d = com_i - com
            I_combined += I_i + m * (np.dot(d, d) * np.eye(3) - np.outer(d, d))

        return total_mass, com, I_combined

    def _resolve_joint_gains(self, joint) -> tuple[float, float]:
        """Resolve kp/kv: material-specified values override actuator gains override coupler defaults."""
        mat = self._entity.material
        mat_kp = getattr(mat, "qipc_default_kp", None)
        mat_kv = getattr(mat, "qipc_default_kv", None)

        if mat_kp is not None:
            kp = float(mat_kp)
        else:
            act_gain = getattr(joint, "dofs_act_gain", None)
            if act_gain is not None and len(act_gain) > 0 and float(act_gain[0]) > 0:
                kp = float(act_gain[0])
            else:
                kp = float(self._entity_config["default_kp"])

        if mat_kv is not None:
            kv = float(mat_kv) if not isinstance(mat_kv, str) else 0.0
        else:
            act_bias = getattr(joint, "dofs_act_bias", None)
            if act_bias is not None and len(act_bias) > 0 and len(act_bias[0]) >= 3 and float(-act_bias[0][2]) > 0:
                kv = float(-act_bias[0][2])
            else:
                default_kv = self._entity_config["default_kv"]
                kv = float(default_kv) if not isinstance(default_kv, str) else 0.0

        return kp, kv

    @staticmethod
    def _compute_merged_volume(entity, members):
        """Compute total volume for a merge group."""
        from qipc.solver.affine_body import compute_mesh_volume_trimesh

        link_by_idx = {link.idx_local: link for link in entity.links}
        total_vol = 0.0

        for m_idx in members:
            link = link_by_idx[m_idx]
            for geom in link.geoms:
                v = geom.init_verts.copy().astype(np.float64)
                R = gu.quat_to_R(np.array(geom.init_quat, dtype=np.float64))
                v = (R @ v.T).T + geom.init_pos
                f = geom.init_faces
                total_vol += abs(compute_mesh_volume_trimesh(v, f))

        return max(total_vol, 1e-12)

    @staticmethod
    def _make_link_to_child_T(child_link):
        """Build the 4x4 transform from parent link frame to child link frame origin."""
        T = np.eye(4, dtype=np.float64)
        T[:3, 3] = child_link.pos
        T[:3, :3] = gu.quat_to_R(np.array(child_link.quat, dtype=np.float64))
        return T

    def _compute_initial_transforms(self):
        """Compute world-frame 4x4 transforms for each link via FK at init_qpos."""
        entity = self._entity
        T_world = {}

        morph = entity.morph
        T_root = np.eye(4, dtype=np.float64)
        if hasattr(morph, "pos") and morph.pos is not None:
            T_root[:3, 3] = np.array(morph.pos, dtype=np.float64)
        if hasattr(morph, "quat") and morph.quat is not None:
            T_root[:3, :3] = gu.quat_to_R(np.array(morph.quat, dtype=np.float64))

        home_qpos = self._entity_config["home_qpos"] if hasattr(self, "_entity_config") else None
        init_qpos = home_qpos if home_qpos is not None else entity.init_qpos

        for link in entity.links:
            if link.parent_idx < 0:
                T_world[link.idx_local] = T_root.copy()
            else:
                parent_local = link.parent_idx - entity.link_start
                T_parent = T_world[parent_local]

                T_child_in_parent = np.eye(4, dtype=np.float64)
                T_child_in_parent[:3, 3] = link.pos
                T_child_in_parent[:3, :3] = gu.quat_to_R(np.array(link.quat, dtype=np.float64))

                T_joint = np.eye(4, dtype=np.float64)
                for joint in link.joints:
                    for d in range(joint.n_dofs):
                        dof_local = joint.dofs_idx_local[d]
                        if dof_local >= len(init_qpos):
                            gs.raise_exception(
                                f"QIPCCoupler: joint '{joint.name}' DOF index {dof_local} "
                                f"exceeds init_qpos length {len(init_qpos)}."
                            )
                        theta = init_qpos[dof_local]
                        if joint.type == gs.JOINT_TYPE.REVOLUTE:
                            axis = np.array(joint.dofs_motion_ang[d], dtype=np.float64)
                            T_joint[:3, :3] = T_joint[:3, :3] @ _rodrigues(axis, float(theta))
                        elif joint.type == gs.JOINT_TYPE.PRISMATIC:
                            axis = joint.dofs_motion_vel[d]
                            T_joint[:3, 3] += axis * theta

                T_world[link.idx_local] = T_parent @ T_child_in_parent @ T_joint

        return T_world
