from __future__ import annotations

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
    """3x3 matrix → quaternion (w,x,y,z) via Shepperd's method. No SVD needed for high-kappa ABD."""
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


@qd.kernel(fastcache=True)
def _kernel_qipc_writeback(
    abd_q: qd.types.ndarray(),
    body_indices: qd.types.ndarray(),
    link_indices: qd.types.ndarray(),
    dofs_pos: qd.types.ndarray(),
    dof_indices: qd.types.ndarray(),
    links_state: array_class.LinksState,
    dofs_state: array_class.DofsState,
):
    """Single kernel: ABD q → pos/quat (skip SVD, high kappa), joint theta → dofs, zero vel."""
    n_links = link_indices.shape[0]
    n_dofs = dof_indices.shape[0]

    for i in range(n_links):
        body_idx = body_indices[i]
        link_idx = link_indices[i]

        # pos = q[0:3]
        links_state.pos[link_idx, 0][0] = abd_q[body_idx, 0]
        links_state.pos[link_idx, 0][1] = abd_q[body_idx, 1]
        links_state.pos[link_idx, 0][2] = abd_q[body_idx, 2]

        # A = q[3:12] treated directly as rotation (valid for high kappa)
        w, x, y, z = _func_mat3_to_quat(
            abd_q[body_idx, 3], abd_q[body_idx, 4], abd_q[body_idx, 5],
            abd_q[body_idx, 6], abd_q[body_idx, 7], abd_q[body_idx, 8],
            abd_q[body_idx, 9], abd_q[body_idx, 10], abd_q[body_idx, 11],
        )
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

    def build(self):
        from qipc import Scene as QIPCScene, trimesh
        from qipc.constitution import AffineBodyConstitution

        assert self._sim.n_envs <= 1, "QIPCCoupler: n_envs > 1 not supported"

        entity = self._sim.rigid_solver.entities[0]
        self._entity = entity

        self._scene = QIPCScene(
            dt=self._sim.dt,
            gravity=tuple(self._sim._gravity),
            **{"contact/enable": self._options.contact_enable},
        )

        abd = AffineBodyConstitution()

        T_world = self._compute_initial_transforms()

        self._link_slots = {}
        for link in entity.links:
            verts, faces = self._collect_link_mesh(link)
            if verts is None:
                continue
            geo = trimesh(verts, faces)
            geo.instances.resize(1)
            geo.transforms = T_world[link.idx_local].reshape(1, 4, 4)

            is_base_fixed = (link.parent_idx < 0 and getattr(entity.morph, "fixed", False))
            if link.inertial_mass is None or link.inertial_mass <= 0:
                gs.raise_exception(
                    f"QIPCCoupler: link '{link.name}' has invalid mass ({link.inertial_mass}). "
                    "All links must have positive mass."
                )
            if link.inertial_pos is None:
                gs.raise_exception(f"QIPCCoupler: link '{link.name}' has no center_of_mass defined.")
            vol = self._compute_link_volume(link)
            inertia_link_frame = rotate_inertia_to_link_frame(link.inertial_i, link.inertial_quat)

            abd.apply_to(
                geo,
                kappa=self._options.rigid_abd_kappa,
                mass=link.inertial_mass,
                center_of_mass=link.inertial_pos,
                inertia=inertia_link_frame,
                volume=vol,
                is_fixed=is_base_fixed,
            )
            self._link_slots[link.idx_local] = self._scene.geometries.create(link.name, geo)

        # Create joints one-by-one, then merge into single JointCollection
        from qipc.scene.joint_collection import JointCollection

        per_joint_jcs = []
        self._genesis_dof_order = []  # entity-local DOF index for each joint (creation order)

        for joint in entity.joints:
            child_link = joint.link
            parent_local = child_link.parent_idx - entity.link_start
            child_local = child_link.idx_local

            if joint.type == gs.JOINT_TYPE.REVOLUTE:
                jtype = "revolute"
                axis_local = joint.dofs_motion_ang[0]
                extra_kwargs = {"kappa_pivot": self._options.joint_kappa_pivot}
            elif joint.type == gs.JOINT_TYPE.PRISMATIC:
                jtype = "prismatic"
                axis_local = joint.dofs_motion_vel[0]
                extra_kwargs = {"kappa_lateral": self._options.joint_kappa_pivot}
            else:
                gs.raise_exception(
                    f"QIPCCoupler: unsupported joint type {joint.type} for joint '{joint.name}'."
                )

            # Anchor/axis in each body's rest frame (= link local frame)
            # anchor_left: joint pivot in parent link local frame = child_link.pos
            # anchor_right: joint pivot in child link local frame = joint.pos (usually [0,0,0])
            # axis_left: axis in parent local frame = R_child_in_parent @ axis_local
            # axis_right: axis in child local frame = axis_local
            R_child_in_parent = gu.quat_to_R(np.array(child_link.quat, dtype=np.float64))
            axis_in_parent = R_child_in_parent @ np.array(axis_local, dtype=np.float64)

            jc = self._scene.add_joint(
                joint.name,
                type=jtype,
                left=self._link_slots[parent_local],
                right=self._link_slots[child_local],
                anchor_left=child_link.pos.tolist(),
                anchor_right=joint.pos.tolist(),
                axis_left=axis_in_parent.tolist(),
                axis_right=np.array(axis_local, dtype=np.float64).tolist(),
                kappa_axis=self._options.joint_kappa_axis,
                enable_controller=True,
                kp=self._options.default_kp,
                kv=self._options.default_kv,
                theta_lower=float(joint.dofs_limit[0, 0]),
                theta_upper=float(joint.dofs_limit[0, 1]),
                **extra_kwargs,
            )
            per_joint_jcs.append(jc)
            self._genesis_dof_order.append(joint.dofs_idx_local[0])

        # Merge into single JC (invalidates per-joint JCs)
        self._jc = JointCollection.merge(per_joint_jcs) if per_joint_jcs else None

        # Convert DOF order to GPU tensor for indexed ops
        self._genesis_dof_order = torch.tensor(
            self._genesis_dof_order, dtype=torch.int64, device=gs.device
        )

        self._scene.init()

        # Build body index mapping (after init)
        self._link_body_indices = {}
        for link in entity.links:
            slot = self._link_slots[link.idx_local]
            body_offset = int(slot.geometry.meta["abd_body_offset"].cpu()[0])
            self._link_body_indices[link.idx_local] = body_offset

        # Pre-compute GPU index tensors for kernel calls (allocated once, reused every frame)
        n_links = len(entity.links)
        link_idx_list = [entity.link_start + link.idx_local for link in entity.links]
        self._link_indices_t = torch.tensor(link_idx_list, dtype=torch.int32, device=gs.device)
        self._body_indices_t = torch.tensor(
            [self._link_body_indices[link.idx_local] for link in entity.links],
            dtype=torch.int64, device=gs.device,
        )

        n_dofs = entity.n_dofs
        dof_idx_list = []
        for joint in entity.joints:
            for d in range(joint.n_dofs):
                dof_idx_list.append(entity.dof_start + joint.dofs_idx_local[d])
        self._dof_indices_t = torch.tensor(dof_idx_list, dtype=torch.int32, device=gs.device)

        # Pre-allocate dofs writeback buffer on GPU
        self._wb_dofs_pos = torch.zeros(n_dofs, dtype=gs.tc_float, device=gs.device)



        # QIPC debug viewer (independent window alongside Genesis viewer)
        self._debug_viewer = None
        if self._options.debug_viewer:
            self._debug_viewer = self._scene.viewer
            self._debug_viewer.up_axis = "z"

        self._substep_count = 0
        self._substeps_per_step = self._sim._substeps
        self._skip_first_step = True  # Genesis runs one step during build() for kernel compilation

    def reset(self, envs_idx=None):
        pass

    def preprocess(self, f):
        """Read ctrl_pos from Genesis, forward to merged QIPC JointCollection."""
        if self._jc is None:
            return
        ctrl_pos_view = qd_to_torch(self._sim.rigid_solver.dofs_state.ctrl_pos)
        entity_ctrl = ctrl_pos_view[self._entity.dof_start:self._entity.dof_end, 0].to(torch.float64)
        # Reorder Genesis DOFs to QIPC joint creation order and set all at once
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
    # State writeback — all GPU, no host transfer
    # -------------------------------------------------------------------------

    def _writeback_state(self):
        """Write QIPC state → Genesis buffers. Single kernel for links, batch theta readback."""
        abd_q = self._scene.affine_body.q  # (n_bodies, 12) GPU tensor, float64

        # Read all joint theta from merged JC (single indexed read from solver tensor)
        if self._jc is not None:
            theta = self._jc.get_dofs_position()  # (n_dofs,) GPU, creation order
            # Scatter into Genesis DOF order
            self._wb_dofs_pos[self._genesis_dof_order] = theta.to(gs.tc_float)

        # Single kernel: ABD q → pos/quat + dofs writeback
        _kernel_qipc_writeback(
            abd_q=abd_q,
            body_indices=self._body_indices_t,
            link_indices=self._link_indices_t,
            dofs_pos=self._wb_dofs_pos,
            dof_indices=self._dof_indices_t,
            links_state=self._sim.rigid_solver.links_state,
            dofs_state=self._sim.rigid_solver.dofs_state,
        )

    # -------------------------------------------------------------------------
    # Build-time helpers (run once, can use host)
    # -------------------------------------------------------------------------

    @staticmethod
    def _compute_link_volume(link):
        """Compute link volume with priority: visual meshes > collision meshes > 1.0 m^3 default."""
        from qipc.solver.affine_body import compute_mesh_volume_trimesh

        # Priority 1: visual meshes
        if hasattr(link, "vgeoms") and len(link.vgeoms) > 0:
            total_vol = 0.0
            for vgeom in link.vgeoms:
                v = vgeom.init_vverts.copy().astype(np.float64)
                R = gu.quat_to_R(np.array(vgeom.init_quat, dtype=np.float64))
                v = (R @ v.T).T + vgeom.init_pos
                f = vgeom.init_vfaces
                total_vol += abs(compute_mesh_volume_trimesh(v, f))
            if total_vol > 1e-12:
                return total_vol

        # Priority 2: collision meshes
        if hasattr(link, "geoms") and len(link.geoms) > 0:
            total_vol = 0.0
            for geom in link.geoms:
                v = geom.init_verts.copy().astype(np.float64)
                R = gu.quat_to_R(np.array(geom.init_quat, dtype=np.float64))
                v = (R @ v.T).T + geom.init_pos
                f = geom.init_faces
                total_vol += abs(compute_mesh_volume_trimesh(v, f))
            if total_vol > 1e-12:
                return total_vol

        # Priority 3: no mesh available
        gs.warn(f"QIPCCoupler: link '{link.name}' has no collision geometry. Using default volume 1.0m^3.")
        return 1.0 

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

        init_qpos = entity.init_qpos

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
                            T_joint[:3, :3] = T_joint[:3, :3] @ gu.axis_angle_to_R(axis, np.array(theta))
                        elif joint.type == gs.JOINT_TYPE.PRISMATIC:
                            axis = joint.dofs_motion_vel[d]
                            T_joint[:3, 3] += axis * theta

                T_world[link.idx_local] = T_parent @ T_child_in_parent @ T_joint

        return T_world

    def _collect_link_mesh(self, link):
        """Union all collision geoms for a link into link-frame vertices and faces."""
        all_verts = []
        all_faces = []
        offset = 0

        for geom in link.geoms:
            v = geom.init_verts.copy().astype(np.float64)
            R = gu.quat_to_R(np.array(geom.init_quat, dtype=np.float64))
            v = (R @ v.T).T + geom.init_pos
            all_verts.append(v)
            all_faces.append(geom.init_faces.copy() + offset)
            offset += len(v)

        if len(all_verts) == 0:
            gs.raise_exception(
                f"QIPCCoupler: link '{link.name}' has no collision geometry. "
                "All links must have at least one collision geom for QIPC."
            )

        return np.concatenate(all_verts, axis=0), np.concatenate(all_faces, axis=0)

