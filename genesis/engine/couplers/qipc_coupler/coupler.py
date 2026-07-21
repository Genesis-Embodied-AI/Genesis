from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import numpy as np
import quadrants as qd
import torch

import genesis as gs
import genesis.utils.array_class as array_class
import genesis.utils.geom as gu
from genesis.repr_base import RBC
from genesis.utils.misc import qd_to_torch


if TYPE_CHECKING:
    from genesis.engine.entities.rigid_entity.rigid_entity import RigidEntity
    from genesis.engine.entities.rigid_entity.rigid_joint import RigidJoint
    from genesis.engine.entities.rigid_entity.rigid_link import RigidLink
    from genesis.engine.simulator import Simulator
    from genesis.options.solvers import QIPCCouplerOptions

    from qipc.scene.joint_collection import JointCollection
    from qipc.scene.scene import Scene as QIPCScene


# ---------------------------------------------------------------------------
# Strong-typed data structures (no plain dict)
# ---------------------------------------------------------------------------


class EntityConfig(NamedTuple):
    """Per-entity QIPC configuration derived from material fields."""

    abd_kappa: float
    kappa_pivot: float
    kappa_axis: float
    default_kp: float
    default_kv: float
    home_qpos: tuple[float, ...] | None


class FreeBaseEntry(NamedTuple):
    """Tracks a free-base entity for qpos writeback."""

    entity: RigidEntity
    body_offset: int


class AbdEntityPreInit(NamedTuple):
    """Per-entity pre-init results: geometry/joints created, body offsets unresolved."""

    entity: RigidEntity
    group_slots: dict[int, object]
    link_to_rep: dict[int, int]
    T_world: dict[int, np.ndarray]
    joint_collections: list[JointCollection]
    genesis_dof_indices: list[int]
    is_free_base: bool


# ---------------------------------------------------------------------------
# Rodrigues rotation
# ---------------------------------------------------------------------------


def _rodrigues(axis: np.ndarray, theta: float) -> np.ndarray:
    """Rodrigues rotation: sign-preserving axis-angle to 3x3 matrix.

    Computes sin(theta) directly so negative angles rotate correctly.
    """
    axis = axis / np.linalg.norm(axis)
    c = np.cos(theta)
    s = np.sin(theta)
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]], dtype=np.float64)
    return np.eye(3, dtype=np.float64) * c + (1 - c) * np.outer(axis, axis) + s * K


# ---------------------------------------------------------------------------
# Quadrants kernels
# ---------------------------------------------------------------------------


@qd.func
def _func_mat3_to_quat(
    r00: gs.qd_float, r01: gs.qd_float, r02: gs.qd_float,
    r10: gs.qd_float, r11: gs.qd_float, r12: gs.qd_float,
    r20: gs.qd_float, r21: gs.qd_float, r22: gs.qd_float,
):
    """3x3 matrix -> quaternion (w,x,y,z) via Shepperd's method."""
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
    dofs_vel: qd.types.ndarray(),
    dof_indices: qd.types.ndarray(),
    free_base_body_indices: qd.types.ndarray(),
    free_base_link_indices: qd.types.ndarray(),
    free_base_q_starts: qd.types.ndarray(),
    links_state: array_class.LinksState,
    dofs_state: array_class.DofsState,
    rigid_info: array_class.RigidInfo,
):
    """Single-kernel writeback: ABD q -> links_state + dofs_state + free-base qpos.

    All state derived from ABD body transforms (first-class truth) in one launch.
    """
    n_links = link_indices.shape[0]
    n_dofs = dof_indices.shape[0]
    n_free = free_base_body_indices.shape[0]

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
        dofs_state.vel[idx, 0] = dofs_vel[i]

    for i in range(n_free):
        i_b = free_base_body_indices[i]
        link_idx = free_base_link_indices[i]
        q_start = free_base_q_starts[i]
        # pos: read from links_state (already written above)
        rigid_info.qpos[q_start, 0] = links_state.pos[link_idx, 0][0]
        rigid_info.qpos[q_start + 1, 0] = links_state.pos[link_idx, 0][1]
        rigid_info.qpos[q_start + 2, 0] = links_state.pos[link_idx, 0][2]
        # quat: read from links_state (already written above)
        rigid_info.qpos[q_start + 3, 0] = links_state.quat[link_idx, 0][0]
        rigid_info.qpos[q_start + 4, 0] = links_state.quat[link_idx, 0][1]
        rigid_info.qpos[q_start + 5, 0] = links_state.quat[link_idx, 0][2]
        rigid_info.qpos[q_start + 6, 0] = links_state.quat[link_idx, 0][3]


# ---------------------------------------------------------------------------
# QIPCCoupler
# ---------------------------------------------------------------------------


class QIPCCoupler(RBC):
    """
    QIPC coupler: uses cuda-graph-qipc as the sole physics engine for rigid/FEM entities.

    Genesis provides all scene data (link meshes, mass/inertia, joint topology);
    QIPC receives structured data and runs the physics. No asset files are loaded by QIPC.

    Design invariant: ABD body transforms are the first-class truth. Joint dof and
    free-base qpos are derived products written back for observation only.
    """

    def __init__(self, simulator: Simulator, options: QIPCCouplerOptions) -> None:
        self._sim: Simulator = simulator
        self._options: QIPCCouplerOptions = options

    @property
    def sim(self) -> Simulator:
        return self._sim

    @property
    def options(self) -> QIPCCouplerOptions:
        return self._options

    # -------------------------------------------------------------------------
    # Build
    # -------------------------------------------------------------------------

    def build(self) -> None:
        from qipc import Scene as QIPCSceneCls
        from qipc import trimesh
        from qipc.constitution import AffineBodyConstitution
        from qipc.geometry import ground as qipc_ground
        from qipc.scene.joint_collection import JointCollection

        assert self._sim.n_envs <= 1, "QIPCCoupler: n_envs > 1 not supported"

        # --- Classify entities: plane vs abd ---
        all_entities: list[RigidEntity] = list(self._sim.rigid_solver.entities)
        plane_entities: list[RigidEntity] = []
        abd_entities: list[RigidEntity] = []
        for entity in all_entities:
            if self._is_plane_entity(entity):
                plane_entities.append(entity)
            else:
                abd_entities.append(entity)

        # --- Create QIPC Scene with contact config ---
        self._scene: QIPCScene = QIPCSceneCls(
            dt=self._sim.dt,
            gravity=tuple(self._sim._gravity),
            **{
                "contact/enable": self._options.contact_enable,
                "contact/d_hat": self._options.contact_d_hat,
                "contact/init_collision_pair_capacity": self._options.init_collision_pair_capacity,
            },
        )

        # --- Global default contact model (per-pair tabular deferred) ---
        # ContactTabular already initializes default (friction_rate=0.05, resistance=1e4)
        # which is reasonable for generic rigid contact. No override needed for now.

        # --- Ground planes ---
        for entity in plane_entities:
            self._create_ground(entity, qipc_ground)

        # --- ABD entities: phase 1 (pre-init, create geometry/joints) ---
        abd = AffineBodyConstitution()
        all_pre_inits: list[AbdEntityPreInit] = []
        for entity in abd_entities:
            pre = self._build_abd_entity_pre_init(entity, abd, trimesh, JointCollection)
            all_pre_inits.append(pre)

        # --- Aggregate JointCollections and dof order (pre-init) ---
        all_jcs: list[JointCollection] = []
        all_genesis_dof_indices: list[int] = []
        for pre in all_pre_inits:
            all_jcs.extend(pre.joint_collections)
            all_genesis_dof_indices.extend(pre.genesis_dof_indices)

        self._jc: JointCollection | None = (
            JointCollection.merge(all_jcs) if all_jcs else None
        )
        self._genesis_dof_order: torch.Tensor = torch.tensor(
            all_genesis_dof_indices, dtype=torch.int64, device=gs.device
        )

        # --- Init QIPC (assigns abd_body_offset to each geometry) ---
        self._scene.init()

        # --- Phase 2 (post-init): resolve body offsets, build writeback tensors ---
        all_link_indices: list[int] = []
        all_body_indices: list[int] = []
        all_rel_transforms: list[np.ndarray] = []
        free_base_entries: list[FreeBaseEntry] = []

        for pre in all_pre_inits:
            link_indices, body_indices, rel_transforms, free_entry = (
                self._resolve_post_init(pre)
            )
            all_link_indices.extend(link_indices)
            all_body_indices.extend(body_indices)
            all_rel_transforms.extend(rel_transforms)
            if free_entry is not None:
                free_base_entries.append(free_entry)

        self._free_base_entries: list[FreeBaseEntry] = free_base_entries

        # --- Build GPU tensors for writeback ---
        self._link_indices_t: torch.Tensor = torch.tensor(
            all_link_indices, dtype=torch.int32, device=gs.device
        )
        self._body_indices_t: torch.Tensor = torch.tensor(
            all_body_indices, dtype=torch.int64, device=gs.device
        )

        rel_data = np.zeros((len(all_rel_transforms), 12), dtype=np.float64)
        for i, rt in enumerate(all_rel_transforms):
            rel_data[i] = rt
        self._rel_transforms_t: torch.Tensor = torch.tensor(
            rel_data, dtype=torch.float64, device=gs.device
        )

        n_controlled_dofs = len(all_genesis_dof_indices)
        self._dof_indices_t: torch.Tensor = torch.tensor(
            all_genesis_dof_indices, dtype=torch.int32, device=gs.device
        )
        self._wb_dofs_pos: torch.Tensor = torch.zeros(
            n_controlled_dofs, dtype=gs.tc_float, device=gs.device
        )
        self._wb_dofs_vel: torch.Tensor = torch.zeros(
            n_controlled_dofs, dtype=gs.tc_float, device=gs.device
        )
        self._prev_theta: torch.Tensor = torch.zeros(
            n_controlled_dofs, dtype=torch.float64, device="cuda"
        )

        # Free-base tensors for unified kernel writeback
        fb_body_indices = [e.body_offset for e in free_base_entries]
        fb_link_indices = [e.entity.link_start for e in free_base_entries]
        fb_q_starts = [e.entity.q_start for e in free_base_entries]
        self._free_base_body_indices_t: torch.Tensor = torch.tensor(
            fb_body_indices or [0], dtype=torch.int64, device=gs.device
        )[:len(free_base_entries)]
        self._free_base_link_indices_t: torch.Tensor = torch.tensor(
            fb_link_indices or [0], dtype=torch.int32, device=gs.device
        )[:len(free_base_entries)]
        self._free_base_q_starts_t: torch.Tensor = torch.tensor(
            fb_q_starts or [0], dtype=torch.int32, device=gs.device
        )[:len(free_base_entries)]

        # --- Debug viewer ---
        self._debug_viewer = None
        if self._options.debug_viewer:
            self._debug_viewer = self._scene.viewer
            self._debug_viewer.up_axis = "z"

        # --- Substep tracking ---
        self._substep_count: int = 0
        self._substeps_per_step: int = self._sim._substeps
        self._is_first_step: bool = True

        # --- Initial writeback ---
        self._writeback_state()

    # -------------------------------------------------------------------------
    # Runtime
    # -------------------------------------------------------------------------

    def reset(self, envs_idx=None) -> None:
        # At build time, Genesis calls reset() to initialize state. Since QIPC
        # was just built from the same initial conditions, this is a no-op.
        # Mid-simulation reset (restoring to a prior state) is not yet supported.
        pass

    def preprocess(self, f: int) -> None:
        """Forward Genesis control targets to QIPC joint controller.

        Unconditionally forwards both position and velocity targets. The
        effective control mode is determined by the gain settings (kp/kv):
        - Position control: kp > 0, target_velocity = 0
        - Velocity control: kp = 0, target_velocity = user target
        - Force control: kp = 0, target_velocity = 0, use control_dofs_force
        """
        if self._jc is None:
            return

        dofs_state = self._sim.rigid_solver.dyn_state.dofs
        ctrl_pos_all: torch.Tensor = qd_to_torch(dofs_state.ctrl_pos)[:, 0].to(torch.float64)
        ctrl_vel_all: torch.Tensor = qd_to_torch(dofs_state.ctrl_vel)[:, 0].to(torch.float64)

        pos_targets: torch.Tensor = ctrl_pos_all[self._genesis_dof_order]
        vel_targets: torch.Tensor = ctrl_vel_all[self._genesis_dof_order]

        self._jc.control_dofs_position(pos_targets)
        self._jc.control_dofs_velocity(vel_targets)

    def couple(self, f: int) -> None:
        self._substep_count += 1
        if self._substep_count < self._substeps_per_step:
            return
        self._substep_count = 0

        if self._is_first_step:
            # Genesis calls couple() once at the end of build() as part of its
            # initialization sequence. QIPC must not step on this call because
            # the scene was just initialized and no user control has been applied.
            self._is_first_step = False
            return

        self._scene.step()

        self._writeback_state()

        gs.logger.debug(
            f"[QIPC] sim={self._scene.solver.step_ms:.2f}ms "
            f"newton={self._scene.solver.newton_iters} "
            f"pcg={self._scene.solver.max_pcg_iters} "
            f"ls={self._scene.solver.max_ls_iters}"
        )

        if self._debug_viewer is not None:
            self._debug_viewer.show()

    def couple_grad(self, f: int) -> None:
        pass

    # -------------------------------------------------------------------------
    # State writeback (transform is first-class truth)
    # -------------------------------------------------------------------------

    def _writeback_state(self) -> None:
        """Write QIPC state -> Genesis buffers in a single kernel launch.

        All state derives from ABD body transforms (first-class truth):
        links_state.pos/quat, dofs_state.pos, dofs_state.vel, and free-base qpos.
        Joint velocity is finite-differenced from theta.
        """
        abd_q: torch.Tensor = self._scene.affine_body.q

        if self._jc is not None:
            theta: torch.Tensor = self._jc.get_dofs_position()
            self._wb_dofs_pos[:] = theta.to(gs.tc_float)
            # Finite-difference velocity: (theta - theta_prev) / dt
            self._wb_dofs_vel[:] = ((theta - self._prev_theta) / self._sim.dt).to(gs.tc_float)
            self._prev_theta[:] = theta

        _kernel_qipc_writeback(
            abd_q=abd_q,
            body_indices=self._body_indices_t,
            link_indices=self._link_indices_t,
            rel_transforms=self._rel_transforms_t,
            dofs_pos=self._wb_dofs_pos,
            dofs_vel=self._wb_dofs_vel,
            dof_indices=self._dof_indices_t,
            free_base_body_indices=self._free_base_body_indices_t,
            free_base_link_indices=self._free_base_link_indices_t,
            free_base_q_starts=self._free_base_q_starts_t,
            links_state=self._sim.rigid_solver.dyn_state.links,
            dofs_state=self._sim.rigid_solver.dyn_state.dofs,
            rigid_info=self._sim.rigid_solver.rigid_info,
        )

    # -------------------------------------------------------------------------
    # Build helpers: entity classification
    # -------------------------------------------------------------------------

    @staticmethod
    def _is_plane_entity(entity: RigidEntity) -> bool:
        """Check if any geom in the entity is a plane."""
        for link in entity.links:
            for geom in link.geoms:
                if geom.type == gs.GEOM_TYPE.PLANE:
                    return True
        return False

    @staticmethod
    def _get_entity_config(entity: RigidEntity) -> EntityConfig:
        """Read per-entity QIPC config from material fields."""
        mat = entity.material
        return EntityConfig(
            abd_kappa=mat.qipc_abd_kappa or 1e8,
            kappa_pivot=mat.qipc_kappa_pivot or 1e8,
            kappa_axis=mat.qipc_kappa_axis or 1e8,
            default_kp=mat.qipc_default_kp or 100.0,
            default_kv=mat.qipc_default_kv or 10.0,
            home_qpos=tuple(mat.qipc_home_qpos) if mat.qipc_home_qpos is not None else None,
        )

    # -------------------------------------------------------------------------
    # Build helpers: ground
    # -------------------------------------------------------------------------

    def _create_ground(self, entity: RigidEntity, ground_factory) -> None:
        """Convert a Genesis Plane entity to a QIPC half-plane ground."""
        for link in entity.links:
            for geom in link.geoms:
                if geom.type != gs.GEOM_TYPE.PLANE:
                    continue
                local_normal: np.ndarray = geom.data[:3].astype(np.float64, copy=False)
                R_geom: np.ndarray = gu.quat_to_R(np.array(geom.init_quat, dtype=np.float64))
                normal: np.ndarray = R_geom @ local_normal
                n_len = np.linalg.norm(normal)
                if n_len < 1e-15:
                    continue
                normal = normal / n_len
                height: float = float(np.dot(np.array(geom.init_pos, dtype=np.float64), normal))
                geo = ground_factory(height=height, N=tuple(normal))
                self._scene.geometries.create(f"ground_{entity.idx}", geo)

    # -------------------------------------------------------------------------
    # Build helpers: per-entity ABD construction
    # -------------------------------------------------------------------------

    def _build_abd_entity_pre_init(
        self,
        entity: RigidEntity,
        abd: object,
        trimesh_factory: object,
        joint_collection_cls: type,
    ) -> AbdEntityPreInit:
        """Phase 1: create QIPC geometry/joints for one entity (before scene.init)."""
        cfg: EntityConfig = self._get_entity_config(entity)

        T_world: dict[int, np.ndarray] = self._compute_initial_transforms(entity, cfg)
        merge_groups, link_to_rep = self._build_merge_groups(entity)

        group_slots: dict[int, object] = {}

        for rep, members in merge_groups:
            slot = self._create_merged_body(
                entity, rep, members, T_world, abd, trimesh_factory,
                abd_kappa=cfg.abd_kappa,
            )
            if slot is None:
                continue
            group_slots[rep] = slot

        # --- Classify joints by type ---
        revolute_joints: list[RigidJoint] = []
        prismatic_joints: list[RigidJoint] = []
        is_free_base: bool = False

        for joint in entity.joints:
            if joint.type == gs.JOINT_TYPE.FIXED:
                continue
            elif joint.type == gs.JOINT_TYPE.REVOLUTE:
                revolute_joints.append(joint)
            elif joint.type == gs.JOINT_TYPE.PRISMATIC:
                prismatic_joints.append(joint)
            elif joint.type == gs.JOINT_TYPE.FREE:
                is_free_base = True

        # Also detect free-base from non-fixed root link without FREE joint
        if not is_free_base:
            base_link = entity.links[0]
            if not base_link.is_fixed:
                is_free_base = True

        # --- Build joints (unified per type) ---
        per_joint_jcs: list[JointCollection] = []
        genesis_dof_indices: list[int] = []

        init_qpos = cfg.home_qpos if cfg.home_qpos is not None else entity.init_qpos

        for joint in revolute_joints:
            jc, dof_idx = self._create_joint(
                entity, joint, "revolute", link_to_rep, group_slots,
                T_world, cfg, init_qpos,
            )
            if jc is not None:
                per_joint_jcs.append(jc)
                genesis_dof_indices.append(dof_idx)

        for joint in prismatic_joints:
            jc, dof_idx = self._create_joint(
                entity, joint, "prismatic", link_to_rep, group_slots,
                T_world, cfg, init_qpos,
            )
            if jc is not None:
                per_joint_jcs.append(jc)
                genesis_dof_indices.append(dof_idx)

        return AbdEntityPreInit(
            entity=entity,
            group_slots=group_slots,
            link_to_rep=link_to_rep,
            T_world=T_world,
            joint_collections=per_joint_jcs,
            genesis_dof_indices=genesis_dof_indices,
            is_free_base=is_free_base,
        )

    @staticmethod
    def _resolve_post_init(
        pre: AbdEntityPreInit,
    ) -> tuple[list[int], list[int], list[np.ndarray], FreeBaseEntry | None]:
        """Phase 2: after scene.init(), resolve abd_body_offset and build writeback arrays."""
        entity = pre.entity
        group_slots = pre.group_slots
        link_to_rep = pre.link_to_rep
        T_world = pre.T_world

        # Resolve link -> body offset mapping (abd_body_offset exists after init)
        link_body_indices: dict[int, int] = {}
        for link in entity.links:
            rep = link_to_rep[link.idx_local]
            if rep not in group_slots:
                continue
            slot = group_slots[rep]
            body_offset = int(slot.geometry.meta["abd_body_offset"].cpu()[0])
            link_body_indices[link.idx_local] = body_offset

        # Free-base entry
        free_entry: FreeBaseEntry | None = None
        if pre.is_free_base:
            base_link = entity.links[0]
            base_rep = link_to_rep.get(base_link.idx_local, base_link.idx_local)
            if base_rep in group_slots:
                slot = group_slots[base_rep]
                body_offset = int(slot.geometry.meta["abd_body_offset"].cpu()[0])
                free_entry = FreeBaseEntry(entity=entity, body_offset=body_offset)

        # Build global-index arrays for writeback
        active_links: list[RigidLink] = [
            link for link in entity.links if link.idx_local in link_body_indices
        ]
        link_indices: list[int] = []
        body_indices: list[int] = []
        rel_transforms_list: list[np.ndarray] = []

        for link in active_links:
            global_link_idx = entity.link_start + link.idx_local
            link_indices.append(global_link_idx)
            body_indices.append(link_body_indices[link.idx_local])

            rep = link_to_rep[link.idx_local]
            if link.idx_local == rep:
                rt = np.zeros(12, dtype=np.float64)
                rt[3] = 1.0
                rt[7] = 1.0
                rt[11] = 1.0
            else:
                T_rep = T_world[rep]
                T_member = T_world[link.idx_local]
                T_rel = np.linalg.inv(T_rep) @ T_member
                rt = np.zeros(12, dtype=np.float64)
                rt[0:3] = T_rel[:3, 3]
                rt[3:12] = T_rel[:3, :3].flatten()
            rel_transforms_list.append(rt)

        return link_indices, body_indices, rel_transforms_list, free_entry

    def _create_joint(
        self,
        entity: RigidEntity,
        joint: RigidJoint,
        jtype: str,
        link_to_rep: dict[int, int],
        group_slots: dict[int, object],
        T_world: dict[int, np.ndarray],
        cfg: EntityConfig,
        init_qpos: tuple[float, ...] | np.ndarray,
    ) -> tuple[JointCollection | None, int]:
        """Create a single QIPC joint. Returns (JointCollection, global_dof_idx) or (None, -1)."""
        child_link: RigidLink = joint.link
        parent_local: int = child_link.parent_idx - entity.link_start
        child_local: int = child_link.idx_local

        parent_rep: int = link_to_rep[parent_local]
        child_rep: int = link_to_rep[child_local]

        if parent_rep == child_rep:
            return None, -1

        if parent_rep not in group_slots or child_rep not in group_slots:
            gs.logger.warning(
                f"QIPCCoupler: skipping joint '{joint.name}' -- "
                f"parent or child body not created."
            )
            return None, -1

        if jtype == "revolute":
            axis_local: np.ndarray = np.array(joint.dofs_motion_ang[0], dtype=np.float64)
            extra_kwargs = {"kappa_pivot": cfg.kappa_pivot}
        else:
            axis_local = np.array(joint.dofs_motion_vel[0], dtype=np.float64)
            extra_kwargs = {"kappa_lateral": cfg.kappa_pivot}

        T_parent_rep: np.ndarray = T_world[parent_rep]
        T_child_rep: np.ndarray = T_world[child_rep]
        T_parent_link: np.ndarray = T_world[parent_local]

        R_parent_rep_inv: np.ndarray = T_parent_rep[:3, :3].T
        R_child_rep_inv: np.ndarray = T_child_rep[:3, :3].T

        T_joint_world: np.ndarray = T_parent_link @ self._make_link_to_child_T(child_link)
        anchor_world: np.ndarray = T_joint_world[:3, 3]

        anchor_left: np.ndarray = R_parent_rep_inv @ (anchor_world - T_parent_rep[:3, 3])
        anchor_right: np.ndarray = R_child_rep_inv @ (anchor_world - T_child_rep[:3, 3])

        R_child_in_parent: np.ndarray = gu.quat_to_R(np.array(child_link.quat, dtype=np.float64))
        axis_world: np.ndarray = T_parent_link[:3, :3] @ R_child_in_parent @ axis_local
        axis_world = axis_world / np.linalg.norm(axis_world)

        axis_left: np.ndarray = R_parent_rep_inv @ axis_world
        axis_right: np.ndarray = R_child_rep_inv @ axis_world

        kp, kv = self._resolve_joint_gains(joint, entity)

        dof_local: int = joint.dofs_idx_local[0]
        global_dof_idx: int = entity.dof_start + dof_local

        jc: JointCollection = self._scene.add_joint(
            joint.name,
            type=jtype,
            left=group_slots[parent_rep],
            right=group_slots[child_rep],
            anchor_left=anchor_left.tolist(),
            anchor_right=anchor_right.tolist(),
            axis_left=axis_left.tolist(),
            axis_right=axis_right.tolist(),
            kappa_axis=cfg.kappa_axis,
            enable_controller=True,
            kp=kp,
            kv=kv,
            theta_lower=float(joint.dofs_limit[0, 0]),
            theta_upper=float(joint.dofs_limit[0, 1]),
            init_theta=float(init_qpos[dof_local]),
            **extra_kwargs,
        )
        return jc, global_dof_idx

    # -------------------------------------------------------------------------
    # Build helpers: merge groups, merged body, inertials, FK, joint gains
    # -------------------------------------------------------------------------

    @staticmethod
    def _build_merge_groups(entity: RigidEntity) -> tuple[list[tuple[int, list[int]]], dict[int, int]]:
        """Group links connected by fixed joints.

        Returns (groups, link_to_rep) where groups is a list of (rep, members)
        tuples and link_to_rep maps each idx_local to its group representative.
        """
        from collections import defaultdict

        fixed_adj: dict[int, list[int]] = defaultdict(list)

        for link in entity.links:
            for joint in link.joints:
                if joint.type == gs.JOINT_TYPE.FIXED:
                    parent_local = link.parent_idx - entity.link_start
                    fixed_adj[link.idx_local].append(parent_local)
                    fixed_adj[parent_local].append(link.idx_local)

        for link in entity.links:
            if link.parent_idx >= 0 and len(link.joints) == 0:
                parent_local = link.parent_idx - entity.link_start
                fixed_adj[link.idx_local].append(parent_local)
                fixed_adj[parent_local].append(link.idx_local)

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

        link_to_rep: dict[int, int] = {}
        for rep, members in groups:
            for m in members:
                link_to_rep[m] = rep

        return groups, link_to_rep

    def _create_merged_body(
        self,
        entity: RigidEntity,
        rep: int,
        members: list[int],
        T_world: dict[int, np.ndarray],
        abd: object,
        trimesh_factory: object,
        *,
        abd_kappa: float,
    ) -> object | None:
        """Create a single ABD body from a merge group. Returns geometry slot or None."""
        T_rep: np.ndarray = T_world[rep]
        T_rep_inv: np.ndarray = np.linalg.inv(T_rep)
        R_rep: np.ndarray = T_rep[:3, :3]
        t_rep: np.ndarray = T_rep[:3, 3]

        all_verts: list[np.ndarray] = []
        all_faces: list[np.ndarray] = []
        vert_offset: int = 0

        link_by_idx: dict[int, RigidLink] = {link.idx_local: link for link in entity.links}

        for m_idx in members:
            link = link_by_idx[m_idx]
            if len(link.geoms) == 0:
                continue

            T_member: np.ndarray = T_world[m_idx]

            for geom in link.geoms:
                v = geom.init_verts.copy().astype(np.float64)
                R_geom = gu.quat_to_R(np.array(geom.init_quat, dtype=np.float64))
                v = (R_geom @ v.T).T + geom.init_pos

                T_geom_in_rep = T_rep_inv @ T_member
                v_h = np.hstack([v, np.ones((len(v), 1))])
                v_rep = (T_geom_in_rep @ v_h.T).T[:, :3]

                all_verts.append(v_rep)
                all_faces.append(geom.init_faces.copy() + vert_offset)
                vert_offset += len(v_rep)

        total_mass, com_world, I_world = self._merge_inertials(entity, members, T_world)
        com_local: np.ndarray = R_rep.T @ (com_world - t_rep)
        I_local: np.ndarray = R_rep.T @ I_world @ R_rep

        is_fixed: bool = any(link_by_idx[m].is_fixed for m in members)

        if all_verts:
            merged_verts: np.ndarray = np.concatenate(all_verts, axis=0)
            merged_faces: np.ndarray = np.concatenate(all_faces, axis=0)

            geo = trimesh_factory(merged_verts, merged_faces)
            geo.instances.resize(1)
            geo.transforms = T_rep.reshape(1, 4, 4)

            if total_mass > 0:
                vol = self._compute_merged_volume(entity, members)
                abd.apply_to(
                    geo,
                    kappa=abd_kappa,
                    mass=total_mass,
                    center_of_mass=com_local,
                    inertia=I_local,
                    volume=vol,
                    is_fixed=is_fixed,
                )
            else:
                abd.apply_to(geo, kappa=abd_kappa, mass_density=1e3, is_fixed=is_fixed)

            rep_link = link_by_idx[rep]
            slot = self._scene.geometries.create(rep_link.name, geo)
            return slot
        elif total_mass > 0:
            vol = total_mass / 1e3
            geo = abd.create_proxy(
                kappa=abd_kappa,
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
    def _merge_inertials(
        entity: RigidEntity,
        members: list[int],
        T_world: dict[int, np.ndarray],
    ) -> tuple[float, np.ndarray, np.ndarray]:
        """Combine inertials of multiple links using parallel axis theorem.

        Returns (total_mass, com_world, inertia_world_at_com).
        """
        link_by_idx: dict[int, RigidLink] = {link.idx_local: link for link in entity.links}

        total_mass: float = 0.0
        weighted_com: np.ndarray = np.zeros(3, dtype=np.float64)
        entries: list[tuple[float, np.ndarray, np.ndarray]] = []

        for m_idx in members:
            link = link_by_idx[m_idx]
            if link.inertial_mass is None or link.inertial_mass <= 0:
                continue

            m = float(link.inertial_mass)
            T_link: np.ndarray = T_world[m_idx]

            inertial_pos: np.ndarray = np.array(link.inertial_pos, dtype=np.float64)
            com_link_h = np.array([*inertial_pos, 1.0], dtype=np.float64)
            com_world_pt: np.ndarray = (T_link @ com_link_h)[:3]

            R_link: np.ndarray = T_link[:3, :3]
            R_inertial: np.ndarray = gu.quat_to_R(np.array(link.inertial_quat, dtype=np.float64))
            R_world_inertial: np.ndarray = R_link @ R_inertial
            I_principal: np.ndarray = np.array(link.inertial_i, dtype=np.float64)
            I_world_link: np.ndarray = R_world_inertial @ I_principal @ R_world_inertial.T

            entries.append((m, com_world_pt, I_world_link))
            total_mass += m
            weighted_com += m * com_world_pt

        if total_mass <= 0:
            return 0.0, np.zeros(3), np.zeros((3, 3))

        com: np.ndarray = weighted_com / total_mass

        I_combined: np.ndarray = np.zeros((3, 3), dtype=np.float64)
        for m, com_i, I_i in entries:
            d = com_i - com
            I_combined += I_i + m * (np.dot(d, d) * np.eye(3) - np.outer(d, d))

        return total_mass, com, I_combined

    def _resolve_joint_gains(self, joint: RigidJoint, entity: RigidEntity) -> tuple[float, float]:
        """Resolve kp/kv: material > actuator gains > coupler defaults."""
        mat = entity.material
        mat_kp: float | None = mat.qipc_default_kp
        mat_kv = mat.qipc_default_kv

        if mat_kp is not None:
            kp = float(mat_kp)
        else:
            act_gain = joint.dofs_act_gain
            if act_gain is not None and len(act_gain) > 0 and float(act_gain[0]) > 0:
                kp = float(act_gain[0])
            else:
                kp = 100.0

        if mat_kv is not None and not isinstance(mat_kv, str):
            kv = float(mat_kv)
        else:
            act_bias = joint.dofs_act_bias
            if act_bias is not None and len(act_bias) > 0 and len(act_bias[0]) >= 3 and float(-act_bias[0][2]) > 0:
                kv = float(-act_bias[0][2])
            else:
                kv = 10.0

        return kp, kv

    @staticmethod
    def _compute_merged_volume(entity: RigidEntity, members: list[int]) -> float:
        """Compute total volume for a merge group."""
        from qipc.solver.affine_body import compute_mesh_volume_trimesh

        link_by_idx: dict[int, RigidLink] = {link.idx_local: link for link in entity.links}
        total_vol: float = 0.0

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
    def _make_link_to_child_T(child_link: RigidLink) -> np.ndarray:
        """Build the 4x4 transform from parent link frame to child link frame origin."""
        T = np.eye(4, dtype=np.float64)
        T[:3, 3] = child_link.pos
        T[:3, :3] = gu.quat_to_R(np.array(child_link.quat, dtype=np.float64))
        return T

    @staticmethod
    def _compute_initial_transforms(entity: RigidEntity, cfg: EntityConfig) -> dict[int, np.ndarray]:
        """Compute world-frame 4x4 transforms for each link via FK at init_qpos."""
        T_world: dict[int, np.ndarray] = {}

        morph = entity.morph
        T_root = np.eye(4, dtype=np.float64)
        T_root[:3, 3] = np.array(morph.pos, dtype=np.float64)
        if morph.quat is not None:
            T_root[:3, :3] = gu.quat_to_R(np.array(morph.quat, dtype=np.float64))

        init_qpos = cfg.home_qpos if cfg.home_qpos is not None else entity.init_qpos

        for link in entity.links:
            if link.parent_idx < 0:
                T_world[link.idx_local] = T_root.copy()
            else:
                parent_local: int = link.parent_idx - entity.link_start
                T_parent: np.ndarray = T_world[parent_local]

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
