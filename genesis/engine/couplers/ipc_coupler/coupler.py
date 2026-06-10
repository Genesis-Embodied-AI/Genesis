import io
import logging
import os
import re
import sys
import tempfile
import weakref
from functools import partial
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Callable, Protocol, cast

import numpy as np
import torch
import trimesh

import genesis as gs
import genesis.utils.geom as gu
from genesis.engine.entities.rigid_entity.rigid_link import RHO_MUJOCO, RHO_OBJECT, RHO_ROBOT
from genesis.engine.materials.FEM.cloth import Cloth
from genesis.engine.materials.FEM.paper import Paper
from genesis.engine.materials.FEM.rope import Rope
from genesis.options.solvers import IPCCouplerOptions, RigidOptions
from genesis.repr_base import RBC
from genesis.utils.mesh import are_meshes_overlapping
from genesis.utils.misc import geometric_mean, harmonic_mean, qd_to_numpy, qd_to_torch, tensor_to_array

if TYPE_CHECKING:
    from genesis.engine.entities import FEMEntity, RigidEntity
    from genesis.engine.entities.rigid_entity import RigidJoint, RigidLink
    from genesis.engine.simulator import Simulator
    from genesis.engine.solvers import FEMSolver, RigidSolver

# Check if libuipc is available
try:
    import uipc

    UIPC_AVAILABLE = True
except ImportError:
    UIPC_AVAILABLE = False

if TYPE_CHECKING or UIPC_AVAILABLE:
    import polyscope as ps
    from uipc.backend import SceneVisitor
    from uipc.constitution import (
        AffineBodyConstitution,
        AffineBodyShell,
        AffineBodyPrismaticJoint,
        AffineBodyRevoluteJoint,
        DiscreteShellBending,
        HookeanSpring,
        KirchhoffRodBending,
        StrainPlasticDiscreteShellBending,
        StressPlasticDiscreteShellBending,
        ElasticModuli,
        ElasticModuli2D,
        ExternalArticulationConstraint,
        SoftPositionConstraint,
        SoftTransformConstraint,
        StableNeoHookean,
        NeoHookeanShell,
        StrainLimitingBaraffWitkinShell,
    )

    # AerodynamicDamping was added in a private uipc fork and is not part of
    # every public release.  Try to import it; if it's missing we simply
    # disable the corresponding code path at runtime instead of failing
    # the whole module import.
    try:
        from uipc.constitution import AerodynamicDamping  # type: ignore[attr-defined]
    except ImportError:
        AerodynamicDamping = None  # type: ignore[assignment, misc]
    try:
        from uipc.constitution import StrainPlasticDiscreteShellBendingModifier  # type: ignore[attr-defined]
    except ImportError:
        StrainPlasticDiscreteShellBendingModifier = None  # type: ignore[assignment, misc]
    try:
        from uipc.constitution import StressPlasticDiscreteShellBendingModifier  # type: ignore[attr-defined]
    except ImportError:
        StressPlasticDiscreteShellBendingModifier = None  # type: ignore[assignment, misc]
    try:
        from uipc.constitution import FiniteElementExternalForce  # type: ignore[attr-defined]
    except ImportError:
        FiniteElementExternalForce = None  # type: ignore[assignment, misc]
    from uipc.core import (
        Engine,
        World,
        Scene,
        SceneIO,
        AffineBodyStateAccessorFeature,
        FiniteElementStateAccessorFeature,
        ContactElement,
        SubsceneElement,
    )
    from uipc.geometry import GeometrySlot, SimplicialComplex, SimplicialComplexSlot
    from uipc.gui import SceneGUI

    from .data import COUPLING_TYPE, ABDLinkData, ArticulatedEntityData
    from .utils import (
        build_ipc_scene_config,
        compute_link_to_link_transform,
        find_abd_merge_target,
        read_ipc_geometry_metadata,
    )


class IPCBeforeWorldInitContext(Protocol):
    """Typed context passed to before_ipc_world_init(ipc, gs)."""

    engine: "Engine"
    world: "World"
    scene: "Scene"


class GenesisSolverContext(Protocol):
    """Typed Genesis module view passed to before_ipc_world_init(ipc, gs)."""

    pass


IPCBeforeWorldInitCallback = Callable[[IPCBeforeWorldInitContext, GenesisSolverContext], None]


# Affine body stiffness in MPa
ABD_KAPPA = 100.0
# Position-space threshold for detecting active IPC contact (restitution tracking)
RESTITUTION_CONTACT_THRESHOLD = 1e-7
COM_AABB_TOL = 2e-3
IPC_SURFACE_PREFIX = "ipc_surface"
GENESIS_SURFACE_PREFIX = "genesis_surface"


def _combine_inertials(
    parent_link: "RigidLink",
    children: "list[RigidLink]",
) -> "tuple[float, np.ndarray, np.ndarray]":
    """Combine inertial properties of a parent link and its merged children.

    Uses the parallel axis theorem to shift each body's inertia tensor to the
    parent link's origin, then sums mass, computes combined COM, and shifts
    the total inertia tensor to the combined COM.

    Parameters
    ----------
    parent_link : RigidLink
        The target link whose frame defines the output.
    children : list[RigidLink]
        Fixed-joint child links being merged into *parent_link*.

    Returns
    -------
    tuple[float, np.ndarray, np.ndarray]
        (total_mass, combined_com, combined_inertia) in parent-local frame.
    """
    total_mass = float(parent_link.inertial_mass)
    # Weighted COM accumulator (in parent-local frame)
    weighted_com = total_mass * np.asarray(parent_link.inertial_pos, dtype=np.float64)
    # Inertia at parent-link origin (parallel axis from parent COM)
    I_origin = _shift_inertia_to_origin(
        np.asarray(parent_link.inertial_i, dtype=np.float64),
        np.asarray(parent_link.inertial_pos, dtype=np.float64),
        total_mass,
    )

    for child in children:
        child_mass = child.inertial_mass
        if child_mass is None or float(child_mass) < gs.EPS:
            continue
        m = float(child_mass)
        child_com_local = np.asarray(child.inertial_pos, dtype=np.float64)
        child_I_local = np.asarray(child.inertial_i, dtype=np.float64)

        # Transform child COM and inertia into parent frame
        rel_pos, rel_quat = compute_link_to_link_transform(child, parent_link)
        rel_pos = np.asarray(rel_pos, dtype=np.float64)
        R_child_to_parent = gu.quat_to_R(np.asarray(rel_quat, dtype=np.float64))

        com_in_parent = R_child_to_parent @ child_com_local + rel_pos
        I_in_parent = R_child_to_parent @ child_I_local @ R_child_to_parent.T

        total_mass += m
        weighted_com += m * com_in_parent
        I_origin += _shift_inertia_to_origin(I_in_parent, com_in_parent, m)

    combined_com = weighted_com / total_mass
    # Shift accumulated inertia from origin to combined COM (reverse parallel axis)
    d = combined_com
    combined_I = I_origin - total_mass * (np.dot(d, d) * np.eye(3) - np.outer(d, d))
    return total_mass, combined_com, combined_I


def _shift_inertia_to_origin(I_com: np.ndarray, com: np.ndarray, mass: float) -> np.ndarray:
    """Shift inertia tensor from COM to the frame origin via parallel axis theorem."""
    d = com
    return I_com + mass * (np.dot(d, d) * np.eye(3) - np.outer(d, d))


def _link_is_fixed_for_ipc(link: "RigidLink") -> bool:
    """Whether a link should be treated as fixed in IPC.

    For ipc_only entities the FREE→FIXED joint conversion makes link.is_fixed
    always True, but the body should only be fixed if the morph was originally
    fixed. For other coupling types, link.is_fixed is correct.
    """
    if link.entity.delegation is not None:
        return link.entity.morph.fixed
    return link.is_fixed


class _NewtonIterCounter:
    """Redirect OS-level stdout/stderr to pipes so that C++ libuipc log spam
    is captured and suppressed.  Extracts Newton iteration counts from the
    convergence summary line and only forwards warnings/errors.

    libuipc writes directly to C file descriptors (fd 1/2), so Python-level
    sys.stdout/stderr replacement is not sufficient — we must use os.dup2.

    Background threads drain the pipes continuously to prevent deadlock from
    the finite pipe buffer (~64KB on Linux).
    """

    _CONVERGED_RE = re.compile(r"Newton Iteration Converged with Iteration Count: (\d+)(?:, Line Search Iters: (\d+))?")
    _LS_MAX_RE = re.compile(r"Line Search Exits with Max Iteration: (\d+)")
    _NEWTON_MAX_RE = re.compile(r"Newton Iteration Exits with Max Iteration: (\d+)")
    # Tolerance checker lines: [x] or [*] followed by <CheckerName> report
    _TOL_CHECK_RE = re.compile(r"\[([\*x])\]\s+<(.+?)>\s+(.*)")
    # Separator line that ends a tolerance block
    _TOL_BLOCK_END_RE = re.compile(r"-{20,}")

    def __init__(self):
        self.newton_iters = 0
        self.ls_iters = 0
        self.ls_max_hits = 0
        self.newton_max_hit = False
        self.last_tol_checks: list[str] = []
        self._cur_tol_checks: list[str] = []
        self._active = False
        self._saved_stderr_fd = None
        self._saved_stdout_fd = None
        self._threads = []

    def start(self):
        if self._active:
            return
        import signal
        import threading

        # Flush before redirecting
        sys.stderr.flush()
        sys.stdout.flush()

        # Save original fds
        self._saved_stderr_fd = os.dup(2)
        self._saved_stdout_fd = os.dup(1)

        # Create pipes and redirect
        pipe_r_err, pipe_w_err = os.pipe()
        pipe_r_out, pipe_w_out = os.pipe()
        os.dup2(pipe_w_err, 2)
        os.dup2(pipe_w_out, 1)
        os.close(pipe_w_err)
        os.close(pipe_w_out)

        # Background threads drain pipes continuously to avoid buffer deadlock
        self._captured = [[], []]
        t_err = threading.Thread(target=self._drain, args=(pipe_r_err, 0), daemon=True)
        t_out = threading.Thread(target=self._drain, args=(pipe_r_out, 1), daemon=True)
        t_err.start()
        t_out.start()
        self._threads = [t_err, t_out]
        self._pipe_fds = [pipe_r_err, pipe_r_out]

        # Install SIGABRT handler to restore fds before crash so libuipc's
        # error message reaches the terminal instead of being lost in the pipe.
        self._prev_sigabrt = signal.getsignal(signal.SIGABRT)

        def _on_abort(signum, frame):
            self._emergency_restore()
            if callable(self._prev_sigabrt) and self._prev_sigabrt not in (signal.SIG_DFL, signal.SIG_IGN):
                self._prev_sigabrt(signum, frame)
            signal.signal(signal.SIGABRT, signal.SIG_DFL)
            os.kill(os.getpid(), signal.SIGABRT)

        signal.signal(signal.SIGABRT, _on_abort)

        self._active = True

    def _emergency_restore(self):
        """Restore original fds without processing captured data."""
        if not self._active:
            return
        try:
            os.dup2(self._saved_stderr_fd, 2)
            os.dup2(self._saved_stdout_fd, 1)
        except OSError:
            pass

    def _drain(self, fd, idx):
        """Read from pipe fd until EOF, storing chunks."""
        try:
            while True:
                data = os.read(fd, 65536)
                if not data:
                    break
                self._captured[idx].append(data)
        except OSError:
            pass

    def stop(self):
        if not self._active:
            return
        import signal

        # Restore SIGABRT handler
        signal.signal(signal.SIGABRT, self._prev_sigabrt)

        # Flush Python streams before restoring
        sys.stderr.flush()
        sys.stdout.flush()

        # Restore original fds (closes the pipe write ends implicitly)
        os.dup2(self._saved_stderr_fd, 2)
        os.dup2(self._saved_stdout_fd, 1)
        os.close(self._saved_stderr_fd)
        os.close(self._saved_stdout_fd)

        # Wait for drain threads to finish (pipe write ends are closed, so reads will EOF)
        for t in self._threads:
            t.join(timeout=2.0)
        for fd in self._pipe_fds:
            os.close(fd)

        # Process captured data — extract counters and forward errors/warnings.
        # When an error or warning is detected, dump ALL captured output so
        # stack traces and surrounding context are not lost.
        has_error = False
        all_lines: list[str] = []
        for chunks in self._captured:
            if not chunks:
                continue
            text = b"".join(chunks).decode("utf-8", errors="replace")
            for line in text.splitlines():
                all_lines.append(line)
                m = self._CONVERGED_RE.search(line)
                if m:
                    self.newton_iters += int(m.group(1))
                    if m.group(2) is not None:
                        self.ls_iters += int(m.group(2))
                m_ls = self._LS_MAX_RE.search(line)
                if m_ls:
                    self.ls_max_hits += 1
                m_nmax = self._NEWTON_MAX_RE.search(line)
                if m_nmax:
                    self.newton_max_hit = True
                m_tol = self._TOL_CHECK_RE.search(line)
                if m_tol:
                    status, name, report = m_tol.group(1), m_tol.group(2), m_tol.group(3)
                    # Strip C++ namespace prefix for readability
                    short_name = name.rsplit("::", 1)[-1]
                    self._cur_tol_checks.append(f"[{status}] {short_name}: {report}")
                if self._TOL_BLOCK_END_RE.search(line) and self._cur_tol_checks:
                    self.last_tol_checks = self._cur_tol_checks
                    self._cur_tol_checks = []
                if "[error]" in line or "[warning]" in line:
                    has_error = True
        if has_error:
            gs.logger.error("[IPC] libuipc log:\n" + "\n".join(all_lines))

        self._threads = []
        self._active = False

    def reset(self):
        """Reset counter for next frame."""
        n = self.newton_iters
        ls = self.ls_iters
        ls_max = self.ls_max_hits
        nmax = self.newton_max_hit
        tol_checks = self.last_tol_checks
        self.newton_iters = 0
        self.ls_iters = 0
        self.ls_max_hits = 0
        self.newton_max_hit = False
        self.last_tol_checks = []
        self._cur_tol_checks = []
        return n, ls, ls_max, nmax, tol_checks


class IPCCoupler(RBC):
    """
    Coupler class for handling Incremental Potential Contact (IPC) simulation coupling.

    This coupler manages the communication between Genesis solvers and the IPC system,
    including rigid bodies (as ABD objects) and FEM bodies in a unified contact framework.
    """

    def __init__(self, simulator: "Simulator", options: IPCCouplerOptions) -> None:
        """
        Initialize IPC Coupler.

        Parameters
        ----------
        simulator : Simulator
            The simulator containing all solvers
        options : IPCCouplerOptions
            IPC configuration options
        """
        # Check if uipc is available
        if not UIPC_AVAILABLE:
            raise ImportError(
                "Python module 'uipc' is required by IPCCoupler but is not installed. Please install it via "
                "`pip install pyuipc`."
            )

        self.sim = simulator
        self.options = options

        assert gs.use_zerocopy, (
            "IPC coupler requires zero-copy, which is not supported on this platform. "
            "Make sure Torch and Quadrants are sharing the same device."
        )

        # Define some proxies for convenience
        self.rigid_solver: "RigidSolver" = self.sim.rigid_solver
        self.fem_solver: "FEMSolver" = self.sim.fem_solver

        # ==== IPC System Infrastructure (created in _init_ipc) ====
        self._ipc_engine: Engine | None = None
        self._ipc_world: World | None = None
        self._ipc_scene: Scene | None = None
        self._ipc_workspace: str | None = None
        self._ipc_subscenes: list[SubsceneElement] = []
        self._ipc_constitution_tabular = None
        self._ipc_contact_tabular = None
        self._ipc_subscene_tabular = None
        self._ipc_objects = None
        self._ipc_animator = None

        # ==== Newton iteration counter (captures libuipc stderr) ====
        self._newton_counter = _NewtonIterCounter()
        self._ipc_frame = 0
        # Saved states for save_state/load_state (frame → snapshot dict)
        self._saved_states: dict[int, dict] = {}

        # ==== IPC Constitutions ====
        self._ipc_abd: AffineBodyConstitution | None = None
        self._ipc_abd_shell: AffineBodyShell | None = None
        self._ipc_stk: StableNeoHookean | None = None
        self._ipc_stc: SoftTransformConstraint | None = None
        self._ipc_nhs: NeoHookeanShell | None = None
        self._ipc_slbws: StrainLimitingBaraffWitkinShell | None = None
        self._ipc_dsb: DiscreteShellBending | None = None
        self._ipc_stress_pdsb: StressPlasticDiscreteShellBending | None = None
        self._ipc_strain_pdsb: StrainPlasticDiscreteShellBending | None = None
        self._ipc_stress_pdsb_modifier: StressPlasticDiscreteShellBendingModifier | None = None
        self._ipc_strain_pdsb_modifier: StrainPlasticDiscreteShellBendingModifier | None = None
        # AerodynamicDamping is optional (only some uipc builds ship it).
        self._ipc_aero: "AerodynamicDamping | None" = None
        self._ipc_hks: HookeanSpring | None = None
        self._ipc_krb: KirchhoffRodBending | None = None
        self._ipc_eac: ExternalArticulationConstraint | None = None
        self._ipc_fem_ext_force: FiniteElementExternalForce | None = None
        self._ipc_soft_pos_constraint: SoftPositionConstraint | None = None

        # ==== IPC Contact Elements ====
        self._ipc_no_collision_contact: ContactElement | None = None
        self._ipc_fems_contact: dict["FEMEntity", ContactElement] = {}
        self._ipc_clothes_contact: dict["FEMEntity", ContactElement] = {}
        self._ipc_ropes_contact: dict["FEMEntity", ContactElement] = {}
        self._ipc_abd_links_contact: dict["RigidLink", ContactElement] = {}
        self._ipc_grounds_contact: dict["RigidEntity", ContactElement] = {}

        # ==== Entity Collision Pair Overrides (pre-build) ====
        # Frozensets of (entity_a, entity_b) whose cross-entity ABD collision is disabled.
        self._disabled_collision_pairs: set[frozenset] = set()

        # ==== Entity Coupling Configuration ====
        self._coup_type_by_entity: dict["RigidEntity", COUPLING_TYPE] = {}
        # Link filter for two_way_soft_constraint coupling
        self._coup_links: dict["RigidEntity", set["RigidLink"]] = {}
        self._coupling_collision_settings: dict["RigidEntity", dict["RigidLink", bool]] = {}
        self._entities_by_coup_type: dict[COUPLING_TYPE, list["RigidEntity"]] = {}

        # ==== FEM Geometry & State ====
        # Per-entity FEM geometry slots, one list per env: entity → list[GeometrySlot] indexed by env
        self._fem_state_feature: "FiniteElementStateAccessorFeature | None" = None
        self._fem_state_geom: SimplicialComplex | None = None
        # FEM entities whose IPC positions need sync: entity → set of dirty env indices.
        self._fem_updated_entities: dict["FEMEntity", set[int]] = {}
        # Pending per-vertex external forces: entity → (n_verts, 3) array or None (clear).
        self._fem_external_forces: dict["FEMEntity", np.ndarray | None] = {}
        # Pending per-vertex soft position constraints: entity → dict with
        # "strength_ratio" (n_verts,) and "aim_position" (n_verts, 3), or None (clear).
        self._fem_soft_position: dict["FEMEntity", dict[str, np.ndarray] | None] = {}

        # ==== ABD Geometry & State ====
        # Cached merged world-frame trimesh per link for neutral-pose overlap check
        self._abd_merged_meshes: dict["RigidLink", trimesh.Trimesh] = {}
        self._abd_state_feature: AffineBodyStateAccessorFeature | None = None
        self._abd_state_geom: SimplicialComplex | None = None
        # ABD links whose IPC state needs sync: link → set of dirty env indices.
        self._abd_updated_links: dict["RigidLink", set[int]] = {}
        # Lookup tables built during _add_rigid_geoms_to_ipc:
        # qpos index → ABD target link
        self._q_to_abd_link: list["RigidLink | None"] = []
        # dof index → ABD target link
        self._dof_to_abd_link: list["RigidLink | None"] = []
        # global link index → ABD target link (handles fixed-joint merging)
        self._link_to_abd_link: list["RigidLink | None"] = []

        # ==== Input/Output Data ====
        self._abd_data_by_link: dict["RigidLink", ABDLinkData] = {}
        self._articulation_data_by_entity: dict["RigidEntity", ArticulatedEntityData] = {}
        self._fem_slots_by_entity: dict["FEMEntity", list] = {}

        # ==== User hooks ====
        # Callbacks invoked after _add_objects_to_ipc() but before _finalize_ipc().
        # Each hook receives the coupler instance as its sole argument.
        self._pre_finalize_hooks: list = []

        # ==== Restitution ====
        # Per-frame velocity corrections: (dof_start, dof_end, correction_array).
        self._restitution_vel_corrections: list[tuple[int, int, np.ndarray]] = []
        self._debug_surface_export_idx = 0
        self._debug_genesis_surface_export_after_genesis_before_ipc_idx = 0
        self._debug_genesis_surface_export_after_ipc_correction_idx = 0

        # ==== GUI ====
        self._ipc_gui: SceneGUI | None = None

    # ============================================================
    # Section 1: Configuration API
    # ============================================================

    def disable_collision_pair(self, entity_a: "RigidEntity", entity_b: "RigidEntity") -> None:
        """Disable IPC collision between two cross-entity ABD rigid bodies.

        Must be called before ``scene.build()``.  Order does not matter.
        """
        self._disabled_collision_pairs.add(frozenset((entity_a, entity_b)))

    def enable_collision_pair(self, entity_a: "RigidEntity", entity_b: "RigidEntity") -> None:
        """Re-enable a previously disabled cross-entity collision pair.

        Must be called before ``scene.build()``.
        """
        self._disabled_collision_pairs.discard(frozenset((entity_a, entity_b)))

    def build(self) -> None:
        """Build IPC system"""
        # IPC coupler builds a single IPC scene shared across all envs, so it requires
        # identical geometry topology (links, joints, geoms) across environments.
        # Batched info options allow per-env topology which is incompatible.
        if self.rigid_solver.is_active:
            rigid_options = cast(RigidOptions, self.rigid_solver._options)
            if rigid_options.batch_links_info or rigid_options.batch_dofs_info or rigid_options.batch_joints_info:
                gs.raise_exception(
                    "IPC coupler does not support batched rigid info (batch_links_info, batch_dofs_info, "
                    "batch_joints_info). Please disable these options when using IPC coupling."
                )

        self._B = self.sim._B

        self._init_ipc()
        self._setup_coupling_config()
        self._add_objects_to_ipc()
        # Run user-registered pre-finalize hooks (e.g. applying RotatingMotor)
        for hook in self._pre_finalize_hooks:
            hook(self)
        self._finalize_ipc()
        self._init_accessors()

        if os.environ.get("GS_ENABLE_IPC_GUI", "0") == "1":
            self._init_ipc_gui()

    def _setup_coupling_config(self):
        """Read coup_type, coup_links, and collision settings from entity materials."""
        assert gs.logger is not None

        entity: "RigidEntity"
        for i_e, entity in enumerate(cast(list["RigidEntity"], self.rigid_solver.entities)):
            if not entity.material.needs_coup:
                continue
            coup_type = entity.material.coup_type
            is_robot = any(j.type not in (gs.JOINT_TYPE.FREE, gs.JOINT_TYPE.FIXED) for j in entity.joints)
            if coup_type is None:
                # Auto-select: robots get articulation coupling, objects get ipc_only
                if is_robot:
                    coup_type = "external_articulation" if entity.base_link.is_fixed else "two_way_soft_constraint"
                else:
                    coup_type = "ipc_only"

            self._coup_type_by_entity[entity] = coup_type = getattr(COUPLING_TYPE, coup_type.upper())
            if coup_type == COUPLING_TYPE.EXTERNAL_ARTICULATION:
                if not entity.base_link.is_fixed:
                    gs.raise_exception(
                        f"Rigid entity {i_e} has a non-fixed base. "
                        f"Use 'two_way_soft_constraint' instead of 'external_articulation'."
                    )
                if not is_robot:
                    gs.raise_exception(
                        f"Rigid entity {i_e} has no articulated joints. Use 'ipc_only' instead of "
                        "'external_articulation'."
                    )
            elif coup_type == COUPLING_TYPE.IPC_ONLY:
                if is_robot:
                    gs.raise_exception(
                        f"Rigid entity {i_e} has articulated joints. Use 'external_articulation' instead of 'ipc_only'."
                    )
            gs.logger.debug(f"Rigid entity {i_e}: coupling type '{coup_type.name.lower()}'")

            # Resolve link filter from material
            link_filter_names = entity.material.coup_links
            if link_filter_names is not None:
                self._coup_links[entity] = set(map(entity.get_link, link_filter_names))
                gs.logger.debug(f"Rigid entity {i_e}: IPC link filter set to {len(link_filter_names)} link(s)")

            if coup_type == COUPLING_TYPE.TWO_WAY_SOFT_CONSTRAINT:
                selected_links = self._resolve_two_way_target_links(entity, is_robot)

            # Resolve collision settings from material
            if not entity.material.enable_coup_collision:
                # Disable collision for all links
                self._coupling_collision_settings[entity] = {link: False for link in entity.links}
                gs.logger.debug(f"Rigid entity {i_e}: IPC collision disabled for all links")
            elif entity.material.coup_collision_links is not None:
                # Positive filter: only named links get collision, others disabled
                allowed = set(entity.material.coup_collision_links)
                self._coupling_collision_settings[entity] = {
                    link: False for link in entity.links if link.name not in allowed
                }
                gs.logger.debug(f"Rigid entity {i_e}: IPC collision limited to {allowed}")

        # Categorize entities by coupling type
        for entity, coup_type in self._coup_type_by_entity.items():
            self._entities_by_coup_type.setdefault(coup_type, []).append(entity)

    def _resolve_two_way_target_links(self, entity: "RigidEntity", is_robot: bool):
        """Resolve and validate target links for two-way coupling."""
        ignore_end_effector_check = self.options.ignore_end_effector_check
        selected_links = self._coup_links.get(entity)
        if selected_links is None:
            if is_robot and not ignore_end_effector_check:
                gs.raise_exception(
                    "Two-way soft coupling for articulated robots requires explicit `coup_links` "
                    "(end-effector links only)."
                )
            selected_links = set(entity.links)

        if not ignore_end_effector_check:
            for link in selected_links:
                # End-effector only: no coupled child link in the same entity.
                # Non-coupled children (e.g. a wheel attached to a sprocket) are fine.
                coupled_children = [
                    child for child in entity.links if child.parent_idx == link.idx and child in selected_links
                ]
                if coupled_children:
                    gs.raise_exception(
                        f"Two-way soft coupling only supports end-effector links. "
                        f"Link '{link.name}' has coupled child links "
                        f"{[c.name for c in coupled_children]} in entity '{entity.uid}'."
                    )
        elif gs.logger is not None and is_robot:
            gs.logger.warning(
                "IPCCouplerOptions.ignore_end_effector_check=True: bypassing articulated two-way "
                "coupling link validation. Use with caution."
            )
        return selected_links

    @staticmethod
    def _validate_link_inertial_com_for_ipc(link: "RigidLink"):
        """Raise if inertial COM is outside collision mesh AABB (IPC assumption check)."""
        if link.inertial_pos is None:
            return

        aabb_min = np.full(3, np.inf, dtype=gs.np_float)
        aabb_max = np.full(3, -np.inf, dtype=gs.np_float)
        has_collision_mesh = False
        for geom in link.geoms:
            if geom.type == gs.GEOM_TYPE.PLANE or geom.n_verts <= 0:
                continue
            verts = gu.transform_by_trans_quat(geom.init_verts, geom.init_pos, geom.init_quat)
            aabb_min = np.minimum(aabb_min, verts.min(axis=0))
            aabb_max = np.maximum(aabb_max, verts.max(axis=0))
            has_collision_mesh = True

        if not has_collision_mesh:
            return

        com = np.asarray(link.inertial_pos, dtype=gs.np_float)
        tol = (aabb_max - aabb_min) * COM_AABB_TOL + COM_AABB_TOL
        if not ((aabb_min - tol < com) & (com < aabb_max + tol)).all():
            com_str = ", ".join(f"{n}={v:0.3f}" for n, v in zip(("x", "y", "z"), com))
            aabb_str = ", ".join(
                f"{n}=({mn:0.3f}, {mx:0.3f})" for n, mn, mx in zip(("x", "y", "z"), aabb_min, aabb_max)
            )
            gs.raise_exception(
                f"IPC two-way coupling assumption violated for link '{link.name}': "
                f"inertial COM [{com_str}] outside collision AABB [{aabb_str}]. "
                "Fix inertial origin or collision geometry alignment."
            )

    def _init_ipc(self) -> None:
        """Initialize IPC system components"""
        assert gs.logger is not None

        # Create IPC scene (deferred from __init__ so solver is_active is available)
        self._ipc_scene = Scene(build_ipc_scene_config(self.options, self.sim))
        self._ipc_constitution_tabular = self._ipc_scene.constitution_tabular()
        self._ipc_contact_tabular = self._ipc_scene.contact_tabular()
        # Disable the default contact model so unregistered pairs are ignored.
        # Genesis registers all required pairs explicitly; this prevents
        # unregistered pairs from silently falling back to an enabled default.
        self._ipc_contact_tabular.default_model(0.0, 0.0, False)
        self._ipc_subscene_tabular = self._ipc_scene.subscene_tabular()
        self._ipc_objects = self._ipc_scene.objects()
        self._ipc_animator = self._ipc_scene.animator()
        self._ipc_no_collision_contact = self._ipc_contact_tabular.create("no_collision_contact")

        if gs.logger.level <= logging.DEBUG:
            uipc.Logger.set_level(uipc.Logger.Level.Info)
            uipc.Timer.enable_all()
        else:
            # TODO: revert to Error after profiling
            uipc.Logger.set_level(uipc.Logger.Level.Info)
            uipc.Timer.disable_all()

        # Create workspace directory for IPC output, named after scene UID.
        workspace = os.path.join(tempfile.gettempdir(), f"genesis_ipc_{self.sim.scene.uid.full()}")
        os.makedirs(workspace, exist_ok=False)
        self._ipc_workspace = workspace

        # Note: gpu_device option may need to be set via CUDA environment variables (CUDA_VISIBLE_DEVICES)
        # before Genesis initialization, as libuipc Engine does not expose device selection in constructor
        self._ipc_engine = Engine("cuda", workspace)
        self._ipc_world = World(self._ipc_engine)

        # Set up sub-scenes for multi-environment to isolate per-environment contacts if batched
        for env_idx in range(self._B):
            ipc_subscene = self._ipc_subscene_tabular.create(f"subscene_{env_idx}")
            for other_ipc_subscene in self._ipc_subscenes:
                self._ipc_subscene_tabular.insert(other_ipc_subscene, ipc_subscene, False)
            self._ipc_subscenes.append(ipc_subscene)

    def _add_objects_to_ipc(self) -> None:
        """Add objects from solvers to IPC system"""
        # Add FEM entities to IPC
        if self.fem_solver.is_active:
            self._add_fem_entities_to_ipc()

        # Add rigid geoms and articulated entities to IPC based on per-entity coupling types
        if self.rigid_solver.is_active:
            self._add_rigid_geoms_to_ipc()
            self._add_articulation_entities_to_ipc()

        # Register all per-entity contact pair models with per-material friction
        self._register_contact_pairs()

    def _add_fem_entities_to_ipc(self) -> None:
        """Add FEM entities to the existing IPC scene (includes both volumetric FEM and cloth)"""

        entity: "FEMEntity"
        for i_e, entity in enumerate(cast(list["FEMEntity"], self.fem_solver.entities)):
            is_rope = isinstance(entity.material, Rope)
            is_cloth = not is_rope and isinstance(entity.material, Cloth)
            if is_rope:
                solver_type = "rope"
            elif is_cloth:
                solver_type = "cloth"
            else:
                solver_type = "fem"

            # ---- Create mesh (env-independent geometry) ----
            # linemesh for rope (1D), trimesh for cloth (2D), tetmesh for volumetric FEM (3D)
            verts = tensor_to_array(entity.init_positions).astype(np.float64, copy=False)
            if is_rope:
                edges = entity.elems.astype(np.int32, copy=False)
                mesh = uipc.geometry.linemesh(verts, edges)
            elif is_cloth:
                faces = entity.surface_triangles.astype(np.int32, copy=False)
                mesh = uipc.geometry.trimesh(verts, faces)
            else:
                mesh = uipc.geometry.tetmesh(verts, entity.elems)
            uipc.geometry.label_surface(mesh)

            # ---- Apply constitutions (env-independent) ----
            # Apply per-entity contact element
            if is_rope:
                self._ipc_ropes_contact[entity] = self._ipc_contact_tabular.create(f"rope_contact_{i_e}")
                self._ipc_ropes_contact[entity].apply_to(mesh)
            elif is_cloth:
                self._ipc_clothes_contact[entity] = self._ipc_contact_tabular.create(f"cloth_contact_{i_e}")
                self._ipc_clothes_contact[entity].apply_to(mesh)
            else:
                self._ipc_fems_contact[entity] = self._ipc_contact_tabular.create(f"fem_contact_{i_e}")
                self._ipc_fems_contact[entity].apply_to(mesh)

            # Apply material constitution based on type
            if is_rope:
                # HookeanSpring for stretch
                if self._ipc_hks is None:
                    self._ipc_hks = HookeanSpring()
                    self._ipc_constitution_tabular.insert(self._ipc_hks)

                self._ipc_hks.apply_to(
                    mesh,
                    moduli=entity.material.E,
                    mass_density=entity.material.rho,
                    thickness=entity.material.thickness,
                )

                # KirchhoffRodBending for bending (optional)
                if entity.material.bending_stiffness is not None:
                    if self._ipc_krb is None:
                        self._ipc_krb = KirchhoffRodBending()
                        self._ipc_constitution_tabular.insert(self._ipc_krb)

                    self._ipc_krb.apply_to(mesh, E=entity.material.bending_stiffness)
            elif is_cloth:
                moduli = ElasticModuli2D.youngs_poisson(entity.material.E, entity.material.nu)
                shell_model = entity.material.model

                if shell_model == "neohookean":
                    if self._ipc_nhs is None:
                        self._ipc_nhs = NeoHookeanShell()
                        self._ipc_constitution_tabular.insert(self._ipc_nhs)
                    self._ipc_nhs.apply_to(
                        mesh, moduli=moduli, mass_density=entity.material.rho, thickness=entity.material.thickness
                    )
                else:
                    if self._ipc_slbws is None:
                        self._ipc_slbws = StrainLimitingBaraffWitkinShell()
                        self._ipc_constitution_tabular.insert(self._ipc_slbws)
                    self._ipc_slbws.apply_to(
                        mesh, moduli=moduli, mass_density=entity.material.rho, thickness=entity.material.thickness
                    )

                if entity.material.bending_stiffness is not None:
                    is_paper = isinstance(entity.material, Paper)
                    if is_paper:
                        # Plastic bending for Paper material
                        if entity.material.plasticity_model == "stress":
                            if self._ipc_stress_pdsb is None:
                                self._ipc_stress_pdsb = StressPlasticDiscreteShellBending()
                                self._ipc_constitution_tabular.insert(self._ipc_stress_pdsb)
                            self._ipc_stress_pdsb.apply_to(
                                mesh,
                                bending_stiffness=entity.material.bending_stiffness,
                                yield_stress=entity.material.yield_stress,
                                hardening_modulus=entity.material.hardening_modulus,
                            )
                            if StressPlasticDiscreteShellBendingModifier is None:
                                gs.logger.warning(
                                    "Material requests stress-plastic bending modifier but the installed "
                                    "uipc build does not expose StressPlasticDiscreteShellBendingModifier; skipping."
                                )
                            else:
                                if self._ipc_stress_pdsb_modifier is None:
                                    self._ipc_stress_pdsb_modifier = StressPlasticDiscreteShellBendingModifier()
                                    self._ipc_constitution_tabular.insert(self._ipc_stress_pdsb_modifier)
                                self._ipc_stress_pdsb_modifier.apply_to(mesh)
                        else:
                            if self._ipc_strain_pdsb is None:
                                self._ipc_strain_pdsb = StrainPlasticDiscreteShellBending()
                                self._ipc_constitution_tabular.insert(self._ipc_strain_pdsb)
                            self._ipc_strain_pdsb.apply_to(
                                mesh,
                                bending_stiffness=entity.material.bending_stiffness,
                                yield_threshold=entity.material.yield_threshold,
                                hardening_modulus=entity.material.hardening_modulus,
                            )
                            if StrainPlasticDiscreteShellBendingModifier is None:
                                gs.logger.warning(
                                    "Material requests strain-plastic bending modifier but the installed "
                                    "uipc build does not expose StrainPlasticDiscreteShellBendingModifier; skipping."
                                )
                            else:
                                if self._ipc_strain_pdsb_modifier is None:
                                    self._ipc_strain_pdsb_modifier = StrainPlasticDiscreteShellBendingModifier()
                                    self._ipc_constitution_tabular.insert(self._ipc_strain_pdsb_modifier)
                                self._ipc_strain_pdsb_modifier.apply_to(mesh)
                    else:
                        # Elastic bending for Cloth material
                        if self._ipc_dsb is None:
                            self._ipc_dsb = DiscreteShellBending()
                            self._ipc_constitution_tabular.insert(self._ipc_dsb)

                        self._ipc_dsb.apply_to(mesh, bending_stiffness=entity.material.bending_stiffness)

                # Aerodynamic damping (optional, for Cloth and Paper).
                # Only available when the installed uipc build exposes
                # ``AerodynamicDamping`` (private fork); skipped otherwise.
                if entity.material.aerodynamic_drag is not None:
                    if AerodynamicDamping is None:
                        gs.logger.warning(
                            "Material requests aerodynamic_drag but the installed "
                            "uipc build does not expose AerodynamicDamping; skipping."
                        )
                    else:
                        if self._ipc_aero is None:
                            self._ipc_aero = AerodynamicDamping()
                            self._ipc_constitution_tabular.insert(self._ipc_aero)
                        self._ipc_aero.apply_to(
                            mesh,
                            drag_coefficient=entity.material.aerodynamic_drag,
                            curvature_scale=entity.material.curvature_drag_scale,
                            inflate_scale=entity.material.curvature_inflate_scale,
                        )
            else:
                if self._ipc_stk is None:
                    self._ipc_stk = StableNeoHookean()
                    self._ipc_constitution_tabular.insert(self._ipc_stk)

                moduli = ElasticModuli.youngs_poisson(entity.material.E, entity.material.nu)
                self._ipc_stk.apply_to(mesh, moduli, mass_density=entity.material.rho)

            # Apply external force constitution (initially zero, activated at runtime)
            if is_cloth:
                if FiniteElementExternalForce is None:
                    gs.logger.warning(
                        "Cloth entity requests FiniteElementExternalForce but the installed "
                        "uipc build does not expose it; external forces will be unavailable."
                    )
                else:
                    if self._ipc_fem_ext_force is None:
                        self._ipc_fem_ext_force = FiniteElementExternalForce()
                        self._ipc_constitution_tabular.insert(self._ipc_fem_ext_force)
                    self._ipc_fem_ext_force.apply_to(mesh, np.array([0.0, 0.0, 0.0]))

            # Apply soft position constraint (initially inactive, activated at runtime)
            if is_cloth:
                if self._ipc_soft_pos_constraint is None:
                    self._ipc_soft_pos_constraint = SoftPositionConstraint()
                    self._ipc_constitution_tabular.insert(self._ipc_soft_pos_constraint)
                self._ipc_soft_pos_constraint.apply_to(mesh, strength_rate=0.0)

            # Per-entity d_hat override
            if entity.material.contact_d_hat is not None:
                mesh.meta().create(uipc.builtin.d_hat, float(entity.material.contact_d_hat))

            # Per-entity gravity override (e.g. (0,0,0) to disable gravity on this entity).
            if entity.material.gravity is not None:
                mesh.vertices().create(
                    uipc.builtin.gravity,
                    np.array(entity.material.gravity, dtype=np.float64),
                )

            # Partition FEM/cloth mesh for faster IPC assembly/solve on large meshes.
            # This follows libuipc sample usage and is applied once on env-independent geometry.
            uipc.geometry.mesh_partition(mesh)

            # ---- Per-environment: create IPC objects, then set per-env attrs on slot geometry ----
            fem_slots: list = []
            for env_idx in range(self._B):
                fem_obj = self._ipc_objects.create(f"{solver_type}_{i_e}_{env_idx}")
                fem_geom_slot, _ = fem_obj.geometries().create(mesh)
                fem_slots.append(fem_geom_slot)

                # All per-env writes go on the slot's own geometry (deep-copied)
                slot_geom = fem_geom_slot.geometry()
                if self._B > 1:
                    self._ipc_subscenes[env_idx].apply_to(slot_geom)
                slot_meta = slot_geom.meta()
                slot_meta.create("solver_type", solver_type)
                slot_meta.create("entity_idx", str(i_e))
                slot_meta.create("entity_name", str(entity.name))
                slot_meta.create("env_idx", str(env_idx))
            self._fem_slots_by_entity[entity] = fem_slots

    def _add_rigid_geoms_to_ipc(self) -> None:
        """Add rigid geoms to the IPC scene as ABD objects, merging geoms by link."""
        assert gs.logger is not None

        gs.logger.debug(f"Registered entity coupling types: {set(self._coup_type_by_entity.values())}")

        # Initialize lookup tables
        self._q_to_abd_link = [None] * self.rigid_solver.n_qs
        self._dof_to_abd_link = [None] * self.rigid_solver.n_dofs
        self._link_to_abd_link = [None] * self.rigid_solver.n_links

        # ========== Build fixed-joint merge map for ext_art entities ==========
        # For external_articulation, fixed-joint child links must be folded into
        # their parent ABD body (IPC has no FixedJoint constraint).  Build a map
        # from each fixed-joint link to its merge target and a reverse map listing
        # all children that merge into each target.
        abd_merge_target: dict["RigidLink", "RigidLink"] = {}
        abd_merge_children: dict["RigidLink", list["RigidLink"]] = {}
        for link in self.rigid_solver.links:
            entity = link.entity
            if self._coup_type_by_entity.get(entity) != COUPLING_TYPE.EXTERNAL_ARTICULATION:
                continue
            target = find_abd_merge_target(link)
            if target is not link:
                abd_merge_target[link] = target
                abd_merge_children.setdefault(target, []).append(link)
                gs.logger.info(f"[IPC] ext_art: merging link '{link.name}' into '{target.name}'")

        # ========== Collect selected links (env-independent) ==========
        selected_links: list["RigidLink"] = []
        for link in self.rigid_solver.links:
            entity = link.entity

            coup_type = self._coup_type_by_entity.get(entity)
            if coup_type is None:
                continue

            # Skip links that are merged into a parent for ext_art
            if link in abd_merge_target:
                continue

            # Link filter for two_way_soft_constraint
            if coup_type == COUPLING_TYPE.TWO_WAY_SOFT_CONSTRAINT:
                link_filter = self._coup_links.get(entity)
                if link_filter is not None and link not in link_filter:
                    continue

            selected_links.append(link)

        # ========== Process each link across environments ==========
        links_pos = qd_to_numpy(self.rigid_solver.links_state.pos, transpose=True)
        links_quat = qd_to_numpy(self.rigid_solver.links_state.quat, transpose=True)

        for link in selected_links:
            entity = link.entity
            entity_coup_type = self._coup_type_by_entity[entity]
            i_e = entity._idx_in_solver

            # ---- Collect geom meshes (env-independent local-frame geometry) ----
            # For ext_art links with merged children, also include children's
            # geoms transformed from child-local frame to this link's local frame.
            merged_children = abd_merge_children.get(link, [])
            geom_sources: list[tuple[list, np.ndarray, np.ndarray]] = [
                (link.geoms, np.zeros(3, dtype=gs.np_float), np.array([1, 0, 0, 0], dtype=gs.np_float))
            ]
            for child in merged_children:
                child_pos, child_quat = compute_link_to_link_transform(child, link)
                geom_sources.append((child.geoms, child_pos, child_quat))

            meshes = []
            has_plane_geom = False
            for geoms, frame_pos, frame_quat in geom_sources:
                for geom in geoms:
                    if geom.type == gs.GEOM_TYPE.PLANE:
                        has_plane_geom = True
                        local_normal = geom.data[:3].astype(np.float64, copy=False)
                        normal = gu.transform_by_quat(local_normal, geom.init_quat)
                        normal = normal / np.linalg.norm(normal)
                        height = np.dot(geom.init_pos, normal)
                        plane_geom = uipc.geometry.ground(height, normal)

                        if entity not in self._ipc_grounds_contact:
                            plane_contact = self._ipc_contact_tabular.create(f"ground_contact_{i_e}")
                            self._ipc_grounds_contact[entity] = plane_contact
                        self._ipc_grounds_contact[entity].apply_to(plane_geom)

                        for env_idx in range(self._B):
                            plane_obj = self._ipc_objects.create(f"rigid_plane_{geom.idx}_{env_idx}")
                            plane_geom_slot, _ = plane_obj.geometries().create(plane_geom)
                            slot_geom = plane_geom_slot.geometry()
                            if self._B > 1:
                                self._ipc_subscenes[env_idx].apply_to(slot_geom)
                            slot_meta = slot_geom.meta()
                            slot_meta.create("solver_type", "rigid")
                            slot_meta.create("entity_name", str(entity.name))
                            slot_meta.create("link_name", str(link.name))
                            slot_meta.create("link_idx", str(link.idx))
                            slot_meta.create("env_idx", str(env_idx))
                    elif geom.n_verts:
                        # Apply geom transform to vertices (geom-local → link-local)
                        geom_verts = gu.transform_by_trans_quat(geom.init_verts, geom.init_pos, geom.init_quat)
                        # If from a merged child, transform child-local → parent-local
                        if frame_pos is not None and np.any(frame_pos) or np.any(frame_quat[1:]):
                            geom_verts = gu.transform_by_trans_quat(geom_verts, frame_pos, frame_quat)

                        try:
                            mesh = uipc.geometry.trimesh(
                                geom_verts.astype(np.float64, copy=False),
                                geom.init_faces.astype(np.int32, copy=False),
                            )
                        except RuntimeError as e:
                            gs.raise_exception_from(f"Failed to process geom {geom.idx} for IPC.", e)

                        meshes.append(mesh)

            # ---- Determine coupling behavior ----
            is_ipc_only = entity_coup_type == COUPLING_TYPE.IPC_ONLY
            is_soft_constraint_target = entity_coup_type == COUPLING_TYPE.TWO_WAY_SOFT_CONSTRAINT

            # ---- Resolve density ----
            rho = entity.material.rho
            if rho is None:
                if entity.solver._enable_mujoco_compatibility:
                    rho = RHO_MUJOCO
                else:
                    rho = RHO_ROBOT if link._is_robot else RHO_OBJECT
            rho = float(rho)

            is_proxy = not meshes

            # Plane geoms are already added as IPC ground objects above;
            # no ABD body is needed for the link.
            if is_proxy and has_plane_geom:
                continue

            if is_proxy:
                # No collision mesh — create a proxy ABD body from inertial properties
                if self._ipc_abd is None:
                    self._ipc_abd = AffineBodyConstitution()
                    self._ipc_constitution_tabular.insert(self._ipc_abd)
                mass_val = float(link.inertial_mass)
                mass_center = np.asarray(link.inertial_pos, dtype=np.float64)
                inertia = np.asarray(link.inertial_i, dtype=np.float64)
                volume = mass_val / rho if rho > 0 else 1.0
                rigid_link_geom = self._ipc_abd.create_proxy(
                    kappa=ABD_KAPPA * uipc.unit.MPa,
                    mass=mass_val,
                    mass_center=mass_center,
                    inertia=inertia,
                    volume=volume,
                )
                self._ipc_no_collision_contact.apply_to(rigid_link_geom)
                gs.logger.info(f"[IPC] link '{link.name}' has no collision mesh — using proxy ABD body")
            else:
                # ---- Merge meshes ----
                rigid_link_geom = meshes[0] if len(meshes) == 1 else uipc.geometry.merge(meshes)
                uipc.geometry.label_surface(rigid_link_geom)

                # Cache merged world-frame trimesh for env 0 (used by neutral overlap check)
                link_T_0 = gu.trans_quat_to_T(links_pos[0, link.idx], links_quat[0, link.idx])
                local_verts = np.asarray(rigid_link_geom.positions().view())[..., 0]
                world_verts = (link_T_0[:3, :3] @ local_verts.T).T + link_T_0[:3, 3]
                faces = rigid_link_geom.triangles().topo().view()[..., 0]
                # Shrink 0.1% toward centroid to match rigid collider's neutral overlap check
                centroid = world_verts.mean(axis=0, keepdims=True)
                world_verts = centroid + (1.0 - 1e-3) * (world_verts - centroid)
                self._abd_merged_meshes[link] = trimesh.Trimesh(vertices=world_verts, faces=faces, process=False)

                # Apply per-link contact element or no-collision marker
                if self._coupling_collision_settings.get(entity, {}).get(link, True):
                    if link not in self._ipc_abd_links_contact:
                        abd_contact = self._ipc_contact_tabular.create(f"abd_link_contact_{link.idx}")
                        self._ipc_abd_links_contact[link] = abd_contact
                    self._ipc_abd_links_contact[link].apply_to(rigid_link_geom)
                else:
                    self._ipc_no_collision_contact.apply_to(rigid_link_geom)

                # Apply ABD constitution — use AffineBodyShell for non-watertight meshes
                # (open surfaces like gripper pads) where volume-based mass would be near-zero.
                mesh_for_check = self._abd_merged_meshes.get(link)
                is_watertight = mesh_for_check is not None and mesh_for_check.is_watertight
                gs.logger.info(
                    f"[IPC ABD] link={link.name}, is_watertight={is_watertight}, mesh_exists={mesh_for_check is not None}, rho={rho}"
                )

                if is_watertight:
                    if self._ipc_abd is None:
                        self._ipc_abd = AffineBodyConstitution()
                        self._ipc_constitution_tabular.insert(self._ipc_abd)
                    self._ipc_abd.apply_to(rigid_link_geom, kappa=ABD_KAPPA * uipc.unit.MPa, mass_density=rho)
                    # Explicitly set thickness=0 so the sanity check merge doesn't
                    # inherit a non-zero default from AffineBodyShell geometries.
                    rigid_link_geom.vertices().create(uipc.builtin.thickness, 0.0)
                else:
                    if self._ipc_abd_shell is None:
                        self._ipc_abd_shell = AffineBodyShell()
                        self._ipc_constitution_tabular.insert(self._ipc_abd_shell)
                    # Use contact_d_hat as shell thickness — a reasonable scale for thin rigid surfaces.
                    shell_thickness = (self.options.contact_d_hat or 0.001) / 2
                    self._ipc_abd_shell.apply_to(
                        rigid_link_geom, kappa=ABD_KAPPA * uipc.unit.MPa, mass_density=rho, thickness=shell_thickness
                    )

            # Per-entity d_hat override
            if entity.material.contact_d_hat is not None:
                rigid_link_geom.meta().create(uipc.builtin.d_hat, float(entity.material.contact_d_hat))

            # Apply SoftTransformConstraint for coupled links
            if is_soft_constraint_target:
                if self._ipc_stc is None:
                    self._ipc_stc = SoftTransformConstraint()
                    self._ipc_constitution_tabular.insert(self._ipc_stc)

                constraint_strength = np.array(entity.material.coup_stiffness)
                self._ipc_stc.apply_to(rigid_link_geom, constraint_strength)

                # Enable constraint once at build time (stays enabled for all frames)
                is_constrained_attr = rigid_link_geom.instances().find(uipc.builtin.is_constrained)
                if is_constrained_attr is not None:
                    uipc.view(is_constrained_attr)[0] = 1

            # Set geometry attributes (env-independent)
            # external_kinetic: 1 = driven by rigid solver, 0 = IPC-only
            external_kinetic_attr = rigid_link_geom.instances().find(uipc.builtin.external_kinetic)
            uipc.view(external_kinetic_attr)[:] = int(not is_ipc_only)

            is_fixed_attr = rigid_link_geom.instances().find(uipc.builtin.is_fixed)
            uipc.view(is_fixed_attr)[:] = int(_link_is_fixed_for_ipc(link))

            # Create ref_dof_prev for external_articulation links.
            # This attribute is re-read every step by the ExternalArticulationConstraint
            # and used as the reference DOF state for computing delta_theta. Without it,
            # set_qpos teleportation causes a huge energy spike because q_prevs (IPC internal)
            # still holds the pre-teleport state.
            is_ext_art = entity_coup_type == COUPLING_TYPE.EXTERNAL_ARTICULATION
            if is_ext_art:
                rigid_link_geom.instances().create("ref_dof_prev", np.zeros(12, dtype=np.float64))

            # ---- Per-environment: create IPC objects, then set per-env attrs on slot geometry ----
            abd_geom_slots: list[GeometrySlot] = []
            for env_idx in range(self._B):
                abd_obj = self._ipc_objects.create(f"rigid_link_{link.idx}_{env_idx}")
                abd_geom_slot, _ = abd_obj.geometries().create(rigid_link_geom)

                # All per-env writes go on the slot's own geometry (deep-copied)
                slot_geom = abd_geom_slot.geometry()
                link_T = gu.trans_quat_to_T(links_pos[env_idx, link.idx], links_quat[env_idx, link.idx])
                uipc.view(slot_geom.transforms())[0] = link_T

                # Initialize ref_dof_prev from the initial transform
                if is_ext_art:
                    ref_dof_prev_attr = slot_geom.instances().find("ref_dof_prev")
                    uipc.view(ref_dof_prev_attr)[0] = uipc.geometry.affine_body.transform_to_q(
                        link_T.astype(np.float64)
                    )
                if self._B > 1:
                    self._ipc_subscenes[env_idx].apply_to(slot_geom)
                slot_meta = slot_geom.meta()
                slot_meta.create("solver_type", "rigid")
                slot_meta.create("entity_name", str(entity.name))
                slot_meta.create("link_name", str(link.name))
                slot_meta.create("link_idx", str(link.idx))
                slot_meta.create("env_idx", str(env_idx))
                abd_geom_slots.append(abd_geom_slot)

            # ---- Store link data ----
            self._abd_data_by_link[link] = ABDLinkData(
                slots=abd_geom_slots,
                aim_transforms=np.tile(np.eye(4, dtype=gs.np_float), (self._B, 1, 1)),
                ipc_transforms=np.tile(np.eye(4, dtype=gs.np_float), (self._B, 1, 1)),
                ipc_velocities=np.zeros((self._B, 4, 4), dtype=gs.np_float),
            )

            # Populate lookup tables
            self._link_to_abd_link[link.idx] = link
            if link.q_start >= 0:
                for qi in range(link.q_start, link.q_end):
                    self._q_to_abd_link[qi] = link
            if link.dof_start >= 0:
                for di in range(link.dof_start, link.dof_end):
                    self._dof_to_abd_link[di] = link
            # Merged children also map to this link's ABD body
            for child in merged_children:
                self._link_to_abd_link[child.idx] = link

    def _add_articulation_entities_to_ipc(self) -> None:
        """
        Add articulated robot entities to IPC using ExternalArticulationConstraint.

        This enables joint-level coupling between Genesis and IPC.
        """
        assert gs.logger is not None

        if COUPLING_TYPE.EXTERNAL_ARTICULATION not in self._coup_type_by_entity.values():
            return

        self._ipc_eac = ExternalArticulationConstraint()
        self._ipc_constitution_tabular.insert(self._ipc_eac)

        joints_xaxis = qd_to_numpy(self.rigid_solver.joints_state.xaxis, transpose=True)
        joints_xanchor = qd_to_numpy(self.rigid_solver.joints_state.xanchor, transpose=True)

        # Process each rigid entity with external_articulation coupling type
        for i_e, entity in enumerate(cast(list["RigidEntity"], self.rigid_solver.entities)):
            # Only process entities with external_articulation coupling type
            if self._coup_type_by_entity.get(entity) != COUPLING_TYPE.EXTERNAL_ARTICULATION:
                continue

            gs.logger.debug(f"Adding articulated entity {i_e} with {entity.n_joints} joints")

            # ---- Collect joint info (env-independent) ----
            joints: list[tuple["RigidJoint", type, "RigidLink", "RigidLink"]] = []
            for joint in entity.joints:
                if joint.type == gs.JOINT_TYPE.FIXED:
                    continue
                elif joint.type == gs.constants.JOINT_TYPE.REVOLUTE:
                    joint_constitution = AffineBodyRevoluteJoint
                elif joint.type == gs.constants.JOINT_TYPE.PRISMATIC:
                    joint_constitution = AffineBodyPrismaticJoint
                else:
                    gs.raise_exception(f"Unsupported joint type: {joint.type}")

                child_link = joint.link
                parent_link = entity.links[max(joint.link.parent_idx, 0) - entity.link_start]
                # Remap merged links to their ABD merge target
                if parent_link not in self._abd_data_by_link:
                    abd_target = self._link_to_abd_link[parent_link.idx]
                    if abd_target is not None:
                        parent_link = abd_target
                if child_link not in self._abd_data_by_link:
                    abd_target = self._link_to_abd_link[child_link.idx]
                    if abd_target is not None:
                        child_link = abd_target
                if parent_link not in self._abd_data_by_link or child_link not in self._abd_data_by_link:
                    gs.raise_exception(
                        "Rigid link has no collision geometry. Coupling type 'external_articulation' is not supported."
                    )
                joints.append((joint, joint_constitution, parent_link, child_link))

            # ---- Create joint geometries per environment ----
            articulation_geom_slots: list[GeometrySlot] = []
            for env_idx in range(self._B):
                joint_geom_slots: list[GeometrySlot] = []
                for joint, joint_constitution, parent_link, child_link in joints:
                    joint_axis = joints_xaxis[env_idx, joint.idx]
                    joint_pos = joints_xanchor[env_idx, joint.idx]

                    v1 = joint_pos - 0.5 * joint_axis
                    v2 = joint_pos + 0.5 * joint_axis
                    vertices = np.array([v1, v2], dtype=np.float64)
                    edges = np.array([[0, 1]], dtype=np.int32)
                    joint_geom = uipc.geometry.linemesh(vertices, edges)
                    if self._B > 1:
                        self._ipc_subscenes[env_idx].apply_to(joint_geom)

                    parent_abd_slot = self._abd_data_by_link[parent_link].slots[env_idx]
                    child_abd_slot = self._abd_data_by_link[child_link].slots[env_idx]
                    joint_constitution().apply_to(
                        joint_geom, [parent_abd_slot], [0], [child_abd_slot], [0], [self.options.joint_strength_ratio]
                    )

                    joint_obj = self._ipc_objects.create(f"joint_{joint.idx}_{env_idx}")
                    joint_geom_slot, _ = joint_obj.geometries().create(joint_geom)
                    joint_geom_slots.append(joint_geom_slot)

                articulation_geom = self._ipc_eac.create_geometry(joint_geom_slots, [0] * len(joint_geom_slots))
                if self._B > 1:
                    self._ipc_subscenes[env_idx].apply_to(articulation_geom)

                articulation_obj = self._ipc_objects.create(f"articulation_entity_{i_e}_{env_idx}")
                articulation_geom_slot, _ = articulation_obj.geometries().create(articulation_geom)
                articulation_geom_slots.append(articulation_geom_slot)

            # Store articulation data with pre-allocated per-step arrays
            n_joints = len(joints)
            self._articulation_data_by_entity[entity] = ArticulatedEntityData(
                slots=articulation_geom_slots,
                q_slice=slice(entity.q_start, entity.q_end),
                dof_slice=slice(entity.dof_start, entity.dof_end),
                joints_child_link=[j.link for j, *_ in joints],
                joints_qs_idx_local=[j.qs_idx_local[0] for j, *_ in joints],
                delta_theta_tilde=np.zeros((self._B, n_joints), dtype=np.float64),
                prev_qpos=np.zeros((self._B, entity.n_qs), dtype=np.float64),
                mass_matrix=np.zeros((self._B, entity.n_dofs, entity.n_dofs), dtype=np.float64),
                ipc_qpos=np.zeros((self._B, entity.n_qs), dtype=gs.np_float),
            )

            gs.logger.debug(f"Successfully added articulated rigid entity {i_e} to IPC.")

    def _register_contact_pairs(self) -> None:
        """Register pairwise contact models for all contact elements.

        Friction is combined by geometric mean, resistance by harmonic mean (series spring).
        Rigid link self-collision filtering mirrors the RigidSolver collider:
        ``enable_self_collision``, ``enable_adjacent_collision``, ``enable_neutral_collision``.
        """
        from genesis.engine.solvers.rigid.collider.collider import are_links_adjacent

        assert gs.logger is not None

        enable_self_collision = self.rigid_solver._enable_self_collision
        enable_adjacent_collision = self.rigid_solver._enable_adjacent_collision
        enable_neutral_collision = self.rigid_solver._enable_neutral_collision

        # Collect non-ABD contact infos (FEM, cloth)
        non_abd_infos: list[tuple[ContactElement, float, float]] = []
        for entity, elem in (
            *self._ipc_clothes_contact.items(),
            *self._ipc_ropes_contact.items(),
            *self._ipc_fems_contact.items(),
        ):
            friction = entity.material.friction_mu
            resistance = entity.material.contact_resistance or self.options.contact_resistance
            non_abd_infos.append((elem, friction, resistance))

        # Collect ABD link contact infos
        abd_link_infos: list[tuple[ContactElement, "RigidLink", float, float]] = []
        for link, elem in self._ipc_abd_links_contact.items():
            friction = link.entity.material.coup_friction
            resistance = link.entity.material.contact_resistance or self.options.contact_resistance
            abd_link_infos.append((elem, link, friction, resistance))

        # ---- Non-ABD × Non-ABD pairs (FEM × FEM) ----
        enable_fem_fem_friction = self.options.enable_fem_fem_friction
        for i, (elem_i, friction_i, resistance_i) in enumerate(non_abd_infos):
            for elem_j, friction_j, resistance_j in non_abd_infos[i:]:
                friction_ij = geometric_mean(friction_i, friction_j) if enable_fem_fem_friction else 0.0
                self._ipc_contact_tabular.insert(
                    elem_i,
                    elem_j,
                    friction_ij,
                    harmonic_mean(resistance_i, resistance_j),
                    True,
                )

        # ---- Non-ABD × ABD link pairs ----
        for elem_na, friction_na, resistance_na in non_abd_infos:
            for elem_abd, _, friction_abd, resistance_abd in abd_link_infos:
                self._ipc_contact_tabular.insert(
                    elem_na,
                    elem_abd,
                    geometric_mean(friction_na, friction_abd),
                    harmonic_mean(resistance_na, resistance_abd),
                    True,
                )

        # ---- ABD link × ABD link pairs (with self-collision filtering) ----
        _n_enabled = 0
        _n_disabled = 0
        for i, (elem_i, link_i, friction_i, resistance_i) in enumerate(abd_link_infos):
            for elem_j, link_j, friction_j, resistance_j in abd_link_infos[i:]:
                friction_ij = geometric_mean(friction_i, friction_j)
                resistance_ij = harmonic_mean(resistance_i, resistance_j)

                if not self.options.enable_rigid_rigid_contact:
                    self._ipc_contact_tabular.insert(elem_i, elem_j, friction_ij, resistance_ij, False)
                    _n_disabled += 1
                    continue

                # Fixed-fixed pairs never collide (mirrors RigidSolver collider)
                if _link_is_fixed_for_ipc(link_i) and _link_is_fixed_for_ipc(link_j):
                    self._ipc_contact_tabular.insert(elem_i, elem_j, friction_ij, resistance_ij, False)
                    _n_disabled += 1
                    continue

                # Same-entity self-collision filtering (mirrors RigidSolver collider)
                if link_i.entity is link_j.entity and link_i is not link_j:
                    if not enable_self_collision:
                        self._ipc_contact_tabular.insert(elem_i, elem_j, friction_ij, resistance_ij, False)
                        _n_disabled += 1
                        gs.logger.debug(f"[IPC CONTACT] DISABLED self-collision: {link_i.name} × {link_j.name}")
                        continue
                    if not enable_adjacent_collision and are_links_adjacent(link_i, link_j):
                        self._ipc_contact_tabular.insert(elem_i, elem_j, friction_ij, resistance_ij, False)
                        _n_disabled += 1
                        gs.logger.debug(f"[IPC CONTACT] DISABLED adjacent: {link_i.name} × {link_j.name}")
                        continue
                    mesh_i = self._abd_merged_meshes.get(link_i)
                    mesh_j = self._abd_merged_meshes.get(link_j)
                    if (
                        not enable_neutral_collision
                        and mesh_i is not None
                        and mesh_j is not None
                        and are_meshes_overlapping(mesh_i, mesh_j)
                    ):
                        self._ipc_contact_tabular.insert(elem_i, elem_j, friction_ij, resistance_ij, False)
                        _n_disabled += 1
                        gs.logger.debug(f"[IPC CONTACT] DISABLED overlapping: {link_i.name} × {link_j.name}")
                        continue

                # Cross-entity collision pair override
                if link_i.entity is not link_j.entity:
                    if frozenset((link_i.entity, link_j.entity)) in self._disabled_collision_pairs:
                        self._ipc_contact_tabular.insert(elem_i, elem_j, friction_ij, resistance_ij, False)
                        _n_disabled += 1
                        gs.logger.debug(f"[IPC CONTACT] DISABLED by pair override: {link_i.name} × {link_j.name}")
                        continue

                gs.logger.debug(f"[IPC CONTACT] ENABLED: {link_i.name} × {link_j.name}")
                _n_enabled += 1
                self._ipc_contact_tabular.insert(elem_i, elem_j, friction_ij, resistance_ij, True)

        gs.logger.info(f"[IPC CONTACT] ABD x ABD pairs: {_n_enabled} enabled, {_n_disabled} disabled")

        # ---- All contact elements (for ground and no-collision registration) ----
        # is_abd: whether the element is an ABD rigid link
        # is_fixed: whether the element's link is fixed wrt the world
        all_contact_infos: list[tuple[ContactElement, float, float, bool, bool]] = []
        for elem, friction, resistance in non_abd_infos:
            all_contact_infos.append((elem, friction, resistance, False, False))
        for elem, link, friction, resistance in abd_link_infos:
            all_contact_infos.append((elem, friction, resistance, True, _link_is_fixed_for_ipc(link)))

        # Register per-plane ground contact pairs
        for entity, ground_elem in self._ipc_grounds_contact.items():
            plane_friction = entity.material.coup_friction
            plane_resistance = entity.material.contact_resistance or self.options.contact_resistance
            for elem, friction, resistance, is_abd, is_fixed in all_contact_infos:
                enabled = (not is_abd or self.options.enable_rigid_ground_contact) and not is_fixed
                friction_ground = geometric_mean(friction, plane_friction)
                resistance_ground = harmonic_mean(resistance, plane_resistance)
                self._ipc_contact_tabular.insert(ground_elem, elem, friction_ground, resistance_ground, enabled)
            self._ipc_contact_tabular.insert(self._ipc_no_collision_contact, ground_elem, 0.0, 0.0, False)

        # Register no_collision pairs (always disabled)
        for elem, *_ in all_contact_infos:
            self._ipc_contact_tabular.insert(self._ipc_no_collision_contact, elem, 0.0, 0.0, False)
        self._ipc_contact_tabular.insert(
            self._ipc_no_collision_contact, self._ipc_no_collision_contact, 0.0, 0.0, False
        )

    def _finalize_ipc(self):
        """Finalize IPC setup and initialize AffineBodyStateAccessorFeature"""
        assert gs.logger is not None
        assert self._ipc_world is not None
        callback: IPCBeforeWorldInitCallback | None = self.options.before_ipc_world_init
        if callback is not None:
            ipc = self._build_before_ipc_world_init_context()
            try:
                callback(ipc, gs)
            except Exception as exc:
                gs.raise_exception_from("`before_ipc_world_init(ipc, gs)` callback failed.", exc)
        self._ipc_world.init(self._ipc_scene)
        # Checkpoint frame 0 so that recover(0) works in reset().
        self._ipc_world.dump()
        gs.logger.info("IPC world initialized successfully")

    def _build_before_ipc_world_init_context(self) -> IPCBeforeWorldInitContext:
        """Build user callback context passed to before_ipc_world_init(ipc, gs)."""
        return cast(
            IPCBeforeWorldInitContext,
            SimpleNamespace(
                engine=self._ipc_engine,
                world=self._ipc_world,
                scene=self._ipc_scene,
            ),
        )

    def _init_accessors(self):
        assert gs.logger is not None
        assert self._ipc_world is not None

        # ---- ABD state accessor ----
        if self._abd_data_by_link:
            abd_links = list(self._abd_data_by_link.keys())
            n_abd_links = len(abd_links)

            self._abd_state_feature = cast(
                AffineBodyStateAccessorFeature, self._ipc_world.features().find(AffineBodyStateAccessorFeature)
            )
            body_count = self._abd_state_feature.body_count()

            # Verify IPC has at least as many ABD bodies as we expect.
            # Extra bodies may exist from user pre-finalize hooks (e.g. pin rods, joints).
            expected = n_abd_links * self._B
            if body_count < expected:
                gs.raise_exception(f"ABD body count too low: got {body_count}, expected at least {expected}.")

            # Create state geometry for batch data transfer
            self._abd_state_geom = self._abd_state_feature.create_geometry()
            self._abd_state_geom.instances().create(uipc.builtin.transform, np.eye(4, dtype=np.float64))
            self._abd_state_geom.instances().create(uipc.builtin.velocity, np.zeros((4, 4), dtype=np.float64))

        # ---- FEM state accessor ----
        if self._fem_slots_by_entity:
            self._fem_state_feature = cast(
                FiniteElementStateAccessorFeature,
                self._ipc_world.features().find(FiniteElementStateAccessorFeature),
            )
            if self._fem_state_feature is not None:
                self._fem_state_geom = self._fem_state_feature.create_geometry()
                self._fem_state_geom.vertices().create(uipc.builtin.position, np.zeros(3, dtype=np.float64))
                self._fem_state_geom.vertices().create(uipc.builtin.velocity, np.zeros(3, dtype=np.float64))

    def _init_ipc_gui(self):
        """Initialize polyscope-based IPC GUI viewer."""
        try:
            if not ps.is_initialized():
                ps.init()
            ps.set_up_dir("z_up")
            self._ipc_gui = SceneGUI(self._ipc_scene, "split")
            self._ipc_gui.register()

            # Match polyscope camera to Genesis viewer options
            viewer_opts = self.sim.scene.viewer_options
            if viewer_opts is not None:
                cam_pos = np.asarray(viewer_opts.camera_pos, dtype=np.float64)
                cam_lookat = np.asarray(viewer_opts.camera_lookat, dtype=np.float64)
                ps.look_at(cam_pos, cam_lookat)

            ps.show(forFrames=1)
            gs.logger.info("IPC GUI initialized successfully")
        except Exception as e:
            gs.logger.warning(f"IPC GUI unavailable: {e}. Continuing without IPC GUI.")
            self._ipc_gui = None

    # ============================================================
    # Section 2: Core implementation
    # ============================================================

    def preprocess(self, f):
        """Preprocessing step before coupling"""
        pass

    def couple(self, f):
        """
        Execute IPC coupling step with per-entity coupling types.

        This unified coupling flow handles all entity types:
        - 'two_way_soft_constraint': Uses Animator + SoftTransformConstraint
        - 'external_articulation': Uses ExternalArticulationConstraint at joint level
        - 'ipc_only': One-way coupling, IPC controls rigid body transforms

        Flow:
        1. Store Genesis rigid states (common)
        2. Pre-advance processing (per entity type)
        3. IPC advance + retrieve (common, only once)
        4. Retrieve FEM states (common)
        5. Post-advance processing (per entity type)
        """
        assert self._ipc_world is not None

        if not self.is_active:
            return

        import time as _time

        # Step 1: Store Genesis rigid states (common)
        _t0 = _time.perf_counter()
        self._store_gs_rigid_states()
        _t1 = _time.perf_counter()
        if self.options._export_pre_coupling_surface:
            self._export_genesis_surface("after_genesis_before_ipc")

        # Step 2: Pre-advance processing — write all per-frame data to IPC geometries
        self._pre_advance_write_ipc_attributes()
        _t2 = _time.perf_counter()

        # Debug: dump IPC body info on first frame
        if self._ipc_frame == 0 and self.options.verbose_ipc_log:
            gs.logger.info("[IPC DEBUG] === IPC bodies at frame 0, before advance ===")
            for link, abd_data in self._abd_data_by_link.items():
                entity = link.entity
                coup_type = self._coup_type_by_entity.get(entity, "?")
                n_verts = abd_data.n_vertices if hasattr(abd_data, "n_vertices") else "?"
                mass = link.inertial_mass if hasattr(link, "inertial_mass") else "?"
                gs.logger.info(
                    f"[IPC DEBUG]   link={link.name}, entity={entity.name}, "
                    f"coup={coup_type}, verts={n_verts}, mass={mass}, "
                    f"n_geoms={len(link.geoms)}"
                )
            gs.logger.info(f"[IPC DEBUG] Total ABD links: {len(self._abd_data_by_link)}")
            gs.logger.info(f"[IPC DEBUG] Coupling types: {set(self._coup_type_by_entity.values())}")

        # Step 3: IPC advance + retrieve (common)
        _verbose = self.options.verbose_ipc_log
        if not _verbose:
            self._newton_counter.start()
        try:
            self._ipc_world.advance()
        except Exception as e:
            if not _verbose:
                self._newton_counter.stop()
            gs.raise_exception(f"[IPC] advance() failed at frame {self._ipc_frame + 1}: {e}")
        if not _verbose:
            self._newton_counter.stop()
        _n_newton, _n_ls, _ls_max, _newton_max, _tol_fails = self._newton_counter.reset()
        _t3 = _time.perf_counter()
        # Check world validity before retrieve — a failed solver leaves corrupted GPU state
        if not self._ipc_world.is_valid():
            gs.raise_exception(
                f"[IPC] World became invalid after advance at frame {self._ipc_frame + 1}. "
                f"The solver likely hit a numerical failure (newton={_n_newton}, ls={_n_ls})."
            )
        self._ipc_world.retrieve()
        _t4 = _time.perf_counter()
        if self.options._export_ipc_surface:
            self._export_ipc_surface()

        # Step 4: Retrieve states
        self._retrieve_ipc_fem_states()
        self._retrieve_ipc_rigid_states()
        _t5 = _time.perf_counter()

        # Step 5: Post-advance — write IPC-resolved state to qpos
        self._post_advance_write_qpos()
        self._sync_rigid_fk()
        _t6 = _time.perf_counter()
        if self.options._export_post_coupling_surface:
            self._export_genesis_surface("after_ipc_correction")

        self._ipc_frame += 1
        _ls_str = f"  ls_maxout={_ls_max}" if _ls_max > 0 else ""
        _nmax_str = "  NEWTON_MAXOUT" if _newton_max else ""
        if _newton_max and _tol_fails:
            # Show tolerance check status from the last Newton iteration
            _nmax_str += " (" + "; ".join(_tol_fails) + ")"
        gs.logger.info(
            f"[IPC] frame {self._ipc_frame:4d}  newton={_n_newton:2d}  ls={_n_ls:3d}  "
            f"advance={(_t3 - _t2) * 1000:.0f}ms  total={(_t6 - _t0) * 1000:.0f}ms"
            f"{_ls_str}{_nmax_str}"
        )

        # Step 6: Update GUI if enabled
        if self._ipc_gui is not None:
            ps.frame_tick()
            self._ipc_gui.update()

    def couple_grad(self, f):
        """Gradient computation for coupling"""
        gs.raise_exception("couple_grad is not available for IPCCoupler. Please use LegacyCoupler instead.")

    def reset(self, envs_idx=None):
        """Reset coupling state. Per-env reset is not supported by libuipc; envs_idx must cover all envs."""
        assert gs.logger is not None
        assert self._ipc_world is not None
        if envs_idx is not None:
            all_envs = set(range(max(self._B, 1)))
            envs_set = set(int(x) for x in envs_idx) if hasattr(envs_idx, "__iter__") else {int(envs_idx)}
            assert envs_set == all_envs, f"IPC coupler only supports full reset, got envs_idx={envs_idx}"

        self._abd_updated_links.clear()
        self._restitution_vel_corrections.clear()
        self._saved_states.clear()
        self._ipc_frame = 0

        self._ipc_world.recover(0)
        self._ipc_world.retrieve()

        # Re-sync cached IPC transforms from the recovered state
        self._retrieve_ipc_rigid_states()

        # Reset rigid solver's qpos_prev to match qpos (scene.reset restores
        # qpos but not qpos_prev, causing delta_theta = qpos - qpos_prev to be
        # huge on the next step).
        if self.rigid_solver.is_active:
            qpos_tc = qd_to_torch(self.rigid_solver.qpos, copy=False)
            qpos_prev_tc = qd_to_torch(self.rigid_solver.qpos_prev, copy=False)
            qpos_prev_tc.copy_(qpos_tc)

            # Reset articulation cached state (read by _store_gs_rigid_states
            # on the next step). IPC geometry attributes (delta_theta_tilde,
            # mass_matrix, ref_dof_prev) are overwritten by
            # _pre_advance_write_ipc_attributes before each advance().
            qpos = qd_to_numpy(self.rigid_solver.qpos, transpose=True)
            for ad in self._articulation_data_by_entity.values():
                ad.delta_theta_tilde[:] = 0.0
                ad.prev_qpos[:] = qpos[..., ad.q_slice]

    # ------------------------------------------------------------------
    # Manual save / load
    # ------------------------------------------------------------------

    def save_state(self) -> int:
        """Save the current IPC + Genesis coupling state and return the frame id.

        The saved frame can later be restored with :meth:`load_state`.
        Internally this calls ``libuipc World.dump()`` and snapshots the
        Genesis-side cached arrays (qpos_prev, articulation data, ABD
        transforms) so they can be restored consistently.

        Returns
        -------
        int
            The IPC frame number of the saved state.
        """
        assert gs.logger is not None
        assert self._ipc_world is not None

        ok = self._ipc_world.dump()
        if not ok:
            gs.raise_exception(f"[IPC] dump() failed at frame {self._ipc_frame}")

        # Snapshot Genesis-side state that is NOT stored inside libuipc
        snapshot: dict = {
            "ipc_frame": self._ipc_frame,
        }
        if self.rigid_solver.is_active:
            snapshot["qpos"] = qd_to_numpy(self.rigid_solver.qpos, transpose=True).copy()
            snapshot["qpos_prev"] = qd_to_numpy(self.rigid_solver.qpos_prev, transpose=True).copy()
        for entity, ad in self._articulation_data_by_entity.items():
            snapshot[("art_delta_theta", id(entity))] = (
                ad.delta_theta_tilde.copy() if ad.delta_theta_tilde is not None else None
            )
            snapshot[("art_prev_qpos", id(entity))] = ad.prev_qpos.copy() if ad.prev_qpos is not None else None
        for link, abd_data in self._abd_data_by_link.items():
            snapshot[("abd_ipc_transforms", id(link))] = (
                abd_data.ipc_transforms.copy() if abd_data.ipc_transforms is not None else None
            )
            snapshot[("abd_ipc_velocities", id(link))] = (
                abd_data.ipc_velocities.copy() if abd_data.ipc_velocities is not None else None
            )
            snapshot[("abd_aim_transforms", id(link))] = (
                abd_data.aim_transforms.copy() if abd_data.aim_transforms is not None else None
            )

        self._saved_states[self._ipc_frame] = snapshot
        gs.logger.info(f"[IPC] Saved state at frame {self._ipc_frame}")
        return self._ipc_frame

    def load_state(self, frame: int) -> None:
        """Restore a previously saved IPC + Genesis coupling state.

        Parameters
        ----------
        frame : int
            The frame number returned by :meth:`save_state`.
        """
        assert gs.logger is not None
        assert self._ipc_world is not None

        if frame not in self._saved_states:
            gs.raise_exception(f"[IPC] No saved state at frame {frame}. Available: {sorted(self._saved_states.keys())}")

        # Recover libuipc side
        ok = self._ipc_world.recover(frame)
        if not ok:
            gs.raise_exception(f"[IPC] recover({frame}) failed")
        self._ipc_world.retrieve()

        # Restore Genesis-side snapshot
        snapshot = self._saved_states[frame]
        self._ipc_frame = snapshot["ipc_frame"]
        self._abd_updated_links.clear()
        self._restitution_vel_corrections.clear()

        # Re-sync cached IPC transforms
        self._retrieve_ipc_rigid_states()

        # Restore qpos and qpos_prev
        if self.rigid_solver.is_active and "qpos" in snapshot:
            qpos_tc = qd_to_torch(self.rigid_solver.qpos, copy=False)
            qpos_prev_tc = qd_to_torch(self.rigid_solver.qpos_prev, copy=False)
            saved_qpos = torch.as_tensor(snapshot["qpos"], dtype=qpos_tc.dtype, device=qpos_tc.device)
            saved_prev = torch.as_tensor(snapshot["qpos_prev"], dtype=qpos_tc.dtype, device=qpos_tc.device)
            # qpos/qpos_prev are stored as (B, n_qs) — match the transposed layout
            if saved_qpos.shape == qpos_tc.shape:
                qpos_tc.copy_(saved_qpos)
                qpos_prev_tc.copy_(saved_prev)
            else:
                # Transposed: (n_qs, B) in qd vs (B, n_qs) in snapshot
                qpos_tc.copy_(saved_qpos.T)
                qpos_prev_tc.copy_(saved_prev.T)

        # Restore articulation cached state
        for entity, ad in self._articulation_data_by_entity.items():
            key_dt = ("art_delta_theta", id(entity))
            key_pq = ("art_prev_qpos", id(entity))
            if key_dt in snapshot and snapshot[key_dt] is not None:
                ad.delta_theta_tilde[:] = snapshot[key_dt]
            if key_pq in snapshot and snapshot[key_pq] is not None:
                ad.prev_qpos[:] = snapshot[key_pq]

        # Restore ABD cached transforms
        for link, abd_data in self._abd_data_by_link.items():
            key_t = ("abd_ipc_transforms", id(link))
            key_v = ("abd_ipc_velocities", id(link))
            key_a = ("abd_aim_transforms", id(link))
            if key_t in snapshot and snapshot[key_t] is not None:
                abd_data.ipc_transforms[:] = snapshot[key_t]
            if key_v in snapshot and snapshot[key_v] is not None:
                abd_data.ipc_velocities[:] = snapshot[key_v]
            if key_a in snapshot and snapshot[key_a] is not None:
                abd_data.aim_transforms[:] = snapshot[key_a]

        # Write restored IPC state back to the rigid solver and run FK
        # so that entity.get_pos()/get_quat() reflect the restored state.
        self._post_advance_write_qpos()
        self._sync_rigid_fk()

        gs.logger.info(f"[IPC] Loaded state from frame {frame}")

    def _mark_abd_link_updated(self, link: "RigidLink", env_set: set[int]):
        """Add a link to the updated set for the given environments."""
        existing = self._abd_updated_links.get(link)
        if existing is None:
            self._abd_updated_links[link] = env_set.copy()
        else:
            existing.update(env_set)

    def mark_abd_updated(self, qs_idx=None, dofs_idx=None, links_idx=None, envs_idx=None):
        """Mark ABD links as needing IPC state sync.

        Parameters
        ----------
        qs_idx : array_like | None
            Global qpos indices that were modified.
        dofs_idx : array_like | None
            Global dof indices that were modified.
        links_idx : array_like | None
            Global link indices that were modified.
        envs_idx : array_like | None
            Environment indices affected. None means all environments.

        If qs_idx, dofs_idx, and links_idx are all None, ALL coupled links are marked.
        """
        if not self._abd_data_by_link:
            return
        all_envs = set(range(self._B)) if self._B > 0 else {0}
        env_set = all_envs if envs_idx is None else set(int(i) for i in envs_idx)

        if qs_idx is None and dofs_idx is None and links_idx is None:
            for link in self._abd_data_by_link:
                self._mark_abd_link_updated(link, env_set)
            return

        if qs_idx is not None:
            if isinstance(qs_idx, slice):
                qs_idx = range(*qs_idx.indices(len(self._q_to_abd_link)))
            for qi in qs_idx:
                link = self._q_to_abd_link[int(qi)]
                if link is not None:
                    self._mark_abd_link_updated(link, env_set)

        if dofs_idx is not None:
            if isinstance(dofs_idx, slice):
                dofs_idx = range(*dofs_idx.indices(len(self._dof_to_abd_link)))
            for di in dofs_idx:
                link = self._dof_to_abd_link[int(di)]
                if link is not None:
                    self._mark_abd_link_updated(link, env_set)

        if links_idx is not None:
            for li in links_idx:
                link = self._link_to_abd_link[int(li)]
                if link is not None:
                    self._mark_abd_link_updated(link, env_set)

    def cache_pre_prediction_transforms(self):
        """
        Sync IPC ABD body transforms from current (pre-prediction) link poses.

        Called by RigidSolver before kernel_predict_integrate. At this point
        links_state reflects actual poses (including any set_qpos changes) before
        prediction overwrites them. Only updated (link, env) pairs are synced.

        For external_articulation entities, we also write the teleported
        transform to each dirty link's slot geometry. This ensures the
        animator callback (which runs inside advance()) reads the teleported
        pose when computing ``ref_dof_prev``, rather than the stale
        previous-frame pose from the slot geometry.
        """
        if not self._abd_updated_links or self._abd_state_feature is None:
            return
        # DEBUG: log when teleport sync is triggered
        dirty_names = [f"{link.name}(envs={envs})" for link, envs in self._abd_updated_links.items()]
        gs.logger.warning(f"[IPC TELEPORT SYNC] {len(self._abd_updated_links)} dirty links: {dirty_names}")

        assert self._abd_state_geom is not None

        links_pos = qd_to_numpy(self.rigid_solver.links_state.pos, transpose=True)
        links_quat = qd_to_numpy(self.rigid_solver.links_state.quat, transpose=True)
        links_transform = gu.trans_quat_to_T(links_pos, links_quat)

        self._abd_state_feature.copy_to(self._abd_state_geom)
        trans_attr = self._abd_state_geom.instances().find(uipc.builtin.transform)
        transforms = trans_attr.view()
        vel_attr = self._abd_state_geom.instances().find(uipc.builtin.velocity)
        velocities = vel_attr.view() if vel_attr is not None else None

        for i_link, (link, abd_data) in enumerate(self._abd_data_by_link.items()):
            dirty_envs = self._abd_updated_links.get(link)
            if dirty_envs is None:
                continue

            for env_idx in dirty_envs:
                abd_body_idx = i_link * self._B + env_idx
                new_T = links_transform[env_idx, link.idx]
                transforms[abd_body_idx] = new_T

                # Zero velocity so IPC's time integrator doesn't see phantom
                # motion from the teleported position.
                if velocities is not None:
                    velocities[abd_body_idx] = np.zeros((4, 4), dtype=new_T.dtype)

                # Update ipc_transforms so _pre_advance_write_ipc_attributes
                # writes the correct ref_dof_prev for ext_art links.
                abd_data.ipc_transforms[env_idx] = new_T

        self._abd_state_feature.copy_from(self._abd_state_geom)

        self._abd_updated_links.clear()

    def set_fem_external_force(self, entity: "FEMEntity", forces: np.ndarray) -> None:
        """Set per-vertex external forces for a FEM entity.

        Forces are written to IPC geometry before each advance().
        Requires FiniteElementExternalForce constitution on the entity (applied
        automatically for cloth/paper in _add_fem_entities_to_ipc).

        Parameters
        ----------
        entity : FEMEntity
            The FEM entity to apply forces to.
        forces : np.ndarray
            Per-vertex force vectors, shape (n_vertices, 3).
        """
        self._fem_external_forces[entity] = forces.astype(np.float64)

    def clear_fem_external_force(self, entity: "FEMEntity") -> None:
        """Clear external forces for a FEM entity.

        Zeros are written on the next advance, then the entry is removed.
        """
        if entity in self._fem_external_forces:
            self._fem_external_forces[entity] = None

    def set_fem_soft_position(
        self,
        entity: "FEMEntity",
        strength_ratio: np.ndarray,
        aim_position: np.ndarray,
    ) -> None:
        """Set per-vertex soft position constraint for a FEM entity.

        Parameters
        ----------
        entity : FEMEntity
            The FEM entity.
        strength_ratio : np.ndarray
            Per-vertex strength, shape (n_vertices,). Use 0 to disable on a vertex.
        aim_position : np.ndarray
            Per-vertex target positions, shape (n_vertices, 3).
        """
        self._fem_soft_position[entity] = {
            "strength_ratio": strength_ratio.astype(np.float64),
            "aim_position": aim_position.astype(np.float64),
        }

    def clear_fem_soft_position(self, entity: "FEMEntity") -> None:
        """Clear soft position constraint for a FEM entity."""
        if entity in self._fem_soft_position:
            self._fem_soft_position[entity] = None

    def freeze_plastic_bending(self, entity: "FEMEntity", new_bending_stiffness: float = 0.0) -> None:
        """Freeze plasticity on all environment copies of an entity's shell mesh.

        Sets per-edge attribute "cancel_plastic" = 1 (all edges) and
        optionally "target_bending_stiffness".  Individual edges can also be
        controlled directly via the slot geometry.

        Args:
            entity: The FEM entity whose plasticity should be frozen.
            new_bending_stiffness: Bending stiffness after freeze (Pa·m).
                                   Pass 0.0 to keep the original value.
        """
        slots = self._fem_slots_by_entity.get(entity)
        if slots is None:
            return
        for env_idx in range(self._B):
            geom = slots[env_idx].geometry()
            freeze_attr = geom.edges().find("cancel_plastic")
            if freeze_attr is not None:
                uipc.view(freeze_attr)[:] = 1
            if new_bending_stiffness > 0.0:
                stiffness_attr = geom.edges().find("target_bending_stiffness")
                if stiffness_attr is not None:
                    uipc.view(stiffness_attr)[:] = new_bending_stiffness

    def mark_fem_updated(self, entity: "FEMEntity", envs_idx=None):
        """Mark a FEM entity as needing IPC position sync."""
        if entity not in self._fem_slots_by_entity:
            return
        all_envs = set(range(self._B)) if self._B > 0 else {0}
        env_set = all_envs if envs_idx is None else set(int(i) for i in envs_idx)
        existing = self._fem_updated_entities.get(entity)
        if existing is None:
            self._fem_updated_entities[entity] = env_set.copy()
        else:
            existing.update(env_set)

    def cache_fem_positions(self):
        """Sync FEM vertex positions from Genesis to IPC before advance.

        Called when the user modifies FEM positions via set_position().
        Writes the new positions (and zero velocity) to IPC's internal
        FEM state via FiniteElementStateAccessorFeature.copy_from().
        """
        if not self._fem_updated_entities or self._fem_state_feature is None:
            return

        assert self._fem_state_geom is not None

        self._fem_state_feature.copy_to(self._fem_state_geom)
        pos_attr = self._fem_state_geom.vertices().find(uipc.builtin.position)
        vel_attr = self._fem_state_geom.vertices().find(uipc.builtin.velocity)
        positions = pos_attr.view()
        velocities = vel_attr.view() if vel_attr is not None else None

        # FEM vertex order in IPC matches the order geometries were added.
        # Each entity-env pair occupies a contiguous block of vertices.
        # Read current positions from the solver state via get_frame().
        offset = 0
        for entity, slots in self._fem_slots_by_entity.items():
            n_verts = entity.n_vertices
            dirty_envs = self._fem_updated_entities.get(entity)
            if dirty_envs is not None:
                state = entity.get_state()
                entity_pos = state.pos.cpu().numpy()
            for env_idx, slot in enumerate(slots):
                if dirty_envs is not None and env_idx in dirty_envs:
                    # IPC stores positions as (N, 3, 1) column vectors
                    positions[offset : offset + n_verts] = entity_pos[env_idx].astype(np.float64).reshape(-1, 3, 1)
                    if velocities is not None:
                        velocities[offset : offset + n_verts] = 0.0
                offset += n_verts

        self._fem_state_feature.copy_from(self._fem_state_geom)
        self._fem_updated_entities.clear()

    @property
    def is_active(self) -> bool:
        """Check if IPC coupling is active"""
        return self._ipc_world is not None

    def _export_ipc_surface(self):
        """Export IPC scene surface snapshots after retrieve().

        Controlled by IPCCouplerOptions private debug fields:
        - _export_ipc_surface
        - _export_surface_dir
        """
        output_dir = self.options._export_surface_dir or self._ipc_workspace
        if output_dir is None:
            output_dir = tempfile.gettempdir()
        os.makedirs(output_dir, exist_ok=True)

        stem = f"{IPC_SURFACE_PREFIX}_{self._debug_surface_export_idx:06d}"
        output_path = os.path.join(output_dir, f"{stem}.obj")

        try:
            scene_io = SceneIO(self._ipc_scene)
            exported = False
            for method_name in ("write_surface", "write_surface_obj", "export_surface"):
                method = getattr(scene_io, method_name, None)
                if method is None:
                    continue
                try:
                    method(output_path)
                except TypeError:
                    # Some bindings may accept (directory, stem) instead of full filepath.
                    method(output_dir, stem)
                exported = True
                break

            if not exported:
                raise AttributeError("SceneIO has no supported surface export method.")

            self._debug_surface_export_idx += 1
        except Exception as exc:
            assert gs.logger is not None
            gs.logger.warning(f"Failed to export IPC debug surface snapshot: {exc}")

    def _export_genesis_surface(self, phase: str):
        """Export current Genesis rigid geometry as one combined OBJ snapshot."""
        if not self.rigid_solver.is_active:
            return
        if phase not in ("after_genesis_before_ipc", "after_ipc_correction"):
            gs.raise_exception(f"Unknown Genesis surface export phase: {phase}")
        # Ensure links_state/geoms_state are refreshed from the latest qpos before exporting.
        self._sync_rigid_fk()

        output_dir = self.options._export_surface_dir or self._ipc_workspace
        if output_dir is None:
            output_dir = tempfile.gettempdir()
        os.makedirs(output_dir, exist_ok=True)

        if phase == "after_genesis_before_ipc":
            frame_idx = self._debug_genesis_surface_export_after_genesis_before_ipc_idx
        else:
            frame_idx = self._debug_genesis_surface_export_after_ipc_correction_idx
        stem = f"{GENESIS_SURFACE_PREFIX}_{phase}_{frame_idx:06d}"
        output_path = os.path.join(output_dir, f"{stem}.obj")

        env_idx = 0
        links_pos = qd_to_numpy(self.rigid_solver.links_state.pos, transpose=True)
        links_quat = qd_to_numpy(self.rigid_solver.links_state.quat, transpose=True)

        obj_lines: list[str] = []
        vert_offset = 1

        for link in self.rigid_solver.links:
            link_pos = links_pos[env_idx, link.idx]
            link_quat = links_quat[env_idx, link.idx]
            for geom in link.geoms:
                if geom.type == gs.GEOM_TYPE.PLANE or geom.n_verts <= 0:
                    continue
                verts_link = gu.transform_by_trans_quat(geom.init_verts, geom.init_pos, geom.init_quat)
                verts_world = gu.transform_by_trans_quat(verts_link, link_pos, link_quat)
                faces = geom.init_faces.astype(np.int64, copy=False)

                obj_lines.append(f"o link_{link.idx}_{link.name}_geom_{geom.idx}")
                for v in verts_world:
                    obj_lines.append(f"v {float(v[0]):.9g} {float(v[1]):.9g} {float(v[2]):.9g}")
                for f in faces:
                    i0, i1, i2 = int(f[0]) + vert_offset, int(f[1]) + vert_offset, int(f[2]) + vert_offset
                    obj_lines.append(f"f {i0} {i1} {i2}")
                vert_offset += len(verts_world)

        try:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write("# Genesis rigid geometry snapshot\n")
                f.write("\n".join(obj_lines))
                f.write("\n")
            if phase == "after_genesis_before_ipc":
                self._debug_genesis_surface_export_after_genesis_before_ipc_idx += 1
            else:
                self._debug_genesis_surface_export_after_ipc_correction_idx += 1
        except Exception as exc:
            assert gs.logger is not None
            gs.logger.warning(f"Failed to export Genesis debug surface snapshot: {exc}")

    @property
    def has_any_rigid_coupling(self) -> bool:
        """
        Check if any rigid entity is coupled to IPC.

        Returns
        -------
        bool
            True if at least one rigid entity has a coupling type (two_way_soft_constraint,
            external_articulation, or ipc_only).
        """
        return bool(self._coup_type_by_entity)

    # ============================================================
    # Section 3: Helpers
    # ============================================================
    # Animator callbacks removed — all per-frame IPC attribute writes are now
    # in _pre_advance_write_ipc_attributes(), avoiding C++→Python→C++ overhead.

    def _retrieve_ipc_fem_states(self):
        # IPC world advance/retrieve is handled at Scene level
        # This method handles both volumetric FEM (3D) and cloth (2D) post-processing

        if not self.fem_solver.is_active:
            return

        # Gather FEM states (both volumetric and cloth) using metadata filtering
        visitor = SceneVisitor(self._ipc_scene)

        # Collect FEM and cloth geometries using metadata
        fem_entities = cast(list["FEMEntity"], self.fem_solver.entities)
        fem_positions_by_entity: dict["FEMEntity", list[np.ndarray]] = {
            entity: [np.array([]) for _ in range(self._B)] for entity in fem_entities
        }
        for fem_geom_slot in visitor.geometries():
            if not isinstance(fem_geom_slot, SimplicialComplexSlot):
                continue

            fem_geom = fem_geom_slot.geometry()
            if fem_geom.dim() not in (1, 2, 3):
                continue
            meta = read_ipc_geometry_metadata(fem_geom)
            if meta is None:
                continue
            solver_type, env_idx, i_e = meta
            if solver_type not in ("fem", "cloth", "rope"):
                continue

            entity = cast("FEMEntity", self.fem_solver.entities[i_e])
            # uipc.view(fem_geom.transforms())[:] = uipc.Matrix4x4.Identity()
            (transformed_geom,) = uipc.geometry.apply_transform(fem_geom)
            fem_positions_by_entity[entity][env_idx] = transformed_geom.positions().view().reshape(-1, 3)

        # Update FEM entities using filtered geometries
        for entity, geom_positions in fem_positions_by_entity.items():
            geom_positions = np.stack(geom_positions, axis=0, dtype=gs.np_float)
            entity.set_pos(0, geom_positions)

    def _retrieve_ipc_rigid_states(self):
        """
        Retrieve ABD transforms/affine matrices after IPC step using AffineBodyStateAccessorFeature.

        O(num_rigid_bodies) instead of O(total_geometries).
        Also populates data arrays for force computation.
        """
        if self._abd_state_feature is None:
            return

        # Single batch copy of ALL ABD states from IPC
        assert self._abd_state_geom is not None
        self._abd_state_feature.copy_to(self._abd_state_geom)

        # Get all transforms at once (array view)
        trans_attr = self._abd_state_geom.instances().find(uipc.builtin.transform)
        # Shape: (num_bodies, 4, 4)
        transforms = trans_attr.view()

        # Get velocities (4x4 matrix representing transform derivative)
        vel_attr = self._abd_state_geom.instances().find(uipc.builtin.velocity)
        # Shape: (num_bodies, 4, 4)
        velocities = vel_attr.view()

        for i_link, (link, abd_data) in enumerate(self._abd_data_by_link.items()):
            for env_idx in range(self._B):
                abd_body_idx = i_link * self._B + env_idx
                abd_data.ipc_transforms[env_idx] = transforms[abd_body_idx]
                abd_data.ipc_velocities[env_idx] = velocities[abd_body_idx]

    def _store_gs_rigid_states(self):
        """
        Store predicted Genesis rigid body states before IPC advance.

        After kernel_predict_integrate + FK, qpos and links_state contain predicted values.
        These are cached so that _pre_advance/_post_advance methods don't need to
        reach back into the rigid solver for reads.

        Note: IPC-only entities have no animator (external_kinetic=0), so stored
        transforms are unused by IPC for them.
        """
        if not self.rigid_solver.is_active:
            return

        # Cache per-entity qpos slices for external articulation
        if self._articulation_data_by_entity:
            qpos = qd_to_numpy(self.rigid_solver.qpos, transpose=True)
            qpos_prev = qd_to_numpy(self.rigid_solver.qpos_prev, transpose=True)
            mass_matrix = qd_to_numpy(self.rigid_solver.mass_mat, transpose=True)

            for ad in self._articulation_data_by_entity.values():
                entity_qpos = qpos[..., ad.q_slice]
                entity_qpos_prev = qpos_prev[..., ad.q_slice]
                ad.delta_theta_tilde[:] = (
                    entity_qpos[..., ad.joints_qs_idx_local] - entity_qpos_prev[..., ad.joints_qs_idx_local]
                )
                ad.prev_qpos[:] = entity_qpos_prev
                entity_mass = mass_matrix[:, ad.dof_slice, ad.dof_slice]
                ad.mass_matrix[:] = entity_mass

                # Debug: check mass matrix for NaN/Inf/singularity
                if self.options.verbose_ipc_log:
                    import numpy as _np

                    for env_idx in range(self._B):
                        M = entity_mass[env_idx]
                        has_nan = _np.any(_np.isnan(M))
                        has_inf = _np.any(_np.isinf(M))
                        diag = _np.diag(M)
                        min_diag = _np.min(diag)
                        max_diag = _np.max(diag)
                        det = _np.linalg.det(M) if M.shape[0] <= 20 else float("nan")
                        gs.logger.info(
                            f"[IPC DEBUG] mass_matrix env={env_idx}: shape={M.shape}, "
                            f"nan={has_nan}, inf={has_inf}, diag_range=[{min_diag:.6e}, {max_diag:.6e}], "
                            f"det={det:.6e}, delta_theta={ad.delta_theta_tilde[env_idx]}"
                        )

        # Store transforms for all rigid links
        links_pos = qd_to_numpy(self.rigid_solver.links_state.pos, transpose=True)
        links_quat = qd_to_numpy(self.rigid_solver.links_state.quat, transpose=True)
        links_transform = gu.trans_quat_to_T(links_pos, links_quat)
        for link, abd_data in self._abd_data_by_link.items():
            abd_data.aim_transforms[:] = links_transform[:, link.idx]

    def _pre_advance_write_ipc_attributes(self):
        """Write all per-frame IPC attributes before advance().

        Replaces animator callbacks — writes aim_transform (two-way),
        ref_dof_prev (ext_art), delta_theta_tilde and mass_matrix (ext_art)
        directly to IPC geometries.
        """
        # 1. Write aim_transform for two_way_soft_constraint links
        for link, abd_data in self._abd_data_by_link.items():
            coup_type = self._coup_type_by_entity.get(link.entity)
            if coup_type != COUPLING_TYPE.TWO_WAY_SOFT_CONSTRAINT:
                continue
            for env_idx in range(self._B):
                geom = abd_data.slots[env_idx].geometry()
                aim_transform_attr = geom.instances().find(uipc.builtin.aim_transform)
                uipc.view(aim_transform_attr)[:] = abd_data.aim_transforms[env_idx]

        # 2. Write ref_dof_prev for external_articulation links
        # Uses ipc_transforms (synced from copy_from via libuipc's write_scene)
        # instead of reading geom.transforms()
        for link, abd_data in self._abd_data_by_link.items():
            coup_type = self._coup_type_by_entity.get(link.entity)
            if coup_type != COUPLING_TYPE.EXTERNAL_ARTICULATION:
                continue
            for env_idx in range(self._B):
                geom = abd_data.slots[env_idx].geometry()
                ref_attr = geom.instances().find("ref_dof_prev")
                if ref_attr is None:
                    continue
                T = abd_data.ipc_transforms[env_idx]
                uipc.view(ref_attr)[0] = uipc.geometry.affine_body.transform_to_q(np.asarray(T, dtype=np.float64))

        # 3. Write delta_theta_tilde and mass_matrix for external_articulation
        for ad in self._articulation_data_by_entity.values():
            for env_idx in range(self._B):
                articulation_geom = ad.slots[env_idx].geometry()

                delta_theta_tilde_attr = articulation_geom["joint"].find("delta_theta_tilde")
                uipc.view(delta_theta_tilde_attr)[:] = ad.delta_theta_tilde[env_idx]

                mass_matrix_attr = articulation_geom["joint_joint"].find("mass")
                uipc.view(mass_matrix_attr).flat[:] = ad.mass_matrix[env_idx]

        # 4. Write FEM external forces
        for entity, forces in list(self._fem_external_forces.items()):
            slots = self._fem_slots_by_entity.get(entity)
            if slots is None:
                continue
            for env_idx in range(self._B):
                geom = slots[env_idx].geometry()
                force_attr = geom.vertices().find("external_force")
                is_constrained = geom.vertices().find(uipc.builtin.is_constrained)
                if force_attr is None or is_constrained is None:
                    continue
                fv = uipc.view(force_attr)
                cv = uipc.view(is_constrained)
                if forces is not None:
                    cv[:] = 1
                    fv[:] = forces.reshape(-1, 3, 1)
                else:
                    cv[:] = 0
                    fv[:] = 0.0
            if forces is None:
                del self._fem_external_forces[entity]

        # 5. Write FEM soft position constraints
        # Must run AFTER section 4 so is_constrained for pinned vertices
        # is not clobbered by the external force clear (cv[:] = 0).
        for entity, data in list(self._fem_soft_position.items()):
            slots = self._fem_slots_by_entity.get(entity)
            if slots is None:
                continue
            for env_idx in range(self._B):
                geom = slots[env_idx].geometry()
                sr_attr = geom.vertices().find("strength_ratio")
                aim_attr = geom.vertices().find(uipc.builtin.aim_position)
                is_constrained = geom.vertices().find(uipc.builtin.is_constrained)
                if sr_attr is None or aim_attr is None:
                    continue
                sr = uipc.view(sr_attr)
                ap = uipc.view(aim_attr)
                if data is not None:
                    sr[:] = data["strength_ratio"]
                    ap[:] = data["aim_position"].reshape(-1, 3, 1)
                    # Ensure is_constrained=1 for vertices with non-zero strength
                    # so the SoftPositionConstraint backend includes them.
                    if is_constrained is not None:
                        cv = uipc.view(is_constrained)
                        pinned = data["strength_ratio"] > 0
                        for k in range(len(pinned)):
                            if pinned[k]:
                                cv[k] = 1
                else:
                    sr[:] = 0.0
            if data is None:
                del self._fem_soft_position[entity]

    def _post_advance_write_qpos(self):
        """
        Write IPC-resolved state back into rigid_global_info.qpos (predicted).

        kernel_restore_integrate will back-compute velocity/acceleration for step_2 to land
        on IPC's target positions.

        For two_way_soft_constraint, non-fixed base links get their IPC-resolved transform
        written to qpos[0:7], and child link joint angles are back-computed from IPC transforms.
        For external_articulation (fixed base only), joint qpos comes from IPC delta_theta.

        When restitution > 0, accumulates per-DOF velocity corrections during active contact
        and flushes them as one-shot impulses when contact ends. This avoids the per-frame
        compounding problem where e^N -> 0 for e<1 over N contact frames.
        """
        if not self._coup_type_by_entity:
            return

        e = self.options.restitution
        dt = self.rigid_solver.substep_dt
        qpos_tc = qd_to_torch(self.rigid_solver.qpos, transpose=True, copy=False)

        # Read predicted qpos (q_pred) before overwriting — needed for restitution
        if e > 0:
            qpos_pred_np = qpos_tc.cpu().numpy().copy()

        # Clear one-shot impulses from previous step
        self._restitution_vel_corrections = []

        # ---- Step 1a: ipc_only base links — write IPC transform to links_state directly ----
        # ipc_only entities have FIXED joints (0 qs/DOFs), so we write to links_state.pos/quat
        # instead of qpos. The FK kernel propagates FIXED root link transforms from links_state.
        links_pos_tc = qd_to_torch(self.rigid_solver.links_state.pos, transpose=True, copy=False)
        links_quat_tc = qd_to_torch(self.rigid_solver.links_state.quat, transpose=True, copy=False)
        for link, abd_data in self._abd_data_by_link.items():
            entity = link.entity
            if not (entity.delegation is not None):
                continue
            if link is not entity.base_link:
                continue

            envs_pos = np.empty((self._B, 3), dtype=gs.np_float)
            envs_quat = np.empty((self._B, 4), dtype=gs.np_float)
            for env_idx in range(self._B):
                envs_pos[env_idx], envs_quat[env_idx] = gu.T_to_trans_quat(abd_data.ipc_transforms[env_idx])
            links_pos_tc[:, link.idx] = torch.from_numpy(envs_pos).to(links_pos_tc.device)
            links_quat_tc[:, link.idx] = torch.from_numpy(envs_quat).to(links_quat_tc.device)

        # ---- Step 1b: Non-fixed base links — write IPC transform to qpos[0:7] ----
        for link, abd_data in self._abd_data_by_link.items():
            entity = link.entity
            if entity.delegation is not None:
                continue
            if link is not entity.base_link or entity.base_link.is_fixed:
                continue

            q_start = entity.q_start
            dof_start = entity.dof_start
            envs_qpos = np.empty((self._B, 7), dtype=gs.np_float)
            for env_idx in range(self._B):
                envs_qpos[env_idx, :3], envs_qpos[env_idx, 3:7] = gu.T_to_trans_quat(abd_data.ipc_transforms[env_idx])
            qpos_tc[:, q_start : q_start + 7] = torch.from_numpy(envs_qpos).to(qpos_tc.device)

            if e > 0:
                self._accumulate_restitution_base_link(
                    dof_start,
                    envs_qpos,
                    qpos_pred_np[:, q_start : q_start + 7],
                    dt,
                )

        # ---- Step 2a: Two-way child links — back-compute joint angles from IPC transforms ----
        if COUPLING_TYPE.TWO_WAY_SOFT_CONSTRAINT in self._entities_by_coup_type:
            qpos0 = qd_to_numpy(self.rigid_solver.qpos0, transpose=True)
            links_pos = qd_to_numpy(self.rigid_solver.links_state.pos, transpose=True)
            links_quat = qd_to_numpy(self.rigid_solver.links_state.quat, transpose=True)

            for link, abd_data in self._abd_data_by_link.items():
                entity = link.entity
                if self._coup_type_by_entity.get(entity) != COUPLING_TYPE.TWO_WAY_SOFT_CONSTRAINT:
                    continue
                if link is entity.base_link or link.parent_idx == -1:
                    continue

                parent_link = entity.links[link.parent_idx - entity.link_start]
                joint = link.joints[0]
                if joint.type not in (gs.JOINT_TYPE.REVOLUTE, gs.JOINT_TYPE.PRISMATIC):
                    continue
                q_idx = joint.q_start
                envs_q = np.empty((self._B, 1), dtype=gs.np_float)
                for env_idx in range(self._B):
                    parent_abd = self._abd_data_by_link.get(parent_link)
                    if parent_abd is not None and parent_abd.ipc_transforms is not None:
                        parent_T = parent_abd.ipc_transforms[env_idx]
                        parent_quat = gu.T_to_trans_quat(parent_T)[1]
                    else:
                        parent_T = gu.trans_quat_to_T(
                            links_pos[env_idx, parent_link.idx], links_quat[env_idx, parent_link.idx]
                        )
                        parent_quat = links_quat[env_idx, parent_link.idx]
                    child_T = abd_data.ipc_transforms[env_idx]
                    child_quat_pre = gu.transform_quat_by_quat(
                        np.asarray(link.quat, dtype=parent_quat.dtype), parent_quat
                    )
                    if joint.type == gs.JOINT_TYPE.REVOLUTE:
                        child_quat = gu.T_to_trans_quat(child_T)[1]
                        qloc = gu.transform_quat_by_quat(child_quat, gu.inv_quat(child_quat_pre))
                        rotvec = gu.quat_to_rotvec(qloc)
                        axis = np.asarray(joint._dofs_motion_ang[0], dtype=rotvec.dtype)
                        angle_ipc = float(np.dot(rotvec, axis))
                    else:  # PRISMATIC
                        child_pos = child_T[:3, 3]
                        parent_pos = parent_T[:3, 3]
                        link_offset_pos = np.asarray(link.pos, dtype=parent_pos.dtype)
                        pos_pre = parent_pos + gu.transform_by_quat(link_offset_pos, parent_quat)
                        axis = np.asarray(joint._dofs_motion_vel[0], dtype=pos_pre.dtype)
                        xaxis = gu.transform_by_quat(axis, child_quat_pre)
                        angle_ipc = float(np.dot(child_pos - pos_pre, xaxis))
                    envs_q[env_idx, 0] = qpos0[env_idx, q_idx] + angle_ipc
                qpos_tc[:, q_idx : q_idx + 1] = torch.from_numpy(envs_q).to(qpos_tc.device)

        # ---- Step 2b: External articulation — read delta_theta, write joint qpos ----
        for ext_art_entity, ad in self._articulation_data_by_entity.items():
            delta_theta_ipc = np.empty((self._B, len(ad.joints_qs_idx_local)), dtype=np.float64)
            for env_idx in range(self._B):
                articulation_geom = ad.slots[env_idx].geometry()
                delta_theta_attr = articulation_geom["joint"].find("delta_theta")
                delta_theta_ipc[env_idx] = delta_theta_attr.view()

            np.copyto(ad.ipc_qpos, ad.prev_qpos, casting="same_kind")
            ad.ipc_qpos[..., ad.joints_qs_idx_local] += delta_theta_ipc
            # Base link qpos[0:7] already handled in Step 1 for non-fixed base;
            # only write joint DOFs here.
            global_qs = [ad.q_slice.start + qi for qi in ad.joints_qs_idx_local]
            qpos_tc[:, global_qs] = torch.from_numpy(ad.ipc_qpos[..., ad.joints_qs_idx_local]).to(qpos_tc.device)

    def _sync_rigid_fk(self):
        """Explicitly run FK to sync qpos with link/geom transforms."""
        if not self.rigid_solver.is_active:
            return
        from genesis.engine.solvers.rigid.abd.forward_kinematics import kernel_forward_kinematics_links_geoms

        kernel_forward_kinematics_links_geoms(
            self.sim.scene._envs_idx,
            links_state=self.rigid_solver.links_state,
            links_info=self.rigid_solver.links_info,
            joints_state=self.rigid_solver.joints_state,
            joints_info=self.rigid_solver.joints_info,
            dofs_state=self.rigid_solver.dofs_state,
            dofs_info=self.rigid_solver.dofs_info,
            geoms_state=self.rigid_solver.geoms_state,
            geoms_info=self.rigid_solver.geoms_info,
            entities_info=self.rigid_solver.entities_info,
            rigid_global_info=self.rigid_solver._rigid_global_info,
            static_rigid_sim_config=self.rigid_solver._static_rigid_sim_config,
        )
        self.rigid_solver._is_forward_pos_updated = True
        self.rigid_solver._is_forward_vel_updated = True

    def _accumulate_restitution_base_link(
        self,
        dof_start: int,
        q_solved: np.ndarray,
        q_pred: np.ndarray,
        dt: float,
    ):
        """Per-frame restitution correction for a free-joint base link (6 DOFs)."""
        correction = np.zeros((self._B, 6), dtype=gs.np_float)

        # Translation correction
        correction[:, :3] = (q_solved[:, :3] - q_pred[:, :3]) / dt

        # Rotation correction
        for env_idx in range(self._B):
            dq = gu.transform_quat_by_quat(q_solved[env_idx, 3:7], gu.inv_quat(q_pred[env_idx, 3:7]))
            rotvec = gu.quat_to_rotvec(dq)
            correction[env_idx, 3:6] = rotvec / dt

        if np.max(np.abs(correction)) > RESTITUTION_CONTACT_THRESHOLD / dt:
            e = self.options.restitution
            self._restitution_vel_corrections.append((dof_start, dof_start + 6, e * correction))

    def apply_restitution_velocity(self):
        """Apply per-frame restitution velocity corrections after step_2.

        Called by rigid_solver.substep_post_coupling after kernel_step_2.
        Each frame: Δv = e * (q_solved - q_pred) / dt for base links in contact.
        """
        if not self._restitution_vel_corrections:
            return
        vel_tc = qd_to_torch(self.rigid_solver.dofs_state.vel, transpose=True, copy=False)
        for dof_start, dof_end, correction in self._restitution_vel_corrections:
            vel_tc[:, dof_start:dof_end] += torch.from_numpy(correction).to(vel_tc.device)
        self._restitution_vel_corrections = []
