from typing import TYPE_CHECKING

import numpy as np
import gstaichi as ti

import genesis as gs
from genesis.options.solvers import IPCCouplerOptions
from genesis.repr_base import RBC

if TYPE_CHECKING:
    from genesis.engine.simulator import Simulator

# Check if libuipc is available
try:
    import uipc

    UIPC_AVAILABLE = True
except ImportError:
    UIPC_AVAILABLE = False
    uipc = None


@ti.data_oriented
class IPCCoupler(RBC):
    """
    Coupler class for handling Incremental Potential Contact (IPC) simulation coupling.

    This coupler manages the communication between Genesis solvers and the IPC system,
    including rigid bodies (as ABD objects) and FEM bodies in a unified contact framework.
    """

    @ti.kernel
    def _compute_external_wrench_kernel(
        self,
        n_links: ti.i32,
        contact_forces: ti.types.ndarray(),  # (n_links, 3)
        contact_torques: ti.types.ndarray(),  # (n_links, 3)
        abd_transforms: ti.types.ndarray(),  # (n_links, 4, 4) - ABD transform matrices
        out_wrenches: ti.types.ndarray(),  # (n_links, 12) - output wrench vectors
    ):
        """
        Compute 12D external wrench from contact forces and torques.
        wrench = [force (3), M_affine (9)]
        where M_affine = skew(torque) * A, A is the rotation part of ABD transform
        """
        for i in range(n_links):
            # Copy force directly to first 3 components
            for j in ti.static(range(3)):
                out_wrenches[i, j] = -0.5 * contact_forces[i, j]

            # Extract torque
            tau = -0.5 * ti.Vector([contact_torques[i, 0], contact_torques[i, 1], contact_torques[i, 2]])

            # Extract rotation matrix A (3x3) from ABD transform (first 3x3 block)
            A = ti.Matrix.zero(ti.f32, 3, 3)
            for row in range(3):
                for col in range(3):
                    A[row, col] = abd_transforms[i, row, col]

            # Compute skew-symmetric matrix S of tau
            S = ti.Matrix.zero(ti.f32, 3, 3)
            S[0, 0] = 0.0
            S[0, 1] = -tau[2]
            S[0, 2] = tau[1]
            S[1, 0] = tau[2]
            S[1, 1] = 0.0
            S[1, 2] = -tau[0]
            S[2, 0] = -tau[1]
            S[2, 1] = tau[0]
            S[2, 2] = 0.0

            # Compute M_affine = S * A (row-major order)
            M_affine = S @ A

            # Write M_affine to last 9 components (row-major)
            idx = 3
            for row in range(3):
                for col in range(3):
                    out_wrenches[i, idx] = M_affine[row, col]
                    idx += 1

    @ti.kernel
    def _compute_coupling_forces_kernel(
        self,
        n_links: ti.i32,
        ipc_transforms: ti.types.ndarray(),  # (n_links, 4, 4)
        aim_transforms: ti.types.ndarray(),  # (n_links, 4, 4)
        link_masses: ti.types.ndarray(),  # (n_links,)
        inertia_tensors: ti.types.ndarray(),  # (n_links, 3, 3)
        translation_strength: ti.f32,
        rotation_strength: ti.f32,
        dt2: ti.f32,
        out_forces: ti.types.ndarray(),  # (n_links, 3)
        out_torques: ti.types.ndarray(),  # (n_links, 3)
    ):
        """
        Compute coupling forces and torques for all links in parallel.
        """
        for i in range(n_links):
            # Extract positions
            pos_current = ti.Vector([ipc_transforms[i, 0, 3], ipc_transforms[i, 1, 3], ipc_transforms[i, 2, 3]])
            pos_aim = ti.Vector([aim_transforms[i, 0, 3], aim_transforms[i, 1, 3], aim_transforms[i, 2, 3]])
            delta_pos = pos_current - pos_aim

            # Extract rotation matrices
            R_current = ti.Matrix.zero(ti.f32, 3, 3)
            R_aim = ti.Matrix.zero(ti.f32, 3, 3)
            for row in range(3):
                for col in range(3):
                    R_current[row, col] = ipc_transforms[i, row, col]
                    R_aim[row, col] = aim_transforms[i, row, col]

            # Compute linear force
            mass = link_masses[i]
            linear_force = translation_strength * mass * delta_pos / dt2

            # Compute relative rotation: R_rel = R_current @ R_aim^T
            R_rel = R_current @ R_aim.transpose()

            # Extract rotation vector from R_rel using Rodrigues formula
            # trace(R) = 1 + 2*cos(theta)
            trace = R_rel[0, 0] + R_rel[1, 1] + R_rel[2, 2]
            theta = ti.acos(ti.min(ti.max((trace - 1.0) / 2.0, -1.0), 1.0))

            # Rotation axis (when theta != 0)
            rotvec = ti.Vector.zero(ti.f32, 3)
            if theta > 1e-6:
                axis_x = R_rel[2, 1] - R_rel[1, 2]
                axis_y = R_rel[0, 2] - R_rel[2, 0]
                axis_z = R_rel[1, 0] - R_rel[0, 1]
                norm = ti.sqrt(axis_x * axis_x + axis_y * axis_y + axis_z * axis_z)
                if norm > 1e-8:
                    rotvec = theta * ti.Vector([axis_x, axis_y, axis_z]) / norm

            # Load inertia tensor
            I_local = ti.Matrix.zero(ti.f32, 3, 3)
            for row in range(3):
                for col in range(3):
                    I_local[row, col] = inertia_tensors[i, row, col]

            # Transform to world frame: I_world = R_current @ I_local @ R_current^T
            I_world = R_current @ I_local @ R_current.transpose()

            # Compute angular torque
            angular_torque = rotation_strength / dt2 * (I_world @ rotvec)

            # Store results
            for j in ti.static(range(3)):
                out_forces[i, j] = linear_force[j]
                out_torques[i, j] = angular_torque[j]

    @ti.kernel
    def _accumulate_contact_forces_kernel(
        self,
        n_contacts: ti.i32,
        contact_vert_indices: ti.types.ndarray(),  # (n_contacts,)
        contact_gradients: ti.types.ndarray(),  # (n_contacts, 3)
        vert_to_link: ti.types.ndarray(),  # (max_vert_idx,) mapping vert -> link_idx (-1 if invalid)
        vert_positions: ti.types.ndarray(),  # (max_vert_idx, 3)
        link_centers: ti.types.ndarray(),  # (n_links, 3)
        out_forces: ti.types.ndarray(),  # (n_links, 3)
        out_torques: ti.types.ndarray(),  # (n_links, 3)
    ):
        """
        Accumulate contact forces and torques for all vertices in parallel.
        """
        for i in range(n_contacts):
            vert_idx = contact_vert_indices[i]
            link_idx = vert_to_link[vert_idx]

            if link_idx >= 0:  # Valid link
                # Force is negative gradient
                force = ti.Vector([-contact_gradients[i, 0], -contact_gradients[i, 1], -contact_gradients[i, 2]])

                # Atomic add force
                for j in ti.static(range(3)):
                    ti.atomic_add(out_forces[link_idx, j], force[j])

                # Compute torque: τ = r × F
                contact_pos = ti.Vector(
                    [vert_positions[vert_idx, 0], vert_positions[vert_idx, 1], vert_positions[vert_idx, 2]]
                )
                center_pos = ti.Vector(
                    [link_centers[link_idx, 0], link_centers[link_idx, 1], link_centers[link_idx, 2]]
                )
                r = contact_pos - center_pos
                torque = r.cross(force)

                # Atomic add torque
                for j in ti.static(range(3)):
                    ti.atomic_add(out_torques[link_idx, j], torque[j])

    @ti.kernel
    def _compute_link_contact_forces_kernel(
        self,
        n_force_entries: ti.i32,
        force_gradients: ti.types.ndarray(),  # (n_force_entries, 3) force gradient for each vertex
        vert_to_link_idx: ti.types.ndarray(),  # (n_force_entries,) link_idx for each force entry
        vert_to_env_idx: ti.types.ndarray(),  # (n_force_entries,) env_idx for each force entry
        vert_positions: ti.types.ndarray(),  # (n_force_entries, 3) vertex positions in world space
        link_centers: ti.types.ndarray(),  # (n_force_entries, 3) link center for each entry
        out_forces: ti.types.ndarray(),  # (max_links, max_envs, 3) output forces
        out_torques: ti.types.ndarray(),  # (max_links, max_envs, 3) output torques
    ):
        """
        Compute contact forces and torques for rigid links from vertex gradients.
        Uses atomic operations to accumulate forces from multiple vertices per link.
        """
        for i in range(n_force_entries):
            link_idx = vert_to_link_idx[i]
            env_idx = vert_to_env_idx[i]

            # Force is negative gradient
            force = ti.Vector([-force_gradients[i, 0], -force_gradients[i, 1], -force_gradients[i, 2]])

            # Atomic add force
            for j in ti.static(range(3)):
                ti.atomic_add(out_forces[link_idx, env_idx, j], force[j])

            # Compute torque: τ = r × F
            contact_pos = ti.Vector([vert_positions[i, 0], vert_positions[i, 1], vert_positions[i, 2]])
            center_pos = ti.Vector([link_centers[i, 0], link_centers[i, 1], link_centers[i, 2]])
            r = contact_pos - center_pos
            torque = r.cross(force)

            # Atomic add torque
            for j in ti.static(range(3)):
                ti.atomic_add(out_torques[link_idx, env_idx, j], torque[j])

    @ti.kernel
    def _batch_read_qpos_kernel(
        self,
        qpos_field: ti.types.ndarray(),  # Source qpos field (n_dofs, n_envs)
        q_start: ti.i32,
        n_qs: ti.i32,
        env_idx: ti.i32,
        out_qpos: ti.types.ndarray(),  # Output array (n_qs,)
    ):
        """
        Batch read qpos for a specific entity and environment.
        """
        for i in range(n_qs):
            out_qpos[i] = qpos_field[q_start + i, env_idx]

    @ti.kernel
    def _compare_qpos_kernel(
        self,
        n_entries: ti.i32,
        qpos_current: ti.types.ndarray(),  # (n_entries,)
        qpos_stored: ti.types.ndarray(),  # (n_entries,)
        tolerance: ti.f32,
        out_modified: ti.types.ndarray(),  # (1,) - output 1 if modified, 0 otherwise
    ):
        """
        Compare two qpos arrays and detect if modified beyond tolerance.
        """
        modified = 0
        for i in range(n_entries):
            diff = ti.abs(qpos_current[i] - qpos_stored[i])
            if diff > tolerance:
                modified = 1
        out_modified[0] = modified

    @ti.kernel
    def _batch_pos_quat_to_transform_kernel(
        self,
        n_links: ti.i32,
        positions: ti.types.ndarray(),  # (n_links, 3)
        quaternions: ti.types.ndarray(),  # (n_links, 4) - wxyz format
        out_transforms: ti.types.ndarray(),  # (n_links, 4, 4)
    ):
        """
        Convert batch of positions and quaternions to 4x4 transform matrices.
        Quaternion format: [w, x, y, z]
        """
        for i in range(n_links):
            # Extract quaternion components
            w = quaternions[i, 0]
            x = quaternions[i, 1]
            y = quaternions[i, 2]
            z = quaternions[i, 3]

            # Compute rotation matrix from quaternion
            # R = I + 2*s*K + 2*K^2, where s = w, K = skew(x,y,z)
            xx = x * x
            yy = y * y
            zz = z * z
            xy = x * y
            xz = x * z
            yz = y * z
            wx = w * x
            wy = w * y
            wz = w * z

            # Rotation matrix (3x3)
            R00 = 1.0 - 2.0 * (yy + zz)
            R01 = 2.0 * (xy - wz)
            R02 = 2.0 * (xz + wy)

            R10 = 2.0 * (xy + wz)
            R11 = 1.0 - 2.0 * (xx + zz)
            R12 = 2.0 * (yz - wx)

            R20 = 2.0 * (xz - wy)
            R21 = 2.0 * (yz + wx)
            R22 = 1.0 - 2.0 * (xx + yy)

            # Build 4x4 transform matrix
            # [R  t]
            # [0  1]
            out_transforms[i, 0, 0] = R00
            out_transforms[i, 0, 1] = R01
            out_transforms[i, 0, 2] = R02
            out_transforms[i, 0, 3] = positions[i, 0]

            out_transforms[i, 1, 0] = R10
            out_transforms[i, 1, 1] = R11
            out_transforms[i, 1, 2] = R12
            out_transforms[i, 1, 3] = positions[i, 1]

            out_transforms[i, 2, 0] = R20
            out_transforms[i, 2, 1] = R21
            out_transforms[i, 2, 2] = R22
            out_transforms[i, 2, 3] = positions[i, 2]

            out_transforms[i, 3, 0] = 0.0
            out_transforms[i, 3, 1] = 0.0
            out_transforms[i, 3, 2] = 0.0
            out_transforms[i, 3, 3] = 1.0

    def __init__(self, simulator: "Simulator", options: "IPCCouplerOptions") -> None:
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
                "libuipc is required for IPC coupling but not found.\n"
                "Please build and install libuipc from source:\n"
                "https://github.com/spiriMirror/libuipc"
            )

        self.sim = simulator
        self.options = options

        # Validate coupling strategy
        valid_strategies = ["two_way_soft_constraint", "contact_proxy"]
        if self.options.coupling_strategy not in valid_strategies:
            raise ValueError(
                f"Invalid coupling_strategy '{self.options.coupling_strategy}'. " f"Must be one of {valid_strategies}"
            )

        # Store solver references
        self.rigid_solver = self.sim.rigid_solver
        self.fem_solver = self.sim.fem_solver

        # IPC system components (will be initialized in build)
        self._ipc_engine = None
        self._ipc_world = None
        self._ipc_scene = None
        self._ipc_abd = None
        self._ipc_stk = None
        self._ipc_abd_contact = None
        self._ipc_fem_contact = None
        self._ipc_scene_subscenes = {}
        self._use_subscenes = False  # Will be set in _init_ipc based on number of environments

        # IPC link filter: maps entity_idx -> set of link_idx to include in IPC
        # If entity_idx not in dict or value is None, all links of that entity participate
        self._ipc_link_filters = {}

        # IPC-only links: maps entity_idx -> set of link_idx that should ONLY exist in IPC
        # These links will not have soft constraints, use full density, and directly set Genesis transforms
        self._ipc_only_links = {}

        # Storage for Genesis rigid body states before IPC advance
        # Maps link_idx -> {env_idx: transform_matrix}
        self._genesis_stored_states = {}

        # Storage for IPC contact forces on rigid links (both coupling mode)
        # Maps link_idx -> {env_idx: {'force': np.array, 'torque': np.array}}
        self._ipc_contact_forces = {}

        # Storage for entity qpos before IPC advance (to detect user-modified qpos)
        # Maps (entity_idx, env_idx) -> qpos_tensor
        self._entity_qpos_before_ipc = {}

        # Storage for external wrench data for rigid links
        # Maps (link_idx, env_idx) -> wrench_vector (12D numpy array)
        self._external_wrench_data = {}

        # Pre-computed mapping from vertex index to rigid link (built once during IPC setup)
        # Maps global_vertex_idx -> (link_idx, env_idx, local_vertex_idx)
        self._vertex_to_link_mapping = {}
        # Global vertex offset for tracking vertex indices across all geometries
        self._global_vertex_offset = 0

        # Pre-allocated Taichi fields for contact force processing
        # Estimate max contacts based on potential rigid links
        max_contacts = 1000  # Conservative estimate, can be tuned based on scene complexity
        max_links = 200  # Max number of links
        max_envs = 100  # Max number of environments

        self.contact_forces_ti = ti.Vector.field(3, dtype=gs.ti_float, shape=max_contacts)
        self.contact_torques_ti = ti.Vector.field(3, dtype=gs.ti_float, shape=max_contacts)
        self.abd_transforms_ti = ti.Matrix.field(4, 4, dtype=gs.ti_float, shape=max_contacts)
        self.out_wrenches_ti = ti.Vector.field(12, dtype=gs.ti_float, shape=max_contacts)
        self.link_indices_ti = ti.field(dtype=ti.i32, shape=max_contacts)
        self.env_indices_ti = ti.field(dtype=ti.i32, shape=max_contacts)

        # Pre-allocated fields for link contact force computation
        # Output arrays dimensioned by max_links x max_envs
        self.link_contact_forces_out = ti.Vector.field(3, dtype=gs.ti_float, shape=(max_links, max_envs))
        self.link_contact_torques_out = ti.Vector.field(3, dtype=gs.ti_float, shape=(max_links, max_envs))

        # Fields for storing vertex-level contact data (for kernel processing)
        self.max_vertex_contacts = 5000  # Max vertex contacts
        self.vertex_force_gradients = ti.Vector.field(3, dtype=gs.ti_float, shape=self.max_vertex_contacts)
        self.vertex_link_indices = ti.field(dtype=ti.i32, shape=self.max_vertex_contacts)
        self.vertex_env_indices = ti.field(dtype=ti.i32, shape=self.max_vertex_contacts)
        self.vertex_positions_world = ti.Vector.field(3, dtype=gs.ti_float, shape=self.max_vertex_contacts)
        self.vertex_link_centers = ti.Vector.field(3, dtype=gs.ti_float, shape=self.max_vertex_contacts)

        # Fields for batch qpos operations
        self.max_qpos_size = 500  # Max qpos size per entity
        self.qpos_buffer = ti.field(dtype=gs.ti_float, shape=self.max_qpos_size)
        self.qpos_comparison_result = ti.field(dtype=ti.i32, shape=1)

        # Fields for batch transform operations
        self.batch_positions = ti.Vector.field(3, dtype=gs.ti_float, shape=max_links)
        self.batch_quaternions = ti.Vector.field(4, dtype=gs.ti_float, shape=max_links)
        self.batch_transforms = ti.Matrix.field(4, 4, dtype=gs.ti_float, shape=max_links)

    def build(self) -> None:
        """Build IPC system"""
        # Initialize IPC system
        self._init_ipc()
        self._add_objects_to_ipc()
        self._finalize_ipc()
        if self.options.enable_ipc_gui:
            self._init_ipc_gui()

    def _init_ipc(self):
        """Initialize IPC system components"""
        from uipc.core import Engine, World, Scene
        from uipc.constitution import AffineBodyConstitution, StableNeoHookean, NeoHookeanShell, DiscreteShellBending

        # Disable IPC logging if requested
        if self.options.disable_ipc_logging:
            from uipc import Logger, Timer

            Logger.set_level(Logger.Level.Error)
            Timer.disable_all()

        # Create IPC engine and world
        import os
        import tempfile

        # Create workspace directory for IPC output
        workspace = os.path.join(tempfile.gettempdir(), "genesis_ipc_workspace")
        os.makedirs(workspace, exist_ok=True)
        self._ipc_engine = Engine("cuda", workspace)
        self._ipc_world = World(self._ipc_engine)

        # Create IPC scene with configuration
        config = Scene.default_config()
        config["dt"] = self.options.dt
        config["gravity"] = [[self.options.gravity[0]], [self.options.gravity[1]], [self.options.gravity[2]]]
        config["contact"]["d_hat"] = self.options.contact_d_hat
        config["contact"]["friction"]["enable"] = self.options.contact_friction_enable
        config["newton"]["velocity_tol"] = self.options.newton_velocity_tol
        config["line_search"]["max_iter"] = self.options.line_search_max_iter
        config["linear_system"]["tol_rate"] = self.options.linear_system_tol_rate
        config["sanity_check"]["enable"] = self.options.sanity_check_enable

        self._ipc_scene = Scene(config)

        # Create constitutions
        self._ipc_abd = AffineBodyConstitution()
        self._ipc_stk = StableNeoHookean()
        self._ipc_nks = NeoHookeanShell()  # For cloth
        self._ipc_dsb = DiscreteShellBending()  # For cloth bending

        # Add constitutions to scene
        self._ipc_scene.constitution_tabular().insert(self._ipc_abd)
        self._ipc_scene.constitution_tabular().insert(self._ipc_stk)
        # Note: Shell constitutions are added on-demand when cloth entities exist

        # Set up contact model (physical parameters)
        self._ipc_scene.contact_tabular().default_model(
            self.options.contact_friction_mu, self.options.contact_resistance
        )

        # Create separate contact elements for ABD, FEM, Cloth, and Ground to control their interactions
        self._ipc_abd_contact = self._ipc_scene.contact_tabular().create("abd_contact")
        self._ipc_fem_contact = self._ipc_scene.contact_tabular().create("fem_contact")
        self._ipc_cloth_contact = self._ipc_scene.contact_tabular().create("cloth_contact")
        self._ipc_ground_contact = self._ipc_scene.contact_tabular().create("ground_contact")

        # Configure contact interactions based on IPC coupler options
        # FEM-FEM: always enabled
        self._ipc_scene.contact_tabular().insert(
            self._ipc_fem_contact,
            self._ipc_fem_contact,
            self.options.contact_friction_mu,
            self.options.contact_resistance,
            True,
        )
        # FEM-ABD: always enabled
        self._ipc_scene.contact_tabular().insert(
            self._ipc_fem_contact,
            self._ipc_abd_contact,
            self.options.contact_friction_mu,
            self.options.contact_resistance,
            True,
        )
        # ABD-ABD: controlled by IPC_self_contact option
        self._ipc_scene.contact_tabular().insert(
            self._ipc_abd_contact,
            self._ipc_abd_contact,
            self.options.contact_friction_mu,
            self.options.contact_resistance,
            self.options.IPC_self_contact,
        )
        # Cloth-Cloth: always enabled for cloth self-collision (necessary to prevent self-penetration)
        self._ipc_scene.contact_tabular().insert(
            self._ipc_cloth_contact,
            self._ipc_cloth_contact,
            self.options.contact_friction_mu,
            self.options.contact_resistance,
            True,
        )  # Always enable cloth self-collision
        # Cloth-FEM: always enabled
        self._ipc_scene.contact_tabular().insert(
            self._ipc_cloth_contact,
            self._ipc_fem_contact,
            self.options.contact_friction_mu,
            self.options.contact_resistance,
            True,
        )
        # Cloth-ABD: always enabled
        self._ipc_scene.contact_tabular().insert(
            self._ipc_cloth_contact,
            self._ipc_abd_contact,
            self.options.contact_friction_mu,
            self.options.contact_resistance,
            True,
        )

        # Ground contact interactions
        # Ground-ABD (rigid bodies): controlled by disable_ipc_ground_contact option
        self._ipc_scene.contact_tabular().insert(
            self._ipc_ground_contact,
            self._ipc_abd_contact,
            self.options.contact_friction_mu,
            self.options.contact_resistance,
            not self.options.disable_ipc_ground_contact,
        )
        # Ground-FEM: always enabled
        self._ipc_scene.contact_tabular().insert(
            self._ipc_ground_contact,
            self._ipc_fem_contact,
            self.options.contact_friction_mu,
            self.options.contact_resistance,
            True,
        )
        # Ground-Cloth: always enabled
        self._ipc_scene.contact_tabular().insert(
            self._ipc_ground_contact,
            self._ipc_cloth_contact,
            self.options.contact_friction_mu,
            self.options.contact_resistance,
            True,
        )

        # Set up subscenes for multi-environment (scene grouping)
        # Only use subscenes when B > 1 to avoid issues with ground collision
        # (ground's subscene support is incomplete in libuipc)
        B = self.sim._B
        self._ipc_scene_subscenes = {}
        self._use_subscenes = B > 1

        if self._use_subscenes:
            for i in range(B):
                self._ipc_scene_subscenes[i] = self._ipc_scene.subscene_tabular().create(f"subscene{i}")

            # Disable contact between different environments
            for i in range(B):
                for j in range(B):
                    if i != j:
                        self._ipc_scene.subscene_tabular().insert(
                            self._ipc_scene_subscenes[i], self._ipc_scene_subscenes[j], False
                        )

        self.abd_data_by_link = {}

    def _add_objects_to_ipc(self):
        """Add objects from solvers to IPC system"""
        # Add FEM entities to IPC
        if self.fem_solver.is_active:
            self._add_fem_entities_to_ipc()

        # Add rigid geoms to IPC
        if self.rigid_solver.is_active:
            self._add_rigid_geoms_to_ipc()

    def _add_fem_entities_to_ipc(self):
        """Add FEM entities to the existing IPC scene (includes both volumetric FEM and cloth)"""
        from uipc.constitution import ElasticModuli
        from uipc.geometry import label_surface, tetmesh, trimesh
        from genesis.engine.materials.FEM.cloth import Cloth as ClothMaterial

        fem_solver = self.fem_solver
        scene = self._ipc_scene
        stk = self._ipc_stk  # StableNeoHookean for volumetric FEM
        nks = self._ipc_nks  # NeoHookeanShell for cloth
        dsb = self._ipc_dsb  # DiscreteShellBending for cloth
        scene_subscenes = self._ipc_scene_subscenes

        fem_solver._mesh_handles = {}
        fem_solver.list_env_obj = []
        fem_solver.list_env_mesh = []

        for i_b in range(self.sim._B):
            fem_solver.list_env_obj.append([])
            fem_solver.list_env_mesh.append([])
            for i_e, entity in enumerate(fem_solver._entities):
                is_cloth = isinstance(entity.material, ClothMaterial)

                # Create object in IPC
                obj_name = f"cloth_{i_b}_{i_e}" if is_cloth else f"fem_{i_b}_{i_e}"
                fem_solver.list_env_obj[i_b].append(scene.objects().create(obj_name))

                # Create mesh: trimesh for cloth (2D shell), tetmesh for volumetric FEM (3D)
                if is_cloth:
                    # Cloth: use surface triangles only
                    verts = entity.init_positions.cpu().numpy().astype(np.float64)
                    faces = entity.surface_triangles.astype(np.int32)
                    mesh = trimesh(verts, faces)
                else:
                    # Volumetric FEM: use tetrahedral mesh
                    mesh = tetmesh(entity.init_positions.cpu().numpy(), entity.elems)

                fem_solver.list_env_mesh[i_b].append(mesh)

                # Add to contact subscene (only for multi-environment)
                if self._use_subscenes:
                    scene_subscenes[i_b].apply_to(mesh)

                # Apply contact element based on type
                if is_cloth:
                    self._ipc_cloth_contact.apply_to(mesh)
                else:
                    self._ipc_fem_contact.apply_to(mesh)

                label_surface(mesh)

                # Apply material constitution based on type
                moduli = ElasticModuli.youngs_poisson(entity.material.E, entity.material.nu)
                if is_cloth:
                    # Apply shell material for cloth
                    nks.apply_to(
                        mesh, moduli=moduli, mass_density=entity.material.rho, thickness=entity.material.thickness
                    )
                    # Apply bending stiffness if specified
                    if entity.material.bending_stiffness is not None:
                        dsb.apply_to(mesh, bending_stiffness=entity.material.bending_stiffness)
                else:
                    # Apply volumetric material for FEM
                    stk.apply_to(mesh, moduli, mass_density=entity.material.rho)

                # Add metadata to identify geometry type
                meta_attrs = mesh.meta()
                meta_attrs.create("solver_type", "cloth" if is_cloth else "fem")
                meta_attrs.create("env_idx", str(i_b))
                meta_attrs.create("entity_idx", str(i_e))

                # Create geometry in IPC scene
                fem_solver.list_env_obj[i_b][i_e].geometries().create(mesh)
                fem_solver._mesh_handles[f"gs_ipc_{i_b}_{i_e}"] = mesh

                # Update global vertex offset (FEM vertices occupy index space but aren't in mapping)
                self._global_vertex_offset += mesh.vertices().size()

    def _add_rigid_geoms_to_ipc(self):
        """Add rigid geoms to the existing IPC scene as ABD objects, merging geoms by link_idx"""
        from uipc.geometry import tetmesh, label_surface, label_triangle_orient, flip_inward_triangles, merge, ground
        from uipc.constitution import AffineBodyExternalWrench
        from genesis.utils import mesh as mu
        import numpy as np
        import trimesh

        rigid_solver = self.rigid_solver
        scene = self._ipc_scene
        abd = self._ipc_abd
        scene_subscenes = self._ipc_scene_subscenes

        # Create and register AffineBodyExternalWrench constitution
        if not hasattr(self, "_ipc_ext_wrench"):
            self._ipc_ext_wrench = AffineBodyExternalWrench()
            scene.constitution_tabular().insert(self._ipc_ext_wrench)

        # Initialize lists following FEM solver pattern
        rigid_solver.list_env_obj = []
        rigid_solver.list_env_mesh = []
        rigid_solver._mesh_handles = {}
        rigid_solver._abd_transforms = {}

        for i_b in range(self.sim._B):
            rigid_solver.list_env_obj.append([])
            rigid_solver.list_env_mesh.append([])

            # Group geoms by link_idx for merging
            link_geoms = {}  # link_idx -> dict with 'meshes', 'link_world_pos', 'link_world_quat', 'entity_idx'
            link_planes = {}  # link_idx -> list of plane geoms (handle separately)

            # First pass: collect and group geoms by link_idx
            for i_g in range(rigid_solver.n_geoms_):
                geom_type = rigid_solver.geoms_info.type[i_g]
                link_idx = rigid_solver.geoms_info.link_idx[i_g]
                entity_idx = rigid_solver.links_info.entity_idx[link_idx]
                entity = rigid_solver._entities[entity_idx]

                # Check if this link should be included in IPC based on coupler's filter
                if entity_idx in self._ipc_link_filters:
                    link_filter = self._ipc_link_filters[entity_idx]
                    if link_filter is not None and link_idx not in link_filter:
                        continue  # Skip this geom/link

                # Initialize link group if not exists
                if link_idx not in link_geoms:
                    link_geoms[link_idx] = {
                        "meshes": [],
                        "link_world_pos": None,
                        "link_world_quat": None,
                        "entity_idx": entity_idx,
                    }
                    link_planes[link_idx] = []

                try:
                    if geom_type == gs.GEOM_TYPE.PLANE:
                        # Handle planes separately (they can't be merged with SimplicialComplex)
                        # Ground/plane will be assigned to ground_contact element for selective collision control
                        pos = rigid_solver.geoms_info.pos[i_g].to_numpy()
                        normal = np.array([0.0, 0.0, 1.0])  # Z-up
                        height = np.dot(pos, normal)
                        plane_geom = ground(height, normal)
                        link_planes[link_idx].append((i_g, plane_geom))

                    else:
                        # For all non-plane geoms, create tetmesh
                        vert_num = rigid_solver.geoms_info.vert_num[i_g]
                        if vert_num == 0:
                            continue  # Skip geoms without vertices

                        # Extract vertex and face data
                        vert_start = rigid_solver.geoms_info.vert_start[i_g]
                        vert_end = rigid_solver.geoms_info.vert_end[i_g]
                        face_start = rigid_solver.geoms_info.face_start[i_g]
                        face_end = rigid_solver.geoms_info.face_end[i_g]

                        # Get vertices and faces
                        geom_verts = rigid_solver.verts_info.init_pos.to_numpy()[vert_start:vert_end]
                        geom_faces = rigid_solver.faces_info.verts_idx.to_numpy()[face_start:face_end]
                        geom_faces = geom_faces - vert_start  # Adjust indices

                        # Apply geom-relative transform to vertices (needed for merging)
                        geom_rel_pos = rigid_solver.geoms_info.pos[i_g].to_numpy()
                        geom_rel_quat = rigid_solver.geoms_info.quat[i_g].to_numpy()

                        # Transform vertices by geom relative transform
                        import genesis.utils.geom as gu

                        geom_rot_mat = gu.quat_to_R(geom_rel_quat)
                        transformed_verts = geom_verts @ geom_rot_mat.T + geom_rel_pos

                        # Create uipc trimesh for rigid body (ABD doesn't need tetmesh)
                        try:
                            from uipc.geometry import trimesh as uipc_trimesh

                            # Create uipc trimesh directly (dim=2, surface mesh for ABD)
                            rigid_mesh = uipc_trimesh(transformed_verts.astype(np.float64), geom_faces.astype(np.int32))

                            # Store uipc mesh (SimplicialComplex) for merging
                            link_geoms[link_idx]["meshes"].append((i_g, rigid_mesh))

                        except Exception as e:
                            gs.logger.warning(f"Failed to convert trimesh to tetmesh for geom {i_g}: {e}")
                            continue

                    # Store link transform info (same for all geoms in link)
                    if link_geoms[link_idx]["link_world_pos"] is None:
                        link_geoms[link_idx]["link_world_pos"] = rigid_solver.links_state.pos[link_idx, i_b]
                        link_geoms[link_idx]["link_world_quat"] = rigid_solver.links_state.quat[link_idx, i_b]

                except Exception as e:
                    gs.logger.warning(f"Failed to process geom {i_g}: {e}")
                    continue

            # Second pass: merge geoms per link and create IPC objects
            link_obj_counter = 0
            for link_idx, link_data in link_geoms.items():
                try:
                    # Handle regular meshes (merge if multiple)
                    if link_data["meshes"]:
                        if len(link_data["meshes"]) == 1:
                            # Single mesh in link
                            geom_idx, merged_mesh = link_data["meshes"][0]
                        else:
                            # Multiple meshes in link - merge them using uipc.geometry.merge
                            meshes_to_merge = [mesh for geom_idx, mesh in link_data["meshes"]]
                            merged_mesh = merge(meshes_to_merge)
                            geom_idx = link_data["meshes"][0][0]  # Use first geom's index for metadata

                        # Apply link world transform
                        from uipc import view, Transform, Vector3, Quaternion

                        trans_view = view(merged_mesh.transforms())
                        t = Transform.Identity()

                        link_world_pos = link_data["link_world_pos"]
                        link_world_quat = link_data["link_world_quat"]

                        # Ensure numpy format
                        link_world_pos = link_world_pos.to_numpy()
                        link_world_quat = link_world_quat.to_numpy()

                        t.translate(Vector3.Values((link_world_pos[0], link_world_pos[1], link_world_pos[2])))
                        uipc_link_quat = Quaternion(link_world_quat)
                        t.rotate(uipc_link_quat)
                        trans_view[0] = t.matrix()

                        # Process surface for contact
                        label_surface(merged_mesh)

                        # Create rigid object
                        rigid_obj = scene.objects().create(f"rigid_link_{i_b}_{link_idx}")
                        rigid_solver.list_env_obj[i_b].append(rigid_obj)
                        rigid_solver.list_env_mesh[i_b].append(merged_mesh)

                        # Add to contact subscene and apply ABD constitution (only for multi-environment)
                        if self._use_subscenes:
                            scene_subscenes[i_b].apply_to(merged_mesh)
                        self._ipc_abd_contact.apply_to(merged_mesh)
                        from uipc.unit import MPa

                        # Check if this link is IPC-only
                        is_ipc_only = (
                            link_data["entity_idx"] in self._ipc_only_links
                            and link_idx in self._ipc_only_links[link_data["entity_idx"]]
                        )

                        entity_rho = rigid_solver._entities[link_data["entity_idx"]].material.rho

                        if is_ipc_only:
                            # IPC-only links use full density (no mass splitting with Genesis)
                            abd.apply_to(
                                merged_mesh,
                                kappa=10.0 * MPa,
                                mass_density=entity_rho,
                            )
                        else:
                            # For two_way_soft_constraint: use half density to avoid double-counting mass
                            # For contact_proxy: use full density (no mass sharing with Genesis)
                            if self.options.coupling_strategy == "contact_proxy":
                                ipc_mass_density = entity_rho
                            else:
                                ipc_mass_density = entity_rho / 2.0

                            abd.apply_to(
                                merged_mesh,
                                kappa=10.0 * MPa,
                                mass_density=ipc_mass_density,
                            )

                            # Apply soft transform constraints only for coupled links (not IPC-only)
                            from uipc.constitution import SoftTransformConstraint

                            if not hasattr(self, "_ipc_stc"):
                                self._ipc_stc = SoftTransformConstraint()
                                scene.constitution_tabular().insert(self._ipc_stc)

                            strength_tuple = self.options.ipc_constraint_strength
                            constraint_strength = np.array(
                                [
                                    strength_tuple[0],  # translation strength
                                    strength_tuple[1],  # rotation strength
                                ]
                            )
                            self._ipc_stc.apply_to(merged_mesh, constraint_strength)

                        # Apply external wrench (initially zero, can be modified by animator)
                        initial_wrench = np.zeros(12, dtype=np.float64)  # Vector12: [fx, fy, fz, dS/dt (9 components)]
                        self._ipc_ext_wrench.apply_to(merged_mesh, initial_wrench)

                        # Add metadata
                        meta_attrs = merged_mesh.meta()
                        meta_attrs.create("solver_type", "rigid")
                        meta_attrs.create("env_idx", str(i_b))
                        meta_attrs.create("link_idx", str(link_idx))  # Use link_idx instead of geom_idx

                        # Build vertex-to-link mapping for contact force computation
                        # Check if this is a 'both' coupling link (needs contact force feedback)
                        is_both_coupling = link_data[
                            "entity_idx"
                        ] not in self._ipc_only_links or link_idx not in self._ipc_only_links.get(
                            link_data["entity_idx"], set()
                        )

                        if is_both_coupling:
                            n_verts = merged_mesh.vertices().size()
                            for local_idx in range(n_verts):
                                global_idx = self._global_vertex_offset + local_idx
                                self._vertex_to_link_mapping[global_idx] = (link_idx, i_b, local_idx)

                        # Update global vertex offset
                        self._global_vertex_offset += merged_mesh.vertices().size()

                        rigid_obj.geometries().create(merged_mesh)

                        # Set up animator for this link
                        if not hasattr(self, "_ipc_animator"):
                            self._ipc_animator = scene.animator()

                        def create_animate_function(env_idx, link_idx, coupler_ref):
                            def animate_rigid_link(info):
                                from uipc import view, builtin
                                import numpy as np

                                geo_slots = info.geo_slots()
                                if len(geo_slots) == 0:
                                    return
                                geo = geo_slots[0].geometry()

                                try:
                                    # Read stored Genesis transform (q_genesis^n)
                                    # This was stored in _store_genesis_rigid_states() before advance()
                                    if hasattr(coupler_ref, "_genesis_stored_states"):
                                        stored_states = coupler_ref._genesis_stored_states
                                        if link_idx in stored_states and env_idx in stored_states[link_idx]:
                                            transform_matrix = stored_states[link_idx][env_idx]

                                            # Enable constraint and set target transform
                                            is_constrained = geo.instances().find(builtin.is_constrained)
                                            aim_transform_attr = geo.instances().find(builtin.aim_transform)

                                            if is_constrained and aim_transform_attr:
                                                view(is_constrained)[0] = 1
                                                view(aim_transform_attr)[:] = transform_matrix

                                    # Update external wrench if user has set it
                                    if hasattr(coupler_ref, "_external_wrench_data"):
                                        wrench_data = coupler_ref._external_wrench_data
                                        key = (link_idx, env_idx)
                                        if key in wrench_data:
                                            wrench_attr = geo.instances().find("external_wrench")
                                            if wrench_attr is not None:
                                                wrench_vector = wrench_data[key]
                                                view(wrench_attr)[:] = wrench_vector.reshape(-1, 1)

                                except Exception as e:
                                    gs.logger.warning(f"Error setting IPC animation target: {e}")

                            return animate_rigid_link

                        animate_func = create_animate_function(i_b, link_idx, self)
                        self._ipc_animator.insert(rigid_obj, animate_func)

                        rigid_solver._mesh_handles[f"rigid_link_{i_b}_{link_idx}"] = merged_mesh
                        link_obj_counter += 1

                    # Handle planes for this link separately
                    for geom_idx, plane_geom in link_planes[link_idx]:
                        plane_obj = scene.objects().create(f"rigid_plane_{i_b}_{geom_idx}")
                        rigid_solver.list_env_obj[i_b].append(plane_obj)
                        rigid_solver.list_env_mesh[i_b].append(None)  # Planes are ImplicitGeometry

                        # Apply ground contact element to plane
                        self._ipc_ground_contact.apply_to(plane_geom)

                        plane_obj.geometries().create(plane_geom)
                        rigid_solver._mesh_handles[f"rigid_plane_{i_b}_{geom_idx}"] = plane_geom
                        link_obj_counter += 1

                        # Planes don't have vertices, no offset update needed

                except Exception as e:
                    gs.logger.warning(f"Failed to create IPC object for link {link_idx}: {e}")
                    continue

        # Scale down Genesis rigid solver masses for links added to IPC
        # Only needed for two_way_soft_constraint (mass sharing between Genesis and IPC)
        # For contact_proxy, no mass scaling needed (forces are transferred via contact)
        if self.options.coupling_strategy != "contact_proxy":
            self._scale_genesis_rigid_link_masses(link_geoms)

    def _scale_genesis_rigid_link_masses(self, link_geoms_dict):
        """
        Scale down Genesis rigid solver mass properties for links that were added to IPC.
        Both Genesis and IPC will simulate these rigid bodies, so we divide by 2 to avoid
        double-counting mass.

        Note: This should only be called for two_way_soft_constraint strategy,
        not for contact_proxy strategy.

        This scales:
        - inertial_mass: scalar mass
        - inertial_i: 3x3 inertia tensor (scales linearly with mass)

        Parameters
        ----------
        link_geoms_dict : dict
            Dictionary mapping link_idx to their geometry data (from _add_rigid_geoms_to_ipc)
        """
        # Safety check: should not be called in contact_proxy mode
        if self.options.coupling_strategy == "contact_proxy":
            gs.logger.warning(
                "_scale_genesis_rigid_link_masses called in contact_proxy mode. "
                "Mass scaling should only be used in two_way_soft_constraint mode. Skipping."
            )
            return

        rigid_solver = self.rigid_solver

        # Get all link indices that were added to IPC
        ipc_link_indices = set(link_geoms_dict.keys())

        if not ipc_link_indices:
            return

        gs.logger.info(f"Scaling Genesis rigid mass for {len(ipc_link_indices)} links added to IPC (dividing by 2)")

        # Scale mass properties for each link
        for link_idx in ipc_link_indices:
            # Scale inertial mass
            original_mass = float(rigid_solver.links_info.inertial_mass[link_idx])
            rigid_solver.links_info.inertial_mass[link_idx] = original_mass / 2.0

            # Scale inertia tensor (inertia scales linearly with mass for same geometry)
            original_inertia = rigid_solver.links_info.inertial_i[link_idx]
            rigid_solver.links_info.inertial_i[link_idx] = original_inertia / 2.0

            gs.logger.debug(
                f"  Link {link_idx}: mass {original_mass:.6f} -> {original_mass/2.0:.6f} kg, " f"inertia scaled by 0.5"
            )

        # After scaling inertial_mass and inertial_i, we need to recompute derived quantities:
        # - mass_mat: mass matrix (computed from inertial_mass and inertial_i)
        # - invweight: inverse weight (computed from mass_mat)
        # - meaninertia: mean inertia (computed from mass_mat)
        gs.logger.info("Recomputing mass matrix and derived quantities after scaling")
        rigid_solver._init_invweight_and_meaninertia(force_update=True)

    def _finalize_ipc(self):
        """Finalize IPC setup"""
        self._ipc_world.init(self._ipc_scene)
        gs.logger.info("IPC world initialized successfully")

    @property
    def is_active(self) -> bool:
        """Check if IPC coupling is active"""
        return self._ipc_world is not None

    def set_link_ipc_coupling_type(self, entity, coupling_type: str, link_names=None, link_indices=None):
        """
        Set IPC coupling type for links of an entity.

        Parameters
        ----------
        entity : RigidEntity
            The rigid entity to configure
        coupling_type : str
            Type of coupling: 'both', 'ipc_only', or 'genesis_only'
            - 'both': Two-way coupling between IPC and Genesis (default behavior)
            - 'ipc_only': Links only simulated in IPC, transforms copied to Genesis (one-way)
            - 'genesis_only': Links only simulated in Genesis, excluded from IPC
        link_names : list of str, optional
            Names of links to configure. If None and link_indices is None, applies to all links.
        link_indices : list of int, optional
            Local indices of links to configure. If None and link_names is None, applies to all links.

        Notes
        -----
        - 'both': Links use half density in IPC, have SoftTransformConstraint, bidirectional forces
        - 'ipc_only': Links use full density in IPC, no SoftTransformConstraint, transforms copied to Genesis
        - 'genesis_only': Links excluded from IPC simulation entirely
        """
        entity_idx = entity._idx

        # Determine which links to configure
        if link_names is None and link_indices is None:
            # Apply to all links
            target_links = set()
            for local_idx in range(entity.n_links):
                solver_link_idx = local_idx + entity._link_start
                target_links.add(solver_link_idx)
        else:
            # Apply to specified links
            target_links = set()

            if link_names is not None:
                for name in link_names:
                    try:
                        link = entity.get_link(name=name)
                        target_links.add(link.idx)

                    except Exception as e:
                        gs.logger.warning(f"Link name '{name}' not found in entity")

            if link_indices is not None:
                for local_idx in link_indices:
                    solver_link_idx = local_idx + entity._link_start
                    target_links.add(solver_link_idx)

        # Apply coupling type
        if coupling_type == "both":
            # Two-way coupling: include in IPC, not in IPC-only
            self._ipc_link_filters[entity_idx] = target_links

            # Remove from IPC-only if present
            if entity_idx in self._ipc_only_links:
                self._ipc_only_links[entity_idx] -= target_links
                if not self._ipc_only_links[entity_idx]:
                    del self._ipc_only_links[entity_idx]

            gs.logger.info(f"Entity {entity_idx}: {len(target_links)} link(s) set to 'both' coupling")

        elif coupling_type == "ipc_only":
            # One-way coupling: IPC -> Genesis
            if entity_idx not in self._ipc_only_links:
                self._ipc_only_links[entity_idx] = set()
            self._ipc_only_links[entity_idx].update(target_links)

            # Also add to IPC link filter
            if entity_idx not in self._ipc_link_filters:
                self._ipc_link_filters[entity_idx] = set()
            self._ipc_link_filters[entity_idx].update(target_links)

            gs.logger.info(f"Entity {entity_idx}: {len(target_links)} link(s) set to 'ipc_only' coupling")

        elif coupling_type == "genesis_only":
            # Genesis-only: remove from both filters
            if entity_idx in self._ipc_link_filters:
                self._ipc_link_filters[entity_idx] -= target_links
                if not self._ipc_link_filters[entity_idx]:
                    del self._ipc_link_filters[entity_idx]

            if entity_idx in self._ipc_only_links:
                self._ipc_only_links[entity_idx] -= target_links
                if not self._ipc_only_links[entity_idx]:
                    del self._ipc_only_links[entity_idx]

            gs.logger.info(
                f"Entity {entity_idx}: {len(target_links)} link(s) set to 'genesis_only' (excluded from IPC)"
            )

        else:
            raise ValueError(
                f"Invalid coupling_type '{coupling_type}'. " f"Must be 'both', 'ipc_only', or 'genesis_only'."
            )

    def preprocess(self, f):
        """Preprocessing step before coupling"""
        pass

    def _store_genesis_rigid_states(self):
        """
        Store current Genesis rigid body states before IPC advance.

        OPTIMIZED VERSION: Uses Taichi kernels for batch reading qpos.

        These stored states will be used by:
        1. Animator: to set aim_transform for IPC soft constraints
        2. Force computation: to ensure action-reaction force consistency
        3. User modification detection: to detect if user called set_qpos
        """
        if not self.rigid_solver.is_active:
            return

        rigid_solver = self.rigid_solver

        # Clear previous stored states
        self._genesis_stored_states.clear()
        self._entity_qpos_before_ipc.clear()

        # Get qpos field as numpy array for kernel access
        qpos_np = rigid_solver._rigid_global_info.qpos.to_numpy()

        # Store qpos for all entities (for user modification detection)
        for entity_idx, entity in enumerate(rigid_solver._entities):
            if entity.n_qs > 0:  # Skip entities without dofs
                q_start = entity._q_start
                n_qs = entity.n_qs

                # Check capacity
                if n_qs > self.max_qpos_size:
                    gs.logger.warning(
                        f"Entity {entity_idx} qpos size {n_qs} exceeds max {self.max_qpos_size}. "
                        f"Using fallback method."
                    )
                    # Fallback to original method
                    for env_idx in range(self.sim._B):
                        qpos = np.zeros(n_qs, dtype=np.float32)
                        for i in range(n_qs):
                            qpos[i] = rigid_solver._rigid_global_info.qpos[q_start + i, env_idx]
                        self._entity_qpos_before_ipc[(entity_idx, env_idx)] = qpos
                else:
                    # Use kernel for batch reading
                    for env_idx in range(self.sim._B):
                        qpos_out = np.zeros(n_qs, dtype=np.float32)
                        self._batch_read_qpos_kernel(qpos_np, q_start, n_qs, env_idx, qpos_out)
                        self._entity_qpos_before_ipc[(entity_idx, env_idx)] = qpos_out.copy()

        # Store transforms for all rigid links
        # OPTIMIZED VERSION: Batch process transforms using kernel
        # Iterate through mesh handles to get all links
        if hasattr(rigid_solver, "_mesh_handles"):
            # Collect all link indices and env indices first
            link_env_pairs = []
            for handle_key in rigid_solver._mesh_handles.keys():
                if handle_key.startswith("rigid_link_"):
                    # Parse: "rigid_link_{env_idx}_{link_idx}"
                    parts = handle_key.split("_")
                    if len(parts) >= 4:
                        env_idx = int(parts[2])
                        link_idx = int(parts[3])
                        link_env_pairs.append((link_idx, env_idx))

            # Batch process transforms
            if link_env_pairs:
                # Get positions and quaternions in batch
                for link_idx, env_idx in link_env_pairs:
                    # Get and store current Genesis transform
                    genesis_transform = self._get_genesis_link_transform(link_idx, env_idx)

                    if link_idx not in self._genesis_stored_states:
                        self._genesis_stored_states[link_idx] = {}
                    self._genesis_stored_states[link_idx][env_idx] = genesis_transform

    def couple(self, f):
        """Execute IPC coupling step"""
        if not self.is_active:
            return

        # Dispatch to strategy-specific coupling logic
        if self.options.coupling_strategy == "two_way_soft_constraint":
            self._couple_two_way_soft_constraint(f)

    def _couple_two_way_soft_constraint(self, f):
        """Two-way coupling using SoftTransformConstraint"""
        # Step 1: Store current Genesis rigid body states (q_genesis^n)
        # This will be used by both animator (to set aim_transform) and
        # force computation (to ensure action-reaction force consistency)
        self._store_genesis_rigid_states()

        # Step 2: Advance IPC simulation
        # Animator reads stored Genesis states and sets them as IPC targets
        self._ipc_world.advance()
        self._ipc_world.retrieve()

        # Step 3: Retrieve IPC results and apply coupling forces
        # Now use IPC's new positions (q_ipc^{n+1}) and stored Genesis states (q_genesis^n)
        # to compute forces: F = M * (q_ipc^{n+1} - q_genesis^n)
        self._retrieve_fem_states(f)  # This handles both volumetric FEM and cloth
        self._retrieve_rigid_states(f)
        # Handle IPC-only links: directly set Genesis transform to IPC result (one-way coupling)
        self._set_genesis_transforms_from_ipc(ipc_only=True)

        if self.options.two_way_coupling:
            self._apply_abd_coupling_forces()

        if self.options.use_contact_proxy:
            self._record_ipc_contact_forces()
            self._apply_ipc_contact_forces()

    def _retrieve_fem_states(self, f):
        # IPC world advance/retrieve is handled at Scene level
        # This method handles both volumetric FEM (3D) and cloth (2D) post-processing

        if not self.fem_solver.is_active:
            return

        # Gather FEM states (both volumetric and cloth) using metadata filtering
        from uipc import builtin
        from uipc.backend import SceneVisitor
        from uipc.geometry import SimplicialComplexSlot, apply_transform, merge
        import numpy as np

        visitor = SceneVisitor(self._ipc_scene)

        # Collect FEM and cloth geometries using metadata
        fem_geo_by_entity = {}
        for geo_slot in visitor.geometries():
            if isinstance(geo_slot, SimplicialComplexSlot):
                geo = geo_slot.geometry()
                # Accept both 3D (volumetric FEM) and 2D (cloth) geometries
                if geo.dim() in [2, 3]:
                    try:
                        # Check solver type using metadata
                        meta_attrs = geo.meta()
                        solver_type_attr = meta_attrs.find("solver_type")

                        if solver_type_attr and solver_type_attr.name() == "solver_type":
                            # Read solver type from metadata
                            try:
                                solver_type_view = solver_type_attr.view()
                                if len(solver_type_view) > 0:
                                    solver_type = str(solver_type_view[0])
                                else:
                                    continue
                            except:
                                continue

                            # Accept both "fem" and "cloth" (both are FEM entities)
                            if solver_type in ["fem", "cloth"]:
                                env_idx_attr = meta_attrs.find("env_idx")
                                entity_idx_attr = meta_attrs.find("entity_idx")

                                if env_idx_attr and entity_idx_attr:
                                    # Read string values and convert to int
                                    env_idx_str = str(env_idx_attr.view()[0])
                                    entity_idx_str = str(entity_idx_attr.view()[0])
                                    env_idx = int(env_idx_str)
                                    entity_idx = int(entity_idx_str)

                                    if entity_idx not in fem_geo_by_entity:
                                        fem_geo_by_entity[entity_idx] = {}

                                    proc_geo = geo
                                    if geo.instances().size() >= 1:
                                        proc_geo = merge(apply_transform(geo))
                                    pos = proc_geo.positions().view().reshape(-1, 3)
                                    fem_geo_by_entity[entity_idx][env_idx] = pos

                    except Exception as e:
                        # Skip this geometry if metadata reading fails
                        continue

        # Update FEM entities using filtered geometries
        for entity_idx, env_positions in fem_geo_by_entity.items():
            if entity_idx < len(self.fem_solver._entities):
                entity = self.fem_solver._entities[entity_idx]
                env_pos_list = []

                for env_idx in range(self.sim._B):
                    if env_idx in env_positions:
                        env_pos_list.append(env_positions[env_idx])
                    else:
                        # Fallback for missing environment
                        env_pos_list.append(np.zeros((0, 3)))

                if env_pos_list:
                    all_env_pos = np.stack(env_pos_list, axis=0, dtype=gs.np_float)
                    entity.set_pos(0, all_env_pos)

    def _retrieve_rigid_states(self, f):
        """
        Handle rigid body IPC: Retrieve ABD transforms/affine matrices after IPC step
        and apply coupling forces back to Genesis rigid bodies
        """
        # IPC world advance/retrieve is handled at Scene level
        # Retrieve ABD transform matrices after IPC simulation

        if not hasattr(self, "_ipc_scene") or not hasattr(self.rigid_solver, "list_env_mesh"):
            return

        from uipc import builtin, view
        from uipc.backend import SceneVisitor
        from uipc.geometry import SimplicialComplexSlot
        import numpy as np
        import genesis.utils.geom as gu

        rigid_solver = self.rigid_solver
        visitor = SceneVisitor(self._ipc_scene)

        # Collect ABD geometries and their constraint data using metadata
        abd_data_by_link = {}  # link_idx -> {env_idx: {transform, gradient, mass}}

        for geo_slot in visitor.geometries():
            if isinstance(geo_slot, SimplicialComplexSlot):
                geo = geo_slot.geometry()
                if geo.dim() in [2, 3]:
                    try:
                        # Check if this is an ABD geometry using metadata
                        meta_attrs = geo.meta()
                        solver_type_attr = meta_attrs.find("solver_type")

                        if solver_type_attr and solver_type_attr.name() == "solver_type":
                            # Actually read solver type from metadata
                            try:
                                solver_type_view = solver_type_attr.view()
                                if len(solver_type_view) > 0:
                                    solver_type = str(solver_type_view[0])
                                else:
                                    continue
                            except:
                                continue

                            if solver_type == "rigid":
                                env_idx_attr = meta_attrs.find("env_idx")
                                link_idx_attr = meta_attrs.find("link_idx")

                                if env_idx_attr and link_idx_attr:
                                    # Read metadata values
                                    env_idx_str = str(env_idx_attr.view()[0])
                                    link_idx_str = str(link_idx_attr.view()[0])
                                    env_idx = int(env_idx_str)
                                    link_idx = int(link_idx_str)

                                    # Initialize link data structure
                                    if link_idx not in abd_data_by_link:
                                        abd_data_by_link[link_idx] = {}

                                    # Get current transform matrix from ABD object (after IPC solve)
                                    # This is q_ipc^{n+1}
                                    transforms = geo.transforms()
                                    transform_matrix = None
                                    if transforms.size() > 0:
                                        transform_matrix = view(transforms)[0].copy()  # 4x4 affine matrix

                                    # Get aim transform that was used by IPC during solve
                                    # This is q_genesis^n (stored before advance)
                                    aim_transform = None
                                    if (
                                        link_idx in self._genesis_stored_states
                                        and env_idx in self._genesis_stored_states[link_idx]
                                    ):
                                        aim_transform = self._genesis_stored_states[link_idx][env_idx]

                                    abd_data_by_link[link_idx][env_idx] = {
                                        "transform": transform_matrix,  # q_ipc^{n+1}
                                        "aim_transform": aim_transform,  # q_genesis^n
                                    }

                    except Exception as e:
                        gs.logger.warning(f"Failed to retrieve ABD geometry data: {e}")
                        continue

        # Store transforms for later access
        self.abd_data_by_link = abd_data_by_link

    def _set_genesis_transforms_from_ipc(self, ipc_only=True):
        """
        Set Genesis transforms from IPC results.

        Parameters
        ----------
        ipc_only : bool
            If True, only process links that are both IPC-only AND in IPC filters.
            If False, process all links in IPC filters (regardless of IPC-only status).

        User modification detection:
        If user manually called set_qpos between _store_genesis_rigid_states and now,
        we skip processing that entity to respect user control.
        """
        import numpy as np
        from scipy.spatial.transform import Rotation as R
        import torch

        rigid_solver = self.rigid_solver
        is_parallelized = self.sim._scene.n_envs > 0

        # Detect which entities were modified by user (set_qpos called after storage)
        # OPTIMIZED VERSION: Uses kernel for qpos comparison
        user_modified_entities = set()
        qpos_np = rigid_solver._rigid_global_info.qpos.to_numpy()

        for (entity_idx, env_idx), stored_qpos in self._entity_qpos_before_ipc.items():
            entity = rigid_solver._entities[entity_idx]
            q_start = entity._q_start
            n_qs = entity.n_qs

            if n_qs > self.max_qpos_size:
                # Fallback for large qpos
                current_qpos = np.zeros(n_qs, dtype=np.float32)
                for i in range(n_qs):
                    current_qpos[i] = rigid_solver._rigid_global_info.qpos[q_start + i, env_idx]
                if not np.allclose(current_qpos, stored_qpos, rtol=1e-6, atol=1e-6):
                    user_modified_entities.add((entity_idx, env_idx))
                    gs.logger.debug(
                        f"Entity {entity_idx} (env {env_idx}) qpos was modified by user, "
                        f"skipping IPC transform update"
                    )
            else:
                # Use kernel for comparison
                current_qpos = np.zeros(n_qs, dtype=np.float32)
                self._batch_read_qpos_kernel(qpos_np, q_start, n_qs, env_idx, current_qpos)

                # Use kernel to compare
                modified_flag = np.zeros(1, dtype=np.int32)
                self._compare_qpos_kernel(n_qs, current_qpos, stored_qpos, 1e-6, modified_flag)

                if modified_flag[0] == 1:
                    user_modified_entities.add((entity_idx, env_idx))
                    gs.logger.debug(
                        f"Entity {entity_idx} (env {env_idx}) qpos was modified by user, "
                        f"skipping IPC transform update"
                    )

        # Step 1: Filter links based on ipc_only flag
        filtered_links = {}  # {(entity_idx, link_idx, env_idx): transform_data}

        if not hasattr(self, "abd_data_by_link"):
            return

        for link_idx, env_data in self.abd_data_by_link.items():
            # Find which entity this link belongs to
            entity_idx = None
            for ent_idx, entity in enumerate(rigid_solver._entities):
                if entity._link_start <= link_idx < entity._link_start + entity.n_links:
                    entity_idx = ent_idx
                    break

            if entity_idx is None:
                continue

            # Check filtering criteria
            if ipc_only:
                # Must be both IPC-only AND in IPC filters
                is_ipc_only = entity_idx in self._ipc_only_links and link_idx in self._ipc_only_links[entity_idx]
                is_in_filter = entity_idx in self._ipc_link_filters and link_idx in self._ipc_link_filters[entity_idx]
                if not (is_ipc_only and is_in_filter):
                    continue
            else:
                # Must be in IPC filters
                is_in_filter = entity_idx in self._ipc_link_filters and link_idx in self._ipc_link_filters[entity_idx]
                if not is_in_filter:
                    continue

            # Store filtered link data
            for env_idx, data in env_data.items():
                filtered_links[(entity_idx, link_idx, env_idx)] = data

        # Step 2: Group filtered links by entity and env
        entity_env_links = {}  # {(entity_idx, env_idx): [(link_idx, transform_data), ...]}

        for (entity_idx, link_idx, env_idx), data in filtered_links.items():
            key = (entity_idx, env_idx)
            if key not in entity_env_links:
                entity_env_links[key] = []
            entity_env_links[key].append((link_idx, data))

        # Step 3: Process each entity-env group
        for (entity_idx, env_idx), link_data_list in entity_env_links.items():
            # Skip if user modified this entity's qpos
            if (entity_idx, env_idx) in user_modified_entities:
                continue

            entity = rigid_solver._entities[entity_idx]

            try:
                # Check if entity has only one link and it's the base link
                if len(link_data_list) == 1:
                    link_idx, data = link_data_list[0]
                    if link_idx == entity.base_link_idx:
                        # Simple case: single base link
                        ipc_transform = data.get("transform")
                        if ipc_transform is None:
                            continue

                        # Extract position and rotation
                        pos = ipc_transform[:3, 3]
                        rot_mat = ipc_transform[:3, :3]
                        quat_xyzw = R.from_matrix(rot_mat).as_quat()
                        quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])

                        # Convert to tensors
                        pos_tensor = torch.as_tensor(pos, dtype=gs.tc_float, device=gs.device).unsqueeze(0)
                        quat_tensor = torch.as_tensor(quat_wxyz, dtype=gs.tc_float, device=gs.device).unsqueeze(0)
                        base_links_idx = torch.tensor([link_idx], dtype=gs.tc_int, device=gs.device)

                        # Set base link transform
                        if is_parallelized:
                            rigid_solver.set_base_links_pos(
                                pos_tensor,
                                base_links_idx,
                                envs_idx=env_idx,
                                relative=False,
                                unsafe=True,
                                skip_forward=False,
                            )
                            rigid_solver.set_base_links_quat(
                                quat_tensor,
                                base_links_idx,
                                envs_idx=env_idx,
                                relative=False,
                                unsafe=True,
                                skip_forward=False,
                            )
                        else:
                            rigid_solver.set_base_links_pos(
                                pos_tensor,
                                base_links_idx,
                                envs_idx=None,
                                relative=False,
                                unsafe=True,
                                skip_forward=False,
                            )
                            rigid_solver.set_base_links_quat(
                                quat_tensor,
                                base_links_idx,
                                envs_idx=None,
                                relative=False,
                                unsafe=True,
                                skip_forward=False,
                            )

                        # Zero velocities
                        if is_parallelized:
                            entity.zero_all_dofs_velocity(envs_idx=env_idx, unsafe=True)
                        else:
                            entity.zero_all_dofs_velocity(envs_idx=None, unsafe=True)

                        continue

                # Complex case: multiple links or non-base link
                # Use inverse kinematics to compute qpos

                # Prepare target positions and quaternions for IK
                links = []
                poss = []
                quats = []

                for link_idx, data in link_data_list:
                    ipc_transform = data.get("transform")
                    if ipc_transform is None:
                        continue

                    # Get link object using local index
                    local_link_idx = link_idx - entity._link_start
                    if local_link_idx < 0 or local_link_idx >= entity.n_links:
                        continue
                    link = entity.links[local_link_idx]

                    # Extract position and quaternion
                    pos = ipc_transform[:3, 3]
                    rot_mat = ipc_transform[:3, :3]
                    quat_xyzw = R.from_matrix(rot_mat).as_quat()
                    quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])

                    links.append(link)
                    poss.append(pos)
                    quats.append(quat_wxyz)

                if not links:
                    continue

                # Call inverse kinematics
                qpos = entity.inverse_kinematics_multilink(
                    links=links,
                    poss=poss,
                    quats=quats,
                    envs_idx=env_idx if is_parallelized else None,
                    return_error=False,
                )

                if qpos is not None:
                    # Set qpos for this entity
                    entity.set_qpos(qpos, envs_idx=env_idx if is_parallelized else None, zero_velocity=True)

            except Exception as e:
                gs.logger.warning(f"Failed to set Genesis transforms for entity {entity_idx}, env {env_idx}: {e}")
                continue

    def _get_genesis_link_transform(self, link_idx, env_idx):
        """
        Get the current transform (4x4 matrix) of a Genesis rigid body link.

        Parameters
        ----------
        link_idx : int
            The link index
        env_idx : int
            The environment index

        Returns
        -------
        np.ndarray
            4x4 transformation matrix
        """
        from uipc import Transform, Vector3, Quaternion
        import numpy as np

        rigid_solver = self.rigid_solver
        is_parallelized = self.sim._scene.n_envs > 0

        # Get current link state from Genesis
        if is_parallelized:
            link_pos = rigid_solver.get_links_pos(links_idx=link_idx, envs_idx=env_idx)
            link_quat = rigid_solver.get_links_quat(links_idx=link_idx, envs_idx=env_idx)
        else:
            link_pos = rigid_solver.get_links_pos(links_idx=link_idx)
            link_quat = rigid_solver.get_links_quat(links_idx=link_idx)

        link_pos = link_pos.detach().cpu().numpy()
        link_quat = link_quat.detach().cpu().numpy()

        # Handle array shapes - squeeze down to 1D
        while len(link_pos.shape) > 1 and link_pos.shape[0] == 1:
            link_pos = link_pos[0]
        while len(link_quat.shape) > 1 and link_quat.shape[0] == 1:
            link_quat = link_quat[0]

        pos_1d = link_pos.flatten()[:3]
        quat_1d = link_quat.flatten()[:4]

        # Create transform matrix
        t = Transform.Identity()
        t.translate(Vector3.Values((pos_1d[0], pos_1d[1], pos_1d[2])))
        uipc_quat = Quaternion(quat_1d)
        t.rotate(uipc_quat)

        return t.matrix().copy()

    def _apply_abd_coupling_forces(self):
        """
        Apply coupling forces from IPC ABD constraint to Genesis rigid bodies using taichi kernel.

        This ensures action-reaction force consistency:
        - IPC constraint force: G_ipc = M * (q_ipc^{n+1} - q_genesis^n)
        - Genesis reaction force: F_genesis = M * (q_ipc^{n+1} - q_genesis^n) = G_ipc

        Where:
        - q_ipc^{n+1}: IPC ABD position after solve (from geo.transforms())
        - q_genesis^n: Genesis position before IPC advance (stored in _genesis_stored_states)
        - M: Mass matrix scaled by constraint strengths
        """
        import numpy as np
        import torch

        rigid_solver = self.rigid_solver
        strength_tuple = self.options.ipc_constraint_strength
        translation_strength = float(strength_tuple[0])
        rotation_strength = float(strength_tuple[1])

        dt = self.sim._dt
        dt2 = dt * dt

        # Collect all link data for batch processing
        link_indices = []
        env_indices = []
        ipc_transforms_list = []
        aim_transforms_list = []
        link_masses_list = []
        inertia_tensors_list = []

        for link_idx, env_data in self.abd_data_by_link.items():
            # Skip IPC-only links (they don't need coupling forces)
            is_ipc_only = False
            for entity_idx, link_set in self._ipc_only_links.items():
                if link_idx in link_set:
                    is_ipc_only = True
                    break

            if is_ipc_only:
                continue  # Skip IPC-only links

            for env_idx, data in env_data.items():
                ipc_transform = data.get("transform")  # Current transform after IPC solve
                aim_transform = data.get("aim_transform")  # Target from Genesis

                if ipc_transform is None or aim_transform is None:
                    continue

                try:
                    link_indices.append(link_idx)
                    env_indices.append(env_idx)
                    ipc_transforms_list.append(ipc_transform)
                    aim_transforms_list.append(aim_transform)
                    link_masses_list.append(float(rigid_solver.links_info.inertial_mass[link_idx]))
                    inertia_tensors_list.append(rigid_solver.links_info.inertial_i[link_idx].to_numpy())
                except Exception as e:
                    gs.logger.warning(f"Failed to collect data for link {link_idx}, env {env_idx}: {e}")
                    continue

        if not link_indices:
            return  # No links to process

        # Convert to numpy arrays for kernel
        n_links = len(link_indices)
        ipc_transforms = np.array(ipc_transforms_list, dtype=np.float32)  # (n_links, 4, 4)
        aim_transforms = np.array(aim_transforms_list, dtype=np.float32)  # (n_links, 4, 4)
        link_masses = np.array(link_masses_list, dtype=np.float32)  # (n_links,)
        inertia_tensors = np.array(inertia_tensors_list, dtype=np.float32)  # (n_links, 3, 3)

        # Allocate output arrays
        out_forces = np.zeros((n_links, 3), dtype=np.float32)
        out_torques = np.zeros((n_links, 3), dtype=np.float32)

        # Call taichi kernel
        self._compute_coupling_forces_kernel(
            n_links,
            ipc_transforms,
            aim_transforms,
            link_masses,
            inertia_tensors,
            translation_strength,
            rotation_strength,
            dt2,
            out_forces,
            out_torques,
        )

        # Apply forces to Genesis rigid bodies - OPTIMIZED batch processing
        is_parallelized = self.sim._scene.n_envs > 0

        if is_parallelized:
            # Group by environment
            env_batches = {}  # {env_idx: {'link_indices': [], 'forces': [], 'torques': []}}
            for i in range(n_links):
                env_idx = env_indices[i]
                if env_idx not in env_batches:
                    env_batches[env_idx] = {"link_indices": [], "forces": [], "torques": []}
                env_batches[env_idx]["link_indices"].append(link_indices[i])
                env_batches[env_idx]["forces"].append(out_forces[i])
                env_batches[env_idx]["torques"].append(out_torques[i])

            # Apply forces per environment
            for env_idx, batch in env_batches.items():
                for j, link_idx in enumerate(batch["link_indices"]):
                    try:
                        force_input = batch["forces"][j].reshape(1, 1, 3)
                        torque_input = batch["torques"][j].reshape(1, 1, 3)
                        rigid_solver.apply_links_external_force(force=force_input, links_idx=link_idx, envs_idx=env_idx)
                        rigid_solver.apply_links_external_torque(
                            torque=torque_input, links_idx=link_idx, envs_idx=env_idx
                        )
                    except Exception as e:
                        gs.logger.warning(f"Failed to apply ABD coupling force for link {link_idx}, env {env_idx}: {e}")
                        continue
        else:
            # Non-parallelized: apply all forces
            for i in range(n_links):
                link_idx = link_indices[i]
                try:
                    force_input = out_forces[i].reshape(1, 3)
                    torque_input = out_torques[i].reshape(1, 3)
                    rigid_solver.apply_links_external_force(force=force_input, links_idx=link_idx)
                    rigid_solver.apply_links_external_torque(torque=torque_input, links_idx=link_idx)
                except Exception as e:
                    gs.logger.warning(f"Failed to apply ABD coupling force for link {link_idx}: {e}")
                    continue

    def couple_grad(self, f):
        """Gradient computation for coupling"""
        # IPC doesn't support gradients yet
        pass

    def reset(self, envs_idx=None):
        """Reset coupling state"""
        # IPC doesn't need special reset logic currently
        pass

    def _init_ipc_gui(self):
        """Initialize IPC GUI for debugging"""
        try:
            import polyscope as ps
            from uipc.gui import SceneGUI

            self.ps = ps

            # Initialize SceneGUI for IPC scene
            self._ipc_scene_gui = SceneGUI(self._ipc_scene)

            # Initialize polyscope if not already done
            if not ps.is_initialized():
                ps.init()

            # Register IPC GUI with polyscope
            self._ipc_scene_gui.register()
            self._ipc_scene_gui.set_edge_width(1)

            # Set up ground plane display in polyscope to match Genesis z=0
            ps.set_up_dir("z_up")
            ps.set_ground_plane_height(0.0)  # Set at z=0 to match Genesis

            # Show polyscope window for first frame to initialize properly
            ps.show(forFrames=1)
            # Flag to control GUI updates
            self.sim._scene._ipc_gui_enabled = True

            gs.logger.info("IPC GUI initialized successfully")

        except Exception as e:
            gs.logger.warning(f"Failed to initialize IPC GUI: {e}")
            self.sim._scene._ipc_gui_enabled = False

    def update_ipc_gui(self):
        """Update IPC GUI"""
        self.ps.frame_tick()  # Non-blocking frame update
        self._ipc_scene_gui.update()

    def _compute_link_contact_forces_and_torques(self, total_force_dict, vertex_to_link, link_vertex_positions):
        """
        Compute total contact forces and torques for each rigid link from vertex gradients.

        OPTIMIZED VERSION: Uses Taichi kernel for parallel computation.

        This function follows the pattern from test_affine_body_contact_force.py, computing
        the total force and torque acting on each rigid body by accumulating contributions
        from all contact vertices.

        Parameters
        ----------
        total_force_dict : dict
            Dictionary mapping vertex indices to contact force gradients {vertex_idx: force_gradient_vector}
        vertex_to_link : dict
            Mapping from global vertex index to (link_idx, env_idx, local_idx)
        link_vertex_positions : dict
            Mapping from (link_idx, env_idx) to list of vertex positions in world space

        Returns
        -------
        dict
            Dictionary mapping (link_idx, env_idx) to {'force': np.array, 'torque': np.array, 'center': np.array}

        Notes
        -----
        - Force is computed as the negative sum of contact gradients: F = -∑grad
        - Torque is computed as τ = ∑(r × F) where r is the vector from link center to contact point
        - Link center is computed as the average of all vertex positions
        """
        import numpy as np

        if not total_force_dict:
            return {}

        # Step 1: Prepare link centers (compute once per (link_idx, env_idx))
        link_centers_dict = {}  # {(link_idx, env_idx): center}
        for (link_idx, env_idx), verts in link_vertex_positions.items():
            link_centers_dict[(link_idx, env_idx)] = np.mean(verts, axis=0)

        # Step 2: Prepare data for kernel - collect all vertex force entries
        vertex_data = []
        for vert_idx, force_grad in total_force_dict.items():
            if vert_idx not in vertex_to_link:
                continue  # Vertex doesn't belong to a both-coupling link

            link_idx, env_idx, local_idx = vertex_to_link[vert_idx]

            # Get vertex position and link center
            if (link_idx, env_idx) in link_vertex_positions:
                contact_pos = link_vertex_positions[(link_idx, env_idx)][local_idx]
                center_pos = link_centers_dict.get((link_idx, env_idx))
                if center_pos is not None:
                    vertex_data.append(
                        {
                            "force_grad": force_grad,
                            "link_idx": link_idx,
                            "env_idx": env_idx,
                            "contact_pos": contact_pos,
                            "center_pos": center_pos,
                        }
                    )

        if not vertex_data:
            return {}

        # Check capacity
        n_entries = len(vertex_data)
        if n_entries > self.max_vertex_contacts:
            gs.logger.warning(
                f"Vertex contact capacity exceeded: {n_entries} > {self.max_vertex_contacts}. "
                f"Truncating to {self.max_vertex_contacts}."
            )
            n_entries = self.max_vertex_contacts
            vertex_data = vertex_data[:n_entries]

        # Step 3: Populate Taichi fields
        for i, data in enumerate(vertex_data):
            self.vertex_force_gradients[i] = data["force_grad"]
            self.vertex_link_indices[i] = data["link_idx"]
            self.vertex_env_indices[i] = data["env_idx"]
            self.vertex_positions_world[i] = data["contact_pos"]
            self.vertex_link_centers[i] = data["center_pos"]

        # Step 4: Clear output arrays
        self.link_contact_forces_out.fill(0.0)
        self.link_contact_torques_out.fill(0.0)

        # Step 5: Call kernel
        force_grads_np = self.vertex_force_gradients.to_numpy()[:n_entries]
        link_idx_np = self.vertex_link_indices.to_numpy()[:n_entries]
        env_idx_np = self.vertex_env_indices.to_numpy()[:n_entries]
        vert_pos_np = self.vertex_positions_world.to_numpy()[:n_entries]
        link_centers_np = self.vertex_link_centers.to_numpy()[:n_entries]

        out_forces_np = self.link_contact_forces_out.to_numpy()
        out_torques_np = self.link_contact_torques_out.to_numpy()

        self._compute_link_contact_forces_kernel(
            n_entries,
            force_grads_np,
            link_idx_np,
            env_idx_np,
            vert_pos_np,
            link_centers_np,
            out_forces_np,
            out_torques_np,
        )

        # Copy results back to Taichi field
        self.link_contact_forces_out.from_numpy(out_forces_np)
        self.link_contact_torques_out.from_numpy(out_torques_np)

        # Step 6: Extract results into dictionary
        link_forces = {}  # {(link_idx, env_idx): {'force': np.array, 'torque': np.array, 'center': np.array}}

        for (link_idx, env_idx), center in link_centers_dict.items():
            force = out_forces_np[link_idx, env_idx]
            torque = out_torques_np[link_idx, env_idx]

            # Only include if there's non-zero force/torque
            if np.any(force != 0.0) or np.any(torque != 0.0):
                link_forces[(link_idx, env_idx)] = {
                    "force": force.copy(),
                    "torque": torque.copy(),
                    "center": center,
                }

        return link_forces

    def _record_ipc_contact_forces(self):
        """
        Record contact forces from IPC for 'both' coupling links.

        This method extracts contact forces and torques from IPC's contact system
        and stores them for later application to Genesis rigid bodies.
        Only processes links that are in _ipc_link_filters but NOT in _ipc_only_links.
        """
        import numpy as np
        from uipc import view
        from uipc.geometry import Geometry

        # Clear previous contact forces
        self._ipc_contact_forces.clear()

        # Get contact feature from IPC world
        features = self._ipc_world.features()
        contact_feature = features.find("core/contact_system")

        if contact_feature is None:
            return  # No contact system available

        # Get available contact primitive types
        prim_types = contact_feature.contact_primitive_types()

        # Accumulate contact gradients (forces) for all vertices
        # NOTE: IPC gradients are actually force * dt^2, so we need to divide by dt^2
        dt = self.options.dt
        dt2 = dt * dt
        total_force_dict = {}  # {vertex_index: force_vector}

        for prim_type in prim_types:
            # Get contact gradient for this primitive type
            vert_grad = Geometry()
            contact_feature.contact_gradient(prim_type, vert_grad)

            # Extract gradient data from instances
            instances = vert_grad.instances()
            i_attr = instances.find("i")  # Vertex indices
            grad_attr = instances.find("grad")  # Gradient vectors

            if i_attr is not None and grad_attr is not None:
                indices = view(i_attr)
                gradients = view(grad_attr)

                # Accumulate gradients for each vertex
                # Gradients from IPC are force * dt^2, so divide by dt^2 to get actual force
                for idx, grad in zip(indices, gradients):
                    grad_vec = np.array(grad).flatten()
                    if idx not in total_force_dict:
                        total_force_dict[idx] = np.zeros(3)
                    total_force_dict[idx] += grad_vec[:3] / dt2  # Convert gradient to force

        if not total_force_dict:
            return  # No contact forces to process

        # Use pre-built vertex-to-link mapping (built once during IPC setup)
        vertex_to_link = self._vertex_to_link_mapping

        # Get current vertex positions for contact force computation
        from uipc.backend import SceneVisitor
        from uipc.geometry import SimplicialComplexSlot

        visitor = SceneVisitor(self._ipc_scene)
        link_vertex_positions = {}  # {(link_idx, env_idx): [vertex_positions]}

        global_vertex_offset = 0

        for geo_slot in visitor.geometries():
            if isinstance(geo_slot, SimplicialComplexSlot):
                geo = geo_slot.geometry()
                if geo.dim() in [2, 3]:
                    n_verts = geo.vertices().size()

                    try:
                        # Check if this is a rigid geometry using metadata
                        meta_attrs = geo.meta()
                        solver_type_attr = meta_attrs.find("solver_type")

                        if solver_type_attr and solver_type_attr.name() == "solver_type":
                            try:
                                solver_type_view = solver_type_attr.view()
                                if len(solver_type_view) > 0:
                                    solver_type = str(solver_type_view[0])
                                else:
                                    continue
                            except:
                                continue

                            if solver_type == "rigid":
                                env_idx_attr = meta_attrs.find("env_idx")
                                link_idx_attr = meta_attrs.find("link_idx")

                                if env_idx_attr and link_idx_attr:
                                    env_idx_str = str(env_idx_attr.view()[0])
                                    link_idx_str = str(link_idx_attr.view()[0])
                                    env_idx = int(env_idx_str)
                                    link_idx = int(link_idx_str)

                                    # Check if any vertex of this geometry is in the mapping
                                    first_vertex_idx = global_vertex_offset
                                    if first_vertex_idx in vertex_to_link:
                                        # Get current ABD transform to transform vertices to world space
                                        transforms = geo.transforms()
                                        if transforms.size() > 0:
                                            transform_matrix = view(transforms)[0]  # 4x4 affine matrix

                                            # Get local vertex positions
                                            positions = view(geo.positions())

                                            # Transform and store vertex positions
                                            if (link_idx, env_idx) not in link_vertex_positions:
                                                link_vertex_positions[(link_idx, env_idx)] = []

                                            for local_idx in range(n_verts):
                                                # Transform local position to world space using ABD transform
                                                local_pos = np.array(positions[local_idx]).flatten()[:3]
                                                local_pos_homogeneous = np.append(local_pos, 1.0)
                                                world_pos = (transform_matrix @ local_pos_homogeneous)[:3]
                                                link_vertex_positions[(link_idx, env_idx)].append(world_pos)

                    except Exception as e:
                        gs.logger.warning(f"Failed to process geometry for contact forces: {e}")
                    finally:
                        # Always update offset
                        global_vertex_offset += n_verts

        # Compute contact forces and torques for each link
        link_forces = self._compute_link_contact_forces_and_torques(
            total_force_dict, vertex_to_link, link_vertex_positions
        )

        # Store forces in the proper format
        for (link_idx, env_idx), data in link_forces.items():
            if link_idx not in self._ipc_contact_forces:
                self._ipc_contact_forces[link_idx] = {}

            self._ipc_contact_forces[link_idx][env_idx] = {"force": data["force"], "torque": data["torque"]}

        # Compute external wrench from contact forces using taichi kernel
        # Collect data directly into pre-allocated Taichi fields
        contact_idx = 0

        for link_idx, env_data in self._ipc_contact_forces.items():
            for env_idx, force_data in env_data.items():
                # Check if we've exceeded pre-allocated capacity
                if contact_idx >= self.contact_forces_ti.shape[0]:
                    gs.logger.warning(
                        f"Contact capacity exceeded: {contact_idx} >= {self.contact_forces_ti.shape[0]}. "
                        f"Increase max_contacts in __init__."
                    )
                    break

                # Get contact force and torque
                force = force_data["force"]
                torque = force_data["torque"]

                # Get ABD transform for this link
                if hasattr(self, "abd_data_by_link") and link_idx in self.abd_data_by_link:
                    if env_idx in self.abd_data_by_link[link_idx]:
                        abd_transform = self.abd_data_by_link[link_idx][env_idx].get("transform")
                        if abd_transform is not None:
                            # Write directly to pre-allocated Taichi fields
                            self.contact_forces_ti[contact_idx] = force
                            self.contact_torques_ti[contact_idx] = torque
                            self.abd_transforms_ti[contact_idx] = abd_transform
                            self.link_indices_ti[contact_idx] = link_idx
                            self.env_indices_ti[contact_idx] = env_idx
                            contact_idx += 1

        if contact_idx > 0:
            # Use a view/slice of the pre-allocated arrays for the kernel call
            # Note: We still need to pass ndarray to kernel, so convert the used portion
            n_links = contact_idx
            contact_forces_arr = self.contact_forces_ti.to_numpy()[:n_links]
            contact_torques_arr = self.contact_torques_ti.to_numpy()[:n_links]
            abd_transforms_arr = self.abd_transforms_ti.to_numpy()[:n_links]
            out_wrenches = np.zeros((n_links, 12), dtype=np.float32)

            # Call taichi kernel to compute wrenches
            self._compute_external_wrench_kernel(
                n_links,
                contact_forces_arr,
                contact_torques_arr,
                abd_transforms_arr,
                out_wrenches,
            )

            # Store wrenches in _external_wrench_data for animator to use
            for i in range(n_links):
                link_idx = self.link_indices_ti[i]
                env_idx = self.env_indices_ti[i]
                wrench_vector = out_wrenches[i].astype(np.float64)  # Convert to float64 for IPC
                self._external_wrench_data[(link_idx, env_idx)] = wrench_vector

    def _apply_ipc_contact_forces(self):
        """
        Apply recorded IPC contact forces to Genesis rigid bodies.

        OPTIMIZED VERSION: Batches forces and torques to reduce API call overhead.

        This method takes the contact forces and torques recorded by _record_ipc_contact_forces
        and applies them to the corresponding Genesis rigid links.
        """
        import torch
        import numpy as np

        if not self._ipc_contact_forces:
            return  # No contact forces to apply

        rigid_solver = self.rigid_solver
        is_parallelized = self.sim._scene.n_envs > 0

        # Collect all forces and torques into batches
        # Group by environment for parallelized scenes, or single batch for non-parallelized
        if is_parallelized:
            # For parallelized scenes, we need to group by env_idx
            env_batches = {}  # {env_idx: {'link_indices': [], 'forces': [], 'torques': []}}

            for link_idx, env_data in self._ipc_contact_forces.items():
                for env_idx, force_data in env_data.items():
                    if env_idx not in env_batches:
                        env_batches[env_idx] = {"link_indices": [], "forces": [], "torques": []}

                    env_batches[env_idx]["link_indices"].append(link_idx)
                    env_batches[env_idx]["forces"].append(force_data["force"] * 0.5)
                    env_batches[env_idx]["torques"].append(force_data["torque"] * 0.5)

            # Apply forces for each environment batch
            for env_idx, batch_data in env_batches.items():
                if not batch_data["link_indices"]:
                    continue

                # Convert to numpy arrays and then to torch tensors
                forces_np = np.array(batch_data["forces"], dtype=np.float32)  # (n_links, 3)
                torques_np = np.array(batch_data["torques"], dtype=np.float32)  # (n_links, 3)
                link_indices = batch_data["link_indices"]

                # Apply each force/torque individually (Genesis API limitation)
                # TODO: Check if Genesis supports batch application
                for i, link_idx in enumerate(link_indices):
                    force_tensor = torch.as_tensor(forces_np[i], dtype=gs.tc_float, device=gs.device).unsqueeze(0)
                    torque_tensor = torch.as_tensor(torques_np[i], dtype=gs.tc_float, device=gs.device).unsqueeze(0)

                    rigid_solver.apply_links_external_force(
                        force=force_tensor,
                        links_idx=link_idx,
                        envs_idx=env_idx,
                        local=False,
                    )
                    rigid_solver.apply_links_external_torque(
                        torque=torque_tensor,
                        links_idx=link_idx,
                        envs_idx=env_idx,
                        local=False,
                    )

        else:
            # For non-parallelized scenes, batch all together
            all_link_indices = []
            all_forces = []
            all_torques = []

            for link_idx, env_data in self._ipc_contact_forces.items():
                for env_idx, force_data in env_data.items():
                    all_link_indices.append(link_idx)
                    all_forces.append(force_data["force"] * 0.5)
                    all_torques.append(force_data["torque"] * 0.5)

            if not all_link_indices:
                return

            # Convert to numpy arrays
            forces_np = np.array(all_forces, dtype=np.float32)  # (n_links, 3)
            torques_np = np.array(all_torques, dtype=np.float32)  # (n_links, 3)

            # Apply each force/torque individually
            for i, link_idx in enumerate(all_link_indices):
                force_tensor = torch.as_tensor(forces_np[i], dtype=gs.tc_float, device=gs.device).unsqueeze(0)
                torque_tensor = torch.as_tensor(torques_np[i], dtype=gs.tc_float, device=gs.device).unsqueeze(0)

                rigid_solver.apply_links_external_force(
                    force=force_tensor,
                    links_idx=link_idx,
                    local=False,
                )
                rigid_solver.apply_links_external_torque(
                    torque=torque_tensor,
                    links_idx=link_idx,
                    local=False,
                )
