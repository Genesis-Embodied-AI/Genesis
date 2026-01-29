"""
Data-oriented classes for IPC coupler Taichi fields.

These classes manage pre-allocated Taichi fields used for GPU-accelerated
IPC coupling computations.
"""

import gstaichi as ti

import genesis as gs


@ti.data_oriented
class IPCTransformData:
    """Data-oriented class for IPC transform processing."""

    def __init__(self, max_links, max_envs, max_abd_links, max_qpos_size):
        # Entity mapping: link_idx -> entity_idx, base_link_idx
        self.link_to_entity_map = ti.field(dtype=ti.i32, shape=max_links)
        self.entity_base_link_map = ti.field(dtype=ti.i32, shape=max_links)
        self.entity_n_links_map = ti.field(dtype=ti.i32, shape=max_links)
        self.entity_link_starts = ti.field(dtype=ti.i32, shape=max_links)

        # Filter flags: for each link, whether it passes ipc_only filter
        self.ipc_only_flags = ti.field(dtype=ti.i32, shape=max_links)
        self.ipc_filter_flags = ti.field(dtype=ti.i32, shape=max_links)

        # User modified entity flags
        self.user_modified_flags = ti.field(dtype=ti.i32, shape=(max_links, max_envs))

        # Input data for filtering
        self.input_link_indices = ti.field(dtype=ti.i32, shape=max_abd_links)
        self.input_transforms = ti.Matrix.field(4, 4, dtype=gs.ti_float, shape=(max_abd_links, max_envs))
        self.input_env_indices = ti.field(dtype=ti.i32, shape=(max_abd_links, max_envs))
        self.input_valid = ti.field(dtype=ti.i32, shape=(max_abd_links, max_envs))

        # Batch output arrays per environment (compacted)
        self.output_count_per_env = ti.field(dtype=ti.i32, shape=max_envs)
        self.output_link_idx = ti.field(dtype=ti.i32, shape=(max_envs, max_links))
        self.output_pos = ti.Vector.field(3, dtype=gs.ti_float, shape=(max_envs, max_links))
        self.output_quat = ti.Vector.field(4, dtype=gs.ti_float, shape=(max_envs, max_links))
        self.output_entity_idx = ti.field(dtype=ti.i32, shape=(max_envs, max_links))

        # Complex case tracking
        self.complex_case_flags = ti.field(dtype=ti.i32, shape=(max_links, max_envs))

        # Reusable buffers for qpos comparison
        # Note: max_qpos_size is passed from IPCCoupler.MAX_QPOS_SIZE
        # The large buffer size (2000) is from IPCCoupler.MAX_QPOS_BUFFER_LARGE
        self.qpos_buffer = ti.field(dtype=gs.ti_float, shape=max_qpos_size)
        self.qpos_buffer_large = ti.field(dtype=gs.ti_float, shape=max_qpos_size * 4)  # For large entities
        self.modified_flag = ti.field(dtype=ti.i32, shape=())
        self.qpos_comparison_result = ti.field(dtype=ti.i32, shape=1)  # For kernel-based comparison

        # Stored Genesis states (used by all coupling strategies)
        # Stored link transforms: pos + quat for all links
        self.stored_link_pos = ti.Vector.field(3, dtype=gs.ti_float, shape=(max_links, max_envs))
        self.stored_link_quat = ti.Vector.field(4, dtype=gs.ti_float, shape=(max_links, max_envs))
        self.stored_link_valid = ti.field(dtype=ti.i32, shape=(max_links, max_envs))  # 1 if stored, 0 otherwise

        # Stored qpos for all entities (used by all coupling strategies)
        # max_links is reused as max entity count (should be sufficient)
        self.stored_qpos = ti.field(dtype=gs.ti_float, shape=(max_links, max_envs, max_qpos_size))
        self.stored_qpos_size = ti.field(dtype=ti.i32, shape=max_links)  # Number of dofs per entity
        self.stored_qpos_start = ti.field(dtype=ti.i32, shape=max_links)  # DOF start index per entity


@ti.data_oriented
class IPCCouplingData:
    """Data-oriented class for IPC coupling force computation."""

    def __init__(self, max_links):
        # Pre-allocated buffers for coupling force computation
        self.link_indices = ti.field(dtype=ti.i32, shape=max_links)
        self.env_indices = ti.field(dtype=ti.i32, shape=max_links)
        self.ipc_transforms = ti.Matrix.field(4, 4, dtype=gs.ti_float, shape=max_links)
        self.aim_transforms = ti.Matrix.field(4, 4, dtype=gs.ti_float, shape=max_links)
        self.link_masses = ti.field(dtype=gs.ti_float, shape=max_links)
        self.inertia_tensors = ti.Matrix.field(3, 3, dtype=gs.ti_float, shape=max_links)
        self.out_forces = ti.Vector.field(3, dtype=gs.ti_float, shape=max_links)
        self.out_torques = ti.Vector.field(3, dtype=gs.ti_float, shape=max_links)
        self.n_items = ti.field(dtype=ti.i32, shape=())


@ti.data_oriented
class ArticulationData:
    """Data-oriented class for joint articulation coupling with Taichi parallelization."""

    def __init__(self, max_entities, max_dofs_per_entity, max_joints_per_entity, max_envs):
        # Entity-level metadata
        self.n_entities = ti.field(dtype=ti.i32, shape=())
        self.entity_indices = ti.field(dtype=ti.i32, shape=max_entities)
        self.entity_env_indices = ti.field(dtype=ti.i32, shape=max_entities)
        self.entity_n_dofs = ti.field(dtype=ti.i32, shape=max_entities)
        self.entity_n_joints = ti.field(dtype=ti.i32, shape=max_entities)
        self.entity_dof_start = ti.field(dtype=ti.i32, shape=max_entities)  # DOF start index in rigid solver

        # Joint to DOF mapping (per entity)
        # joint_dof_indices[entity_idx, joint_idx] = local DOF index
        self.joint_dof_indices = ti.field(dtype=ti.i32, shape=(max_entities, max_joints_per_entity))

        # DOF data (per entity, per environment)
        self.ref_dof_prev = ti.field(dtype=gs.ti_float, shape=(max_entities, max_envs, max_dofs_per_entity))
        self.qpos_current = ti.field(dtype=gs.ti_float, shape=(max_entities, max_envs, max_dofs_per_entity))
        self.qvel_genesis = ti.field(dtype=gs.ti_float, shape=(max_entities, max_envs, max_dofs_per_entity))
        self.qpos_new = ti.field(dtype=gs.ti_float, shape=(max_entities, max_envs, max_dofs_per_entity))

        # Joint data (per entity, per environment)
        self.delta_theta_tilde = ti.field(dtype=gs.ti_float, shape=(max_entities, max_envs, max_joints_per_entity))
        self.delta_theta_ipc = ti.field(dtype=gs.ti_float, shape=(max_entities, max_envs, max_joints_per_entity))

        # Mass matrix (per entity, flattened column-major)
        max_mass_size = max_joints_per_entity * max_joints_per_entity
        self.mass_matrix = ti.field(dtype=gs.ti_float, shape=(max_entities, max_mass_size))

        # Previous timestep link transforms for ref_dof_prev computation
        # Stores link indices and transform matrices from previous step
        # Dictionary: {(entity_idx, joint_idx, env_idx): transform_matrix_4x4}
        self.prev_link_transforms = {}
