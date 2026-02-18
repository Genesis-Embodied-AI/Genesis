"""
Data-oriented classes for IPC coupler Taichi fields.

These classes manage pre-allocated Taichi fields used for GPU-accelerated
IPC coupling computations.
"""

import quadrants as qd

import genesis as gs


@qd.data_oriented
class IPCTransformData:
    """Data-oriented class for IPC transform processing."""

    def __init__(self, max_links, max_envs, max_abd_links, max_qpos_size):
        # Entity mapping: link_idx -> entity_idx, base_link_idx
        self.link_to_entity_map = qd.field(dtype=qd.i32, shape=max_links)
        self.entity_base_link_map = qd.field(dtype=qd.i32, shape=max_links)
        self.entity_n_links_map = qd.field(dtype=qd.i32, shape=max_links)
        self.entity_link_starts = qd.field(dtype=qd.i32, shape=max_links)

        # Filter flags: for each link, whether it passes ipc_only filter
        self.ipc_only_flags = qd.field(dtype=qd.i32, shape=max_links)
        self.ipc_filter_flags = qd.field(dtype=qd.i32, shape=max_links)

        # User modified entity flags
        self.user_modified_flags = qd.field(dtype=qd.i32, shape=(max_links, max_envs))

        # Input data for filtering
        self.input_link_indices = qd.field(dtype=qd.i32, shape=max_abd_links)
        self.input_transforms = qd.Matrix.field(4, 4, dtype=gs.qd_float, shape=(max_abd_links, max_envs))
        self.input_env_indices = qd.field(dtype=qd.i32, shape=(max_abd_links, max_envs))
        self.input_valid = qd.field(dtype=qd.i32, shape=(max_abd_links, max_envs))

        # Batch output arrays per environment (compacted)
        self.output_count_per_env = qd.field(dtype=qd.i32, shape=max_envs)
        self.output_link_idx = qd.field(dtype=qd.i32, shape=(max_envs, max_links))
        self.output_pos = qd.Vector.field(3, dtype=gs.qd_float, shape=(max_envs, max_links))
        self.output_quat = qd.Vector.field(4, dtype=gs.qd_float, shape=(max_envs, max_links))
        self.output_entity_idx = qd.field(dtype=qd.i32, shape=(max_envs, max_links))

        # Complex case tracking
        self.complex_case_flags = qd.field(dtype=qd.i32, shape=(max_links, max_envs))

        # Reusable buffers for qpos comparison
        # Note: max_qpos_size is passed from IPCCoupler.MAX_QPOS_SIZE
        # The large buffer size (2000) is from IPCCoupler.MAX_QPOS_BUFFER_LARGE
        self.qpos_buffer = qd.field(dtype=gs.qd_float, shape=max_qpos_size)
        self.qpos_buffer_large = qd.field(dtype=gs.qd_float, shape=max_qpos_size * 4)  # For large entities
        self.modified_flag = qd.field(dtype=qd.i32, shape=())
        self.qpos_comparison_result = qd.field(dtype=qd.i32, shape=1)  # For kernel-based comparison

        # Stored Genesis states (used by all coupling strategies)
        # Stored link transforms: pos + quat for all links
        self.stored_link_pos = qd.Vector.field(3, dtype=gs.qd_float, shape=(max_links, max_envs))
        self.stored_link_quat = qd.Vector.field(4, dtype=gs.qd_float, shape=(max_links, max_envs))
        self.stored_link_valid = qd.field(dtype=qd.i32, shape=(max_links, max_envs))  # 1 if stored, 0 otherwise

        # Stored qpos for all entities (used by all coupling strategies)
        # max_links is reused as max entity count (should be sufficient)
        self.stored_qpos = qd.field(dtype=gs.qd_float, shape=(max_links, max_envs, max_qpos_size))
        self.stored_qpos_size = qd.field(dtype=qd.i32, shape=max_links)  # Number of dofs per entity
        self.stored_qpos_start = qd.field(dtype=qd.i32, shape=max_links)  # DOF start index per entity


class IPCCouplingData:
    """Data-oriented class for IPC coupling force computation using numpy arrays."""

    def __init__(self, max_links):
        # Pre-allocated numpy buffers for coupling force computation
        # These will be passed directly to Taichi kernels using qd.types.ndarray()
        import numpy as np

        self.link_indices = np.empty(max_links, dtype=np.int32)
        self.env_indices = np.empty(max_links, dtype=np.int32)
        self.ipc_transforms = np.empty((max_links, 4, 4), dtype=gs.np_float)
        self.aim_transforms = np.empty((max_links, 4, 4), dtype=gs.np_float)
        self.link_masses = np.empty(max_links, dtype=gs.np_float)
        self.inertia_tensors = np.empty((max_links, 3, 3), dtype=gs.np_float)
        self.out_forces = np.empty((max_links, 3), dtype=gs.np_float)
        self.out_torques = np.empty((max_links, 3), dtype=gs.np_float)
        self.n_items = 0  # Track actual number of items used
