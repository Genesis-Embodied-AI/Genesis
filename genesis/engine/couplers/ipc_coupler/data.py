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


class IPCCouplingData:
    """Data-oriented class for IPC coupling force computation using numpy arrays."""

    def __init__(self, max_links):
        # Pre-allocated numpy buffers for coupling force computation
        # These will be passed directly to Taichi kernels using ti.types.ndarray()
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
