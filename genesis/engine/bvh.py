import genesis as gs
import taichi as ti
from genesis.repr_base import RBC
import numpy as np


@ti.data_oriented
class AABB(RBC):
    """
    AABB (Axis-Aligned Bounding Box) class for managing collections of bounding boxes in batches.

    This class defines an axis-aligned bounding box (AABB) structure and provides a Taichi dataclass
    for efficient computation and intersection testing on the GPU. Each AABB is represented by its
    minimum and maximum 3D coordinates. The class supports batch processing of multiple AABBs.

    Attributes:
        n_batches (int): Number of batches of AABBs.
        n_aabbs (int): Number of AABBs per batch.
        ti_aabb (taichi.dataclass): Taichi dataclass representing an individual AABB with min and max vectors.
        aabbs (taichi.field): Taichi field storing all AABBs in the specified batches.

    Args:
        n_batches (int): Number of batches to allocate.
        n_aabbs (int): Number of AABBs per batch.

    Example:
        aabb_manager = AABB(n_batches=4, n_aabbs=128)
    """

    def __init__(self, n_batches, n_aabbs):
        self.n_batches = n_batches
        self.n_aabbs = n_aabbs

        @ti.dataclass
        class ti_aabb:
            min: gs.ti_vec3
            max: gs.ti_vec3

            @ti.func
            def intersects(self, other) -> bool:
                """
                Check if this AABB intersects with another AABB.
                """
                return (
                    self.min[0] <= other.max[0]
                    and self.max[0] >= other.min[0]
                    and self.min[1] <= other.max[1]
                    and self.max[1] >= other.min[1]
                    and self.min[2] <= other.max[2]
                    and self.max[2] >= other.min[2]
                )

        self.ti_aabb = ti_aabb

        self.aabbs = ti_aabb.field(
            shape=(n_batches, n_aabbs),
            needs_grad=False,
            layout=ti.Layout.SOA,
        )


@ti.data_oriented
class LBVH(RBC):
    """
    Linear BVH is a simple BVH that is used to accelerate collision detection. It supports parallel building and
    querying of the BVH tree. Only supports axis-aligned bounding boxes (AABBs).

    Attributes
    -----
        aabbs : ti.field
        The input AABBs to be organized in the BVH, shape (n_batches, n_aabbs).
        n_aabbs : int
            Number of AABBs per batch.
        n_batches : int
            Number of batches.
        max_n_query_results : int
            Maximum number of query results allowed.
        max_stack_depth : int
            Maximum stack depth for BVH traversal.
        aabb_centers : ti.field
            Centers of the AABBs, shape (n_batches, n_aabbs).
        aabb_min : ti.field
            Minimum coordinates of AABB centers per batch, shape (n_batches).
        aabb_max : ti.field
            Maximum coordinates of AABB centers per batch, shape (n_batches).
        scale : ti.field
            Scaling factors for normalizing AABB centers, shape (n_batches).
        morton_codes : ti.field
            Morton codes for each AABB, shape (n_batches, n_aabbs).
        hist : ti.field
            Histogram for radix sort, shape (n_batches, 256).
        prefix_sum : ti.field
            Prefix sum for histogram, shape (n_batches, 256).
        offset : ti.field
            Offset for radix sort, shape (n_batches, n_aabbs).
        tmp_morton_codes : ti.field
            Temporary storage for radix sort, shape (n_batches, n_aabbs).
        Node : ti.dataclass
            Node structure for the BVH tree, containing left, right, parent indices and bounding box.
        nodes : ti.field
            BVH nodes, shape (n_batches, n_aabbs * 2 - 1).
        internal_node_visited : ti.field
            Flags indicating if an internal node has been visited during traversal, shape (n_batches, n_aabbs - 1).
        query_result : ti.field
            Query results as a vector of (batch id, self id, query id), shape (max_n_query_results).
        query_result_count : ti.field
            Counter for the number of query results.

    Notes
    ------
        For algorithmic details, see:
        https://research.nvidia.com/sites/default/files/pubs/2012-06_Maximizing-Parallelism-in/karras2012hpg_paper.pdf
    """

    def __init__(self, aabb: AABB, max_n_query_result_per_aabb: int = 8):
        self.aabbs = aabb.aabbs
        self.n_aabbs = aabb.n_aabbs
        self.n_batches = aabb.n_batches

        # Maximum number of query results
        self.max_n_query_results = min(self.n_aabbs * max_n_query_result_per_aabb * self.n_batches, 0x7FFFFFFF)
        # Maximum stack depth for traversal
        self.max_stack_depth = 64
        self.aabb_centers = ti.field(gs.ti_vec3, shape=(self.n_batches, self.n_aabbs))
        self.aabb_min = ti.field(gs.ti_vec3, shape=(self.n_batches))
        self.aabb_max = ti.field(gs.ti_vec3, shape=(self.n_batches))
        self.scale = ti.field(gs.ti_vec3, shape=(self.n_batches))
        self.morton_codes = ti.field(ti.u64, shape=(self.n_batches, self.n_aabbs))

        # Histogram for radix sort
        self.hist = ti.field(ti.u32, shape=(self.n_batches, 256))
        # Prefix sum for histogram
        self.prefix_sum = ti.field(ti.u32, shape=(self.n_batches, 256))
        # Offset for radix sort
        self.offset = ti.field(ti.u32, shape=(self.n_batches, self.n_aabbs))
        # Temporary storage for radix sort
        self.tmp_morton_codes = ti.field(ti.u64, shape=(self.n_batches, self.n_aabbs))

        @ti.dataclass
        class Node:
            """
            Node structure for the BVH tree.

            Attributes:
                left (int): Index of the left child node.
                right (int): Index of the right child node.
                parent (int): Index of the parent node.
                bound (ti_aabb): Bounding box of the node, represented as an AABB.
            """

            left: ti.i32
            right: ti.i32
            parent: ti.i32
            bound: aabb.ti_aabb

        self.Node = Node

        # Nodes of the BVH, first n_aabbs - 1 are internal nodes, last n_aabbs are leaf nodes
        self.nodes = self.Node.field(shape=(self.n_batches, self.n_aabbs * 2 - 1))
        # Whether an internal node has been visited during traversal
        self.internal_node_active = ti.field(ti.u1, shape=(self.n_batches, self.n_aabbs - 1))
        self.internal_node_ready = ti.field(ti.u1, shape=(self.n_batches, self.n_aabbs - 1))
        self.updated = ti.field(ti.u1, shape=())

        # Query results, vec3 of batch id, self id, query id
        self.query_result = ti.field(gs.ti_ivec3, shape=(self.max_n_query_results))
        # Count of query results
        self.query_result_count = ti.field(ti.i32, shape=())

    def build(self):
        """
        Build the BVH from the axis-aligned bounding boxes (AABBs).
        """
        self.compute_aabb_centers_and_scales()
        self.compute_morton_codes()
        self.radix_sort_morton_codes()
        self.build_radix_tree()
        self.compute_bounds()

    @ti.kernel
    def compute_aabb_centers_and_scales(self):
        for i_b, i_a in ti.ndrange(self.n_batches, self.n_aabbs):
            self.aabb_centers[i_b, i_a] = (self.aabbs[i_b, i_a].min + self.aabbs[i_b, i_a].max) / 2

        for i_b in ti.ndrange(self.n_batches):
            self.aabb_min[i_b] = self.aabb_centers[i_b, 0]
            self.aabb_max[i_b] = self.aabb_centers[i_b, 0]

        for i_b, i_a in ti.ndrange(self.n_batches, self.n_aabbs):
            ti.atomic_min(self.aabb_min[i_b], self.aabbs[i_b, i_a].min)
            ti.atomic_max(self.aabb_max[i_b], self.aabbs[i_b, i_a].max)

        for i_b in ti.ndrange(self.n_batches):
            scale = self.aabb_max[i_b] - self.aabb_min[i_b]
            for i in ti.static(range(3)):
                self.scale[i_b][i] = ti.select(scale[i] > gs.EPS, 1.0 / scale[i], 1.0)

    @ti.kernel
    def compute_morton_codes(self):
        """
        Compute the Morton codes for each AABB.

        The first 32 bits is the Morton code for the x, y, z coordinates, and the last 32 bits is the index of the AABB
        in the original array. The x, y, z coordinates are scaled to a 10-bit integer range [0, 1024) and interleaved to
        form the Morton code.
        """
        for i_b, i_a in ti.ndrange(self.n_batches, self.n_aabbs):
            center = self.aabb_centers[i_b, i_a] - self.aabb_min[i_b]
            scaled_center = center * self.scale[i_b]
            morton_code_x = ti.floor(scaled_center[0] * 1023.0, dtype=ti.u32)
            morton_code_y = ti.floor(scaled_center[1] * 1023.0, dtype=ti.u32)
            morton_code_z = ti.floor(scaled_center[2] * 1023.0, dtype=ti.u32)
            morton_code_x = self.expand_bits(morton_code_x)
            morton_code_y = self.expand_bits(morton_code_y)
            morton_code_z = self.expand_bits(morton_code_z)
            morton_code = (morton_code_x << 2) | (morton_code_y << 1) | (morton_code_z)
            self.morton_codes[i_b, i_a] = (ti.u64(morton_code) << 32) | ti.u64(i_a)

    @ti.func
    def expand_bits(self, v):
        """
        Expands a 10-bit integer into 30 bits by inserting 2 zeros before each bit.
        """
        v = (v * ti.u32(0x00010001)) & ti.u32(0xFF0000FF)
        v = (v * ti.u32(0x00000101)) & ti.u32(0x0F00F00F)
        v = (v * ti.u32(0x00000011)) & ti.u32(0xC30C30C3)
        v = (v * ti.u32(0x00000005)) & ti.u32(0x49249249)
        return v

    def radix_sort_morton_codes(self):
        """
        Radix sort the morton codes, using 8 bits at a time.
        """
        for i in range(8):
            self._kernel_radix_sort_morton_codes_one_round(i)

    @ti.kernel
    def _kernel_radix_sort_morton_codes_one_round(self, i: int):
        # Clear histogram
        self.hist.fill(0)

        # Fill histogram
        for i_b in range(self.n_batches):
            # This is now sequential
            # TODO Parallelize, need to use groups to handle data to remain stable, could be not worth it
            for i_a in range(self.n_aabbs):
                code = (self.morton_codes[i_b, i_a] >> (i * 8)) & 0xFF
                self.offset[i_b, i_a] = ti.atomic_add(self.hist[i_b, ti.i32(code)], 1)

        # Compute prefix sum
        for i_b in ti.ndrange(self.n_batches):
            self.prefix_sum[i_b, 0] = 0
            for j in range(1, 256):  # sequential prefix sum
                self.prefix_sum[i_b, j] = self.prefix_sum[i_b, j - 1] + self.hist[i_b, j - 1]

        # Reorder morton codes
        for i_b, i_a in ti.ndrange(self.n_batches, self.n_aabbs):
            code = (self.morton_codes[i_b, i_a] >> (i * 8)) & 0xFF
            idx = ti.i32(self.offset[i_b, i_a] + self.prefix_sum[i_b, ti.i32(code)])
            self.tmp_morton_codes[i_b, idx] = self.morton_codes[i_b, i_a]

        # Swap the temporary and original morton codes
        for i_b, i_a in ti.ndrange(self.n_batches, self.n_aabbs):
            self.morton_codes[i_b, i_a] = self.tmp_morton_codes[i_b, i_a]

    @ti.kernel
    def build_radix_tree(self):
        """
        Build the radix tree from the sorted morton codes.

        The tree is built in parallel for every internal node.
        """
        # Initialize the first node
        for i_b in ti.ndrange(self.n_batches):
            self.nodes[i_b, 0].parent = -1

        # Initialize the leaf nodes
        for i_b, i in ti.ndrange(self.n_batches, self.n_aabbs):
            self.nodes[i_b, i + self.n_aabbs - 1].left = -1
            self.nodes[i_b, i + self.n_aabbs - 1].right = -1

        # Parallel build for every internal node
        for i_b, i in ti.ndrange(self.n_batches, self.n_aabbs - 1):
            d = ti.select(
                self.delta(i, i + 1, i_b) > self.delta(i, i - 1, i_b),
                1,
                -1,
            )

            delta_min = self.delta(i, i - d, i_b)
            l_max = ti.u32(2)
            while self.delta(i, i + l_max * d, i_b) > delta_min:
                l_max *= 2
            l = ti.u32(0)

            t = l_max // 2
            while t > 0:
                if self.delta(i, i + (l + t) * d, i_b) > delta_min:
                    l += t
                t //= 2
            j = i + l * d
            delta_node = self.delta(i, j, i_b)
            s = ti.u32(0)
            t = (l + 1) // 2
            while t > 0:
                if self.delta(i, i + (s + t) * d, i_b) > delta_node:
                    s += t
                t = ti.select(t > 1, (t + 1) // 2, 0)

            gamma = i + ti.i32(s) * d + ti.min(d, 0)
            left = ti.select(ti.min(i, j) == gamma, gamma + self.n_aabbs - 1, gamma)
            right = ti.select(ti.max(i, j) == gamma + 1, gamma + self.n_aabbs, gamma + 1)
            self.nodes[i_b, i].left = ti.i32(left)
            self.nodes[i_b, i].right = ti.i32(right)
            self.nodes[i_b, ti.i32(left)].parent = i
            self.nodes[i_b, ti.i32(right)].parent = i

    @ti.func
    def delta(self, i, j, i_b):
        """
        Compute the longest common prefix (LCP) of the morton codes of two AABBs.
        """
        result = -1
        if j >= 0 and j < self.n_aabbs:
            result = 64
            x = self.morton_codes[i_b, ti.i32(i)] ^ self.morton_codes[i_b, ti.i32(j)]
            for b in range(64):
                if x & (ti.u64(1) << (63 - b)):
                    result = b
                    break
        return result

    def compute_bounds(self):
        """
        Compute the bounds of the BVH nodes.

        Starts from the leaf nodes and works upwards layer by layer.
        """
        self._kernel_compute_bounds_init()
        while self.updated[None]:
            self._kernel_compute_bounds_one_layer()

    @ti.kernel
    def _kernel_compute_bounds_init(self):
        self.updated[None] = True
        self.internal_node_active.fill(0)
        self.internal_node_ready.fill(0)

        for i_b, i in ti.ndrange(self.n_batches, self.n_aabbs):
            idx = ti.i32(self.morton_codes[i_b, i])
            self.nodes[i_b, i + self.n_aabbs - 1].bound.min = self.aabbs[i_b, idx].min
            self.nodes[i_b, i + self.n_aabbs - 1].bound.max = self.aabbs[i_b, idx].max
            parent_idx = self.nodes[i_b, i + self.n_aabbs - 1].parent
            if parent_idx != -1:
                self.internal_node_active[i_b, parent_idx] = 1

    @ti.kernel
    def _kernel_compute_bounds_one_layer(self):
        self.updated[None] = False
        for i_b, i in ti.ndrange(self.n_batches, self.n_aabbs - 1):
            if self.internal_node_active[i_b, i] == 0:
                continue
            left_bound = self.nodes[i_b, self.nodes[i_b, i].left].bound
            right_bound = self.nodes[i_b, self.nodes[i_b, i].right].bound
            self.nodes[i_b, i].bound.min = ti.min(left_bound.min, right_bound.min)
            self.nodes[i_b, i].bound.max = ti.max(left_bound.max, right_bound.max)
            parent_idx = self.nodes[i_b, i].parent
            if parent_idx != -1:
                self.internal_node_ready[i_b, parent_idx] = 1
            self.internal_node_active[i_b, i] = 0
            self.updated[None] = True

        for i_b, i in ti.ndrange(self.n_batches, self.n_aabbs - 1):
            if self.internal_node_ready[i_b, i] == 0:
                continue
            self.internal_node_active[i_b, i] = 1
            self.internal_node_ready[i_b, i] = 0

    @ti.kernel
    def query(self, aabbs: ti.template()):
        """
        Query the BVH for intersections with the given AABBs.

        The results are stored in the query_result field.
        """
        self.query_result_count[None] = 0

        n_querys = aabbs.shape[1]
        for i_b, i_q in ti.ndrange(self.n_batches, n_querys):
            query_stack = ti.Vector.zero(ti.i32, 64)
            stack_depth = 1

            while stack_depth > 0:
                stack_depth -= 1
                node_idx = query_stack[stack_depth]
                node = self.nodes[i_b, node_idx]
                # Check if the AABB intersects with the node's bounding box
                if aabbs[i_b, i_q].intersects(node.bound):
                    # If it's a leaf node, add the AABB index to the query results
                    if node.left == -1 and node.right == -1:
                        idx = ti.atomic_add(self.query_result_count[None], 1)
                        if idx < self.max_n_query_results:
                            code = self.morton_codes[i_b, node_idx - (self.n_aabbs - 1)]
                            self.query_result[idx] = gs.ti_ivec3(
                                i_b, ti.i32(code & ti.u64(0xFFFFFFFF)), i_q
                            )  # Store the AABB index
                    else:
                        # Push children onto the stack
                        if node.right != -1:
                            query_stack[stack_depth] = node.right
                            stack_depth += 1
                        if node.left != -1:
                            query_stack[stack_depth] = node.left
                            stack_depth += 1
