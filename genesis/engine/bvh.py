import genesis as gs
import taichi as ti
from genesis.repr_base import RBC


@ti.data_oriented
class AABB(RBC):

    def __init__(self, n_aabbs, n_batches):
        self.n_aabbs = n_aabbs
        self.n_batches = n_batches

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
    A bounding volume hierarchy (BVH) is a data structure that allows for efficient spatial partitioning of objects in a scene.
    It is used to accelerate collision detection and ray tracing. Linear BVH is a simple BVH that is used to accelerate collision detection and ray tracing using parallelization.
    """

    def __init__(self, aabb: AABB, max_n_query_result_per_aabb: int = 8):
        self.aabbs = aabb.aabbs
        self.n_aabbs = aabb.n_aabbs
        self.n_batches = aabb.n_batches
        self.max_n_query_results = (
            self.n_aabbs * max_n_query_result_per_aabb
        )  # Maximum number of query results per batch
        self.max_stack_depth = 64  # Maximum stack depth for traversal
        self.aabb_centers = ti.field(gs.ti_vec3, shape=(self.n_batches, self.n_aabbs))
        self.aabb_min = ti.field(gs.ti_vec3, shape=(self.n_batches))
        self.aabb_max = ti.field(gs.ti_vec3, shape=(self.n_batches))
        self.scale = ti.field(gs.ti_vec3, shape=(self.n_batches))
        self.morton_codes = ti.field(ti.u64, shape=(self.n_batches, self.n_aabbs))

        self.hist = ti.field(ti.u32, shape=(self.n_batches, 256))  # Histogram for radix sort
        self.prefix_sum = ti.field(ti.u32, shape=(self.n_batches, 256))  # Prefix sum for histogram
        self.offset = ti.field(ti.u32, shape=(self.n_batches, self.n_aabbs))  # Offset for radix sort
        self.tmp_morton_codes = ti.field(
            ti.u64, shape=(self.n_batches, self.n_aabbs)
        )  # Temporary storage for radix sort

        @ti.dataclass
        class Node:
            left: ti.i32  # Index of the left child
            right: ti.i32  # Index of the right child
            parent: ti.i32  # Index of the parent node
            bound: aabb.ti_aabb  # Bounding box of the node

        self.Node = Node

        self.nodes = self.Node.field(
            shape=(self.n_batches, self.n_aabbs * 2 - 1)
        )  # Nodes of the BVH, first n_aabbs - 1 are internal nodes, last n_aabbs are leaf nodes
        self.internal_node_visited = ti.field(
            ti.u8, shape=(self.n_batches, self.n_aabbs - 1)
        )  # If an internal node has been visited during traversal

        self.query_result = ti.field(
            gs.ti_ivec2, shape=(self.n_batches, self.max_n_query_results)
        )  # Query results, vec2 first is self id, second is query id
        self.query_result_count = ti.field(ti.i32, shape=(self.n_batches))  # Count of query results per batch

    @ti.kernel
    def build(self):
        """
        Build the BVH from the given axis-aligned bounding boxes (AABBs).
        The AABBs are expected to be in the format of a 2D array with shape (n_aabbs, n_batches),
        where n_aabbs is the number of AABBs and n_batches is the number of batches.
        Each AABB is represented by its minimum and maximum corners.

        Notes
        ------
        https://research.nvidia.com/sites/default/files/pubs/2012-06_Maximizing-Parallelism-in/karras2012hpg_paper.pdf
        """

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
                self.scale[i_b][i] = ti.select(scale[i] > 1e-7, 1.0 / scale[i], 1)

        self.compute_morton_codes()
        self.radix_sort_morton_codes()
        self.build_radix_tree()
        self.compute_bounds()

    @ti.func
    def compute_morton_codes(self):
        for i_b, i_a in ti.ndrange(self.n_batches, self.n_aabbs):
            center = self.aabb_centers[i_b, i_a] - self.aabb_min[i_b]
            scaled_center = center * self.scale[i_b]
            morton_code_x = ti.floor(scaled_center[0] * 1024.0, dtype=ti.u32)
            morton_code_y = ti.floor(scaled_center[1] * 1024.0, dtype=ti.u32)
            morton_code_z = ti.floor(scaled_center[2] * 1024.0, dtype=ti.u32)
            morton_code_x = self.expand_bits(morton_code_x)
            morton_code_y = self.expand_bits(morton_code_y)
            morton_code_z = self.expand_bits(morton_code_z)
            morton_code = (morton_code_x << 2) | (morton_code_y << 1) | (morton_code_z)
            self.morton_codes[i_b, i_a] = (ti.u64(morton_code) << 32) | ti.u64(i_a)

    # Expands a 10-bit integer into 30 bits by inserting 2 zeros after each bit.
    @ti.func
    def expand_bits(self, v):
        v = (v * ti.u32(0x00010001)) & ti.u32(0xFF0000FF)
        v = (v * ti.u32(0x00000101)) & ti.u32(0x0F00F00F)
        v = (v * ti.u32(0x00000011)) & ti.u32(0xC30C30C3)
        v = (v * ti.u32(0x00000005)) & ti.u32(0x49249249)
        return v

    # radix sort the morton codes, using 8 bits at a time
    @ti.func
    def radix_sort_morton_codes(self):
        for i in ti.static(range(8)):
            # Clear histogram
            for i_b, j in ti.ndrange(self.n_batches, 256):
                self.hist[i_b, j] = 0
            # Fill histogram
            for i_b, i_a in ti.ndrange(self.n_batches, self.n_aabbs):
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

    @ti.func
    def build_radix_tree(self):
        # Initialize the first node
        for i_b in ti.ndrange(self.n_batches):
            self.nodes[i_b, 0].parent = -1

        # Initialize the leaf nodes
        for i_b, i in ti.ndrange(self.n_batches, self.n_aabbs):
            self.nodes[i_b, i + self.n_aabbs - 1].left = -1
            self.nodes[i_b, i + self.n_aabbs - 1].right = -1
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

    @ti.func
    def compute_bounds(self):
        """
        Compute the bounds of the BVH nodes. Starts from the leaf nodes and works upwards.
        """
        for i_b, i in ti.ndrange(self.n_batches, self.n_aabbs - 1):
            self.internal_node_visited[i_b, i] = ti.u8(0)

        for i_b, i in ti.ndrange(self.n_batches, self.n_aabbs):
            idx = ti.i32(self.morton_codes[i_b, i])
            self.nodes[i_b, i + self.n_aabbs - 1].bound.min = self.aabbs[i_b, idx].min
            self.nodes[i_b, i + self.n_aabbs - 1].bound.max = self.aabbs[i_b, idx].max

            cur_idx = self.nodes[i_b, i + self.n_aabbs - 1].parent
            while cur_idx != -1:
                visited = ti.u1(ti.atomic_or(self.internal_node_visited[i_b, cur_idx], ti.u8(1)))
                if not visited:
                    break
                left_bound = self.nodes[i_b, self.nodes[i_b, cur_idx].left].bound
                right_bound = self.nodes[i_b, self.nodes[i_b, cur_idx].right].bound
                self.nodes[i_b, cur_idx].bound.min = ti.min(left_bound.min, right_bound.min)
                self.nodes[i_b, cur_idx].bound.max = ti.max(left_bound.max, right_bound.max)
                cur_idx = self.nodes[i_b, cur_idx].parent

    @ti.kernel
    def query(self, aabbs: ti.template()):
        """
        Query the BVH for intersections with the given AABBs.
        The results are stored in the query_result field.
        """
        for i_b in ti.ndrange(self.n_batches):
            self.query_result_count[i_b] = 0

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
                        idx = ti.atomic_add(self.query_result_count[i_b], 1)
                        if idx < self.max_n_query_results:
                            code = self.morton_codes[i_b, node_idx - (self.n_aabbs - 1)]
                            self.query_result[i_b, idx] = gs.ti_ivec2(
                                ti.i32(code & ti.u64(0xFFFFFFFF)), i_q
                            )  # Store the AABB index
                    else:
                        # Push children onto the stack
                        if node.right != -1:
                            query_stack[stack_depth] = node.right
                            stack_depth += 1
                        if node.left != -1:
                            query_stack[stack_depth] = node.left
                            stack_depth += 1
