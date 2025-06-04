import genesis as gs
import taichi as ti
from genesis.repr_base import RBC


@ti.dataclass
class AABB:
    min: gs.ti_vec3
    max: gs.ti_vec3


@ti.data_oriented
class LBVH(RBC):
    """
    A bounding volume hierarchy (BVH) is a data structure that allows for efficient spatial partitioning of objects in a scene.
    It is used to accelerate collision detection and ray tracing. Linear BVH is a simple BVH that is used to accelerate collision detection and ray tracing using parallelization.
    """

    @ti.dataclass
    class Node:
        left: ti.i32  # Index of the left child
        right: ti.i32  # Index of the right child
        parent: ti.i32  # Index of the parent node
        bound: AABB  # Bounding box of the node

    def __init__(self, n_aabbs, n_batches):
        self.n_aabbs = n_aabbs
        self.n_batches = n_batches
        self.aabb_centers = ti.field(gs.ti_vec3, shape=(n_aabbs, n_batches))
        self.aabb_min = ti.field(gs.ti_vec3, shape=(n_batches))
        self.aabb_max = ti.field(gs.ti_vec3, shape=(n_batches))
        self.scale = ti.field(gs.ti_vec3, shape=(n_batches))
        self.morton_codes = ti.field(ti.u64, shape=(n_aabbs, n_batches))

        self.hist = ti.field(ti.u32, shape=(256, n_batches))  # Histogram for radix sort
        self.prefix_sum = ti.field(ti.u32, shape=(256, n_batches))  # Prefix sum for histogram
        self.offset = ti.field(ti.u32, shape=(n_aabbs, n_batches))  # Offset for radix sort
        self.tmp_morton_codes = ti.field(ti.u64, shape=(n_aabbs, n_batches))  # Temporary storage for radix sort

        self.nodes = self.Node.field(
            shape=(n_aabbs * 2 - 1, n_batches)
        )  # Nodes of the BVH, first n_aabbs - 1 are internal nodes, last n_aabbs are leaf nodes
        self.internal_node_visited = ti.field(
            ti.u8, shape=(n_aabbs - 1, n_batches)
        )  # If an internal node has been visited during traversal

    @ti.kernel
    def build(self, aabbs: ti.template()):
        """
        Build the BVH from the given axis-aligned bounding boxes (AABBs).
        The AABBs are expected to be in the format of a 2D array with shape (n_aabbs, n_batches),
        where n_aabbs is the number of AABBs and n_batches is the number of batches.
        Each AABB is represented by its minimum and maximum corners.

        Notes
        ------
        https://research.nvidia.com/sites/default/files/pubs/2012-06_Maximizing-Parallelism-in/karras2012hpg_paper.pdf
        """
        assert aabbs.shape[0] == self.n_aabbs
        assert aabbs.shape[1] == self.n_batches

        for i_a, i_b in ti.ndrange(self.n_aabbs, self.n_batches):
            self.aabb_centers[i_a, i_b] = (aabbs[i_a, i_b].min + aabbs[i_a, i_b].max) / 2

        for i_b in ti.ndrange(self.n_batches):
            self.aabb_min[i_b] = self.aabb_centers[0, i_b]
            self.aabb_max[i_b] = self.aabb_centers[0, i_b]

        for i_a, i_b in ti.ndrange(self.n_aabbs, self.n_batches):
            ti.atomic_min(self.aabb_min[i_b], aabbs[i_a, i_b].min)
            ti.atomic_max(self.aabb_max[i_b], aabbs[i_a, i_b].max)

        for i_b in ti.ndrange(self.n_batches):
            scale = self.aabb_max[i_b] - self.aabb_min[i_b]
            for i in ti.static(range(3)):
                self.scale[i_b][i] = ti.select(scale[i] > 1e-7, 1.0 / scale[i], 1)

        self.compute_morton_codes()
        self.radix_sort_morton_codes()
        self.build_radix_tree()
        self.compute_bounds(aabbs)

    @ti.func
    def compute_morton_codes(self):
        for i_a, i_b in ti.ndrange(self.n_aabbs, self.n_batches):
            center = self.aabb_centers[i_a, i_b] - self.aabb_min[i_b]
            scaled_center = center * self.scale[i_b]
            morton_code_x = ti.floor(scaled_center[0] * 1024.0, dtype=ti.u32)
            morton_code_y = ti.floor(scaled_center[1] * 1024.0, dtype=ti.u32)
            morton_code_z = ti.floor(scaled_center[2] * 1024.0, dtype=ti.u32)
            morton_code_x = self.expand_bits(morton_code_x)
            morton_code_y = self.expand_bits(morton_code_y)
            morton_code_z = self.expand_bits(morton_code_z)
            morton_code = (morton_code_x << 2) | (morton_code_y << 1) | (morton_code_z)
            self.morton_codes[i_a, i_b] = (ti.u64(morton_code) << 32) | ti.u64(i_a)

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
            for j, i_b in ti.ndrange(256, self.n_batches):
                self.hist[j, i_b] = 0
            # Fill histogram
            for i_a, i_b in ti.ndrange(self.n_aabbs, self.n_batches):
                code = (self.morton_codes[i_a, i_b] >> (i * 8)) & 0xFF
                self.offset[i_a, i_b] = ti.atomic_add(self.hist[ti.i32(code), i_b], 1)
            # Compute prefix sum
            for i_b in ti.ndrange(self.n_batches):
                self.prefix_sum[0, i_b] = 0
                for j in range(1, 256):
                    self.prefix_sum[j, i_b] = self.prefix_sum[j - 1, i_b] + self.hist[j - 1, i_b]
            # Reorder morton codes
            for i_a, i_b in ti.ndrange(self.n_aabbs, self.n_batches):
                code = (self.morton_codes[i_a, i_b] >> (i * 8)) & 0xFF
                idx = ti.i32(self.offset[i_a, i_b] + self.prefix_sum[ti.i32(code), i_b])
                self.tmp_morton_codes[idx, i_b] = self.morton_codes[i_a, i_b]

            # Swap the temporary and original morton codes
            for i_a, i_b in ti.ndrange(self.n_aabbs, self.n_batches):
                self.morton_codes[i_a, i_b] = self.tmp_morton_codes[i_a, i_b]

    @ti.func
    def build_radix_tree(self):
        # Initialize the first node
        for i_b in ti.ndrange(self.n_batches):
            self.nodes[0, i_b].parent = -1

        # Initialize the leaf nodes
        for i, i_b in ti.ndrange(self.n_aabbs, self.n_batches):
            self.nodes[i + self.n_aabbs - 1, i_b].left = -1
            self.nodes[i + self.n_aabbs - 1, i_b].right = -1

        for i, i_b in ti.ndrange(self.n_aabbs - 1, self.n_batches):
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
            self.nodes[i, i_b].left = ti.i32(left)
            self.nodes[i, i_b].right = ti.i32(right)
            self.nodes[ti.i32(left), i_b].parent = i
            self.nodes[ti.i32(right), i_b].parent = i

    @ti.func
    def delta(self, i, j, i_b):
        """
        Compute the longest common prefix (LCP) of the morton codes of two AABBs.
        """
        result = -1
        if j >= 0 and j < self.n_aabbs:
            result = 64
            x = self.morton_codes[ti.i32(i), i_b] ^ self.morton_codes[ti.i32(j), i_b]
            for b in range(64):
                if x & (ti.u64(1) << (63 - b)):
                    result = b
                    break
        return result

    @ti.func
    def compute_bounds(self, aabbs: ti.template()):
        """
        Compute the bounds of the BVH nodes. Starts from the leaf nodes and works upwards.
        """
        for i, i_b in ti.ndrange(self.n_aabbs - 1, self.n_batches):
            self.internal_node_visited[i, i_b] = ti.u8(0)

        for i, i_b in ti.ndrange(self.n_aabbs, self.n_batches):
            idx = ti.i32(self.morton_codes[i, i_b])
            self.nodes[i + self.n_aabbs - 1, i_b].bound.min = aabbs[idx, i_b].min
            self.nodes[i + self.n_aabbs - 1, i_b].bound.max = aabbs[idx, i_b].max

            cur_idx = self.nodes[i + self.n_aabbs - 1, i_b].parent
            while cur_idx != -1:
                visited = ti.u1(ti.atomic_or(self.internal_node_visited[cur_idx, i_b], ti.u8(1)))
                if not visited:
                    break
                left_bound = self.nodes[self.nodes[cur_idx, i_b].left, i_b].bound
                right_bound = self.nodes[self.nodes[cur_idx, i_b].right, i_b].bound
                self.nodes[cur_idx, i_b].bound.min = ti.min(left_bound.min, right_bound.min)
                self.nodes[cur_idx, i_b].bound.max = ti.max(left_bound.max, right_bound.max)
                cur_idx = self.nodes[cur_idx, i_b].parent
