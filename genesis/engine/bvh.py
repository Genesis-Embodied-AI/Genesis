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

    def __init__(self, n_aabbs, n_batches):
        self.n_aabbs = n_aabbs
        self.n_batches = n_batches
        self.aabb_centers = ti.field(gs.ti_vec3, shape=(n_aabbs, n_batches))
        self.aabb_min = ti.field(gs.ti_vec3, shape=(n_batches))
        self.aabb_max = ti.field(gs.ti_vec3, shape=(n_batches))
        self.scale = ti.field(gs.ti_vec3, shape=(n_batches))
        self.morton_codes = ti.field(gs.ti_u32, shape=(n_aabbs, n_batches))

    @ti.kernel
    def build(self, aabbs):
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

    @ti.kernel
    def test(self):
        for i_b in ti.ndrange(self.n_batches):
            print(self.scale[i_b])
