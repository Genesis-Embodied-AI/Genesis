from typing import TYPE_CHECKING

import numpy as np
from typing_extensions import override

import genesis as gs
from genesis.utils.misc import qd_to_numpy, qd_to_torch
from genesis.utils.raycast import Ray, RayHit
from genesis.ext.pyrender.camera import OrthographicCamera

from .base import ViewerPlugin

if TYPE_CHECKING:
    from genesis.engine.scene import Scene
    from genesis.ext.pyrender.node import Node


class Raycaster:
    """
    BVH-accelerated single-ray cast for the viewer.

    The per-env raycast (`kernel_cast_ray`) writes one hit per env into a batched `RaycastResult`; this class then
    reduces across envs in torch to pick the closest hit. Cross-env reduction is intentionally a viewer-side concern,
    not part of the kernel, because parallel envs are otherwise meant to be isolated.

    After each `cast()`, `last_hit_env_idx` exposes the env that produced the returned `RayHit` (None when no hit).
    """

    def __init__(self, scene: "Scene"):
        # NOTE: delayed imports to avoid pulling in rigid_solver / array_class before gs is fully initialized.
        import genesis.utils.array_class as array_class
        from genesis.engine.bvh import AABB, LBVH

        self.scene = scene
        self.solver = scene.sim.rigid_solver
        self.envs_idx = scene._envs_idx
        self.last_hit_env_idx: int | None = None

        n_faces = self.solver.faces_info.geom_idx.shape[0]
        if n_faces == 0:
            gs.logger.warning("No faces found in scene, viewer raycasting will not work.")
            self.aabb = None
            self.bvh = None
            self.result = None
            return

        n_envs_max = len(self.envs_idx)
        self.aabb = AABB(n_batches=n_envs_max, n_aabbs=n_faces)
        self.bvh = LBVH(
            self.aabb,
            max_n_query_result_per_aabb=0,  # Not used for ray queries
            n_radix_sort_groups=min(64, n_faces),
        )
        self.result = array_class.get_raycast_result(n_envs_max)

        self.update()

        # Pre-compile to avoid a race condition with Quadrants on the first interactive cast.
        self.cast(ray_origin=np.zeros(3, dtype=gs.np_float), ray_direction=np.zeros(3, dtype=gs.np_float))

    def update(self) -> None:
        """Refresh per-env vertex positions, AABBs and rebuild the BVH."""
        if self.bvh is None:
            return
        from genesis.utils.raycast_qd import kernel_update_verts_and_aabbs

        kernel_update_verts_and_aabbs(
            geoms_info=self.solver.geoms_info,
            geoms_state=self.solver.geoms_state,
            verts_info=self.solver.verts_info,
            faces_info=self.solver.faces_info,
            free_verts_state=self.solver.free_verts_state,
            fixed_verts_state=self.solver.fixed_verts_state,
            static_rigid_sim_config=self.solver._static_rigid_sim_config,
            aabb_state=self.aabb,
        )
        self.bvh.build()

    def cast(
        self,
        ray_origin: np.ndarray,
        ray_direction: np.ndarray,
        max_range: float = 1000.0,
        envs_idx=None,
    ) -> RayHit | None:
        """
        Cast a single ray against the BVH of each env in parallel and return the closest hit across envs.

        Parameters
        ----------
        ray_origin : np.ndarray, shape (3,)
            Ray origin in world coordinates.
        ray_direction : np.ndarray, shape (3,)
            Normalized ray direction.
        max_range : float, optional
            Per-env BVH traversal max distance.
        envs_idx : optional
            Indices of envs to raycast. Defaults to all envs.
        """
        from genesis.utils.raycast_qd import kernel_cast_ray

        kernel_cast_ray(
            self.solver.fixed_verts_state,
            self.solver.free_verts_state,
            self.solver.verts_info,
            self.solver.faces_info,
            self.bvh.nodes,
            self.bvh.morton_codes,
            np.ascontiguousarray(ray_origin, dtype=gs.np_float),
            np.ascontiguousarray(ray_direction, dtype=gs.np_float),
            max_range,
            envs_idx if envs_idx is not None else self.envs_idx,
            self.solver._rigid_global_info,
            self.result,
            gs.EPS,
        )

        # Reduce per-env hits to the closest one. Distance is +inf for envs that didn't hit, so argmin alone
        # picks the closest hitting env; geom_idx then confirms the winner actually hit (argmin on all-inf
        # would otherwise return env 0).
        distances = qd_to_torch(self.result.distance, copy=None)
        winner = int(distances.argmin())
        geom_idx = int(qd_to_numpy(self.result.geom_idx, row_mask=winner, keepdim=False))
        if geom_idx < 0:
            self.last_hit_env_idx = None
            return None

        distance = float(distances[winner])
        position = qd_to_numpy(self.result.hit_point, row_mask=winner, keepdim=False, transpose=True)
        normal = qd_to_numpy(self.result.normal, row_mask=winner, keepdim=False, transpose=True)
        geom = self.solver.geoms[geom_idx] if 0 <= geom_idx < len(self.solver.geoms) else None

        self.last_hit_env_idx = winner if self.scene.n_envs > 0 else None
        return RayHit(distance, position, normal, geom)


class RaycasterViewerPlugin(ViewerPlugin):
    """
    Base viewer plugins using mouse raycast
    """

    def __init__(self) -> None:
        super().__init__()
        self._raycaster: "Raycaster | None" = None

    def build(self, viewer, camera: "Node", scene: "Scene"):
        super().build(viewer, camera, scene)

        self._raycaster = Raycaster(self.scene)

    @override
    def update_on_sim_step(self) -> None:
        super().update_on_sim_step()

        self._raycaster.update()

    def _screen_position_to_ray(self, x: float, y: float) -> Ray:
        """
        Converts 2D screen position to a ray.

        Parameters
        ----------
        x : float
            The x coordinate on the screen.
        y : float
            The y coordinate on the screen.

        Returns
        -------
        origin : np.ndarray, shape (3,)
            The origin of the ray in world coordinates.
        direction : np.ndarray, shape (3,)
            The direction of the ray in world coordinates.
        """

        viewport_size = self.viewer._viewport_size
        w_raw = float(viewport_size[0])
        h_raw = float(viewport_size[1])
        h = max(h_raw, 1e-8)
        x_c = float(x) - 0.5 * w_raw
        y_c = float(y) - 0.5 * h_raw
        sx = 2.0 * x_c / h
        sy = 2.0 * y_c / h

        # NOTE: ignoring pixel aspect ratio; projection may change after build (e.g. O key)
        mtx = self.camera.matrix
        position = mtx[:3, 3]
        forward = -mtx[:3, 2]
        right = mtx[:3, 0]
        up = mtx[:3, 1]

        cam = self.camera.camera
        if isinstance(cam, OrthographicCamera):
            ymag = float(cam.ymag)
            origin = position + right * (sx * ymag) + up * (sy * ymag)
            direction = forward / np.linalg.norm(forward)
            return Ray(origin, direction)

        tan_half = float(np.tan(0.5 * float(cam.yfov)))
        direction = forward + right * (sx * tan_half) + up * (sy * tan_half)
        direction /= np.linalg.norm(direction)
        return Ray(position, direction)
