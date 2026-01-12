import csv
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
from typing_extensions import override

import genesis as gs
import genesis.utils.geom as gu
from genesis.options.viewer_plugins import MeshPointSelectorPlugin as MeshPointSelectorPluginOptions
from genesis.utils.misc import tensor_to_array

from ..raycaster import Ray, ViewerRaycaster
from ..viewer_plugin import EVENT_HANDLE_STATE, EVENT_HANDLED, register_viewer_plugin
from .help_text import HelpTextPlugin

if TYPE_CHECKING:
    from genesis.engine.entities.rigid_entity import RigidLink
    from genesis.engine.scene import Scene
    from genesis.ext.pyrender.node import Node


class SelectedPoint(NamedTuple):
    """
    Represents a selected point on a rigid mesh surface.

    Attributes
    ----------
    link : RigidLink
        The rigid link that the point belongs to.
    local_position : np.ndarray, shape (3,)
        The position of the point in the link's local coordinate frame.
    local_normal : np.ndarray, shape (3,)
        The surface normal at the point in the link's local coordinate frame.
    """

    link: "RigidLink"
    local_position: np.ndarray  # shape (3,)
    local_normal: np.ndarray  # shape (3,)


@register_viewer_plugin(MeshPointSelectorPluginOptions)
class MeshPointSelectorPlugin(HelpTextPlugin):
    """
    Interactive viewer plugin that enables using mouse clicks to select points on rigid meshes.
    Selected points are stored in local coordinates relative to their link's frame.
    """

    def __init__(
        self,
        viewer,
        options: MeshPointSelectorPluginOptions,
        camera: "Node",
        scene: "Scene",
    ) -> None:
        super().__init__(viewer, options, camera, scene)

        self.prev_mouse_pos: tuple[int, int] = (self.viewer._viewport_size[0] // 2, self.viewer._viewport_size[1] // 2)
        self.selected_points: list[SelectedPoint] = []
        self.raycaster: ViewerRaycaster = ViewerRaycaster(self.scene)

    def _snap_to_grid(self, point: np.ndarray) -> np.ndarray:
        """
        Snap a point to the grid based on grid_snap settings.

        Parameters
        ----------
        point : np.ndarray, shape (3,)
            The point to snap.

        Returns
        -------
        np.ndarray, shape (3,)
            The point snapped to the grid.
        """
        grid_snap = np.array(self.options.grid_snap)
        # Snap each axis if the snap value is non-negative
        return np.where(grid_snap >= 0, np.round(point / grid_snap) * grid_snap, point)

    @override
    def on_mouse_motion(self, x: int, y: int, dx: int, dy: int) -> EVENT_HANDLE_STATE:
        super().on_mouse_motion(x, y, dx, dy)
        self.prev_mouse_pos = (x, y)
        return None

    @override
    def on_mouse_press(self, x: int, y: int, button: int, modifiers: int) -> EVENT_HANDLE_STATE:
        super().on_mouse_press(x, y, button, modifiers)
        if button == 1:  # left mouse button
            ray = self._screen_position_to_ray(x, y)
            ray_hit = self.raycaster.cast_ray(ray.origin, ray.direction)

            if ray_hit is not None and ray_hit.geom:
                link = ray_hit.geom.link
                world_pos = ray_hit.position
                world_normal = ray_hit.normal

                # Get link pose
                link_pos = tensor_to_array(link.get_pos())
                link_quat = tensor_to_array(link.get_quat())

                local_pos = gu.inv_transform_by_trans_quat(world_pos, link_pos, link_quat)
                local_normal = gu.inv_transform_by_quat(world_normal, link_quat)

                # Apply grid snapping to local position
                local_pos = self._snap_to_grid(local_pos)

                selected_point = SelectedPoint(link=link, local_position=local_pos, local_normal=local_normal)
                self.selected_points.append(selected_point)

                return EVENT_HANDLED
        return None

    @override
    def update_on_sim_step(self) -> None:
        self.raycaster.update_bvh()

    @override
    def on_draw(self) -> None:
        super().on_draw()
        if self.scene._visualizer is not None and self.scene._visualizer.is_built:
            self.scene.clear_debug_objects()
            mouse_ray: Ray = self._screen_position_to_ray(*self.prev_mouse_pos)

            closest_hit = self.raycaster.cast_ray(mouse_ray.origin, mouse_ray.direction)
            if closest_hit is not None:
                snap_pos = self._snap_to_grid(closest_hit.position)
                # Draw hover preview
                self.scene.draw_debug_sphere(
                    snap_pos,
                    self.options.sphere_radius,
                    self.options.hover_color,
                )
                self.scene.draw_debug_arrow(
                    snap_pos,
                    tuple(n * 0.1 for n in closest_hit.normal),
                    self.options.sphere_radius / 2,
                    self.options.hover_color,
                )

            if self.selected_points:
                world_positions = []
                for point in self.selected_points:
                    link_pos = tensor_to_array(point.link.get_pos())
                    link_quat = tensor_to_array(point.link.get_quat())
                    local_pos_arr = np.array(point.local_position, dtype=np.float32)
                    current_world_pos = gu.transform_by_trans_quat(local_pos_arr, link_pos, link_quat)
                    world_positions.append(current_world_pos)

                if len(world_positions) == 1:
                    self.scene.draw_debug_sphere(
                        world_positions[0],
                        self.options.sphere_radius,
                        self.options.sphere_color,
                    )
                else:
                    positions_array = np.array(world_positions)
                    self.scene.draw_debug_spheres(
                        positions_array, self.options.sphere_radius, self.options.sphere_color
                    )

    @override
    def on_close(self) -> None:
        super().on_close()

        if not self.selected_points:
            print("[MeshPointSelectorPlugin] No points selected.")
            return

        output_file = self.options.output_file
        try:
            with open(output_file, "w", newline="") as csvfile:
                writer = csv.writer(csvfile)

                writer.writerow(
                    [
                        "point_idx",
                        "link_idx",
                        "local_pos_x",
                        "local_pos_y",
                        "local_pos_z",
                        "local_normal_x",
                        "local_normal_y",
                        "local_normal_z",
                    ]
                )

                for i, point in enumerate(self.selected_points, 1):
                    writer.writerow(
                        [
                            i,
                            point.link.idx,
                            point.local_position[0],
                            point.local_position[1],
                            point.local_position[2],
                            point.local_normal[0],
                            point.local_normal[1],
                            point.local_normal[2],
                        ]
                    )

            gs.logger.info(
                f"[MeshPointSelectorPlugin] Wrote {len(self.selected_points)} selected points to '{output_file}'"
            )

        except Exception as e:
            gs.logger.error(f"[MeshPointSelectorPlugin] Error writing to '{output_file}': {e}")
