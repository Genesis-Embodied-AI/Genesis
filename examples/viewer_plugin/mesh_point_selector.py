import csv
import os
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
from typing_extensions import override

import genesis as gs
import genesis.utils.geom as gu
import genesis.vis.keybindings as kb
from genesis.utils.misc import tensor_to_array
from genesis.vis.viewer_plugins import EVENT_HANDLE_STATE, EVENT_HANDLED, RaycasterViewerPlugin

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


class MeshPointSelectorPlugin(RaycasterViewerPlugin):
    """
    Interactive viewer plugin that enables using mouse clicks to select points on rigid meshes.
    Selected points are stored in local coordinates relative to their link's frame.
    """

    def __init__(
        self,
        sphere_radius: float = 0.005,
        sphere_color: tuple = (0.1, 0.3, 1.0, 1.0),
        hover_color: tuple = (0.3, 0.5, 1.0, 1.0),
        grid_snap: tuple[float, float, float] = (-1.0, -1.0, -1.0),
        output_file: str = "selected_points.csv",
    ) -> None:
        super().__init__()
        self.sphere_radius = sphere_radius
        self.sphere_color = sphere_color
        self.hover_color = hover_color
        self.grid_snap = grid_snap
        self.output_file = output_file

        self.selected_points: dict[int, SelectedPoint] = {}
        self._prev_mouse_pos: tuple[int, int] = (0, 0)

    def build(self, viewer, camera: "Node", scene: "Scene"):
        super().build(viewer, camera, scene)
        self._prev_mouse_pos: tuple[int, int] = (self.viewer._viewport_size[0] // 2, self.viewer._viewport_size[1] // 2)

    def _get_pos_hash(self, pos: np.ndarray) -> int:
        """
        Generate a hash for a given position to use as a unique identifier.

        Parameters
        ----------
        pos : np.ndarray, shape (3,)
            The position to hash.

        Returns
        -------
        int
            The hash of the position.
        """
        return hash((round(pos[0], 6), round(pos[1], 6), round(pos[2], 6)))

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
        grid_snap = np.array(self.grid_snap)
        # Snap each axis if the snap value is non-negative
        return np.where(grid_snap >= 0, np.round(point / grid_snap) * grid_snap, point)

    @override
    def on_mouse_motion(self, x: int, y: int, dx: int, dy: int) -> EVENT_HANDLE_STATE:
        self._prev_mouse_pos = (x, y)

    @override
    def on_mouse_press(self, x: int, y: int, button: int, modifiers: int) -> EVENT_HANDLE_STATE:
        if button == 1:  # left click
            ray = self._screen_position_to_ray(x, y)
            ray_hit = self._raycaster.cast(*ray)

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

                pos_hash = self._get_pos_hash(local_pos)
                if pos_hash in self.selected_points:
                    # Deselect point if already selected
                    del self.selected_points[pos_hash]
                else:
                    selected_point = SelectedPoint(link, local_pos, local_normal)
                    self.selected_points[pos_hash] = selected_point

                return EVENT_HANDLED
        return None

    @override
    def on_draw(self) -> None:
        super().on_draw()
        if self.scene._visualizer is not None and self.scene._visualizer.is_built:
            self.scene.clear_debug_objects()
            mouse_ray = self._screen_position_to_ray(*self._prev_mouse_pos)

            closest_hit = self._raycaster.cast(*mouse_ray)
            if closest_hit is not None:
                snap_pos = self._snap_to_grid(closest_hit.position)

                # Draw hover preview
                self.scene.draw_debug_sphere(
                    snap_pos,
                    self.sphere_radius,
                    self.hover_color,
                )
                self.scene.draw_debug_arrow(
                    snap_pos,
                    tuple(n * 0.05 for n in closest_hit.normal),
                    self.sphere_radius / 2,
                    self.hover_color,
                )

            if self.selected_points:
                world_positions = []
                for point in self.selected_points.values():
                    link_pos = tensor_to_array(point.link.get_pos())
                    link_quat = tensor_to_array(point.link.get_quat())
                    local_pos_arr = np.array(point.local_position, dtype=np.float32)
                    current_world_pos = gu.transform_by_trans_quat(local_pos_arr, link_pos, link_quat)
                    world_positions.append(current_world_pos)

                if len(world_positions) == 1:
                    self.scene.draw_debug_sphere(
                        world_positions[0],
                        self.sphere_radius,
                        self.sphere_color,
                    )
                else:
                    positions_array = np.array(world_positions)
                    self.scene.draw_debug_spheres(positions_array, self.sphere_radius, self.sphere_color)

    @override
    def on_close(self) -> None:
        super().on_close()

        if not self.selected_points:
            print("[MeshPointSelectorPlugin] No points selected.")
            return

        output_file = self.output_file
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

                for i, point in enumerate(self.selected_points.values(), 1):
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


if __name__ == "__main__":
    gs.init(backend=gs.gpu)

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            gravity=(0.0, 0.0, 0.0),
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.6, 0.6, 0.6),
            camera_lookat=(0.0, 0.0, 0.2),
            camera_fov=40,
        ),
        vis_options=gs.options.VisOptions(
            show_world_frame=True,
        ),
        profiling_options=gs.options.ProfilingOptions(
            show_FPS=False,
        ),
        show_viewer=True,
    )

    hand = scene.add_entity(
        morph=gs.morphs.URDF(
            file="urdf/shadow_hand/shadow_hand.urdf",
            collision=True,
            pos=(0.0, 0.0, 0.0),
            euler=(0.0, 0.0, 180.0),
            fixed=True,
            merge_fixed_links=False,
        ),
    )

    scene.viewer.add_plugin(
        MeshPointSelectorPlugin(
            sphere_radius=0.004,
            grid_snap=(-1.0, 0.01, 0.01),
            output_file="selected_points.csv",
        )
    )

    scene.build()

    is_running = True

    def stop():
        global is_running
        is_running = False

    scene.viewer.register_keybinds(
        kb.Keybind("quit", kb.Key.ESCAPE, kb.KeyAction.PRESS, callback=stop),
    )

    try:
        while is_running:
            scene.step()

            if "PYTEST_VERSION" in os.environ:
                break
    except KeyboardInterrupt:
        gs.logger.info("Simulation interrupted, exiting.")
    finally:
        gs.logger.info("Simulation finished.")
