import csv
from typing import TYPE_CHECKING, NamedTuple

from typing_extensions import override

import genesis as gs
from genesis.options.viewer_interactions import MeshPointSelectorPlugin as MeshPointSelectorPluginOptions

from ..base_interaction import EVENT_HANDLE_STATE, EVENT_HANDLED, register_viewer_plugin
from ..utils import Pose, Ray, Vec3, ViewerRaycaster
from .viewer_controls import ViewerDefaultControls

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
    local_position : Vec3
        The position of the point in the link's local coordinate frame.
    local_normal : Vec3
        The surface normal at the point in the link's local coordinate frame.
    """
    link: "RigidLink"
    local_position: Vec3
    local_normal: Vec3



@register_viewer_plugin(MeshPointSelectorPluginOptions)
class MeshPointSelectorPlugin(ViewerDefaultControls):
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
        viewport_size: tuple[int, int],
    ) -> None:
        super().__init__(viewer, options, camera, scene, viewport_size)
        self.prev_mouse_pos: tuple[int, int] = (viewport_size[0] // 2, viewport_size[1] // 2)

        # List of selected points with link, local position, and local normal
        self.selected_points: list[SelectedPoint] = []

        self.raycaster: ViewerRaycaster = ViewerRaycaster(self.scene)

    def _snap_to_grid(self, position: Vec3) -> Vec3:
        """
        Snap a position to the grid based on grid_snap settings.
        
        Parameters
        ----------
        position : Vec3
            The position to snap.
            
        Returns
        -------
        Vec3
            The snapped position.
        """
        snap_x, snap_y, snap_z = self.options.grid_snap
        
        # Snap each axis if the snap value is non-negative
        x = round(position.x / snap_x) * snap_x if snap_x >= 0 else position.x
        y = round(position.y / snap_y) * snap_y if snap_y >= 0 else position.y
        z = round(position.z / snap_z) * snap_z if snap_z >= 0 else position.z
            
        return Vec3.from_xyz(x, y, z)

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
            ray_hit = self.raycaster.cast_ray(ray.origin.v, ray.direction.v)

            if ray_hit.is_hit and ray_hit.geom:
                link = ray_hit.geom.link
                world_pos = ray_hit.position
                world_normal = ray_hit.normal
                
                pose: Pose = Pose.from_link(link)
                local_pos = pose.inverse_transform_point(world_pos)
                local_normal = pose.inverse_transform_direction(world_normal)

                # Apply grid snapping to local position
                local_pos = self._snap_to_grid(local_pos)

                selected_point = SelectedPoint(
                    link=link,
                    local_position=local_pos,
                    local_normal=local_normal
                )
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

            closest_hit = self.raycaster.cast_ray(mouse_ray.origin.v, mouse_ray.direction.v)
            if closest_hit.is_hit:
                snap_pos = self._snap_to_grid(closest_hit.position)
                # Draw hover preview
                self.scene.draw_debug_sphere(
                    snap_pos.v,
                    self.options.sphere_radius,
                    self.options.hover_color,
                )
                self.scene.draw_debug_arrow(
                    snap_pos.v,
                    closest_hit.normal.v * 0.1,
                    self.options.sphere_radius / 2,
                    self.options.hover_color,
                )

            if self.selected_points:
                world_positions = []
                for point in self.selected_points:
                    pose = Pose.from_link(point.link)
                    current_world_pos = pose.transform_point(point.local_position)
                    world_positions.append(current_world_pos.v)

                if len(world_positions) == 1:
                    self.scene.draw_debug_sphere(
                        world_positions[0],
                        self.options.sphere_radius,
                        self.options.sphere_color,
                    )
                else:
                    import numpy as np

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
            with open(output_file, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                
                writer.writerow([
                    'point_idx',
                    'link_idx',
                    'local_pos_x',
                    'local_pos_y',
                    'local_pos_z',
                    'local_normal_x',
                    'local_normal_y',
                    'local_normal_z'
                ])
                
                for i, point in enumerate(self.selected_points, 1):
                    writer.writerow([
                        i,
                        point.link.idx,
                        point.local_position.x,
                        point.local_position.y,
                        point.local_position.z,
                        point.local_normal.x,
                        point.local_normal.y,
                        point.local_normal.z,
                    ])

            gs.logger.info(f"[MeshPointSelectorPlugin] Wrote {len(self.selected_points)} selected points to '{output_file}'")

        except Exception as e:
            gs.logger.error(f"[MeshPointSelectorPlugin] Error writing to '{output_file}': {e}")
