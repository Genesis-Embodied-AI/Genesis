import math
import sys
import os
import time  # Added time import for timeout functionality
# Ensure we can import genesis from the local source tree
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import random
from typing import TYPE_CHECKING, List, Tuple, Optional

import numpy as np
import genesis as gs
from genesis.vis.camera import Camera

from quadcopter_controller import DronePIDController

if TYPE_CHECKING:
    from genesis.engine.entities.drone_entity import DroneEntity


base_rpm = 14468.429183500699
min_rpm = 0.5 * base_rpm
max_rpm = 2.0 * base_rpm

# Bounds settings
BOUNDS_MIN = np.array([-5.0, -5.0, 0.0])
BOUNDS_MAX = np.array([5.0, 5.0, 5.0])

class MissionControl:
    """
    Simulates a ground station or onboard computer that tracks the drone
    and manages the mission (waypoints, boundaries, ring traversal).
    """
    def __init__(self, rings: List[dict]):
        self.rings = rings
        self.current_ring_idx = 0
        self.state = "APPROACH" # APPROACH, TRAVERSE
        self.mission_status = "ACTIVE" # ACTIVE, COMPLETED, FAILED
        self.waypoints = []
        self._plan_mission()

    def _plan_mission(self):
        """Generates a sequence of waypoints to traverse all rings."""
        for ring in self.rings:
            pos = np.array(ring['pos'])
            normal = np.array(ring['normal'])
            radius = ring['radius']

            # Approach point (before the ring)
            approach_pt = pos - normal * 1.0 # 1 meter before
            # Traverse point (center of ring)
            traverse_pt = pos
            # Exit point (after the ring)
            exit_pt = pos + normal * 1.0     # 1 meter after

            self.waypoints.append({'pos': approach_pt, 'type': 'APPROACH'})
            self.waypoints.append({'pos': traverse_pt, 'type': 'TRAVERSE'})
            self.waypoints.append({'pos': exit_pt, 'type': 'EXIT'})

    def check_bounds(self, drone_pos: np.ndarray) -> bool:
        """Checks if drone is within flight safety zone."""
        if (np.any(drone_pos < BOUNDS_MIN) or np.any(drone_pos > BOUNDS_MAX)):
            print(f"[MissionControl] ALARM: Drone out of bounds! Pos: {drone_pos}")
            self.mission_status = "FAILED"
            return False
        return True

    def get_target(self, drone_pos: np.ndarray, drone_vel: np.ndarray) -> Tuple[np.ndarray, bool]:
        """Calculates the dynamic target for the drone."""
        if self.mission_status != "ACTIVE":
            return drone_pos, False # Hover in place if failed/done

        if not self.waypoints:
            print("[MissionControl] Mission Accomplished!")
            self.mission_status = "COMPLETED"
            return drone_pos, False

        current_wp = self.waypoints[0]
        target_pos = current_wp['pos']

        # Distance to current waypoint
        dist = np.linalg.norm(target_pos - drone_pos)

        # Switching logic
        # If we are close enough to the waypoint, move to next
        threshold = 0.3 if current_wp['type'] == 'TRAVERSE' else 0.5

        if dist < threshold:
            print(f"[MissionControl] Reached waypoint: {current_wp['type']}")
            self.waypoints.pop(0)
            if self.waypoints:
                target_pos = self.waypoints[0]['pos']
            else:
                self.mission_status = "COMPLETED"

        # Dynamic adjustment: Look ahead
        # If we have a next waypoint, blend it slightly for smooth turns
        if len(self.waypoints) > 1:
            next_wp = self.waypoints[1]['pos']
            # Simple blend if close
            if dist < 1.0:
                alpha = 1.0 - dist
                target_pos = target_pos * (1 - alpha) + next_wp * alpha

        return target_pos, True

def create_ring_visual(scene: gs.Scene, pos: Tuple[float, float, float], normal: Tuple[float, float, float], radius: float):
    """Creates a visual ring using small cylinders approximating a torus."""
    # Normalize normal vector
    n = np.array(normal)
    n = n / np.linalg.norm(n)

    # Create basis vectors for the ring plane
    # Arbitrary vector not parallel to n
    if abs(n[0]) < 0.9:
        a = np.array([1, 0, 0])
    else:
        a = np.array([0, 1, 0])

    b = np.cross(n, a)
    b = b / np.linalg.norm(b)
    c = np.cross(n, b)

    segments = 12
    seg_angle = 2 * np.pi / segments
    tube_radius = 0.05
    seg_length = 2 * radius * math.tan(np.pi / segments) * 1.1 # slightly overlap

    # We want to place cylinders along the circumference
    for i in range(segments):
        angle = i * seg_angle
        # Center of the segment
        seg_center_local = b * math.cos(angle) * radius + c * math.sin(angle) * radius
        seg_pos = np.array(pos) + seg_center_local

        # Orientation: Tangent to the circle
        tangent = -b * math.sin(angle) + c * math.cos(angle)

        # Genesis cylinders are aligned along Z axis (usually).
        # We need to rotate Z axis to match the tangent vector.
        # Quat from Z to tangent.
        z_axis = np.array([0, 0, 1])

        # Rotation axis
        rot_axis = np.cross(z_axis, tangent)
        rot_sin = np.linalg.norm(rot_axis)
        rot_cos = np.dot(z_axis, tangent)

        if rot_sin < 1e-6:
            # Parallel or anti-parallel
            rx, ry, rz = 0, 0, 0
            # If anti-parallel, need 180 flip. skip for simplicity as cylinder is symmetric
        else:
            rot_axis = rot_axis / rot_sin
            rot_angle = math.atan2(rot_sin, rot_cos)
            # Axis-angle to quat not directly supported in spawn?
            # We can use user-defined conversion or lookat logic.
            # However, for simplicity, let's just use quat from scipy or implement axis-angle to quat.
            pass

        # Simplified: Just place small spheres if orientation is hard
        # Or let's use the lookat parameter of camera which we don't have for entities easily without quats.
        # However, for simplicity, let's use small spheres to form the ring. It's easier and looks like dots.
        pass

    # Use spheres for simplicity and "virtual" look
    for i in range(20):
        angle = i * (2 * np.pi / 20)
        p = np.array(pos) + b * math.cos(angle) * radius + c * math.sin(angle) * radius
        scene.add_entity(
            morph=gs.morphs.Sphere(pos=tuple(p), radius=0.05),
            surface=gs.surfaces.Rough(color=(1.0, 0.8, 0.0)), # Gold rings
            material=gs.materials.Rigid()
        )

# Remove ObstacleAvoidance class as it simulates "Avoiding" but we want "Traversing"
# We will use MissionControl instead.

def hover(drone: "DroneEntity"):
    drone.set_propellels_rpm([base_rpm, base_rpm, base_rpm, base_rpm])


def clamp(rpm):
    return max(min_rpm, min(int(rpm), max_rpm))


def update_camera_chase(cam: Camera, drone_pos: Tuple[float, float, float], drone_vel: Tuple[float, float, float], target: Tuple[float, float, float]):
    """
    Updates camera to follow drone from behind (Third-person view).
    """
    speed_sq = drone_vel[0]**2 + drone_vel[1]**2

    # Smooth heading calculation
    if speed_sq > 0.1:
        desired_heading = math.atan2(drone_vel[1], drone_vel[0])
    else:
        desired_heading = math.atan2(target[1] - drone_pos[1], target[0] - drone_pos[0])

    dist = 2.0
    height = 0.8 # Lower camera to see pillars better

    cx = drone_pos[0] - math.cos(desired_heading) * dist
    cy = drone_pos[1] - math.sin(desired_heading) * dist
    cz = drone_pos[2] + height

    # Smooth camera movement (approximated by just setting pose here,
    # but could be interpolated if we had state. For now direct set is fine)
    cam.set_pose(pos=(cx, cy, cz), lookat=drone_pos)


def fly_mission(mission_control: MissionControl, controller: "DronePIDController", scene: gs.Scene, cam: Camera, timeout: float = 60.0):
    """
    Executes the mission with a timeout safeguard for slower machines.

    Args:
        timeout: Maximum wall-clock time in seconds to run the simulation.
    """
    drone = controller.drone
    step = 0
    max_steps = 3000
    start_time = time.time()

    print(f"Mission started with timeout: {timeout} seconds")

    while step < max_steps:
        # Check timeout
        if time.time() - start_time > timeout:
            print(f"Mission Timeout reached ({timeout}s). Stopping simulation to generate demo video.")
            break

        # State estimation
        pos_tensor = drone.get_pos()
        vel_tensor = drone.get_vel()
        pos_np = pos_tensor.cpu().numpy()
        vel_np = vel_tensor.cpu().numpy()

        # Mission Control (The "Other" Simulation Program/Logic)
        # Check Safety
        if not mission_control.check_bounds(pos_np):
            print("Mission Aborted: Out of bounds.")
            break

        # Get Guidance
        target_pos, keep_flying = mission_control.get_target(pos_np, vel_np)

        if not keep_flying and mission_control.mission_status == "COMPLETED":
            print("Mission Completed Successfully!")
            break

        # Control
        rpms = controller.update(target_pos)
        rpms = [clamp(r) for r in rpms]
        drone.set_propellels_rpm(rpms)

        scene.step()

        # Visualization & Tracking
        target_tuple = tuple(target_pos)
        pos_tuple = tuple(pos_np)

        # Draw line to target
        scene.draw_debug_line(pos_tuple, target_tuple, radius=0.005, color=(0.0, 1.0, 0.0, 0.5))

        # Camera
        update_camera_chase(cam, pos_tuple, tuple(vel_np), target_tuple)
        cam.render()

        step += 1

    if step >= max_steps:
        print("Mission Timed Out.")


def main():
    gs.init(backend=gs.gpu)

    ##### scene #####
    scene = gs.Scene(show_viewer=True, sim_options=gs.options.SimOptions(dt=0.01))

    ##### entities #####
    plane = scene.add_entity(morph=gs.morphs.Plane())

    drone = scene.add_entity(morph=gs.morphs.Drone(file="urdf/drones/cf2x.urdf", pos=(0, 0, 0.2)))

    # Define Rings (Pos, Normal, Radius)
    # A path that curves and changes height
    rings_config = [
        {'pos': (1.5, 1.5, 1.5),  'normal': (1, 1, 0),    'radius': 0.4},
        {'pos': (-1.0, 2.5, 2.0), 'normal': (-1, 0.5, 0), 'radius': 0.4},
        {'pos': (-2.0, 0.0, 1.0), 'normal': (0, -1, 0),   'radius': 0.4},
        {'pos': (0.0, -2.0, 1.5), 'normal': (1, 0, 0.5),  'radius': 0.4}
    ]

    # Visualize Rings
    for ring in rings_config:
        create_ring_visual(scene, ring['pos'], ring['normal'], ring['radius'])

    # Initialize Mission Control
    mission = MissionControl(rings_config)

    # PID Params
    pid_params = [
        [2.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [20.0, 0.0, 20.0],
        [20.0, 0.0, 20.0],
        [25.0, 0.0, 20.0],
        [10.0, 0.0, 1.0],
        [10.0, 0.0, 1.0],
        [2.0, 0.0, 0.2],
    ]

    controller = DronePIDController(drone=drone, dt=0.01, base_rpm=base_rpm, pid_params=pid_params)

    # Initial Camera Setup
    init_pos = drone.morph.pos
    cam = scene.add_camera(
        pos=(init_pos[0] - 2, init_pos[1], init_pos[2] + 1),
        lookat=init_pos,
        GUI=False,
        res=(1280, 720),
        fov=45
    )

    ##### build #####
    scene.build()

    cam.start_recording()

    print("Starting Mission: Ring Traversal")
    # Set a reasonable timeout for a demo (e.g., 60 seconds)
    fly_mission(mission, controller, scene, cam, timeout=60.0)

    cam.stop_recording(save_to_filename="../../videos/fly_route_rings.mp4")
    print("Video saved to ../../videos/fly_route_rings.mp4")



if __name__ == "__main__":
    main()
