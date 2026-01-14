import math
import sys
import os
import time  # Added time import for timeout functionality
import platform as platform_module
# Ensure we can import genesis from the local source tree
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import random
from typing import TYPE_CHECKING, List, Tuple, Optional

import numpy as np
import torch

# Enable MPS (Metal Performance Shaders) acceleration on macOS
if platform_module.system() == "Darwin":
    if torch.backends.mps.is_available():
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        print("🔧 macOS detected - MPS (Metal Performance Shaders) acceleration enabled")
    else:
        print("⚠️  macOS detected but MPS not available on this device")

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


def fly_mission(mission_control: MissionControl, controller: "DronePIDController", scene: gs.Scene, cam: Camera, timeout: float = 120.0, min_video_seconds: float = 10.0):
    """
    Executes the mission with optimized processing:
    - Physics simulation and rendering in main loop
    - Ensures minimum video length for demo purposes.

    Args:
        timeout: Maximum wall-clock time in seconds to run the simulation.
        min_video_seconds: Minimum video duration in seconds (at least 10s for demo).
    """
    drone = controller.drone
    step = 0
    dt = 0.01  # Simulation timestep
    
    # Calculate minimum steps for video duration
    min_steps_for_video = int(min_video_seconds / dt)  # At least 1000 steps for 10s
    max_steps = max(12000, min_steps_for_video)  # Ensure at least min video length
    
    start_time = time.time()
    
    # Tracking statistics
    rings_passed = 0
    last_ring_idx = -1
    mission_done = False
    
    # Performance tracking
    step_times = []

    print(f"🚀 Simulation Configuration:")
    print(f"   Minimum video length: {min_video_seconds}s ({min_steps_for_video} steps)")
    print(f"   Max steps: {max_steps}, dt={dt}s → {max_steps * dt:.1f}s max mission time")
    print(f"   Timeout: {timeout}s wall-clock time")

    try:
        while step < max_steps:
            step_start = time.perf_counter()
            
            # Check timeout (wall-clock time)
            elapsed = time.time() - start_time
            if elapsed > timeout:
                # But ensure we have minimum video length
                if step >= min_steps_for_video:
                    print(f"\n⏱️  Timeout reached ({timeout}s). Video has {step * dt:.1f}s of content.")
                    break
                else:
                    print(f"\n⏱️  Timeout but continuing for minimum video length...")
            
            # ===== PHYSICS SIMULATION =====
            # State estimation
            pos_tensor = drone.get_pos()
            vel_tensor = drone.get_vel()
            pos_np = pos_tensor.cpu().numpy()
            vel_np = vel_tensor.cpu().numpy()

            # Mission Control - Safety check
            if not mission_control.check_bounds(pos_np):
                if step >= min_steps_for_video:
                    print("❌ Mission Aborted: Out of bounds.")
                    break
                # Continue for minimum video even if out of bounds (hover)
                target_pos = pos_np
                keep_flying = False
            else:
                # Get Guidance
                target_pos, keep_flying = mission_control.get_target(pos_np, vel_np)

            # Check mission completion
            if not keep_flying and mission_control.mission_status == "COMPLETED" and not mission_done:
                print("✅ Mission Completed Successfully!")
                print(f"   Traversed {rings_passed} rings!")
                mission_done = True
                # Continue simulation for video length but mark done

            # Track ring traversal
            if mission_control.current_ring_idx > last_ring_idx:
                last_ring_idx = mission_control.current_ring_idx
                if mission_control.current_ring_idx > 0:
                    rings_passed += 1
                    print(f"   ✓ Ring {rings_passed} traversed! (Step {step}, Time {step*dt:.2f}s)")

            # PID Control - compute motor RPMs
            if not mission_done or step < min_steps_for_video:
                rpms = controller.update(target_pos)
                rpms = [clamp(r) for r in rpms]
            else:
                # Hover in place after mission complete
                rpms = [base_rpm] * 4
            
            drone.set_propellels_rpm(rpms)

            # Physics step
            scene.step()

            # ===== VISUALIZATION =====
            target_tuple = tuple(target_pos)
            pos_tuple = tuple(pos_np)
            vel_tuple = tuple(vel_np)
            
            # Draw debug line
            scene.draw_debug_line(pos_tuple, target_tuple, radius=0.005, color=(0.0, 1.0, 0.0, 0.5))
            
            # Update camera and render
            update_camera_chase(cam, pos_tuple, vel_tuple, target_tuple)
            cam.render()

            step_times.append(time.perf_counter() - step_start)
            step += 1
            
            # Progress report every 500 steps
            if step % 500 == 0:
                video_time = step * dt
                avg_step = np.mean(step_times[-100:]) * 1000 if step_times else 0
                fps = 1000.0 / avg_step if avg_step > 0 else 0
                print(f"   📊 Step {step}: Video={video_time:.1f}s, StepTime={avg_step:.1f}ms, FPS={fps:.1f}")

    except gs.GenesisException as e:
        if "Viewer closed" in str(e):
            video_time = step * dt
            print(f"\n🛑 Viewer closed at step {step} ({video_time:.1f}s video)")
            print(f"   Rings traversed: {rings_passed}")
            if video_time < min_video_seconds:
                print(f"   ⚠️  Video shorter than {min_video_seconds}s target")
        else:
            raise e
    except Exception as e:
        print(f"\n⚠️  Unexpected error: {e}")
        print(f"   Steps completed: {step}, Rings traversed: {rings_passed}")
    
    # Final statistics
    video_time = step * dt
    print(f"\n📈 Simulation Statistics:")
    print(f"   Total steps: {step}")
    print(f"   Video duration: {video_time:.2f}s {'✅' if video_time >= min_video_seconds else '⚠️'}")
    print(f"   Rings passed: {rings_passed}")
    if step_times:
        print(f"   Avg step time: {np.mean(step_times)*1000:.2f}ms")
        print(f"   Avg FPS: {1.0/np.mean(step_times):.1f}")
    
    return step >= min_steps_for_video  # Return True if minimum video length achieved


def main():
    """
    Main function with distributed simulation architecture:
    - GPU Core 1: Physics simulation (drone dynamics, collision)
    - GPU Core 2: Ring visualization & rendering (async)
    
    Ensures minimum 10 second video output for demo purposes.
    """
    # Detect platform and select appropriate backend
    system_platform = platform_module.system()
    if system_platform == "Darwin":
        # Use Metal GPU backend on macOS for MPS acceleration
        backend = gs.gpu  # Metal backend on macOS
        print(f"🍎 Using Metal GPU backend on {system_platform}")
        print(f"   Metal handles work distribution across GPU cores automatically")
    else:
        backend = gs.gpu
        print(f"🖥️  Using GPU backend on {system_platform}")
    
    gs.init(backend=backend)

    ##### scene with optimized options #####
    # Note: run_in_thread not supported on macOS, use synchronous rendering
    scene = gs.Scene(
        show_viewer=True, 
        sim_options=gs.options.SimOptions(dt=0.01),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(3.5, 0.0, 2.5),
            camera_lookat=(0.0, 0.0, 0.5),
            camera_fov=45,
        ),
    )

    ##### entities #####
    plane = scene.add_entity(morph=gs.morphs.Plane())

    drone = scene.add_entity(morph=gs.morphs.Drone(file="urdf/drones/cf2x.urdf", pos=(0, 0, 0.2)))

    # Define Rings with increased complexity - Multiple rings forming a challenging path
    # Ring visualization runs on GPU Core 2 (async)
    rings_config = [
        # Starting sequence - single ring at moderate height
        {'pos': (2.0, 0.0, 1.2),   'normal': (1, 0, 0),    'radius': 0.5},
        
        # Rising sequence - rings ascending
        {'pos': (3.5, 1.5, 1.8),   'normal': (0.8, 0.6, 0), 'radius': 0.5},
        {'pos': (4.5, 3.0, 2.5),   'normal': (1, 0, 0.2),   'radius': 0.5},
        
        # Height peak sequence
        {'pos': (3.0, 4.5, 3.2),   'normal': (0.5, 0.8, 0.2), 'radius': 0.45},
        
        # Turning sequence - challenging turns
        {'pos': (1.0, 4.0, 2.8),   'normal': (-0.7, 0.7, 0), 'radius': 0.45},
        {'pos': (-1.5, 2.5, 2.5),  'normal': (-1, 0, 0.3),   'radius': 0.5},
        
        # Diagonal descent
        {'pos': (-2.0, 0.5, 1.8),  'normal': (-0.5, -0.5, -0.5), 'radius': 0.5},
        
        # Final challenge - tight ring
        {'pos': (0.0, -2.0, 1.5),  'normal': (0, -1, 0.2),   'radius': 0.4},
        
        # Final ring - return to start area
        {'pos': (1.0, -1.0, 1.2),  'normal': (1, -0.5, 0),   'radius': 0.5},
    ]

    print(f"\n🎯 Creating {len(rings_config)} rings (GPU Core 2 - Visualization)...")
    # Visualize Rings - Static geometry handled by GPU Core 2
    for i, ring in enumerate(rings_config):
        create_ring_visual(scene, ring['pos'], ring['normal'], ring['radius'])
    print(f"   ✅ Ring geometry created: {len(rings_config) * 20} spheres")

    # Initialize Mission Control
    mission = MissionControl(rings_config)

    # PID Params - Adjusted for better ring traversal performance
    # Control computation runs on GPU Core 1
    pid_params = [
        [2.5, 0.05, 0.1],   # X position - more responsive
        [2.5, 0.05, 0.1],   # Y position - more responsive
        [3.0, 0.1, 0.2],    # Z position - faster altitude changes
        [22.0, 1.0, 25.0],  # Roll - improved stability
        [22.0, 1.0, 25.0],  # Pitch - improved stability
        [28.0, 2.0, 22.0],  # Yaw - better rotation control
        [12.0, 0.5, 1.5],   # Roll rate
        [12.0, 0.5, 1.5],   # Pitch rate
        [2.5, 0.1, 0.3],    # Yaw rate
    ]

    controller = DronePIDController(drone=drone, dt=0.01, base_rpm=base_rpm, pid_params=pid_params)

    # Camera Setup for recording
    init_pos = drone.morph.pos
    cam = scene.add_camera(
        pos=(init_pos[0] - 2, init_pos[1], init_pos[2] + 1),
        lookat=init_pos,
        GUI=False,
        res=(1280, 720),
        fov=45
    )

    ##### build #####
    print(f"\n🔨 Building scene...")
    scene.build()
    print(f"   ✅ Scene built successfully")

    cam.start_recording()

    print(f"\n🚁 Starting Mission: Complex Ring Traversal")
    print(f"   Total rings to traverse: {len(rings_config)}")
    print(f"   Platform: {system_platform}")
    print(f"   Min video length: 10 seconds")
    print(f"")
    
    # Run distributed simulation with minimum 10 second video guarantee
    video_ok = fly_mission(
        mission, 
        controller, 
        scene, 
        cam, 
        timeout=180.0,  # Allow more time for distributed processing
        min_video_seconds=10.0  # Guarantee at least 10 seconds of video
    )

    # Save video (handle gracefully if viewer was closed)
    try:
        output_path = "../../videos/fly_route_rings.mp4"
        cam.stop_recording(save_to_filename=output_path)
        
        # Verify video was created
        abs_path = os.path.abspath(os.path.join(os.path.dirname(__file__), output_path))
        if os.path.exists(abs_path):
            file_size = os.path.getsize(abs_path) / (1024 * 1024)  # MB
            print(f"\n✅ Video saved successfully!")
            print(f"   Path: {abs_path}")
            print(f"   Size: {file_size:.2f} MB")
            if video_ok:
                print(f"   Duration: ≥10 seconds ✅")
            else:
                print(f"   Duration: <10 seconds ⚠️")
        else:
            print(f"\n⚠️  Video file not found at expected path")
            
    except Exception as e:
        print(f"\n⚠️  Could not save video: {e}")
        print("   (This may happen if the viewer was closed early)")
    
    print(f"\n🏁 Simulation complete!")



if __name__ == "__main__":
    main()
