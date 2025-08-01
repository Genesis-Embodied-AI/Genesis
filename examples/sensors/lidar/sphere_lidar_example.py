#!/usr/bin/env python3
"""
Genesis LiDAR Sensor Example with Go2 Robot

This script demonstrates how to use the LiDAR sensor in Genesis with Go2 quadruped robot.
Key features:
1. Uses the Genesis LidarSensor class 
2. Proper integration with Genesis physics simulation
3. Real-time LiDAR point cloud visualization with colored points
4. Go2 quadruped robot for realistic testing
5. Multi-environment support with different robot movement patterns
6. Interactive visualization showing detected obstacles in the environment
"""

import numpy as np
import genesis as gs
import time
from typing import Optional


def create_lidar_example_scene(n_envs=1):
    """Create a Genesis scene for LiDAR sensor demonstration."""
    print(f"Creating Genesis scene for LiDAR demonstration with {n_envs} environment(s)...")
    
    # Initialize Genesis
    gs.init(backend=gs.gpu)
    
    # Create scene
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=0.02, 
            substeps=2,
            gravity=(0.0, 0.0, -9.81)
        ),
        viewer_options=gs.options.ViewerOptions(
            max_FPS=60,
            camera_pos=(8.0, 8.0, 6.0) if n_envs == 1 else (25.0, 15.0, 12.0),  # Much higher and wider for multi-env
            camera_lookat=(0.0, 0.0, 1.0) if n_envs == 1 else (10.0, 0.0, 1.0),  # Look at center of multi-env
            camera_fov=60 if n_envs == 1 else 80,  # Wider FOV for multi-env
        ),
        vis_options=gs.options.VisOptions(
            rendered_envs_idx=list(range(n_envs))  # Render all environments
        ),
        rigid_options=gs.options.RigidOptions(
            dt=0.02,
            constraint_solver=gs.constraint_solver.Newton,
        ),
        show_viewer=True,
    )
    
    return scene


def setup_environment_with_obstacles(scene, n_envs=1):
    """Setup environment with obstacles - Genesis will replicate across environments."""
    print(f"Setting up environment with obstacles (Genesis will replicate to {n_envs} environments)...")
    
    # Ground plane for all environments
    terrain = scene.add_entity(
        gs.morphs.Plane()
    )
    
    # Add 360-degree obstacles around the robot for comprehensive LiDAR testing
    # Genesis will automatically replicate these to all environments with proper spacing
    obstacles = []
    
    # Create a circular arrangement of obstacles around the robot
    # Robot is at (0, 0, 1), Genesis will handle environment spacing automatically
    
    # Inner circle of pillars at radius 3m
    inner_radius = 3.0
    n_inner_pillars = 8
    for i in range(n_inner_pillars):
        angle = 2 * np.pi * i / n_inner_pillars
        x = inner_radius * np.cos(angle)
        y = inner_radius * np.sin(angle)
        
        # Genesis will replicate this obstacle to all environments
        obstacle = scene.add_entity(
            gs.morphs.Box(
                size=(0.3, 0.3, 1.5),
                pos=(x, y, 0.75),
                fixed=True
            )
        )
        obstacles.append(obstacle)
    
    # Outer circle of larger obstacles at radius 5m
    outer_radius = 5.0
    n_outer_obstacles = 6
    for i in range(n_outer_obstacles):
        angle = 2 * np.pi * i / n_outer_obstacles + np.pi / 6  # Offset for variety
        x = outer_radius * np.cos(angle)
        y = outer_radius * np.sin(angle)
        
        # Genesis will replicate this obstacle to all environments
        obstacle = scene.add_entity(
            gs.morphs.Box(
                size=(0.5, 0.5, 2.0),
                pos=(x, y, 1.0),
                fixed=True
            )
        )
        obstacles.append(obstacle)
    
    # Add some random spheres at medium distance for variety
    sphere_positions = [
        (2.0, 1.5, 0.4),
        (-1.8, 2.2, 0.5),
        (1.2, -2.8, 0.3),
        (-2.5, -1.0, 0.6),
        (0.8, 2.5, 0.4),
        (-2.2, 0.5, 0.3),
    ]
    
    for pos in sphere_positions:
        # Genesis will replicate this obstacle to all environments
        obstacle = scene.add_entity(
            gs.morphs.Sphere(
                radius=0.2 + np.random.random() * 0.2,  # Random radius 0.2-0.4
                pos=pos,
                fixed=True
            )
        )
        obstacles.append(obstacle)
    
    # Add some closer obstacles for near-range testing
    close_obstacles = [
        {'pos': (1.5, 0.0, 0.5), 'size': (0.2, 0.2, 1.0)},
        {'pos': (0.0, 1.5, 0.5), 'size': (0.2, 0.2, 1.0)},
        {'pos': (-1.5, 0.0, 0.5), 'size': (0.2, 0.2, 1.0)},
        {'pos': (0.0, -1.5, 0.5), 'size': (0.2, 0.2, 1.0)},
    ]
    
    for config in close_obstacles:
        # Genesis will replicate this obstacle to all environments
        obstacle = scene.add_entity(
            gs.morphs.Box(
                size=config['size'],
                pos=config['pos'],
                fixed=True
            )
        )
        obstacles.append(obstacle)

    total_obstacles = n_inner_pillars + n_outer_obstacles + len(sphere_positions) + len(close_obstacles)
    print(f"Added {total_obstacles} obstacle templates (Genesis will replicate to {n_envs * total_obstacles} total):")
    print(f"  - {n_inner_pillars} inner pillars at {inner_radius}m radius")
    print(f"  - {n_outer_obstacles} outer obstacles at {outer_radius}m radius") 
    print(f"  - {len(sphere_positions)} random spheres at medium distance")
    print(f"  - {len(close_obstacles)} close obstacles for near-range testing")
    
    return terrain, obstacles


def create_go2_robot_with_lidar(scene, n_envs=1):
    """Create Go2 robot with LiDAR sensor - Genesis will replicate across environments."""
    print(f"Creating Go2 robot with LiDAR sensor (Genesis will replicate to {n_envs} environments)...")
    
    # Initial robot position and orientation
    base_init_pos = np.array([0.0, 0.0, 0.3])  # Start at 30cm height for Go2
    base_init_quat = np.array([1.0, 0.0, 0.0, 0.0])  # No rotation
    
    # Create single Go2 robot - Genesis will replicate to all environments
    # Genesis will automatically space environments, so we use origin position
    robot = scene.add_entity(
        gs.morphs.URDF(
            file="urdf/go2/urdf/go2.urdf",
            pos=base_init_pos,
            quat=base_init_quat,
            fixed=False  # Allow robot to move
        )
    )
    
    # Add LiDAR sensor configuration - optimized for 360-degree testing with Go2 robot
    lidar_config = {
        'n_scan_lines': 16,         # Reduced for multi-env performance
        'n_points_per_line': 32,    # Good 360-degree coverage
        'fov_vertical': 30.0,       # degrees
        'fov_horizontal': 360.0,    # degrees (full 360-degree rotation)
        'max_range': 8.0,           # meters (reduced to focus on our obstacle arrangement)
        'min_range': 0.1            # meters
    }
    
    # Create LiDAR sensor attached to the robot's base link
    lidar_sensor = gs.sensors.LidarSensor(
        entity=robot,
        link_idx=None,              # Use base link
        use_local_frame=False,      # Return points in world frame
        config=lidar_config
    )
    
    print(f"Go2 robot created at position {base_init_pos} with quaternion {base_init_quat}")
    print(f"Genesis will automatically space robots across {n_envs} environments") 
    print(f"LiDAR sensor config: {lidar_config}")
    print("LiDAR hit points will be visualized as colored spheres (red, green, blue for different envs).")
    
    return robot, lidar_sensor, base_init_pos, base_init_quat


def run_lidar_simulation(scene, robot, lidar_sensor, base_init_pos, base_init_quat, n_envs=1, num_steps=200):
    """Run the simulation with LiDAR sensor readings for single or multi-environment."""
    print(f"Running LiDAR simulation with {n_envs} environment(s)...")
    
    # Convert to lists for consistent handling
    if n_envs == 1:
        robots = [robot]
        lidar_sensors = [lidar_sensor]
    else:
        # For multi-env, we need to handle the case where robot and lidar_sensor are single objects
        # that will be replicated by Genesis, not lists
        robots = [robot] * n_envs  # Create list with same robot reference
        lidar_sensors = [lidar_sensor] * n_envs  # Create list with same sensor reference
    
    # Build the scene - Genesis will replicate entities across environments
    scene.build(n_envs=n_envs)
    
    # After building, let's check robot positions to confirm proper spacing
    robot_pos = robots[0].get_pos()
    print(f"\nAfter scene.build(), robot positions:")
    if hasattr(robot_pos, 'shape') and len(robot_pos.shape) > 1:
        for env_idx in range(min(n_envs, robot_pos.shape[0])):
            print(f"  Environment {env_idx}: Robot at {robot_pos[env_idx]}")
    else:
        print(f"  Single environment: Robot at {robot_pos}")
    
    # For multi-environment testing, let's position robots at slightly different locations
    # within each environment to make the visualization more interesting
    if n_envs > 1:
        print(f"\nPositioning Go2 robots at different locations within each environment...")
        
        # Create different starting positions for each environment
        env_positions = []
        env_quaternions = []
        for env_idx in range(n_envs):
            # Slight variation in x,y position and orientation within each environment
            offset_x = (env_idx - (n_envs-1)/2) * 1.0  # Spread them out more for Go2
            offset_y = env_idx * 0.5  # Different y positions
            # Different orientations for variety
            yaw_offset = env_idx * 0.5  # Different yaw rotations
            quat = [np.cos(yaw_offset/2), 0, 0, np.sin(yaw_offset/2)]  # Rotation around Z axis
            
            env_positions.append([base_init_pos[0] + offset_x, base_init_pos[1] + offset_y, base_init_pos[2]])
            env_quaternions.append(quat)
        
        # Set positions for all environments at once
        import torch
        new_positions = torch.tensor(env_positions, device=gs.device)
        new_quaternions = torch.tensor(env_quaternions, device=gs.device)
        
        # Set robot positions and orientations for each environment
        for env_idx in range(n_envs):
            robots[0].set_pos(new_positions[env_idx:env_idx+1], envs_idx=[env_idx])
            robots[0].set_quat(new_quaternions[env_idx:env_idx+1], envs_idx=[env_idx])
            print(f"  Environment {env_idx}: Moved robot to {new_positions[env_idx].cpu().numpy()} with yaw {env_idx * 0.5:.2f} rad")
    
    # Print mesh information after scene is built (use first sensor)
    print("\n=== LiDAR Mesh Information ===")
    lidar_sensors[0].print_mesh_summary()
    
    # Save the extracted mesh for debugging (use first sensor)
    mesh_saved = lidar_sensors[0].save_scene_mesh("/tmp/lidar_scene_mesh.obj")
    if mesh_saved:
        print("Scene mesh saved to /tmp/lidar_scene_mesh.obj for inspection")
    
    # Get detailed mesh info
    mesh_info = lidar_sensors[0].get_mesh_info()
    if mesh_info:
        print(f"\nMesh extraction took {mesh_info['extraction_time']:.1f}ms")
        print(f"Ray casting will use {mesh_info['total_triangles']} triangles for collision detection")
    
    print("=" * 50)
    
    # Simulation parameters
    step_count = 0
    lidar_update_interval = 5  # Update LiDAR every 5 steps
    vis_update_interval = 10   # Update visualization every 10 steps
    sphere_radius = 0.05       # Radius for LiDAR point visualization (bigger for visibility)
    
    # Point cloud visualization variables (one per environment)
    current_point_cloud_nodes = [None] * n_envs
    
    print(f"Starting simulation for {num_steps} steps...")
    print("LiDAR Configuration:")
    print(f"  - Scan lines: {lidar_sensors[0].config['n_scan_lines']}")
    print(f"  - Points per line: {lidar_sensors[0].config['n_points_per_line']}")
    print(f"  - Max range: {lidar_sensors[0].config['max_range']}m")
    print(f"  - Total rays per scan: {lidar_sensors[0].config['n_scan_lines'] * lidar_sensors[0].config['n_points_per_line']}")
    print(f"  - LiDAR points will be visualized as RED SPHERES with radius {sphere_radius}m")
    
    try:
        for step in range(num_steps):
            # Step the simulation
            scene.step()
            
            # Read LiDAR sensor periodically
            if step % lidar_update_interval == 0:
                # For multi-environment, read from the single sensor but specify environment indices
                start_time = time.time()
                
                # Read LiDAR data for all environments at once
                hit_points, hit_distances = lidar_sensors[0].read()
                
                end_time = time.time()
                read_time = (end_time - start_time) * 1000  # Convert to milliseconds
                
                # Process each environment's data
                for env_idx in range(n_envs):
                    # Analyze the readings for this environment
                    if hit_points is not None and hit_distances is not None:
                        # Get statistics for this environment
                        env_hit_points = hit_points[env_idx] if n_envs > 1 else hit_points[0]
                        env_hit_distances = hit_distances[env_idx] if n_envs > 1 else hit_distances[0]
                        
                        valid_hits = env_hit_distances < lidar_sensors[0].config['max_range']
                        n_hits = np.sum(valid_hits)
                        total_rays = lidar_sensors[0].config['n_scan_lines'] * lidar_sensors[0].config['n_points_per_line']
                        hit_rate = 100.0 * n_hits / total_rays
                        
                        if n_hits > 0:
                            # Extract valid hit points for visualization
                            valid_points = env_hit_points[valid_hits]
                            valid_distances = env_hit_distances[valid_hits]
                            
                            min_dist = np.min(valid_distances)
                            max_dist = np.max(valid_distances)
                            mean_dist = np.mean(valid_distances)
                            
                            env_prefix = f"Env{env_idx}" if n_envs > 1 else ""
                            if env_idx == 0:  # Only show timing for first environment to avoid spam
                                print(f"Step {step:3d} {env_prefix}: LiDAR read time: {read_time:5.1f}ms, "
                                      f"Hit rate: {hit_rate:5.1f}% ({n_hits}/{total_rays}), "
                                      f"Distance range: {min_dist:.2f}-{max_dist:.2f}m (mean: {mean_dist:.2f}m)")
                            else:
                                print(f"Step {step:3d} {env_prefix}: "
                                      f"Hit rate: {hit_rate:5.1f}% ({n_hits}/{total_rays}), "
                                      f"Distance range: {min_dist:.2f}-{max_dist:.2f}m (mean: {mean_dist:.2f}m)")
                            
                            # Debug: Print some sample points (only for first environment and first few steps)
                            if env_idx == 0 and step <= 10:  # Only print for first env and first few steps
                                print(f"         Sample hit points (first 3):")
                                for i in range(min(3, len(valid_points))):
                                    print(f"           Point {i}: {valid_points[i]} (dist: {valid_distances[i]:.2f}m)")
                            
                            # Update point cloud visualization
                            if step % vis_update_interval == 0:
                                # Clear previous point cloud for this environment
                                if current_point_cloud_nodes[env_idx] is not None:
                                    scene.clear_debug_object(current_point_cloud_nodes[env_idx])
                                
                                # Visualize new point cloud as red spheres for better visibility
                                # Use different colors for different environments
                                colors = [(1.0, 0.0, 0.0, 0.8), (0.0, 1.0, 0.0, 0.8), (0.0, 0.0, 1.0, 0.8)]
                                color = colors[env_idx % len(colors)]
                                
                                current_point_cloud_nodes[env_idx] = scene.draw_debug_spheres(
                                    valid_points,
                                    radius=sphere_radius,
                                    color=color
                                )
                        else:
                            env_prefix = f"Env{env_idx}" if n_envs > 1 else ""
                            if env_idx == 0:  # Only show timing for first environment
                                print(f"Step {step:3d} {env_prefix}: LiDAR read time: {read_time:5.1f}ms, No hits detected")
                            else:
                                print(f"Step {step:3d} {env_prefix}: No hits detected")
                    else:
                        env_prefix = f"Env{env_idx}" if n_envs > 1 else ""
                        print(f"Step {step:3d} {env_prefix}: LiDAR sensor returned None")
            
            step_count += 1
            
            # Add some Go2 robot movement for demonstration using set_pos
            if step % 10 == 0 and step > 0:  # Update robot positions every 10 steps
                # Apply different movement patterns to robots in different environments
                for env_idx in range(n_envs):
                    # Get current time for smooth motion
                    t = step * 0.02  # simulation time (dt = 0.02)
                    
                    # Calculate new position based on different movement patterns
                    if env_idx == 0:
                        # Environment 0: Circular motion
                        radius = 1.5
                        center_x = 0.0
                        center_y = 0.0
                        new_x = center_x + radius * np.cos(t * 0.5)  # slow circular motion
                        new_y = center_y + radius * np.sin(t * 0.5)
                        new_z = base_init_pos[2]  # keep same height
                        # Rotate robot to face movement direction
                        yaw = t * 0.5 + np.pi/2  # face tangent to circle
                        new_quat = [np.cos(yaw/2), 0, 0, np.sin(yaw/2)]
                        
                    elif env_idx == 1:
                        # Environment 1: Figure-8 motion
                        scale = 1.2
                        new_x = scale * np.sin(t * 0.4)
                        new_y = scale * np.sin(t * 0.8)  # double frequency for figure-8
                        new_z = base_init_pos[2]
                        # Face movement direction
                        dx = scale * 0.4 * np.cos(t * 0.4)
                        dy = scale * 0.8 * np.cos(t * 0.8)
                        yaw = np.arctan2(dy, dx)
                        new_quat = [np.cos(yaw/2), 0, 0, np.sin(yaw/2)]
                        
                    else:
                        # Environment 2+: Linear back-and-forth motion
                        amplitude = 2.0
                        new_x = amplitude * np.sin(t * 0.3)  # back and forth in X
                        new_y = 0.5 * env_idx  # offset in Y for each env
                        new_z = base_init_pos[2]
                        # Face movement direction
                        if np.cos(t * 0.3) > 0:
                            yaw = 0.0  # facing +X
                        else:
                            yaw = np.pi  # facing -X
                        new_quat = [np.cos(yaw/2), 0, 0, np.sin(yaw/2)]
                    
                    # Apply new position and orientation to robot in specific environment
                    import torch
                    new_pos_tensor = torch.tensor([new_x, new_y, new_z], device=gs.device).unsqueeze(0)
                    new_quat_tensor = torch.tensor(new_quat, device=gs.device).unsqueeze(0)
                    
                    # Set robot position and orientation
                    robots[0].set_pos(new_pos_tensor, envs_idx=[env_idx], zero_velocity=False)
                    robots[0].set_quat(new_quat_tensor, envs_idx=[env_idx], zero_velocity=False)
                    
                    if step % 50 == 0:  # Print less frequently to avoid spam
                        print(f"Step {step} Env{env_idx}: Moved robot to [{new_x:.2f}, {new_y:.2f}, {new_z:.2f}], yaw={np.degrees(yaw):.1f}°")
                
                # Show current robot positions
                if step % 50 == 0:  # Print less frequently
                    current_pos = robots[0].get_pos()
                    if hasattr(current_pos, 'shape') and len(current_pos.shape) > 1:
                        print(f"          Current robot positions:")
                        for env_idx in range(min(n_envs, current_pos.shape[0])):
                            print(f"            Env{env_idx}: {current_pos[env_idx].cpu().numpy()}")
                    else:
                        print(f"          Current robot position: {current_pos}")
                
    except KeyboardInterrupt:
        print(f"\nSimulation stopped by user at step {step_count}")
        # Clean up visualization
        for env_idx in range(n_envs):
            if current_point_cloud_nodes[env_idx] is not None:
                scene.clear_debug_object(current_point_cloud_nodes[env_idx])
    
    print(f"Simulation completed after {step_count} steps")


def main():
    """Main function to run the LiDAR sensor demonstration."""
    print("=== Genesis LiDAR Sensor Example with Go2 Robot Multi-Environment ===")
    print("This example demonstrates the LiDAR sensor with Go2 robots in multiple environments.")
    print("LiDAR hit points will be shown as colored spheres.")
    print()
    
    # Configuration for multi-environment testing
    n_envs = 3  # Use 3 environments for better variety testing
    
    if n_envs > 1:
        print(f"Running with {n_envs} environments for multi-environment testing")
        print("Each environment will have:")
        print("  - Go2 quadruped robot with LiDAR sensor")
        print("  - Identical obstacle layout")
        print("  - Different colored LiDAR points (red, green, blue)")
        print("  - Different movement patterns:")
        print("    * Env 0: Circular motion")
        print("    * Env 1: Figure-8 motion") 
        print("    * Env 2+: Linear back-and-forth motion")
    else:
        print("Running with single environment")
        print("  - Go2 quadruped robot with LiDAR sensor")
    
    try:
        # Create scene
        scene = create_lidar_example_scene(n_envs)
        
        # Setup environment 
        terrain, obstacles = setup_environment_with_obstacles(scene, n_envs)
        
        # Create robot with LiDAR
        robot, lidar_sensor, base_init_pos, base_init_quat = create_go2_robot_with_lidar(scene, n_envs)
        
        # Run simulation
        run_lidar_simulation(scene, robot, lidar_sensor, base_init_pos, base_init_quat, n_envs, num_steps=500)
        
        print("\nExample completed successfully!")
        
    except Exception as e:
        print(f"Error running example: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
