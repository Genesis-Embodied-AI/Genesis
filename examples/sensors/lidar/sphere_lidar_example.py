#!/usr/bin/env python3
"""
Genesis LiDAR Sensor Example with Sphere Robot

This script demonstrates how to use the LiDAR sensor in Genesis with a simple sphere robot.
Key features:
1. Uses the Genesis LidarSensor class 
2. Proper integration with Genesis physics simulation
3. Real-time LiDAR point cloud visualization with red points
4. Simple sphere robot for focus on LiDAR functionality
5. Interactive visualization showing detected obstacles in the environment
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
            camera_pos=(8.0, 8.0, 6.0) if n_envs == 1 else (15.0, 15.0, 8.0),  # Higher camera for multi-env
            camera_lookat=(0.0, 0.0, 1.0),
            camera_fov=60,
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
    """Setup environment with obstacles for interesting LiDAR visualization."""
    print(f"Setting up environment with 360-degree obstacles for {n_envs} environment(s)...")
    
    # Ground plane for all environments
    terrain = scene.add_entity(
        gs.morphs.Plane()
    )
    
    # Add 360-degree obstacles around the robot for comprehensive LiDAR testing
    obstacles = []
    
    # Environment spacing for multi-env scenarios
    env_spacing = 20.0  # Space environments 20m apart
    
    for env_idx in range(n_envs):
        # Calculate environment offset
        env_offset_x = env_idx * env_spacing
        env_offset_y = 0.0
        
        print(f"Setting up obstacles for environment {env_idx} at offset ({env_offset_x}, {env_offset_y})")
        
        # Create a circular arrangement of obstacles around the robot
        # Robot is at (env_offset_x, env_offset_y, 1), so we place obstacles in a circle around it
        
        # Inner circle of pillars at radius 3m
        inner_radius = 3.0
        n_inner_pillars = 8
        for i in range(n_inner_pillars):
            angle = 2 * np.pi * i / n_inner_pillars
            x = env_offset_x + inner_radius * np.cos(angle)
            y = env_offset_y + inner_radius * np.sin(angle)
            
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
            x = env_offset_x + outer_radius * np.cos(angle)
            y = env_offset_y + outer_radius * np.sin(angle)
            
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
            obstacle = scene.add_entity(
                gs.morphs.Sphere(
                    radius=0.2 + np.random.random() * 0.2,  # Random radius 0.2-0.4
                    pos=(pos[0] + env_offset_x, pos[1] + env_offset_y, pos[2]),
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
            obstacle = scene.add_entity(
                gs.morphs.Box(
                    size=config['size'],
                    pos=(config['pos'][0] + env_offset_x, config['pos'][1] + env_offset_y, config['pos'][2]),
                    fixed=True
                )
            )
            obstacles.append(obstacle)
    
    total_per_env = n_inner_pillars + n_outer_obstacles + len(sphere_positions) + len(close_obstacles)
    print(f"Added {len(obstacles)} total obstacles ({total_per_env} per environment):")
    print(f"  - {n_inner_pillars} inner pillars at {inner_radius}m radius per env")
    print(f"  - {n_outer_obstacles} outer obstacles at {outer_radius}m radius per env") 
    print(f"  - {len(sphere_positions)} random spheres at medium distance per env")
    print(f"  - {len(close_obstacles)} close obstacles for near-range testing per env")
    
    return terrain, obstacles


def create_sphere_robot_with_lidar(scene, n_envs=1):
    """Create sphere robot(s) with LiDAR sensor(s)."""
    print(f"Creating sphere robot with LiDAR sensor for {n_envs} environment(s)...")
    
    # Environment spacing for multi-env scenarios  
    env_spacing = 20.0
    
    # Create sphere robot(s) - one per environment
    robots = []
    lidar_sensors = []
    
    for env_idx in range(n_envs):
        # Calculate robot position for this environment
        robot_x = env_idx * env_spacing
        robot_y = 0.0
        robot_z = 1.0
        
        robot = scene.add_entity(
            morph=gs.morphs.Sphere(
                radius=0.2,
                pos=(robot_x, robot_y, robot_z),
                fixed=False
            ),
            material=gs.materials.Rigid(
                rho=1000.0,  # Density
                friction=0.3
            )
        )
        robots.append(robot)
        
        # Add LiDAR sensor configuration - optimized for 360-degree testing
        lidar_config = {
            'n_scan_lines': 16,         # Reduced for multi-env performance
            'n_points_per_line': 32,    # Good 360-degree coverage
            'fov_vertical': 30.0,       # degrees
            'fov_horizontal': 360.0,    # degrees (full 360-degree rotation)
            'max_range': 8.0,           # meters (reduced to focus on our obstacle arrangement)
            'min_range': 0.1            # meters
        }
        
        # Create LiDAR sensor attached to the robot
        lidar_sensor = gs.sensors.LidarSensor(
            entity=robot,
            link_idx=None,              # Use base link
            use_local_frame=False,      # Return points in world frame
            config=lidar_config
        )
        lidar_sensors.append(lidar_sensor)
        
        print(f"Environment {env_idx}: Robot at ({robot_x}, {robot_y}, {robot_z})")
    
    print(f"LiDAR sensor config: {lidar_config}")
    print("LiDAR hit points will be visualized as RED spheres.")
    
    # For single environment, return single objects; for multi-env, return lists
    if n_envs == 1:
        return robots[0], lidar_sensors[0]
    else:
        return robots, lidar_sensors


def run_lidar_simulation(scene, robot, lidar_sensor, n_envs=1, num_steps=200):
    """Run the simulation with LiDAR sensor readings for single or multi-environment."""
    print(f"Running LiDAR simulation with {n_envs} environment(s)...")
    
    # Handle both single and multi-environment cases
    if n_envs == 1:
        robots = [robot]
        lidar_sensors = [lidar_sensor]
    else:
        robots = robot  # Already a list
        lidar_sensors = lidar_sensor  # Already a list
    
    # Build the scene
    scene.build(n_envs=n_envs)
    
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
                # Process each environment
                for env_idx in range(n_envs):
                    start_time = time.time()
                    
                    # Read LiDAR data for this environment
                    hit_points, hit_distances = lidar_sensors[env_idx].read()
                    
                    end_time = time.time()
                    read_time = (end_time - start_time) * 1000  # Convert to milliseconds
                    
                    # Analyze the readings
                    if hit_points is not None and hit_distances is not None:
                        # Get statistics for this environment
                        valid_hits = hit_distances[0] < lidar_sensors[env_idx].config['max_range']
                        n_hits = np.sum(valid_hits)
                        total_rays = lidar_sensors[env_idx].config['n_scan_lines'] * lidar_sensors[env_idx].config['n_points_per_line']
                        hit_rate = 100.0 * n_hits / total_rays
                        
                        if n_hits > 0:
                            # Extract valid hit points for visualization
                            valid_points = hit_points[0][valid_hits]
                            valid_distances = hit_distances[0][valid_hits]
                            
                            min_dist = np.min(valid_distances)
                            max_dist = np.max(valid_distances)
                            mean_dist = np.mean(valid_distances)
                            
                            env_prefix = f"Env{env_idx}" if n_envs > 1 else ""
                            print(f"Step {step:3d} {env_prefix}: LiDAR read time: {read_time:5.1f}ms, "
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
                            print(f"Step {step:3d} {env_prefix}: LiDAR read time: {read_time:5.1f}ms, No hits detected")
                    else:
                        env_prefix = f"Env{env_idx}" if n_envs > 1 else ""
                        print(f"Step {step:3d} {env_prefix}: LiDAR sensor returned None")
            
            step_count += 1
            
            # Add some simple robot movement for demonstration
            if step % 50 == 0 and step > 0:  # Less frequent movement
                for env_idx in range(n_envs):
                    # Apply a small impulse to robot
                    impulse = np.array([
                        (np.random.random() - 0.5) * 1.0,  # x direction
                        (np.random.random() - 0.5) * 1.0,  # y direction  
                        0.5                                 # slight upward impulse
                    ])
                    
                    robot_pos = robots[env_idx].get_pos()
                    env_prefix = f"Env{env_idx}" if n_envs > 1 else ""
                    print(f"Step {step} {env_prefix}: Applying impulse {impulse} to robot at position {robot_pos}")
                
    except KeyboardInterrupt:
        print(f"\nSimulation stopped by user at step {step_count}")
        # Clean up visualization
        for env_idx in range(n_envs):
            if current_point_cloud_nodes[env_idx] is not None:
                scene.clear_debug_object(current_point_cloud_nodes[env_idx])
    
    print(f"Simulation completed after {step_count} steps")


def main():
    """Main function to run the LiDAR sensor demonstration."""
    print("=== Genesis LiDAR Sensor Example with Multi-Environment ===")
    print("This example demonstrates the LiDAR sensor with sphere robots in multiple environments.")
    print("LiDAR hit points will be shown as red spheres.")
    print()
    
    # Configuration for multi-environment testing
    n_envs = 3
    
    if n_envs > 1:
        print(f"Running with {n_envs} environments for multi-environment testing")
    else:
        print("Running with single environment")
    
    try:
        # Create scene
        scene = create_lidar_example_scene(n_envs)
        
        # Setup environment 
        terrain, obstacles = setup_environment_with_obstacles(scene, n_envs)
        
        # Create robot with LiDAR
        robot, lidar_sensor = create_sphere_robot_with_lidar(scene, n_envs)
        
        # Run simulation
        run_lidar_simulation(scene, robot, lidar_sensor, n_envs, num_steps=200)
        
        print("\nExample completed successfully!")
        
    except Exception as e:
        print(f"Error running example: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
