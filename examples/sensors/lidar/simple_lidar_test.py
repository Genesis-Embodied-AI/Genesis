#!/usr/bin/env python3
"""
Simple LiDAR Sensor Test with Single Environment

This script creates a minimal test for the LiDAR sensor with proper scene geometry extraction.
"""

import numpy as np
import genesis as gs
import time


def main():
    """Simple test for LiDAR sensor with proper scene geometry extraction."""
    print("=== Simple LiDAR Sensor Test - Single Environment ===")
    print("Testing LiDAR sensor with dynamic scene geometry extraction...")
    
    # Initialize Genesis
    gs.init(backend=gs.gpu)
    
    # Create simple scene
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=0.02, 
            substeps=2,
            gravity=(0.0, 0.0, -9.81)
        ),
        viewer_options=gs.options.ViewerOptions(
            max_FPS=60,
            camera_pos=(5.0, 5.0, 3.0),
            camera_lookat=(0.0, 0.0, 1.0),
            camera_fov=50,
        ),
        rigid_options=gs.options.RigidOptions(
            dt=0.02,
            constraint_solver=gs.constraint_solver.Newton,
        ),
        show_viewer=True,
    )
    
    # Add ground plane
    print("Adding ground plane...")
    terrain = scene.add_entity(
        gs.morphs.Plane()
    )
    
    # Add 360-degree obstacles for comprehensive testing
    print("Adding 360-degree obstacle arrangement...")
    obstacles = []
    
    # Create a simple circular arrangement of obstacles
    radius = 3.0
    n_obstacles = 6
    for i in range(n_obstacles):
        angle = 2 * np.pi * i / n_obstacles
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        
        obstacle = scene.add_entity(
            gs.morphs.Box(
                size=(0.3, 0.3, 1.0),
                pos=(x, y, 0.5),
                fixed=True
            )
        )
        obstacles.append(obstacle)
    
    # Add a few closer obstacles
    close_obstacles = [
        (1.5, 0.0, 0.3),
        (0.0, 1.5, 0.4),
        (-1.5, 0.0, 0.3),
        (0.0, -1.5, 0.4),
    ]
    
    for pos in close_obstacles:
        obstacle = scene.add_entity(
            gs.morphs.Sphere(
                radius=0.2,
                pos=pos,
                fixed=True
            )
        )
        obstacles.append(obstacle)
    
    print(f"Added {len(obstacles)} obstacles in 360-degree arrangement")
    
    # Create sphere robot
    print("Creating sphere robot...")
    robot = scene.add_entity(
        morph=gs.morphs.Sphere(
            radius=0.2,
            pos=(0.0, 0.0, 1.0),
            fixed=False
        ),
        material=gs.materials.Rigid(
            rho=1000.0,
            friction=0.3
        )
    )
    
    # Simple LiDAR configuration for 360-degree testing
    lidar_config = {
        'n_scan_lines': 4,          # Very small for testing
        'n_points_per_line': 16,    # Increased for 360-degree coverage
        'fov_vertical': 30.0,       # degrees
        'fov_horizontal': 360.0,    # degrees (full rotation)
        'max_range': 6.0,           # meters (reduced to focus on our obstacles)
        'min_range': 0.1            # meters
    }
    
    # Create LiDAR sensor
    print("Creating LiDAR sensor...")
    lidar_sensor = gs.sensors.LidarSensor(
        entity=robot,
        link_idx=None,
        use_local_frame=False,
        config=lidar_config
    )
    
    print(f"LiDAR sensor created with {lidar_config['n_scan_lines']} scan lines, {lidar_config['n_points_per_line']} points per line")
    print(f"Total rays: {lidar_config['n_scan_lines'] * lidar_config['n_points_per_line']}")
    
    # Build scene (single environment - n_envs=1)
    print("Building scene with single environment...")
    scene.build(n_envs=1)
    
    # Debug: Extract and save scene geometry for inspection
    print("\n=== DEBUGGING MESH EXTRACTION ===")
    try:
        # Print mesh information summary
        lidar_sensor.print_mesh_summary()
        
        # Save mesh using the new method
        mesh_saved = lidar_sensor.save_scene_mesh("/tmp/simple_lidar_mesh.obj")
        if mesh_saved:
            print("Scene mesh saved to /tmp/simple_lidar_mesh.obj for inspection")
        
        # Get detailed mesh info and save individual entity meshes
        mesh_info = lidar_sensor.get_mesh_info()
        if mesh_info:
            print(f"\nDetailed mesh extraction information:")
            print(f"  - Extraction time: {mesh_info['extraction_time']:.1f}ms")
            print(f"  - Total entities processed: {len(mesh_info['entities'])}")
            print(f"  - Total geometries: {mesh_info['geometry_count']}")
            
            # Save individual entity meshes for debugging
            print(f"\nSaving individual entity meshes:")
            vertices, triangles = lidar_sensor.get_scene_mesh()
            
            for i, entity_info in enumerate(mesh_info['entities']):
                if len(entity_info['geometries']) > 0:
                    # Extract vertices and triangles for this entity
                    start_idx = entity_info['geometries'][0]['vertex_offset']
                    end_idx = start_idx + entity_info['vertex_count']
                    
                    entity_vertices = vertices[start_idx:end_idx]
                    
                    # Find triangles that belong to this entity
                    entity_triangles = []
                    triangle_offset = 0
                    for j, prev_entity in enumerate(mesh_info['entities'][:i]):
                        triangle_offset += prev_entity['triangle_count']
                    
                    entity_triangle_end = triangle_offset + entity_info['triangle_count']
                    entity_triangles = triangles[triangle_offset:entity_triangle_end] - start_idx
                    
                    # Save individual entity mesh
                    try:
                        import trimesh
                        entity_mesh = trimesh.Trimesh(vertices=entity_vertices, faces=entity_triangles)
                        filename = f"/tmp/entity_{i}_{entity_info['type']}.obj"
                        entity_mesh.export(filename)
                        print(f"  Entity {i} ({entity_info['type']}): {entity_info['vertex_count']} vertices, {entity_info['triangle_count']} triangles -> {filename}")
                        
                        # Print bounds for entity
                        bounds = entity_mesh.bounds
                        print(f"    Bounds: min={bounds[0]}, max={bounds[1]}")
                        
                    except Exception as e:
                        print(f"  Failed to save entity {i}: {e}")
        
        # Get raw mesh data  
        vertices, triangles = lidar_sensor.get_scene_mesh()
        print(f"\nRaw mesh data: {len(vertices)} vertices and {len(triangles)} triangles")
        
        # Print some vertex samples
        print(f"Sample vertices (first 5):")
        for i in range(min(5, len(vertices))):
            print(f"  Vertex {i}: {vertices[i]}")
            
        print(f"Sample triangles (first 5):")
        for i in range(min(5, len(triangles))):
            print(f"  Triangle {i}: {triangles[i]}")
            
    except Exception as e:
        print(f"ERROR in mesh extraction: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Test LiDAR for a few steps
    print("\n=== TESTING LIDAR SENSOR ===")
    for step in range(20):  # Reduced to 20 steps for quick testing
        scene.step()
        
        if step % 5 == 0:  # Test every 5 steps
            start_time = time.time()
            hit_points, hit_distances = lidar_sensor.read()
            end_time = time.time()
            
            read_time = (end_time - start_time) * 1000
            
            if hit_points is not None and hit_distances is not None:
                # Get statistics
                valid_hits = hit_distances[0] < lidar_sensor.config['max_range']
                n_hits = np.sum(valid_hits)
                total_rays = lidar_config['n_scan_lines'] * lidar_config['n_points_per_line']
                hit_rate = 100.0 * n_hits / total_rays
                
                print(f"Step {step:2d}: Read time: {read_time:5.1f}ms, Hit rate: {hit_rate:5.1f}% ({n_hits}/{total_rays})")
                
                if n_hits > 0:
                    valid_distances = hit_distances[0][valid_hits]
                    valid_points = hit_points[0][valid_hits]
                    
                    min_dist = np.min(valid_distances)
                    max_dist = np.max(valid_distances)
                    mean_dist = np.mean(valid_distances)
                    
                    print(f"         Distance range: {min_dist:.2f}-{max_dist:.2f}m (mean: {mean_dist:.2f}m)")
                    
                    # Print some sample hit points for debugging
                    print(f"         Sample hit points (first 3):")
                    for i in range(min(3, len(valid_points))):
                        print(f"           Point {i}: {valid_points[i]} (dist: {valid_distances[i]:.2f}m)")
                    
                    # Visualize some points
                    if step == 10:  # Only visualize once to avoid clutter
                        sample_indices = np.random.choice(len(valid_points), 
                                                         min(20, len(valid_points)), 
                                                         replace=False)
                        sample_points = valid_points[sample_indices]
                        
                        scene.draw_debug_spheres(
                            sample_points,
                            radius=0.05,
                            color=(1.0, 0.0, 0.0, 0.8)
                        )
                        print(f"         Visualized {len(sample_points)} LiDAR hit points as red spheres")
                else:
                    print(f"         No hits detected")
            else:
                print(f"Step {step:2d}: LiDAR returned None")
    
    print("\nTest completed!")


if __name__ == "__main__":
    main()
