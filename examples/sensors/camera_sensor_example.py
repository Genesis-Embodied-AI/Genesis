"""
Example demonstrating camera sensors with different rendering backends.

This example shows:
1. Creating cameras as sensors using add_sensor()
2. Multiple cameras sharing the same backend renderer
3. Both .render() and .read() API patterns
4. Adding lights per backend
5. Rendering across multiple environments (batched)
6. Attaching/detaching cameras to rigid links (dynamic camera mounting)
"""

import genesis as gs
from genesis.options.sensors import RasterizerCameraOptions, RaytracerCameraOptions, BatchRendererCameraOptions

########################## init ##########################
gs.init(seed=0, precision="32", backend=gs.gpu, logging_level="info")

########################## create a scene ##########################
scene = gs.Scene(
    rigid_options=gs.options.RigidOptions(
        enable_collision=True,
        gravity=(0, 0, -9.8),
    ),
    show_viewer=False,  # No viewer needed - cameras are independent
)

########################## entities ##########################
# Floor
plane = scene.add_entity(
    morph=gs.morphs.Plane(),
    surface=gs.surfaces.Rough(color=(0.4, 0.4, 0.4)),
)

# Sphere - will be used for camera attachment demo
sphere = scene.add_entity(
    morph=gs.morphs.Sphere(pos=(0.0, 0.0, 2.0), radius=0.5),
    surface=gs.surfaces.Smooth(color=(1.0, 0.5, 0.5)),
)

# Box - another dynamic entity for variety
box = scene.add_entity(
    morph=gs.morphs.Box(pos=(1.0, 1.0, 1.0), size=(0.3, 0.3, 0.3)),
    surface=gs.surfaces.Rough(color=(0.5, 1.0, 0.5)),
)

########################## Example 1: Rasterizer Cameras ##########################
print("\n=== Rasterizer Cameras ===")

raster_cam0 = scene.add_sensor(
    RasterizerCameraOptions(
        res=(512, 512),
        pos=(3.0, 0.0, 2.0),
        lookat=(0.0, 0.0, 1.0),
        up=(0.0, 0.0, 1.0),
        fov=60.0,
        near=0.1,
        far=100.0,
    )
)

raster_cam1 = scene.add_sensor(
    RasterizerCameraOptions(
        res=(256, 256),
        pos=(0.0, 3.0, 2.0),
        lookat=(0.0, 0.0, 1.0),
        up=(0.0, 0.0, 1.0),
        fov=45.0,
    )
)

# Camera attached to moving sphere
raster_cam_attached = scene.add_sensor(
    RasterizerCameraOptions(
        res=(320, 240),
        pos=(0.0, 0.0, 3.0),  # Initial position (will be overridden by attachment)
        lookat=(0.0, 0.0, 0.0),
        up=(0.0, 0.0, 1.0),
        fov=70.0,
    )
)

# Add lights (shared across all rasterizer cameras)
raster_cam0.add_light(
    pos=(2.0, 2.0, 5.0),
    color=(1.0, 1.0, 1.0),
    intensity=5.0,
)

########################## Example 2: Raytracer Cameras (TODO: Not implemented yet) ##########################
# print("\n=== Raytracer Cameras ===")
# raytrace_cam0 = scene.add_sensor(RaytracerCameraOptions(...))

########################## Example 3: Batch Renderer Cameras (TODO: Not implemented yet) ##########################
# print("\n=== Batch Renderer Cameras ===")
# batch_cam0 = scene.add_sensor(BatchRendererCameraOptions(...))

########################## build ##########################
scene.build(n_envs=2)  # Build with 2 environments for batched rendering

########################## attach camera to sphere ##########################
print("\n=== Attaching Camera to Sphere ===")

# Create offset transform for camera attachment
# Camera will be positioned 1.0 unit above the sphere, looking down
import numpy as np

offset_T = np.eye(4, dtype=np.float32)
offset_T[2, 3] = 1.0  # 1 meter above the sphere center

# Attach camera to sphere's rigid link
sphere_link = sphere.links[0]  # Get the first (and only) link
raster_cam_attached.attach(sphere_link, offset_T)
print(f"Camera attached to sphere link: {sphere_link}")

########################## simulate and render ##########################
print("\n=== Simulation Loop ===")

for i in range(100):
    scene.step()

    # Render every 10 steps
    if i % 10 == 0:
        print(f"\n--- Step {i} ---")

        # Method 1: render() then read()
        raster_cam0.render()
        data = raster_cam0.read()
        print(f"Rasterizer cam0 RGB shape: {data.rgb.shape}")
        # Expected: (2, 512, 512, 3) for 2 envs

        # Method 2: render() with envs_idx, then read specific env
        raster_cam1.render()
        data_env0 = raster_cam1.read(envs_idx=0)
        print(f"Rasterizer cam1 RGB (env 0) shape: {data_env0.rgb.shape}")
        # Expected: (256, 256, 3) for single env

        # Attached camera automatically follows the sphere
        raster_cam_attached.render()
        data_attached = raster_cam_attached.read()
        print(f"Attached camera RGB shape: {data_attached.rgb.shape}")

        # Detach camera at step 50
        if i == 50:
            print(">>> Detaching camera from sphere...")
            raster_cam_attached.detach()
            print(">>> Camera detached! It will now stay at its current position.")

        # Save images every 10 steps
        import matplotlib.pyplot as plt
        import os

        os.makedirs("camera_sensor_output", exist_ok=True)

        # Save rasterizer cam0 output (both envs)
        plt.imsave(f"camera_sensor_output/raster_cam0_env0_step{i:03d}.png", data.rgb[0])
        plt.imsave(f"camera_sensor_output/raster_cam0_env1_step{i:03d}.png", data.rgb[1])

        # Save rasterizer cam1 output (both envs)
        plt.imsave(f"camera_sensor_output/raster_cam1_env0_step{i:03d}.png", data_env0.rgb)
        data_env1 = raster_cam1.read(envs_idx=1)
        plt.imsave(f"camera_sensor_output/raster_cam1_env1_step{i:03d}.png", data_env1.rgb)

        # Save attached camera output (both envs)
        plt.imsave(f"camera_sensor_output/attached_cam_env0_step{i:03d}.png", data_attached.rgb[0])
        plt.imsave(f"camera_sensor_output/attached_cam_env1_step{i:03d}.png", data_attached.rgb[1])

        if i == 0:
            print("\n✓ Saving images to camera_sensor_output/")
        if i == 50:
            print("✓ Saved detachment frame")

print("\n=== Simulation Complete ===")
