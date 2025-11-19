"""
Example demonstrating camera sensors with different rendering backends.

This example shows:
1. Creating cameras as sensors using add_sensor() with three backends:
   - Rasterizer (OpenGL-based, fast for debugging)
   - Raytracer (LuisaRender, high-quality path tracing) - optional if LuisaRenderPy installed
   - BatchRenderer (Madrona, efficient multi-camera GPU rendering)
2. Multiple cameras sharing the same backend renderer
3. Both .render() and .read() API patterns
4. Adding lights per backend (each backend has its own light system)
5. Rendering across multiple environments (batched)
6. Attaching/detaching cameras to rigid links for ALL backends (dynamic camera mounting)
   - Cameras follow a falling sphere from step 0-50
   - At step 50, all cameras detach and remain static
   - Demonstrates dynamic viewpoint changes vs. static cameras
7. Comparing rendering quality across different backends
8. Graceful handling of optional dependencies (LuisaRenderPy)
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
    renderer=gs.renderers.RayTracer(
        env_surface=gs.surfaces.Emission(
            emissive_texture=gs.textures.ColorTexture(color=(0.2, 0.3, 0.5)),
        ),
        env_radius=20.0,
    ),
    show_viewer=False,
)

########################## entities ##########################
plane = scene.add_entity(
    morph=gs.morphs.Plane(),
    surface=gs.surfaces.Rough(color=(0.4, 0.4, 0.4)),
)

sphere = scene.add_entity(
    morph=gs.morphs.Sphere(pos=(0.0, 0.0, 2.0), radius=0.5),
    surface=gs.surfaces.Smooth(color=(1.0, 0.5, 0.5)),
)

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

# Check if LuisaRenderPy is available
# Note: Raytracer is experimental and may have issues. Set ENABLE_RAYTRACER=True to test it.
ENABLE_RAYTRACER = True  # Set to True to test raytracer (currently has known issues with segfaults)

raytrace_cam0 = None
raytrace_cam_attached = None
if ENABLE_RAYTRACER:
    try:
        import LuisaRenderPy

        raytrace_cam0 = scene.add_sensor(
            RaytracerCameraOptions(
                res=(512, 512),
                pos=(3.5, 0.0, 2.5),
                lookat=(0.0, 0.0, 1.0),
                up=(0.0, 0.0, 1.0),
                fov=55.0,
                model="pinhole",
                spp=64,
                denoise=False,
                env_surface=gs.surfaces.Emission(
                    emissive_texture=gs.textures.ColorTexture(color=(0.2, 0.3, 0.5)),
                ),
                env_radius=20.0,
            )
        )

        # Camera attached to sphere for raytracer
        raytrace_cam_attached = scene.add_sensor(
            RaytracerCameraOptions(
                res=(320, 240),
                pos=(0.0, 0.0, 3.0),
                lookat=(0.0, 0.0, 0.0),
                up=(0.0, 0.0, 1.0),
                fov=70.0,
                spp=64,
            )
        )

        # Add sphere light
        raytrace_cam0.add_light(
            pos=(0.0, 0.0, 5.0),
            color=(10.0, 10.0, 10.0),
            intensity=1.0,
        )

        print("✓ Raytracer cameras created (LuisaRenderPy available)")
    except ImportError:
        print("⊘ Skipping Raytracer cameras (LuisaRenderPy not installed)")
        print("  Install from: https://github.com/LuisaGroup/LuisaRender")
else:
    print("⊘ Raytracer disabled (set ENABLE_RAYTRACER=True to enable - experimental)")

########################## Example 3: Batch Renderer Cameras ##########################
print("\n=== Batch Renderer Cameras ===")

# Note: All batch renderer cameras must have same resolution
batch_cam0 = scene.add_sensor(
    BatchRendererCameraOptions(
        res=(256, 256),
        pos=(2.5, 1.0, 2.0),
        lookat=(0.0, 0.0, 1.0),
        up=(0.0, 0.0, 1.0),
        fov=50.0,
        use_rasterizer=True,
    )
)

batch_cam1 = scene.add_sensor(
    BatchRendererCameraOptions(
        res=(256, 256),  # Must match batch_cam0
        pos=(1.0, 2.5, 2.0),
        lookat=(0.0, 0.0, 1.0),
        up=(0.0, 0.0, 1.0),
        fov=50.0,
        use_rasterizer=True,
    )
)

# Camera attached to sphere for batch renderer
batch_cam_attached = scene.add_sensor(
    BatchRendererCameraOptions(
        res=(256, 256),  # Must match other batch cameras
        pos=(0.0, 0.0, 3.0),
        lookat=(0.0, 0.0, 0.0),
        up=(0.0, 0.0, 1.0),
        fov=60.0,
        use_rasterizer=True,
    )
)

# Add lights
batch_cam0.add_light(
    pos=(5.0, 5.0, 5.0),
    dir=(-1.0, -1.0, -1.0),
    color=(1.0, 1.0, 1.0),
    intensity=3.0,
    directional=True,
)

########################## build ##########################
scene.build(n_envs=2)  # Build with 2 environments for batched rendering

########################## attach cameras to sphere ##########################
print("\n=== Attaching Cameras to Sphere ===")

# Create offset transform for camera attachment
# Camera will be positioned 1.0 unit above the sphere, looking down
import numpy as np

offset_T = np.eye(4, dtype=np.float32)
offset_T[2, 3] = 1.0  # 1 meter above the sphere center

# Get sphere's rigid link
sphere_link = sphere.links[0]  # Get the first (and only) link

# Attach rasterizer camera
raster_cam_attached.attach(sphere_link, offset_T)
print(f"✓ Rasterizer camera attached to sphere link: {sphere_link}")

# Attach batch renderer camera
batch_cam_attached.attach(sphere_link, offset_T)
print(f"✓ Batch renderer camera attached to sphere link: {sphere_link}")

# Attach raytracer camera (if available)
if raytrace_cam_attached is not None:
    raytrace_cam_attached.attach(sphere_link, offset_T)
    print(f"✓ Raytracer camera attached to sphere link: {sphere_link}")

########################## simulate and render ##########################
print("\n=== Simulation Loop ===")

for i in range(100):
    scene.step()

    # Render every 10 steps
    if i % 10 == 0:
        print(f"\n--- Step {i} ---")

        # ========== Rasterizer Cameras ==========
        # Method 1: render() then read()
        raster_cam0.render()
        data_raster0 = raster_cam0.read()
        print(f"  Rasterizer cam0 RGB shape: {data_raster0.rgb.shape}")

        # Method 2: render() with envs_idx, then read specific env
        raster_cam1.render()
        data_raster1_env0 = raster_cam1.read(envs_idx=0)
        print(f"  Rasterizer cam1 RGB (env 0) shape: {data_raster1_env0.rgb.shape}")

        # Attached rasterizer camera automatically follows the sphere
        raster_cam_attached.render()
        data_raster_attached = raster_cam_attached.read()
        print(f"  Rasterizer attached RGB shape: {data_raster_attached.rgb.shape}")

        # ========== Raytracer Cameras ==========
        if raytrace_cam0 is not None:
            raytrace_cam0.render()
            print("Raytracer cam0 rendered")
            data_raytrace0 = raytrace_cam0.read()
            print(f"  Raytracer cam0 RGB shape: {data_raytrace0.rgb.shape}")

            # Attached raytracer camera
            raytrace_cam_attached.render()
            print("Raytracer attached rendered")
            data_raytrace_attached = raytrace_cam_attached.read()
            print(f"  Raytracer attached RGB shape: {data_raytrace_attached.rgb.shape}")

        # ========== Batch Renderer Cameras ==========
        # Note: Batch renderer renders all cameras at once
        batch_cam0.render()  # This renders all batch cameras
        batch_cam1.render()  # (internally coordinated to avoid duplicate work)
        batch_cam_attached.render()

        data_batch0 = batch_cam0.read()
        data_batch1 = batch_cam1.read()
        data_batch_attached = batch_cam_attached.read()
        print(f"  Batch cam0 RGB shape: {data_batch0.rgb.shape}")
        print(f"  Batch cam1 RGB shape: {data_batch1.rgb.shape}")
        print(f"  Batch attached RGB shape: {data_batch_attached.rgb.shape}")

        # Detach cameras at step 50
        if i == 50:
            print("\n>>> Detaching all attached cameras from sphere...")
            raster_cam_attached.detach()
            batch_cam_attached.detach()
            if raytrace_cam_attached is not None:
                raytrace_cam_attached.detach()
            print(">>> All cameras detached! They will now stay at their current positions.\n")

        # Save images every 10 steps
        import matplotlib.pyplot as plt
        import os

        os.makedirs("camera_sensor_output", exist_ok=True)

        # Save rasterizer outputs
        plt.imsave(f"camera_sensor_output/raster_cam0_env0_step{i:03d}.png", data_raster0.rgb[0])
        plt.imsave(f"camera_sensor_output/raster_cam0_env1_step{i:03d}.png", data_raster0.rgb[1])
        plt.imsave(f"camera_sensor_output/raster_cam1_env0_step{i:03d}.png", data_raster1_env0.rgb)
        data_raster1_env1 = raster_cam1.read(envs_idx=1)
        plt.imsave(f"camera_sensor_output/raster_cam1_env1_step{i:03d}.png", data_raster1_env1.rgb)
        plt.imsave(f"camera_sensor_output/raster_attached_env0_step{i:03d}.png", data_raster_attached.rgb[0])
        plt.imsave(f"camera_sensor_output/raster_attached_env1_step{i:03d}.png", data_raster_attached.rgb[1])

        # Save raytracer outputs (only renders env 0)
        if raytrace_cam0 is not None:
            plt.imsave(f"camera_sensor_output/raytrace_cam0_env0_step{i:03d}.png", data_raytrace0.rgb[0])
            plt.imsave(f"camera_sensor_output/raytrace_attached_env0_step{i:03d}.png", data_raytrace_attached.rgb[0])

        # Save batch renderer outputs
        plt.imsave(f"camera_sensor_output/batch_cam0_env0_step{i:03d}.png", data_batch0.rgb[0])
        plt.imsave(f"camera_sensor_output/batch_cam0_env1_step{i:03d}.png", data_batch0.rgb[1])
        plt.imsave(f"camera_sensor_output/batch_cam1_env0_step{i:03d}.png", data_batch1.rgb[0])
        plt.imsave(f"camera_sensor_output/batch_cam1_env1_step{i:03d}.png", data_batch1.rgb[1])
        plt.imsave(f"camera_sensor_output/batch_attached_env0_step{i:03d}.png", data_batch_attached.rgb[0])
        plt.imsave(f"camera_sensor_output/batch_attached_env1_step{i:03d}.png", data_batch_attached.rgb[1])

        if i == 0:
            print("\n✓ Saving images to camera_sensor_output/")
        if i == 50:
            print("✓ Saved detachment frame")

print("\n=== Simulation Complete ===")
print("✓ Backend renderers tested successfully!")
print("  - Rasterizer: Fast OpenGL rendering with camera attachment ✓")
if raytrace_cam0 is not None:
    print("  - Raytracer: High-quality path tracing with camera attachment ✓")
else:
    print("  - Raytracer: Disabled (set ENABLE_RAYTRACER=True to enable)")
print("  - BatchRenderer: Efficient multi-camera batched rendering with camera attachment ✓")
print(f"\n✓ Images saved to camera_sensor_output/")
print("  - Tested attach/detach functionality for Rasterizer and BatchRenderer")
print("  - Cameras follow sphere from step 0-50, then stay static after detachment")
print("  - Compare the rendering quality across backends!")
