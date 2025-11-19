"""
Example demonstrating camera sensors with different rendering backends.

Creating cameras as sensors using add_sensor() with three backends
Rasterizer, Raytracer and BatchRenderer.
Test the attach/detach, add light, batch rendering functionalities.
"""

import genesis as gs
from genesis.options.sensors import RasterizerCameraOptions, RaytracerCameraOptions, BatchRendererCameraOptions

########################## init ##########################
gs.init(seed=0, precision="32", backend=gs.gpu, logging_level="info")

########################## create a scene ##########################
# Check if LuisaRenderPy is available
ENABLE_RAYTRACER = True  # Set to True to test raytracer (currently has known issues with segfaults)

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

########################## Camera Configurations ##########################
# Define camera configurations as lists for easier management

rasterizer_configs = [
    {
        "name": "raster_cam0",
        "options": RasterizerCameraOptions(
            res=(512, 512),
            pos=(3.0, 0.0, 2.0),
            lookat=(0.0, 0.0, 1.0),
            up=(0.0, 0.0, 1.0),
            fov=60.0,
            near=0.1,
            far=100.0,
        ),
        "attach": False,
        "lights": [{"pos": (2.0, 2.0, 5.0), "color": (1.0, 1.0, 1.0), "intensity": 5.0}],
    },
    {
        "name": "raster_cam1",
        "options": RasterizerCameraOptions(
            res=(512, 512),
            pos=(0.0, 3.0, 2.0),
            lookat=(0.0, 0.0, 1.0),
            up=(0.0, 0.0, 1.0),
            fov=60.0,
            near=0.1,
            far=100.0,
        ),
        "attach": False,
        "lights": [],
    },
    {
        "name": "raster_cam_attached",
        "options": RasterizerCameraOptions(
            res=(320, 240),
            pos=(0.0, 0.0, 3.0),  # Initial position (will be overridden by attachment)
            lookat=(0.0, 0.0, 0.0),
            up=(0.0, 0.0, 1.0),
            fov=70.0,
            near=0.1,
            far=100.0,
        ),
        "attach": True,
        "lights": [],
    },
]

raytracer_configs = []
if ENABLE_RAYTRACER:
    try:
        import LuisaRenderPy

        raytracer_configs = [
            {
                "name": "raytrace_cam0",
                "options": RaytracerCameraOptions(
                    res=(512, 512),
                    pos=(3.0, 0.0, 2.0),
                    lookat=(0.0, 0.0, 1.0),
                    up=(0.0, 0.0, 1.0),
                    fov=60.0,
                    model="pinhole",
                    spp=64,
                    denoise=False,
                    env_surface=gs.surfaces.Emission(
                        emissive_texture=gs.textures.ColorTexture(color=(0.2, 0.3, 0.5)),
                    ),
                    env_radius=20.0,
                ),
                "attach": False,
                "lights": [{"pos": (2.0, 2.0, 5.0), "color": (10.0, 10.0, 10.0), "intensity": 1.0}],
            },
            {
                "name": "raytrace_cam1",
                "options": RaytracerCameraOptions(
                    res=(512, 512),
                    pos=(0.0, 3.0, 2.0),
                    lookat=(0.0, 0.0, 1.0),
                    up=(0.0, 0.0, 1.0),
                    fov=60.0,
                    model="pinhole",
                    spp=64,
                    denoise=False,
                ),
                "attach": False,
                "lights": [],
            },
            {
                "name": "raytrace_cam_attached",
                "options": RaytracerCameraOptions(
                    res=(320, 240),
                    pos=(0.0, 0.0, 3.0),
                    lookat=(0.0, 0.0, 0.0),
                    up=(0.0, 0.0, 1.0),
                    fov=70.0,
                    spp=64,
                ),
                "attach": True,
                "lights": [],
            },
        ]
        print("✓ Raytracer cameras created (LuisaRenderPy available)")
    except ImportError:
        print("⊘ Skipping Raytracer cameras (LuisaRenderPy not installed)")
        print("  Install from: https://github.com/LuisaGroup/LuisaRender")
else:
    print("⊘ Raytracer disabled (set ENABLE_RAYTRACER=True to enable - experimental)")

batch_renderer_configs = [
    {
        "name": "batch_cam0",
        "options": BatchRendererCameraOptions(
            res=(512, 512),
            pos=(3.0, 0.0, 2.0),
            lookat=(0.0, 0.0, 1.0),
            up=(0.0, 0.0, 1.0),
            fov=60.0,
            use_rasterizer=True,
        ),
        "attach": False,
        "lights": [{"pos": (2.0, 2.0, 5.0), "color": (1.0, 1.0, 1.0), "intensity": 5.0, "directional": False}],
    },
    {
        "name": "batch_cam1",
        "options": BatchRendererCameraOptions(
            res=(512, 512),
            pos=(0.0, 3.0, 2.0),
            lookat=(0.0, 0.0, 1.0),
            up=(0.0, 0.0, 1.0),
            fov=60.0,
            use_rasterizer=True,
        ),
        "attach": False,
        "lights": [],
    },
    {
        "name": "batch_cam_attached",
        "options": BatchRendererCameraOptions(
            res=(512, 512),  # Must match other batch cameras
            pos=(0.0, 0.0, 3.0),
            lookat=(0.0, 0.0, 0.0),
            up=(0.0, 0.0, 1.0),
            fov=70.0,
            use_rasterizer=True,
        ),
        "attach": True,
        "lights": [],
    },
]

########################## Create Cameras ##########################
print("\n=== Rasterizer Cameras ===")
cameras = {}

# Create rasterizer cameras
for config in rasterizer_configs:
    camera = scene.add_sensor(config["options"])
    cameras[config["name"]] = camera

    # Add lights
    for light_config in config["lights"]:
        camera.add_light(**light_config)

print(f"✓ Created {len(rasterizer_configs)} rasterizer cameras")

# Create raytracer cameras
if raytracer_configs:
    for config in raytracer_configs:
        camera = scene.add_sensor(config["options"])
        cameras[config["name"]] = camera

        # Add lights
        for light_config in config["lights"]:
            camera.add_light(**light_config)

# Create batch renderer cameras
print("\n=== Batch Renderer Cameras ===")
for config in batch_renderer_configs:
    camera = scene.add_sensor(config["options"])
    cameras[config["name"]] = camera

    # Add lights
    for light_config in config["lights"]:
        camera.add_light(**light_config)

print(f"✓ Created {len(batch_renderer_configs)} batch renderer cameras")


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

# Attach cameras that are configured to be attached
attached_cameras = []
for config_group in [rasterizer_configs, raytracer_configs, batch_renderer_configs]:
    for config in config_group:
        if config["attach"]:
            camera = cameras[config["name"]]
            camera.attach(sphere_link, offset_T)
            attached_cameras.append(camera)
            print(f"✓ {config['name']} attached to sphere link: {sphere_link}")

print(f"✓ Attached {len(attached_cameras)} cameras to sphere")

########################## simulate and render ##########################
print("\n=== Simulation Loop ===")

# Define reading patterns for different camera types
read_patterns = {
    "raster_cam0": {"method": "all_envs", "print_suffix": ""},
    "raster_cam1": {"method": "single_env", "env_idx": 0, "print_suffix": " (env 0)"},
    "raster_cam_attached": {"method": "all_envs", "print_suffix": ""},
    "raytrace_cam0": {"method": "all_envs", "print_suffix": " (auto-rendered on read)"},
    "raytrace_cam1": {"method": "single_env", "env_idx": 0, "print_suffix": " (env 0, auto-rendered on read)"},
    "raytrace_cam_attached": {"method": "all_envs", "print_suffix": " (auto-rendered on read)"},
    "batch_cam0": {"method": "all_envs", "print_suffix": ""},
    "batch_cam1": {"method": "single_env", "env_idx": 0, "print_suffix": " (env 0)"},
    "batch_cam_attached": {"method": "all_envs", "print_suffix": ""},
}

# Define image saving patterns
save_patterns = {
    "raster_cam0": [{"env_idx": 0, "suffix": "_env0"}, {"env_idx": 1, "suffix": "_env1"}],
    "raster_cam1": [
        {"env_idx": 0, "suffix": "_env0"},
        {"read_env": 1, "suffix": "_env1"},
    ],  # Special case: read env 1 separately
    "raster_cam_attached": [{"env_idx": 0, "suffix": "_env0"}, {"env_idx": 1, "suffix": "_env1"}],
    "raytrace_cam0": [{"env_idx": 0, "suffix": "_env0"}],
    "raytrace_cam1": [{"env_idx": 0, "suffix": "_env0"}],
    "raytrace_cam_attached": [{"env_idx": 0, "suffix": "_env0"}],
    "batch_cam0": [{"env_idx": 0, "suffix": "_env0"}, {"env_idx": 1, "suffix": "_env1"}],
    "batch_cam1": [
        {"env_idx": 0, "suffix": "_env0"},
        {"read_env": 1, "suffix": "_env1"},
    ],  # Special case: read env 1 separately
    "batch_cam_attached": [{"env_idx": 0, "suffix": "_env0"}, {"env_idx": 1, "suffix": "_env1"}],
}

import matplotlib.pyplot as plt
import os

os.makedirs("camera_sensor_output", exist_ok=True)


# Helper to convert torch tensors to numpy arrays for saving
def to_numpy_for_save(tensor_or_array):
    if hasattr(tensor_or_array, "cpu"):
        return tensor_or_array.cpu().numpy()
    return tensor_or_array


for i in range(100):
    scene.step()

    # Render every 10 steps
    if i % 10 == 0:
        print(f"\n--- Step {i} ---")

        # Read and print camera data
        camera_data = {}
        for cam_name, camera in cameras.items():
            if cam_name not in read_patterns:
                continue

            pattern = read_patterns[cam_name]
            if pattern["method"] == "all_envs":
                data = camera.read()
            elif pattern["method"] == "single_env":
                data = camera.read(envs_idx=pattern["env_idx"])

            camera_data[cam_name] = data
            print(f"  {cam_name.replace('_', ' ').title()} RGB shape: {data.rgb.shape}{pattern['print_suffix']}")

        # Detach cameras at step 50
        if i == 50:
            print("\n>>> Detaching all attached cameras from sphere...")
            for camera in attached_cameras:
                camera.detach()
            print(">>> All cameras detached! They will now stay at their current positions.\n")

        # Save images
        for cam_name, patterns in save_patterns.items():
            if cam_name not in camera_data:
                continue

            data = camera_data[cam_name]
            for pattern in patterns:
                if "read_env" in pattern:
                    # Special case: read a different environment
                    data_specific = cameras[cam_name].read(envs_idx=pattern["read_env"])
                    rgb_data = data_specific.rgb
                else:
                    # Use data from the main read
                    env_idx = pattern["env_idx"]
                    rgb_data = data.rgb[env_idx] if data.rgb.ndim > 3 else data.rgb

                filename = f"camera_sensor_output/{cam_name}{pattern['suffix']}_step{i:03d}.png"
                plt.imsave(filename, to_numpy_for_save(rgb_data))

        if i == 0:
            print("\n✓ Saving images to camera_sensor_output/")
        if i == 50:
            print("✓ Saved detachment frame")

print("\n=== Simulation Complete ===")
print("✓ Backend renderers tested successfully!")
print(
    f"  - Rasterizer: Fast OpenGL rendering ({len([c for c in rasterizer_configs if not c['attach']])} static + {len([c for c in rasterizer_configs if c['attach']])} attached cameras) ✓"
)

if raytracer_configs:
    attached_raytracer = len([c for c in raytracer_configs if c["attach"]])
    static_raytracer = len([c for c in raytracer_configs if not c["attach"]])
    print(
        f"  - Raytracer: High-quality path tracing ({static_raytracer} static + {attached_raytracer} attached cameras) ✓"
    )
else:
    print("  - Raytracer: Disabled (set ENABLE_RAYTRACER=True to enable)")

attached_batch = len([c for c in batch_renderer_configs if c["attach"]])
static_batch = len([c for c in batch_renderer_configs if not c["attach"]])
print(
    f"  - BatchRenderer: Efficient multi-camera batched rendering ({static_batch} static + {attached_batch} attached cameras) ✓"
)

print(f"\n✓ Images saved to camera_sensor_output/")
print(f"  - Tested attach/detach functionality ({len(attached_cameras)} cameras attached to moving sphere)")
print("  - Cameras follow sphere from step 0-50, then stay static after detachment")
print("  - Compare the rendering quality across backends!")
