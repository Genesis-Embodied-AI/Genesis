"""
Example demonstrating camera sensors with different rendering backends.

Creating cameras as sensors using add_sensor() with three backends
Rasterizer, Raytracer and BatchRenderer.
Test the attachment, add light, batch rendering functionalities.
"""

import os
import matplotlib.pyplot as plt
import genesis as gs
from genesis.utils.misc import tensor_to_array
from genesis.options.sensors import RasterizerCameraOptions, RaytracerCameraOptions, BatchRendererCameraOptions

########################## init ##########################
gs.init(seed=0, precision="32", backend=gs.gpu, logging_level="info")

########################## check dependencies ##########################
# Try to import LuisaRenderPy to determine if raytracer is available
try:
    import LuisaRenderPy

    ENABLE_RAYTRACER = True
    print("✓ LuisaRenderPy available - Raytracer will be enabled")
except ImportError:
    ENABLE_RAYTRACER = False
    print("⊘ LuisaRenderPy not available - Raytracer will be disabled")

try:
    import gs_madrona

    ENABLE_MADRONA = True
    print("✓ gs_madrona available - BatchRenderer will be enabled")
except ImportError:
    ENABLE_MADRONA = False
    print("⊘ gs_madrona not available - BatchRenderer will be disabled")
ENABLE_MADRONA = ENABLE_MADRONA and (gs.backend == gs.cuda)
########################## create a scene ##########################
# Choose renderer based on raytracer availability
if ENABLE_RAYTRACER:
    renderer = gs.renderers.RayTracer(
        env_surface=gs.surfaces.Emission(
            emissive_texture=gs.textures.ColorTexture(color=(0.2, 0.3, 0.5)),
        ),
        env_radius=20.0,
    )
else:
    # Use Rasterizer as fallback renderer
    renderer = gs.renderers.Rasterizer()

scene = gs.Scene(
    rigid_options=gs.options.RigidOptions(
        enable_collision=True,
        gravity=(0, 0, -9.8),
    ),
    renderer=renderer,
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
# Define common camera parameters
CAMERA_COMMON_KWARGS = dict(
    {
        "up": (0.0, 0.0, 1.0),
        "near": 0.1,
        "far": 100.0,
    }
)

CAMERA_SENSORS_KWARGS = [
    (
        "cam0",
        (3.0, 0.0, 2.0),
        (0.0, 0.0, 1.0),
        60.0,
        None,  # No attachment
        [{"pos": (2.0, 2.0, 5.0), "color": (1.0, 1.0, 1.0), "intensity": 5.0}],
    ),
    ("cam1", (0.0, 3.0, 2.0), (0.0, 0.0, 1.0), 60.0, None, []),
    (
        "cam_attached",
        (0.0, 0.0, 1.0),
        (0.0, 0.0, 0.0),
        70.0,
        {"entity_idx": None, "link_idx_local": 0, "pos_offset": (0.0, 0.0, 0.0), "euler_offset": (0.0, 0.0, 0.0)},
        [],
    ),
]


def create_camera_configs(backend_name, options_class, sphere_entity_idx=None, **backend_specific):
    """Create camera configurations for a specific backend."""
    configs = []
    for cam_suffix, pos, lookat, fov, attachment, lights in CAMERA_SENSORS_KWARGS:
        name = f"{backend_name}_{cam_suffix}"
        res = (500, 600)

        # Create options with common and backend-specific parameters
        options_kwargs = {
            "res": res,
            "pos": pos,
            "lookat": lookat,
            "up": CAMERA_COMMON_KWARGS["up"],
            "fov": fov,
            **backend_specific,
        }

        # Handle attachment
        if attachment is not None:
            # For attached cameras, set the entity_idx to the sphere's index
            if sphere_entity_idx is not None:
                options_kwargs.update(
                    {
                        "entity_idx": sphere_entity_idx,
                        "link_idx_local": attachment["link_idx_local"],
                        "pos_offset": attachment["pos_offset"],
                        "euler_offset": attachment["euler_offset"],
                    }
                )
            else:
                # If sphere_entity_idx is not provided, create as static camera
                pass

        # Add backend-specific parameters
        if backend_name == "raster":
            options_kwargs.update({"near": CAMERA_COMMON_KWARGS["near"], "far": CAMERA_COMMON_KWARGS["far"]})
        elif backend_name == "raytrace":
            options_kwargs.update(
                {
                    "model": "pinhole",
                    "spp": 64,
                    "denoise": False,
                }
            )
            if attachment is None:  # Only add env surface for non-attached cameras
                options_kwargs.update(
                    {
                        "env_surface": gs.surfaces.Emission(
                            emissive_texture=gs.textures.ColorTexture(color=(0.2, 0.3, 0.5)),
                        ),
                        "env_radius": 20.0,
                    }
                )
        elif backend_name == "batch":
            options_kwargs.update({"use_rasterizer": True})
            # Adjust lights for batch renderer
            if lights:
                adjusted_lights = [{**light, "directional": False} for light in lights]
                lights = adjusted_lights

        # Adjust lights for raytracer (different intensity/color)
        if backend_name == "raytrace" and lights:
            adjusted_lights = [{**light, "color": (10.0, 10.0, 10.0), "intensity": 1.0} for light in lights]
            lights = adjusted_lights

        options = options_class(**options_kwargs)
        configs.append(
            {
                "name": name,
                "options": options,
                "attachment": attachment,
                "lights": lights,
            }
        )

    return configs


# Create configurations for each backend
rasterizer_configs = create_camera_configs("raster", RasterizerCameraOptions, sphere_entity_idx=sphere.idx)
raytracer_configs = (
    create_camera_configs("raytrace", RaytracerCameraOptions, sphere_entity_idx=sphere.idx) if ENABLE_RAYTRACER else []
)
batch_renderer_configs = (
    create_camera_configs("batch", BatchRendererCameraOptions, sphere_entity_idx=sphere.idx) if ENABLE_MADRONA else []
)

########################## Create Cameras ##########################
cameras = {}
config_groups = []
config_groups += [("Rasterizer", rasterizer_configs)] if rasterizer_configs else []
config_groups += [("Raytracer", raytracer_configs)] if raytracer_configs else []
config_groups += [("Batch Renderer", batch_renderer_configs)] if batch_renderer_configs else []

for group_name, configs in config_groups:
    print(f"\n=== {group_name} Cameras ===")
    for config in configs:
        camera = scene.add_sensor(config["options"])
        cameras[config["name"]] = camera

        for light_config in config["lights"]:
            camera.add_light(**light_config)

    print(f"✓ Created {len(configs)} {group_name.lower()} cameras")


########################## build ##########################
n_envs = 1
scene.build(n_envs=n_envs)  # Build with 1 environment

########################## identify attached cameras ##########################
print("\n=== Identifying Attached Cameras ===")

# Identify cameras that are configured to be attached
attached_cameras = []
for group_name, configs in config_groups:
    for config in configs:
        if config["attachment"] is not None:
            camera = cameras[config["name"]]
            attached_cameras.append(camera)
            print(f"✓ {config['name']} is attached to sphere")

print(f"✓ Identified {len(attached_cameras)} attached cameras")

########################## simulate and render ##########################
os.makedirs("camera_sensor_output", exist_ok=True)

for i in range(100):
    scene.step()
    # Render every 10 steps
    if i % 10 == 0:
        print(f"\n--- Step {i} ---")

        # Read and print camera data
        camera_data = {}
        for cam_name, camera in cameras.items():
            # Read camera data (handles both single and multi-environment cases)
            data = camera.read()
            camera_data[cam_name] = data
            print(f"  {cam_name.replace('_', ' ').title()} RGB shape: {data.rgb.shape}")

        # Save images (always from environment 0 for visualization)
        for cam_name, data in camera_data.items():
            rgb_data = data.rgb[0] if data.rgb.ndim > 3 else data.rgb
            suffix = "_env0" if n_envs > 1 else ""
            filename = f"camera_sensor_output/{cam_name}{suffix}_step{i:03d}.png"
            plt.imsave(filename, tensor_to_array(rgb_data))
