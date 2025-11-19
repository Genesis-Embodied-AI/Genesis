"""
Minimal example for testing camera sensor implementation.
Starts with just rasterizer backend.
"""

import genesis as gs
from genesis.options.sensors import RasterizerCameraOptions
from genesis.utils.image_exporter import FrameImageExporter

########################## init ##########################
gs.init(seed=0, precision="32", backend=gs.cpu, logging_level="info")

########################## create a scene ##########################
scene = gs.Scene(
    show_viewer=False,  # No viewer - cameras are independent
)

########################## entities ##########################
plane = scene.add_entity(gs.morphs.Plane())
sphere = scene.add_entity(
    gs.morphs.Sphere(pos=(0.0, 0.0, 2.0), radius=1.0),
    surface=gs.surfaces.Smooth(color=(1.0, 0.5, 0.5)),
)

########################## cameras as sensors ##########################
cam0 = scene.add_sensor(
    RasterizerCameraOptions(
        res=(512, 512),
        pos=(3.5, 0.0, 1.5),
        lookat=(0.0, 0.0, 0.7),
        up=(0.0, 0.0, 1.0),
        fov=60.0,
    )
)

cam1 = scene.add_sensor(
    RasterizerCameraOptions(
        res=(256, 256),
        pos=(0.0, 3.5, 1.5),
        lookat=(0.0, 0.0, 0.7),
        up=(0.0, 0.0, 1.0),
        fov=45.0,
    )
)

# Add lights
cam0.add_light(
    pos=(2.0, 2.0, 5.0),
    color=(1.0, 1.0, 1.0),
    intensity=5.0,
)

########################## build ##########################
scene.build()

########################## simulate and render ##########################
exporter = FrameImageExporter("camera_sensor_output")

for t in range(10):
    scene.step()

    # Render and read camera 0
    cam0.render()
    data0 = cam0.read()
    print(f"Step {t}, Cam0 RGB shape: {data0.rgb.shape}")
    exporter.export_frame_single_camera(t, 0, rgb=data0.rgb)

    # Render and read camera 1
    cam1.render()
    data1 = cam1.read()
    print(f"Step {t}, Cam1 RGB shape: {data1.rgb.shape}")
    exporter.export_frame_single_camera(t, 1, rgb=data1.rgb)

print("Done! Check camera_sensor_output/ for images")
