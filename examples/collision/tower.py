import numpy as np
import genesis as gs
import argparse

object_type = "duck"

parser = argparse.ArgumentParser()
parser.add_argument(
    "--object",
    type=str,
    default=object_type,
    choices=["sphere", "cylinder", "duck"],
)
args = parser.parse_args()
object_type = args.object

gs.init(backend=gs.cpu, precision="32")

scene = gs.Scene(
    viewer_options=gs.options.ViewerOptions(
        camera_pos=(20, -20, 20),
        camera_lookat=(0.0, 0.0, 5.0),
        camera_fov=30,
        max_FPS=60,
    ),
    sim_options=gs.options.SimOptions(
        dt=0.005,
    ),
    rigid_options=gs.options.RigidOptions(
        box_box_detection=False,
        max_collision_pairs=2000,
        use_gjk_collision=True,
        enable_mujoco_compatibility=False,
    ),
    show_viewer=True,
)

plane = scene.add_entity(gs.morphs.Plane(pos=(0, 0, 0)))

# create pyramid of boxes
box_width = 0.25
box_length = 2.0
box_height = 0.1
num_stacks = 50
height_offset = 0.0
for i in range(num_stacks):
    horizontal = i % 2 == 0

    if horizontal:
        box_size = np.array([box_width, box_length, box_height])
        box_pos_0 = (-0.4 * box_length, 0, i * (height_offset + box_size[2]) + 0.5 * box_size[2])
        box_pos_1 = (+0.4 * box_length, 0, i * (height_offset + box_size[2]) + 0.5 * box_size[2])
    else:
        box_size = np.array([box_length, box_width, box_height])
        box_pos_0 = (0, -0.4 * box_length, i * (height_offset + box_size[2]) + 0.5 * box_size[2])
        box_pos_1 = (0, +0.4 * box_length, i * (height_offset + box_size[2]) + 0.5 * box_size[2])

    scene.add_entity(
        gs.morphs.Box(size=box_size, pos=box_pos_0),
    )
    scene.add_entity(
        gs.morphs.Box(size=box_size, pos=box_pos_1),
    )

# Drop a huge mesh
if object_type == "duck":
    duck_scale = 1.0
    duck = scene.add_entity(
        morph=gs.morphs.Mesh(
            file="meshes/duck.obj",
            scale=duck_scale,
            pos=(0, 0, num_stacks * (height_offset + box_height) + 10 * duck_scale),
        ),
    )
elif object_type == "sphere":
    sphere_radius = 2.0
    scene.add_entity(
        morph=gs.morphs.Sphere(
            radius=sphere_radius, pos=(0.0, 0.0, num_stacks * (height_offset + box_height) + 5 * sphere_radius)
        ),
    )
elif object_type == "cylinder":
    cylinder_radius = 2.0
    cylinder_height = 1.0
    scene.add_entity(
        morph=gs.morphs.Cylinder(
            radius=cylinder_radius,
            height=cylinder_height,
            pos=(0.0, 0.0, num_stacks * (height_offset + box_height) + 5 * cylinder_height),
        ),
    )

scene.build()
for i in range(5000):
    scene.step()
