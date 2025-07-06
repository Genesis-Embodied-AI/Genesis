import numpy as np
import genesis as gs
import argparse

pile_type = "static"
num_cubes = 5

parser = argparse.ArgumentParser()
parser.add_argument("--pile_type", type=str, default=pile_type, choices=["static", "falling"])
parser.add_argument("--num_cubes", type=int, default=num_cubes, choices=range(5, 11))
parser.add_argument("--cpu", action="store_true", help="Use CPU backend instead of GPU")
args = parser.parse_args()

pile_type = args.pile_type
num_cubes = args.num_cubes
cpu = args.cpu
backend = gs.cpu if cpu else gs.gpu

gs.init(backend=backend, precision="32")

scene = gs.Scene(
    viewer_options=gs.options.ViewerOptions(
        camera_pos=(0, -5.5, 2.5),
        camera_lookat=(0, 0.0, 1.5),
        camera_fov=30,
        max_FPS=60,
    ),
    sim_options=gs.options.SimOptions(
        dt=0.01,
    ),
    rigid_options=gs.options.RigidOptions(
        box_box_detection=False,
        max_collision_pairs=1000,
        use_gjk_collision=True,
        enable_mujoco_compatibility=False,
    ),
    show_viewer=True,
)

plane = scene.add_entity(gs.morphs.Plane(pos=(0, 0, 0)))

# create pyramid of boxes
box_size = 0.25
if pile_type == "static":
    box_spacing = box_size
else:
    box_spacing = 1.1 * box_size
vec_one = np.array([1.0, 1.0, 1.0])
box_pos_offset = (0 - 0.5, 1, 0.0) + 0.5 * box_size * vec_one
boxes = {}
for i in range(num_cubes):
    for j in range(num_cubes - i):
        box = scene.add_entity(
            gs.morphs.Box(size=box_size * vec_one, pos=box_pos_offset + box_spacing * np.array([i + 0.5 * j, 0, j])),
        )

scene.build()

for i in range(1000):
    scene.step()
