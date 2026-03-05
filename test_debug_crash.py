import numpy as np
import genesis as gs

gs.init(backend=gs.gpu, logging_level="info")

scene = gs.Scene(
    rigid_options=gs.options.RigidOptions(dt=0.01),
    show_viewer=False,
    show_FPS=False,
)

scene.add_entity(gs.morphs.Plane())
n_cubes = 6
box_size = 0.25
box_spacing = (1.0 - 1e-3) * box_size
box_pos_offset = (-0.5, 1.0, 0.0) + 0.5 * np.array([box_size, box_size, box_size])
for i in range(n_cubes):
    for j in range(n_cubes - i):
        scene.add_entity(
            gs.morphs.Box(
                size=[box_size, box_size, box_size],
                pos=box_pos_offset + box_spacing * np.array([i + 0.5 * j, 0.0, j]),
            ),
        )

n_envs = 4096
scene.build(n_envs=n_envs)

import torch
import quadrants as qd

for i in range(10000):
    scene.step()
    if i % 100 == 0:
        qd.sync()
        print(f"Step {i} ok", flush=True)
