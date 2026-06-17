"""Many independent boxes on a plane, solved with the CPU islands + sparse-Jacobian composition.

Each well-separated box is its own 6-dof island, so the per-island solve factorizes each block independently
while the sparse Jacobian keeps the per-iteration products O(nonzeros). The step time then scales near-linearly
in the number of boxes. Run on the CPU backend so the two compose (on GPU sparse is dropped and islands run
alone). Prints the live step rate in the console while the viewer is open.
"""

import math
import time

import numpy as np

import genesis as gs


def main():
    n_boxes = 1500

    gs.init(backend=gs.cpu, performance_mode=True)

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=1.0 / 60.0,
            gravity=(0.0, 0.0, -9.8),
        ),
        rigid_options=gs.options.RigidOptions(
            use_contact_island=True,
            sparse_solve=True,
            max_collision_pairs=10000,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(28.0, 28.0, 22.0),
            camera_lookat=(8.0, 8.0, 0.0),
            camera_fov=40,
        ),
        vis_options=gs.options.VisOptions(
            show_world_frame=False,
        ),
        show_viewer=True,
    )

    scene.add_entity(gs.morphs.Plane())

    rng = np.random.default_rng(0)
    n_side = math.ceil(math.sqrt(n_boxes))
    spacing = 0.5
    placed = 0
    for i in range(n_side):
        for j in range(n_side):
            if placed >= n_boxes:
                break
            euler = tuple(rng.uniform(-180.0, 180.0, size=3))
            scene.add_entity(
                gs.morphs.Box(
                    size=(0.1, 0.1, 0.1),
                    pos=(i * spacing, j * spacing, 1.5),
                    euler=euler,
                ),
                vis_mode="collision",
            )
            placed += 1

    scene.build(n_envs=1)

    solver = scene.rigid_solver
    print(
        f"n_boxes={placed} n_dofs={solver.n_dofs} "
        f"islands={solver._use_contact_island} sparse={solver.constraint_solver.sparse_solve}"
    )

    t_window = time.perf_counter()
    for i in range(1_000_000):
        scene.step()
        if (i + 1) % 60 == 0:
            now = time.perf_counter()
            per_step = (now - t_window) / 60.0
            t_window = now
            print(f"step {i + 1}: {per_step * 1e3:.2f} ms/step ({1.0 / per_step:.0f} steps/s)")


if __name__ == "__main__":
    main()
