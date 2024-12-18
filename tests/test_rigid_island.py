import random
import unittest

import numpy as np
import torch

import genesis as gs

seed = 0
torch.manual_seed(seed)

# If using CUDA
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU

# Set seed for other libraries like numpy and random
np.random.seed(seed)
random.seed(seed)


class TestRigidISLAND(unittest.TestCase):
    def test_island(self):
        n_frame_fps = 10

        def run_cubes(solver, n_envs, n_cubes, is_island):
            gs.init(backend=gs.gpu, logging_level="warning", precision="32", seed=0)

            scene = gs.Scene(
                show_viewer=False,
                viewer_options=gs.options.ViewerOptions(
                    camera_pos=(3.5, 0.0, 2.5),
                    camera_lookat=(0.0, 0.0, 0.5),
                    camera_fov=40,
                ),
                rigid_options=gs.options.RigidOptions(
                    dt=0.01,
                    constraint_solver=solver,
                    use_contact_island=is_island,
                ),
            )

            ########################## entities ##########################
            plane = scene.add_entity(
                gs.morphs.Plane(),
            )
            # cube = scene.add_entity(
            #     gs.morphs.MJCF(file='xml/one_box.xml'),
            #     visualize_contact=True,
            # )

            for i in range(n_cubes):
                cube = scene.add_entity(
                    gs.morphs.Box(
                        size=(0.1, 0.1, 0.1),
                        pos=(0.0, 0.2 * i, 0.045),
                    ),
                )

            ########################## build ##########################
            scene.build(n_envs=n_envs)

            vec_fps = []
            for i in range(1000):
                scene.step()
                vec_fps.append(scene.FPS_tracker.total_fps)

            total_fps = 1.0 / (1.0 / np.array(vec_fps[-n_frame_fps:])).mean()
            result = f"{is_island} island \t| {n_cubes} cubes \t| {solver} \t| {total_fps:,.2f} fps \t| {n_envs} envs"

            gs.destroy()
            return result

        solvers = [gs.constraint_solver.CG, gs.constraint_solver.Newton]
        results = []

        n_envs = 8192
        for solver in solvers:
            for n_cubes in [1, 10]:
                for is_island in [True, False]:
                    string = run_cubes(solver, n_envs, n_cubes, is_island)
                    print(string)
                    results.append(string)

        with open("speed_test.txt", "a") as file:
            for string in results:
                file.write(string + "\n")


if __name__ == "__main__":
    unittest.main()
