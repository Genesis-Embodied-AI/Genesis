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


class TestRigidSpeed(unittest.TestCase):
    def test_speed(self):
        n_envs = 30000
        n_frame_fps = 10

        def run_anymal_c(solver):
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
                ),
            )

            ########################## entities ##########################
            scene.add_entity(
                gs.morphs.Plane(),
            )
            robot = scene.add_entity(
                gs.morphs.URDF(
                    file="urdf/anymal_c/urdf/anymal_c.urdf",
                    pos=(0, 0, 0.8),
                ),
            )
            ########################## build ##########################
            scene.build(n_envs=n_envs)

            joint_names = [
                "RH_HAA",
                "LH_HAA",
                "RF_HAA",
                "LF_HAA",
                "RH_HFE",
                "LH_HFE",
                "RF_HFE",
                "LF_HFE",
                "RH_KFE",
                "LH_KFE",
                "RF_KFE",
                "LF_KFE",
            ]
            motor_dofs = [robot.get_joint(name).dof_idx_local for name in joint_names]

            robot.set_dofs_kp(np.full(12, 1000), motor_dofs)
            if n_envs > 0:
                robot.control_dofs_position(np.zeros((n_envs, 12)), motor_dofs)
            else:
                robot.control_dofs_position(np.zeros(12), motor_dofs)

            vec_fps = []
            for i in range(1000):
                scene.step()
                vec_fps.append(scene.FPS_tracker.total_fps)

            total_fps = 1.0 / (1.0 / np.array(vec_fps[-n_frame_fps:])).mean()
            result = f"anymal \t| {solver} \t| {total_fps:,.2f} fps \t| {n_envs} envs"
            gs.destroy()
            return result

        def run_batched_franka(solver):
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
                ),
            )

            ########################## entities ##########################
            plane = scene.add_entity(
                gs.morphs.Plane(),
            )
            franka = scene.add_entity(
                gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"),
                visualize_contact=True,
            )

            ########################## build ##########################
            scene.build(n_envs=n_envs, env_spacing=(1.0, 1.0))

            vec_fps = []
            for i in range(1000):
                scene.step()
                vec_fps.append(scene.FPS_tracker.total_fps)

            total_fps = 1.0 / (1.0 / np.array(vec_fps[-n_frame_fps:])).mean()
            result = f"frank \t| {solver} \t| {total_fps:,.2f} fps \t| {n_envs} envs"

            gs.destroy()
            return result

        def run_random(solver):
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
                ),
            )

            ########################## entities ##########################
            plane = scene.add_entity(
                gs.morphs.Plane(),
            )

            robot = scene.add_entity(
                gs.morphs.URDF(
                    file="urdf/anymal_c/urdf/anymal_c.urdf",
                    pos=(0, 0, 0.8),
                ),
                visualize_contact=True,
            )

            ########################## build ##########################
            scene.build(n_envs=n_envs, env_spacing=(1.0, 1.0))

            vec_fps = []
            robot.set_dofs_kp(np.full(12, 1000), np.arange(6, 18))
            dofs = torch.arange(6, 18).cuda()
            robot.control_dofs_position(torch.zeros((n_envs, 12), device="cuda"), dofs)
            for i in range(1000):
                robot.control_dofs_position(torch.rand((n_envs, 12), device="cuda") * 0.1 - 0.05, dofs)
                scene.step()
                vec_fps.append(scene.FPS_tracker.total_fps)

            total_fps = 1.0 / (1.0 / np.array(vec_fps[-n_frame_fps:])).mean()
            result = f"random \t| {solver} \t| {total_fps:,.2f} fps \t| {n_envs} envs"
            gs.destroy()
            return result

        funcs = [run_random, run_anymal_c, run_batched_franka]
        solvers = [gs.constraint_solver.CG, gs.constraint_solver.Newton]
        results = []

        for func in funcs:
            for solver in solvers:
                string = func(solver)
                print(string)
                results.append(string)

        with open("speed_test.txt", "a") as file:
            for string in results:
                file.write(string + "\n")


if __name__ == "__main__":
    unittest.main()
