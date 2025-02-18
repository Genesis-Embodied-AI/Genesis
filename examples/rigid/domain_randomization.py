import argparse

import numpy as np
import torch

import genesis as gs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", default=False)
    args = parser.parse_args()

    ########################## init ##########################
    gs.init(seed=0, precision="32", logging_level="debug")

    ########################## create a scene ##########################
    scene = gs.Scene(
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.0, -2, 1.5),
            camera_lookat=(0.0, 0.0, 0.5),
            camera_fov=40,
            max_FPS=200,
        ),
        show_viewer=args.vis,
        rigid_options=gs.options.RigidOptions(
            dt=0.01,
            constraint_solver=gs.constraint_solver.Newton,
        ),
    )

    ########################## entities ##########################
    scene.add_entity(
        gs.morphs.Plane(),
    )
    robot = scene.add_entity(
        gs.morphs.URDF(
            file="urdf/go2/urdf/go2.urdf",
            pos=(0, 0, 0.4),
        ),
    )
    ########################## build ##########################
    n_envs = 8
    scene.build(n_envs=n_envs)

    ########################## domain randomization ##########################
    robot.set_friction_ratio(
        friction_ratio=0.5 + torch.rand(scene.n_envs, robot.n_links),
        ls_idx_local=np.arange(0, robot.n_links),
    )
    from IPython import embed

    embed()

    # set mass of a single link
    link = robot.get_link("RR_thigh")
    rigid = scene.sim.rigid_solver
    ori_mass = rigid.links_info.inertial_mass.to_numpy()
    print("original mass", link.get_mass(), ori_mass)
    link.set_mass(1)
    new_mass = rigid.links_info.inertial_mass.to_numpy()
    print("diff mass", new_mass - ori_mass)

    robot.set_mass_shift(
        mass_shift=-0.5 + torch.rand(scene.n_envs, robot.n_links),
        ls_idx_local=np.arange(0, robot.n_links),
    )
    robot.set_COM_shift(
        com_shift=-0.05 + 0.1 * torch.rand(scene.n_envs, robot.n_links, 3),
        ls_idx_local=np.arange(0, robot.n_links),
    )

    joint_names = [
        "FR_hip_joint",
        "FR_thigh_joint",
        "FR_calf_joint",
        "FL_hip_joint",
        "FL_thigh_joint",
        "FL_calf_joint",
        "RR_hip_joint",
        "RR_thigh_joint",
        "RR_calf_joint",
        "RL_hip_joint",
        "RL_thigh_joint",
        "RL_calf_joint",
    ]
    motor_dofs = [robot.get_joint(name).dof_idx_local for name in joint_names]

    robot.set_dofs_kp(np.full(12, 20), motor_dofs)
    robot.set_dofs_kv(np.full(12, 1), motor_dofs)
    default_dof_pos = np.array(
        [
            0.0,
            0.8,
            -1.5,
            0.0,
            0.8,
            -1.5,
            0.0,
            1.0,
            -1.5,
            0.0,
            1.0,
            -1.5,
        ]
    )
    robot.control_dofs_position(default_dof_pos, motor_dofs)

    for i in range(1000):
        scene.step()


if __name__ == "__main__":
    main()
