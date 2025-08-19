import argparse

import numpy as np
import genesis as gs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", default=False)
    parser.add_argument("-n", "--n_envs", type=int, default=4)
    args = parser.parse_args()

    ########################## init ##########################
    gs.init(backend=gs.gpu, precision="32")
    ########################## create a scene ##########################
    scene = gs.Scene(
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(3, -1, 1.5),
            camera_lookat=(0.0, 0.0, 0.5),
            camera_fov=30,
            max_FPS=60,
        ),
        sim_options=gs.options.SimOptions(
            dt=0.01,
        ),
        show_viewer=args.vis,
    )

    ########################## entities ##########################
    plane = scene.add_entity(
        gs.morphs.Plane(),
    )
    franka = scene.add_entity(
        gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"),
    )

    morph_heterogeneous = [
        gs.morphs.Box(
            size=(0.02, 0.02, 0.02),
            pos=(0.65, 0.0, 0.02),
        ),
        gs.morphs.Sphere(
            radius=0.015,
            pos=(0.65, 0.0, 0.02),
        ),
        gs.morphs.Sphere(
            radius=0.025,
            pos=(0.65, 0.0, 0.02),
        ),
    ]
    grasping_object = scene.add_entity(
        gs.morphs.Box(
            size=(0.04, 0.04, 0.04),
            pos=(0.65, 0.0, 0.02),
        ),
        # comment out this line turn off heterogeneous simulation
        morph_heterogeneous=morph_heterogeneous,
    )
    ########################## build ##########################
    scene.build(n_envs=args.n_envs, env_spacing=(1, 1))

    motors_dof = np.arange(7)
    fingers_dof = np.arange(7, 9)
    l_qpos = [-1.0124, 1.5559, 1.3662, -1.6878, -1.5799, 1.7757, 1.4602, 0.04, 0.04]
    if args.n_envs == 0:
        franka.set_qpos(np.array(l_qpos))
    else:
        franka.set_qpos(np.array([l_qpos] * args.n_envs))
    scene.step()

    end_effector = franka.get_link("hand")
    qpos = franka.inverse_kinematics(
        link=end_effector,
        pos=np.array([[0.65, 0.0, 0.135]] * args.n_envs),
        quat=np.array([[0, 1, 0, 0]] * args.n_envs),
    )
    print(qpos.shape, motors_dof.shape)
    franka.control_dofs_position(qpos[..., :-2], motors_dof)

    # hold
    for i in range(100):
        print("hold", i)
        scene.step()

    # grasp
    finder_pos = -0.0
    for i in range(100):
        print("grasp", i)
        franka.control_dofs_position(qpos[..., :-2], motors_dof)
        franka.control_dofs_position(np.array([[finder_pos, finder_pos]] * args.n_envs), fingers_dof)
        scene.step()

    # lift
    qpos = franka.inverse_kinematics(
        link=end_effector,
        pos=np.array([[0.65, 0.0, 0.3]] * args.n_envs),
        quat=np.array([[0, 1, 0, 0]] * args.n_envs),
    )
    for i in range(200):
        print("lift", i)
        franka.control_dofs_position(qpos[..., :-2], motors_dof)
        franka.control_dofs_position(np.array([[finder_pos, finder_pos]] * args.n_envs), fingers_dof)
        scene.step()


if __name__ == "__main__":
    main()
