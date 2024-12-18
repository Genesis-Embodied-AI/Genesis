import argparse

import numpy as np

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
            camera_pos=(2.5, 0.0, 1.5),
            camera_lookat=(0.0, 0.0, 0.5),
            camera_fov=40,
        ),
        show_viewer=args.vis,
        rigid_options=gs.options.RigidOptions(
            gravity=(0, 0, 0),
            enable_collision=False,
            enable_joint_limit=False,
        ),
    )

    target_1 = scene.add_entity(
        gs.morphs.Mesh(
            file="meshes/axis.obj",
            scale=0.05,
        ),
        surface=gs.surfaces.Default(color=(1, 0.5, 0.5, 1)),
    )

    target_2 = scene.add_entity(
        gs.morphs.Mesh(
            file="meshes/axis.obj",
            scale=0.05,
        ),
        surface=gs.surfaces.Default(color=(0.5, 1.0, 0.5, 1)),
    )
    target_3 = scene.add_entity(
        gs.morphs.Mesh(
            file="meshes/axis.obj",
            scale=0.05,
        ),
        surface=gs.surfaces.Default(color=(0.5, 0.5, 1.0, 1)),
    )
    ########################## entities ##########################
    robot = scene.add_entity(
        morph=gs.morphs.URDF(
            scale=1.0,
            file="urdf/shadow_hand/shadow_hand.urdf",
        ),
        surface=gs.surfaces.Reflective(color=(0.4, 0.4, 0.4)),
    )

    ########################## build ##########################
    scene.build()
    scene.reset()

    target_quat = np.array([1, 0, 0, 0])
    index_finger_distal = robot.get_link("index_finger_distal")
    middle_finger_distal = robot.get_link("middle_finger_distal")
    forearm = robot.get_link("forearm")

    center = np.array([0.5, 0.5, 0.2])
    r1 = 0.1
    r2 = 0.13

    for i in range(2000):
        index_finger_pos = center + np.array([np.cos(i / 90 * np.pi), np.sin(i / 90 * np.pi), 0]) * r1
        middle_finger_pos = center + np.array([np.cos(i / 90 * np.pi), np.sin(i / 90 * np.pi), 0]) * r2
        forearm_pos = index_finger_pos - np.array([0, 0, 0.40])

        target_1.set_qpos(np.concatenate([index_finger_pos, target_quat]))
        target_2.set_qpos(np.concatenate([middle_finger_pos, target_quat]))
        target_3.set_qpos(np.concatenate([forearm_pos, target_quat]))

        qpos = robot.inverse_kinematics_multilink(
            links=[index_finger_distal, middle_finger_distal, forearm],
            poss=[index_finger_pos, middle_finger_pos, forearm_pos],
        )

        robot.set_qpos(qpos)
        scene.step()


if __name__ == "__main__":
    main()
