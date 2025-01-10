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

    dofs_idx_local = []
    for v in robot.joints:
        if v.name in [
            "wrist_joint",
            "index_finger_joint1",
            "index_finger_join2",
            "index_finger_joint3",
        ]:
            dof_idx_local_v = v.dof_idx_local
            if isinstance(dof_idx_local_v, list):
                dofs_idx_local.extend(dof_idx_local_v)
            else:
                assert isinstance(dof_idx_local_v, int)
                dofs_idx_local.append(dof_idx_local_v)

    center = np.array([0.033, -0.01, 0.42])
    r1 = 0.05

    for i in range(2000):
        index_finger_pos = center + np.array([0, np.cos(i / 90 * np.pi) - 1.0, np.sin(i / 90 * np.pi) - 1.0]) * r1

        target_1.set_qpos(np.concatenate([index_finger_pos, target_quat]))

        qpos = robot.inverse_kinematics_multilink(
            links=[index_finger_distal],  # IK targets
            poss=[index_finger_pos],
            dofs_idx_local=dofs_idx_local,  # IK wrt these dofs
        )

        robot.set_qpos(qpos)
        scene.step()


if __name__ == "__main__":
    main()
