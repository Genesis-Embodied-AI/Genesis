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
            camera_pos=(0.0, -2, 1.5),
            camera_lookat=(0.0, 0.0, 0.5),
            camera_fov=40,
            max_FPS=200,
        ),
        show_viewer=args.vis,
        rigid_options=gs.options.RigidOptions(
            enable_joint_limit=False,
            enable_collision=False,
            gravity=(0, 0, -0),
        ),
    )

    ########################## entities ##########################

    plane = scene.add_entity(
        gs.morphs.Plane(),
    )
    robot = scene.add_entity(
        gs.morphs.MJCF(file="xml/universal_robots_ur5e/ur5e.xml"),
    )

    target_entity = scene.add_entity(
        gs.morphs.Mesh(
            file="meshes/axis.obj",
            scale=0.10,
        ),
        surface=gs.surfaces.Default(color=(1, 0.5, 0.5, 1)),
    )
    ########################## build ##########################
    scene.build()

    robot.set_qpos([-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0.0])

    target_quat = np.array([0, 1, 0, 0])
    center = np.array([0.5, 0, 0.5])
    r = 0.1
    damping = 1e-4
    diag = damping * np.eye(6)

    ee_link = robot.get_link("ee_virtual_link")

    for i in range(0, 2000):
        target_pos = center + np.array([np.cos(i / 360 * np.pi), np.sin(i / 360 * np.pi), 0]) * r

        target_entity.set_qpos(np.concatenate([target_pos, target_quat]))

        # Position error.
        error_pos = target_pos - ee_link.get_pos().cpu().numpy()

        # Orientation error.
        ee_quat = ee_link.get_quat().cpu().numpy()
        error_quat = gs.transform_quat_by_quat(gs.inv_quat(ee_quat), target_quat)
        error_rotvec = gs.quat_to_rotvec(error_quat)

        error = np.concatenate([error_pos, error_rotvec])

        # jacobian
        jac = robot.get_jacobian(link=ee_link).cpu().numpy()
        dq = jac.T @ np.linalg.solve(jac @ jac.T + diag, error)
        q = robot.get_qpos().cpu().numpy() + dq

        # control
        robot.control_dofs_position(q, np.arange(6))
        scene.step()


if __name__ == "__main__":
    main()
