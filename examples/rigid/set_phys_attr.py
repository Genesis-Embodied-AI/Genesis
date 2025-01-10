import argparse

import numpy as np

import genesis as gs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", default=False)
    args = parser.parse_args()

    ########################## init ##########################
    gs.init(backend=gs.gpu)

    ########################## create a scene ##########################
    viewer_options = gs.options.ViewerOptions(
        camera_pos=(0, -3.5, 2.5),
        camera_lookat=(0.0, 0.0, 0.5),
        camera_fov=40,
        max_FPS=60,
    )

    scene = gs.Scene(
        viewer_options=viewer_options,
        sim_options=gs.options.SimOptions(
            dt=0.01,
        ),
        show_viewer=args.vis,
        rigid_options=gs.options.RigidOptions(
            # NOTE: Batching dofs/links info to set different physical attributes across environments (in parallel)
            #       By default, both are False as it's faster and thus only turn this on if necessary
            batch_dofs_info=True,
            batch_links_info=True,
        ),
    )

    ########################## entities ##########################
    plane = scene.add_entity(
        gs.morphs.Plane(),
    )
    franka = scene.add_entity(
        gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"),
    )
    ########################## build ##########################
    scene.build(n_envs=2)  # test with 2 different environments

    jnt_names = [
        "joint1",
        "joint2",
        "joint3",
        "joint4",
        "joint5",
        "joint6",
        "joint7",
        "finger_joint1",
        "finger_joint2",
    ]
    dofs_idx = [franka.get_joint(name).dof_idx_local for name in jnt_names]

    lnk_names = [
        "link0",
        "link1",
        "link2",
        "link3",
        "link4",
        "link5",
        "link6",
        "link7",
        "hand",
        "left_finger",
        "right_finger",
    ]
    links_idx = [franka.get_link(name).idx_local for name in lnk_names]

    # Optional: set control gains
    franka.set_dofs_kp(
        np.array(
            [
                [4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100],
                [100, 100, 2000, 2000, 2000, 3500, 3500, 4500, 4500],
            ]
        ),
        dofs_idx,
    )
    print("=== kp ===\n", franka.get_dofs_kp())
    franka.set_dofs_kv(
        np.array(
            [
                [450, 450, 350, 350, 200, 200, 200, 10, 10],
                [10, 10, 200, 200, 200, 350, 350, 450, 450],
            ]
        ),
        dofs_idx,
    )
    print("=== kv ===\n", franka.get_dofs_kv())
    franka.set_dofs_force_range(
        np.array(
            [
                [-87, -87, -87, -87, -12, -12, -12, -100, -100],
                [-120, -100, -12, -12, -12, -87, -87, -87, -87],
            ]
        ),
        np.array(
            [
                [87, 87, 87, 87, 12, 12, 12, 100, 100],
                [100, 100, 12, 12, 12, 87, 87, 87, 87],
            ]
        ),
        dofs_idx,
    )
    print("=== force range ===\n", franka.get_dofs_force_range())
    franka.set_dofs_armature(
        np.array(
            [
                [0.1] * len(dofs_idx),
                [0.2] * len(dofs_idx),
            ]
        ),
        dofs_idx,
    )
    print("=== armature ===\n", franka.get_dofs_armature())
    franka.set_dofs_stiffness(
        np.array(
            [
                [0.0] * len(dofs_idx),
                [0.1] * len(dofs_idx),
            ]
        ),
        dofs_idx,
    )
    print("=== stiffness ===\n", franka.get_dofs_stiffness())
    franka.set_dofs_invweight(
        np.array(
            [
                [5.5882, 0.9693, 6.8053, 3.9007, 7.8085, 6.6139, 9.4213, 8.6984, 8.6984],
                [8.6984, 8.6984, 9.4213, 6.6139, 7.8085, 3.9007, 6.8053, 0.9693, 5.5882],
            ]
        ),
        dofs_idx,
    )
    print("=== invweight ===\n", franka.get_dofs_invweight())
    franka.set_dofs_damping(
        np.array(
            [
                [1.0] * len(dofs_idx),
                [2.0] * len(dofs_idx),
            ]
        ),
        dofs_idx,
    )
    print("=== damping ===\n", franka.get_dofs_damping())
    franka.set_links_inertial_mass(
        np.array(
            [
                [0.6298, 4.9707, 0.6469, 3.2286, 3.5879, 1.2259, 1.6666, 0.7355, 0.7300, 0.0150, 0.0150],
                [0.015, 0.015, 0.73, 0.7355, 1.6666, 1.2259, 3.5879, 3.2286, 0.6469, 4.9707, 0.6298],
            ]
        ),
        links_idx,
    )
    print("=== links inertial mass ===\n", franka.get_links_inertial_mass())
    franka.set_links_invweight(
        np.array(
            [
                [0.0, 3.6037e-05, 0.00030664, 0.025365, 0.036351, 0.072328, 0.089559, 0.11661, 0.11288, 3.0179, 3.0179],
                [3.0179, 3.0179, 0.11288, 0.11661, 0.089559, 0.072328, 0.036351, 0.025365, 0.00030664, 3.6037e-05, 0.0],
            ]
        ),
        links_idx,
    )
    print("=== links invweight ===\n", franka.get_links_invweight())

    # Hard reset
    for i in range(150):
        if i < 50:
            franka.set_dofs_position(
                np.array([1, 1, 0, 0, 0, 0, 0, 0.04, 0.04])[None, :].repeat(scene.n_envs, 0), dofs_idx
            )
        elif i < 100:
            franka.set_dofs_position(
                np.array([-1, 0.8, 1, -2, 1, 0.5, -0.5, 0.04, 0.04])[None, :].repeat(scene.n_envs, 0), dofs_idx
            )
        else:
            franka.set_dofs_position(np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])[None, :].repeat(scene.n_envs, 0), dofs_idx)

        scene.step()

    # PD control
    for i in range(1250):
        if i == 0:
            franka.control_dofs_position(
                np.array([1, 1, 0, 0, 0, 0, 0, 0.04, 0.04])[None, :].repeat(scene.n_envs, 0),
                dofs_idx,
            )
        elif i == 250:
            franka.control_dofs_position(
                np.array([-1, 0.8, 1, -2, 1, 0.5, -0.5, 0.04, 0.04])[None, :].repeat(scene.n_envs, 0),
                dofs_idx,
            )
        elif i == 500:
            franka.control_dofs_position(
                np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])[None, :].repeat(scene.n_envs, 0),
                dofs_idx,
            )
        elif i == 750:
            # control first dof with velocity, and the rest with position
            franka.control_dofs_position(
                np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])[1:][None, :].repeat(scene.n_envs, 0),
                dofs_idx[1:],
            )
            franka.control_dofs_velocity(
                np.array([1.0, 0, 0, 0, 0, 0, 0, 0, 0])[:1][None, :].repeat(scene.n_envs, 0),
                dofs_idx[:1],
            )
        elif i == 1000:
            franka.control_dofs_force(
                np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])[None, :].repeat(scene.n_envs, 0),
                dofs_idx,
            )
        # This is the internal control force computed based on the given control command
        # If using force control, it's the same as the given control command
        print("control force:", franka.get_dofs_control_force(dofs_idx))

        scene.step()


if __name__ == "__main__":
    main()
