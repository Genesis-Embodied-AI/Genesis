import time
import argparse
import numpy as np
import genesis as gs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", default=False)
    args = parser.parse_args()

    ########################## init ##########################
    gs.init(backend=gs.gpu)
    sim_dt = 0.01
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
            dt=sim_dt,
        ),
        show_viewer=args.vis,
    )

    ########################## entities ##########################
    plane = scene.add_entity(
        gs.morphs.Plane(),
    )
    franka = scene.add_entity(
        gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"),
        # material=gs.materials.Rigid(gravity_compensation=0.),
    )
    ########################## build ##########################
    scene.build()

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

    # Optional: set control gains
    # franka.set_dofs_kp(
    #     np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100]),
    #     dofs_idx,
    # )
    # franka.set_dofs_kv(
    #     np.array([450, 450, 350, 350, 200, 200, 200, 10, 10]),
    #     dofs_idx,
    # )
    # franka.set_dofs_force_range(
    #     np.array([-87, -87, -87, -87, -12, -12, -12, -100, -100]),
    #     np.array([87, 87, 87, 87, 12, 12, 12, 100, 100]),
    #     dofs_idx,
    # )

    last_link_vel = None
    # PD control
    for i in range(1250):
        # if i == 0:
        #     franka.control_dofs_position(
        #         np.array([1, 1, 0, 0, 0, 0, 0, 0.04, 0.04]),
        #         dofs_idx,
        #     )
        # elif i == 250:
        #     franka.control_dofs_position(
        #         np.array([-1, 0.8, 1, -2, 1, 0.5, -0.5, 0.04, 0.04]),
        #         dofs_idx,
        #     )
        # elif i == 500:
        #     franka.control_dofs_position(
        #         np.array([0, 0, 0, 0, 0, 0, 0, 0, 0]),
        #         dofs_idx,
        #     )
        # elif i == 750:
        #     # control first dof with velocity, and the rest with position
        #     franka.control_dofs_position(
        #         np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])[1:],
        #         dofs_idx[1:],
        #     )
        #     franka.control_dofs_velocity(
        #         np.array([1.0, 0, 0, 0, 0, 0, 0, 0, 0])[:1],
        #         dofs_idx[:1],
        #     )
        # elif i == 1000:
        #     franka.control_dofs_force(
        #         np.array([0, 0, 0, 0, 0, 0, 0, 0, 0]),
        #         dofs_idx,
        #     )
        # This is the internal control force computed based on the given control command
        # If using force control, it's the same as the given control command
        # print("control force:", franka.get_dofs_control_force(dofs_idx))

        scene.step()

        rigid = scene.sim.rigid_solver

        links_acc = franka.get_links_acc()
        links_pos = franka.get_links_pos()
        scene.clear_debug_objects()

        _link_vel = rigid.links_state.cd_vel.to_numpy()[:, 0]
        _link_acc = rigid.links_state.cdd_vel.to_numpy()[:, 0]
        if last_link_vel is not None:
            finite_diff_acc = (_link_vel - last_link_vel) / sim_dt
        for i in range(links_acc.shape[0]):
            link_pos = links_pos[i]
            link_acc = links_acc[i]
            # link_acc = link_acc / link_acc.norm() * 0.1

            scene.draw_debug_arrow(
                pos=link_pos.tolist(),
                vec=link_acc.tolist(),
            )

            if last_link_vel is not None:
                scene.draw_debug_arrow(
                    pos=link_pos.tolist(),
                    vec=finite_diff_acc[i],
                    color=(0, 1, 0),
                )

        last_link_vel = _link_vel

        time.sleep(0.1)


if __name__ == "__main__":
    main()
