import argparse
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

    joints_name = (
        "joint1",
        "joint2",
        "joint3",
        "joint4",
        "joint5",
        "joint6",
        "joint7",
        "finger_joint1",
        "finger_joint2",
    )

    last_link_vel = None
    # PD control
    for i in range(500):
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


if __name__ == "__main__":
    main()
