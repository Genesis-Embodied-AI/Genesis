import genesis as gs
import argparse
import time


def main_equality_connect():
    gs.init(backend=gs.cpu)

    scene = gs.Scene(
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(10, 0, 10),
            camera_lookat=(0.0, 0.0, 3),
            camera_fov=60,
        ),
    )
    franka = scene.add_entity(
        gs.morphs.MJCF(
            file="xml/four_bar_linkage.xml",
        ),
    )
    scene.build()
    for i in range(1000):
        scene.step()


def main_equality_weld():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", default=False)
    args = parser.parse_args()
    ########################## init ##########################
    gs.init(backend=gs.cpu)
    scene = gs.Scene(
        show_viewer=args.vis,
    )
    ########################## entities ##########################
    robot1 = scene.add_entity(
        gs.morphs.MJCF(file="xml/four_bar_linkage_weld.xml"),
    )

    ########################## build ##########################
    scene.build()

    rigid = scene.sim.rigid_solver
    qpos = rigid.qpos.to_numpy()[:, 0]
    qpos[0], qpos[1], qpos[2] = 0.2, 0.2, 0.2
    rigid.qpos.from_numpy(qpos[:, None])

    for i in range(1000):
        time.sleep(1)
        scene.step()


if __name__ == "__main__":
    main_equality_weld()
    # main_equality_connect()
