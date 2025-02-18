import argparse

import genesis as gs


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", default=False)
    parser.add_argument("-c", "--cpu", action="store_true", default=False)
    parser.add_argument("-s", "--sep", action="store_true", default=False)
    parser.add_argument("-r", "--record", action="store_true", default=False)
    parser.add_argument("-n", "--num_env", type=int, default=1)
    args = parser.parse_args()

    ########################## init ##########################
    gs.init(backend=gs.cpu if args.cpu else gs.gpu)

    ########################## create a scene ##########################
    scene = gs.Scene(
        vis_options=gs.options.VisOptions(
            plane_reflection=False,
            n_rendered_envs=args.num_env,
            env_separate_rigid=args.sep,
            show_world_frame=False,
            show_link_frame=False,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(3.5, 0.0, 2.5),
            camera_lookat=(0.0, 0.0, 0.5),
            camera_fov=40,
        ),
        show_viewer=args.vis,
        rigid_options=gs.options.RigidOptions(
            # constraint_solver=gs.constraint_solver.Newton,
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

    ########################## cameras ##########################
    cam_0 = scene.add_camera(
        res=(1280, 960),
        pos=(3.5, 0.0, 2.5),
        lookat=(0, 0, 0.5),
        fov=30,
        GUI=True,
    )
    ########################## build ##########################
    scene.build(n_envs=args.num_env, env_spacing=(0.5, 0.5))

    if args.record:
        cam_0.start_recording()

    horizon = 1000
    for i in range(horizon):
        scene.step()

        color, depth, seg, normal = cam_0.render(
            rgb=True, depth=True, segmentation=True, colorize_seg=True, normal=True
        )
        print(f"Step {i}:", args.num_env, color.shape, depth.shape, seg.shape, normal.shape)

    if args.record:
        cam_0.stop_recording(save_to_filename="video.mp4")


if __name__ == "__main__":
    main()
