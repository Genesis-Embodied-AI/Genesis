import argparse
import numpy as np

import genesis as gs
from genesis.utils.geom import trans_to_T
from genesis.utils.image_exporter import FrameImageExporter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", default=False)
    parser.add_argument("-c", "--cpu", action="store_true", default=False)
    parser.add_argument("-b", "--n_envs", type=int, default=3)
    parser.add_argument("-s", "--n_steps", type=int, default=2)
    parser.add_argument("-r", "--render_all_cameras", action="store_true", default=False)
    parser.add_argument("-o", "--output_dir", type=str, default="data/test")
    parser.add_argument("-u", "--use_rasterizer", action="store_true", default=False)
    parser.add_argument("-d", "--debug", action="store_true", default=False)
    parser.add_argument("-l", "--seg_level", type=str, default="link")
    args = parser.parse_args()

    ########################## init ##########################
    gs.init(backend=gs.cpu if args.cpu else gs.gpu)

    ########################## create a scene ##########################
    scene = gs.Scene(
        vis_options=gs.options.VisOptions(
            segmentation_level=args.seg_level,
        ),
        renderer=gs.options.renderers.BatchRenderer(
            use_rasterizer=args.use_rasterizer,
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
    debug_cam = scene.add_camera(
        res=(720, 1280),
        pos=(1.5, -0.5, 1.0),
        lookat=(0.0, 0.0, 0.5),
        fov=60,
        GUI=args.vis,
        debug=True,
    )
    cam_0 = scene.add_camera(
        res=(512, 512),
        pos=(1.5, 0.5, 1.5),
        lookat=(0.0, 0.0, 0.5),
        fov=45,
        GUI=args.vis,
    )
    cam_0.attach(franka.links[6], trans_to_T(np.array([0.0, 0.5, 0.0])))
    cam_1 = scene.add_camera(
        res=(512, 512),
        pos=(1.5, -0.5, 1.5),
        lookat=(0.0, 0.0, 0.5),
        fov=45,
        GUI=args.vis,
    )
    cam_2 = scene.add_camera(
        res=(512, 512),
        pos=(0.0, 0.1, 5.0),
        lookat=(0.0, 0.0, 0.0),
        fov=45,
        GUI=args.vis,
    )

    scene.add_light(
        pos=(0.0, 0.0, 1.5),
        dir=(1.0, 1.0, -2.0),
        color=(1.0, 0.0, 0.0),
        directional=True,
        castshadow=True,
        cutoff=45.0,
        intensity=0.5,
    )
    scene.add_light(
        pos=(4, -4, 4),
        dir=(0, 0, -1),
        directional=False,
        castshadow=True,
        cutoff=80.0,
        intensity=1.0,
        attenuation=0.1,
    )

    ########################## build ##########################
    scene.build(n_envs=args.n_envs)

    # Create an image exporter
    exporter = FrameImageExporter(args.output_dir)

    if args.debug:
        debug_cam.start_recording()
    for i in range(args.n_steps):
        scene.step()
        if args.debug:
            debug_cam.render()
        if args.render_all_cameras:
            color, depth, seg, normal = scene.render_all_cameras(
                rgb=True, depth=i % 2 == 1, segmentation=i % 2 == 1, normal=True
            )
            exporter.export_frame_all_cameras(i, rgb=color, depth=depth, segmentation=seg, normal=normal)
        else:
            color, depth, seg, normal = cam_1.render(
                rgb=False,
                depth=True,
                segmentation=True,
                colorize_seg=True,
                normal=False,
            )
            exporter.export_frame_single_camera(i, cam_1.idx, rgb=seg, depth=depth, segmentation=None, normal=normal)
    if args.debug:
        debug_cam.stop_recording("debug_cam.mp4")


if __name__ == "__main__":
    main()
