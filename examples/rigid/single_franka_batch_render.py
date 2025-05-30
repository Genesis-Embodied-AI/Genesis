import argparse
import numpy as np

import genesis as gs
from genesis.options.renderers import BatchRenderer
from genesis.utils.geom import trans_to_T
from genesis.utils.image_exporter import FrameImageExporter

N_ENVS = 3
N_STEPS = 2
IS_RENDER_ALL_CAMERAS = True
OUTPUT_DIR = "img_output/test"


def main():
    global N_ENVS, N_STEPS, IS_RENDER_ALL_CAMERAS, OUTPUT_DIR

    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", default=False)
    parser.add_argument("-c", "--cpu", action="store_true", default=False)
    args = parser.parse_args()

    ########################## init ##########################
    gs.init(backend=gs.cpu if args.cpu else gs.gpu)

    ########################## create a scene ##########################
    scene = gs.Scene(
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(3.5, 0.0, 2.5),
            camera_lookat=(0.0, 0.0, 0.5),
            camera_fov=40,
        ),
        show_viewer=args.vis,
        renderer=gs.options.renderers.BatchRenderer(
            use_rasterizer=False,
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
        res=(512, 512),
        pos=(1.5, 0.5, 1.5),
        lookat=(0.0, 0.0, 0.5),
        fov=45,
        GUI=True,
    )
    cam_0.attach(franka.links[6], trans_to_T(np.array([0.0, 0.5, 0.0])))
    cam_1 = scene.add_camera(
        res=(512, 512),
        pos=(1.5, -0.5, 1.5),
        lookat=(0.0, 0.0, 0.5),
        fov=45,
        GUI=True,
    )
    scene.add_light(
        pos=[0.0, 0.0, 1.5],
        dir=[1.0, 1.0, -2.0],
        directional=1,
        castshadow=1,
        cutoff=45.0,
        intensity=0.5,
    )
    scene.add_light(
        pos=[4, -4, 4],
        dir=[-1, 1, -1],
        directional=0,
        castshadow=1,
        cutoff=45.0,
        intensity=0.5,
    )

    ########################## build ##########################
    scene.build(n_envs=N_ENVS)

    # Create an image exporter
    exporter = FrameImageExporter(OUTPUT_DIR)

    for i in range(N_STEPS):
        scene.step()
        if IS_RENDER_ALL_CAMERAS:
            rgb, depth, _, _ = scene.render_all_cameras()
            exporter.export_frame_all_cameras(i, rgb=rgb, depth=depth)
        else:
            rgb, depth, _, _ = cam_0.render()
            exporter.export_frame_single_camera(i, cam_0.idx, rgb=rgb, depth=depth)


if __name__ == "__main__":
    main()
