import argparse
import os

import numpy as np
from huggingface_hub import snapshot_download

import genesis as gs
from genesis.utils.misc import ti_to_numpy
import genesis.utils.geom as gu


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--num_steps", type=int, default=5)
    parser.add_argument("-v", "--vis", action="store_true", default=False)
    args = parser.parse_args()

    gs.init()

    scene = gs.Scene(
        rigid_options=gs.options.RigidOptions(
            dt=0.01,
            gravity=(0, 0, -9.8),
            enable_collision=True,
            enable_joint_limit=True,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(4.0, 2.0, 2.5),
            camera_lookat=(0.0, 0.0, 1.0),
            camera_fov=40,
        ),
        show_viewer=args.vis,
    )

    entities = scene.add_stage(
        stage=gs.morphs.USD(
            file="data/251209_parse_usd/Kitchen001.usd",
        ),
        # vis_mode="collision",
        # visualize_contact=True,
    )

    for entity in entities:
        print(entity.morph.prim_path)
        for joint in entity.joints:
            print(joint.name, joint.type)

    cam1 = scene.add_camera(
        res=(1280, 960),
        pos=(-3.0, 3.0, 3.0),
        lookat=(-2.0, 2.0, 2.0),
        GUI=False,
        fov=40,
    )

    scene.build()

    # args.num_steps = 1000000
    for i in range(args.num_steps):
        scene.step()
        rgb = cam1.render(rgb=True)[0]
        gs.tools.save_img_arr(rgb, f"data/demo/kitchen_cam1_step{i:03d}.png")

if __name__ == "__main__":
    main()
