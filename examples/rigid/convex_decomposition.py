import argparse
import os

from huggingface_hub import snapshot_download

import genesis as gs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", default=False)
    parser.add_argument("-c", "--cpu", action="store_true", default=False)
    args = parser.parse_args()

    ########################## init ##########################
    gs.init(backend=gs.cpu if args.cpu else gs.gpu, precision="32", seed=0)

    ########################## create a scene ##########################
    scene = gs.Scene(
        rigid_options=gs.options.RigidOptions(
            dt=0.01,
        ),
        show_viewer=args.vis,
        show_FPS=False,
    )

    ########################## entities ##########################
    scene.add_entity(
        gs.morphs.Mesh(
            file="meshes/tank.obj",
            scale=5.0,
            fixed=True,
            # euler=(90, 0, 90),
            euler=(80, 10, 90),
            pos=(0.05, -0.1, 0.0),
        ),
        vis_mode="collision",
    )
    for i, asset_name in enumerate(("donut_0", "mug_1", "cup_2", "apple_15")):
        asset_path = snapshot_download(
            repo_type="dataset",
            repo_id="Genesis-Intelligence/assets",
            revision="4d96c3512df4421d4dd3d626055d0d1ebdfdd7cc",
            allow_patterns=f"{asset_name}/*",
            max_workers=1,
        )
        scene.add_entity(
            gs.morphs.MJCF(
                file=f"{asset_path}/{asset_name}/output.xml",
                pos=(0.0, 0.15 * (i - 1.5), 0.7),
            ),
            vis_mode="collision",
            # visualize_contact=True,
        )

    ########################## build ##########################
    scene.build()
    horizon = 2000 if "PYTEST_VERSION" not in os.environ else 5
    for i in range(horizon):
        scene.step()


if __name__ == "__main__":
    main()
