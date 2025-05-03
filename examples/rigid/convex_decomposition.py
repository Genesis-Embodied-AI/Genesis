import argparse

from huggingface_hub import snapshot_download

import genesis as gs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", default=False)
    parser.add_argument("-c", "--cpu", action="store_true", default=False)
    args = parser.parse_args()

    ########################## init ##########################
    gs.init(backend=gs.cpu if args.cpu else gs.gpu, precision="64" if args.cpu else "32", seed=0)

    ########################## create a scene ##########################
    scene = gs.Scene(
        rigid_options=gs.options.RigidOptions(
            dt=0.004,
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
            repo_type="dataset", repo_id="Genesis-Intelligence/assets", allow_patterns=f"{asset_name}/*"
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
    for i in range(2000):
        scene.step()


if __name__ == "__main__":
    main()
