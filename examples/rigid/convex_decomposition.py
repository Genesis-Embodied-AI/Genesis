import argparse

from huggingface_hub import snapshot_download

import genesis as gs


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", default=False)
    parser.add_argument("-c", "--cpu", action="store_true", default=False)
    args = parser.parse_args()

    ########################## init ##########################
    gs.init(backend=gs.cpu if args.cpu else gs.gpu)

    ########################## create a scene ##########################
    scene = gs.Scene(
        rigid_options=gs.options.RigidOptions(
            dt=0.005,
        ),
        show_viewer=args.vis,
    )

    ########################## entities ##########################
    scene.add_entity(
        gs.morphs.Mesh(
            file="meshes/tank.obj",
            scale=5.0,
            fixed=True,
            euler=(90, 0, 90),
        ),
        # vis_mode="collision",
    )
    for i, asset_name in enumerate(("donut_0", "mug_1", "cup_2", "apple_15")):
        asset_path = snapshot_download(
            repo_type="dataset", repo_id="Genesis-Intelligence/assets", allow_patterns=f"{asset_name}/*"
        )
        scene.add_entity(
            gs.morphs.MJCF(
                file=f"{asset_path}/{asset_name}/output.xml",
                pos=(0.0, 0.15 * (i - 1.5), 0.4),
            ),
            # vis_mode="collision",
        )

    ########################## build ##########################
    scene.build()
    for i in range(4000):
        scene.step()


if __name__ == "__main__":
    main()
