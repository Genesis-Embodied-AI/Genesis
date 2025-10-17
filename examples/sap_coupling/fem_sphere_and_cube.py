import argparse
import sys
import numpy as np
import genesis as gs
from huggingface_hub import snapshot_download


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--cpu", action="store_true", default=(sys.platform == "darwin"))
    parser.add_argument("-v", "--vis", action="store_true", default=False)
    args = parser.parse_args()

    gs.init(backend=gs.cpu if args.cpu else gs.gpu, precision="64", performance_mode=True)

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=1 / 60,
            substeps=2,
        ),
        fem_options=gs.options.FEMOptions(
            use_implicit_solver=True,
        ),
        coupler_options=gs.options.SAPCouplerOptions(),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(1.5, -1.5, 1.5),
            camera_lookat=(0, 0, 0),
            max_FPS=60,
        ),
        show_viewer=args.vis,
    )
    sphere = scene.add_entity(
        morph=gs.morphs.Sphere(
            pos=(0.0, 0.0, 0.1),
            radius=0.1,
        ),
        material=gs.materials.FEM.Elastic(
            model="linear_corotated",
            E=1e5,
            nu=0.4,
        ),
    )
    asset_path = snapshot_download(
        repo_type="dataset",
        repo_id="Genesis-Intelligence/assets",
        revision="4d96c3512df4421d4dd3d626055d0d1ebdfdd7cc",
        allow_patterns="cube8.obj",
        max_workers=1,
    )
    cube = scene.add_entity(
        morph=gs.morphs.Mesh(
            file=f"{asset_path}/cube8.obj",
            pos=(0.0, 0.0, 0.4),
            scale=0.1,
        ),
        material=gs.materials.FEM.Elastic(
            model="linear_corotated",
        ),
    )
    scene.build()

    for _ in range(200):
        scene.step()


if __name__ == "__main__":
    main()
