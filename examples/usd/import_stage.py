import argparse
import os

from huggingface_hub import snapshot_download

import genesis as gs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--num_steps", type=int, default=1000)
    parser.add_argument("--show-viewer", action="store_true", default=False)
    args = parser.parse_args()

    args.num_steps = 1 if "PYTEST_VERSION" in os.environ else args.num_steps
    args.show_viewer = False if "PYTEST_VERSION" in os.environ else args.show_viewer
    args.show_viewer = True # just for testing
    
    gs.init(backend=gs.cpu)

    scene = gs.Scene(
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(3.5, 0.0, 2.5),
            camera_lookat=(0.0, 0.0, 0.5),
            camera_fov=40,
            enable_interaction=True,
        ),
        rigid_options=gs.options.RigidOptions(
            # constraint_solver=gs.constraint_solver.Newton,
            gravity=(0, 0, -9.8),
            enable_collision=True,
            enable_joint_limit=True,
            max_collision_pairs=1000,
        ),
        show_viewer=args.show_viewer,
    )

    # Download asset from HuggingFace
    asset_path = snapshot_download(
        repo_type="dataset",
        repo_id="Genesis-Intelligence/assets",
        revision="main",
        allow_patterns="usd/Lightwheel_Kitchen001/Kitchen001/*",
        max_workers=1,
    )

    # load a stage from USD file
    entities = scene.add_stage(f"{asset_path}/usd/Lightwheel_Kitchen001/Kitchen001/Kitchen001.usd")

    # Build the scene
    scene.build()

    # Run the simulation for visualization
    if args.show_viewer:
        while scene.viewer.is_alive():
            scene.step()
    else:
        for _ in range(args.num_steps):
            scene.step()


if __name__ == "__main__":
    main()
