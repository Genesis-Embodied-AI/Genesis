"""
Camera Sensor Multi-Environment Example

This example demonstrates using camera sensors with multiple parallel environments (n_envs > 1).
It shows:
1. Static cameras rendering different object poses per environment
2. Attached cameras that follow entities with per-environment poses
3. Support for both Rasterizer and RayTracer renderers
"""

import argparse
import torch
from PIL import Image

import genesis as gs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", default=False, help="Show viewer")
    parser.add_argument("-n", "--n_envs", type=int, default=4, help="Number of parallel environments")
    parser.add_argument("-o", "--output", type=str, default="camera_multi_env", help="Output prefix for images")
    parser.add_argument(
        "-r",
        "--renderer",
        type=str,
        choices=["rasterizer", "raytracer"],
        default="rasterizer",
        help="Renderer to use (rasterizer or raytracer)",
    )
    args = parser.parse_args()

    gs.init(backend=gs.cpu)

    # Select renderer based on argument
    if args.renderer == "raytracer":
        renderer = gs.renderers.RayTracer()
        CameraOptions = gs.sensors.RaytracerCameraOptions
        camera_kwargs = {"spp": 4}  # Low samples for speed
        print("Using RayTracer renderer (note: multi-env renders sequentially)")
    else:
        renderer = gs.renderers.Rasterizer()
        CameraOptions = gs.sensors.RasterizerCameraOptions
        camera_kwargs = {}
        print("Using Rasterizer renderer")

    # Create scene
    scene = gs.Scene(
        rigid_options=gs.options.RigidOptions(
            enable_collision=True,
            gravity=(0, 0, -9.8),
        ),
        renderer=renderer,
        show_viewer=args.vis,
    )

    # Ground plane
    plane = scene.add_entity(
        morph=gs.morphs.Plane(),
        surface=gs.surfaces.Rough(color=(0.4, 0.4, 0.4)),
    )

    # Robot arm placeholder (box)
    robot = scene.add_entity(
        morph=gs.morphs.Box(pos=(0.0, 0.0, 0.5), size=(0.2, 0.2, 0.5)),
        surface=gs.surfaces.Smooth(color=(0.2, 0.5, 0.8)),
    )

    # Target objects - will be at different positions per env
    target = scene.add_entity(
        morph=gs.morphs.Sphere(pos=(1.5, 0.0, 0.5), radius=0.15),
        surface=gs.surfaces.Smooth(color=(1.0, 0.3, 0.3)),
    )

    # Static camera - observes the scene from a fixed position
    static_camera = scene.add_sensor(
        CameraOptions(
            res=(256, 256),
            pos=(3.0, 0.0, 2.0),
            lookat=(0.0, 0.0, 0.5),
            fov=60.0,
            **camera_kwargs,
        )
    )

    # Attached camera - mounted on the "robot" (box)
    # This camera will have different poses in each environment because
    # the robot it's attached to can be at different positions
    attached_camera = scene.add_sensor(
        CameraOptions(
            res=(256, 256),
            pos=(0.3, 0.0, 0.5),  # Offset from robot center
            lookat=(1.5, 0.0, 0.5),  # Look towards target
            fov=60.0,
            entity_idx=robot.idx,
            link_idx_local=0,
            **camera_kwargs,
        )
    )

    # Build scene with multiple environments
    scene.build(n_envs=args.n_envs, env_spacing=(0.0, 0.0))
    print(f"Built scene with {args.n_envs} environments")

    # Set different robot and target positions for each environment
    for i in range(args.n_envs):
        # Move robot to different positions
        robot_x = -0.5 + i * 0.3
        robot.set_pos(torch.tensor([robot_x, 0.0, 0.5], device=gs.device), envs_idx=[i])

        # Move target to different positions
        target_y = -0.5 + i * 0.3
        target.set_pos(torch.tensor([1.5, target_y, 0.5], device=gs.device), envs_idx=[i])

    # Simulate and render
    scene.step()

    # Read images from cameras
    print("Rendering static camera...")
    static_data = static_camera.read()
    print("Rendering attached camera...")
    attached_data = attached_camera.read()

    print(f"Static camera output shape: {static_data.rgb.shape}")
    print(f"Attached camera output shape: {attached_data.rgb.shape}")

    # Create a combined image grid: 2 rows (static, attached) x n_envs columns
    h, w = static_data.rgb.shape[1:3]
    combined = Image.new("RGB", (w * args.n_envs, h * 2))

    for i in range(args.n_envs):
        # Static camera (top row)
        static_img = Image.fromarray(static_data.rgb[i].cpu().numpy())
        combined.paste(static_img, (i * w, 0))

        # Attached camera (bottom row)
        attached_img = Image.fromarray(attached_data.rgb[i].cpu().numpy())
        combined.paste(attached_img, (i * w, h))

    output_path = f"{args.output}.png"
    combined.save(output_path)
    print(f"Saved combined image: {output_path}")
    print(f"  Top row: static camera (env 0 to {args.n_envs - 1})")
    print(f"  Bottom row: attached camera (env 0 to {args.n_envs - 1})")

    # Compute and print image differences to verify per-env poses
    if args.n_envs >= 2:
        print("\nImage difference analysis (verifies per-env poses):")
        for cam_name, data in [("Static", static_data), ("Attached", attached_data)]:
            diff = (data.rgb[0].float() - data.rgb[1].float()).abs().mean()
            print(f"  {cam_name} camera: mean abs diff between env0 and env1 = {diff:.2f}")


if __name__ == "__main__":
    main()
