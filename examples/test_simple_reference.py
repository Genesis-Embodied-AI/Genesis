#!/usr/bin/env python3
"""
Create a simple reference scene in Genesis that matches the Blender reference scene.

This scene has (Genesis Z-up coordinates):
- Ground plane at z=0
- 6 colored spheres centered at z=4:
  - Red at X+ (2, 0, 4)
  - Cyan at X- (-2, 0, 4)
  - Yellow at Y+ (0, 2, 4)
  - Blue at Y- (0, -2, 4)
  - Green at Z+ (0, 0, 6)
  - Purple at Z- (0, 0, 2)
- 6 cameras ALL at position (0, 0, 4), each rotated to look at its corresponding sphere
- 1 point light at (0, 0, 10)

Run with viewer: python test_simple_reference.py --viewer
Run without viewer: python test_simple_reference.py
Export to Apollo: python test_simple_reference.py --export
"""

import genesis as gs
from genesis.utils.apollo_scene_exporter import ApolloSceneExporter
from gs_apollo import apollo_py_sdk as ap
import argparse
from pathlib import Path


def create_reference_scene(show_viewer=False):
    """Create the reference scene matching Blender layout."""
    
    # Create scene with optional viewer
    if show_viewer:
        scene = gs.Scene(
            renderer=gs.renderers.BatchRenderer(),
            show_viewer=True,
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(5.0, -5.0, 5.0),  # View from an angle
                camera_lookat=(0.0, 0.0, 4.0),
                camera_fov=40,
            ),
        )
    else:
        scene = gs.Scene(renderer=gs.renderers.BatchRenderer())

    # Ground plane at z=0
    plane = scene.add_entity(
        gs.morphs.Plane(
            pos=(0.0, 0.0, 0.0),
            plane_size=(20.0, 20.0),
        )
        #,
        # surface=gs.surfaces.Rough(
        #     color=(0.5, 0.5, 0.5),
        # )
    )

    # Red sphere at X+ : (2, 0, 4)
    red_sphere = scene.add_entity(
        gs.morphs.Sphere(
            pos=(2.0, 0.0, 4.0),
            radius=0.3,
        ),
        surface=gs.surfaces.Rough(
            color=(1.0, 0.0, 0.0),
        ),
    )

    # Cyan sphere at X- : (-2, 0, 4)
    cyan_sphere = scene.add_entity(
        gs.morphs.Sphere(
            pos=(-2.0, 0.0, 4.0),
            radius=0.3,
        ),
        surface=gs.surfaces.Rough(
            color=(0.0, 1.0, 1.0),
        ),
    )

    # Yellow sphere at Y+ : (0, 2, 4)
    yellow_sphere = scene.add_entity(
        gs.morphs.Sphere(
            pos=(0.0, 2.0, 4.0),
            radius=0.3,
        ),
        surface=gs.surfaces.Rough(
            color=(1.0, 1.0, 0.0),
        ),
    )

    # Blue sphere at Y- : (0, -2, 4)
    blue_sphere = scene.add_entity(
        gs.morphs.Sphere(
            pos=(0.0, -2.0, 4.0),
            radius=0.3,
        ),
        surface=gs.surfaces.Rough(
            color=(0.0, 0.0, 1.0),
        ),
    )

    # Green sphere at Z+ : (0, 0, 6)
    green_sphere = scene.add_entity(
        gs.morphs.Sphere(
            pos=(0.0, 0.0, 6.0),
            radius=0.3,
        ),
        surface=gs.surfaces.Rough(
            color=(0.0, 1.0, 0.0),
        ),
    )

    # Purple sphere at Z- : (0, 0, 2)
    purple_sphere = scene.add_entity(
        gs.morphs.Sphere(
            pos=(0.0, 0.0, 2.0),
            radius=0.3,
        ),
        surface=gs.surfaces.Rough(
            color=(0.5, 0.0, 0.5),
        ),
    )

    # All cameras at position (0, 0, 4), looking at different spheres
    camera_pos = (0.0, 0.0, 4.0)

    # Camera looking at red sphere (X+)
    cam_red = scene.add_camera(
        res=(1440, 960),
        pos=camera_pos,
        lookat=(2.0, 0.0, 4.0),
        fov=25,
        far=1000.0,
        GUI=False,
    )

    # Camera looking at cyan sphere (X-)
    cam_cyan = scene.add_camera(
        res=(1440, 960),
        pos=camera_pos,
        lookat=(-2.0, 0.0, 4.0),
        fov=25,
        far=1000.0,
        GUI=False,
    )

    # Camera looking at yellow sphere (Y+)
    cam_yellow = scene.add_camera(
        res=(1440, 960),
        pos=camera_pos,
        lookat=(0.0, 2.0, 4.0),
        fov=25,
        far=1000.0,
        GUI=False,
    )

    # Camera looking at blue sphere (Y-)
    cam_blue = scene.add_camera(
        res=(1440, 960),
        pos=camera_pos,
        lookat=(0.0, -2.0, 4.0),
        fov=25,
        far=1000.0,
        GUI=False,
    )

    # Camera looking at green sphere (Z+)
    cam_green = scene.add_camera(
        res=(1440, 960),
        pos=camera_pos,
        lookat=(0.0, 0.0, 6.0),
        fov=25,
        far=1000.0,
        GUI=False,
    )

    # Camera looking at purple sphere (Z-)
    cam_purple = scene.add_camera(
        res=(1440, 960),
        pos=camera_pos,
        lookat=(0.0, 0.0, 2.0),
        fov=25,
        far=1000.0,
        GUI=False,
    )

    # Point light at (0, 0, 10)
    scene.add_light(
        pos=(0.0, 0.0, 10.0),
        dir=(0.0, 0.0, -1.0),
        color=(1.0, 1.0, 1.0),
        intensity=1000.0,
        directional=False,
        castshadow=False,
    )

    scene.build(n_envs=0)

    # Return scene and cameras dict
    cameras = {
        'red': cam_red,
        'cyan': cam_cyan,
        'yellow': cam_yellow,
        'blue': cam_blue,
        'green': cam_green,
        'purple': cam_purple,
    }
    return scene, cameras


def main():
    parser = argparse.ArgumentParser(description='Create simple reference scene')
    parser.add_argument('--viewer', action='store_true',
                        help='Show interactive viewer')
    parser.add_argument('--viewer-camera', type=str, choices=['red', 'cyan', 'yellow', 'blue', 'green', 'purple'],
                        help='Set viewer camera to match one of the scene cameras')
    parser.add_argument('--export', action='store_true',
                        help='Export scene to Apollo JSON format')
    parser.add_argument('-o', '--output', type=str, default='test_exports_sdk/simple_reference.json',
                        help='Output path for exported JSON')
    args = parser.parse_args()

    # Startup Apollo SDK if exporting
    if args.export:
        ap.startup()

    # Initialize Genesis
    gs.init(backend=gs.gpu)

    # Create scene
    print("Creating reference scene...")
    scene, cameras = create_reference_scene(show_viewer=args.viewer or args.viewer_camera is not None)
    print("✓ Scene created")

    # Set viewer camera if requested
    if args.viewer_camera:
        camera_map = {
            'red': cameras['red'],
            'cyan': cameras['cyan'],
            'yellow': cameras['yellow'],
            'blue': cameras['blue'],
            'green': cameras['green'],
            'purple': cameras['purple'],
        }
        cam = camera_map[args.viewer_camera]
        # Flatten the arrays to 1D
        pos = cam.pos.flatten()
        lookat = cam.lookat.flatten()
        print(f"\nSetting viewer camera to match '{args.viewer_camera}' camera...")
        print(f"  Position: {pos}")
        print(f"  Lookat: {lookat}")
        scene.visualizer.viewer.set_camera_pose(pos=pos, lookat=lookat)

    # Export if requested
    if args.export:
        print(f"\nExporting scene to {args.output}...")

        exporter = ApolloSceneExporter()
        scene_asset = exporter.generate_from_scene(scene, export_path=args.output)

        # Add names to cameras (in order they were added to the scene)
        # camera_names = ["cam_red", "cam_cyan", "cam_yellow", "cam_blue", "cam_green", "cam_purple"]
        # num_cameras = scene_asset.camera_size()
        # print(f"Adding names to {num_cameras} cameras...")
        # for i in range(min(num_cameras, len(camera_names))):
        #     cam = scene_asset.get_camera(i)
        #     cam.name = camera_names[i]
        #     scene_asset.set_camera(i, cam)
        #     print(f"  Camera {i}: {camera_names[i]}")

        # Actually write the JSON file
        ap.export_scene_file(args.output, scene_asset)

        # Verify file was created
        output_file = Path(args.output)
        if output_file.exists():
            file_size = output_file.stat().st_size
            print(f"✓ Scene exported to {args.output} ({file_size} bytes)")
        else:
            print(f"✗ Export failed: File not created")

        ap.shutdown()

    # If viewer mode, keep it open (without stepping to avoid physics simulation)
    if args.viewer:
        print("\nViewer is open. Press Ctrl+C to exit.")
        try:
            import time
            while True:
                time.sleep(0.1)  # Just keep the viewer open without simulation
        except KeyboardInterrupt:
            print("\nExiting...")


if __name__ == "__main__":
    main()

