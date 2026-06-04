import argparse

import numpy as np

import genesis as gs
from genesis.ext.pyrender.overlay import ImGuiOverlayPlugin


FPS = 60


def launch(filename=None, collision=False, rotate=False, scale=1.0, show_link_frame=False, deprecated=False):
    gs.init(backend=gs.cpu)

    if deprecated:
        gs.logger.warning("'gs view' is deprecated and will be removed in a future release. Use 'gs launch' instead.")

    scene = gs.Scene(
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(3.5, 0.0, 2.5),
            camera_lookat=(0.0, 0.0, 0.5),
            camera_fov=40,
            max_FPS=FPS,
            enable_gui=True,
        ),
        vis_options=gs.options.VisOptions(
            show_link_frame=show_link_frame,
            show_world_frame=True,
        ),
        show_viewer=True,
    )

    # With no file given, open an empty interactive scene; entities can be added live through the overlay's
    # "Add Entity / Stage" panel and applied with "Rebuild Scene".
    entities = []
    if filename is not None:
        filename_lower = filename.lower()
        morphs = gs.options.morphs
        material = gs.materials.Rigid()
        # Morphs load collision geometry by default, so the overlay's collision vis-mode has something to show; the -c
        # flag only selects which representation is displayed first.
        surface = gs.surfaces.Default(vis_mode="visual" if not collision else "collision")

        if filename_lower.endswith(morphs.USD_FORMATS):
            morph = gs.morphs.USD(file=filename, scale=scale)
            entities = scene.add_stage(morph=morph, vis_mode=surface.vis_mode)
        elif filename_lower.endswith((morphs.URDF_FORMAT, morphs.XACRO_FORMAT)):
            morph_cls = gs.morphs.URDF
            entities = [
                scene.add_entity(
                    morph_cls(file=filename, scale=scale),
                    material=material,
                    surface=surface,
                )
            ]
        elif filename_lower.endswith(morphs.MJCF_FORMAT):
            morph_cls = gs.morphs.MJCF
            entities = [
                scene.add_entity(
                    morph_cls(file=filename, scale=scale),
                    material=material,
                    surface=surface,
                )
            ]
        elif filename_lower.endswith(morphs.MESH_FORMATS):
            morph_cls = gs.morphs.Mesh
            entities = [
                scene.add_entity(
                    morph_cls(file=filename, scale=scale),
                    material=material,
                    surface=surface,
                )
            ]
        else:
            gs.raise_exception(
                f"Unsupported file format for 'gs launch'. Expected {morphs.URDF_FORMAT}, {morphs.XACRO_FORMAT}, "
                f"{morphs.MJCF_FORMAT}, {morphs.MESH_FORMATS}, or {morphs.USD_FORMATS}."
            )

    scene.build(compile_kernels=False)

    # 'enable_gui=True' auto-attaches the ImGui overlay, which owns the simulation through an InteractiveScene:
    # play/pause/step/reset and per-joint sliders (editable while paused) live in the overlay. Start paused so the
    # asset can be inspected and posed before the physics simulation is stepped.
    plugin = next(p for p in scene.viewer.plugins if isinstance(p, ImGuiOverlayPlugin))
    plugin.interactive_scene.pause()

    t = 0
    while scene.viewer.is_alive():
        # Rotate entity if requested, independently of the play/pause state, as a visual turntable.
        if rotate:
            t += 1 / FPS
            quat = gs.utils.geom.xyz_to_quat(np.array([0, 0, t * 50]), rpy=True, degrees=True)
            for entity in entities:
                entity.set_quat(quat)
        scene.step()


def play(filename=None, collision=False, scale=1.0):
    gs.init()

    scene = gs.Scene(
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(2.0, 2.0, 1.5),
            camera_lookat=(0.0, 0.0, 0.5),
            enable_gui=True,
        ),
        show_viewer=True,
    )

    if filename is None:
        scene.add_entity(gs.morphs.Plane())
        scene.add_entity(gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"))
    else:
        filename_lower = filename.lower()
        morphs = gs.options.morphs
        surface = gs.surfaces.Default(vis_mode="visual" if not collision else "collision")

        if filename_lower.endswith(morphs.USD_FORMATS):
            scene.add_stage(
                morph=gs.morphs.USD(file=filename, scale=scale),
                vis_mode=surface.vis_mode,
            )
        elif filename_lower.endswith(morphs.URDF_FORMAT):
            scene.add_entity(gs.morphs.URDF(file=filename, scale=scale), surface=surface)
        elif filename_lower.endswith(morphs.MJCF_FORMAT):
            scene.add_entity(gs.morphs.MJCF(file=filename, scale=scale), surface=surface)
        elif filename_lower.endswith(morphs.MESH_FORMATS):
            scene.add_entity(gs.morphs.Mesh(file=filename, scale=scale), surface=surface)
        else:
            gs.raise_exception(
                f"Unsupported file format for 'gs play'. Expected {morphs.URDF_FORMAT}, "
                f"{morphs.MJCF_FORMAT}, {morphs.MESH_FORMATS}, or {morphs.USD_FORMATS}."
            )

    scene.build()

    while scene.viewer.is_alive():
        scene.step()


def animate(filename_pattern, fps):
    import glob

    from PIL import Image

    gs.init()
    files = sorted(glob.glob(filename_pattern))
    imgs = []
    for file in files:
        print(f"Loading {file}")
        imgs.append(np.array(Image.open(file)))
    gs.tools.animate(imgs, "video.mp4", fps)


def main():
    parser = argparse.ArgumentParser(description="Genesis CLI")
    subparsers = parser.add_subparsers(dest="command")

    launch_args = argparse.ArgumentParser(add_help=False)
    launch_args.add_argument(
        "filename",
        type=str,
        nargs="?",
        default=None,
        help="Optional asset file (Mesh/URDF/MJCF/USD). Defaults to an empty interactive scene.",
    )
    launch_args.add_argument(
        "-c", "--collision", action="store_true", default=False, help="Whether to visualize collision geometry"
    )
    launch_args.add_argument("-r", "--rotate", action="store_true", default=False, help="Whether to rotate the entity")
    launch_args.add_argument("-s", "--scale", type=float, default=1.0, help="Scale of the entity")
    launch_args.add_argument("-l", "--link_frame", action="store_true", default=False, help="Show link frame")

    subparsers.add_parser("launch", parents=[launch_args], help="Visualize a given asset (Mesh/URDF/MJCF/USD)")
    subparsers.add_parser("view", parents=[launch_args], help="[DEPRECATED] Alias of 'launch'.")

    parser_play = subparsers.add_parser("play", help="Interactive viewer with ImGui joint controls and simulation")
    parser_play.add_argument(
        "filename",
        type=str,
        nargs="?",
        default=None,
        help="Optional asset file (URDF/MJCF/Mesh/USD). Defaults to a demo scene.",
    )
    parser_play.add_argument(
        "-c", "--collision", action="store_true", default=False, help="Visualize collision geometry"
    )
    parser_play.add_argument("-s", "--scale", type=float, default=1.0, help="Scale of the entity")

    parser_animate = subparsers.add_parser("animate", help="Compile a list of image files into a video")
    parser_animate.add_argument("filename_pattern", type=str, help="Image files, via glob pattern")
    parser_animate.add_argument("--fps", type=int, default=30, help="FPS of the output video")

    args = parser.parse_args()

    if args.command in ("launch", "view"):
        launch(
            args.filename,
            args.collision,
            args.rotate,
            args.scale,
            args.link_frame,
            deprecated=args.command == "view",
        )
    elif args.command == "play":
        play(args.filename, args.collision, args.scale)
    elif args.command == "animate":
        animate(args.filename_pattern, args.fps)
    elif args.command is None:
        parser.print_help()


if __name__ == "__main__":
    main()
