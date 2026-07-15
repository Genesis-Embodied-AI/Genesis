"""
Load the Lightwheel Kitchen USD assets in Genesis.

Usage
-----
    python examples/usd/kitchen.py                    # sample assets, headless
    python examples/usd/kitchen.py --asset dishwasher
    python examples/usd/kitchen.py --vis              # interactive viewer (drag with the mouse)
    python examples/usd/kitchen.py --full --vis       # the whole kitchen scene

With ``-v``/``--vis`` the scene runs indefinitely in the interactive viewer with the
MouseInteractionPlugin (drag entities with the mouse); press ``Esc`` to quit.
"""

import argparse
import os

import numpy as np
from huggingface_hub import snapshot_download

import genesis as gs
import genesis.vis.keybindings as kb
from genesis.utils.misc import tensor_to_array

SAMPLE_ASSETS = {
    "dishwasher": ("Lightwheel_Kitchen/Dishwasher054/Dishwasher054.usd", ["Lightwheel_Kitchen/Dishwasher054/*"]),
    "bottle": (
        "Lightwheel_Kitchen/Kitchen_Other/Kitchen_Bottle006.usd",
        ["Lightwheel_Kitchen/Kitchen_Other/Kitchen_Bottle006.usd", "Lightwheel_Kitchen/Kitchen_Other/texture/*"],
    ),
}
FULL_ROOM_ASSETS = ("Lightwheel_Kitchen/KitchenRoom.usd", ["Lightwheel_Kitchen/*"])


def place_on_ground(entities, gap=0.05):
    """Lay entities out in a row on the ground plane."""
    x = 0.0
    for entity in entities:
        lo, hi = tensor_to_array(entity.get_AABB())
        size = hi - lo
        target_min = np.array([x, -0.5 * size[1], gap], dtype=np.float32)
        entity.set_pos(tensor_to_array(entity.get_pos()) + (target_min - lo))
        x += size[0] + gap


def load_asset(scene, root, rel_path, fixed):
    usd_file = os.path.join(root, rel_path)
    if not os.path.isfile(usd_file):
        gs.raise_exception(f"USD file not found: {usd_file}")

    gs.logger.info(f"Loading {rel_path} ...")
    entities = scene.add_stage(
        morph=gs.morphs.USD(
            file=usd_file,
            fixed=fixed,
            convexify=False,  # Don't force convex hulls; honor the asset's MeshCollisionAPI approximation per geom.
            decimate=True,  # Simplify collision meshes (fewer faces) for speed and stability.
            align=False,  # Keep the USD root-link frames (don't re-center to the center of mass).
        ),
        vis_mode="visual",  # Render the entity's own USD materials, not the randomized per-collision colors.
    )
    gs.logger.info(f"  -> parsed {len(entities)} entit{'y' if len(entities) == 1 else 'ies'}")
    for e in entities:
        gs.logger.info(f"     {e.__class__.__name__}: n_links={e.n_links} n_joints={e.n_joints} n_geoms={e.n_geoms}")
    return entities


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--asset", default="all", choices=[*SAMPLE_ASSETS, "all"], help="Which sample asset(s) to load."
    )
    parser.add_argument("--full", action="store_true", help="Load the entire kitchen scene instead of the samples.")
    parser.add_argument("-n", "--num_steps", type=int, default=0, help="Number of sim steps after build (headless).")
    parser.add_argument("-v", "--vis", action="store_true", default=False, help="Show the interactive viewer.")
    args = parser.parse_args()

    gs.init(backend=gs.cpu)

    scene = gs.Scene(
        show_viewer=args.vis,
        show_FPS=False,
        rigid_options=gs.options.RigidOptions(
            enable_neutral_collision=True,  # Enable so articulated parts (e.g. dishwasher) don't clip
        ),
    )
    scene.add_entity(gs.morphs.Plane())

    if args.full:
        patterns = FULL_ROOM_ASSETS[1]
        rel_paths = [FULL_ROOM_ASSETS[0]]
        fixed = None  # Keep the scene's authored fixed/free states.
    else:
        selected = list(SAMPLE_ASSETS) if args.asset == "all" else [args.asset]
        patterns = [pattern for key in selected for pattern in SAMPLE_ASSETS[key][1]]
        rel_paths = [SAMPLE_ASSETS[key][0] for key in selected]
        fixed = False  # Free base so every sample (incl. the authored-fixed dishwasher) drops and is draggable.

    asset_path = snapshot_download(
        repo_type="dataset",
        repo_id="Genesis-Intelligence/assets",
        revision="c3d4f971ac7da1ca2257adc7663b3aaea82c9a32",
        allow_patterns=patterns,
    )

    all_entities = []
    for rel_path in rel_paths:
        all_entities += load_asset(scene, asset_path, rel_path, fixed=fixed)

    if args.vis:
        # Drag entities around with the mouse; the plugin must be attached before build.
        scene.viewer.add_plugin(
            gs.vis.viewer_plugins.MouseInteractionPlugin(
                use_force=True,
                color=(0.1, 0.6, 0.8, 0.6),
            )
        )

    scene.build()
    if not args.full:
        place_on_ground(all_entities)
    gs.logger.info(
        f"Scene built successfully with {len(all_entities)} entit{'y' if len(all_entities) == 1 else 'ies'}."
    )

    if args.vis:
        is_running = True

        def stop():
            nonlocal is_running
            is_running = False

        scene.viewer.register_keybinds(
            kb.Keybind("quit", kb.Key.ESCAPE, kb.KeyAction.RELEASE, callback=stop),
        )

        while is_running and scene.viewer.is_alive():
            scene.step()
    else:
        for _ in range(args.num_steps):
            scene.step()


if __name__ == "__main__":
    main()
