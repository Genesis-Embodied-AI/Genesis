"""Interactive demo of runtime heterogeneous variant switching.

A single entity is built from a list of morphs -- randomly sized boxes/spheres/cylinders, or (with --articulated)
2-link pendulum chains -- and each environment shows one of them, switched live via `entity.set_entity_variant`;
both the physics and the rendered geometry follow. Run with -v/--vis to open the viewer and press SPACE to
randomize the arrangement interactively; without it the demo performs a single switch as a headless smoke check.
"""

import argparse

import numpy as np

import genesis as gs
from genesis.utils.procedural import build_articulated_chain
from genesis.vis.keybindings import Key, KeyAction, Keybind


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", help="Open the interactive viewer to drive switching live")
    parser.add_argument("-g", "--gpu", action="store_true", help="Run on GPU instead of CPU")
    parser.add_argument("-b", "--num-envs", type=int, default=4, help="Number of parallel environments")
    parser.add_argument("--n-variants", type=int, default=3, help="Number of morph variants to build")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument(
        "--articulated", action="store_true", help="Use 2-link pendulum chains instead of primitive shapes"
    )
    args = parser.parse_args()

    gs.init(backend=gs.gpu if args.gpu else gs.cpu)

    rng = np.random.default_rng(args.seed)

    # A box, sphere and cylinder in rotation (or 2-link chains under --articulated), each at a random size.
    variants = []
    for i in range(args.n_variants):
        if args.articulated:
            # Vary only the radius: the rebind does not move joint anchors, so a different length would misplace links.
            variants.append(
                gs.morphs.MJCF(
                    file=build_articulated_chain(
                        n_links=2,
                        link_radius=rng.uniform(0.015, 0.07),
                        link_length=0.25,
                    ),
                    pos=(0.0, 0.0, 0.8),
                )
            )
        elif i % 3 == 0:
            side = rng.uniform(0.12, 0.28)
            variants.append(gs.morphs.Box(size=(side, side, side), pos=(0.0, 0.0, 0.3)))
        elif i % 3 == 1:
            variants.append(gs.morphs.Sphere(radius=rng.uniform(0.08, 0.16), pos=(0.0, 0.0, 0.3)))
        else:
            variants.append(
                gs.morphs.Cylinder(radius=rng.uniform(0.07, 0.13), height=rng.uniform(0.15, 0.35), pos=(0.0, 0.0, 0.3))
            )

    scene = gs.Scene(
        show_viewer=args.vis,
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(4.0, 4.0, 3.0),
            camera_lookat=(0.0, 0.0, 0.4 if args.articulated else 0.1),
        ),
    )
    scene.add_entity(
        gs.morphs.Plane(),
    )
    het = scene.add_entity(
        morph=variants,
    )
    if args.vis:
        # Mouse-drag to sanity-check collision and mass; spring force works for free and anchored bodies alike.
        scene.viewer.add_plugin(
            gs.vis.viewer_plugins.MouseInteractionPlugin(
                use_force=True,
            )
        )
    scene.build(
        n_envs=args.num_envs,
        env_spacing=(1.0, 1.0),
    )

    n_variants = len(variants)
    is_running = True
    pending_randomize = True  # start from a random arrangement

    def randomize():
        nonlocal pending_randomize
        pending_randomize = True

    def stop():
        nonlocal is_running
        is_running = False

    if args.vis:
        scene.viewer.register_keybinds(
            Keybind("randomize_variants", Key.SPACE, KeyAction.PRESS, callback=randomize),
            Keybind("quit", Key.ESCAPE, KeyAction.RELEASE, callback=stop),
        )
        print("\nSPACE randomizes each environment's variant, drag objects with the mouse, ESC to quit.\n")

    while is_running:
        # Keybind callbacks fire on the viewer thread; mutate the scene here on the stepping thread.
        if pending_randomize:
            pending_randomize = False
            het.set_entity_variant(rng.integers(0, n_variants, size=args.num_envs))
            if args.articulated:
                # Kick the hinges so the fresh chain swings.
                het.set_dofs_velocity(rng.uniform(-5.0, 5.0, size=het.n_dofs))
            else:
                # Drop the fresh geometry so the switch reads clearly.
                het.set_pos((0.0, 0.0, 0.3))
                het.set_dofs_velocity(np.zeros(6))
        scene.step()

        # Without a viewer there is no interactive quit, so one switch suffices as a headless smoke check.
        if not args.vis:
            break


if __name__ == "__main__":
    main()
