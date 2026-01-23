"""
Hibernation Example

This example demonstrates the hibernation feature in Genesis rigid body simulation.
Hibernation allows stationary rigid bodies to "sleep", improving simulation performance.

Note: Hibernation only works in fields mode (non-ndarray mode). Run with:
    GS_ENABLE_NDARRAY=0 python examples/hibernation.py -v

The scenario this example demonstrates:
1. Two boxes settle separately on ground -> both hibernate, 2 contact islands
2. Move one box above the other using set_pos (wakes it up)
3. Box falls and collides -> both boxes awake
4. Stacked boxes settle and hibernate -> 1 contact island (merged)

See PR https://github.com/Genesis-Embodied-AI/Genesis/pull/1542 for context.
"""

import argparse

import numpy as np

import genesis as gs


def main():
    parser = argparse.ArgumentParser(description="Hibernation demo with two boxes")
    parser.add_argument("-v", "--vis", action="store_true", default=False, help="Enable visualization")
    parser.add_argument("-c", "--cpu", action="store_true", default=False, help="Use CPU backend")
    args = parser.parse_args()

    gs.init(backend=gs.cpu if args.cpu else gs.gpu, logging_level="info")

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.01),
        rigid_options=gs.options.RigidOptions(
            use_contact_island=True,
            use_hibernation=True,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0, -2, 1.5),
            camera_lookat=(0, 0, 0.2),
            camera_up=(0, 0, 1),
        ),
        show_viewer=args.vis,
    )

    scene.add_entity(gs.morphs.Plane())

    # Two boxes placed separately on ground
    box1 = scene.add_entity(
        gs.morphs.Box(pos=(-0.3, 0, 0.15), size=(0.1, 0.1, 0.1)),
        surface=gs.surfaces.Default(color=(1.0, 0.2, 0.2, 1.0)),
    )
    box2 = scene.add_entity(
        gs.morphs.Box(pos=(0.3, 0, 0.15), size=(0.1, 0.1, 0.1)),
        surface=gs.surfaces.Default(color=(0.2, 0.2, 1.0, 1.0)),
    )

    scene.build()

    solver = scene.sim.rigid_solver
    box1_idx = box1._idx_in_solver
    box2_idx = box2._idx_in_solver

    def is_hibernated(entity_idx):
        return solver.entities_state.hibernated[entity_idx, 0] == 1

    def get_n_islands():
        return solver.constraint_solver.contact_island.n_islands[0]

    def print_status(step, phase):
        box1_status = "HIBERNATED" if is_hibernated(box1_idx) else "AWAKE"
        box2_status = "HIBERNATED" if is_hibernated(box2_idx) else "AWAKE"
        n_islands = get_n_islands()
        print(f"[Step {step:4d}] {phase}: Box1={box1_status}, Box2={box2_status}, Islands={n_islands}")

    # Phase 1: Let boxes settle and hibernate separately
    print("=" * 60)
    print("Phase 1: Letting boxes settle and hibernate separately...")
    print("=" * 60)

    for step in range(500):
        scene.step()
        if step % 20 == 0:
            print_status(step, "Settling")
        if is_hibernated(box1_idx) and is_hibernated(box2_idx):
            print_status(step, "Settling")
            print("  -> Both boxes are now hibernated!")
            print(f"  -> Contact islands: {get_n_islands()} (should be 2)")
            break

    # Phase 2: Move box1 above box2 (this wakes up box1)
    print()
    print("=" * 60)
    print("Phase 2: Moving box1 above box2 (triggers wake-up)...")
    print("=" * 60)

    box2_pos = box2.get_pos()
    offset = 0.01
    box1.set_pos(np.array([float(box2_pos[0]) + offset, float(box2_pos[1]) + offset, 0.3]))

    print(f"  -> Box1 moved to z={float(box1.get_pos()[2]):.3f}")
    print(f"  -> Box1 hibernated: {is_hibernated(box1_idx)} (should be False)")

    # Let box1 fall and collide with box2
    print()
    print("=" * 60)
    print("Phase 3: Letting box1 fall and collide with box2...")
    print("=" * 60)

    for step in range(50):
        scene.step()
        if step % 10 == 0:
            print_status(step, "Falling")

    print(
        f"  -> Both boxes awake after collision: box1={not is_hibernated(box1_idx)}, box2={not is_hibernated(box2_idx)}"
    )

    # Phase 4: Let stacked boxes settle and hibernate
    print()
    print("=" * 60)
    print("Phase 4: Letting stacked boxes settle and hibernate...")
    print("=" * 60)

    for step in range(500):
        scene.step()
        if step % 50 == 0:
            print_status(step, "Settling")
        if is_hibernated(box1_idx) and is_hibernated(box2_idx):
            print_status(step, "Settling")
            print("  -> Both boxes are now hibernated!")
            print(f"  -> Contact islands: {get_n_islands()} (should be 1 - stacked)")
            break

    print()
    print("=" * 60)
    print("Simulation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
