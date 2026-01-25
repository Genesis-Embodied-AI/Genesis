"""
Hibernation Performance Example

This example demonstrates the performance benefit of hibernation in Genesis rigid body simulation.
Hibernation allows stationary rigid bodies to "sleep", skipping physics computations for objects
that have settled, which significantly improves simulation performance.

The scenario creates many boxes that fall and settle on a ground plane. Once settled, hibernated
objects require minimal computation, while non-hibernated simulations continue computing physics
for all objects every step.

Usage:
    python examples/hibernation.py           # Run performance comparison
    python examples/hibernation.py -v        # With visualization (slower due to rendering)
    python examples/hibernation.py -n 50     # Use 50 boxes instead of default 20
"""

import argparse
import time

import genesis as gs


def run_simulation(use_hibernation: bool, n_boxes: int, n_steps: int, show_viewer: bool) -> float:
    """Run simulation and return total time for the stepping phase."""
    scene = gs.Scene(
        rigid_options=gs.options.RigidOptions(
            use_contact_island=True,
            use_hibernation=use_hibernation,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0, -3, 2),
            camera_lookat=(0, 0, 0.5),
            camera_up=(0, 0, 1),
        ),
        show_viewer=show_viewer,
    )

    scene.add_entity(gs.morphs.Plane())

    # Create a grid of boxes that will fall and settle
    spacing = 0.25
    box_size = 0.1
    grid_size = int(n_boxes**0.5) + 1

    for i in range(n_boxes):
        row = i // grid_size
        col = i % grid_size
        x = (col - grid_size / 2) * spacing
        y = (row - grid_size / 2) * spacing
        z = 0.5 + (i % 3) * 0.2  # Stagger heights slightly
        scene.add_entity(
            gs.morphs.Box(pos=(x, y, z), size=(box_size, box_size, box_size)),
        )

    scene.build()

    # Warm-up phase: let boxes fall and settle
    warmup_steps = 200
    for _ in range(warmup_steps):
        scene.step()

    # Timed phase: measure performance after objects have settled
    start_time = time.perf_counter()
    for _ in range(n_steps):
        scene.step()
    elapsed = time.perf_counter() - start_time
    scene.destroy()

    return elapsed


def main():
    parser = argparse.ArgumentParser(description="Hibernation performance comparison")
    parser.add_argument("-v", "--vis", action="store_true", default=False, help="Enable visualization")
    parser.add_argument("-c", "--cpu", action="store_true", default=False, help="Use CPU backend")
    parser.add_argument("-n", "--n-boxes", type=int, default=20, help="Number of boxes (default: 20)")
    parser.add_argument("-s", "--steps", type=int, default=500, help="Number of timed steps (default: 500)")
    args = parser.parse_args()

    gs.init(backend=gs.cpu if args.cpu else gs.gpu, performance_mode=True)

    print("=" * 70)
    print("Hibernation Performance Comparison")
    print("=" * 70)
    print(f"Configuration: {args.n_boxes} boxes, {args.steps} timed steps")
    print(f"Backend: {'CPU' if args.cpu else 'GPU'}")
    print()

    # Run without hibernation
    print("Running simulation WITHOUT hibernation...")
    time_without = run_simulation(
        use_hibernation=False,
        n_boxes=args.n_boxes,
        n_steps=args.steps,
        show_viewer=args.vis,
    )
    print(f"  Time: {time_without:.3f}s ({args.steps / time_without:.1f} steps/sec)")

    # Run with hibernation
    print("Running simulation WITH hibernation...")
    time_with = run_simulation(
        use_hibernation=True,
        n_boxes=args.n_boxes,
        n_steps=args.steps,
        show_viewer=args.vis,
    )
    print(f"  Time: {time_with:.3f}s ({args.steps / time_with:.1f} steps/sec)")

    # Results
    print()
    print("=" * 70)
    print("Results")
    print("=" * 70)
    speedup = time_without / time_with if time_with > 0 else float("inf")
    print(f"Without hibernation: {time_without:.3f}s")
    print(f"With hibernation:    {time_with:.3f}s")
    print(f"Speedup:             {speedup:.2f}x faster with hibernation")
    print()
    print("Note: Hibernation benefit increases with more settled objects and longer simulations.")
    print("      The speedup comes from skipping physics computations for sleeping objects.")


if __name__ == "__main__":
    main()
