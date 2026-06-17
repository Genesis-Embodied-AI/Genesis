"""Per-kernel breakdown of the N-box step, to localize what scales with N.

python tests/bench_box_profile.py <dense|island|sparse> <n_bodies>
"""

import math
import sys

import quadrants as qd

_orig_init = qd.init


def _init_with_profiler(*a, **k):
    k["kernel_profiler"] = True
    return _orig_init(*a, **k)


qd.init = _init_with_profiler

import genesis as gs


def main():
    config = sys.argv[1]
    n_bodies = int(sys.argv[2]) if len(sys.argv) > 2 else 256

    gs.init(backend=gs.cpu, logging_level="warning")
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=1.0 / 60.0, gravity=(0.0, 0.0, -9.8)),
        rigid_options=gs.options.RigidOptions(
            use_contact_island=config in ("island", "both"),
            constraint_solver=gs.constraint_solver.Newton,
            sparse_solve=config in ("sparse", "both"),
            max_collision_pairs=8000,
        ),
        show_viewer=False,
    )
    scene.add_entity(gs.morphs.Plane())
    n_side = math.ceil(math.sqrt(n_bodies))
    placed = 0
    for i in range(n_side):
        for j in range(n_side):
            if placed >= n_bodies:
                break
            scene.add_entity(gs.morphs.Box(size=(0.1, 0.1, 0.1), pos=(i * 0.5, j * 0.5, 0.08)))
            placed += 1
    scene.build(n_envs=1)

    for _ in range(10):
        scene.step()
    qd.profiler.clear_kernel_profiler_info()
    for _ in range(20):
        scene.step()
    print(f"\n===== config={config} n_bodies={n_bodies} =====")
    qd.profiler.print_kernel_profiler_info()


if __name__ == "__main__":
    main()
