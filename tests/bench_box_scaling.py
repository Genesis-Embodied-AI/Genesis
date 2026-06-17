"""Scaling profile: N disconnected boxes on a plane (each box -> its own 6-dof island).

Compares the three CPU solve strategies as N grows, to check whether step time scales linearly:
  - dense : one O(n_dofs^3) Cholesky over the whole system (use_contact_island=False, sparse_solve=False)
  - island: per-island block isolation, sum_k O(n_k^3) (use_contact_island=True, sparse_solve=False)
  - sparse: skyline-envelope Cholesky (use_contact_island=False, sparse_solve=True)

Disconnected boxes make the global Hessian block-diagonal, so islands are exactly linear in N. Sparse is
linear only if the fill-reducing permutation keeps the envelope tight; a loose envelope scales superlinearly.

  python tests/bench_box_scaling.py <dense|island|sparse> <n_bodies> [cpu|gpu]
"""

import math
import sys
import time

import genesis as gs


def main():
    config = sys.argv[1]
    n_bodies = int(sys.argv[2]) if len(sys.argv) > 2 else 64
    backend = gs.gpu if (len(sys.argv) > 3 and sys.argv[3] == "gpu") else gs.cpu

    use_island = config in ("island", "both")
    sparse_solve = config in ("sparse", "both")
    spacing = 0.5

    gs.init(backend=backend, logging_level="warning")
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=1.0 / 60.0,
            gravity=(0.0, 0.0, -9.8),
        ),
        rigid_options=gs.options.RigidOptions(
            use_contact_island=use_island,
            constraint_solver=gs.constraint_solver.Newton,
            sparse_solve=sparse_solve,
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
            scene.add_entity(gs.morphs.Box(size=(0.1, 0.1, 0.1), pos=(i * spacing, j * spacing, 0.08)))
            placed += 1
    scene.build(n_envs=1)

    solver = scene.rigid_solver
    eff = f"island={solver._use_contact_island} sparse={solver.constraint_solver.sparse_solve}"

    n_warmup, n_measure = 5, 10
    for _ in range(n_warmup):
        scene.step()
    t0 = time.perf_counter()
    for _ in range(n_measure):
        scene.step()
    per_step = (time.perf_counter() - t0) / n_measure

    print(f"RESULT config={config} n_bodies={n_bodies} n_dofs={solver.n_dofs} {per_step * 1e3:.3f} ms/step [{eff}]")


if __name__ == "__main__":
    main()
