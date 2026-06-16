"""Phase-0 ship/kill benchmark for the island reboot.

Thesis: partitioning the global Newton solve into per-island blocks cuts the dense Cholesky from one
O(n_dofs^3) factorization to sum_i O(n_dofs_i^3). On a many-independent-body scene (N boxes on a ground
plane -> N islands of 6 dofs each) that is an ~N^2 reduction in factorization FLOPs, BEFORE any per-island
GPU parallelism. If the functional GPU monolith-island path does not already beat islands-off here, the
Step-3b per-island-parallel re-architecture is not worth its cost.

Run one configuration per process (monolith arm forced so the only difference is island decomposition):
  QD_PERFDISPATCH_FORCE=func_solve_body:func_solve_body_monolith python tests/bench_island_phase0.py <on|off> <n_side>
"""

import sys
import time

import genesis as gs


def main():
    use_island = sys.argv[1] == "on"
    n_side = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    n_envs = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    backend = gs.cpu if (len(sys.argv) > 4 and sys.argv[4] == "cpu") else gs.gpu
    solver = gs.constraint_solver.CG if (len(sys.argv) > 5 and sys.argv[5] == "cg") else gs.constraint_solver.Newton
    spacing = 0.5
    # Fewer steps where each step is heavy (CPU scalar, or many envs).
    if backend == gs.cpu:
        n_warmup, n_measure = 5, 12
    elif n_envs > 16:
        n_warmup, n_measure = 10, 30
    else:
        n_warmup, n_measure = 30, 120

    gs.init(backend=backend, logging_level="warning")
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=1.0 / 60.0,
            gravity=(0.0, 0.0, -9.8),
        ),
        rigid_options=gs.options.RigidOptions(
            use_contact_island=use_island,
            constraint_solver=solver,
            sparse_solve=False,
        ),
        show_viewer=False,
    )
    scene.add_entity(gs.morphs.Plane())
    for i in range(n_side):
        for j in range(n_side):
            scene.add_entity(gs.morphs.Box(size=(0.1, 0.1, 0.1), pos=(i * spacing, j * spacing, 0.08)))
    scene.build(n_envs=n_envs)

    for _ in range(n_warmup):
        scene.step()
    t0 = time.perf_counter()
    for _ in range(n_measure):
        scene.step()
    per_step = (time.perf_counter() - t0) / n_measure

    label = "island ON " if use_island else "island OFF"
    print(f"RESULT {label} n_bodies={n_side * n_side} {per_step * 1e3:.4f} ms/step {1.0 / per_step:.1f} steps/s")


if __name__ == "__main__":
    main()
