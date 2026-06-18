"""High-dofs-per-env island crossover bench: 1 env, N boxes -> 6N dofs in one Hessian.

Tests the hypothesis that islands win once the whole-env Hessian no longer fits GPU shared memory
(hessian_fits_shared False), forcing the non-island global solve off the fast tiled path while each
small island stays on it.

  python tests/bench_island_highdof.py <dense|island> <n_bodies>
  # cooperative-island arm: prefix with QD_PERFDISPATCH_FORCE=func_solve_body:func_solve_decomposed
"""

import math
import sys
import time

import genesis as gs


def main():
    config = sys.argv[1]
    n_bodies = int(sys.argv[2])

    gs.init(backend=gs.gpu, logging_level="warning")
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=1.0 / 60.0, gravity=(0.0, 0.0, -9.8)),
        rigid_options=gs.options.RigidOptions(
            use_contact_island=config == "island",
            constraint_solver=gs.constraint_solver.Newton,
            sparse_solve=False,
            max_collision_pairs=max(8 * n_bodies, 8000),
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

    solver = scene.rigid_solver
    cfg = solver._static_rigid_sim_config
    fits = cfg.hessian_fits_shared
    isl = solver._use_contact_island

    for _ in range(3):
        scene.step()
    t0 = time.perf_counter()
    for _ in range(8):
        scene.step()
    per_step = (time.perf_counter() - t0) / 8

    print(
        f"RESULT config={config} n_bodies={n_bodies} n_dofs={solver.n_dofs} "
        f"island={isl} hessian_fits_shared={fits} {per_step * 1e3:.3f} ms/step"
    )


if __name__ == "__main__":
    main()
