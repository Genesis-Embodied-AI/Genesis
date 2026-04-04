"""Instrumented box_pyramid_6 benchmark to diagnose performance regression.

Runs with both monolith and decomposed solver paths (forced via env var),
adds CUDA event timing around solver phases, and captures torch profiler traces.

Usage:
    python bench_bp6_instrumented.py                    # auto solver selection
    QD_PERFDISPATCH_FORCE=func_solve_body:func_solve_body_monolith python bench_bp6_instrumented.py
    QD_PERFDISPATCH_FORCE=func_solve_body:func_solve_decomposed python bench_bp6_instrumented.py
    python bench_bp6_instrumented.py --both             # run both paths sequentially
"""
import sys
import types
import time
import os
import argparse

sys.path.insert(0, "tests")
tests_pkg = types.ModuleType("tests")
tests_pkg.__path__ = ["tests"]
sys.modules["tests"] = tests_pkg

import genesis as gs

gs.vis.visualizer.Visualizer.build = lambda self: None

gs.init()

from tests.test_rigid_benchmarks import make_box_pyramid
import quadrants as qd
import torch

N_ENVS = 4096
WARMUP = 50
MEASURE = 100


def run_benchmark(label):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")

    scene, step_fn, _ = make_box_pyramid(N_ENVS, n_cubes=6)
    qd.sync()

    rigid_solver = scene.rigid_solver
    cs = rigid_solver.constraint_solver

    print(f"n_dofs = {rigid_solver.n_dofs}")
    print(f"n_envs = {N_ENVS}")
    static_cfg = rigid_solver._static_rigid_sim_config
    print(f"enable_tiled_cholesky_hessian = {static_cfg.enable_tiled_cholesky_hessian}")
    print(f"solver_type = {static_cfg.solver_type}")
    print(f"QD_PERFDISPATCH_FORCE = {os.environ.get('QD_PERFDISPATCH_FORCE', '(not set)')}")
    print()

    # --- Warmup ---
    for _ in range(WARMUP):
        step_fn()
    qd.sync()

    # --- Overall FPS ---
    t0 = time.perf_counter()
    for _ in range(MEASURE):
        step_fn()
    qd.sync()
    dt = time.perf_counter() - t0
    fps = MEASURE * N_ENVS / dt
    print(f"Overall FPS: {fps:.0f}  ({dt*1000/MEASURE:.2f} ms/step)")
    print()

    # --- CUDA event timing around solver phases ---
    from genesis.engine.solvers.rigid.constraint import solver as solver_mod

    solve_init_times = []
    solve_body_times = []
    resolve_times = []

    _orig_resolve = cs.resolve.__func__

    def timed_resolve(self):
        start_all = torch.cuda.Event(enable_timing=True)
        end_all = torch.cuda.Event(enable_timing=True)
        start_init = torch.cuda.Event(enable_timing=True)
        end_init = torch.cuda.Event(enable_timing=True)
        start_body = torch.cuda.Event(enable_timing=True)
        end_body = torch.cuda.Event(enable_timing=True)

        start_all.record()
        start_init.record()
        solver_mod.func_solve_init(
            self._solver.dofs_info, self._solver.dofs_state, self._solver.entities_info,
            self.constraint_state, self._solver._rigid_global_info, self._solver._static_rigid_sim_config,
        )
        end_init.record()

        start_body.record()
        solver_mod.func_solve_body(
            self._solver.entities_info, self._solver.dofs_info, self._solver.dofs_state,
            self.constraint_state, self._solver._rigid_global_info, self._solver._static_rigid_sim_config,
            self._n_iterations,
        )
        end_body.record()

        solver_mod.func_update_qacc(
            self._solver.dofs_state, self.constraint_state,
            self._solver._static_rigid_sim_config, self._solver._errno,
        )
        if self._solver._options.noslip_iterations > 0:
            self.noslip()
        solver_mod.func_update_contact_force(
            self._solver.links_state, self._collider._collider_state,
            self.constraint_state, self._solver._static_rigid_sim_config,
        )
        end_all.record()
        torch.cuda.synchronize()

        solve_init_times.append(start_init.elapsed_time(end_init))
        solve_body_times.append(start_body.elapsed_time(end_body))
        resolve_times.append(start_all.elapsed_time(end_all))

    import types as _types
    cs.resolve = _types.MethodType(timed_resolve, cs)

    for _ in range(MEASURE):
        step_fn()

    cs.resolve = _types.MethodType(_orig_resolve, cs)

    if solve_init_times:
        n = len(solve_init_times)
        skip = n // 5
        init_avg = sum(solve_init_times[skip:]) / (n - skip)
        body_avg = sum(solve_body_times[skip:]) / (n - skip)
        resolve_avg = sum(resolve_times[skip:]) / (n - skip)
        print(f"Solver timing (avg of {n-skip} steps, skipping first {skip}):")
        print(f"  func_solve_init:  {init_avg:.3f} ms")
        print(f"  func_solve_body:  {body_avg:.3f} ms")
        print(f"  resolve (total):  {resolve_avg:.3f} ms")
        print()

    # --- Torch profiler ---
    try:
        from torch.profiler import profile, ProfilerActivity
        print("Capturing torch profiler trace (5 steps)...")
        with profile(activities=[ProfilerActivity.CUDA], record_shapes=False, with_stack=False) as prof:
            for _ in range(5):
                step_fn()
        print("\nTop 20 CUDA kernels by total time:")
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
    except Exception as e:
        print(f"Profiler failed: {e}")

    return fps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--both", action="store_true", help="Run both monolith and decomposed paths")
    args = parser.parse_args()

    if args.both:
        results = {}
        for solver_name, env_val in [
            ("monolith", "func_solve_body:func_solve_body_monolith"),
            ("decomposed", "func_solve_body:func_solve_decomposed"),
        ]:
            os.environ["QD_PERFDISPATCH_FORCE"] = env_val
            results[solver_name] = run_benchmark(f"Solver: {solver_name} (forced)")

        print(f"\n{'='*60}")
        print("  SUMMARY")
        print(f"{'='*60}")
        for name, fps in results.items():
            print(f"  {name:15s}: {fps:.0f} FPS")
    else:
        run_benchmark("Solver: auto (or forced via QD_PERFDISPATCH_FORCE)")


if __name__ == "__main__":
    main()
