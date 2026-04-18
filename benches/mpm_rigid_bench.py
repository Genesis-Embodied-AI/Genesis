"""
MPM + Rigid benchmark harness.

Drives a deterministic Franka-squeezes-elastic-cube scene (adapted from
examples/coupling/grasp_soft_cube.py) and reports wall-clock per step,
peak GPU memory, and a correctness fingerprint (mean particle position)
so variants can be compared against a baseline run.

Usage:
    python benches/mpm_rigid_bench.py --label baseline --warmup 20 --steps 200
    python benches/mpm_rigid_bench.py --label plan_a  --warmup 20 --steps 200 \
        --out bench_results.csv

Append the same --out path to accumulate a CSV across commits.
"""

import argparse
import csv
import os
import subprocess
import sys
import time
from pathlib import Path

# Ensure the editable repo root takes priority over any site-packages `genesis` namespace
# stub (common when a prior `pip install genesis-world` left behind an assets/ext dir). Without
# this, `import genesis` may resolve to an empty namespace package that lacks `gs.init`.
_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np
import torch

import genesis as gs


def _git_sha():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parent.parent,
            stderr=subprocess.DEVNULL,
            encoding="utf-8",
        ).strip()
    except Exception:
        return "unknown"


def _ik_pos(n_envs: int, z: float):
    p = np.array([0.64, 0.0, z])
    return np.tile(p, (n_envs, 1)) if n_envs > 0 else p


def _ik_quat(n_envs: int):
    q = np.array([0.0, 1.0, 0.0, 0.0])
    return np.tile(q, (n_envs, 1)) if n_envs > 0 else q


def build_scene(n_envs: int, grid_density: int, substeps: int, cube_size: float = 0.04):
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=5e-3, substeps=substeps),
        mpm_options=gs.options.MPMOptions(
            lower_bound=(0.55, -0.1, -0.05),
            upper_bound=(0.75, 0.1, 0.3),
            grid_density=grid_density,
        ),
        show_viewer=False,
    )
    scene.add_entity(gs.morphs.Plane())
    cube = scene.add_entity(
        material=gs.materials.MPM.Elastic(),
        morph=gs.morphs.Box(
            size=(cube_size, cube_size, cube_size),
            pos=(0.65, 0.0, cube_size / 2 + 0.005),
        ),
    )
    franka = scene.add_entity(
        gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"),
        material=gs.materials.Rigid(coup_friction=1.0),
    )
    scene.build(n_envs=n_envs)

    franka.set_dofs_kp(np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100]))
    franka.set_dofs_kv(np.array([450, 450, 350, 350, 200, 200, 200, 10, 10]))
    franka.set_dofs_force_range(
        np.array([-87, -87, -87, -87, -12, -12, -12, -100, -100]),
        np.array([87, 87, 87, 87, 12, 12, 12, 100, 100]),
    )
    end_effector = franka.get_link("hand")
    qpos = franka.inverse_kinematics(
        link=end_effector,
        pos=_ik_pos(n_envs, 0.135),
        quat=_ik_quat(n_envs),
    )
    # qpos shape: (n_envs, 9) if batched, else (9,)
    if n_envs > 0:
        qpos[:, -2:] = 0.03
        franka.set_dofs_position(qpos[:, :-2], np.arange(7))
        franka.set_dofs_position(qpos[:, -2:], np.arange(7, 9))
        franka.control_dofs_position(qpos[:, :-2], np.arange(7))
        franka.control_dofs_force(np.tile(np.array([-1.0, -1.0]), (n_envs, 1)), np.arange(7, 9))
    else:
        qpos[-2:] = 0.03
        franka.set_dofs_position(qpos[:-2], np.arange(7))
        franka.set_dofs_position(qpos[-2:], np.arange(7, 9))
        franka.control_dofs_position(qpos[:-2], np.arange(7))
        franka.control_dofs_force(np.array([-1.0, -1.0]), np.arange(7, 9))

    return scene, cube, franka, end_effector


def drive_step(i: int, n_envs: int, franka, end_effector):
    """Deterministic motion: 100 settle, 300 lift, 100 hold."""
    if i < 100:
        return
    if i < 400:
        lift_i = i - 100
        z = 0.135 + 0.0005 * lift_i
        qpos = franka.inverse_kinematics(
            link=end_effector,
            pos=_ik_pos(n_envs, z),
            quat=_ik_quat(n_envs),
        )
        if n_envs > 0:
            franka.control_dofs_position(qpos[:, :-2], np.arange(7))
        else:
            franka.control_dofs_position(qpos[:-2], np.arange(7))


def fingerprint(cube) -> tuple[float, float, float]:
    """Mean particle position as a coarse correctness fingerprint."""
    poss = cube.get_particles_pos()
    if poss.ndim == 3:
        poss = poss[0]
    mean = poss.mean(dim=0).detach().cpu().numpy()
    return float(mean[0]), float(mean[1]), float(mean[2])


def run(
    label: str,
    steps: int,
    warmup: int,
    n_envs: int,
    grid_density: int,
    substeps: int,
    out: str | None,
    cube_size: float,
):
    gs.init(backend=gs.gpu, logging_level="warning")

    scene, cube, franka, end_effector = build_scene(n_envs, grid_density, substeps, cube_size=cube_size)
    n_particles = cube.n_particles

    # Warmup (JIT compile kernels, settle contact).
    for i in range(warmup):
        drive_step(i, n_envs, franka, end_effector)
        scene.step()

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    t0 = time.perf_counter()
    for i in range(warmup, warmup + steps):
        drive_step(i, n_envs, franka, end_effector)
        scene.step()
    torch.cuda.synchronize()
    t1 = time.perf_counter()

    wall = t1 - t0
    peak_mib = torch.cuda.max_memory_allocated() / (1024**2)
    fp = fingerprint(cube)

    step_ms = wall / steps * 1000.0
    substep_ms = step_ms / substeps
    print(f"[{label}] steps={steps} substeps={substeps} envs={n_envs} particles={n_particles}")
    print(f"  wall            : {wall:.3f} s")
    print(f"  per-step        : {step_ms:.3f} ms")
    print(f"  per-substep     : {substep_ms:.3f} ms")
    print(f"  peak GPU mem    : {peak_mib:.1f} MiB")
    print(f"  fingerprint xyz : ({fp[0]:.6f}, {fp[1]:.6f}, {fp[2]:.6f})")

    if out:
        new_file = not os.path.exists(out)
        with open(out, "a", newline="") as f:
            w = csv.writer(f)
            if new_file:
                w.writerow(
                    [
                        "label",
                        "git_sha",
                        "n_envs",
                        "grid_density",
                        "substeps",
                        "n_particles",
                        "steps",
                        "wall_s",
                        "step_ms",
                        "substep_ms",
                        "peak_mem_MiB",
                        "fp_x",
                        "fp_y",
                        "fp_z",
                    ]
                )
            w.writerow(
                [
                    label,
                    _git_sha(),
                    n_envs,
                    grid_density,
                    substeps,
                    n_particles,
                    steps,
                    f"{wall:.6f}",
                    f"{step_ms:.4f}",
                    f"{substep_ms:.4f}",
                    f"{peak_mib:.2f}",
                    f"{fp[0]:.6f}",
                    f"{fp[1]:.6f}",
                    f"{fp[2]:.6f}",
                ]
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", default="run")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--grid-density", type=int, default=128)
    parser.add_argument("--substeps", type=int, default=15)
    parser.add_argument("--cube-size", type=float, default=0.04)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    run(
        label=args.label,
        steps=args.steps,
        warmup=args.warmup,
        n_envs=args.n_envs,
        grid_density=args.grid_density,
        substeps=args.substeps,
        out=args.out,
        cube_size=args.cube_size,
    )


if __name__ == "__main__":
    main()
