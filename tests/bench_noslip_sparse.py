"""Benchmark: sparse_solve impact on noslip performance.

Compares simulation time with sparse_solve=False vs sparse_solve=True
on a table scene with noslip_iterations=2 across different batch sizes.
The sparse advantage grows with scene complexity (more DOFs per env).

Usage:
    python tests/bench_noslip_sparse.py
"""

import subprocess
import sys
import os
import tempfile

_MARVIN_URDF = "simvla-genesis-test-scene/assets/robot/marvin_bimanual/urdf/marvin_pika.urdf"

_BENCH_SCRIPT = """
import time
import math
import genesis as gs
import numpy as np

sparse = "--sparse" in __import__("sys").argv
n_envs = int(__import__("sys").argv[__import__("sys").argv.index("--n_envs") + 1])

gs.init(backend=gs.cpu, seed=0, precision="64", logging_level="warning")

TABLE_Z = 0.762
PLACE_Z = TABLE_Z + 0.08

scene = gs.Scene(
    sim_options=gs.options.SimOptions(dt=1.0 / 30, substeps=4, gravity=(0, 0, -9.81)),
    rigid_options=gs.options.RigidOptions(
        noslip_iterations=2,
        max_collision_pairs=128,
        sparse_solve=sparse,
    ),
    show_viewer=False,
    profiling_options=gs.options.ProfilingOptions(show_FPS=False),
)

{robot_block}

# Table
scene.add_entity(morph=gs.morphs.Box(
    pos=(0.5, 0, TABLE_Z / 2), size=(0.8, 0.6, TABLE_Z / 2), fixed=True))

# 16 objects on the table
n_objects = 16
cols = max(1, int(math.sqrt(n_objects * 1.5)))
for i in range(n_objects):
    x = 0.25 + 0.12 * (i % cols)
    y = -0.25 + 0.12 * (i // cols)
    scene.add_entity(
        material=gs.materials.Rigid(friction=0.5),
        morph=gs.morphs.Box(pos=(x, y, PLACE_Z), size=(0.04, 0.04, 0.04)),
    )

scene.build(n_envs=n_envs)

# Warmup
WARMUP_SECS = 3.0
t0 = time.time()
while time.time() - t0 < WARMUP_SECS:
    scene.step()

# Timed run
RECORD_SECS = 5.0
n_steps = 0
t0 = time.time()
while time.time() - t0 < RECORD_SECS:
    scene.step()
    n_steps += 1
elapsed = time.time() - t0

label = "sparse=True " if sparse else "sparse=False"
env_steps = n_steps * max(n_envs, 1)
fps = env_steps / elapsed
ms = elapsed / n_steps * 1000
print(f"BENCH {label} n_envs={n_envs}: {n_steps} steps in {elapsed:.1f}s, {fps:.0f} env-FPS, {ms:.2f} ms/step")
"""

_MARVIN_ROBOT = """
scene.add_entity(
    morph=gs.morphs.URDF(
        file="{urdf}",
        pos=(0, 0, 1.08), fixed=True, convexify=True, merge_fixed_links=False,
    ),
)
""".replace("{urdf}", _MARVIN_URDF)

_PANDA_ROBOTS = """
# Two Panda arms for more DOFs
scene.add_entity(morph=gs.morphs.URDF(
    file="urdf/panda_bullet/panda.urdf", pos=(0, -0.2, TABLE_Z), fixed=True))
scene.add_entity(morph=gs.morphs.URDF(
    file="urdf/panda_bullet/panda.urdf", pos=(0, 0.2, TABLE_Z), euler=(0,0,180), fixed=True))
"""


def run_bench(script_path, sparse, n_envs):
    cmd = [sys.executable, script_path, "--n_envs", str(n_envs)]
    if sparse:
        cmd.append("--sparse")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    for line in (result.stdout + result.stderr).split("\n"):
        if "BENCH" in line:
            return line.strip()
    return None


if __name__ == "__main__":
    has_marvin = os.path.exists(_MARVIN_URDF)
    robot_name = "marvin bimanual" if has_marvin else "2x Panda"
    robot_block = _MARVIN_ROBOT if has_marvin else _PANDA_ROBOTS

    script_code = _BENCH_SCRIPT.replace("{robot_block}", robot_block)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(script_code)
        script = f.name

    print("=" * 65)
    print("Benchmark: sparse_solve impact on noslip performance")
    print(f"Robot: {robot_name} + 16 boxes, noslip_iterations=2")
    print("Warmup: 3s, Record: 5s per config")
    print("=" * 65)
    print(f"{'n_envs':>7s}  {'dense (ms)':>10s}  {'sparse (ms)':>11s}  {'speedup':>8s}")
    print("-" * 42)

    for n_envs in [1, 2, 4, 8]:
        results = {}
        for sparse in [False, True]:
            line = run_bench(script, sparse, n_envs)
            if line:
                ms = float(line.split("ms/step")[0].split()[-1])
                results[sparse] = ms

        if False in results and True in results:
            speedup = (1 - results[True] / results[False]) * 100
            print(f"{n_envs:>7d}  {results[False]:>10.2f}  {results[True]:>11.2f}  {speedup:>7.1f}%")
        else:
            print(f"{n_envs:>7d}  FAILED")

    os.unlink(script)
