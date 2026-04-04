"""Benchmark H patching vs full rebuild on dex_hand."""
import sys
import types
import time
import os

sys.path.insert(0, "tests")
tests_pkg = types.ModuleType("tests")
tests_pkg.__path__ = ["tests"]
sys.modules["tests"] = tests_pkg

import genesis as gs

# Monkey-patch the visualizer build to skip OpenGL/EGL on headless nodes
_orig_vis_build = gs.vis.visualizer.Visualizer.build
def _noop_build(self):
    pass
gs.vis.visualizer.Visualizer.build = _noop_build

gs.init()

from tests.test_rigid_benchmarks import make_dex_hand
import quadrants as qd

N_ENVS = 4096
WARMUP = 100
MEASURE = 200

scene, step_fn, _ = make_dex_hand(N_ENVS)
qd.sync()

for i in range(WARMUP):
    step_fn()
qd.sync()

t0 = time.perf_counter()
for i in range(MEASURE):
    step_fn()
qd.sync()
dt = time.perf_counter() - t0

fps = MEASURE * N_ENVS / dt
print(f"FPS: {fps:.0f}  ({dt*1000/MEASURE:.2f} ms/step)")
