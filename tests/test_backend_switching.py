"""Test that gs.init() / gs.destroy() can cycle between field and ndarray backends."""

import os
import pathlib
import subprocess
import sys

import numpy as np
import pytest
import quadrants as qd

import genesis as gs

RET_SUCCESS = 42
RET_SKIP = 43

FILE_PATH = pathlib.Path(__file__)
MODULE_ROOT_DIR = FILE_PATH.parents[1]
MODULE = ".".join((FILE_PATH.parent.name, FILE_PATH.stem))


@pytest.mark.parametrize("backend", [None])  # Disable genesis initialization at worker level
@pytest.mark.parametrize(
    "order",
    [
        (True, False, True),
        (False, True, False),
    ],
    ids=["ndarray-field-ndarray", "field-ndarray-field"],
)
def test_backend_switching(backend, order):
    """Three consecutive init/destroy cycles switching between backends.

    Each cycle builds a rigid-body scene (box on plane, 10 steps) and verifies
    that _tensor_backend() and V/V_VEC resolve the correct backend.
    """
    for cycle_idx, use_nd in enumerate(order):
        old_val = os.environ.get("GS_ENABLE_NDARRAY")
        os.environ["GS_ENABLE_NDARRAY"] = "1" if use_nd else "0"

        try:
            gs.init(backend=gs.cpu, seed=0)

            assert gs.use_ndarray == use_nd, f"Cycle {cycle_idx}: expected use_ndarray={use_nd}, got {gs.use_ndarray}"

            from genesis.utils.array_class import V, V_VEC, _tensor_backend

            expected_backend = qd.Backend.NDARRAY if use_nd else qd.Backend.FIELD
            assert _tensor_backend() == expected_backend, (
                f"Cycle {cycle_idx}: expected _tensor_backend()={expected_backend}, got {_tensor_backend()}"
            )

            t = V(qd.i32, (4,))
            t.fill(cycle_idx + 1)
            arr = t.to_numpy()
            np.testing.assert_array_equal(arr, np.full(4, cycle_idx + 1))

            v = V_VEC(3, qd.f32, (2,))
            assert v.to_numpy().shape == (2, 3), f"Cycle {cycle_idx}: unexpected V_VEC shape {v.to_numpy().shape}"

            scene = gs.Scene(show_viewer=False)
            scene.add_entity(gs.morphs.Plane())
            scene.add_entity(gs.morphs.Box(size=(0.4, 0.4, 0.4), pos=(0.0, 0.0, 0.5)))
            scene.build()
            for _ in range(10):
                scene.step()

        finally:
            gs.destroy()
            if old_val is None:
                os.environ.pop("GS_ENABLE_NDARRAY", None)
            else:
                os.environ["GS_ENABLE_NDARRAY"] = old_val


@pytest.mark.parametrize("backend", [None])
def test_set_gravity_accepts_field_and_tensor():
    """set_gravity uses ``gravity: qd.Tensor`` annotation which must accept both a raw qd.field() (subclass solvers
    like MPM) and a qd.Tensor wrapper (base_solver / rigid solver).
    """
    os.environ["GS_ENABLE_NDARRAY"] = "0"
    try:
        gs.init(backend=gs.cpu, seed=0)

        scene = gs.Scene(
            show_viewer=False,
            rigid_options=gs.options.RigidOptions(gravity=(0.0, 0.0, -9.81)),
            mpm_options=gs.options.MPMOptions(gravity=(0.0, 0.0, -9.81)),
        )
        scene.add_entity(gs.morphs.Plane())
        scene.add_entity(gs.morphs.Box(size=(0.4, 0.4, 0.4), pos=(0.0, 0.0, 0.5)))
        scene.add_entity(gs.morphs.Sphere(pos=(0.0, 0.0, 0.5), radius=0.1), material=gs.materials.MPM.Liquid())
        scene.build()

        new_gravity = [0.0, 0.0, -5.0]

        # Rigid solver: _gravity is a qd.Tensor (from base_solver.build via V())
        rigid = scene.sim.rigid_solver
        assert isinstance(rigid._gravity, qd.Tensor), f"Expected qd.Tensor, got {type(rigid._gravity)}"
        rigid.set_gravity(new_gravity)
        np.testing.assert_allclose(rigid.get_gravity(), new_gravity, atol=1e-6)

        # MPM solver: _gravity is a raw qd.field() (subclass override)
        mpm = scene.sim.mpm_solver
        assert isinstance(mpm._gravity, qd.Field), f"Expected qd.Field, got {type(mpm._gravity)}"
        mpm.set_gravity(new_gravity)
        np.testing.assert_allclose(mpm._gravity.to_numpy().flatten(), new_gravity, atol=1e-6)

    finally:
        gs.destroy()
        os.environ.pop("GS_ENABLE_NDARRAY", None)


def _basic_sim_child(args: list[str]):
    """Child process: init genesis, build a scene, step, destroy."""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", type=str, choices=["cpu", "gpu"], default="cpu")
    args = parser.parse_args(args)

    backend = getattr(gs.constants.backend, args.backend)
    try:
        gs.utils.get_device(backend)
    except gs.GenesisException:
        print(f"Backend '{backend}' not available on this machine", file=sys.stderr)
        sys.exit(RET_SKIP)

    gs.init(backend=backend, precision="32")

    scene = gs.Scene(show_viewer=False)
    scene.add_entity(gs.morphs.Plane())
    scene.add_entity(gs.morphs.Box(size=(0.4, 0.4, 0.4), pos=(0.0, 0.0, 0.5)))
    scene.build()
    for _ in range(10):
        scene.step()

    sys.exit(RET_SUCCESS)


@pytest.mark.parametrize("backend", [None])
@pytest.mark.parametrize("test_backend", ["cpu"])
@pytest.mark.parametrize("use_ndarray", [False, True])
def test_basic_sim_subprocess(test_backend: str, use_ndarray: bool):
    """Run a basic simulation in a fresh subprocess to verify genesis import + init works cleanly."""
    cmd_line = [
        sys.executable,
        "-m",
        MODULE,
        _basic_sim_child.__name__,
        "--backend",
        test_backend,
    ]
    env = dict(os.environ)
    env["GS_ENABLE_NDARRAY"] = "1" if use_ndarray else "0"

    proc = subprocess.run(cmd_line, capture_output=True, text=True, encoding="utf-8", env=env, cwd=MODULE_ROOT_DIR)
    if proc.returncode == RET_SKIP:
        pytest.skip(proc.stderr)
    elif proc.returncode != RET_SUCCESS:
        print(proc.stdout)
        print("-" * 100)
        print(proc.stderr)
    assert proc.returncode == RET_SUCCESS


if __name__ == "__main__":
    globals()[sys.argv[1]](sys.argv[2:])
