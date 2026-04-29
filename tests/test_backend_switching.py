"""Test that gs.init() / gs.destroy() can cycle between field and ndarray backends."""

import os

import numpy as np
import pytest
import quadrants as qd

import genesis as gs


def _run_cycle(use_nd, cycle_idx):
    """Run one init/build/step/destroy cycle."""
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
    for i, use_nd in enumerate(order):
        _run_cycle(use_nd=use_nd, cycle_idx=i)


@pytest.mark.parametrize("backend", [None])
def test_set_gravity_accepts_field_and_tensor():
    """set_gravity uses ``gravity: qd.Tensor`` annotation which must accept both a raw qd.field() (subclass solvers)
    and a qd.Tensor wrapper (base_solver).
    """
    os.environ["GS_ENABLE_NDARRAY"] = "0"
    try:
        gs.init(backend=gs.cpu, seed=0)

        scene = gs.Scene(show_viewer=False)
        scene.add_entity(gs.morphs.Plane())
        scene.add_entity(gs.morphs.Box(size=(0.4, 0.4, 0.4), pos=(0.0, 0.0, 0.5)))
        scene.build()

        solver = scene.sim.solvers[0]
        if solver._gravity is None:
            pytest.skip("Solver has no gravity")

        new_gravity = [0.0, 0.0, -5.0]
        solver.set_gravity(new_gravity)
        result = solver.get_gravity()
        np.testing.assert_allclose(result, new_gravity, atol=1e-6)

    finally:
        gs.destroy()
        os.environ.pop("GS_ENABLE_NDARRAY", None)
