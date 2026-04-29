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
def test_backend_switching_ndarray_field_ndarray(backend):
    """Three consecutive init/destroy cycles: ndarray -> field -> ndarray.

    Each cycle builds a rigid-body scene (box on plane, 10 steps) and verifies
    that _tensor_backend() and V/V_VEC resolve the correct backend.

    Note: field-first -> ndarray currently fails due to a Quadrants qd.reset()
    limitation with AnyArray wrapping; starting with ndarray avoids this.
    """
    _run_cycle(use_nd=True, cycle_idx=0)
    _run_cycle(use_nd=False, cycle_idx=1)
    _run_cycle(use_nd=True, cycle_idx=2)
