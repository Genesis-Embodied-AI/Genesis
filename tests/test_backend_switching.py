"""Test that gs.init() / gs.destroy() can cycle between field and ndarray backends."""

import os

import numpy as np
import pytest
import quadrants as qd

import genesis as gs


@pytest.mark.parametrize("backend", [None])  # Disable genesis initialization at worker level
def test_backend_switching_field_ndarray_field(backend):
    """Three consecutive init/destroy cycles: field -> ndarray -> field.

    Verifies that _tensor_backend() and V/V_VEC resolve the correct backend
    after each destroy/re-init cycle, and that created tensors work correctly.
    """
    use_ndarray_sequence = [False, True, False]

    for i, use_nd in enumerate(use_ndarray_sequence):
        old_val = os.environ.get("GS_ENABLE_NDARRAY")
        os.environ["GS_ENABLE_NDARRAY"] = "1" if use_nd else "0"

        try:
            gs.init(backend=gs.cpu, seed=0)

            assert gs.use_ndarray == use_nd, f"Cycle {i}: expected use_ndarray={use_nd}, got {gs.use_ndarray}"

            from genesis.utils.array_class import V, V_VEC, _tensor_backend

            expected_backend = qd.Backend.NDARRAY if use_nd else qd.Backend.FIELD
            assert _tensor_backend() == expected_backend, (
                f"Cycle {i}: expected _tensor_backend()={expected_backend}, got {_tensor_backend()}"
            )

            t = V(qd.i32, (4,))
            t.fill(i + 1)
            arr = t.to_numpy()
            np.testing.assert_array_equal(arr, np.full(4, i + 1))

            v = V_VEC(3, qd.f32, (2,))
            assert v.to_numpy().shape == (2, 3), f"Cycle {i}: unexpected V_VEC shape {v.to_numpy().shape}"

        finally:
            gs.destroy()
            if old_val is None:
                os.environ.pop("GS_ENABLE_NDARRAY", None)
            else:
                os.environ["GS_ENABLE_NDARRAY"] = old_val
