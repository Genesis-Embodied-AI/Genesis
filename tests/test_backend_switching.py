"""Test that gs.init() / gs.destroy() can cycle between field and ndarray backends."""

import os

import numpy as np
import pytest
import quadrants as qd

import genesis as gs


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
