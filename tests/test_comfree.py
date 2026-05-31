"""Tests for the ComFree (Complementarity-Free) constraint solver.

Validates that the ComFree solver:
1. Can be instantiated and stepped without errors
2. Produces physically plausible results (objects fall, contacts prevent penetration)
3. Reproduces analytic free-fall when there are no contacts
4. Works across multiple parallel environments and simple stacking

These tests rely on the session-scoped ``initialize_genesis`` autouse fixture
(see ``tests/conftest.py``) for ``gs.init`` / ``gs.destroy`` and the parametrized
backend / precision, so they must not initialize Genesis themselves.
"""

import numpy as np
import pytest

import genesis as gs


def _make_scene(show_viewer, dt=0.002, stiffness=0.3, damping=0.005, n_envs=0, enable_collision=True):
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=dt, gravity=(0, 0, -9.81)),
        rigid_options=gs.options.RigidOptions(
            dt=dt,
            constraint_solver=gs.constraint_solver.ComFree,
            comfree_stiffness=stiffness,
            comfree_damping=damping,
            enable_collision=enable_collision,
            enable_joint_limit=True,
        ),
        show_viewer=show_viewer,
    )
    scene.add_entity(gs.morphs.Plane())
    box = scene.add_entity(
        gs.morphs.Box(size=(0.2, 0.2, 0.2), pos=(0, 0, 0.5)),
        material=gs.materials.Rigid(rho=1000),
    )
    scene.build(n_envs=n_envs)
    return scene, box


@pytest.mark.required
def test_comfree_instantiation(show_viewer):
    scene, _ = _make_scene(show_viewer)
    assert scene.sim.rigid_solver.constraint_solver.__class__.__name__ == "ComFreeSolver"


@pytest.mark.required
def test_comfree_step(show_viewer):
    scene, _ = _make_scene(show_viewer)
    for _ in range(10):
        scene.step()


def test_box_falls_under_gravity(show_viewer):
    scene, box = _make_scene(show_viewer)
    initial_z = float(box.get_pos().numpy()[2])
    for _ in range(5):
        scene.step()
    z = float(box.get_pos().numpy()[2])
    assert z < initial_z, f"Box z={z} should be less than initial z={initial_z}"


def test_box_contacts_floor(show_viewer):
    scene, box = _make_scene(show_viewer)
    for _ in range(500):
        scene.step()
    z = float(box.get_pos().numpy()[2])
    assert z > -0.05, f"Box fell through floor: z={z}"
    assert z < 0.2, f"Box didn't settle: z={z}"


def test_freefall_matches_analytic(show_viewer):
    """Without contacts ComFree must integrate gravity exactly like the smooth dynamics."""
    dt = 0.01
    z0 = 2.0
    n_steps = 50
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=dt, gravity=(0, 0, -9.81)),
        rigid_options=gs.options.RigidOptions(
            dt=dt,
            constraint_solver=gs.constraint_solver.ComFree,
            enable_collision=False,
        ),
        show_viewer=show_viewer,
    )
    box = scene.add_entity(
        gs.morphs.Box(size=(0.2, 0.2, 0.2), pos=(0, 0, z0)),
        material=gs.materials.Rigid(rho=1000),
    )
    scene.build()
    for _ in range(n_steps):
        scene.step()
    z = float(box.get_pos().numpy()[2])
    t = n_steps * dt
    expected_z = z0 - 0.5 * 9.81 * t * t
    assert abs(z - expected_z) < 0.15, f"Free-fall z={z}, expected ~{expected_z}"


def test_batched_simulation(show_viewer):
    """ComFree works with multiple parallel environments."""
    n_envs = 4
    scene, box = _make_scene(show_viewer, n_envs=n_envs)
    for _ in range(300):
        scene.step()
    pos = box.get_pos().numpy()
    assert pos.shape == (n_envs, 3)
    for i in range(n_envs):
        assert pos[i, 2] > -0.05, f"Env {i}: z={pos[i, 2]}"
        assert pos[i, 2] < 0.2, f"Env {i}: z={pos[i, 2]}"


def test_two_box_stack(show_viewer):
    """Two boxes should stack on top of each other without penetrating the floor."""
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.002, gravity=(0, 0, -9.81)),
        rigid_options=gs.options.RigidOptions(
            dt=0.002,
            constraint_solver=gs.constraint_solver.ComFree,
            comfree_stiffness=0.3,
            comfree_damping=0.005,
            enable_collision=True,
        ),
        show_viewer=show_viewer,
    )
    scene.add_entity(gs.morphs.Plane())
    box1 = scene.add_entity(
        gs.morphs.Box(size=(0.2, 0.2, 0.2), pos=(0, 0, 0.3)),
        material=gs.materials.Rigid(rho=1000),
    )
    box2 = scene.add_entity(
        gs.morphs.Box(size=(0.2, 0.2, 0.2), pos=(0, 0, 0.7)),
        material=gs.materials.Rigid(rho=1000),
    )
    scene.build()
    for _ in range(800):
        scene.step()

    z1 = float(box1.get_pos().numpy()[2])
    z2 = float(box2.get_pos().numpy()[2])
    assert z1 > 0.0, f"Box1 penetrated: z={z1}"
    assert z2 > z1, f"Box2 below box1: z2={z2}, z1={z1}"
    assert z2 < 0.5, f"Box2 too high: z={z2}"
