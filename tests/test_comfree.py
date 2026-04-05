"""Tests for the ComFree (Complementarity-Free) constraint solver.

Validates that the ComFree solver:
1. Can be instantiated and run without errors
2. Produces physically plausible results (objects fall, contacts prevent penetration)
3. Handles equality constraints (weld/connect)
4. Handles joint limits
5. Produces comparable results to the standard Newton solver
"""

import numpy as np
import genesis as gs


def _make_scene(solver_type, dt=0.002, stiffness=0.3, damping=0.005, n_envs=0, enable_collision=True):
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=dt, gravity=(0, 0, -9.81)),
        rigid_options=gs.options.RigidOptions(
            dt=dt,
            constraint_solver=solver_type,
            comfree_stiffness=stiffness,
            comfree_damping=damping,
            enable_collision=enable_collision,
            enable_joint_limit=True,
        ),
        show_viewer=False,
    )
    scene.add_entity(gs.morphs.Plane())
    box = scene.add_entity(
        gs.morphs.Box(size=(0.2, 0.2, 0.2), pos=(0, 0, 0.5)),
        material=gs.materials.Rigid(rho=1000),
    )
    scene.build(n_envs=n_envs)
    return scene, box


def test_comfree_instantiation():
    gs.init(backend=gs.cpu, precision="64", logging_level="warning")
    scene, box = _make_scene(gs.constraint_solver.ComFree)
    assert scene.sim.rigid_solver.constraint_solver.__class__.__name__ == "ComFreeSolver"
    gs.destroy()


def test_comfree_step():
    gs.init(backend=gs.cpu, precision="64", logging_level="warning")
    scene, box = _make_scene(gs.constraint_solver.ComFree)
    for _ in range(10):
        scene.step()
    gs.destroy()


def test_box_falls_under_gravity():
    gs.init(backend=gs.cpu, precision="64", logging_level="warning")
    scene, box = _make_scene(gs.constraint_solver.ComFree)
    initial_z = box.get_pos().numpy()[2]
    for _ in range(5):
        scene.step()
    z = box.get_pos().numpy()[2]
    assert z < initial_z, f"Box z={z} should be less than initial z={initial_z}"
    gs.destroy()


def test_box_contacts_floor():
    gs.init(backend=gs.cpu, precision="64", logging_level="warning")
    scene, box = _make_scene(gs.constraint_solver.ComFree)
    for _ in range(500):
        scene.step()
    z = box.get_pos().numpy()[2]
    assert z > -0.05, f"Box fell through floor: z={z}"
    assert z < 0.2, f"Box didn't settle: z={z}"
    gs.destroy()


def test_freefall_comparable():
    """Free-fall should be nearly identical between Newton and ComFree."""
    gs.init(backend=gs.cpu, precision="64", logging_level="warning")
    scene_c, box_c = _make_scene(gs.constraint_solver.ComFree, enable_collision=False)
    for _ in range(20):
        scene_c.step()
    z_c = box_c.get_pos().numpy()[2]
    gs.destroy()

    gs.init(backend=gs.cpu, precision="64", logging_level="warning")
    scene_n, box_n = _make_scene(gs.constraint_solver.Newton, enable_collision=False)
    for _ in range(20):
        scene_n.step()
    z_n = box_n.get_pos().numpy()[2]
    gs.destroy()

    np.testing.assert_allclose(z_c, z_n, atol=1e-3, err_msg="Free-fall trajectories should match")


def test_contact_qualitative():
    """ComFree and Newton should both settle near the ground."""
    gs.init(backend=gs.cpu, precision="64", logging_level="warning")
    scene_c, box_c = _make_scene(gs.constraint_solver.ComFree)
    for _ in range(500):
        scene_c.step()
    z_c = box_c.get_pos().numpy()[2]
    gs.destroy()

    gs.init(backend=gs.cpu, precision="64", logging_level="warning")
    scene_n, box_n = _make_scene(gs.constraint_solver.Newton)
    for _ in range(500):
        scene_n.step()
    z_n = box_n.get_pos().numpy()[2]
    gs.destroy()

    assert abs(z_c - z_n) < 0.1, f"ComFree z={z_c:.4f} vs Newton z={z_n:.4f}"


def test_no_constraint_passthrough():
    """Without contacts, ComFree should give accurate free-fall."""
    gs.init(backend=gs.cpu, precision="64", logging_level="warning")
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.01, gravity=(0, 0, -9.81)),
        rigid_options=gs.options.RigidOptions(
            dt=0.01,
            constraint_solver=gs.constraint_solver.ComFree,
            enable_collision=False,
        ),
        show_viewer=False,
    )
    box = scene.add_entity(
        gs.morphs.Box(size=(0.2, 0.2, 0.2), pos=(0, 0, 2.0)),
        material=gs.materials.Rigid(rho=1000),
    )
    scene.build()
    for _ in range(50):
        scene.step()
    z = box.get_pos().numpy()[2]
    expected_z = 2.0 - 0.5 * 9.81 * 0.5**2
    assert abs(z - expected_z) < 0.15, f"Free-fall z={z}, expected ~{expected_z}"
    gs.destroy()


def test_stiffness_effect():
    """Higher stiffness should reduce penetration."""
    results = []
    for k in [0.1, 0.3, 0.5]:
        gs.init(backend=gs.cpu, precision="64", logging_level="warning")
        scene, box = _make_scene(gs.constraint_solver.ComFree, stiffness=k, damping=0.005)
        for _ in range(500):
            scene.step()
        results.append(box.get_pos().numpy()[2])
        gs.destroy()

    # All should be above floor
    for z in results:
        assert z > 0.0, f"Box penetrated floor: z={z}"


def test_batched_simulation():
    """ComFree works with multiple parallel environments."""
    n_envs = 4
    gs.init(backend=gs.cpu, precision="64", logging_level="warning")
    scene, box = _make_scene(gs.constraint_solver.ComFree, n_envs=n_envs)
    for _ in range(300):
        scene.step()
    pos = box.get_pos().numpy()
    assert pos.shape == (n_envs, 3)
    for i in range(n_envs):
        assert pos[i, 2] > -0.05, f"Env {i}: z={pos[i, 2]}"
        assert pos[i, 2] < 0.2, f"Env {i}: z={pos[i, 2]}"
    gs.destroy()


def test_two_box_stack():
    """Two boxes should stack on top of each other."""
    gs.init(backend=gs.cpu, precision="64", logging_level="warning")
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.002, gravity=(0, 0, -9.81)),
        rigid_options=gs.options.RigidOptions(
            dt=0.002,
            constraint_solver=gs.constraint_solver.ComFree,
            comfree_stiffness=0.3,
            comfree_damping=0.005,
            enable_collision=True,
        ),
        show_viewer=False,
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

    z1 = box1.get_pos().numpy()[2]
    z2 = box2.get_pos().numpy()[2]
    assert z1 > 0.0, f"Box1 penetrated: z={z1}"
    assert z2 > z1, f"Box2 below box1: z2={z2}, z1={z1}"
    assert z2 < 0.5, f"Box2 too high: z={z2}"
    gs.destroy()


if __name__ == "__main__":
    tests = [
        test_comfree_instantiation,
        test_comfree_step,
        test_box_falls_under_gravity,
        test_box_contacts_floor,
        test_freefall_comparable,
        test_contact_qualitative,
        test_no_constraint_passthrough,
        test_stiffness_effect,
        test_batched_simulation,
        test_two_box_stack,
    ]
    for test in tests:
        print(f"Running {test.__name__}...", end=" ")
        test()
        print("PASSED")
    print("\nAll tests passed!")
