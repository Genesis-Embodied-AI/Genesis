import pytest

import genesis as gs


pytestmark = pytest.mark.required


@pytest.mark.parametrize("requires_grad", [False, True])
def test_rigid_only_uses_rigid_solver_system(show_viewer, requires_grad):
    scene = gs.Scene(sim_options=gs.options.SimOptions(requires_grad=requires_grad), show_viewer=show_viewer)
    scene.add_entity(gs.morphs.Box(size=(0.1, 0.1, 0.1)))
    scene.build()

    assert scene.sim._use_rigid_solver_system
    assert scene.sim._engine is not None
    assert scene.rigid_solver._system is not None
    assert scene.rigid_solver.data_manager is scene.rigid_solver._system.data_manager

    system = scene.rigid_solver._system
    assert scene.rigid_solver.collider._data is system.contact_system.collider_data
    assert scene.rigid_solver.constraint_solver is system.constraint_system.constraint_solver
    assert not hasattr(system.data, "collider_state")
    assert not hasattr(system.data, "constraint_state")
    system._is_forward_pos_updated = True
    system._is_forward_vel_updated = True
    scene.rigid_solver.set_qpos(scene.rigid_solver.get_qpos(), skip_forward=True)
    assert not system._is_forward_pos_updated
    assert not system._is_forward_vel_updated
    system._is_backward = True
    assert scene.rigid_solver._is_backward

    scene.step()


@pytest.mark.parametrize("material", [gs.materials.Kinematic(), gs.materials.MPM.Elastic()])
def test_mixed_scene_keeps_system_ownership_behind_legacy_api(show_viewer, material):
    scene = gs.Scene(show_viewer=show_viewer)
    scene.add_entity(gs.morphs.Box(size=(0.1, 0.1, 0.1)))
    other = scene.add_entity(
        morph=gs.morphs.Box(pos=(0.5, 0.0, 0.5), size=(0.1, 0.1, 0.1)),
        material=material,
    )
    scene.build()

    assert scene.sim._use_rigid_solver_system
    assert not scene.sim._rigid_only
    assert scene.sim._engine is not None
    assert scene.rigid_solver._system is not None
    assert not hasattr(other.solver, "_system")

    system = scene.rigid_solver._system
    assert scene.rigid_solver.collider._data is system.contact_system.collider_data
    assert scene.rigid_solver.constraint_solver is system.constraint_system.constraint_solver

    scene.step()


@pytest.mark.parametrize(
    "enable_collision,disable_constraint",
    [(False, False), (False, True), (True, False), (True, True)],
)
def test_disabled_rigid_features_keep_owner_data_wired(show_viewer, enable_collision, disable_constraint):
    scene = gs.Scene(
        rigid_options=gs.options.RigidOptions(
            enable_collision=enable_collision,
            disable_constraint=disable_constraint,
        ),
        show_viewer=show_viewer,
    )
    scene.add_entity(gs.morphs.Box(size=(0.1, 0.1, 0.1)))
    scene.build()

    system = scene.rigid_solver._system
    assert scene.rigid_solver.collider._data is system.contact_system.collider_data
    assert scene.rigid_solver.constraint_solver is system.constraint_system.constraint_solver
    assert system.contact_system.enabled is enable_collision
    assert system.constraint_system.enabled is (not disable_constraint)

    scene.step()


def test_inactive_rigid_padding_remains_system_owned(show_viewer):
    scene = gs.Scene(show_viewer=show_viewer)
    ghost = scene.add_entity(
        gs.morphs.Box(size=(0.1, 0.1, 0.1)),
        material=gs.materials.Kinematic(),
    )
    scene.build()

    assert not scene.rigid_solver.is_active
    assert scene.rigid_solver._system is not None
    assert scene.rigid_solver.collider._data is scene.rigid_solver._system.contact_system.collider_data
    assert scene.rigid_solver.constraint_solver is scene.rigid_solver._system.constraint_system.constraint_solver
    assert not hasattr(ghost.solver, "_system")

    scene.step()
