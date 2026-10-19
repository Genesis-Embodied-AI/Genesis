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

    scene.step()


@pytest.mark.parametrize("material", [gs.materials.Kinematic(), gs.materials.MPM.Elastic()])
def test_non_rigid_solver_keeps_every_solver_legacy(show_viewer, material):
    scene = gs.Scene(show_viewer=show_viewer)
    scene.add_entity(gs.morphs.Box(size=(0.1, 0.1, 0.1)))
    scene.add_entity(
        morph=gs.morphs.Box(pos=(0.5, 0.0, 0.5), size=(0.1, 0.1, 0.1)),
        material=material,
    )
    scene.build()

    assert not scene.sim._use_rigid_solver_system
    assert scene.sim._engine is None
    assert scene.rigid_solver._system is None

    scene.step()
