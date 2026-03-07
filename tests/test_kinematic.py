import pytest

import genesis as gs

from genesis.utils.misc import tensor_to_array
from tests.utils import assert_allclose


@pytest.mark.required
def test_setters(show_viewer, tol):
    scene = gs.Scene(
        show_viewer=show_viewer,
        show_FPS=False,
    )

    ghost_box = scene.add_entity(
        morph=gs.morphs.Box(
            size=(0.4, 0.2, 0.1),
        ),
        material=gs.materials.Kinematic(),
        surface=gs.surfaces.Default(
            color=(1.0, 0.0, 0.0, 0.7),
        ),
    )
    ghost_robot = scene.add_entity(
        morph=gs.morphs.URDF(
            file="urdf/simple/two_cube_prismatic.urdf",
            fixed=False,
        ),
        material=gs.materials.Kinematic(),
        surface=gs.surfaces.Default(
            color=(0.0, 1.0, 0.0, 0.7),
        ),
    )

    scene.build(n_envs=2)

    assert_allclose(ghost_box.get_vAABB(), ((-0.20, -0.10, -0.05), (0.20, 0.10, 0.05)), tol=tol)
    assert_allclose(ghost_robot.get_vAABB(), ((-0.05, -0.05, -0.05), (0.15, 0.05, 0.05)), tol=tol)

    ghost_box.set_pos([1.0, 2.0, 3.0])
    assert_allclose(ghost_box.get_vAABB(), ((0.80, 1.90, 2.95), (1.20, 2.10, 3.05)), tol=tol)
    ghost_box.set_pos([0.0, 0.0, 0.0])
    ghost_box.set_quat([[1.0, 1.0, 0.0, 0.0], [1.0, 0.0, 1.0, 0.0]])
    assert_allclose(ghost_box.get_vAABB()[0], ((-0.20, -0.05, -0.10), (0.20, 0.05, 0.1)), tol=tol)
    assert_allclose(ghost_box.get_vAABB()[1], ((-0.05, -0.10, -0.2), (0.05, 0.10, 0.2)), tol=tol)

    ghost_robot.set_dofs_position([0.1, -0.1], dofs_idx_local=-1)
    assert_allclose(ghost_robot.get_vAABB()[0], ((-0.05, -0.05, -0.05), (0.25, 0.05, 0.05)), tol=tol)
    assert_allclose(ghost_robot.get_vAABB()[1], ((-0.05, -0.05, -0.05), (0.05, 0.05, 0.05)), tol=tol)

    ghost_robot.set_qpos([1.0, 2.0, 3.0, 1.0, 1.0, 0.0, 0.0, 1.0])
    assert_allclose(ghost_robot.get_vAABB(), ((0.95, 1.95, 2.95), (2.15, 2.05, 3.05)), tol=tol)

    frozen_vaabb = [tensor_to_array(entity.get_vAABB()) for entity in scene.entities]
    for _ in range(5):
        scene.step()
    assert_allclose([tensor_to_array(entity.get_vAABB()) for entity in scene.entities], frozen_vaabb, tol=gs.EPS)


@pytest.mark.required
def test_track_rigid(show_viewer, tol):
    scene = gs.Scene(
        show_viewer=show_viewer,
        show_FPS=False,
    )

    scene.add_entity(gs.morphs.Plane())
    robot = scene.add_entity(
        morph=gs.morphs.URDF(
            file="urdf/go2/urdf/go2.urdf",
            pos=(0, 0.5, 0.42),
        ),
    )
    ghost = scene.add_entity(
        morph=gs.morphs.URDF(
            file="urdf/go2/urdf/go2.urdf",
            pos=(0, 0.5, 0.42),
        ),
        material=gs.materials.Kinematic(),
        surface=gs.surfaces.Default(
            color=(0.2, 0.0, 1.0, 0.7),
        ),
    )
    scene.build(n_envs=2, env_spacing=(0.5, 0.5))

    for _ in range(20):
        scene.step()
        ghost.set_qpos(robot.get_qpos())
        assert_allclose(ghost.get_vAABB(), robot.get_vAABB(), tol=tol)
    assert_allclose(ghost.get_links_pos(), robot.get_links_pos(), tol=tol)

    ghost.set_dofs_velocity(robot.get_dofs_velocity())
    assert_allclose(ghost.get_links_vel(), robot.get_links_vel(), tol=tol)

    frozen_ghost_vaabb = ghost.get_vAABB()
    frozen_robot_vaabb = robot.get_vAABB()
    for _ in range(20):
        scene.step()

    assert_allclose(ghost.get_vAABB(), frozen_ghost_vaabb, tol=gs.EPS)
    with pytest.raises(AssertionError):
        assert_allclose(robot.get_vAABB(), frozen_robot_vaabb, atol=0.1)
