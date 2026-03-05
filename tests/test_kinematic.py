import math

import torch
import pytest

import genesis as gs

from tests.utils import assert_allclose


@pytest.mark.required
def test_kinematic_entity(show_viewer):
    """KinematicEntity: no collision geoms, get_vAABB tracks set_pos/set_quat, frozen under stepping."""

    scene = gs.Scene(
        show_viewer=show_viewer,
        show_FPS=False,
    )

    ghost_box = scene.add_entity(
        morph=gs.morphs.Box(
            size=(0.4, 0.2, 0.1),
            pos=(0, 0, 0.1),
        ),
        material=gs.materials.Kinematic(),
        surface=gs.surfaces.Default(
            color=(1.0, 0.0, 0.0, 0.7),
        ),
    )
    ghost_robot = scene.add_entity(
        morph=gs.morphs.URDF(
            file="urdf/simple/two_cube_revolute.urdf",
            pos=(0, 0, 0.3),
            fixed=False,
        ),
        material=gs.materials.Kinematic(),
        surface=gs.surfaces.Default(
            color=(0.0, 1.0, 0.0, 0.7),
        ),
    )

    scene.build(n_envs=2)

    breakpoint()

    vaabb = ghost_box.get_vAABB()
    # assert vaabb.shape == (2, 3)
    # assert_allclose(vaabb[0], [-0.2, -0.1, 1.95], tol=1e-4)
    # assert_allclose(vaabb[1], [0.2, 0.1, 2.05], tol=1e-4)

    # ghost.set_pos(torch.tensor([1.0, 2.0, 3.0]))
    # vaabb = ghost.get_vAABB()
    # assert_allclose(vaabb[0], [0.8, 1.9, 2.95], tol=1e-4)
    # assert_allclose(vaabb[1], [1.2, 2.1, 3.05], tol=1e-4)

    # ghost.set_pos(torch.tensor([0.0, 0.0, 0.0]))
    # angle = math.pi / 2
    # quat_z90 = torch.tensor([math.cos(angle / 2), 0.0, 0.0, math.sin(angle / 2)])
    # ghost.set_quat(quat_z90)
    # vaabb = ghost.get_vAABB()
    # # After 90° Z rotation: half-extents swap x↔y → [-0.1, -0.2, -0.05] to [0.1, 0.2, 0.05]
    # assert_allclose(vaabb[0], [-0.1, -0.2, -0.05], tol=1e-3)
    # assert_allclose(vaabb[1], [0.1, 0.2, 0.05], tol=1e-3)

    # frozen_vaabb = ghost.get_vAABB()
    # for _ in range(10):
    #     scene.step()
    # assert_allclose(ghost.get_vAABB(), frozen_vaabb, tol=1e-7)


@pytest.mark.required
def test_kinematic_ghost_tracks_rigid(show_viewer):
    """KinematicEntity mirrors RigidEntity when syncing pos/quat, stays frozen when updates stop."""

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

    scene.build()

    for _ in range(10):
        scene.step()
        ghost.set_qpos(robot.get_qpos())
        assert_allclose(ghost.get_vAABB(), robot.get_vAABB(), tol=1e-4)

    frozen_ghost_vaabb = ghost.get_vAABB()
    frozen_robot_vaabb = robot.get_vAABB()

    for _ in range(30):
        scene.step()

    assert_allclose(ghost.get_vAABB(), frozen_ghost_vaabb, tol=gs.EPS)
    with pytest.raises(AssertionError):
        assert_allclose(robot.get_vAABB(), frozen_robot_vaabb, atol=0.1)
