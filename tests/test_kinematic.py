import numpy as np
import pytest

import genesis as gs


def compute_entity_vgeom_aabbs(entity, solver):
    """Compute per-vgeom AABBs by transforming mesh vertices with visual render transforms.

    Returns an array of shape (n_vgeoms, 2, 3) where [i, 0] is the min corner
    and [i, 1] is the max corner of the i-th vgeom's world-space AABB.
    """
    solver.forward_kinematics()
    solver.update_vgeoms_render_T()
    vgeoms_T = solver._vgeoms_render_T  # (n_vgeoms_, B, 4, 4)

    aabbs = []
    for vgeom in entity.vgeoms:
        verts = np.asarray(vgeom.vmesh.trimesh.vertices, dtype=np.float32)  # (V, 3)
        T = vgeoms_T[vgeom.idx, 0]  # env 0, shape (4, 4)

        ones = np.ones((verts.shape[0], 1), dtype=np.float32)
        verts_h = np.concatenate([verts, ones], axis=1)  # (V, 4)
        world_verts = (verts_h @ T.T)[:, :3]  # (V, 3)

        aabb = np.stack([world_verts.min(axis=0), world_verts.max(axis=0)])  # (2, 3)
        aabbs.append(aabb)

    return np.stack(aabbs)  # (n_vgeoms, 2, 3)


@pytest.mark.required
def test_kinematic_ghost_tracks_rigid():
    """KinematicEntity mirrors a RigidEntity when given matching qpos,
    and stays frozen when updates stop while the rigid entity diverges."""

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.01),
        show_viewer=False,
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
    )

    scene.build()

    # Kinematic entities should have zero collision geoms but visual geoms
    assert ghost.n_geoms == 0, f"Expected 0 collision geoms, got {ghost.n_geoms}"
    assert ghost.n_vgeoms > 0, f"Expected visual geoms, got {ghost.n_vgeoms}"

    # ------------------------------------------------------------------
    # Phase 1: Ghost tracks rigid (sync qpos -> matching bounding boxes)
    # ------------------------------------------------------------------
    for _ in range(5):
        scene.step()
        robot_qpos = robot.get_dofs_position()
        ghost.set_dofs_position(robot_qpos)
        scene.step()  # trigger FK on kinematic solver

        rigid_aabbs = compute_entity_vgeom_aabbs(robot, scene.rigid_solver)
        kinematic_aabbs = compute_entity_vgeom_aabbs(ghost, scene.kinematic_solver)

        np.testing.assert_allclose(
            rigid_aabbs,
            kinematic_aabbs,
            atol=1e-4,
            err_msg="Ghost vgeom AABBs should match rigid vgeom AABBs when qpos is synced",
        )

    # ------------------------------------------------------------------
    # Phase 2: Ghost freezes, rigid diverges
    # ------------------------------------------------------------------
    frozen_ghost_aabbs = compute_entity_vgeom_aabbs(ghost, scene.kinematic_solver)
    frozen_robot_aabbs = compute_entity_vgeom_aabbs(robot, scene.rigid_solver)

    for _ in range(200):
        scene.step()

    ghost_aabbs_after = compute_entity_vgeom_aabbs(ghost, scene.kinematic_solver)
    robot_aabbs_after = compute_entity_vgeom_aabbs(robot, scene.rigid_solver)

    # Ghost should not have moved
    np.testing.assert_allclose(
        ghost_aabbs_after,
        frozen_ghost_aabbs,
        atol=1e-7,
        err_msg="Ghost AABBs should remain frozen when qpos is not updated",
    )

    # Rigid bounding boxes should have changed (robot falls under gravity)
    assert not np.allclose(robot_aabbs_after, frozen_robot_aabbs, atol=1e-2), (
        "Rigid AABBs should have diverged from their frozen snapshot after stepping without sync"
    )
