"""Test that sparse_solve=True does not produce NaN constraint forces.

The bug: several constraint types (equality joint, friction loss) did not
populate jac_relevant_dofs when sparse_solve=True. This caused sparse
iterations over these constraints to see 0 relevant DOFs, producing
zero/wrong forces that led to NaN in the solver.

Additionally, the sparse Hessian build (H = M + J^T D J) wrote to the
upper triangle for cross-entity constraints because jac_relevant_dofs
was not globally sorted in descending order.

Minimal repro: Franka Panda robot (has mimic joint equality constraints
on gripper fingers) + boxes on a table with noslip enabled. Without the
fix, this scene produces NaN at step 5. With the fix, it runs 200 steps.
"""

import genesis as gs
import numpy as np
import pytest


@pytest.fixture(autouse=True)
def genesis_init():
    gs.init(backend=gs.cpu, seed=0, precision="64", logging_level="warning")
    yield


def _build_panda_table_scene(noslip_iterations=2, sparse_solve=False):
    """Panda robot + table + 16 boxes. The Panda's gripper has mimic joint
    equality constraints that trigger the sparse_solve bug."""
    TABLE_Z = 0.762
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=1.0 / 30, substeps=4, gravity=(0, 0, -9.81)),
        rigid_options=gs.options.RigidOptions(
            noslip_iterations=noslip_iterations,
            max_collision_pairs=128,
            sparse_solve=sparse_solve,
        ),
        show_viewer=False,
    )

    scene.add_entity(morph=gs.morphs.URDF(file="urdf/panda_bullet/panda.urdf", pos=(0, 0, TABLE_Z), fixed=True))
    scene.add_entity(morph=gs.morphs.Box(pos=(0.5, 0, TABLE_Z / 2), size=(0.8, 0.6, TABLE_Z / 2), fixed=True))

    boxes = []
    for i in range(16):
        x = 0.25 + 0.12 * (i % 5)
        y = -0.25 + 0.12 * (i // 5)
        box = scene.add_entity(
            material=gs.materials.Rigid(friction=0.5),
            morph=gs.morphs.Box(pos=(x, y, TABLE_Z + 0.08), size=(0.04, 0.04, 0.04)),
        )
        boxes.append(box)

    scene.build()
    return scene, boxes


def test_sparse_solve_no_nan():
    """sparse_solve=True + noslip must not produce NaN over 200 steps.

    Without the fix, this crashes at step 5 with:
      'Invalid constraint forces causing nan'
    """
    scene, boxes = _build_panda_table_scene(noslip_iterations=2, sparse_solve=True)

    for step in range(200):
        scene.step()

    # Verify no NaN in positions
    for i, box in enumerate(boxes):
        pos = box.get_pos().cpu().numpy().flatten()
        assert not np.any(np.isnan(pos)), f"box_{i} has NaN position at step 200"
        assert np.all(np.abs(pos) < 10), f"box_{i} has unreasonable position: {pos}"


def test_sparse_solve_matches_dense():
    """sparse_solve=True results should be close to dense solve results."""
    results = {}
    for sparse in [False, True]:
        gs._is_initialized = False
        gs.init(backend=gs.cpu, seed=0, precision="64", logging_level="warning")
        scene, boxes = _build_panda_table_scene(noslip_iterations=2, sparse_solve=sparse)

        for _ in range(60):
            scene.step()

        results[sparse] = [box.get_pos().cpu().numpy().flatten() for box in boxes]

    for i in range(len(results[False])):
        diff = np.max(np.abs(results[False][i] - results[True][i]))
        assert diff < 0.1, f"box_{i} diverged too much: {diff:.4f}m (dense vs sparse)"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
