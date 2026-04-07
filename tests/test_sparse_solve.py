"""Test that sparse_solve=True does not produce NaN constraint forces.

The bug: jac_relevant_dofs was populated in per-entity descending order,
but cross-entity constraints (e.g., a contact between a robot link and
a free-floating box) concatenated two descending subsequences without
globally sorting. This caused the sparse Hessian build (H = M + J^T D J)
to write to the upper triangle (uninitialized memory), producing NaN.

The fix: insertion-sort jac_relevant_dofs into globally descending order
after population, and use max/min to ensure lower-triangle writes.

Minimal repro requires a robot with enough DOFs and complex link structure
to create cross-entity constraints with non-trivially ordered DOF indices.
Simple robots (Panda 9-DOF, Go2 18-DOF) don't trigger it because their
DOF ranges happen to be naturally sorted. The marvin bimanual robot's
gripper DOFs create the problematic interleaving.

Note: This test requires the marvin_pika.urdf asset. It will be skipped
if the asset is not available (e.g., in minimal CI environments).
"""

import os
import genesis as gs
import numpy as np
import pytest

_MARVIN_URDF = "simvla-genesis-test-scene/assets/robot/marvin_bimanual/urdf/marvin_pika.urdf"


@pytest.mark.skipif(
    not os.path.exists(_MARVIN_URDF),
    reason=f"Requires {_MARVIN_URDF} (not available in minimal CI)",
)
def test_sparse_solve_no_nan():
    """sparse_solve=True must complete 120 steps without NaN when the marvin
    bimanual robot is present, creating cross-entity constraints with
    non-trivially ordered DOF indices."""
    gs.init(backend=gs.cpu, seed=0, precision="64", logging_level="warning")

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=1.0 / 30, substeps=4, gravity=(0, 0, -9.81)),
        rigid_options=gs.options.RigidOptions(
            max_collision_pairs=128,
            sparse_solve=True,
        ),
        show_viewer=False,
    )

    # Marvin bimanual robot (many DOFs with complex interleaved structure)
    scene.add_entity(
        morph=gs.morphs.URDF(
            file=_MARVIN_URDF,
            pos=(0, 0, 1.08),
            fixed=True,
            convexify=True,
            merge_fixed_links=False,
        ),
    )

    # Table
    TABLE_Z = 0.762
    scene.add_entity(morph=gs.morphs.Box(pos=(0.5, 0, TABLE_Z / 2), size=(0.8, 0.6, TABLE_Z / 2), fixed=True))

    # Several boxes on the table
    boxes = []
    positions = [
        (0.45, -0.15, TABLE_Z + 0.08),
        (0.55, 0.10, TABLE_Z + 0.08),
        (0.60, -0.08, TABLE_Z + 0.08),
        (0.40, 0.05, TABLE_Z + 0.08),
        (0.55, -0.02, TABLE_Z + 0.08),
    ]
    for pos in positions:
        box = scene.add_entity(
            material=gs.materials.Rigid(friction=0.5),
            morph=gs.morphs.Box(pos=pos, size=(0.04, 0.04, 0.04)),
        )
        boxes.append(box)

    # A box to drop mid-simulation
    drop_box = scene.add_entity(morph=gs.morphs.Box(pos=(3, 0, -5), size=(0.03, 0.03, 0.03)))

    scene.build()

    # Phase 1: settle (40 steps)
    for _ in range(40):
        scene.step()

    # Phase 2: drop box (triggers new cross-entity contacts)
    drop_box.set_pos(np.array([0.5, 0.0, TABLE_Z + 0.30]))
    for _ in range(80):
        scene.step()

    # Verify no NaN
    for i, box in enumerate(boxes):
        pos = box.get_pos().cpu().numpy().flatten()
        assert not np.any(np.isnan(pos)), f"box_{i} has NaN position"
        assert np.all(np.abs(pos) < 10), f"box_{i} unreasonable position: {pos}"

    pos = drop_box.get_pos().cpu().numpy().flatten()
    assert not np.any(np.isnan(pos)), "drop_box has NaN position"

    print("PASSED: sparse_solve=True, 120 steps with marvin robot + boxes, no NaN")


if __name__ == "__main__":
    if not os.path.exists(_MARVIN_URDF):
        print(f"SKIPPED: {_MARVIN_URDF} not found")
    else:
        test_sparse_solve_no_nan()
