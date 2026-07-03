import genesis as gs
import numpy as np
import pytest

from genesis.utils.misc import tensor_to_array

from .utils import assert_allclose


@pytest.mark.slow("gpu")  # gpu ~250s
@pytest.mark.required
@pytest.mark.parametrize("backend", [gs.cpu, gs.gpu])
def test_sparse_solve_no_nan(backend, precision):
    TABLE_Z = 0.762
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=1.0 / 30, substeps=4, gravity=(0, 0, -9.81)),
        rigid_options=gs.options.RigidOptions(
            noslip_iterations=2,
            max_collision_pairs=128,
            sparse_solve=True,
        ),
        show_viewer=False,
    )

    scene.add_entity(
        morph=gs.morphs.URDF(
            file="urdf/panda_bullet/panda.urdf",
            pos=(0, 0, TABLE_Z),
            fixed=True,
        )
    )
    scene.add_entity(
        morph=gs.morphs.Box(
            pos=(0.5, 0, TABLE_Z / 2),
            size=(0.8, 0.6, TABLE_Z / 2),
            fixed=True,
        )
    )

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

    for step in range(20):
        scene.step()

    # Verify no NaN in positions
    for i, box in enumerate(boxes):
        pos = tensor_to_array(box.get_pos())
        assert not np.any(np.isnan(pos)), f"box_{i} has NaN position"
        assert np.all(np.abs(pos) < 10), f"box_{i} has unreasonable position: {pos}"


@pytest.mark.required
def test_self_collision_sparse_dense_consistency():
    # Pose folding the arm into link2 / left_finger contact: rows coupling two links of the same kinematic tree carry
    # both ancestor chains in their dof support, which the sparse consumers must treat as a set.
    vels = []
    for sparse_solve in (False, True):
        scene = gs.Scene(
            rigid_options=gs.options.RigidOptions(
                noslip_iterations=2,
                sparse_solve=sparse_solve,
            ),
            show_viewer=False,
        )
        franka = scene.add_entity(
            morph=gs.morphs.MJCF(
                file="xml/franka_emika_panda/panda.xml",
            ),
        )
        scene.build()
        franka.set_qpos([-0.2326, -1.6055, 1.7372, -2.7745, 0.1091, 0.9083, 0.4493, 0.0384, 0.0258])
        scene.step()
        vels.append(tensor_to_array(franka.get_dofs_velocity()))
    assert_allclose(vels[0], vels[1], rtol=1e-4, atol=1e-5)
