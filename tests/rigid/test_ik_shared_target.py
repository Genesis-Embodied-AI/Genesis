import numpy as np

import genesis as gs

from ..utils.assertions import assert_equal


def test_inverse_kinematics_accepts_shared_target_batched(show_viewer):
    """Regression test for #3246.

    `inverse_kinematics` documents `pos` as shape (3,) and `quat` as shape (4,). In a
    batched scene this shared target form must be accepted (broadcast to all selected
    environments), not rejected as if the 3/4 coordinates were per-environment rows.
    A shared target and the equivalent tiled per-env target must yield the same solution.
    """
    scene = gs.Scene(show_viewer=show_viewer)
    franka = scene.add_entity(gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"))

    n_envs = 2
    scene.build(n_envs=n_envs)

    end_effector = franka.get_link("hand")

    pos = np.array([0.4, 0.0, 0.3])
    quat = np.array([0.0, 1.0, 0.0, 0.0])

    # Documented shared target (pos shape (3,), quat shape (4,)).
    # Before the fix this raised "First dimension of `pos` must be equal to `scene.n_envs`."
    qpos_shared = franka.inverse_kinematics(link=end_effector, pos=pos, quat=quat)

    # Equivalent per-environment (tiled) target.
    qpos_tiled = franka.inverse_kinematics(
        link=end_effector,
        pos=np.tile(pos, (n_envs, 1)),
        quat=np.tile(quat, (n_envs, 1)),
    )

    assert qpos_shared.shape == qpos_tiled.shape
    assert_equal(qpos_shared, qpos_tiled)
