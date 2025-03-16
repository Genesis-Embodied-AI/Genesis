import pytest

import numpy as np
import mujoco
import genesis as gs

from .utils import simulate_and_check_mujoco_consistency


@pytest.mark.parametrize("model_name", ["box_plan"])
@pytest.mark.parametrize("gs_solver", [gs.constraint_solver.CG])
@pytest.mark.parametrize("gs_integrator", [gs.integrator.implicitfast])
@pytest.mark.parametrize("backend", ["cpu"], indirect=True)
def test_box_plan_dynamics(gs_sim, mj_sim):
    (gs_robot,) = gs_sim.entities
    cube_pos = np.array([0.0, 0.0, 0.6])
    cube_quat = np.random.rand(4)
    cube_quat /= np.linalg.norm(cube_quat)
    qpos = np.concatenate((cube_pos, cube_quat))
    qvel = np.zeros((gs_robot.n_dofs,))
    simulate_and_check_mujoco_consistency(gs_sim, mj_sim, qpos, qvel, num_steps=150)
