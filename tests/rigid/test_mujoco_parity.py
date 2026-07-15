import math

import mujoco
import numpy as np
import pytest
import torch

import genesis as gs
import genesis.utils.geom as gu

from ..utils import (
    assert_allclose,
    assert_equal,
    check_mujoco_model_consistency,
    init_simulators,
    simulate_and_check_mujoco_consistency,
)


@pytest.mark.required
@pytest.mark.parametrize("model_name", ["box_plan"])
@pytest.mark.parametrize(
    "gs_solver, gs_integrator",
    [
        (gs.constraint_solver.CG, gs.integrator.implicitfast),
        (gs.constraint_solver.CG, gs.integrator.Euler),
        (gs.constraint_solver.Newton, gs.integrator.implicitfast),
        (gs.constraint_solver.Newton, gs.integrator.Euler),
        # Elliptic (second-order) friction cone must match MuJoCo's elliptic cone. The box lands and slides with an
        # initial tangential + angular velocity so the tangential cone rows are exercised in both the sliding (cone
        # boundary) and sticking (bottom) regimes.
        pytest.param(
            gs.constraint_solver.CG,
            gs.integrator.implicitfast,
            marks=pytest.mark.friction_cone(gs.friction_cone.elliptic),
            id="CG-implicitfast-elliptic",
        ),
        pytest.param(
            gs.constraint_solver.Newton,
            gs.integrator.implicitfast,
            marks=pytest.mark.friction_cone(gs.friction_cone.elliptic),
            id="Newton-implicitfast-elliptic",
        ),
    ],
)
@pytest.mark.parametrize("backend", [gs.cpu])
def test_box_plane_dynamics(gs_sim, mj_sim, tol):
    cube_pos = np.array([0.0, 0.0, 0.6])
    cube_quat = np.random.rand(4)
    cube_quat /= np.linalg.norm(cube_quat)
    qpos = np.concatenate((cube_pos, cube_quat))
    qvel = np.random.rand(6) * 0.2
    simulate_and_check_mujoco_consistency(gs_sim, mj_sim, qpos, qvel, num_steps=150, tol=tol)


@pytest.mark.required
@pytest.mark.adjacent_collision(True)
@pytest.mark.parametrize("model_name", ["chain_capsule_hinge_mesh"])  # FIXME: , "chain_capsule_hinge_capsule"])
@pytest.mark.parametrize("gs_solver", [gs.constraint_solver.CG, gs.constraint_solver.Newton])
@pytest.mark.parametrize("gs_integrator", [gs.integrator.implicitfast, gs.integrator.Euler])
@pytest.mark.parametrize("gjk_collision", [True, False])
@pytest.mark.parametrize("backend", [gs.cpu])
def test_simple_kinematic_chain(gs_sim, mj_sim, tol):
    simulate_and_check_mujoco_consistency(gs_sim, mj_sim, num_steps=200, tol=tol)


@pytest.mark.required
@pytest.mark.parametrize("xml_path", ["xml/walker.xml"])
@pytest.mark.parametrize(
    "gs_solver",
    [
        gs.constraint_solver.CG,
        # gs.constraint_solver.Newton,  # FIXME: This test is not passing because collision detection is too sensitive
    ],
)
@pytest.mark.parametrize("gs_integrator", [gs.integrator.implicitfast, gs.integrator.Euler])
@pytest.mark.parametrize("gjk_collision", [True, False])
@pytest.mark.parametrize("backend", [gs.cpu])
def test_walker(gs_sim, mj_sim, gjk_collision, tol):
    # Force numpy seed because this test is very sensitive to the initial condition
    np.random.seed(0)
    (gs_robot,) = gs_sim.entities
    qpos = np.zeros((gs_robot.n_qs,))
    qpos[2] += 0.5
    qvel = np.random.rand(gs_robot.n_dofs) * 0.2

    # Make sure it is possible to set the configuration vector without failure
    qpos = gs_robot.get_dofs_position()
    gs_robot.set_dofs_position(qpos)
    assert_allclose(gs_robot.get_dofs_position(), qpos, tol=gs.EPS)
    qpos = torch.rand(gs_robot.n_dofs).clip(*gs_robot.get_dofs_limit())
    gs_robot.set_dofs_position(qpos)
    assert_allclose(gs_robot.get_dofs_position(), qpos, tol=gs.EPS)

    # Cannot simulate any longer because collision detection is very sensitive
    simulate_and_check_mujoco_consistency(gs_sim, mj_sim, qpos, qvel, num_steps=90, tol=tol)


@pytest.mark.required
@pytest.mark.parametrize("xml_path", ["xml/one_ball_joint.xml"])
@pytest.mark.parametrize("gs_solver", [gs.constraint_solver.CG, gs.constraint_solver.Newton])
@pytest.mark.parametrize("gs_integrator", [gs.integrator.implicitfast, gs.integrator.Euler])
@pytest.mark.parametrize("backend", [gs.cpu])
def test_one_ball_joint(gs_sim, mj_sim, tol):
    # FIXME: Mujoco is detecting collision for some reason...
    mj_sim.model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT

    simulate_and_check_mujoco_consistency(gs_sim, mj_sim, num_steps=600, tol=tol)


@pytest.mark.required
@pytest.mark.parametrize("xml_path", ["xml/rope_ball.xml", "xml/rope_hinge.xml"])
@pytest.mark.parametrize("gs_solver", [gs.constraint_solver.CG, gs.constraint_solver.Newton])
@pytest.mark.parametrize("gs_integrator", [gs.integrator.implicitfast, gs.integrator.Euler])
@pytest.mark.parametrize("gjk_collision", [True, False])
@pytest.mark.parametrize("backend", [gs.cpu])
def test_rope_ball(gs_sim, mj_sim, gs_solver, tol):
    # Make sure it is possible to set the configuration vector without failure
    qpos = gs_sim.rigid_solver.get_dofs_position()
    gs_sim.rigid_solver.set_dofs_position(qpos)
    assert_allclose(gs_sim.rigid_solver.get_dofs_position(), qpos, tol=gs.EPS)
    qpos = torch.rand(gs_sim.rigid_solver.n_dofs).clip(*gs_sim.rigid_solver.get_dofs_limit())
    gs_sim.rigid_solver.set_dofs_position(qpos)
    assert_allclose(gs_sim.rigid_solver.get_dofs_position(), qpos, tol=gs.EPS)

    check_mujoco_model_consistency(gs_sim, mj_sim, tol=tol)
    simulate_and_check_mujoco_consistency(gs_sim, mj_sim, num_steps=300, tol=1e-8)


@pytest.mark.required
@pytest.mark.parametrize("xml_path", ["linear_deformable.urdf"])
@pytest.mark.parametrize("gs_solver", [gs.constraint_solver.CG])
@pytest.mark.parametrize("gs_integrator", [gs.integrator.implicitfast])
@pytest.mark.parametrize("gjk_collision", [True, False])
@pytest.mark.parametrize("backend", [gs.cpu])
def test_urdf_rope(gs_sim, mj_sim, gs_solver, xml_path):
    # Must increase sol params to improve numerical stability
    sol_params = gu.default_solver_params()
    sol_params[0] = 0.02
    gs_sim.rigid_solver.set_global_sol_params(sol_params)
    mj_sim.model.jnt_solref[:, 0] = sol_params[0]
    mj_sim.model.geom_solref[:, 0] = sol_params[0]
    mj_sim.model.eq_solref[:, 0] = sol_params[0]

    # FIXME: Tolerance must be very large due to small masses and compounding of errors over long kinematic chains
    simulate_and_check_mujoco_consistency(gs_sim, mj_sim, num_steps=300, tol=5e-5)


@pytest.mark.required
@pytest.mark.mujoco_compatibility(True)
@pytest.mark.parametrize("xml_path", ["xml/tet_tet.xml", "xml/tet_ball.xml", "xml/tet_capsule.xml"])
@pytest.mark.parametrize("gs_solver", [gs.constraint_solver.CG, gs.constraint_solver.Newton])
@pytest.mark.parametrize("gs_integrator", [gs.integrator.implicitfast, gs.integrator.Euler])
@pytest.mark.parametrize("gjk_collision", [True])
@pytest.mark.parametrize("multi_contact", [True, False])
@pytest.mark.parametrize("backend", [gs.cpu])
def test_tet_primitive_shapes(gs_sim, mj_sim, gs_integrator, gs_solver, xml_path, multi_contact, tol):
    # Make sure it is possible to set the configuration vector without failure
    gs_sim.rigid_solver.set_dofs_position(gs_sim.rigid_solver.get_dofs_position())

    check_mujoco_model_consistency(gs_sim, mj_sim, tol=tol)
    # FIXME: Because of very small numerical error, error could be this large even if there is no logical error.
    # Multi-contact perturbation introduces slightly larger errors due to GJK implementation differences.
    simulate_and_check_mujoco_consistency(gs_sim, mj_sim, num_steps=700, tol=2e-6)


@pytest.mark.required
@pytest.mark.mujoco_compatibility(False)
@pytest.mark.parametrize("xml_path", ["xml/humanoid.xml"])
@pytest.mark.parametrize("gs_solver", [gs.constraint_solver.Newton])
@pytest.mark.parametrize("gs_integrator", [gs.integrator.Euler])
@pytest.mark.parametrize("gjk_collision", [True])
@pytest.mark.parametrize("backend", [gs.cpu, gs.gpu])
def test_stickman(gs_sim, mj_sim, tol):
    # Make sure that "static" model information are matching
    check_mujoco_model_consistency(gs_sim, mj_sim, tol=tol)

    # Initialize the simulation
    init_simulators(gs_sim)

    # Make sure that the simulation is deterministic
    (gs_robot,) = gs_sim.entities
    gs_sim.scene.reset()
    gs_sim.scene.step()
    dofs_vel = gs_robot.get_dofs_velocity()
    for _ in range(50):
        gs_sim.scene.reset()
        gs_sim.scene.step()
        assert_equal(gs_robot.get_dofs_velocity(), dofs_vel)

    # Run the simulation for a while
    qvel_norminf_all = []
    for i in range(750):
        gs_sim.scene.step()
        if i > 700:
            (gs_robot,) = gs_sim.entities
            qvel = gs_robot.get_dofs_velocity()
            qvel_norminf = torch.linalg.norm(qvel, ord=math.inf)
            qvel_norminf_all.append(qvel_norminf)
    assert_allclose(torch.quantile(torch.stack(qvel_norminf_all, dim=0), 0.5), 0.0, tol=0.1)

    qpos = gs_robot.get_dofs_position()
    assert torch.linalg.norm(qpos[:2]) < 1.3
    body_z = gs_sim.rigid_solver.links_state.pos.to_numpy()[:-1, 0, 2]
    np.testing.assert_array_less(0, body_z + gs.EPS)
