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
@pytest.mark.friction_torsional(True)
@pytest.mark.parametrize(
    "model_name",
    [
        "sphere_plane_spin",
        pytest.param("sphere_plane_roll", marks=pytest.mark.friction_rolling(True)),
    ],
)
@pytest.mark.parametrize(
    "gs_solver, gs_integrator",
    [
        (gs.constraint_solver.Newton, gs.integrator.Euler),
        pytest.param(
            gs.constraint_solver.Newton,
            gs.integrator.Euler,
            marks=pytest.mark.friction_cone(gs.friction_cone.elliptic),
            id="Newton-Euler-elliptic",
        ),
    ],
)
@pytest.mark.parametrize("backend", [gs.cpu])
def test_torsional_and_rolling_friction(gs_sim, mj_sim, tol):
    # Sliding while spinning and rolling couples every friction axis through slip, stick, and rest. The slight
    # initial penetration makes the contact exist from the first step.
    qpos = np.array([0.0, 0.0, 0.0999, 1.0, 0.0, 0.0, 0.0])
    qvel = np.array([0.5, 0.0, 0.0, 0.0, 4.0, 3.0])
    simulate_and_check_mujoco_consistency(gs_sim, mj_sim, qpos=qpos, qvel=qvel, num_steps=60, tol=tol)


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
    simulate_and_check_mujoco_consistency(gs_sim, mj_sim, num_steps=300, tol=5e-9)


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

    # The smooth acceleration divides chain-accumulated rounding by link masses of a tenth of a gram, putting its
    # agreement floor four decades above the working precision.
    simulate_and_check_mujoco_consistency(gs_sim, mj_sim, num_steps=300, tol=5e-5)


@pytest.mark.required
@pytest.mark.parametrize(
    "model_name, gjk_collision",
    [
        ("xml/tet_tet.xml", True),
        ("xml/tet_ball.xml", True),
        ("xml/tet_capsule.xml", True),
        # Multi-vertex contact patches between discrete meshes, recovered by clipping the touching faces; the rows
        # above settle on single-point vertex-face contacts.
        ("tet_meshball", True),
        # The same patches through the MPR pipeline, whose manifold comes from exhaustive mesh supports and perturbed
        # re-detections.
        ("tet_meshball", False),
    ],
)
@pytest.mark.parametrize("gs_solver", [gs.constraint_solver.CG, gs.constraint_solver.Newton])
@pytest.mark.parametrize("gs_integrator", [gs.integrator.implicitfast, gs.integrator.Euler])
@pytest.mark.parametrize("multi_contact", [True, False])
@pytest.mark.parametrize("backend", [gs.cpu])
def test_tet_primitive_shapes(gs_sim, mj_sim, gs_integrator, gs_solver, multi_contact, tol):
    # Make sure it is possible to set the configuration vector without failure
    gs_sim.rigid_solver.set_dofs_position(gs_sim.rigid_solver.get_dofs_position())

    check_mujoco_model_consistency(gs_sim, mj_sim, tol=tol)
    # FIXME: Because of very small numerical error, error could be this large even if there is no logical error.
    # Multi-contact perturbation introduces slightly larger errors due to GJK implementation differences.
    # Both implementations agree to machine precision on most steps, but the capsule scene holds a grazing contact
    # whose occasional hard solves amplify rounding-order differences into distinct CG iterate paths.
    simulate_and_check_mujoco_consistency(gs_sim, mj_sim, num_steps=700, tol=5e-6)


@pytest.mark.required
@pytest.mark.parametrize("model_name", ["humanoid_ball_floor"])
@pytest.mark.parametrize("gs_solver", [gs.constraint_solver.Newton])
@pytest.mark.parametrize("gs_integrator", [gs.integrator.Euler])
@pytest.mark.parametrize("gjk_collision", [True])
@pytest.mark.parametrize("backend", [gs.cpu, gs.gpu])
def test_stickman(gs_sim, mj_sim, tol):
    # Make sure that the simulation is deterministic
    init_simulators(gs_sim)
    (gs_robot,) = gs_sim.entities
    gs_sim.scene.reset()
    gs_sim.scene.step()
    dofs_vel = gs_robot.get_dofs_velocity()
    for _ in range(50):
        gs_sim.scene.reset()
        gs_sim.scene.step()
        assert_equal(gs_robot.get_dofs_velocity(), dofs_vel)

    # A falling humanoid puts every capsule of the model on the ground in turn, so the contact set it exercises is far
    # richer than the other models here. Consistency is asserted step by step against MuJoCo rather than through the
    # pose it eventually settles in, which depends on a chaotic tumble and says nothing about compatibility.
    simulate_and_check_mujoco_consistency(gs_sim, mj_sim, num_steps=500, tol=5e-9 if gs.np_float == np.float64 else tol)
