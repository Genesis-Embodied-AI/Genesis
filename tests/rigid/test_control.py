import mujoco
import numpy as np
import pytest
import torch

import genesis as gs
from genesis.utils.misc import qd_to_torch, tensor_to_array

from ..utils import (
    assert_allclose,
    assert_equal,
    check_mujoco_data_consistency,
    check_mujoco_model_consistency,
    get_hf_dataset,
    init_simulators,
)


@pytest.mark.required
@pytest.mark.parametrize("model_name", ["general_actuator"])
@pytest.mark.parametrize("gs_solver", [gs.constraint_solver.CG])
@pytest.mark.parametrize("gs_integrator", [gs.integrator.Euler])
@pytest.mark.parametrize("backend", [gs.cpu])
def test_general_actuator(gs_sim, mj_sim, tol):
    (entity,) = gs_sim.entities

    # get_dofs_kp raises for all DOFs (joint 1 is non-PD-reducible from parser)
    with pytest.raises(gs.GenesisException):
        entity.get_dofs_kp()

    # but succeeds for the PD joint (joint 0)
    entity.get_dofs_kp(dofs_idx_local=[0])

    # Set different control modes per DOF via public API
    entity.control_dofs_force(0.0, dofs_idx_local=[0])
    entity.control_dofs_velocity(0.0, dofs_idx_local=[1])
    entity.control_dofs_position(0.0, dofs_idx_local=[2])
    ctrl_mode = gs_sim.rigid_solver.dofs_state.ctrl_mode.to_numpy()[:, 0]
    assert ctrl_mode[entity.dof_start + 0] == gs.CTRL_MODE.FORCE
    assert ctrl_mode[entity.dof_start + 1] == gs.CTRL_MODE.VELOCITY
    assert ctrl_mode[entity.dof_start + 2] == gs.CTRL_MODE.POSITION

    # control_dofs_position overrides all to POSITION
    entity.control_dofs_position([0.0, 0.0, 0.0])
    ctrl_mode = gs_sim.rigid_solver.dofs_state.ctrl_mode.to_numpy()[:, 0]
    assert (ctrl_mode[entity.dof_start : entity.dof_start + 3] == gs.CTRL_MODE.POSITION).all()

    # Disable constraints, keep actuation enabled
    mj_sim.model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONSTRAINT
    gs_sim.rigid_solver._enable_collision = False
    gs_sim.rigid_solver._enable_joint_limit = False
    gs_sim.rigid_solver._disable_constraint = True
    gs_sim.rigid_solver.collider.clear()
    gs_sim.rigid_solver.constraint_solver.clear()

    # Compare all dynamic quantities against MuJoCo with both PD and general actuators active.
    check_mujoco_model_consistency(gs_sim, mj_sim, tol=tol)
    init_simulators(gs_sim, mj_sim, qpos=np.array([0.2, 0.1, 0.0]), qvel=np.array([0.1, -0.1, 0.0]))

    mj_sim.data.ctrl[:] = [0.5, 0.3, 1.0]
    entity.control_dofs_position([0.5, 0.3, 0.0])
    entity.control_dofs_force(5.0, dofs_idx_local=[2])  # motor: gear(5) * gainprm(1) * ctrl(1) = 5

    # Pre-step so that Genesis computes qf_applied (needed for data consistency checks)
    mj_sim.data.qpos[:] = gs_sim.rigid_solver.qpos.to_numpy()[:, 0]
    mj_sim.data.qvel[:] = gs_sim.rigid_solver.dofs_state.vel.to_numpy()[:, 0]
    mujoco.mj_step(mj_sim.model, mj_sim.data)
    gs_sim.scene.step()

    for _ in range(99):
        check_mujoco_data_consistency(gs_sim, mj_sim, tol=tol, ignore_constraints=True)

        mj_sim.data.qpos[:] = gs_sim.rigid_solver.qpos.to_numpy()[:, 0]
        mj_sim.data.qvel[:] = gs_sim.rigid_solver.dofs_state.vel.to_numpy()[:, 0]
        mujoco.mj_step(mj_sim.model, mj_sim.data)
        gs_sim.scene.step()

    # Validate setter/getter round-trips for actuator parameters
    entity.set_dofs_act_gain([200.0], dofs_idx_local=[1])
    assert_allclose(entity.get_dofs_act_gain()[1], 200.0, tol=1e-6)
    entity.set_dofs_act_bias([0.5], [-100.0], [-5.0], dofs_idx_local=[1])
    b0, b1, b2 = entity.get_dofs_act_bias()
    assert_allclose(b0[1], 0.5, tol=1e-6)
    assert_allclose(b1[1], -100.0, tol=1e-6)
    assert_allclose(b2[1], -5.0, tol=1e-6)

    # set_dofs_kp restores PD on joint 1: act_gain=kp, act_bias[0]=0, act_bias[1]=-kp
    entity.set_dofs_kp([50.0], dofs_idx_local=[1])
    assert_allclose(entity.get_dofs_kp(dofs_idx_local=[0, 1]), [100.0, 50.0], tol=1e-6)
    b0, b1, _ = entity.get_dofs_act_bias()
    assert_allclose(b0[1], 0.0, tol=1e-6)
    assert_allclose(b1[1], -50.0, tol=1e-6)


@pytest.mark.required
@pytest.mark.parametrize("backend", [gs.cpu])
def test_position_control(show_viewer):
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            substeps=1,  # This is essential to be able to emulate native PD control
        ),
        rigid_options=gs.options.RigidOptions(
            batch_links_info=True,
            batch_dofs_info=True,
            disable_constraint=True,
            integrator=gs.integrator.approximate_implicitfast,
        ),
        show_viewer=show_viewer,
        show_FPS=False,
    )
    robot = scene.add_entity(
        gs.morphs.MJCF(
            file="xml/franka_emika_panda/panda.xml",
        ),
    )
    scene.build(n_envs=2, env_spacing=(1.0, 1.0))

    MOTORS_POS_TARGET = torch.tensor(
        [0.6900, -0.1100, -0.7200, -2.7300, -0.1500, 2.6400, 0.8900, 0.0400, 0.0400],
        dtype=gs.tc_float,
        device=gs.device,
    )
    MOTORS_VEL_TARGET = torch.rand_like(MOTORS_POS_TARGET)
    MOTORS_KP = torch.tensor(
        [4500.0, 4500.0, 3500.0, 3500.0, 2000.0, 2000.0, 2000.0, 100.0, 100.0],
        dtype=gs.tc_float,
        device=gs.device,
    )
    MOTORS_KD = torch.tensor(
        [450.0, 450.0, 350.0, 350.0, 200.0, 200.0, 200.0, 10.0, 10.0],
        dtype=gs.tc_float,
        device=gs.device,
    )

    # FIXME: We do NOT raise exception anymore when setting control targets that would have no effect
    # robot.set_dofs_kp(torch.zeros_like(MOTORS_KP), envs_idx=0)
    # robot.set_dofs_kv(torch.zeros_like(MOTORS_KD), envs_idx=0)
    # with pytest.raises(gs.GenesisException):
    #     robot.control_dofs_position(MOTORS_POS_TARGET, envs_idx=0)
    # with pytest.raises(gs.GenesisException):
    #     robot.control_dofs_position_velocity(MOTORS_POS_TARGET, MOTORS_VEL_TARGET, envs_idx=0)
    # with pytest.raises(gs.GenesisException):
    #     robot.control_dofs_velocity(MOTORS_VEL_TARGET, envs_idx=0)
    # robot.set_dofs_kv(MOTORS_KD, envs_idx=0)
    # robot.control_dofs_velocity(MOTORS_VEL_TARGET, envs_idx=0)
    # with pytest.raises(gs.GenesisException):
    #     robot.control_dofs_position(MOTORS_POS_TARGET, envs_idx=0)
    # robot.control_dofs_position_velocity(MOTORS_POS_TARGET, MOTORS_VEL_TARGET, envs_idx=0)

    robot.set_dofs_kp(MOTORS_KP, envs_idx=0)
    robot.set_dofs_kv(MOTORS_KD, envs_idx=0)
    robot.control_dofs_position(MOTORS_POS_TARGET, envs_idx=0)
    robot.control_dofs_position_velocity(MOTORS_POS_TARGET, MOTORS_VEL_TARGET, envs_idx=0)

    # Must update DoF armature to emulate implicit damping for force control.
    # This is equivalent to the first-order correction term involved in implicit integration scheme,
    # in the particular case where `approximate_implicitfast` integrator is used.
    # Note that the low-level internal API is used because invweights must NOT be updated, otherwise
    # the test cannot pass. This is unecessary and not recommended for practical applications.
    # robot.set_dofs_armature(robot.get_dofs_armature(envs_idx=1) + MOTORS_KD * scene.sim._substep_dt, envs_idx=1)
    dofs_armature = scene.rigid_solver.dofs_info.armature.to_numpy()
    dofs_armature[:, 1] += tensor_to_array(MOTORS_KD * scene.sim._substep_dt)
    scene.rigid_solver.dofs_info.armature.from_numpy(dofs_armature)

    force_range = qd_to_torch(scene.rigid_solver.dofs_info.force_range)
    for i in range(200):
        dofs_pos = robot.get_qpos(envs_idx=1)
        dofs_vel = robot.get_dofs_velocity(envs_idx=1)
        dofs_torque = MOTORS_KP * (MOTORS_POS_TARGET - dofs_pos) + MOTORS_KD * (MOTORS_VEL_TARGET - dofs_vel)
        dofs_torque.clamp_(force_range[:, 1, 0], force_range[:, 1, 1])
        robot.control_dofs_force(dofs_torque, envs_idx=1)
        scene.step()
        qf_applied = scene.rigid_solver.dofs_state.qf_applied.to_numpy().T
        # dofs_torque = robot.get_dofs_control_force()
        assert_allclose(qf_applied[1], dofs_torque, tol=1e-6)
        assert_allclose(qf_applied[0], qf_applied[1], tol=1e-6)

    A = 0.1
    f = 1.0
    scene.reset()
    robot.set_dofs_kp(MOTORS_KP, envs_idx=1)
    robot.set_dofs_kv(MOTORS_KD, envs_idx=1)
    force_range[:, 1, 0] = float("-inf")
    force_range[:, 1, 1] = float("+inf")
    scene.rigid_solver.dofs_info.force_range.from_numpy(tensor_to_array(force_range))
    for i in range(1000):
        t = scene.t * scene.dt
        pos_target = A * np.sin(2 * np.pi * f * t)
        vel_target = A * 2 * np.pi * f * np.cos(2 * np.pi * f * t)
        robot.control_dofs_position_velocity(torch.full((9,), pos_target), torch.full((9,), vel_target), envs_idx=1)
        scene.step()
        assert_allclose(pos_target, robot.get_dofs_position(envs_idx=1), tol=1e-2)


@pytest.mark.required
@pytest.mark.parametrize("backend", [gs.cpu, gs.gpu])
@pytest.mark.parametrize("robot_path", ["xml/franka_emika_panda/panda.xml"])
def test_reset_control(robot_path, tol):
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=0.01,
        ),
        rigid_options=gs.options.RigidOptions(
            enable_collision=False,
        ),
        show_viewer=False,
        show_FPS=False,
    )
    robot = scene.add_entity(gs.morphs.MJCF(file=robot_path))
    scene.build()
    qpos = np.random.rand(robot.n_dofs)
    robot.set_dofs_position(qpos)
    robot.control_dofs_position(torch.zeros((robot.n_dofs,), dtype=gs.tc_float, device=gs.device))
    old_control_force = robot.get_dofs_control_force()
    scene.reset()
    new_control_force = robot.get_dofs_control_force()
    assert old_control_force.abs().max() > gs.EPS
    assert_allclose(new_control_force, 0, tol=gs.EPS)


@pytest.mark.required
def test_drone_propellers_force_substep_consistency(show_viewer, tol):
    BASE_RPM = 15000

    scene_ref = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=0.004,
            substeps=1,
        ),
        show_viewer=show_viewer,
    )
    drone_ref = scene_ref.add_entity(
        morph=gs.morphs.Drone(
            file="urdf/drones/cf2x.urdf",
            pos=(0, 0, 1),
        ),
    )
    scene_ref.build(n_envs=2)

    # This not only tests setter, but also proper reset (tracking and clearing applied external force)
    drone_ref.set_propellers_rpm(BASE_RPM)
    with np.testing.assert_raises(gs.GenesisException):
        drone_ref.set_propellers_rpm(BASE_RPM)
    scene_ref.reset()
    drone_ref.set_propellers_rpm((BASE_RPM,) * 4)
    scene_ref.reset()
    drone_ref.set_propellers_rpm(torch.full((scene_ref.n_envs, 4), fill_value=BASE_RPM))
    scene_ref.reset()

    for _ in range(500):
        drone_ref.set_propellers_rpm(BASE_RPM)
        scene_ref.step()

    scene_test = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=0.02,
            substeps=5,
        ),
        show_viewer=show_viewer,
    )
    drone_test = scene_test.add_entity(
        morph=gs.morphs.Drone(
            file="urdf/drones/cf2x.urdf",
            pos=(0, 0, 1.0),
        ),
    )
    scene_test.build()
    for _ in range(100):
        drone_test.set_propellers_rpm(BASE_RPM)
        scene_test.step()

    pos_ref = drone_ref.get_dofs_position()
    pos_test = drone_test.get_dofs_position()
    assert_allclose(pos_ref, pos_test, tol=tol)


@pytest.mark.required
@pytest.mark.parametrize("backend", [gs.cpu])
def test_drone_advanced(show_viewer):
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=0.005,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(2.5, 0.0, 1.5),
            camera_lookat=(0.0, 0.0, 0.5),
        ),
        show_viewer=show_viewer,
        show_FPS=False,
    )
    plane = scene.add_entity(gs.morphs.Plane())
    asset_path = get_hf_dataset(pattern="drone_sus/*")
    drones = []
    for offset, merge_fixed_links in ((-0.3, False), (0.3, True)):
        drone = scene.add_entity(
            morph=gs.morphs.Drone(
                file=f"{asset_path}/drone_sus/drone_sus.urdf",
                merge_fixed_links=merge_fixed_links,
                pos=(0.0, offset, 1.5),
            ),
            vis_mode="collision",
            visualize_contact=True,
        )
        drones.append(drone)
    scene.build()

    for drone in drones:
        assert drone.base_link.parent_idx == -1
        assert_equal([link.root_idx for link in drone.links], drone.base_link.idx)
        assert all(drone.link_start <= link.parent_idx < link.idx for link in drone.links[1:])

    for drone in drones:
        chain_dofs = range(6, drone.n_dofs)
        drone.set_dofs_armature(drone.get_dofs_armature(chain_dofs) + 1e-3, chain_dofs)

    # Wait for the drones to land on the ground and hold straight
    for i in range(400):
        for drone in drones:
            drone.set_propellers_rpm(50000.0)
        scene.step()
        if i > 350:
            assert scene.rigid_solver.collider._collider_state.n_contacts.to_numpy()[0] == 2
            assert_allclose(scene.rigid_solver.get_dofs_velocity(), 0, tol=2e-3)

    # Push the drones symmetrically and wait for them to collide
    drones[0].set_dofs_velocity([0.2], [1])
    drones[1].set_dofs_velocity([-0.2], [1])
    for i in range(150):
        for drone in drones:
            drone.set_propellers_rpm(50000.0)
        scene.step()
        if scene.rigid_solver.collider._collider_state.n_contacts.to_numpy()[0] > 2:
            break
    else:
        raise AssertionError

    tol = 1e-2
    pos_1 = drones[0].get_pos()
    pos_2 = drones[1].get_pos()
    assert abs(pos_1[0] - pos_2[0]) < tol
    assert abs(pos_1[1] + pos_2[1]) < tol
    assert abs(pos_1[2] - pos_2[2]) < tol
    quat_1 = drones[0].get_quat()
    quat_2 = drones[1].get_quat()
    assert abs(quat_1[1] + quat_2[1]) < tol
    assert abs(quat_1[2] - quat_2[2]) < tol
    assert abs(quat_1[2] - quat_2[2]) < tol
