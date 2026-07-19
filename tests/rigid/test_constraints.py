import mujoco
import numpy as np
import pytest
import torch

import genesis as gs
import genesis.utils.geom as gu
from genesis.utils.misc import tensor_to_array

from ..utils import (
    assert_allclose,
    simulate_and_check_mujoco_consistency,
)


@pytest.mark.parametrize("model_name", ["mimic_hinges"])
@pytest.mark.parametrize("gs_solver", [gs.constraint_solver.CG, gs.constraint_solver.Newton])
@pytest.mark.parametrize("gs_integrator", [gs.integrator.implicitfast, gs.integrator.Euler])
@pytest.mark.parametrize("backend", [gs.cpu])
def test_equality_joint(gs_sim, mj_sim, gs_solver, tol):
    # there is an equality constraint
    assert gs_sim.rigid_solver.n_equalities == 1

    qpos = np.array((0.0, -1.0))
    qvel = np.array((1.0, -0.3))
    simulate_and_check_mujoco_consistency(gs_sim, mj_sim, qpos, qvel, num_steps=300, tol=tol)

    # check if the two joints are equal
    gs_qpos = gs_sim.rigid_solver.qpos.to_numpy()[:, 0]
    assert_allclose(gs_qpos[0], gs_qpos[1], tol=tol)


@pytest.mark.required
@pytest.mark.parametrize("xml_path", ["xml/four_bar_linkage_weld.xml", "weld.xml", "connect.xml"])
@pytest.mark.parametrize("gs_solver", [gs.constraint_solver.Newton])
@pytest.mark.parametrize("gs_integrator", [gs.integrator.Euler])
@pytest.mark.parametrize("backend", [gs.cpu])
def test_equality_link(gs_sim, mj_sim, gs_solver, xml_path):
    # Must disable self-collision caused by closing the kinematic chain (adjacent link filtering is not enough)
    gs_sim.rigid_solver._enable_collision = False
    mj_sim.model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT

    # Must the time constant of the constraints to improve numerical stability
    TIME_CONSTANT = 0.02
    for entity in gs_sim.entities:
        for equality in entity.equalities:
            equality.set_sol_params((TIME_CONSTANT, *tensor_to_array(equality.sol_params)[1:]))
    mj_sim.model.eq_solref[:, 0] = TIME_CONSTANT

    # Randomize the initial condition for force convergence of the constraints
    np.random.seed(0)
    qpos = np.random.rand(gs_sim.rigid_solver.n_qs) * 0.1

    # Note that the world frame in which weld constraint is computed is different between Mujoco and Genesis for sites.
    # Mujoco is using site 1, whereas Genesis is using parent link frame of site 1 since it has no notion of site.
    ignore_constraints = np.any(
        (mj_sim.model.eq_objtype == mujoco.mjtObj.mjOBJ_SITE) & (mj_sim.model.eq_type == mujoco.mjtEq.mjEQ_WELD)
    )
    simulate_and_check_mujoco_consistency(
        gs_sim, mj_sim, qpos, num_steps=300, tol=1e-7, ignore_constraints=ignore_constraints
    )


@pytest.mark.slow  # ~250s
@pytest.mark.required
def test_dynamic_weld(show_viewer, tol):
    CUBE_POS = (0.65, 0.0, 0.02)

    scene = gs.Scene(
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(5.5, 0.0, 2.5),
            camera_lookat=(1.0, 0.0, 0.0),
        ),
        show_viewer=show_viewer,
        show_FPS=False,
    )
    scene.add_entity(
        gs.morphs.Plane(),
    )
    cube = scene.add_entity(
        gs.morphs.Box(
            size=(0.04, 0.04, 0.04),
            pos=CUBE_POS,
        ),
        surface=gs.surfaces.Default(
            color=(1, 0, 0),
        ),
    )
    robot = scene.add_entity(
        gs.morphs.MJCF(
            file="xml/universal_robots_ur5e/ur5e.xml",
        ),
    )
    scene.build(n_envs=4, env_spacing=(3.0, 3.0))

    end_effector = robot.get_link("ee_virtual_link")

    # Compute up and down robot configurations
    ee_pos_up = np.array((0.65, 0.0, 0.5), dtype=gs.np_float)
    ee_pos_down = np.array((0.65, 0.0, 0.15), dtype=gs.np_float)
    qpos_up = robot.inverse_kinematics(
        link=end_effector,
        pos=np.tile(ee_pos_up, (4, 1)),
        quat=np.tile(np.array((0.0, 1.0, 0.0, 0.0), dtype=gs.np_float), (4, 1)),
    )
    qpos_down = robot.inverse_kinematics(
        link=end_effector,
        pos=np.tile(ee_pos_down, (4, 1)),
        quat=np.tile(np.array((0.0, 1.0, 0.0, 0.0), dtype=gs.np_float), (4, 1)),
    )

    # move to pre-grasp pose
    robot.control_dofs_position(qpos_up)
    for i in range(120):
        scene.step()

    # reach
    robot.control_dofs_position(qpos_down)
    for i in range(70):
        scene.step()

    # add weld constraint and move back up
    scene.sim.rigid_solver.add_weld_constraint(cube.base_link.idx, end_effector.idx, envs_idx=(0, 1, 2))
    robot.control_dofs_position(qpos_up)
    for _ in range(60):
        scene.step()
    cubes_pos, cubes_quat = cube.get_pos(), tensor_to_array(cube.get_quat())
    assert_allclose(gu.quat_to_rotvec(cubes_quat), 0.0, tol=1e-3)
    assert_allclose(torch.diff(cubes_pos[[0, 1, 2]], dim=0), 0.0, tol=tol)
    assert_allclose(cubes_pos[3], CUBE_POS, tol=1e-3)
    assert_allclose(cubes_pos[-1] - cubes_pos[0], ee_pos_down - ee_pos_up, tol=1e-2)

    # drop
    scene.sim.rigid_solver.delete_weld_constraint(cube.base_link.idx, end_effector.idx, envs_idx=(0, 1))
    for _ in range(110):
        scene.step()
    cubes_pos, cubes_quat = cube.get_pos(), tensor_to_array(cube.get_quat())
    assert_allclose(gu.quat_to_rotvec(cubes_quat), 0.0, tol=1e-3)
    assert_allclose(torch.diff(cubes_pos[[0, 1, 3]], dim=0), 0.0, tol=1e-2)
    assert_allclose(cubes_pos[2] - cubes_pos[0], ee_pos_up - ee_pos_down, tol=1e-3)


@pytest.mark.slow  # ~200s
@pytest.mark.required
def test_dynamic_weld_scene_reset():
    scene = gs.Scene(
        rigid_options=gs.options.RigidOptions(
            max_dynamic_constraints=10,
        ),
        show_viewer=False,
    )
    box1 = scene.add_entity(
        gs.morphs.Box(
            size=(0.1, 0.1, 0.1),
            pos=(0, 0, 0.5),
        )
    )
    box2 = scene.add_entity(
        gs.morphs.Box(
            size=(0.1, 0.1, 0.1),
            pos=(0.2, 0, 0.5),
        )
    )
    scene.build(n_envs=2)

    solver = scene.rigid_solver
    n_eq_base = solver.rigid_info.n_equalities[None]

    solver.add_weld_constraint(box1.base_link_idx, box2.base_link_idx)
    assert solver.constraint_solver.constraint_state.qd_n_equalities[0] == n_eq_base + 1
    assert solver.constraint_solver.constraint_state.qd_n_equalities[1] == n_eq_base + 1

    scene.reset(state=scene.get_state(), envs_idx=[0])
    assert solver.constraint_solver.constraint_state.qd_n_equalities[0] == n_eq_base
    assert solver.constraint_solver.constraint_state.qd_n_equalities[1] == n_eq_base + 1


@pytest.mark.required
def test_urdf_mimic(show_viewer, tol):
    # create and build the scene
    scene = gs.Scene(
        show_viewer=show_viewer,
    )
    hand = scene.add_entity(
        gs.morphs.URDF(
            file="urdf/panda_bullet/hand.urdf",
            fixed=True,
        ),
    )
    scene.build()
    assert scene.rigid_solver.n_equalities == 1

    qvel = scene.rigid_solver.dyn_state.dofs.vel.to_numpy()
    qvel[-1] = 1
    scene.rigid_solver.dyn_state.dofs.vel.from_numpy(qvel)
    for i in range(200):
        scene.step()

    gs_qpos = scene.rigid_solver.qpos.to_numpy()[:, 0]
    assert_allclose(gs_qpos[-1], gs_qpos[-2], tol=tol)


@pytest.mark.slow  # ~200s
@pytest.mark.required
def test_get_constraints_api(show_viewer, tol):
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(gravity=(0.0, 0.0, 0.0)),
        show_viewer=show_viewer,
    )
    robot = scene.add_entity(
        gs.morphs.MJCF(
            file="xml/franka_emika_panda/panda.xml",
        ),
    )
    cube = scene.add_entity(
        gs.morphs.Box(
            size=(0.05, 0.05, 0.05),
            pos=(0.2, 0.0, 0.05),
        )
    )
    scene.build(n_envs=2)

    link_a, link_b = robot.base_link.idx, cube.base_link.idx
    scene.sim.rigid_solver.add_weld_constraint(link_a, link_b, envs_idx=[1])
    with np.testing.assert_raises(AssertionError):
        scene.sim.rigid_solver.add_weld_constraint(link_a, link_b, envs_idx=[1])

    for as_tensor, to_torch in ((True, True), (True, False), (False, True), (False, False)):
        weld_const_info = scene.sim.rigid_solver.get_weld_constraints(as_tensor, to_torch)
        link_a_, link_b_ = weld_const_info["link_a"], weld_const_info["link_b"]
        if as_tensor:
            assert_allclose((link_a_[0], link_b_[0]), ((-1,), (-1,)), tol=0)
        else:
            assert_allclose((link_a_[0], link_b_[0]), ((), ()), tol=0)
        assert_allclose((link_a_[1], link_b_[1]), ((link_a,), (link_b,)), tol=0)


@pytest.mark.slow  # ~200s
@pytest.mark.required
@pytest.mark.parametrize("n_envs, batched", [(0, False), (3, True)])
def test_set_sol_params(n_envs, batched, tol):
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=0.01,
            substeps=1,
        ),
        rigid_options=gs.options.RigidOptions(
            batch_joints_info=batched,
        ),
        show_viewer=False,
        show_FPS=False,
    )
    robot = scene.add_entity(
        gs.morphs.MJCF(
            file="xml/franka_emika_panda/panda.xml",
            pos=(0.0, 0.4, 0.1),
            euler=(0, 0, 90),
        ),
    )
    scene.build(n_envs=2)
    assert scene.sim._substep_dt == 0.01

    for objs, batched in ((robot.joints, batched), (robot.geoms, False), (robot.equalities, True)):
        for obj in objs:
            sol_params = obj.sol_params + 1.0
            obj.set_sol_params(sol_params)
            with pytest.raises(AssertionError):
                assert_allclose(obj.sol_params, sol_params, tol=tol)
            obj.set_sol_params([0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0])
            assert_allclose(obj.sol_params, [2.0e-02, 0.5, 1e-4, 1e-4, 0.0, 1e-4, 1.0], tol=tol)


@pytest.mark.required
def test_comfree_incompatible_options():
    with pytest.raises(gs.GenesisException):
        gs.options.RigidOptions(
            constraint_solver=gs.constraint_solver.ComFree,
            friction_cone=gs.friction_cone.elliptic,
        )
    with pytest.raises(gs.GenesisException):
        gs.options.RigidOptions(
            constraint_solver=gs.constraint_solver.ComFree,
            noslip_iterations=1,
        )


@pytest.mark.required
@pytest.mark.parametrize("n_envs", [0, 2])
def test_comfree_solver(n_envs, comfree_rig, show_viewer):
    DT = 0.01
    GRAVITY = 9.81
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=DT,
            gravity=(0.0, 0.0, -GRAVITY),
        ),
        rigid_options=gs.options.RigidOptions(
            constraint_solver=gs.constraint_solver.ComFree,
        ),
        show_viewer=show_viewer,
    )
    scene.add_entity(
        gs.morphs.Plane(),
    )
    resting_box = scene.add_entity(
        gs.morphs.Box(
            size=(0.2, 0.2, 0.2),
            pos=(0.0, 0.0, 0.11),
        ),
    )
    falling_box = scene.add_entity(
        gs.morphs.Box(
            size=(0.2, 0.2, 0.2),
            pos=(1.0, 0.0, 2.0),
        ),
    )
    rig = scene.add_entity(
        gs.morphs.MJCF(
            file=comfree_rig,
        ),
    )
    scene.build(n_envs=n_envs)

    rig.set_dofs_velocity([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, -2.0, 0.0])
    # Start the pendulum close to its 30 deg limit so it engages the limit gently.
    rig.set_dofs_position(np.radians(25.0), dofs_idx_local=[8])

    # Constraint-free flight must match the smooth dynamics: semi-implicit Euler under gravity in closed form.
    n_flight_steps = 30
    for _ in range(n_flight_steps):
        scene.step()
    z_flight = 2.0 - GRAVITY * DT**2 * n_flight_steps * (n_flight_steps + 1) / 2
    assert_allclose(np.atleast_1d(tensor_to_array(falling_box.get_pos()))[..., 2], z_flight, tol=1e-5)

    # 120 more steps: the falling box lands (~0.6s of flight) and settles, everything else reaches steady state.
    pendulum_qpos = []
    for _ in range(120):
        scene.step()
        pendulum_qpos.append(tensor_to_array(rig.get_qpos())[..., -1])
    pendulum_qpos = np.stack(pendulum_qpos)

    # Both boxes rest on the plane, within the compliant penetration of the analytical contact force.
    for box in (resting_box, falling_box):
        boxes_z = np.atleast_1d(tensor_to_array(box.get_pos()))[..., 2]
        assert ((0.095 < boxes_z) & (boxes_z < 0.102)).all()
        assert_allclose(box.get_vel(), 0.0, tol=1e-4)

    # The connect equality holds the hung body: its rows must pull as well as push.
    hung_z = np.atleast_1d(tensor_to_array(rig.get_link("hung").get_pos()))[..., 2]
    assert ((0.62 < hung_z) & (hung_z < 0.71)).all()

    # Frictionloss decelerates both spin directions symmetrically.
    spins_vel = tensor_to_array(rig.get_dofs_velocity())[..., 6:8]
    assert_allclose(spins_vel[..., 0], -spins_vel[..., 1], tol=1e-6)
    assert (np.abs(spins_vel) < 0.15).all()

    # The pendulum is caught by its range limit: the transient overshoot stays within the limit compliance,
    # and it comes to rest leaning on the limit under its gravity load.
    assert (pendulum_qpos < np.radians(33.5)).all()
    assert ((np.radians(29.0) < pendulum_qpos[-1]) & (pendulum_qpos[-1] < np.radians(31.5))).all()

    # Identical envs must produce identical trajectories.
    if n_envs > 0:
        boxes_pos = tensor_to_array(falling_box.get_pos())
        assert_allclose(boxes_pos, boxes_pos[0], tol=1e-7)
