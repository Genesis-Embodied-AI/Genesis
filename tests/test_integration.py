import numpy as np
import pytest

import genesis as gs

from .utils import assert_allclose


@pytest.mark.parametrize("mode", [0, 1, 2])
@pytest.mark.parametrize("backend", [gs.cpu, gs.gpu])
def test_pick_and_place(mode, show_viewer):
    # Add DoF armature to improve numerical stability if not using 'approximate_implicitfast' integrator.
    #
    # This is necessary because the first-order correction term involved in the implicit integration schemes
    # 'implicitfast' and 'Euler' are only able to stabilize each entity independently, from the forces that were
    # obtained from the instable accelerations. As a result, eveything is fine as long as the entities are not
    # interacting with each other, but it induces unrealistic motion otherwise. In this case, the acceleration of the
    # cube being lifted is based on the acceleration that the gripper would have without implicit damping.
    #
    # The only way to correct this would be to take into account the derivative of the Jacobian of the constraints in
    # the first-order correction term. Doing this is challenging and would significantly increase the computation cost.
    #
    # In practice, it is more common to just go for a higher order integrator such as RK4.
    if mode == 0:
        integrator = gs.integrator.approximate_implicitfast
        substeps = 1
        armature = 0.0
    elif mode == 1:
        integrator = gs.integrator.implicitfast
        substeps = 4
        armature = 0.0
    elif mode == 2:
        integrator = gs.integrator.Euler
        substeps = 1
        armature = 2.0

    # Create and build the scene
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=0.01,
            substeps=substeps,
        ),
        rigid_options=gs.options.RigidOptions(
            box_box_detection=True,
            integrator=integrator,
        ),
        show_viewer=show_viewer,
        show_FPS=False,
    )
    plane = scene.add_entity(
        gs.morphs.Plane(),
    )
    cube = scene.add_entity(
        gs.morphs.Box(
            size=(0.05, 0.05, 0.05),
            pos=(0.65, 0.0, 0.025),
        ),
        surface=gs.surfaces.Plastic(color=(1, 0, 0)),
    )
    cube_2 = scene.add_entity(
        gs.morphs.Box(
            size=(0.05, 0.05, 0.05),
            pos=(0.4, 0.2, 0.025),
        ),
        surface=gs.surfaces.Plastic(color=(0, 1, 0)),
    )
    franka = scene.add_entity(
        gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"),
        vis_mode="collision",
        visualize_contact=True,
    )
    scene.build()

    franka.set_dofs_armature(franka.get_dofs_armature() + armature)

    motors_dof = np.arange(7)
    fingers_dof = np.arange(7, 9)
    end_effector = franka.get_link("hand")

    # set control gains
    franka.set_dofs_kp(
        np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100]),
    )
    franka.set_dofs_kv(
        np.array([450, 450, 350, 350, 200, 200, 200, 10, 10]),
    )
    franka.set_dofs_force_range(
        np.array([-87, -87, -87, -87, -12, -12, -12, -100, -100]),
        np.array([87, 87, 87, 87, 12, 12, 12, 100, 100]),
    )

    # move to pre-grasp pose
    qpos = franka.inverse_kinematics(
        link=end_effector,
        pos=np.array([0.65, 0.0, 0.22]),
        quat=np.array([0, 1, 0, 0]),
    )
    # gripper open pos
    qpos[-2:] = 0.04
    path = franka.plan_path(qpos_goal=qpos, num_waypoints=300, resolution=0.05, max_retry=10)
    # execute the planned path
    franka.control_dofs_position(np.array([0.15, 0.15]), fingers_dof)
    for waypoint in path:
        franka.control_dofs_position(waypoint)
        scene.step()

    # Get more time to the robot to reach the last waypoint
    for i in range(120):
        scene.step()

    # reach
    qpos = franka.inverse_kinematics(
        link=end_effector,
        pos=np.array([0.65, 0.0, 0.13]),
        quat=np.array([0, 1, 0, 0]),
    )
    franka.control_dofs_position(qpos[:-2], motors_dof)
    for i in range(60):
        scene.step()

    # grasp
    franka.control_dofs_position(qpos[:-2], motors_dof)
    franka.control_dofs_force(np.array([-1.0, -1.0]), fingers_dof)
    for i in range(50):
        scene.step()

    # lift
    qpos = franka.inverse_kinematics(
        link=end_effector,
        pos=np.array([0.65, 0.0, 0.28]),
        quat=np.array([0, 1, 0, 0]),
    )
    franka.control_dofs_position(qpos[:-2], motors_dof)
    for i in range(50):
        scene.step()

    # reach
    qpos = franka.inverse_kinematics(
        link=end_effector,
        pos=np.array([0.4, 0.2, 0.2]),
        quat=np.array([0, 1, 0, 0]),
    )
    path = franka.plan_path(
        qpos_goal=qpos,
        num_waypoints=100,
        resolution=0.05,
        max_retry=10,
        ee_link_name="hand",
        with_entity=cube,
    )
    for waypoint in path:
        franka.control_dofs_position(waypoint[:-2], motors_dof)
        scene.step()

    # Get more time to the robot to reach the last waypoint
    for i in range(50):
        scene.step()

    # release
    franka.control_dofs_position(np.array([0.15, 0.15]), fingers_dof)

    for i in range(550):
        scene.step()
        if i > 550:
            qvel = cube.get_dofs_velocity()
            assert_allclose(qvel, 0, atol=0.02)

    qpos = cube.get_dofs_position()
    assert_allclose(qpos[2], 0.075, atol=2e-3)
