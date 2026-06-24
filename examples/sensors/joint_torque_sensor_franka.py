"""
Joint torque sensing on a Franka Emika Panda using the JointTorqueSensor.

The sensor measures the torque at each actuator output shaft:

    tau_sensor = tau_control − I_arm · q̈ + tau_frictionloss

In free space tau_sensor ≈ gravity + Coriolis load; when the arm presses against
the wall box it also captures the contact reaction.
"""
import argparse
import os
import time
import matplotlib.pyplot as plt

import numpy as np
from tqdm import tqdm

import genesis as gs
from genesis.recorders.plotters import IS_MATPLOTLIB_AVAILABLE, IS_PYQTGRAPH_AVAILABLE


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-dt", "--timestep", type=float, default=1e-2)
    parser.add_argument("-v", "--vis", action="store_true", default=True)
    parser.add_argument("-nv", "--no-vis", action="store_false", dest="vis")
    parser.add_argument("-c", "--cpu", action="store_true")
    parser.add_argument("-t", "--seconds", type=float, default=5.0)
    args = parser.parse_args()
    steps = int(args.seconds / args.timestep) if "PYTEST_VERSION" not in os.environ else 5

    ########################## init ##########################
    gs.init(backend=gs.cpu if args.cpu else gs.gpu)

    ########################## scene ##########################
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=args.timestep),
        vis_options=gs.options.VisOptions(show_world_frame=False),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(3.5, 0.0, 2.5),
            camera_lookat=(0.0, 0.0, 0.5),
            camera_fov=40,
        ),
        profiling_options=gs.options.ProfilingOptions(show_FPS=False),
        show_viewer=args.vis,
    )

    ########################## entities ##########################
    scene.add_entity(gs.morphs.Plane())
    franka = scene.add_entity(gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"))

    # Fixed wall: the robot will press its end-effector into it in the second half.
    scene.add_entity(
        gs.morphs.Box(pos=(0.60, 0.0, 0.50), size=(0.04, 0.20, 0.20), fixed=True)
    )

    end_effector = franka.get_link("hand")
    motors_dof = (0, 1, 2, 3, 4, 5, 6)

    ########################## sensor ##########################
    franka_entity_idx = franka.idx  # scene-level entity index
    torque_sensor = scene.add_sensor(
        gs.options.sensors.JointTorqueSensor(
            entity_idx=franka_entity_idx,
            dofs_idx_local=motors_dof,  # None would select all DOFs
        )
    )

    ########################## recording ##########################
    def data_func():
        tau = torque_sensor.read()                            # (7,) joint torques
        tau_ctrl = franka.get_dofs_control_force(motors_dof)  # PD output
        qdot = franka.get_dofs_velocity(motors_dof)
        return {
            "tau_sensor (J0-J2)": tau[:3],
            "tau_ctrl (J0-J2)": tau_ctrl[:3],
            # residual ≈ 0 in free space, nonzero under contact load
            "residual (J0-J2)": (tau - tau_ctrl)[:3],
            "qdot (J0-J2) [rad/s]": qdot[:3],
        }

    if args.vis:
        if IS_PYQTGRAPH_AVAILABLE:
            scene.start_recording(
                data_func,
                gs.recorders.PyQtLinePlot(title="JointTorqueSensor — Franka"),
            )
        elif IS_MATPLOTLIB_AVAILABLE:
            gs.logger.info("pyqtgraph not found, falling back to matplotlib.")
            scene.start_recording(
                data_func,
                gs.recorders.MPLLinePlot(
                    title="JointTorqueSensor — Franka", history_length=steps
                ),
            )

    scene.start_recording(
        data_func=data_func,
        rec_options=gs.recorders.NPZFile(filename="joint_torque_sensor_franka.npz"),
    )

    ########################## build ##########################
    scene.build()

    # franka.set_dofs_armature(np.zeros(franka.n_dofs))
    # franka.set_dofs_frictionloss(np.zeros(franka.n_dofs) + 20.2)


    franka.set_dofs_kp(
        np.array([4500.0, 4500.0, 3500.0, 3500.0, 2000.0, 2000.0, 2000.0, 100.0, 100.0])
    )
    franka.set_dofs_kv(
        np.array([450.0, 450.0, 350.0, 350.0, 200.0, 200.0, 200.0, 10.0, 10.0])
    )
    franka.set_dofs_force_range(
        np.array([-87.0, -87.0, -87.0, -87.0, -12.0, -12.0, -12.0, -100.0, -100.0]),
        np.array([87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0, 100.0, 100.0]),
    )

    ee_quat = np.array([0.0, 1.0, 0.0, 0.0])
    pos_hold = np.array([0.45, 0.0, 0.50])
    qpos_hold = franka.inverse_kinematics(link=end_effector, pos=pos_hold, quat=ee_quat)
    franka.set_qpos(qpos_hold)

    pos_push = np.array([0.70, 0.0, 0.50])
    qpos_push = franka.inverse_kinematics(link=end_effector, pos=pos_push, quat=ee_quat)

    contact_step = steps // 2

    ########################## simulate ##########################
    try:
        for i in tqdm(range(steps)):
            scene.step()
            target = qpos_hold if i < contact_step else qpos_push
            franka.control_dofs_position(target[:-2], motors_dof)
    except KeyboardInterrupt:
        gs.logger.info("Simulation interrupted.")
    finally:
        gs.logger.info("Simulation finished.")
        plt.show()
        while True:
            time.sleep(0.1)
