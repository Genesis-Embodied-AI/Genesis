"""
Joint torque sensing on a Franka Emika Panda using the JointTorqueSensor.

The sensor measures the torque at each actuator output shaft:

    τ_sensor = τ_control − I_arm · q̈ + τ_frictionloss

In free space τ_sensor ≈ gravity + Coriolis load; when the arm presses against
the wall box it also captures the contact reaction.

Three live plots show, for joints J0, J2 and J4, the comparison between the
control torque, the sensor reading and the gravity + Coriolis reference.
"""

import argparse
import os
import matplotlib.pyplot as plt

import numpy as np
import torch
from tqdm import tqdm

import genesis as gs
from genesis.recorders.plotters import IS_MATPLOTLIB_AVAILABLE, IS_PYQTGRAPH_AVAILABLE


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-dt", "--timestep", type=float, default=1e-2, help="Simulation time step")
    parser.add_argument("-v", "--vis", action="store_true", help="Show visualization GUI", default=True)
    parser.add_argument("-nv", "--no-vis", action="store_false", dest="vis", help="Disable visualization GUI")
    parser.add_argument("-c", "--cpu", action="store_true", help="Use CPU instead of GPU")
    parser.add_argument("-t", "--seconds", type=float, default=5.0, help="Number of seconds to simulate")
    args = parser.parse_args()

    steps = int(args.seconds / args.timestep) if "PYTEST_VERSION" not in os.environ else 5

    ########################## init ##########################
    gs.init(backend=gs.cpu if args.cpu else gs.gpu)

    ########################## create a scene ##########################
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
    scene.add_entity(gs.morphs.Box(pos=(0.60, 0.0, 0.50), size=(0.04, 0.30, 0.20), fixed=True))

    end_effector = franka.get_link("hand")
    motors_dof = (0, 1, 2, 3, 4, 5, 6)

    ########################## sensor ##########################
    torque_sensor = scene.add_sensor(
        gs.options.sensors.JointTorqueSensor(
            entity_idx=franka.idx,
            dofs_idx_local=motors_dof,
        )
    )

    ########################## recording ##########################
    # Three subplots, one per representative joint (J0, J1).
    # Each subplot shows: control torque / sensor torque / difference.
    channel_labels = ("tau_ctrl", "tau_sensor", "difference")
    labels = {"J0": channel_labels, "J1": channel_labels, "J2": channel_labels}

    def data_func():
        tau_sensor = torque_sensor.read()
        tau_ctrl = franka.get_dofs_control_force(motors_dof)
        return {
            "J0": torch.stack([tau_ctrl[0], tau_sensor[0], tau_sensor[0] - tau_ctrl[0]]),
            "J1": torch.stack([tau_ctrl[1], tau_sensor[1], tau_sensor[1] - tau_ctrl[1]]),
            "J2": torch.stack([tau_ctrl[2], tau_sensor[2], tau_sensor[2] - tau_ctrl[2]]),
        }

    if args.vis:
        if IS_PYQTGRAPH_AVAILABLE:
            scene.start_recording(
                data_func,
                gs.recorders.PyQtLinePlot(title="JointTorqueSensor — Franka", labels=labels),
            )
        elif IS_MATPLOTLIB_AVAILABLE:
            gs.logger.info("pyqtgraph not found, falling back to matplotlib.")
            scene.start_recording(
                data_func,
                gs.recorders.MPLLinePlot(title="JointTorqueSensor — Franka", labels=labels, history_length=steps),
            )
        else:
            print("matplotlib or pyqtgraph not found, skipping real-time plotting.")

    ########################## build ##########################
    scene.build()

    franka.set_dofs_armature(np.full(franka.n_dofs, 1.0))
    franka.set_dofs_frictionloss(np.full(franka.n_dofs, 10.0))

    franka.set_dofs_kp(np.array([4500.0, 4500.0, 3500.0, 3500.0, 2000.0, 2000.0, 2000.0, 100.0, 100.0]))
    franka.set_dofs_kv(np.array([450.0, 450.0, 350.0, 350.0, 200.0, 200.0, 200.0, 10.0, 10.0]))
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

    ########################## simulate ##########################
    try:
        contact_step = steps // 2
        for i in tqdm(range(steps)):
            target = qpos_hold if i < contact_step else qpos_push
            franka.control_dofs_position(target[:-2], motors_dof)
            scene.step()
    except KeyboardInterrupt:
        gs.logger.info("Simulation interrupted, exiting.")
    finally:
        gs.logger.info("Simulation finished.")
        if args.vis:
            plt.show()
        scene.stop_recording()


if __name__ == "__main__":
    main()
