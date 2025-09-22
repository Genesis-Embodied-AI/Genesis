import argparse

import numpy as np
from tqdm import tqdm

import genesis as gs
from genesis.recorders.plotters import IS_MATPLOTLIB_AVAILABLE, IS_PYQTGRAPH_AVAILABLE


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-dt", "--timestep", type=float, default=1e-2, help="Simulation time step")
    parser.add_argument("-v", "--vis", action="store_true", help="Show visualization GUI", default=True)
    parser.add_argument("-nv", "--no-vis", action="store_false", dest="vis", help="Disable visualization GUI")
    parser.add_argument("-c", "--cpu", action="store_true", help="Use CPU instead of GPU")
    parser.add_argument("-t", "--seconds", type=float, default=3, help="Number of seconds to simulate")
    args = parser.parse_args()

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
    franka = scene.add_entity(
        gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"),
    )
    end_effector = franka.get_link("hand")
    motors_dof = (0, 1, 2, 3, 4, 5, 6)

    imu = scene.add_sensor(
        gs.sensors.IMU(
            entity_idx=franka.idx,
            link_idx_local=end_effector.idx_local,
            # noise parameters
            acc_axes_skew=(0.0, 0.01, 0.02),
            gyro_axes_skew=(0.03, 0.04, 0.05),
            acc_noise=(0.01, 0.01, 0.01),
            gyro_noise=(0.01, 0.01, 0.01),
            acc_random_walk=(0.001, 0.001, 0.001),
            gyro_random_walk=(0.001, 0.001, 0.001),
            delay=0.01,
            jitter=0.01,
            interpolate=True,
        )
    )
    labels = {"lin_acc": ("acc_x", "acc_y", "acc_z"), "ang_vel": ("gyro_x", "gyro_y", "gyro_z")}
    if args.vis:
        if IS_PYQTGRAPH_AVAILABLE:
            imu.start_recording(gs.recorders.PyQtPlot(title="IMU Measured Data", labels=labels))
            scene.start_recording(
                imu.read_ground_truth,
                gs.recorders.PyQtPlot(title="IMU Ground Truth Data", labels=labels),
            )
        elif IS_MATPLOTLIB_AVAILABLE:
            gs.logger.info("pyqtgraph not found, falling back to matplotlib.")
            imu.start_recording(gs.recorders.MPLPlot(title="IMU Measured Data", labels=labels))
            scene.start_recording(
                imu.read_ground_truth,
                gs.recorders.MPLPlot(title="IMU Ground Truth Data", labels=labels),
            )
        else:
            print("matplotlib or pyqtgraph not found, skipping real-time plotting.")

    imu.start_recording(gs.recorders.NPZFile(filename="imu_data.npz"))

    ########################## build ##########################
    scene.build()

    franka.set_dofs_kp(
        np.array([4500.0, 4500.0, 3500.0, 3500.0, 2000.0, 2000.0, 2000.0, 100.0, 100.0]),
    )
    franka.set_dofs_kv(
        np.array([450.0, 450.0, 350.0, 350.0, 200.0, 200.0, 200.0, 10.0, 10.0]),
    )
    franka.set_dofs_force_range(
        np.array([-87.0, -87.0, -87.0, -87.0, -12.0, -12.0, -12.0, -100.0, -100.0]),
        np.array([87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0, 100.0, 100.0]),
    )

    circle_center = np.array([0.4, 0.0, 0.5])
    circle_radius = 0.15
    rate = np.deg2rad(2.0)

    try:
        steps = int(args.seconds / args.timestep)
        for i in tqdm(range(steps)):
            scene.step()

            # control franka to move in a circle and draw the target positions
            pos = circle_center + np.array([np.cos(i * rate), np.sin(i * rate), 0]) * circle_radius
            qpos = franka.inverse_kinematics(
                link=end_effector,
                pos=pos,
                quat=np.array([0.0, 1.0, 0.0, 0.0]),
            )
            franka.control_dofs_position(qpos[:-2], motors_dof)
            scene.draw_debug_sphere(pos, radius=0.01, color=(1.0, 0.0, 0.0, 0.5))

    except KeyboardInterrupt:
        gs.logger.info("Simulation interrupted, exiting.")
    finally:
        gs.logger.info("Simulation finished.")

        scene.stop_recording()

        print("Ground truth data:")
        print(imu.read_ground_truth())
        print("Measured data:")
        print(imu.read())


if __name__ == "__main__":
    main()
