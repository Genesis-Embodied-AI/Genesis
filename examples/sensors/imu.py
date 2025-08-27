import argparse

import numpy as np
from custom_data_handlers import IS_PYQTGRAPH_AVAILABLE, PyQtGraphPlotter
from tqdm import tqdm

import genesis as gs
from genesis.sensors.data_handlers import NPZFileWriter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", default=True)
    parser.add_argument("-c", "--cpu", action="store_true", default=False)
    args = parser.parse_args()

    ########################## init ##########################
    gs.init(backend=gs.cpu if args.cpu else gs.gpu)

    ########################## create a scene ##########################
    scene = gs.Scene(
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(3.5, 0.0, 2.5),
            camera_lookat=(0.0, 0.0, 0.5),
            camera_fov=40,
        ),
        sim_options=gs.options.SimOptions(
            dt=0.01,
        ),
        profiling_options=gs.options.ProfilingOptions(
            show_FPS=False,
        ),
        show_viewer=args.vis,
    )

    ########################## entities ##########################
    scene.add_entity(gs.morphs.Plane())
    franka = scene.add_entity(
        gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"),
    )
    end_effector = franka.get_link("hand")
    motors_dof = np.arange(7)

    imu = scene.add_sensor(
        gs.sensors.IMU(
            entity_idx=franka.idx,
            link_idx_local=end_effector.idx_local,
            # noise parameters
            acc_axes_skew=(0.0, 0.01, 0.02),
            gyro_axes_skew=(0.03, 0.04, 0.05),
            acc_noise_std=(0.01, 0.01, 0.01),
            gyro_noise_std=(0.01, 0.01, 0.01),
            acc_bias_drift_std=(0.001, 0.001, 0.001),
            gyro_bias_drift_std=(0.001, 0.001, 0.001),
            delay=0.01,
            jitter=0.01,
            interpolate_for_delay=True,
        )
    )
    if IS_PYQTGRAPH_AVAILABLE:
        imu.add_recorder(
            handler=PyQtGraphPlotter(title="IMU Accelerometer Measured Data", labels=["acc_x", "acc_y", "acc_z"]),
            rec_options=gs.options.RecordingOptions(
                preprocess_func=lambda data, ground_truth_data: data["lin_acc"],
            ),
        )
        imu.add_recorder(
            handler=PyQtGraphPlotter(title="IMU Accelerometer Ground Truth Data", labels=["acc_x", "acc_y", "acc_z"]),
            rec_options=gs.options.RecordingOptions(
                preprocess_func=lambda data, ground_truth_data: ground_truth_data["lin_acc"],
            ),
        )
    else:
        print("pyqtgraph not found, skipping real-time plotting and saving IMU data as .npz instead.")
        imu.add_recorder(
            handler=NPZFileWriter(filename="imu_data.npz"),
        )
    imu.start_recording()

    ########################## build ##########################
    scene.build()

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

    circle_center = np.array([0.4, 0.0, 0.5])
    circle_radius = 0.15
    rate = 2 / 180 * np.pi

    def control_franka_circle_path(i):
        pos = circle_center + np.array([np.cos(i * rate), np.sin(i * rate), 0]) * circle_radius
        qpos = franka.inverse_kinematics(
            link=end_effector,
            pos=pos,
            quat=np.array([0, 1, 0, 0]),
        )
        franka.control_dofs_position(qpos[:-2], motors_dof)
        scene.draw_debug_sphere(pos, radius=0.01, color=(1.0, 0.0, 0.0, 0.5))

    try:
        for i in tqdm(range(1000)):
            scene.step()
            control_franka_circle_path(i)
    except KeyboardInterrupt:
        gs.logger.info("Simulation interrupted, exiting.")
    except gs.GenesisException as e:
        gs.logger.error(e)
    finally:
        gs.logger.info("Simulation finished.")

        print("Ground truth data:")
        print(imu.read_ground_truth())
        print("Measured data:")
        print(imu.read())

    imu.stop_recording()


if __name__ == "__main__":
    main()
