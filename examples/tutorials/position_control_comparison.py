# This example compares the position control accuracy between 'control_dofs_position' and
# 'control_dofs_position_velocity' when tracking a dynamic trajectory.
# While both are equivalent in static, the former lacks the target velocity term of  true PD controller in robotics,
# making it underperform compared to 'control_dofs_position_velocity'.
import argparse
import math

import matplotlib.pyplot as plt

import genesis as gs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", default=False)
    parser.add_argument("-c", "--cpu", action="store_true", default=False)
    args = parser.parse_args()

    ########################## init ##########################
    gs.init(backend=gs.cpu if args.cpu else gs.gpu)

    ########################## create a scene ##########################
    scene = gs.Scene(
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0, -3.5, 2.5),
            camera_lookat=(0.0, 0.0, 0.5),
            camera_fov=30,
            max_FPS=None,
        ),
        sim_options=gs.options.SimOptions(
            dt=0.005,
        ),
        show_viewer=False,
        show_FPS=True,
    )

    ########################## entities ##########################
    plane = scene.add_entity(
        gs.morphs.Plane(),
    )
    franka = scene.add_entity(
        gs.morphs.MJCF(
            file="xml/franka_emika_panda/panda.xml",
        ),
    )
    ########################## build ##########################
    scene.build()

    joints_name = (
        "joint1",
        "joint2",
        "joint3",
        "joint4",
        "joint5",
        "joint6",
        "joint7",
        "finger_joint1",
        "finger_joint2",
    )
    motors_dof_idx = [franka.get_joint(name).dofs_idx_local[0] for name in joints_name]

    ############ Optional: set control gains ############
    # set positional gains
    franka.set_dofs_kp(
        kp=[4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100],
        dofs_idx_local=motors_dof_idx,
    )
    # set velocity gains
    franka.set_dofs_kv(
        kv=[450, 450, 350, 350, 200, 200, 200, 10, 10],
        dofs_idx_local=motors_dof_idx,
    )
    # set force range for safety
    franka.set_dofs_force_range(
        lower=[-87, -87, -87, -87, -12, -12, -12, -100, -100],
        upper=[87, 87, 87, 87, 12, 12, 12, 100, 100],
        dofs_idx_local=motors_dof_idx,
    )
    # Hard reset
    # Follow a sinusoid trajectory
    A = 0.5  # motion amplitude, rad
    f = 1.0  # motion frequency, Hz

    # Use control_dofs_position
    pos_simulation_result = []
    franka.set_dofs_position([A, 0, 0, 0, 0, 0, 0, 0, 0], motors_dof_idx)
    t0 = scene.t
    while (t := (scene.t - t0) * scene.dt) < 2.0:
        target_position = A * (1 + math.sin(2 * math.pi * f * t))

        current_position = float(franka.get_qpos()[0])
        pos_simulation_result.append([t, current_position, target_position])
        franka.control_dofs_position([target_position, 0, 0, 0, 0, 0, 0, 0, 0], motors_dof_idx)
        scene.step()

    # Use control_dofs_position_velocity
    pos_vel_simulation_result = []
    franka.set_dofs_position([A, 0, 0, 0, 0, 0, 0, 0, 0], motors_dof_idx)
    t0 = scene.t
    while (t := (scene.t - t0) * scene.dt) < 2.0:
        target_position = A * (1 + math.sin(2 * math.pi * f * t))
        target_velocity = 2 * math.pi * f * A * math.cos(2 * math.pi * f * t)

        current_position = float(franka.get_qpos()[0])
        pos_vel_simulation_result.append([t, current_position, target_position])
        franka.control_dofs_position_velocity(
            [target_position, 0, 0, 0, 0, 0, 0, 0, 0],
            [target_velocity, 0, 0, 0, 0, 0, 0, 0, 0],
            motors_dof_idx,
        )
        scene.step()

    # Plot results
    pos_simulation_result = tuple(zip(*pos_simulation_result))
    pos_vel_simulation_result = tuple(zip(*pos_vel_simulation_result))

    plt.plot(pos_simulation_result[0], pos_simulation_result[1], label="control_dofs_position")
    plt.plot(pos_vel_simulation_result[0], pos_vel_simulation_result[1], label="control_dofs_position_velocity")
    plt.plot(pos_vel_simulation_result[0], pos_vel_simulation_result[2], color="black", label="Target position")
    plt.xlabel("Time (s)")
    plt.ylabel("Joint position (rad)")
    plt.title("Comparison of joint position tracking with two different controllers")
    plt.grid()
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
