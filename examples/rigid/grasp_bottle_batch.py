import argparse

import numpy as np
import torch
import genesis as gs
from genesis.utils.path_planing import RRT, RRTConnect


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", default=False)
    parser.add_argument("-a", "--algo", type=str, default="rrt", choices=["rrt", "rrt_connect"])
    args = parser.parse_args()

    ########################## init ##########################
    gs.init(backend=gs.gpu)

    ########################## create a scene ##########################
    viewer_options = gs.options.ViewerOptions(
        camera_pos=(3, -1, 1.5),
        camera_lookat=(0.0, 0.0, 0.0),
        camera_fov=30,
        max_FPS=60,
    )

    scene = gs.Scene(
        viewer_options=viewer_options,
        rigid_options=gs.options.RigidOptions(
            dt=0.01,
        ),
        show_FPS=False,
        show_viewer=args.vis,
    )

    ########################## entities ##########################
    plane = scene.add_entity(
        gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True),
    )
    bottle = scene.add_entity(
        material=gs.materials.Rigid(rho=300),
        morph=gs.morphs.URDF(
            file="urdf/3763/mobility_vhacd.urdf",
            scale=0.09,
            pos=(0.65, 0.0, 0.036),
            euler=(0, 90, 0),
        ),
        # visualize_contact=True,
    )
    franka = scene.add_entity(
        gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"),
    )
    
    cam = scene.add_camera(
        res=(1280, 960),
        pos=(3.0, 0.0, 1.5),
        lookat=(0.5, 0.0, 0.5),
        fov=30,
        GUI=False,
    )

    ########################## build ##########################
    n_envs = 100
    envs_idx=torch.arange(n_envs)
    scene.build(n_envs=n_envs)

    end_effector = franka.get_link("hand")
    motors_dof = np.arange(7)
    fingers_dof = np.arange(7, 9)

    # Optional: set control gains
    franka.set_qpos(np.array([[1.56, -0.72, -0.02, -2.09, 0.04, 1.33, 2.4, 0.01, 0.01]]*n_envs))
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
        pos=np.array([[0.65, 0.0, 0.25]]*n_envs),
        quat=np.array([[0, 1, 0, 0]]*n_envs),
    )
    qpos[:,-2:] = 0.04

    

    if args.algo == "rrt_connect":
        planner = RRTConnect(franka)
    elif args.algo == "rrt":
        planner = RRT(franka)
    else:
        raise

    path = planner.plan(qpos[envs_idx], envs_idx=envs_idx, num_waypoints=300)

    cam.start_recording()
    for waypoint in path:
        franka.control_dofs_position(waypoint, envs_idx=envs_idx)
        scene.step()
        cam.render()

    for i in range(30):
        scene.step()

    # reach
    qpos = franka.inverse_kinematics(
        link=end_effector,
        pos=np.array([[0.65, 0.0, 0.142]]*n_envs),
        quat=np.array([[0, 1, 0, 0]]*n_envs),
    )
    franka.control_dofs_position(qpos[:,:-2], motors_dof)
    for i in range(100):
        scene.step()
        cam.render()

    # grasp
    franka.control_dofs_position(qpos[:,:-2], motors_dof)
    franka.control_dofs_position(np.array([[0, 0]]*n_envs), fingers_dof)  # you can use position control
    for i in range(100):
        scene.step()
        cam.render()

    # lift
    qpos = franka.inverse_kinematics(
        link=end_effector,
        pos=np.array([[0.65, 0.0, 0.3]]*n_envs),
        quat=np.array([[0, 1, 0, 0]]*n_envs),
    )
    franka.control_dofs_position(qpos[:,:-2], motors_dof)
    franka.control_dofs_force(np.array([[-20, -20]]*n_envs), fingers_dof)  # can also use force control
    for i in range(100):
        scene.step()
        cam.render()
        
    cam.stop_recording(save_to_filename="grasp_bottle.mp4")


if __name__ == "__main__":
    main()
