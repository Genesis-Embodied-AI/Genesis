import os
import argparse

import numpy as np
from tqdm import tqdm

import genesis as gs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--solver", choices=["explicit", "implicit"], default="implicit", help="FEM solver type (default: explicit)"
    )
    parser.add_argument("--dt", type=float, help="Time step (auto-selected based on solver if not specified)")
    parser.add_argument(
        "--substeps", type=int, help="Number of substeps (auto-selected based on solver if not specified)"
    )
    parser.add_argument("--vis", "-v", action="store_true", help="Show visualization GUI")
    parser.add_argument("-c", "--cpu", action="store_true", default=True)
    args = parser.parse_args()

    steps = int(1.0 / dt if "PYTEST_VERSION" not in os.environ else 5)

    gs.init(backend=gs.cpu if args.cpu else gs.gpu, logging_level=None)

    if args.solver == "explicit":
        dt = args.dt if args.dt is not None else 1e-4
        substeps = args.substeps if args.substeps is not None else 5
    else:  # implicit
        dt = args.dt if args.dt is not None else 1e-3
        substeps = args.substeps if args.substeps is not None else 1

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=dt,
            substeps=substeps,
            gravity=(0, 0, -9.81),
        ),
        fem_options=gs.options.FEMOptions(
            use_implicit_solver=args.solver == "implicit",
            enable_vertex_constraints=True,
        ),
        profiling_options=gs.options.ProfilingOptions(
            show_FPS=False,
        ),
        show_viewer=args.vis,
    )

    # Setup scene entities
    scene.add_entity(gs.morphs.Plane())

    cube = scene.add_entity(
        morph=gs.morphs.Box(
            pos=(0.5, 0.0, 0.05),
            size=(0.2, 0.2, 0.2),
        ),
        material=gs.materials.FEM.Elastic(
            E=1.0e4,  # stiffness
            nu=0.45,  # compressibility (0 to 0.5)
            rho=1000.0,  # density
            model="linear_corotated",
        ),
    )
    arm = scene.add_entity(
        morph=gs.morphs.MJCF(
            file="xml/franka_emika_panda/panda.xml",
            pos=(0, 0, 0),
        ),
    )

    # Setup camera for recording
    video_fps = 1 / dt
    max_fps = 100
    frame_interval = max(1, int(video_fps / max_fps)) if max_fps > 0 else 1
    print("video_fps:", video_fps, "frame_interval:", frame_interval)
    cam = scene.add_camera(
        res=(640, 480),
        pos=(-2.0, 3.0, 2.0),
        lookat=(0.5, 0.5, 0.5),
        fov=30,
    )

    scene.build()
    cam.start_recording()

    try:
        joint_names = [j.name for j in arm.joints]
        dofs_idx_local = []
        for j in arm.joints:
            # print("joint name:", j.name, "dofs_idx_local:", j.dofs_idx_local)
            dofs_idx_local += j.dofs_idx_local
        end_joint = arm.get_joint(joint_names[-1])

        arm.set_dofs_kp(
            np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100]),
        )
        arm.set_dofs_kv(
            np.array([450, 450, 350, 350, 200, 200, 200, 10, 10]),
        )
        arm.set_dofs_force_range(
            np.array([-87, -87, -87, -87, -12, -12, -12, -100, -100]),
            np.array([87, 87, 87, 87, 12, 12, 12, 100, 100]),
        )

        for i in range(100):
            arm.set_dofs_position(
                np.array([0.9643, -0.3213, -0.6685, -2.3139, -0.2890, 2.0335, -1.6014, 0.0306, 0.0306]), dofs_idx_local
            )
            scene.step()
            if i % frame_interval == 0:
                cam.render()

        print("cube init pos", cube.init_positions)
        pin_idx = [1, 5]
        cube.set_vertex_constraints(
            verts_idx=pin_idx,
            link=end_joint.link,
        )
        print("Cube initial positions:", cube.init_positions[pin_idx])
        scene.draw_debug_spheres(poss=cube.init_positions[pin_idx], radius=0.02, color=(1.0, 0.0, 1.0, 0.8))

        arm_target_pos = (0.3, 0.2, 0.8)
        scene.draw_debug_spheres(poss=[arm_target_pos], radius=0.02, color=(0.0, 1.0, 0.0, 0.8))
        qpos = arm.inverse_kinematics(
            link=end_joint.link,
            pos=np.array(arm_target_pos, gs.np_float),
            quat=np.array((0.0, 1.0, 0.0, 0.0), gs.np_float),
        )
        arm_path_waypoints = arm.plan_path(qpos_goal=qpos, num_waypoints=steps)

        for i, waypoint in tqdm(enumerate(arm_path_waypoints), total=len(arm_path_waypoints)):
            arm.control_dofs_position(waypoint)
            scene.step()
            if i % frame_interval == 0:
                cam.render()

        print("Now dropping the cube")
        cube.remove_vertex_constraints()
        for i in tqdm(range(steps), total=steps):
            arm.control_dofs_position(arm_path_waypoints[-1])
            scene.step()
            if i % frame_interval == 0:
                cam.render()

    except KeyboardInterrupt:
        gs.logger.info("Simulation interrupted, exiting.")
    finally:
        gs.logger.info("Simulation finished.")

        actual_fps = video_fps / frame_interval
        video_filename = f"cube_link_arm_{args.solver}_dt={dt}_substeps={substeps}.mp4"
        cam.stop_recording(save_to_filename=video_filename, fps=actual_fps)
        gs.logger.info(f"Saved video to {video_filename}")


if __name__ == "__main__":
    main()
