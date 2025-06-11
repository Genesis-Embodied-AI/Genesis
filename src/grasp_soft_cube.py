import argparse

import numpy as np

import genesis as gs


def make_step(scene, cam, franka):
    """フランカを目標位置に移動させるステップ関数"""
    scene.step()
    cam.render()
    scene.clear_debug_objects()
    links_force_torque = franka.get_links_force_torque([9, 10]) # 手先のlocal_indexは9, 10
    print(links_force_torque)
    #force
    scale = 0.1
    scene.draw_debug_arrow(
        pos=franka.get_link("left_finger").get_pos().tolist(),
        vec=(links_force_torque[0][:3]*scale).tolist(),
        color=(1, 0, 0),
    )
    scene.draw_debug_arrow(
        pos=franka.get_link("right_finger").get_pos().tolist(),
        vec=(links_force_torque[1][:3]*scale).tolist(),
        color=(1, 0, 0),
    )
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", default=False)
    parser.add_argument("-o", "--outfile", default="grasp_soft_cube.mp4")
    args = parser.parse_args()

    ########################## init ##########################
    gs.init(backend=gs.gpu)

    ########################## create a scene ##########################

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=5e-3,
            substeps=15,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(3, -1, 1.5),
            camera_lookat=(0.0, 0.0, 0.0),
            camera_fov=30,
            max_FPS=60,
        ),
        show_viewer=args.vis,
        vis_options=gs.options.VisOptions(
            visualize_mpm_boundary=True,
        ),
        mpm_options=gs.options.MPMOptions(
            lower_bound=(0.55, -0.1, -0.05),
            upper_bound=(0.75, 0.1, 0.3),
            grid_density=128,
        ),
    )

    cam = scene.add_camera(
        res=(1280, 720),
        pos=(3, -1, 1.5),
        lookat=(0.0, 0.0, 0.0),
        fov=30,
        GUI=False,
    )

    ########################## entities ##########################
    plane = scene.add_entity(
        gs.morphs.Plane(),
    )
    cube = scene.add_entity(
        material=gs.materials.MPM.Elastic(),
        morph=gs.morphs.Box(
            size=(0.04, 0.04, 0.04),
            pos=(0.65, 0.0, 0.025),
            euler=(0, 0, 0),
        ),
        surface=gs.surfaces.Default(
            vis_mode="particle",
        ),
    )
    franka = scene.add_entity(
        gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"),
        material=gs.materials.Rigid(coup_friction=1.0),
    )

    ########################## build ##########################
    scene.build()

    motors_dof = np.arange(7)
    fingers_dof = np.arange(7, 9)

    # Optional: set control gains
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

    end_effector = franka.get_link("hand")
    # move to pre-grasp pose
    qpos = franka.inverse_kinematics(
        link=end_effector,
        pos=np.array([0.64, 0.0, 0.135]),
        quat=np.array([0, 1, 0, 0]),
    )
    qpos[-2:] = 0.03
    franka.set_dofs_position(qpos[:-2], motors_dof)
    franka.set_dofs_position(qpos[-2:], fingers_dof)

    # grasp with 1N force
    franka.control_dofs_position(qpos[:-2], motors_dof)
    franka.control_dofs_force(np.array([-1, -1]), fingers_dof)
    # franka.control_dofs_position(np.array([0, 0]), fingers_dof) # you can also use position control

    cam.start_recording()
    for _ in range(100):
        make_step(scene, cam, franka)

    # lift
    for i in range(300):
        qpos = franka.inverse_kinematics(
            link=end_effector,
            pos=np.array([0.64, 0.0, 0.135 + 0.0005 * i]),
            quat=np.array([0, 1, 0, 0]),
        )
        franka.control_dofs_position(qpos[:-2], motors_dof)
        make_step(scene, cam, franka)

    #add power of fingers
    for i in range(100):
        franka.control_dofs_force(np.array([-0.1*i-1, -0.1*i-1]), fingers_dof)
        make_step(scene, cam, franka)
    cam.stop_recording(save_to_filename=args.outfile, fps=60)
    print(f"saved -> {args.outfile}")
if __name__ == "__main__":
    main()
