import argparse
import numpy as np
import genesis as gs
import pandas as pd
def make_step(scene, cam, franka, df):
    """フランカを目標位置に移動させるステップ関数"""
    scene.step()
    cam.render()
    dofs = franka.get_dofs_position()
    links_force_torque = franka.get_links_force_torque([9, 10]) # 手先のlocal_indexは9, 10
    links_force_torque = [x.item() for x in links_force_torque[0]] + [x.item() for x in links_force_torque[1]]
    df.loc[len(df)] = [
        scene.t,
        links_force_torque[0], links_force_torque[1], links_force_torque[2],
        links_force_torque[3], links_force_torque[4], links_force_torque[5],
        links_force_torque[6], links_force_torque[7], links_force_torque[8],
        links_force_torque[9], links_force_torque[10], links_force_torque[11],
        dofs[0], dofs[1], dofs[2], dofs[3], dofs[4], dofs[5], dofs[6], dofs[7], dofs[8]
    ]
    # #force

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--video", default="data/videos/grasp_can.mp4")
    parser.add_argument("-o", "--outfile", default="data/csv/grasp_can.csv")
    args = parser.parse_args()
    df = pd.DataFrame(columns=["step", "left_fx", "left_fy", "left_fz", "left_tx", "left_ty", "left_tz",
                           "right_fx", "right_fy", "right_fz", "right_tx", "right_ty", "right_tz",
                           "dof_0", "dof_1", "dof_2", "dof_3", "dof_4", "dof_5", "dof_6", "dof_7", "dof_8"])
    ########################## init ##########################
    # gs.init(backend=gs.cpu, debug=True, logging_level="error")
    gs.init(backend=gs.cpu)
    ########################## create a scene ##########################
    viewer_options = gs.options.ViewerOptions(
        camera_pos=(3, -1, 1.5),
        camera_lookat=(0.0, 0.0, 0.0),
        camera_fov=30,
        max_FPS=60,
    )
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=5e-3,
            substeps=15,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(3, -1, 1.5),
            camera_lookat=(0.0, 0.0, 0.0),
            camera_fov=30,
        ),
        show_viewer=False,
        vis_options=gs.options.VisOptions(
            visualize_mpm_boundary=True,
        ),
        mpm_options=gs.options.MPMOptions(
            lower_bound=(0.0, -0.1, -0.05),
            upper_bound=(0.75, 1.0, 1.0),
            grid_density=128,
        ),
    )
    # ---- 追加: オフスクリーンカメラ ------------------------
    cam = scene.add_camera(
        res=(1280, 720),
        # X 軸方向からのサイドビュー、Z を 0.1（缶の中心高さ程度）にして水平に
        pos=(2.0, 2.0, 0.1),
        lookat=(0.0, 0.0, 0.1),
        fov=30,
    )
    # --------------------------------------------------------
    ########################## entities ##########################
    plane = scene.add_entity(
        gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True),
    )
    chips_can = scene.add_entity(
        # material=gs.materials.Rigid(),
        # material=gs.materials.MPM.Elastic( #Rubber
        #     E=2500,
        #     nu=0.499,
        #     rho=920,
        #     sampler="pbs",
        #     model="neohooken"
        # ),
        material=gs.materials.MPM.ElastoPlastic( #PET
            E=2.45e6,
            nu=0.4,
            rho=1400,
            use_von_mises=True,
            von_mises_yield_stress=18000,
        ),
        # material=gs.materials.MPM.ElastoPlastic( #PP
        #     E=2.0e6,
        #     nu=0.42,
        #     rho=900,
        #     use_von_mises=True,
        #     von_mises_yield_stress=33000,
        # ),
        material=gs.materials.MPM.ElastoPlastic( #aluminum
            E=69e6,
            nu=0.33,
            rho=2700,
            use_von_mises=True,
            von_mises_yield_stress=95000,
        ),
        material=gs.materials.MPM.ElastoPlastic( #steel
            E=180e6,
            nu=0.25,
            rho=7860,
            use_von_mises=True,
            von_mises_yield_stress=502000,
        ),
        morph=gs.morphs.Mesh(
            file="data/objects/002_master_chef_can/poisson/textured.obj",
            scale=0.6, #record
            pos=(0.45, 0.45, 0.0),
            euler=(0, 0, 0), #record
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
    x = 0.45
    y = 0.45
    z = 0.6
    qpos = franka.inverse_kinematics(
        link=end_effector,
        pos=np.array([x, y, z]),
        quat=np.array([0, 1, 0, 0]),
    )
    qpos[-2:] = 0.04
    franka.set_dofs_position(qpos[:-2], motors_dof)
    franka.set_dofs_position(qpos[-2:], fingers_dof)
    cam.start_recording()
    #=================この中を調整========================
    # reach
    for i in range (100):
        franka.set_dofs_position(qpos[:-2], motors_dof)
        franka.set_dofs_position(qpos[-2:], fingers_dof)
        make_step(scene, cam, franka, df)
    for i in range(190):
        #record optimized moments
        z -= 0.0025
        qpos = franka.inverse_kinematics(
            link=end_effector,
            pos=np.array([x, y, z]),
            quat=np.array([0, 1, 0, 0]),
        )
        qpos[-2:] = 0.04
        franka.control_dofs_position(qpos[:-2], motors_dof)
        franka.control_dofs_position(qpos[-2:], fingers_dof)
        make_step(scene, cam, franka, df)
    print("x, y, z: ", x, y, z)
    # grasp
    for i in range(400):
        qpos = franka.inverse_kinematics(
            link=end_effector,
            pos=np.array([x, y, z]),
            quat=np.array([0, 1, 0, 0]),
        )
        franka.control_dofs_position(qpos[:-2], motors_dof)
        franka.control_dofs_force(np.array([-0.01*i, -0.01*i]), fingers_dof)
        make_step(scene, cam, franka, df)
    
    for i in range(200):
        z += 0.0025
        qpos = franka.inverse_kinematics(
            link=end_effector,
            pos=np.array([x, y, z]),
            quat = np.array([0, 1, 0, 0]),
        )
        franka.control_dofs_position(qpos[:-2], motors_dof)
        franka.control_dofs_force(np.array([-5, -5]), fingers_dof)
        make_step(scene, cam, franka, df)
    
    for i in range(100):
        franka.control_dofs_position(qpos[:-2], motors_dof)
        franka.control_dofs_force(np.array([-5+0.05*i, -5+0.05*i]), fingers_dof)
        make_step(scene, cam, franka, df)
    # ---- 追加: 録画終了・保存 -------------------------------
    cam.stop_recording(save_to_filename=args.video, fps=1000)
    print(f"saved -> {args.video}")
    df.to_csv(args.outfile, index=False)
    print(f"saved -> {args.outfile}")
    # --------------------------------------------------------
if __name__ == "__main__":
    main()
