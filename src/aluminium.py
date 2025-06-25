import argparse
import numpy as np
import genesis as gs
import pandas as pd
import os
import torch
import imageio.v3 as iio
"""https://pypi.org/project/imageio/"""

def make_step(scene, cam, franka, df, photo_path):
    """フランカを目標位置に移動させるステップ関数"""
    scene.step()
    t = int(scene.t) - 1
    # if t % 20 == 0:
    #     cam.render()
    if t % 500 == 0:
        rgb, _, _, _ = cam.render(rgb=True)

        filepath = photo_path + f"_{t:05d}.png"
        iio.imwrite(filepath, rgb)
    dofs = franka.get_dofs_position()
    dofs = [x.item() for x in dofs]
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

def aluminium(object_name, object_euler, object_scale, grasp_pos, object_path, qpos_init, coup_friction=0.5):
    video_object_path = f"data/videos/{object_name}"
    outfile_object_path = f"data/csv/{object_name}"
    photo_object_path = f"data/photos/{object_name}"
    photo_material_path = f"{photo_object_path}/aluminium"
    if coup_friction == 0.5:
        photo_friction_path = f"{photo_material_path}/050"
    elif coup_friction == 1.0:
        photo_friction_path = f"{photo_material_path}/100"
    elif coup_friction == 5.0:
        photo_friction_path = f"{photo_material_path}/500"
    else:
        raise ValueError("coup_friction must be 0.5, 1.0, or 5.0")
    if not os.path.exists(video_object_path):
        os.makedirs(video_object_path)
    if not os.path.exists(outfile_object_path):
        os.makedirs(outfile_object_path)
    if not os.path.exists(photo_object_path):
        os.makedirs(photo_object_path)
    if not os.path.exists(photo_material_path):
        os.makedirs(photo_material_path)
    if not os.path.exists(photo_friction_path):
        os.makedirs(photo_friction_path)
    default_video_path = f"{video_object_path}/{object_name}_aluminium_{coup_friction}.mp4"
    default_outfile_path = f"{outfile_object_path}/{object_name}_aluminium_{coup_friction}.csv"
    base_photo_name = f"{photo_friction_path}/{object_name}_aluminium_{coup_friction}"
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--video", default=default_video_path)
    parser.add_argument("-o", "--outfile", default=default_outfile_path)
    args = parser.parse_args()
    df = pd.DataFrame(columns=["step", "left_fx", "left_fy", "left_fz", "left_tx", "left_ty", "left_tz",
                           "right_fx", "right_fy", "right_fz", "right_tx", "right_ty", "right_tz",
                           "dof_0", "dof_1", "dof_2", "dof_3", "dof_4", "dof_5", "dof_6", "dof_7", "dof_8"])
    ########################## init ##########################
    # if torch.cuda.is_available():
    #     device = torch.device("cuda")
    #     gs.init(backend=gs.gpu)
    # else:
    #     device = torch.device("cpu")
    #     gs.init(backend=gs.cpu, debug=True)
    device = torch.device("cpu")
    gs.init(backend=gs.cpu, logging_level="debug")
    ########################## create a scene ##########################
    viewer_options = gs.options.ViewerOptions(
        camera_pos=(3, -1, 1.5),
        camera_lookat=(0.0, 0.0, 0.0),
        camera_fov=30,
        max_FPS=60,
    )
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=1e-3,
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
        material=gs.materials.Rigid( #Aluminium
            rho=2700,
            coup_friction=coup_friction,
            friction=coup_friction,
        ),
        morph=gs.morphs.Mesh(
            file=object_path,
            scale=object_scale, #record
            pos=(0.45, 0.45, 0.0),
            euler=object_euler, #record
        ),
    )
    franka = scene.add_entity(
        gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"),
        material=gs.materials.Rigid(coup_friction=coup_friction, friction=coup_friction),
    )

    ########################## build ##########################
    scene.build()
    motors_dof = torch.arange(7, dtype=gs.tc_int, device=device)
    fingers_dof = torch.arange(7, 9, dtype=gs.tc_int, device=device)
    # Optional: set control gains
    franka.set_dofs_kp(
        torch.tensor([4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100], device=gs.device),
    )
    franka.set_dofs_kv(
        torch.tensor([450, 450, 350, 350, 200, 200, 200, 10, 10], device=gs.device),
    )
    franka.set_dofs_force_range(
        torch.tensor([-87, -87, -87, -87, -12, -12, -12, -100, -100], device=gs.device),
        torch.tensor([87, 87, 87, 87, 12, 12, 12, 100, 100], device=gs.device),
    )
    end_effector = franka.get_link("hand")
    # move to pre-grasp pose
    dz = 0.0001
    x = 0.45
    y = 0.45
    z = 0.5
    z_steps = int((z - grasp_pos[2]) // dz)
    dx = (x - grasp_pos[0]) / z_steps
    dy = (y - grasp_pos[1]) / z_steps
    print("x,y,z: ", x, y, z)
    qpos = qpos_init.clone().to(gs.device)
    print("qpos: ", qpos)
    franka.set_dofs_position(qpos[:-2], motors_dof)
    franka.set_dofs_position(qpos[-2:], fingers_dof)
    cam.start_recording()
    #=================この中を調整========================
    # 事前にテンソルを１回だけ作成（ループ内で再利用）
    pos_tensor = torch.empty(3, dtype=gs.tc_float, device=gs.device)
    quat_tensor = torch.tensor([0, 1, 0, 0], dtype=gs.tc_float, device=gs.device)
    force_tensor = torch.empty(2, dtype=gs.tc_float, device=gs.device)
    finger_pos_tensor = torch.empty(2, dtype=gs.tc_float, device=gs.device)

    # reach
    for i in range(z_steps):
        x -= dx; y -= dy; z -= dz
        pos_tensor[0], pos_tensor[1], pos_tensor[2] = x, y, z
        qpos = franka.inverse_kinematics(
            link=end_effector,
            pos=pos_tensor,
            quat=quat_tensor,
        )
        qpos[-2:] = 0.04
        franka.control_dofs_position(qpos[:-2], motors_dof)
        franka.control_dofs_position(qpos[-2:], fingers_dof)
        make_step(scene, cam, franka, df, base_photo_name)

    # grasp
    for i in range(300):
        # pos, quat は変わらないので再利用
        qpos = franka.inverse_kinematics(
            link=end_effector,
            pos=pos_tensor,
            quat=quat_tensor,
        )
        force_tensor.fill_(-0.1 * i)
        franka.control_dofs_position(qpos[:-2], motors_dof)
        franka.control_dofs_force(force_tensor, fingers_dof)
        make_step(scene, cam, franka, df, base_photo_name)

    # lift up
    for i in range(z_steps):
        z += dz
        pos_tensor[2] = z
        qpos = franka.inverse_kinematics(
            link=end_effector,
            pos=pos_tensor,
            quat=quat_tensor,
        )
        force_tensor.fill_(-30)
        franka.control_dofs_position(qpos[:-2], motors_dof)
        franka.control_dofs_force(force_tensor, fingers_dof)
        make_step(scene, cam, franka, df, base_photo_name)

    # 小幅往復
    for i in range(3000):
        qpos[-4] -= 0.0002
        franka.control_dofs_position(qpos[:-2], motors_dof)
        franka.control_dofs_force(force_tensor, fingers_dof)
        make_step(scene, cam, franka, df, base_photo_name)

    for i in range(3000):
        qpos[-4] += 0.0002
        franka.control_dofs_position(qpos[:-2], motors_dof)
        franka.control_dofs_force(force_tensor, fingers_dof)
        make_step(scene, cam, franka, df, base_photo_name)

    for i in range(3000):
        qpos[0] += 0.0002
        franka.control_dofs_position(qpos[:-2], motors_dof)
        franka.control_dofs_force(force_tensor, fingers_dof)
        make_step(scene, cam, franka, df, base_photo_name)

    for i in range(1000):
        force_tensor.fill_(-30 + 0.03 * i)
        franka.control_dofs_position(qpos[:-2], motors_dof)
        franka.control_dofs_force(force_tensor, fingers_dof)
        make_step(scene, cam, franka, df, base_photo_name)

    # フィンガー開閉
    for i in range(400):
        finger_pos_tensor.fill_(0.0001 * i)
        franka.control_dofs_position(qpos[:-2], motors_dof)
        franka.control_dofs_position(finger_pos_tensor, fingers_dof)
        make_step(scene, cam, franka, df, base_photo_name)
    

    # ---- 追加: 録画終了・保存 -------------------------------
    cam.stop_recording(save_to_filename=args.video, fps=1000/250)
    print(f"saved -> {args.video}")
    df.to_csv(args.outfile, index=False)
    print(f"saved -> {args.outfile}")
    gs.destroy()
    # --------------------------------------------------------