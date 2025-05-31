import numpy as np
import genesis as gs

def make_step(scene, cam, franka):
    """フランカを目標位置に移動させるステップ関数"""
    scene.step()
    cam.render()
    scene.clear_debug_objects()
    links_force_torque = franka.get_links_force_torque([9, 10]) # 手先のlocal_indexは9, 10
    #force
    # scene.draw_debug_arrow(
    #     pos=franka.get_link("left_finger").get_pos().tolist(),
    #     vec=links_force_torque[0][:3].tolist(),
    #     color=(1, 0, 0),
    # )
    # scene.draw_debug_arrow(
    #     pos=franka.get_link("right_finger").get_pos().tolist(),
    #     vec=links_force_torque[1][:3].tolist(),
    #     color=(1, 0, 0),
    # )
    #torque
    scene.draw_debug_arrow(
        pos=franka.get_link("left_finger").get_pos().tolist(),
        vec=links_force_torque[0][3:].tolist(),
        color=(1, 0, 0),
    )
    scene.draw_debug_arrow(
        pos=franka.get_link("right_finger").get_pos().tolist(),
        vec=links_force_torque[1][3:].tolist(),
        color=(1, 0, 0),
    )

# ───────── 初期化 ─────────
gs.init(backend=gs.gpu)               # GPU / Vulkan

# ───────── Scene ─────────
scene = gs.Scene(
    show_viewer=False,                # ← GUI を開かない
    viewer_options=gs.options.ViewerOptions(
        camera_pos    =(3, -1, 1.5),
        camera_lookat =(0.0, 0.0, 0.5),
        camera_fov    =30,
        max_FPS       =60,
    ),
    sim_options = gs.options.SimOptions(
        dt=0.01,
        substeps=4,
    ),
)

# オフスクリーンカメラ（GUI=False）
cam = scene.add_camera(
    res   =(1280, 720),
    pos   =(3, -1, 1.5),
    lookat=(0.0, 0.0, 0.5),
    fov   =30,
    GUI   =False,
)

# ───────── Entities ─────────
plane  = scene.add_entity(gs.morphs.Plane())
cube   = scene.add_entity(gs.morphs.Box(size=(0.04,0.04,0.04), pos=(0.65,0.0,0.02)))
franka = scene.add_entity(gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"))

# ───────── Build ─────────
scene.build()

motors_dof  = np.arange(7)
fingers_dof = np.arange(7, 9)

franka.set_dofs_kp(np.array([4500,4500,3500,3500,2000,2000,2000,100,100]))
franka.set_dofs_kv(np.array([ 450, 450, 350, 350, 200, 200, 200, 10, 10]))
franka.set_dofs_force_range(
    np.array([-87,-87,-87,-87,-12,-12,-12,-100,-100]),
    np.array([ 87, 87, 87, 87, 12, 12, 12, 100, 100]),
)

end_effector = franka.get_link("hand")
# print("end_effector:", end_effector)
# left_finger = franka.get_link("left_finger")
# right_finger = franka.get_link("right_finger")
# print("left_finger:", left_finger)
# print("right_finger:", right_finger)
# ── IK → パス生成 ─────────────────
q_goal = franka.inverse_kinematics(
    link=end_effector,
    pos =np.array([0.65,0.0,0.25]),
    quat=np.array([0,1,0,0]),
)
q_goal[-2:] = 0.04                    # 指少し開く
path = franka.plan_path(q_goal, num_waypoints=200)

# ───────── 録画開始 ─────────
cam.start_recording()

# ── 1. 物体上方へ移動 ────────────
for wp in path:
    franka.control_dofs_position(wp)
    make_step(scene, cam, franka)

# ── 2. 下降して掴む ───────────────
q_pick = franka.inverse_kinematics(
    link=end_effector,
    pos =np.array([0.65,0.0,0.135]),
    quat=np.array([0,1,0,0]),
)
for _ in range(100):
    franka.control_dofs_position(q_pick[:-2], motors_dof)
    make_step(scene, cam, franka)
# ── 3. 指を閉じて把持 ─────────────
for _ in range(100):
    franka.control_dofs_force(np.array([-0.5,-0.5]), fingers_dof)
    franka.control_dofs_position(q_pick[:-2], motors_dof)
    make_step(scene, cam, franka)
# ── 4. リフトアップ ──────────────
q_lift = franka.inverse_kinematics(
    link=end_effector,
    pos =np.array([0.65,0.0,0.30]),
    quat=np.array([0,1,0,0]),
)
for _ in range(200):
    franka.control_dofs_position(q_lift[:-2], motors_dof)
    make_step(scene, cam, franka)
# ───────── 録画終了・保存 ───────
cam.stop_recording(save_to_filename="pickup.mp4", fps=60)
print("saved -> pickup.mp4")