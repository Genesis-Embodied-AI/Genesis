import mujoco
import mujoco_viewer
import numpy as np

model = mujoco.MjModel.from_xml_path("genesis/assets/xml/four_bar_linkage.xml")
data = mujoco.MjData(model)
# model.opt.gravity[2] = -9.81
model.opt.timestep = 0.01
# create the viewer object
viewer = mujoco_viewer.MujocoViewer(model, data)
data.qpos = [0.1, 0.1, 0.1]
# simulate and render
for i in range(100000):
    # data.ctrl[0] = 100000 * np.sin(0.01 * i)  # Oscillate hinge_1
    # data.ctrl[1] = 100000 * np.cos(0.01 * i)  # Oscillate hinge_2
    # data.ctrl[2] = 5
    print("i-----------------", i)
    mujoco.mj_step(model, data)
    for j in range(3):
        print(data.efc_pos[j], data.efc_aref[j], data.efc_vel[j], data.efc_D[j])
    print(data.efc_J)
    viewer.render()
    if i >= 0:
        from IPython import embed; embed()

# close
viewer.close()
