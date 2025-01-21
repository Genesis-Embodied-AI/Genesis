import mujoco
import mujoco_viewer
import numpy as np

model = mujoco.MjModel.from_xml_path("genesis/assets/xml/four_bar_linkage.xml")
data = mujoco.MjData(model)

# create the viewer object
viewer = mujoco_viewer.MujocoViewer(model, data)

# simulate and render
for i in range(100000):
    data.ctrl[0] = 100000 * np.sin(0.01 * i)  # Oscillate hinge_1
    data.ctrl[1] = 100000 * np.cos(0.01 * i)  # Oscillate hinge_2
    # data.ctrl[2] = 5
    mujoco.mj_step(model, data)
    viewer.render()

# close
viewer.close()
