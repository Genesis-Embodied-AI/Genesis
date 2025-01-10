import numpy as np
import time
import genesis as gs

########################## init ##########################
gs.init(backend=gs.gpu, precision="32")
########################## create a scene ##########################
scene = gs.Scene(
    viewer_options=gs.options.ViewerOptions(
        camera_pos=(3, -1, 1.5),
        camera_lookat=(0.0, 0.0, 0.5),
        camera_fov=30,
        res=(960, 640),
        max_FPS=60,
    ),
    sim_options=gs.options.SimOptions(
        dt=0.01,
    ),
    rigid_options=gs.options.RigidOptions(
        box_box_detection=True,
    ),
    show_viewer=True,
)

########################## entities ##########################
plane = scene.add_entity(
    gs.morphs.Plane(),
)
franka = scene.add_entity(
    gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"),
)

cube = scene.add_entity(
    gs.morphs.Box(
        size=(0.04, 0.04, 0.04),
        pos=(0.65, 0.0, 0.02),
    )
)
########################## build ##########################
scene.build()

motors_dof = np.arange(7)
fingers_dof = np.arange(7, 9)
qpos = np.array([-1.0124, 1.5559, 1.3662, -1.6878, -1.5799, 1.7757, 1.4602, 0.04, 0.04])
franka.set_qpos(qpos)
scene.step()

end_effector = franka.get_link("hand")
qpos = franka.inverse_kinematics(
    link=end_effector,
    pos=np.array([0.65, 0.0, 0.135]),
    quat=np.array([0, 1, 0, 0]),
)

franka.control_dofs_position(qpos[:-2], motors_dof)

# hold
for i in range(100):
    print("hold", i)
    scene.step()

# grasp
finder_pos = -0.0
for i in range(100):
    print("grasp", i)
    franka.control_dofs_position(qpos[:-2], motors_dof)
    franka.control_dofs_position(np.array([finder_pos, finder_pos]), fingers_dof)
    scene.step()

# lift
qpos = franka.inverse_kinematics(
    link=end_effector,
    pos=np.array([0.65, 0.0, 0.3]),
    quat=np.array([0, 1, 0, 0]),
)
for i in range(200):
    print("lift", i)
    franka.control_dofs_position(qpos[:-2], motors_dof)
    franka.control_dofs_position(np.array([finder_pos, finder_pos]), fingers_dof)
    scene.step()
