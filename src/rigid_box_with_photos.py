import os
import sys
import genesis as gs
import threading, time

import numpy as np

import imageio.v3 as iio
"""https://pypi.org/project/imageio/"""


gs.init(backend=gs.gpu)

scene = gs.Scene(
    show_viewer=True,
    viewer_options=gs.options.ViewerOptions(
        res=(1280, 960),
        camera_pos=(3.5, 1.0, 1.5),
        camera_lookat=(0.0, 0.0, 0.5),
        camera_fov=40,
        max_FPS=100,
    ),
    vis_options=gs.options.VisOptions(
        show_world_frame=True,
        world_frame_size=1.0,
        show_link_frame=False,
        show_cameras=False,
        plane_reflection=True,
        ambient_light=(0.1, 0.1, 0.1),
    ),
    rigid_options=gs.options.RigidOptions(
        box_box_detection=True,
    ),
    # renderer=gs.renderers.RayTracer(),
    renderer=gs.renderers.Rasterizer(),
)


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

cam = scene.add_camera(
    res=(1280, 960),
    pos    = (3.5, 0.0, 2.5),
    lookat = (0, 0, 0.5),
    fov    = 40,
    GUI    = False,
)

########################## build ##########################
scene.build()

if sys.platform == "darwin":
    scene._visualizer._viewer._pyrender_viewer._renderer.dpscale = 1

def sim_loop():
    rgb, _, _, _ = cam.render(rgb=True)

    filename = "frame_init.png"
    filepath = os.path.join("photos_while_running", filename)

    iio.imwrite(filepath, rgb)


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


    """Render camera rgb values and unpack"""
    rgb, _, _, _ = cam.render(rgb=True)
    filename = "frame_starting.png"
    filepath = os.path.join("photos_while_running", filename)

    """Write to file"""
    iio.imwrite(filepath, rgb)

    # grasp
    finder_pos = -0.0
    for i in range(100):
        #print("grasp", i)
        franka.control_dofs_position(qpos[:-2], motors_dof)
        franka.control_dofs_position(np.array([finder_pos, finder_pos]), fingers_dof)
        scene.step()
        "Save image at the end of sequence"
        if i == 99:
            rgb, _, _, _ = cam.render(rgb=True)
            filename = f"frame_grasp.png"
            filepath = os.path.join("photos_while_running", filename)
            iio.imwrite(filepath, rgb)


    # lift
    qpos = franka.inverse_kinematics(
        link=end_effector,
        pos=np.array([0.65, 0.0, 0.3]),
        quat=np.array([0, 1, 0, 0]),
    )
    for i in range(200):
        #print("lift", i)
        franka.control_dofs_position(qpos[:-2], motors_dof)
        franka.control_dofs_position(np.array([finder_pos, finder_pos]), fingers_dof)
        scene.step()
        "Save image at the end of sequence"
        if i == 199:
            rgb, _, _, _ = cam.render(rgb=True)
            filename = "frame_lift.png"
            filepath = os.path.join("photos_while_running", filename)

            iio.imwrite(filepath, rgb)



threading.Thread(target=sim_loop, daemon=True).start()
scene.viewer.start()
