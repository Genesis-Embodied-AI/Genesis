import numpy as np
import genesis as gs


########################## init ##########################
gs.init(precision="32", logging_level="info")

######################## create a scene ##########################
scene = gs.Scene(
    sim_options=gs.options.SimOptions(
        dt=3e-3,
        substeps=10,
    ),
    rigid_options=gs.options.RigidOptions(
        gravity=(0, 0, -9.8),
        enable_collision=True,
        enable_self_collision=False,
        enable_adjacent_collision=False,
        # Prevent the rigid contact solver from being too stiff otherwise this would cause large impulse, especially
        # because the simulation timestep must be very small to ensure numerical stability of rigid body dynamics.
        constraint_timeconst=0.02,
    ),
    mpm_options=gs.options.MPMOptions(
        lower_bound=(0.0, 0.0, -0.2),
        upper_bound=(1.0, 1.0, 1.0),
        gravity=(0, 0, 0),  # mimic gravity compensation
        enable_CPIC=True,
    ),
    viewer_options=gs.options.ViewerOptions(
        camera_pos=(1.5, 1.3, 0.5),
        camera_lookat=(0.0, 0.0, 0.0),
        camera_fov=40,
    ),
    vis_options=gs.options.VisOptions(
        show_world_frame=True,
        visualize_mpm_boundary=False,
    ),
    show_viewer=True,
    show_FPS=False,
)

########################## entities ##########################
scene.add_entity(morph=gs.morphs.Plane())

robot = scene.add_entity(
    morph=gs.morphs.URDF(
        file="urdf/simple/two_link_arm.urdf",
        pos=(0.5, 0.5, 0.3),
        euler=(0.0, 0.0, 0.0),
        scale=0.2,
        fixed=True,
    ),
    material=gs.materials.Hybrid(
        material_rigid=gs.materials.Rigid(
            gravity_compensation=1.0,
        ),
        material_soft=gs.materials.MPM.Muscle(  # to allow setting group
            E=1e4,
            nu=0.45,
            rho=1000.0,
            model="neohooken",
        ),
        thickness=0.05,
        damping=1000.0,
    ),
)

ball = scene.add_entity(
    morph=gs.morphs.Sphere(
        pos=(0.8, 0.6, 0.1),
        radius=0.1,
    ),
    material=gs.materials.Rigid(rho=1000, friction=0.5),
)

########################## build ##########################
scene.build()

########################## run ##########################
scene.reset()
for i in range(1000):
    dofs_ctrl = np.array(
        [
            1.0 * np.sin(2 * np.pi * i * 0.001),
        ]
        * robot.n_dofs
    )

    robot.control_dofs_velocity(dofs_ctrl)

    scene.step()
