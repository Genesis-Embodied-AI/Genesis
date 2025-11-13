import genesis as gs

gs.init(backend=gs.cpu)

scene = gs.Scene(
    viewer_options=gs.options.ViewerOptions(
        camera_pos=(3.5, 0.0, 2.5),
        camera_lookat=(0.0, 0.0, 0.5),
        camera_fov=40,
    ),
    rigid_options=gs.options.RigidOptions(
        # constraint_solver=gs.constraint_solver.Newton,
        gravity=(0, 0, 0),
    ),
    show_viewer=True
)

# Load the racer drone URDF
drone = scene.add_entity(
    gs.morphs.URDF(
        file='genesis/assets/urdf/drones/racer.urdf',
    ),
)

# Build the scene
scene.build()

# Run the simulation for visualization
while(scene.viewer.is_alive()):
    scene.step()

