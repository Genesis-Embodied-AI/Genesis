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
        gravity=(0, 0, -1),
        enable_collision = True,
        enable_joint_limit = True,
    ),
    show_viewer=True,
)

# load a stage from USD file
entities = scene.add_stage("D:\\Assets\\Lightwheel_Kitchen001\\Kitchen001\\Kitchen001.usd")
# entities = scene.add_stage("d:\\Assets\\Fixed\\G1.usd")
# Build the scene
scene.build()

# Run the simulation for visualization
while(scene.viewer.is_alive()):
    scene.step()
    pass