import genesis as gs

gs.init(backend=gs.cpu)

scene = gs.Scene(
    viewer_options=gs.options.ViewerOptions(
        camera_pos=(3.5, 0.0, 2.5),
        camera_lookat=(0.0, 0.0, 0.5),
        camera_fov=40,
        enable_interaction=True,
    ),
    rigid_options=gs.options.RigidOptions(
        # constraint_solver=gs.constraint_solver.Newton,
        gravity=(0, 0, -9.8),
        enable_collision=True,
        enable_joint_limit=True,
        max_collision_pairs=1000,
    ),
    show_viewer=True,
)

AssetRoot = "D:/Assets"

# load a stage from USD file
entities = scene.add_stage(f"{AssetRoot}/Lightwheel_Kitchen001/Kitchen001/Kitchen001.usd")

# Build the scene
scene.build()

# Run the simulation for visualization
while scene.viewer.is_alive():
    scene.step()
    pass
