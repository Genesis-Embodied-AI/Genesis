import genesis as gs
gs.init(backend=gs.cpu)
from genesis.engine.entities import RigidEntity

scene = gs.Scene(
    viewer_options=gs.options.ViewerOptions(
        camera_pos=(3.5, 0.0, 2.5),
        camera_lookat=(0.0, 0.0, 0.5),
        camera_fov=40,
    ),
    rigid_options=gs.options.RigidOptions(
        # constraint_solver=gs.constraint_solver.Newton,
        gravity=(0, 0, -1),
        enable_collision = False,
        enable_joint_limit = True,
    ),
    show_viewer=True,
)

# load an articulation from USD file
entity:RigidEntity= scene.add_entity(
    gs.morphs.USDArticulation(
        file="D:\\Assets\\Fixed\\G1.usd"
    ),
)

# Build the scene
scene.build()

# Run the simulation for visualization
while(scene.viewer.is_alive()):
    # scene.step()
    pass