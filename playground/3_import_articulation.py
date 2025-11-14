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
        gravity=(0, -1, 0),
        enable_collision = False
    ),
    show_viewer=True,
)

# Load the racer drone URDF
entity:RigidEntity= scene.add_entity(
    gs.morphs.USDArticulation(
        file="D:\\MyStorage\\Project\\GenesisProject\\Genesis\\playground\\assets\\input_mesh.usda",
    ),
)


# Build the scene
scene.build()

# Run the simulation for visualization
while(scene.viewer.is_alive()):
    # entity.set_dofs_position([scene.t/10], dofs_idx_local=0)
    scene.step()