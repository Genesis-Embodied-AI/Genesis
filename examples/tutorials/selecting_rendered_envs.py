import genesis as gs

gs.init()

scene = gs.Scene(
    show_viewer=True,
    viewer_options=gs.options.ViewerOptions(
        res=(1280, 960),
        camera_pos=(3.5, 0.0, 2.5),
        camera_lookat=(0.0, 0.0, 0.5),
        camera_fov=50,
    ),
    vis_options=gs.options.VisOptions(
        rendered_envs_idx=list(range(10, 15)),  # render the 11th to 15th environments
        # rendered_envs_idx=list(range(5)), # render the first 5 environments
    ),
)

plane = scene.add_entity(
    gs.morphs.Plane(),
)
franka = scene.add_entity(
    gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"),
)


scene.build(n_envs=20, env_spacing=(1.0, 1.0))

for i in range(1000):
    scene.step()
