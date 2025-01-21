import genesis as gs

gs.init(backend=gs.cpu)

scene = gs.Scene(
    viewer_options=gs.options.ViewerOptions(
        camera_pos=(10, 0, 10),
        camera_lookat=(0.0, 0.0, 3),
        camera_fov=60,
    ),
)

# plane = scene.add_entity(
#     gs.morphs.Plane(),
# )
franka = scene.add_entity(
    gs.morphs.MJCF(
        file="xml/four_bar_linkage.xml",
        # pos=(0, 0, 0.5),
        # scale=0.05
    ),
)

scene.build()
for i in range(10000):
    scene.step()
    # import ipdb; ipdb.set_trace()
