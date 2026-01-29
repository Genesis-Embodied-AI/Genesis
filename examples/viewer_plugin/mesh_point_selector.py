import genesis as gs

if __name__ == "__main__":
    gs.init(backend=gs.gpu)

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            gravity=(0.0, 0.0, 0.0),
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.5, 0.2, 1.0),
            camera_lookat=(0.0, 0.0, 1.0),
            camera_fov=40,
        ),
        profiling_options=gs.options.ProfilingOptions(
            show_FPS=False,
        ),
        show_viewer=True,
    )

    hand = scene.add_entity(
        morph=gs.morphs.URDF(
            file="urdf/shadow_hand/shadow_hand.urdf",
            collision=True,
            pos=(0.0, 0.0, 1.0),
            euler=(0.0, 0.0, 0.0),
            fixed=True,
            merge_fixed_links=False,
        ),
    )

    scene.viewer.add_plugin(
        gs.vis.viewer_plugins.MeshPointSelectorPlugin(
            sphere_radius=0.004,
            grid_snap=(-1.0, 0.01, 0.01),
        )
    )

    scene.build()

    while True:
        scene.step()
