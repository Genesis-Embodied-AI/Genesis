from huggingface_hub import snapshot_download

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
            viewer_plugin=gs.options.viewer_plugins.MeshPointSelectorPlugin(
                sphere_radius=0.004,
                grid_snap=(-1.0, 0.01, 0.01),
            ),
        ),
        profiling_options=gs.options.ProfilingOptions(
            show_FPS=False,
        ),
        show_viewer=True,
    )

    asset_path = snapshot_download(
        repo_id="Genesis-Intelligence/assets",
        allow_patterns="allegro_hand/*",
        repo_type="dataset",
    )
    hand = scene.add_entity(
        morph=gs.morphs.URDF(
            file=f"{asset_path}/allegro_hand/allegro_hand_right_glb.urdf",
            collision=True,
            pos=(0.0, 0.0, 1.0),
            euler=(0.0, 0.0, 0.0),
            fixed=True,
            merge_fixed_links=False,
        ),
    )

    scene.build()

    while True:
        scene.step()
