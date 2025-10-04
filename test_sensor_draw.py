import genesis as gs


def test_sensors_draw_debug(png_snapshot, n_envs):
    """Test that sensor debug drawing works correctly and renders visible debug elements."""
    CAM_RES = (640, 480)

    scene = gs.Scene(
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(2.0, 2.0, 2.0),
            camera_lookat=(0.0, 0.0, 0.2),
            camera_fov=30,
            res=CAM_RES,
        ),
        profiling_options=gs.options.ProfilingOptions(show_FPS=False),
        show_viewer=True,
    )

    scene.add_entity(gs.morphs.Plane())

    floating_box = scene.add_entity(
        gs.morphs.Box(
            size=(0.1, 0.1, 0.1),
            pos=(0.0, 0.0, 0.5),
            fixed=True,
        )
    )
    scene.add_sensor(
        gs.sensors.IMU(
            entity_idx=floating_box.idx,
            pos_offset=(0.0, 0.0, 0.1),
            draw_debug=True,
        )
    )

    ground_box = scene.add_entity(
        gs.morphs.Box(
            size=(0.4, 0.2, 0.1),
            pos=(-0.25, 0.0, 0.05),
        )
    )
    scene.add_sensor(
        gs.sensors.Contact(
            entity_idx=ground_box.idx,
            draw_debug=True,
            debug_sphere_radius=0.08,
        )
    )
    scene.add_sensor(
        gs.sensors.ContactForce(
            entity_idx=ground_box.idx,
            draw_debug=True,
            debug_scale=0.01,
        )
    )
    scene.add_sensor(
        gs.sensors.Raycaster(
            pattern=gs.sensors.raycaster.GridPattern(
                resolution=0.2,
                size=(0.4, 0.4),
                direction=(0.0, 0.0, -1.0),
            ),
            entity_idx=floating_box.idx,
            pos_offset=(0.2, 0.0, -0.1),
            return_world_frame=True,
            draw_debug=True,
        )
    )
    scene.add_sensor(
        gs.sensors.Raycaster(
            pattern=gs.sensors.raycaster.SphericalPattern(
                n_points=(6, 6),
                fov=(60.0, (-120.0, -60.0)),
            ),
            entity_idx=floating_box.idx,
            pos_offset=(0.0, 0.5, 0.0),
            return_world_frame=False,
            draw_debug=True,
            debug_sphere_radius=0.01,
            debug_ray_start_color=(1.0, 1.0, 0.5, 1.0),
            debug_ray_hit_color=(0.5, 1.0, 1.0, 1.0),
        )
    )

    scene.build(n_envs=n_envs)

    for _ in range(5):
        scene.step()


if __name__ == "__main__":
    gs.init(backend=gs.cpu)
    # test_sensors_draw_debug(None, 0)
    test_sensors_draw_debug(None, 2)
