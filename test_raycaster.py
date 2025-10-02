import numpy as np

import genesis as gs
from tests.utils import assert_allclose


def expand_batch_dim(values: tuple[float, ...], n_envs: int) -> tuple[float, ...] | np.ndarray:
    """Helper function to expand expected values for n_envs dimension."""
    if n_envs == 0:
        return values
    return np.tile(np.array(values), (n_envs,) + (1,) * len(values))


def test_raycaster_hits(show_viewer, tol, n_envs, only_cast_fixed):
    """Test if the Raycaster sensor with GridPattern rays pointing to ground returns the correct distance."""
    EXPECTED_DISTANCE = 0.7
    NUM_RAYS_XY = 3
    BOX_HEIGHT = 0.2
    SPHERE_POS = (4.0, 0.0, 1.0)
    RAYCAST_GRID_SIZE = 0.5

    scene = gs.Scene(
        profiling_options=gs.options.ProfilingOptions(show_FPS=False),
        show_viewer=show_viewer,
    )

    scene.add_entity(
        gs.morphs.Plane(
            is_free=False,  # TODO: remove after PR #1795 is merged
        )
    )

    box_obstacle = scene.add_entity(
        gs.morphs.Box(
            size=(RAYCAST_GRID_SIZE / 2.0, RAYCAST_GRID_SIZE / 2.0, BOX_HEIGHT),
            # pos=(0.0, 0.0, -BOX_HEIGHT),  # init below ground to not interfere with first raycast
            pos=(RAYCAST_GRID_SIZE, RAYCAST_GRID_SIZE, EXPECTED_DISTANCE / 2.0 + BOX_HEIGHT / 2.0),
        ),
    )
    grid_sensor_box = scene.add_entity(
        gs.morphs.Box(
            size=(0.1, 0.1, 0.1),
            pos=(0.0, 0.0, EXPECTED_DISTANCE + BOX_HEIGHT),
            fixed=True,
        ),
    )
    grid_raycaster = scene.add_sensor(
        gs.sensors.Raycaster(
            pattern=gs.sensors.raycaster.GridPattern(
                resolution=1.0 / (NUM_RAYS_XY - 1.0),
                size=(1.0, 1.0),
                direction=(0.0, 0.0, -1.0),  # pointing downwards to ground
            ),
            entity_idx=grid_sensor_box.idx,
            pos_offset=(0.0, 0.0, -BOX_HEIGHT),
            return_world_frame=True,
            only_cast_fixed=only_cast_fixed,
            draw_debug=True,
        )
    )

    spherical_sensor = scene.add_entity(
        gs.morphs.Sphere(
            radius=EXPECTED_DISTANCE,
            pos=SPHERE_POS,
            fixed=True,
        ),
    )
    spherical_raycaster = scene.add_sensor(
        gs.sensors.Raycaster(
            pattern=gs.sensors.raycaster.SphericalPattern(
                n_points=(NUM_RAYS_XY, NUM_RAYS_XY),
            ),
            entity_idx=spherical_sensor.idx,
            return_world_frame=False,
            only_cast_fixed=only_cast_fixed,
        )
    )

    scene.build(n_envs=n_envs)

    scene.step()

    grid_hits = grid_raycaster.read().points
    grid_distances = grid_raycaster.read().distances
    spherical_distances = spherical_raycaster.read().distances

    expected_shape = (NUM_RAYS_XY, NUM_RAYS_XY) if n_envs == 0 else (n_envs, NUM_RAYS_XY, NUM_RAYS_XY)
    assert grid_distances.shape == spherical_distances.shape == expected_shape

    grid_distance_min = grid_distances.min()
    assert grid_distances.min() < EXPECTED_DISTANCE - tol, "Raycaster grid pattern should have hit obstacle"

    ground_hit_mask = grid_distances > grid_distance_min + tol
    assert_allclose(
        grid_hits[ground_hit_mask][..., 2],
        0.0,
        tol=tol,
        err_msg="Raycaster grid pattern should hit ground (z≈0)",
    )
    assert_allclose(
        grid_distances[ground_hit_mask],
        EXPECTED_DISTANCE,
        tol=tol,
        err_msg=f"Raycaster grid pattern should measure {EXPECTED_DISTANCE}m to ground plane",
    )
    assert_allclose(
        spherical_distances,
        EXPECTED_DISTANCE,
        tol=1e-2,  # since sphere mesh is discretized, we need a larger tolerance here
        err_msg=f"Raycaster spherical pattern should measure {EXPECTED_DISTANCE}m to the sphere around it",
    )

    for _ in range(5):
        scene.step()

    assert grid_raycaster.read().distances.min() > grid_distance_min, "Raycaster should hit falling obstacle"


if __name__ == "__main__":
    gs.init()

    test_raycaster_hits(show_viewer=False, tol=1e-3, n_envs=0, only_cast_fixed=False)
    test_raycaster_hits(show_viewer=False, tol=1e-3, n_envs=2, only_cast_fixed=False)
    # test_raycaster_hits(show_viewer=False, tol=1e-3, n_envs=10, only_cast_fixed=False)
    # test_raycaster_hits(show_viewer=False, tol=1e-3, n_envs=10, only_cast_fixed=True)
