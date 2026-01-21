import numpy as np
import pytest

import genesis as gs

from .utils import assert_allclose


@pytest.mark.required
def test_lidar_parallel_env_distances(show_viewer):
    """Verify each environment receives a different lidar distance when geometries differ."""
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=1e-2, substeps=1),
        vis_options=gs.options.VisOptions(rendered_envs_idx=(0,)),
        show_viewer=show_viewer,
    )
    scene.add_entity(gs.morphs.Plane())

    sensor_mount = scene.add_entity(
        gs.morphs.Box(size=(0.1, 0.1, 0.1), pos=(0.0, 0.0, 0.5), fixed=True, collision=False)
    )
    obstacle = scene.add_entity(
        gs.morphs.Box(size=(0.2, 0.2, 0.2), pos=(1.0, 0.0, 0.5), fixed=True),
    )

    lidar = scene.add_sensor(
        gs.sensors.Lidar(
            entity_idx=sensor_mount.idx,
            pattern=gs.options.sensors.SphericalPattern(n_points=(1, 1), fov=(0.0, 0.0)),
            max_range=5.0,
        )
    )

    scene.build(n_envs=2, env_spacing=(5.0, 5.0))

    sensor_positions = np.array([[0.0, 0.0, 0.5], [20.0, 0.0, 0.5]], dtype=np.float32)
    obstacle_positions = np.array([[1.1, 0.0, 0.5], [22.5, 0.0, 0.5]], dtype=np.float32)
    sensor_mount.set_pos(sensor_positions)
    obstacle.set_pos(obstacle_positions)

    scene.step(update_visualizer=show_viewer)
    if show_viewer:
        scene.visualizer.update(force=True)

    distances = lidar.read().distances
    assert distances.shape == (2, 1, 1)
    lidar_distances = distances[:, 0, 0]

    front_positions = obstacle_positions[:, 0] - 0.5 * 0.2
    expected_distances = front_positions - sensor_positions[:, 0]
    assert_allclose(lidar_distances, expected_distances, tol=1e-4)
