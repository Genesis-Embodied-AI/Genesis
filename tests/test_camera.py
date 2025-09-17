import numpy as np
import pytest

import genesis as gs


@pytest.mark.required
@pytest.mark.parametrize("n_envs", [0, 1, 2])
@pytest.mark.parametrize("backend", [gs.gpu])
def test_envs_idx(n_envs):
    scene = gs.Scene(
        renderer=gs.renderers.BatchRenderer(use_rasterizer=True),
        show_viewer=False,
        show_FPS=False,
    )

    scene.add_entity(gs.morphs.Plane())
    single_camera = scene.add_camera(
        pos=np.array([0, -1, 1]),
        lookat=np.array([0, 0, 0]),
        res=(64, 64),
        GUI=False,
        env_idx=0 if n_envs > 0 else None,
        debug=True,
    )

    parallel_camera = scene.add_camera(
        pos=np.array([0, -1, 1]),
        lookat=np.array([0, 0, 0]),
        res=(64, 64),
        GUI=False,
    )

    scene.build(n_envs=n_envs)

    if n_envs > 0:
        assert single_camera._pos.shape == (3,)
        assert parallel_camera._pos.shape == (n_envs, 3)
    else:
        assert single_camera._pos.shape == (3,)
        assert parallel_camera._pos.shape == (1, 3)
