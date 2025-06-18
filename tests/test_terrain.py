import pytest
import genesis as gs

from .utils import assert_allclose


def test_terrain_size():
    scene_ref = gs.Scene(show_viewer=True)
    terrain_ref = scene_ref.add_entity(
        morph=gs.morphs.Terrain(
            n_subterrains=(2, 2),
            subterrain_size=(12.0, 12.0),
            horizontal_scale=0.25,
            subterrain_types="wave_terrain",
            pattern_scale=1.0,
        )
    )

    height_ref = terrain_ref.geoms[0].metadata["height_field"]
    height_ref_min = terrain_ref.terrain_scale[1] * height_ref.min()
    height_ref_max = terrain_ref.terrain_scale[1] * height_ref.max()

    scene_test = gs.Scene(show_viewer=True)
    terrain_test = scene_test.add_entity(
        morph=gs.morphs.Terrain(
            n_subterrains=(2, 2),
            subterrain_size=(12.0, 12.0),
            horizontal_scale=0.25,
            subterrain_types="wave_terrain",
            pattern_scale=2.0,
        )
    )

    height_test = terrain_test.geoms[0].metadata["height_field"]
    height_test_min = terrain_test.terrain_scale[1] * height_test.min()
    height_test_max = terrain_test.terrain_scale[1] * height_test.max()

    amp_ref = height_ref_max - height_ref_min
    amp_test = height_test_max - height_test_min
    assert_allclose(amp_ref * 2, amp_test, tol=1e-7)
