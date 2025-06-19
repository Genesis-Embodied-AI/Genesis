import pytest
import genesis as gs

from .utils import assert_allclose


def test_terrain_size():
    scene_ref = gs.Scene(show_viewer=False)
    terrain_ref = scene_ref.add_entity(
        morph=gs.morphs.Terrain(
            n_subterrains=(2, 2),
            subterrain_size=(12.0, 12.0),
            horizontal_scale=0.25,
            subterrain_types="wave_terrain",
        )
    )

    height_ref = terrain_ref.geoms[0].metadata["height_field"]

    scene_test = gs.Scene(show_viewer=False)
    terrain_test = scene_test.add_entity(
        morph=gs.morphs.Terrain(
            n_subterrains=(2, 2),
            subterrain_size=(12.0, 12.0),
            horizontal_scale=0.25,
            subterrain_types="wave_terrain",
            subterrain_parameters={"wave_terrain": {"amplitude": 2.0}},
        )
    )

    height_test = terrain_test.geoms[0].metadata["height_field"]

    assert_allclose((height_ref * 2).all(), height_test.all(), tol=1e-7)
