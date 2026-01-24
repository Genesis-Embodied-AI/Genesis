import pytest

import genesis as gs


@pytest.mark.required
def test_scene_destroy_cleans_up_simulator():
    scene = gs.Scene(show_viewer=False)
    scene.add_entity(morph=gs.morphs.Plane())
    scene.build()
    scene.step()

    assert scene._sim is not None

    scene.destroy()

    assert scene._sim is None
    assert scene._visualizer is None


@pytest.mark.required
def test_scene_destroy_idempotent():
    scene = gs.Scene(show_viewer=False)
    scene.add_entity(morph=gs.morphs.Plane())
    scene.build()
    scene.step()

    scene.destroy()
    assert scene._sim is None

    scene.destroy()
    assert scene._sim is None
