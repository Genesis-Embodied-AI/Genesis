"""Integration tests for ImGuiOverlayPlugin with a real Genesis scene."""

import numpy as np
import pytest

import genesis as gs
from genesis.ext.pyrender.imgui_overlay import ImGuiOverlayPlugin


@pytest.mark.required
def test_imgui_overlay_plugin(show_viewer):
    """Test ImGuiOverlayPlugin with a real Panda scene: state, should_step, cache, qpos update."""
    scene = gs.Scene(show_viewer=show_viewer, show_FPS=False)
    scene.add_entity(gs.morphs.Plane())
    panda = scene.add_entity(gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"))
    scene.build()

    plugin = ImGuiOverlayPlugin()
    plugin.scene = scene
    # viewer reference not needed for non-rendering tests; set to None
    plugin.viewer = None

    # -- Initial state --
    assert plugin.paused is False
    assert plugin._available is False

    # -- should_step logic --
    assert plugin.should_step() is True
    plugin.paused = True
    assert plugin.should_step() is False
    plugin._step_requested = True
    assert plugin.should_step() is True
    assert plugin.should_step() is False  # consumed
    plugin.paused = False

    # -- _cache_entity_data with real entities --
    plugin._cache_entity_data()
    assert len(plugin._entity_cache) > 0

    # Find the Panda entry (has DOFs)
    panda_data = None
    for data in plugin._entity_cache.values():
        if data["n_qs"] > 0:
            panda_data = data
            break
    assert panda_data is not None
    assert panda_data["n_qs"] == panda.n_qs
    assert len(panda_data["q_names"]) == panda.n_qs
    assert not panda_data["has_free_joint"]

    # -- set_qpos update (what _apply_qpos_update delegates to) --
    new_qpos = np.zeros(panda.n_qs, dtype=np.float64)
    new_qpos[0] = 0.5
    panda.set_qpos(new_qpos)
    updated_qpos = panda.get_qpos().cpu().numpy()
    np.testing.assert_allclose(updated_qpos, new_qpos, atol=1e-6)
