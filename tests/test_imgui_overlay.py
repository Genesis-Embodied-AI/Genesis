"""Unit tests for ImGuiOverlayPlugin (no imgui-bundle or renderer required).

All tests use pure stubs/mocks — no gs.Scene.build(), no EGL, no GPU.
"""

import pytest
import numpy as np
from unittest.mock import MagicMock

import genesis as gs


def _make_stub_joint(name, joint_type, n_qs, n_dofs=None, dofs_limit=None):
    """Create a stub joint for _cache_entity_data testing."""
    joint = MagicMock()
    joint.name = name
    joint.type = joint_type
    joint.n_qs = n_qs
    joint.n_dofs = n_dofs if n_dofs is not None else n_qs
    if dofs_limit is not None:
        joint.dofs_limit = np.array(dofs_limit, dtype=np.float64)
    else:
        joint.dofs_limit = np.array([[-3.14, 3.14]] * n_qs, dtype=np.float64)
    return joint


def _make_stub_entity(idx, joints, n_dofs=None, n_qs=None):
    """Create a stub entity with joints for _cache_entity_data."""
    entity = MagicMock()
    entity.idx = idx
    entity.joints = joints
    total_qs = sum(j.n_qs for j in joints if j.type != gs.JOINT_TYPE.FIXED)
    entity.n_dofs = n_dofs if n_dofs is not None else total_qs
    entity.n_qs = n_qs if n_qs is not None else total_qs
    entity.name = f"entity_{idx}"
    return entity


def _make_plugin_with_scene(entities):
    """Create an ImGuiOverlayPlugin with a stubbed scene containing entities."""
    from genesis.ext.pyrender.imgui_overlay import ImGuiOverlayPlugin

    plugin = ImGuiOverlayPlugin()
    mock_scene = MagicMock()
    mock_scene.rigid_solver.entities = entities
    mock_scene.rigid_solver.is_active = False
    plugin.scene = mock_scene
    plugin.viewer = MagicMock()
    return plugin


@pytest.mark.required
def test_imgui_overlay_plugin():
    """Test ImGuiOverlayPlugin core logic: initial state, should_step(), _cache_entity_data()."""
    from genesis.ext.pyrender.imgui_overlay import ImGuiOverlayPlugin

    plugin = ImGuiOverlayPlugin()

    # Test initial state
    assert plugin.paused is False
    assert plugin._available is False

    # Test should_step() logic
    assert plugin.should_step() is True
    plugin.paused = True
    assert plugin.should_step() is False
    plugin._step_requested = True
    assert plugin.should_step() is True
    assert plugin.should_step() is False

    # Test _cache_entity_data() with a Panda-like entity (9 revolute joints)
    joints = [_make_stub_joint(f"joint{i}", gs.JOINT_TYPE.REVOLUTE, n_qs=1) for i in range(9)]
    entity = _make_stub_entity(idx=0, joints=joints, n_dofs=9, n_qs=9)
    plugin = _make_plugin_with_scene([entity])

    plugin._cache_entity_data()
    assert len(plugin._entity_cache) == 1
    entity_data = list(plugin._entity_cache.values())[0]
    assert "q_names" in entity_data
    assert entity_data["n_qs"] == 9

    # Verify speed/get_speed don't exist (AC-1 negative check)
    with pytest.raises(AttributeError):
        _ = plugin.speed
    with pytest.raises(AttributeError):
        plugin.get_speed()


@pytest.mark.required
def test_imgui_overlay_spherical_joint():
    """Test _cache_entity_data handles spherical (ball) joints without IndexError."""
    # Spherical joint: n_qs=4 (quaternion), n_dofs=3
    joint = _make_stub_joint("ball", gs.JOINT_TYPE.SPHERICAL, n_qs=4, n_dofs=3)
    entity = _make_stub_entity(idx=0, joints=[joint], n_dofs=3, n_qs=4)
    plugin = _make_plugin_with_scene([entity])

    # Should NOT raise IndexError for spherical joints
    plugin._cache_entity_data()

    assert len(plugin._entity_cache) == 1
    entity_data = list(plugin._entity_cache.values())[0]

    # Ball joint has 4 quaternion components (qw, qx, qy, qz)
    assert entity_data["n_qs"] == 4
    assert all(entity_data["q_is_quaternion"])  # All should be True


@pytest.mark.required
def test_imgui_overlay_multi_env():
    """Test _apply_qpos_update passes envs_idx=0 for multi-env scenes."""
    from genesis.ext.pyrender.imgui_overlay import ImGuiOverlayPlugin

    plugin = ImGuiOverlayPlugin()
    mock_scene = MagicMock()
    mock_scene.rigid_solver.is_active = False
    plugin.scene = mock_scene
    plugin.viewer = MagicMock()

    mock_entity = MagicMock()
    test_qpos = [0.1] * 9

    # Multi-env: set_qpos must be called WITH envs_idx=0
    plugin._apply_qpos_update(mock_entity, test_qpos, is_multi_env=True)
    assert mock_entity.set_qpos.call_count == 1
    _, kwargs = mock_entity.set_qpos.call_args
    assert kwargs.get("envs_idx") == 0, "envs_idx=0 should be passed for multi-env"


@pytest.mark.required
def test_imgui_overlay_single_env_update():
    """Test _apply_qpos_update does NOT pass envs_idx for single-env scenes."""
    from genesis.ext.pyrender.imgui_overlay import ImGuiOverlayPlugin

    plugin = ImGuiOverlayPlugin()
    mock_scene = MagicMock()
    mock_scene.rigid_solver.is_active = False
    plugin.scene = mock_scene
    plugin.viewer = MagicMock()

    mock_entity = MagicMock()
    test_qpos = [0.2] * 9

    # Single-env: set_qpos must be called WITHOUT envs_idx
    plugin._apply_qpos_update(mock_entity, test_qpos, is_multi_env=False)
    assert mock_entity.set_qpos.call_count == 1
    call_args, call_kwargs = mock_entity.set_qpos.call_args
    assert "envs_idx" not in call_kwargs, "envs_idx should NOT be passed for single-env"
    assert len(call_args) == 1, "Only qpos should be passed as positional arg for single-env"
    np.testing.assert_array_equal(call_args[0], np.asarray(test_qpos))


@pytest.mark.required
def test_imgui_overlay_apply_qpos_update_signature():
    """Test _apply_qpos_update passes correct arguments based on is_multi_env.

    This test uses a mock to verify that:
    - Single-env (is_multi_env=False): set_qpos is called WITHOUT envs_idx
    - Multi-env (is_multi_env=True): set_qpos is called WITH envs_idx=0
    """
    from genesis.ext.pyrender.imgui_overlay import ImGuiOverlayPlugin

    plugin = ImGuiOverlayPlugin()

    # Mock scene and viewer so _apply_qpos_update's post-set_qpos path works
    mock_scene = MagicMock()
    mock_scene.rigid_solver.is_active = False
    plugin.scene = mock_scene
    plugin.viewer = MagicMock()

    # Create a mock entity with a mocked set_qpos method
    mock_entity = MagicMock()
    test_qpos = [0.1, 0.2, 0.3]

    # Test single-env case: should NOT pass envs_idx
    mock_entity.reset_mock()
    plugin._apply_qpos_update(mock_entity, test_qpos, is_multi_env=False)

    # Verify set_qpos was called exactly once
    assert mock_entity.set_qpos.call_count == 1
    call_args, call_kwargs = mock_entity.set_qpos.call_args
    assert "envs_idx" not in call_kwargs, "envs_idx should NOT be passed for single-env"
    assert len(call_args) == 1, "Only qpos should be passed as positional arg for single-env"
    np.testing.assert_array_equal(call_args[0], np.asarray(test_qpos))

    # Test multi-env case: should pass envs_idx=0
    mock_entity.reset_mock()
    plugin._apply_qpos_update(mock_entity, test_qpos, is_multi_env=True)

    assert mock_entity.set_qpos.call_count == 1
    call_args, call_kwargs = mock_entity.set_qpos.call_args
    assert call_kwargs.get("envs_idx") == 0, "envs_idx=0 should be passed for multi-env"
    np.testing.assert_array_equal(call_args[0], np.asarray(test_qpos))
