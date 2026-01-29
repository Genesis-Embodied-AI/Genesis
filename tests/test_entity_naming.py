"""Tests for the entity naming system."""

import pytest

import genesis as gs


@pytest.fixture
def scene():
    """Create a basic scene and destroy it after the test."""
    s = gs.Scene(show_viewer=False)
    yield s
    s.destroy()


def test_auto_and_user_names(scene):
    """Test auto-generated and user-specified entity names."""
    # Auto-generated name
    box = scene.add_entity(gs.morphs.Box(size=(0.1, 0.1, 0.1)))
    assert box.name.startswith("box_")
    assert len(box.name) == len("box_") + 8  # 8-char UID suffix

    # User-specified name
    sphere = scene.add_entity(gs.morphs.Sphere(radius=0.1), name="my_sphere")
    assert sphere.name == "my_sphere"

    # Duplicate name raises error
    with pytest.raises(Exception, match="already exists"):
        scene.add_entity(gs.morphs.Cylinder(radius=0.1, height=0.2), name="my_sphere")


def test_get_entity_by_name(scene):
    """Test retrieving entity by name."""
    box = scene.add_entity(gs.morphs.Box(size=(0.1, 0.1, 0.1)), name="test_box")
    assert scene.get_entity(name="test_box") is box

    # Non-existent name raises error
    with pytest.raises(Exception, match="not found"):
        scene.get_entity(name="nonexistent")


def test_get_entity_by_uid(scene):
    """Test retrieving entity by UID prefix."""
    box = scene.add_entity(gs.morphs.Box(size=(0.1, 0.1, 0.1)))
    uid_prefix = str(box.uid)[:4]
    assert scene.get_entity(uid=uid_prefix) is box

    # Non-existent UID raises error
    with pytest.raises(Exception, match="not found"):
        scene.get_entity(uid="zzzzzzzz")


def test_entity_names_property(scene):
    """Test scene.entity_names returns tuple in creation order."""
    scene.add_entity(gs.morphs.Box(size=(0.1, 0.1, 0.1)), name="first")
    scene.add_entity(gs.morphs.Sphere(radius=0.1), name="second")
    names = scene.entity_names
    assert isinstance(names, tuple)
    assert names == ("first", "second")
