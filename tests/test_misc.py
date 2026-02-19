"""Tests for the entity naming system."""

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


@pytest.mark.required
def test_auto_and_user_names():
    """Test auto-generated and user-specified entity names."""
    scene = gs.Scene()

    # Auto-generated name
    box = scene.add_entity(gs.morphs.Box(size=(0.1, 0.1, 0.1)))
    assert box.name.startswith("box_")

    # Multiple identical entities should have unique names
    box2 = scene.add_entity(gs.morphs.Box(size=(0.1, 0.1, 0.1)))
    assert box2.name.startswith("box_")
    assert box.name != box2.name

    # User-specified name
    sphere = scene.add_entity(gs.morphs.Sphere(radius=0.1), name="my_sphere")
    assert sphere.name == "my_sphere"

    # Duplicate name raises error
    with pytest.raises(Exception, match="already exists"):
        scene.add_entity(gs.morphs.Cylinder(radius=0.1, height=0.2), name="my_sphere")


@pytest.mark.required
def test_get_entity_by_name():
    """Test retrieving entity by name."""
    scene = gs.Scene()

    box = scene.add_entity(gs.morphs.Box(size=(0.1, 0.1, 0.1)), name="test_box")
    assert scene.get_entity(name="test_box") is box

    # Non-existent name raises error
    with pytest.raises(Exception, match="not found"):
        scene.get_entity(name="nonexistent")


@pytest.mark.required
def test_get_entity_by_uid():
    """Test retrieving entity by short UID."""
    scene = gs.Scene()

    box = scene.add_entity(gs.morphs.Box(size=(0.1, 0.1, 0.1)))

    # Short UID lookup (7-character prefix shown in terminal)
    assert scene.get_entity(uid=box.uid.short()) is box

    # Non-existent UID raises error
    with pytest.raises(Exception, match="not found"):
        scene.get_entity(uid=gs.UID().short())


@pytest.mark.required
def test_entity_names_property():
    """Test scene.entity_names returns names in creation order."""
    scene = gs.Scene()

    # Use "B" then "A" to confirm insertion order (not sorted)
    scene.add_entity(gs.morphs.Box(size=(0.1, 0.1, 0.1)), name="B")
    scene.add_entity(gs.morphs.Sphere(radius=0.1), name="A")
    assert tuple(scene.entity_names) == ("B", "A")


@pytest.mark.required
def test_urdf_mjcf_names_from_file():
    """Test that URDF/MJCF entities use robot/model names from files."""
    scene = gs.Scene()

    # URDF: plane.urdf has <robot name="plane">
    urdf_entity = scene.add_entity(gs.morphs.URDF(file="urdf/plane/plane.urdf"))
    assert urdf_entity.name.startswith("plane_")

    # MJCF: panda.xml has <mujoco model="panda">
    mjcf_entity = scene.add_entity(gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"))
    assert mjcf_entity.name.startswith("panda_")

    # Multiple URDF entities should have unique names
    urdf_entity2 = scene.add_entity(gs.morphs.URDF(file="urdf/plane/plane.urdf"))
    assert urdf_entity2.name.startswith("plane_")
    assert urdf_entity.name != urdf_entity2.name
