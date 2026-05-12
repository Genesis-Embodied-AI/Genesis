"""Tests for the entity naming system."""

import pytest
from pydantic import BaseModel

import genesis as gs
from genesis.options.surfaces import Surface
from genesis.options.textures import ColorTexture


@pytest.mark.required
def test_coacd_options_pca_validation():
    gs.options.CoacdOptions(pca=False)
    with pytest.raises(gs.GenesisException, match="pca=True"):
        gs.options.CoacdOptions(pca=True)


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


@pytest.mark.required
def test_surface_shortcut_resolution():
    # Plastic family: color resolves to diffuse_texture; the Rough subclass roughness default (1.0) feeds
    # roughness_texture and default_roughness.
    rough = gs.surfaces.Rough(color=(0.4, 0.4, 0.4))
    assert rough.color == (0.4, 0.4, 0.4)
    assert rough.roughness == 1.0
    assert rough.diffuse_texture.color == (0.4, 0.4, 0.4)
    assert rough.roughness_texture.color == (1.0,)
    assert rough.default_roughness == 1.0

    # Glass: color resolves to specular_texture and the thickness shortcut is honored on the same path.
    glass = gs.surfaces.Glass(color=(0.6, 0.8, 1.0), thickness=0.02)
    assert glass.specular_texture.color == (0.6, 0.8, 1.0)
    assert glass.thickness_texture.color == (0.02,)

    # BSDF exercises multiple shortcuts at once.
    bsdf = gs.surfaces.BSDF(color=(0.2, 0.3, 0.4), roughness=0.3, metallic=0.5)
    assert bsdf.diffuse_texture.color == (0.2, 0.3, 0.4)
    assert bsdf.roughness_texture.color == (0.3,)
    assert bsdf.metallic_texture.color == (0.5,)
    assert bsdf.default_roughness == 0.3

    # Emission: color resolves to emissive_texture.
    emit = gs.surfaces.Emission(color=(1.0, 1.0, 0.0))
    assert emit.emissive_texture.color == (1.0, 1.0, 0.0)

    # Explicit default_roughness wins over the roughness shortcut.
    override = gs.surfaces.Rough(roughness=0.7, default_roughness=0.5)
    assert override.default_roughness == 0.5

    # Nesting an already-resolved surface in another Pydantic model must not re-trigger resolution.
    class Wrapper(BaseModel):
        surface: Surface

    for surface in (rough, glass, bsdf, emit):
        Wrapper(surface=surface)
    Wrapper(surface=rough)
    assert rough.diffuse_texture.color == (0.4, 0.4, 0.4)
    assert rough.roughness_texture.color == (1.0,)

    # Passing both the shortcut and its resolved texture at construction is a user error.
    with pytest.raises(Exception, match="'color' and 'diffuse_texture' cannot both be set"):
        gs.surfaces.Rough(color=(1.0, 0.0, 0.0), diffuse_texture=ColorTexture(color=(0.0, 1.0, 0.0)))
    with pytest.raises(Exception, match="'thickness' and 'thickness_texture' cannot both be set"):
        gs.surfaces.Glass(thickness=0.02, thickness_texture=ColorTexture(color=(0.05,)))
