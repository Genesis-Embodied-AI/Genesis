"""Tests for the entity naming system."""

import pytest

import genesis as gs


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
def test_surface_shortcut_resolution_is_idempotent():
    """``Surface._resolve_shortcuts`` must be safe to run more than once.

    Pydantic re-runs ``model_validator(mode="after")`` validators when an
    instance is nested inside another model. Without idempotency, the second
    pass sees both the shortcut (e.g. ``color``) and the resolved texture
    field (``diffuse_texture``) populated and raises
    ``"'color' and 'diffuse_texture' cannot both be set."``.
    """
    from pydantic import BaseModel
    from genesis.options.surfaces import Surface

    class Wrapper(BaseModel):
        surface: Surface

    # Plastic-family: color → diffuse_texture, roughness (default) → roughness_texture.
    rough = gs.surfaces.Rough(color=(0.4, 0.4, 0.4))
    assert rough.color is None, "color shortcut should be cleared after resolution"
    assert rough.roughness is None, "roughness shortcut should be cleared after resolution"
    assert rough.diffuse_texture.color == (0.4, 0.4, 0.4)
    assert rough.roughness_texture.color == (1.0,)
    Wrapper(surface=rough)  # used to raise

    # Glass: color → specular_texture, plus the thickness shortcut.
    glass = gs.surfaces.Glass(color=(0.6, 0.8, 1.0), thickness=0.02)
    assert glass.color is None
    assert glass.thickness is None
    assert glass.specular_texture.color == (0.6, 0.8, 1.0)
    assert glass.thickness_texture.color == (0.02,)
    Wrapper(surface=glass)

    # BSDF: exercise multiple shortcuts simultaneously (color, roughness, metallic).
    bsdf = gs.surfaces.BSDF(color=(0.2, 0.3, 0.4), roughness=0.3, metallic=0.5)
    assert bsdf.color is None and bsdf.roughness is None and bsdf.metallic is None
    assert bsdf.diffuse_texture.color == (0.2, 0.3, 0.4)
    assert bsdf.roughness_texture.color == (0.3,)
    assert bsdf.metallic_texture.color == (0.5,)
    Wrapper(surface=bsdf)

    # Emit: color → emissive_texture.
    emit = gs.surfaces.Emission(color=(1.0, 1.0, 0.0))
    assert emit.color is None
    assert emit.emissive_texture.color == (1.0, 1.0, 0.0)
    Wrapper(surface=emit)

    # Re-using the same already-resolved surface in multiple wrappers must not
    # mutate it further (i.e. ``_resolve_shortcuts`` is a no-op on resolved
    # instances). This exercises double re-validation.
    Wrapper(surface=rough)
    Wrapper(surface=rough)
    assert rough.diffuse_texture.color == (0.4, 0.4, 0.4)
    assert rough.roughness_texture.color == (1.0,)


@pytest.mark.required
def test_surface_default_roughness_sync():
    """``default_roughness`` must mirror the ``roughness`` shortcut unless the
    user passes an explicit ``default_roughness`` (in which case the user wins).

    The sync was moved to the top of ``_resolve_shortcuts`` when the loop that
    consumes ``self.roughness`` started clearing it; this guards that ordering.
    """
    # roughness shortcut drives default_roughness when no explicit override.
    rough = gs.surfaces.Rough(roughness=0.3)
    assert rough.default_roughness == 0.3

    # Explicit default_roughness wins over the roughness shortcut.
    rough_override = gs.surfaces.Rough(roughness=0.7, default_roughness=0.5)
    assert rough_override.default_roughness == 0.5

    # No roughness shortcut → default_roughness keeps its declared default.
    plain = gs.surfaces.Plastic()
    assert plain.default_roughness == 1.0


@pytest.mark.required
def test_surface_shortcut_conflict_still_detected_on_first_construct():
    """Setting both the shortcut and its resolved texture field on construction
    must still raise — only re-validation of an already-resolved instance is
    made safe by the idempotency fix."""
    from genesis.options.textures import ColorTexture

    with pytest.raises(Exception, match="'color' and 'diffuse_texture' cannot both be set"):
        gs.surfaces.Rough(color=(1.0, 0.0, 0.0), diffuse_texture=ColorTexture(color=(0.0, 1.0, 0.0)))

    with pytest.raises(Exception, match="'thickness' and 'thickness_texture' cannot both be set"):
        gs.surfaces.Glass(thickness=0.02, thickness_texture=ColorTexture(color=(0.05,)))
