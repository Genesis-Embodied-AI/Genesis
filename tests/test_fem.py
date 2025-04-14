import pytest
import genesis as gs


@pytest.fixture
def fem_scene():
    """Fixture for basic FEM scene setup"""
    try:
        gs.init(seed=0, precision="32", logging_level="debug")
    except Exception as err:
        # Ignore exception when genesis has been already initialized
        if "Genesis already initialized." not in str(err):
            raise err

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            substeps=10,
            gravity=(0, 0, 0),
        ),
        fem_options=gs.options.FEMOptions(
            dt=5e-4,
            damping=45.0,
        ),
        show_viewer=False,
    )

    yield scene

    # Cleanup
    scene = None


@pytest.fixture
def fem_material():
    """Fixture for common FEM material properties"""
    return gs.materials.FEM.Muscle(
        E=3.0e4,
        nu=0.45,
        rho=1000.0,
        model="stable_neohooken",
    )


def test_multiple_fem_entities(fem_scene, fem_material):
    """Test adding multiple FEM entities to the scene"""
    # Add first FEM entity
    fem_scene.add_entity(
        morph=gs.morphs.Sphere(
            pos=(0.5, -0.2, 0.3),
            radius=0.1,
        ),
        material=fem_material,
    )

    # Add second FEM entity
    fem_scene.add_entity(
        morph=gs.morphs.Box(size=(0.1, 0.1, 0.1), pos=(0.0, 0.0, 0.5)),
        material=fem_material,
    )

    # Build and run the scene
    fem_scene.build()
    fem_scene.reset()

    # Run simulation steps
    for _ in range(100):
        fem_scene.step()
