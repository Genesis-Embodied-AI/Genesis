import pytest

import genesis as gs


@pytest.fixture(scope="session")
def fem_material():
    """Fixture for common FEM material properties"""
    return gs.materials.FEM.Muscle(
        E=3.0e4,
        nu=0.45,
        rho=1000.0,
        model="stable_neohooken",
    )


@pytest.mark.parametrize("backend", [gs.cpu])
def test_multiple_fem_entities(fem_material, show_viewer):
    """Test adding multiple FEM entities to the scene"""
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=5e-4,
            substeps=10,
            gravity=(0.0, 0.0, 0.0),
        ),
        fem_options=gs.options.FEMOptions(
            damping=45.0,
        ),
        show_viewer=show_viewer,
    )

    # Add first FEM entity
    scene.add_entity(
        morph=gs.morphs.Sphere(
            pos=(0.5, -0.2, 0.3),
            radius=0.1,
        ),
        material=fem_material,
    )

    # Add second FEM entity
    scene.add_entity(
        morph=gs.morphs.Box(
            size=(0.1, 0.1, 0.1),
            pos=(0.0, 0.0, 0.5),
        ),
        material=fem_material,
    )

    # Build the scene
    scene.build()

    # Run simulation
    for _ in range(100):
        scene.step()
