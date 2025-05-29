import numpy as np
import pytest

import genesis as gs


@pytest.fixture(scope="session")
def fem_material():
    """Fixture for common FEM material properties"""
    return gs.materials.FEM.Elastic()


@pytest.mark.parametrize("backend", [gs.cpu])
def test_multiple_fem_entities(fem_material, show_viewer):
    """Test adding multiple FEM entities to the scene"""
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=1e-2,
        ),
        fem_options=gs.options.FEMOptions(
            use_explicit_solver=True,
            n_newton_iterations=3,
            n_pcg_iterations=100,
            pcg_threshold=1e-6,
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
    for _ in range(500):
        scene.step()

    for entity in scene.entities:
        state = entity.get_state()
        vel = state.vel.detach().cpu().numpy()
        assert np.allclose(vel, 0.0, atol=2e-3), f"Entity {entity.uid} velocity is not near zero."
        pos = state.pos.detach().cpu().numpy()
        min_pos_z = np.min(pos[..., 2])
        assert np.isclose(
            min_pos_z, 0.0, atol=5e-2
        ), f"Entity {entity.uid} minimum Z position {min_pos_z} is not close to 0.0."
