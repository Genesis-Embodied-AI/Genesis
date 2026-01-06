"""
Unit test for plane primitive mass calculation.

This test ensures that plane entities have non-zero mass and can be used in physics simulations.
Regression test for issue where planes had epsilon mass due to infinite plane geometry having no volume.
"""

import numpy as np
import pytest

import genesis as gs


def test_plane_has_nonzero_mass():
    """Test that a plane entity has a reasonable non-zero mass."""
    scene = gs.Scene(show_viewer=False)

    # Create a plane with default parameters
    plane = scene.add_entity(
        morph=gs.morphs.Plane(),
    )

    scene.build()

    # Get the plane's mass
    mass = plane.get_mass()

    # The plane should have substantial mass, not epsilon
    # Default plane_size is (1e3, 1e3), thickness is 1e-2
    # Volume = 1000 * 1000 * 0.01 = 10,000 m³
    # With default density ~200 kg/m³ (Genesis default for Rigid material), mass should be ~2,000,000 kg
    assert mass > 1.0, f"Plane mass {mass} is too small (should be > 1 kg)"
    assert mass < 1e10, f"Plane mass {mass} is unreasonably large"

    # More specifically, check it's in the expected range for default parameters
    # Note: Default rigid material density is 200 kg/m³ in Genesis
    expected_mass = 1000 * 1000 * 0.01 * 200  # plane_size * thickness * default_rho
    assert abs(mass - expected_mass) / expected_mass < 0.1, f"Plane mass {mass} differs from expected {expected_mass}"

    print(f"✓ Plane has correct mass: {mass} kg")


def test_plane_custom_size_mass():
    """Test that plane mass scales correctly with custom plane_size."""
    scene = gs.Scene(show_viewer=False)

    # Create a smaller plane
    custom_size = (10.0, 10.0)
    plane = scene.add_entity(
        morph=gs.morphs.Plane(plane_size=custom_size),
    )

    scene.build()

    mass = plane.get_mass()

    # Expected: 10 * 10 * 0.01 * 200 = 200 kg (default rho is 200 kg/m³)
    expected_mass = custom_size[0] * custom_size[1] * 0.01 * 200
    assert abs(mass - expected_mass) / expected_mass < 0.1, f"Plane mass {mass} differs from expected {expected_mass}"

    print(f"✓ Custom plane has correct mass: {mass} kg (expected ~{expected_mass} kg)")


def test_plane_can_be_used_in_simulation():
    """Test that a plane with proper mass can be used in a simple simulation."""
    scene = gs.Scene(show_viewer=False, sim_options=gs.options.SimOptions(dt=0.01, substeps=10))

    # Create plane and a box above it
    plane = scene.add_entity(
        morph=gs.morphs.Plane(),
    )

    box = scene.add_entity(
        morph=gs.morphs.Box(pos=(0, 0, 1.0), size=(0.2, 0.2, 0.2)),
    )

    scene.build()

    # Verify plane has reasonable mass
    plane_mass = plane.get_mass()
    assert plane_mass > 1.0, f"Plane mass {plane_mass} is too small"

    # Run simulation - should not crash
    for _ in range(10):
        scene.step()

    # Box should have fallen due to gravity
    box_pos = box.get_pos()
    assert box_pos[2] < 1.0, "Box should have fallen"

    print(f"✓ Simulation with plane (mass={plane_mass} kg) works correctly")

