# Tests for SPH simulation - initial density and pressure validation
#
# Background:
# SPH is sensitive to the initial particle distribution, as it directly determines the initial density and pressure fields.
# To ensure numerical stability, particles must be initialized using a regular sampler that enforces near-uniform spacing.
# Irregular samplers (e.g. pbs, random) introduce local density fluctuations at initialization, which lead to large
# spurious pressure forces and can cause the simulation to become unstable or diverge.

import pytest
import numpy as np

import genesis as gs


def create_sph_scene(show_viewer, particle_size=0.01, pressure_solver="WCSPH"):
    """Create a basic SPH scene with regular sampler."""
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=4e-3,
            substeps=10,
        ),
        sph_options=gs.options.SPHOptions(
            lower_bound=(-0.5, -0.5, 0.0),
            upper_bound=(0.5, 0.5, 1),
            particle_size=particle_size,
            pressure_solver=pressure_solver,
        ),
        show_viewer=show_viewer,
    )

    scene.add_entity(
        morph=gs.morphs.Plane(),
    )

    liquid = scene.add_entity(
        material=gs.materials.SPH.Liquid(sampler="regular"),
        morph=gs.morphs.Box(
            pos=(0.0, 0.0, 0.65),
            size=(0.4, 0.4, 0.4),
        ),
        surface=gs.surfaces.Default(
            color=(0.4, 0.8, 1.0),
            vis_mode="particle",
        ),
    )

    scene.build()

    return scene, liquid


@pytest.fixture(scope="function")
def sph_scene(show_viewer):
    """Fixture to create a basic SPH scene with regular sampler."""
    return create_sph_scene(show_viewer)


def get_sph_density_pressure(scene, n_particles):
    """
    Helper function to compute and retrieve density and pressure from SPH solver.

    This triggers the density computation kernel and returns the computed values.

    Parameters
    ----------
    scene : gs.Scene
        The built scene containing SPH solver.
    n_particles : int
        Number of particles to retrieve.

    Returns
    -------
    densities : np.ndarray
        Array of density values for all particles.
    pressures : np.ndarray
        Array of pressure values for all particles.
    """
    sph_solver = scene.sph_solver

    # Reorder particles into spatial grid cells for efficient neighbor queries,
    # then compute density. This mirrors what happens in substep_pre_coupling.
    # We must reorder first because _kernel_compute_rho operates on particles_reordered.
    sph_solver._kernel_reorder_particles(0)
    sph_solver._kernel_compute_rho(0)

    # Get density values from reordered particles (where _kernel_compute_rho stores results)
    densities = sph_solver.particles_reordered.rho.to_numpy()[:n_particles, 0]

    # Get material properties for pressure computation
    rho0 = sph_solver.particles_info[0].rho
    stiffness = sph_solver.particles_info[0].stiffness
    exponent = sph_solver.particles_info[0].exponent

    # Compute pressure using the same formula as the solver:
    # p = stiffness * ((max(rho, rho0) / rho0)^exponent - 1)
    densities_clamped = np.maximum(densities, rho0)
    pressures = stiffness * (np.power(densities_clamped / rho0, exponent) - 1.0)

    return densities, pressures


@pytest.mark.required
@pytest.mark.parametrize("pressure_solver", ["WCSPH", "DFSPH"])
def test_sph_initial_density_regular_sampler(show_viewer, pressure_solver, tol):
    """
    Test that SPH with regular sampler produces reasonable initial density distribution.

    With a regular sampler, particles are placed at uniform spacing, which results in:
    - Initial density values that are consistent across interior particles
    - Lower density at boundaries (expected due to fewer neighbors)
    - Density standard deviation that is bounded (not wildly fluctuating)

    The test verifies that:
    1. Mean density is within expected bounds (50-100% of rest density)
    2. Density standard deviation is bounded (< 10% of rest density)
    3. No particles have extreme density values (> 1.5 * rho0 or < 0.3 * rho0)
    """
    scene, _ = create_sph_scene(show_viewer, pressure_solver=pressure_solver)
    sph_solver = scene.sph_solver
    n_particles = sph_solver.n_particles
    rho0 = sph_solver.particles_info[0].rho

    # Get initial density values
    densities, _ = get_sph_density_pressure(scene, n_particles)

    # Test 1: Mean density should be within reasonable bounds
    # For regular sampler, mean is typically 70-90% of rho0 due to boundary effects
    density_mean = densities.mean()
    assert 0.5 * rho0 < density_mean < rho0, (
        f"Mean initial density {density_mean:.2f} is outside expected range "
        f"[{0.5 * rho0:.2f}, {rho0:.2f}] for rest density {rho0}"
    )

    # Test 2: Density standard deviation should be bounded
    # Regular sampler produces relatively uniform density distribution
    density_std = densities.std()
    assert density_std < 0.1 * rho0, (
        f"Density standard deviation {density_std:.2f} is too high "
        f"(expected < {0.1 * rho0:.2f} for rest density {rho0})"
    )

    # Test 3: No extreme density values
    assert densities.min() > 0.3 * rho0, (
        f"Minimum density {densities.min():.2f} is too low (expected > {0.3 * rho0:.2f})"
    )
    assert densities.max() < 1.5 * rho0, (
        f"Maximum density {densities.max():.2f} is too high (expected < {1.5 * rho0:.2f})"
    )


@pytest.mark.required
@pytest.mark.parametrize("pressure_solver", ["WCSPH", "DFSPH"])
def test_sph_initial_pressure_regular_sampler(show_viewer, pressure_solver, tol):
    """
    Test that SPH with regular sampler produces near-zero initial pressure.

    With properly initialized particles using a regular sampler:
    - Initial density is at or below rest density (rho <= rho0)
    - When rho < rho0, the pressure formula clamps density to rho0
    - This results in pressure = stiffness * ((rho0/rho0)^exp - 1) = 0

    This is critical for stability: non-zero initial pressure would cause
    spurious forces that destabilize the simulation.
    """
    scene, _ = create_sph_scene(show_viewer, pressure_solver=pressure_solver)
    sph_solver = scene.sph_solver
    n_particles = sph_solver.n_particles
    rho0 = sph_solver.particles_info[0].rho
    stiffness = sph_solver.particles_info[0].stiffness

    # Get initial pressure values
    densities, pressures = get_sph_density_pressure(scene, n_particles)

    # Test: For regular sampler, initial density should be <= rho0
    # which means pressure should be 0 (clamped)
    # All particles should have near-zero pressure initially
    assert np.allclose(pressures, 0.0, atol=1e-6), (
        f"Initial pressure is not zero. Mean: {pressures.mean():.6f}, "
        f"Max: {pressures.max():.6f}. "
        f"This indicates density > rho0 for some particles, which shouldn't happen "
        f"with regular sampler."
    )

    # Also verify that density is indeed <= rho0 for most particles
    fraction_above_rho0 = (densities > rho0).sum() / n_particles
    assert fraction_above_rho0 < 0.01, (
        f"{fraction_above_rho0 * 100:.1f}% of particles have density > rho0. "
        f"Regular sampler should produce density <= rho0 for almost all particles."
    )


@pytest.mark.required
@pytest.mark.parametrize("pressure_solver", ["WCSPH", "DFSPH"])
def test_sph_simulation_stability_regular_sampler(show_viewer, pressure_solver, tol):
    """
    Test that SPH simulation with regular sampler remains stable over multiple steps.

    A stable simulation should:
    1. Not produce NaN or Inf values in particle positions/velocities
    2. Have bounded maximum velocities (not exploding)
    3. Particles should remain within the simulation domain

    This test runs a few simulation steps and verifies stability.
    """
    scene, liquid = create_sph_scene(show_viewer, pressure_solver=pressure_solver)
    sph_solver = scene.sph_solver
    n_particles = sph_solver.n_particles

    # Get initial positions for comparison
    initial_pos = liquid.get_particles_pos().clone()

    # Run simulation for a few steps
    num_steps = 5
    max_velocities = []
    for step in range(num_steps):
        scene.step()
        pos = liquid.get_particles_pos()
        vel = liquid.get_particles_vel()
        assert not pos.isnan().any(), f"NaN values detected in positions at step {step}"
        assert not pos.isinf().any(), f"Inf values detected in positions at step {step}"
        assert not vel.isnan().any(), f"NaN values detected in velocities at step {step}"
        assert not vel.isinf().any(), f"Inf values detected in velocities at step {step}"

        # Track maximum velocity
        max_vel = vel.norm(dim=-1).max()
        max_velocities.append(max_vel)

    # Maximum velocity should be bounded (not exploding)
    # For stable simulation with gravity, velocities should be reasonable
    max_velocity_overall = max(max_velocities)
    assert max_velocity_overall < 10.0, (
        f"Maximum velocity {max_velocity_overall:.2f} exceeds reasonable bounds. This may indicate instability."
    )

    # Get final positions
    final_pos = liquid.get_particles_pos()

    # Check particles haven't moved too far (not exploding)
    displacement = (final_pos - initial_pos).norm(dim=-1).max()
    assert displacement < 1.0, (
        f"Maximum displacement {displacement:.4f} is too large after {num_steps} steps. This may indicate instability."
    )


@pytest.mark.required
@pytest.mark.parametrize("pressure_solver", ["WCSPH", "DFSPH"])
@pytest.mark.parametrize("particle_size", [0.01, 0.02])
def test_sph_density_consistency_different_particle_sizes(show_viewer, pressure_solver, particle_size, tol):
    """
    Test that initial density behavior is consistent across different particle sizes.

    The relative density distribution (normalized by rho0) should be similar
    regardless of particle size, as the SPH kernel is scaled accordingly.
    """
    scene, _ = create_sph_scene(
        show_viewer,
        particle_size=particle_size,
        pressure_solver=pressure_solver,
    )

    sph_solver = scene.sph_solver
    n_particles = sph_solver.n_particles
    rho0 = sph_solver.particles_info[0].rho

    # Get density values
    densities, pressures = get_sph_density_pressure(scene, n_particles)

    # Normalized metrics should be similar regardless of particle size
    normalized_mean = densities.mean() / rho0
    normalized_std = densities.std() / rho0

    # Mean density should be in similar range (50-100% of rho0)
    assert 0.5 < normalized_mean < 1.0, (
        f"Normalized mean density {normalized_mean:.3f} is outside expected range [0.5, 1.0] "
        f"for particle_size={particle_size}"
    )

    # Standard deviation should be bounded
    assert normalized_std < 0.1, (
        f"Normalized density std {normalized_std:.3f} is too high for particle_size={particle_size}"
    )

    # Pressure should still be zero
    assert np.allclose(pressures, 0.0, atol=1e-6), (
        f"Initial pressure is not zero for particle_size={particle_size}. Mean: {pressures.mean():.6f}"
    )


# =============================================================================
# DFSPH (Divergence-Free SPH) Pressure Solver Tests
# =============================================================================


# TODO: Add more tests for validating the underlying physical behavior of DFSPH.
@pytest.mark.required
def test_dfsph_simulation_builds_and_runs(show_viewer):
    """
    Test that DFSPH pressure solver builds and runs without errors.

    This is a basic smoke test to verify that the DFSPH solver is properly
    initialized and can execute simulation steps without crashing.
    """
    scene, liquid = create_sph_scene(show_viewer, pressure_solver="DFSPH")

    assert scene.sph_solver is not None
    assert scene.sph_solver.n_particles > 0

    # Run a few simulation steps
    for _ in range(3):
        scene.step()

    pos = liquid.get_particles_pos()
    assert pos.shape[0] == scene.sph_solver.n_particles
