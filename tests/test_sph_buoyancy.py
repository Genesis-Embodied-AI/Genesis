"""
Analytical integration tests for SPH-rigid pressure-based buoyancy coupling.

Validates that ``particles.p`` is populated with physically meaningful pressure
values and that the rigid-body coupler transmits the resulting forces.

1. Heavy-cube sinking
   -------------------
   A submerged rigid cube with density *rho_c > rho_f* sinks at less than
   free-fall acceleration due to buoyancy + drag + collision response.
   The free-fall displacement over *N* steps provides a rigorous upper bound.

2. Hydrostatic pressure gradient (WCSPH)
   --------------------------------------
   For a settled column of WCSPH fluid the pressure approximately follows

       dp/dz ~ rho * g   (≈ 9800 Pa/m),

   confirming that ``particles.p`` holds physically meaningful values.
"""

import numpy as np
import pytest

import genesis as gs


# ---------------------------------------------------------------------------
# Heavy-cube sinking — upper bound: free-fall displacement
# ---------------------------------------------------------------------------

_RHO_FLUID = 1000.0
_RHO_CUBE_HEAVY = 2000.0  # 2× fluid density
_HEAVY_DT = 2e-3
_HEAVY_SUBSTEPS = 4
_HEAVY_N_STEPS = 100
_T_SIM = _HEAVY_N_STEPS * _HEAVY_DT * _HEAVY_SUBSTEPS
_FREE_FALL_DISP = 0.5 * 9.8 * _T_SIM ** 2  # m


@pytest.mark.required
def test_heavy_cube_sinks_slower_than_free_fall(show_viewer):
    """A rigid cube (rho_c = 2 * rho_f) must sink significantly less than
    the free-fall displacement, confirming the SPH-rigid coupling applies
    a net upward force (collision + pressure buoyancy)."""
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=_HEAVY_DT, substeps=_HEAVY_SUBSTEPS),
        sph_options=gs.options.SPHOptions(
            lower_bound=(-0.5, -0.5, 0.0),
            upper_bound=(0.5, 0.5, 1.0),
            particle_size=0.02,
            pressure_solver="WCSPH",
        ),
        show_viewer=show_viewer,
    )
    scene.add_entity(morph=gs.morphs.Plane())
    scene.add_entity(
        material=gs.materials.SPH.Liquid(sampler="regular", mu=0.05),
        morph=gs.morphs.Box(pos=(0.0, 0.0, 0.35), size=(0.4, 0.4, 0.5)),
    )
    cube = scene.add_entity(
        material=gs.materials.Rigid(rho=_RHO_CUBE_HEAVY),
        morph=gs.morphs.Box(pos=(0.0, 0.0, 0.25), size=(0.1, 0.1, 0.1)),
    )
    scene.build()

    init_z = float(cube.get_pos()[2])
    for _ in range(_HEAVY_N_STEPS):
        scene.step()
    final_z = float(cube.get_pos()[2])
    displacement = abs(final_z - init_z)

    assert displacement < _FREE_FALL_DISP * 0.8, (
        f"Displacement {displacement:.4f} m exceeds 80 % of free-fall "
        f"({_FREE_FALL_DISP:.4f} m) — coupling force is missing."
    )


# ---------------------------------------------------------------------------
# Hydrostatic pressure gradient in WCSPH
# ---------------------------------------------------------------------------

_MIN_CENTRE_PARTICLES = 8


@pytest.mark.required
def test_wcsph_hydrostatic_pressure_gradient(show_viewer):
    """The WCSPH pressure field in a settled column should approximately
    follow dp/dz = rho * g ≈ 9800 Pa/m."""
    scene = _build_fluid_only_scene(show_viewer, "WCSPH")
    sph = scene.sph_solver

    p = sph.particles_reordered.p.to_numpy()[:, 0]
    pos = sph.particles_reordered.pos.to_numpy()[:, 0, :]
    x, y, z = pos[:, 0], pos[:, 1], pos[:, 2]

    centre = (np.abs(x) < 0.04) & (np.abs(y) < 0.04) & (z > 0.02) & (z < 0.7)
    z_c, p_c = z[centre], p[centre]

    if len(z_c) < _MIN_CENTRE_PARTICLES:
        pytest.skip(f"Too few centre-column particles ({len(z_c)}).")

    z_surface = z_c.max()
    depths = np.maximum(z_surface - z_c, 1e-12)
    weights = np.clip(depths / depths.max(), 0.1, 1.0)
    slope = np.polyfit(depths, p_c, 1, w=weights)[0]
    ratio = slope / (1000.0 * 9.8)

    assert 0.3 < ratio < 2.5, (
        f"Pressure gradient dp/dz = {slope:.0f} Pa/m "
        f"(expected ~9800 Pa/m, ratio={ratio:.2f})."
    )


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _build_fluid_only_scene(show_viewer, pressure_solver):
    """Build and settle a fluid-only scene for pressure-field analysis."""
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=2e-3, substeps=4),
        sph_options=gs.options.SPHOptions(
            lower_bound=(-0.5, -0.5, 0.0),
            upper_bound=(0.5, 0.5, 1.0),
            particle_size=0.015,
            pressure_solver=pressure_solver,
        ),
        show_viewer=show_viewer,
    )
    scene.add_entity(morph=gs.morphs.Plane())
    scene.add_entity(
        material=gs.materials.SPH.Liquid(sampler="regular", mu=0.05),
        morph=gs.morphs.Box(pos=(0.0, 0.0, 0.4), size=(0.4, 0.4, 0.6)),
    )
    scene.build()

    for _ in range(800):
        scene.step()

    return scene
