import pytest

import genesis as gs


@pytest.mark.required
@pytest.mark.parametrize("n_envs", [0, 2])
@pytest.mark.parametrize("pressure_solver", ["WCSPH", "DFSPH"])
def test_rigid_flotation_follows_density_ratio(n_envs, pressure_solver, show_viewer):
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=1e-2,
            substeps=10,
            gravity=(0.0, 0.0, -9.81),
        ),
        sph_options=gs.options.SPHOptions(
            lower_bound=(-0.25, -0.25, 0.0),
            upper_bound=(0.25, 0.25, 1.0),
            particle_size=0.02,
            pressure_solver=pressure_solver,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(1.2, 0.0, 0.7),
            camera_lookat=(0.0, 0.0, 0.3),
            camera_fov=40,
        ),
        show_viewer=show_viewer,
    )
    scene.add_entity(
        morph=gs.morphs.Plane(),
    )
    scene.add_entity(
        material=gs.materials.SPH.Liquid(sampler="regular"),
        morph=gs.morphs.Box(
            pos=(0.0, 0.0, 0.25),
            size=(0.5, 0.5, 0.5),
        ),
    )
    light_ball = scene.add_entity(
        material=gs.materials.Rigid(rho=200.0),
        morph=gs.morphs.Sphere(
            pos=(-0.12, 0.0, 0.25),
            radius=0.06,
        ),
    )
    heavy_ball = scene.add_entity(
        material=gs.materials.Rigid(rho=1500.0),
        morph=gs.morphs.Sphere(
            pos=(0.12, 0.0, 0.25),
            radius=0.06,
        ),
    )
    scene.build(n_envs=n_envs)

    for _ in range(50):
        scene.step()

    # The static fluid pressure must push the light ball (rho well below the fluid rest density) up toward the
    # surface, while the heavy ball (rho well above) must keep sinking: flotation discriminates on the density
    # ratio, which is Archimedes' principle.
    light_z = light_ball.get_pos()[..., 2]
    heavy_z = heavy_ball.get_pos()[..., 2]
    assert (light_z > 0.28).all(), f"Light ball must rise under buoyancy, got z={light_z}"
    assert (heavy_z < 0.22).all(), f"Heavy ball must sink, got z={heavy_z}"
