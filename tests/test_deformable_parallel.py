import numpy as np
import genesis as gs
import pytest
import os


@pytest.mark.parametrize("backend", [gs.gpu])
def test_deformable_parallel(show_viewer):

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=2e-3,
            substeps=10,
        ),
        pbd_options=gs.options.PBDOptions(
            particle_size=1e-2,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(3.5, 0.0, 2.5),
            camera_lookat=(0.0, 0.0, 0.5),
            camera_fov=40,
        ),
        sph_options=gs.options.SPHOptions(
            lower_bound=(-0.03, -0.03, -0.08),
            upper_bound=(0.33, 0.33, 1.0),
        ),
        fem_options=gs.options.FEMOptions(
            damping=45.0,
        ),
        mpm_options=gs.options.MPMOptions(
            lower_bound=(0.5, -0.1, -0.05),
            upper_bound=(0.7, 0.1, 0.3),
        ),
        vis_options=gs.options.VisOptions(
            rendered_envs_idx=[1],
        ),
        show_viewer=show_viewer,
    )

    ########################## entities ##########################
    frictionless_rigid = gs.materials.Rigid(needs_coup=True, coup_friction=0.0)

    plane = scene.add_entity(
        material=frictionless_rigid,
        morph=gs.morphs.Plane(),
    )

    # gs_root = os.path.dirname(os.path.abspath(__file__))
    # path_cloth = os.path.join(gs_root, "meshes", "cloth.obj")
    path_cloth = "meshes/cloth.obj"

    # pbd
    cloth = scene.add_entity(
        material=gs.materials.PBD.Cloth(),
        morph=gs.morphs.Mesh(
            file=path_cloth,
            scale=0.6,
            pos=(0.0, 0.8, 0.3),
            euler=(180.0, 0.0, 0.0),
        ),
        surface=gs.surfaces.Default(
            color=(0.2, 0.4, 0.8, 1.0),
        ),
    )

    # sph
    water = scene.add_entity(
        material=gs.materials.SPH.Liquid(),
        morph=gs.morphs.Box(
            pos=(0.15, 0.15, 0.22),
            size=(0.25, 0.25, 0.4),
        ),
        surface=gs.surfaces.Default(
            color=(0.2, 0.6, 1.0, 1.0),
            vis_mode="particle",
        ),
    )

    # mpm
    mpm_cube = scene.add_entity(
        material=gs.materials.MPM.Elastic(rho=200),
        morph=gs.morphs.Box(
            pos=(0.6, 0, 0.1),
            size=(0.1, 0.1, 0.1),
        ),
        surface=gs.surfaces.Default(
            color=(0.9, 0.8, 0.2, 1.0),
            vis_mode="particle",
        ),
    )

    # fem
    E, nu = 3.0e4, 0.45
    rho = 1000.0
    eneity_fem = scene.add_entity(
        morph=gs.morphs.Box(
            pos=(0.8, 0.8, 0.1),
            size=(0.1, 0.1, 0.1),
        ),
        material=gs.materials.FEM.Elastic(
            E=E,
            nu=nu,
            rho=rho,
            model="stable_neohooken",
        ),
    )

    ########################## build ##########################
    scene.build(n_envs=4)

    horizon = 2000

    for i in range(horizon):
        scene.step()

    # speed is around 0
    vel = cloth._solver.get_state(0).vel.cpu().numpy()
    np.testing.assert_allclose(vel, 0, atol=1e-2)

    vel = water._solver.get_state(0).vel.cpu().numpy()
    np.testing.assert_allclose(vel, 0, atol=2e-2)

    vel = mpm_cube._solver.get_state(0).vel.cpu().numpy()
    np.testing.assert_allclose(vel, 0, atol=1e-2)

    vel = eneity_fem._solver.get_state(0).vel.cpu().numpy()
    np.testing.assert_allclose(vel, 0, atol=1e-2)
