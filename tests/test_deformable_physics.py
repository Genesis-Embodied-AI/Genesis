import numpy as np
import pytest
import torch

import genesis as gs

from .utils import assert_allclose


pytestmark = [
    pytest.mark.field_only,
]


@pytest.mark.required
@pytest.mark.parametrize("n_envs", [0, 2])
@pytest.mark.parametrize("muscle_material", [gs.materials.MPM.Muscle, gs.materials.FEM.Muscle])
def test_muscle(n_envs, muscle_material, show_viewer):
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=5e-4,
            substeps=10,
            gravity=(0, 0, -9.8),
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(1.5, 0, 0.8),
            camera_lookat=(0.0, 0.0, 0.0),
            camera_fov=40,
        ),
        mpm_options=gs.options.MPMOptions(
            lower_bound=(-1.0, -1.0, -0.2),
            upper_bound=(1.0, 1.0, 1.0),
        ),
        vis_options=gs.options.VisOptions(
            show_world_frame=True,
            visualize_mpm_boundary=True,
        ),
        show_viewer=show_viewer,
        show_FPS=False,
    )

    scene.add_entity(
        morph=gs.morphs.Plane(),
        material=gs.materials.Rigid(
            coup_friction=5.0,
        ),
    )
    worm = scene.add_entity(
        morph=gs.morphs.Mesh(
            file="meshes/worm/worm.obj",
            pos=(0.3, 0.3, 0.001),
            scale=0.1,
            euler=(90, 0, 0),
        ),
        material=muscle_material(
            E=5e5,
            nu=0.45,
            rho=10000.0,
            n_groups=4,
        ),
        surface=gs.surfaces.Default(
            diffuse_texture=gs.textures.ImageTexture(
                image_path="meshes/worm/bdy_Base_Color.png",
            ),
        ),
    )
    if n_envs > 0:
        scene.build(n_envs=n_envs)
    else:
        scene.build()

    if isinstance(worm.material, gs.materials.MPM.Muscle):
        pos = worm.get_state().pos[0]
        n_units = worm.n_particles
    else:  # isinstance(worm.material, gs.materials.FEM.Muscle):
        pos = worm.get_state().pos[0, worm.get_el2v()].mean(1)
        n_units = worm.n_elements

    pos = pos.cpu().numpy()
    pos_max, pos_min = pos.max(0), pos.min(0)
    pos_range = pos_max - pos_min
    lu_thresh, fh_thresh = 0.3, 0.6
    muscle_group = np.zeros((n_units,), dtype=int)
    mask_upper = pos[:, 2] > (pos_min[2] + pos_range[2] * lu_thresh)
    mask_fore = pos[:, 1] < (pos_min[1] + pos_range[1] * fh_thresh)
    muscle_group[mask_upper & mask_fore] = 0  # upper fore body
    muscle_group[mask_upper & ~mask_fore] = 1  # upper hind body
    muscle_group[~mask_upper & mask_fore] = 2  # lower fore body
    muscle_group[~mask_upper & ~mask_fore] = 3  # lower hind body

    muscle_direction = np.array([[0.0, 1.0, 0.0]] * n_units, dtype=gs.np_float)

    worm.set_muscle(
        muscle_group=muscle_group,
        muscle_direction=muscle_direction,
    )

    scene.reset()
    for i in range(200):
        actuation = np.array([0.0, 0.0, 0.0, 1.0 * (0.5 + np.sin(0.005 * np.pi * i))])
        if n_envs > 1:
            actuation = np.tile(actuation, (n_envs, 1))
        worm.set_actuation(actuation)
        scene.step()


@pytest.mark.required
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
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(3.5, 0.0, 2.5),
            camera_lookat=(0.0, 0.0, 0.5),
            camera_fov=40,
        ),
        vis_options=gs.options.VisOptions(
            rendered_envs_idx=[1],
        ),
        show_viewer=show_viewer,
    )

    plane = scene.add_entity(
        morph=gs.morphs.Plane(),
        material=gs.materials.Rigid(
            needs_coup=True,
            coup_friction=0.0,
        ),
    )
    cloth = scene.add_entity(
        morph=gs.morphs.Mesh(
            file="meshes/cloth.obj",
            scale=0.6,
            pos=(0.0, 0.8, 0.3),
            euler=(180.0, 0.0, 0.0),
        ),
        material=gs.materials.PBD.Cloth(),
        surface=gs.surfaces.Default(
            color=(0.2, 0.4, 0.8, 1.0),
        ),
    )
    water = scene.add_entity(
        morph=gs.morphs.Box(
            pos=(0.15, 0.15, 0.22),
            size=(0.25, 0.25, 0.4),
        ),
        material=gs.materials.SPH.Liquid(),
        surface=gs.surfaces.Default(
            color=(0.2, 0.6, 1.0, 1.0),
        ),
    )
    mpm_cube = scene.add_entity(
        morph=gs.morphs.Box(
            pos=(0.6, 0, 0.1),
            size=(0.1, 0.1, 0.1),
        ),
        material=gs.materials.MPM.Elastic(rho=200),
        surface=gs.surfaces.Default(
            color=(0.9, 0.8, 0.2, 1.0),
        ),
    )

    entity_fem = scene.add_entity(
        morph=gs.morphs.Box(
            pos=(0.8, 0.8, 0.1),
            size=(0.1, 0.1, 0.1),
        ),
        material=gs.materials.FEM.Elastic(
            E=3.0e4,
            nu=0.45,
            rho=1000.0,
            model="stable_neohookean",
        ),
    )
    scene.build(n_envs=2)

    for i in range(1500):
        scene.step()

    assert_allclose(cloth.get_particles_vel(), 0.0, atol=1e-5)
    assert_allclose(mpm_cube.get_particles_vel(), 0.0, atol=1e-4)
    assert_allclose(entity_fem._solver.get_state(0).vel, 0, atol=1e-3)
    assert_allclose(water.get_particles_vel(), 0.0, atol=5e-2)


@pytest.mark.required
@pytest.mark.parametrize("n_envs", [0, 2])
@pytest.mark.parametrize("material_type", [gs.materials.PBD.Cloth])
@pytest.mark.parametrize("backend", [gs.gpu])
def test_attach_cloth(n_envs, material_type, show_viewer, tol):
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=4e-3,
            substeps=10,
        ),
        show_viewer=show_viewer,
    )
    plane = scene.add_entity(
        morph=gs.morphs.Plane(),
    )
    cloth_1 = scene.add_entity(
        morph=gs.morphs.Mesh(
            file="meshes/cloth.obj",
            pos=(0, 2.0, 0.5),
            scale=1.0,
        ),
        material=material_type(),
        surface=gs.surfaces.Default(
            color=(1.0, 1.0, 0.2, 1.0),
            vis_mode="visual",
        ),
    )
    cloth_2 = scene.add_entity(
        morph=gs.morphs.Mesh(
            file="meshes/cloth.obj",
            pos=(0, 0, 0.1),
            scale=2.0,
        ),
        material=material_type(),
        surface=gs.surfaces.Default(
            color=(1.0, 1.0, 0.2, 1.0),
            vis_mode="visual",
        ),
    )
    scene.build(n_envs=n_envs)

    # Make sure that 'get_position', 'get_velocity' is working
    for cloth in (cloth_1, cloth_2):
        init_com = torch.tensor([2.0, 0.0, 1.0])
        cloth.set_position(init_com)
        cloth.process_input()
        poss = cloth.get_particles_pos()
        assert_allclose(poss.mean(dim=-2), init_com, tol=1e-2)
        poss += torch.tensor([0.5, -1.0, -0.5])
        vels = torch.rand_like(poss)
        cloth.set_position(poss)
        cloth.set_velocity(vels)
        assert_allclose(cloth.get_particles_vel(), 0.0, tol=tol)
        cloth.process_input()
        assert_allclose(cloth.get_particles_pos(), poss, tol=tol)
        assert_allclose(cloth.get_particles_vel(), vels, tol=tol)
    scene.reset()

    # Simulate for a while
    for i in range(40):
        scene.step()

    # Make sure that the cloth is landing perfectly vertically and laying on the ground without moving
    poss = cloth_2.get_particles_pos()
    assert_allclose(poss[..., :2], cloth_2._mesh.verts[..., :2], tol=tol)
    assert_allclose(poss[..., 2], cloth_2._particle_size / 2, tol=tol)
    vels = cloth_2.get_particles_vel()
    assert_allclose(vels, 0.0, tol=tol)

    # Attach top-left corner and simulate for a while
    particle_idx = cloth_2.find_closest_particle((-1, -1, 0))
    particle_pos_ref = (-0.5, -0.5, 0.05)
    cloth_2.set_particles_pos(particle_pos_ref, particle_idx)
    for i in range(60):
        scene.step()

    # Make sure that the corner is at the target position, some points are still on the ground, and none are moving
    poss = cloth_2.get_particles_pos()
    if scene.n_envs > 0:
        particle_pos = poss[torch.arange(scene.n_envs), particle_idx]
    else:
        particle_pos = poss[particle_idx]
    assert_allclose(particle_pos, particle_pos_ref, tol=tol)
    assert_allclose(poss[..., 2].min(dim=-1).values, cloth_2._particle_size / 2, tol=tol)
    vels = cloth_2.get_particles_vel()
    assert_allclose(vels[..., 2].mean(dim=-1), 0.0, tol=1e-3)

    # Release cloth
    cloth_2.release_particle(particle_idx)
    for i in range(30):
        scene.step()

    # Make sure that the cloth is laying on the ground without moving
    poss = cloth_2.get_particles_pos()
    assert_allclose(poss[..., 2], cloth_2._particle_size / 2, tol=tol)
    assert -0.6 < poss[..., :1].min() and poss[..., :1].max() < 0.6
    vels = cloth_2.get_particles_vel()
    assert_allclose(vels[..., 2].mean(dim=-1), 0.0, tol=tol)
