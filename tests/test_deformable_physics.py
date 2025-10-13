import math
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

    scene.get_state()
    for i in range(1500):
        scene.step()

    assert_allclose(cloth.get_particles_vel(), 0.0, atol=1e-5)
    assert_allclose(mpm_cube.get_particles_vel(), 0.0, atol=1e-4)
    assert_allclose(entity_fem._solver.get_state(0).vel, 0, atol=1e-3)
    assert_allclose(water.get_particles_vel(), 0.0, atol=5e-2)


@pytest.mark.required
def test_sf_solver(show_viewer):
    import gstaichi as ti

    res = 384
    orbit_tau = 0.2
    orbit_radius = 0.3
    orbit_radius_vel = 0.0

    jet_radius = 0.02

    sub_orbit_radius = 0.03
    sub_orbit_tau = 3.0

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=1e-2,
        ),
        sf_options=gs.options.SFOptions(
            res=res,
            solver_iters=200,
            decay=0.025,
        ),
        show_viewer=show_viewer,
    )

    @ti.data_oriented
    class Jet(object):
        def __init__(
            self,
            world_center,
            jet_radius,
            orbit_radius,
            orbit_radius_vel,
            orbit_init_degree,
            orbit_tau,
            sub_orbit_radius,
            sub_orbit_tau,
        ):
            self.world_center = ti.Vector(world_center)
            self.orbit_radius = orbit_radius
            self.orbit_radius_vel = orbit_radius_vel
            self.orbit_init_radian = math.radians(orbit_init_degree)
            self.orbit_tau = orbit_tau

            self.jet_radius = jet_radius

            self.num_sub_jets = 3
            self.sub_orbit_radian_delta = 2.0 * math.pi / self.num_sub_jets
            self.sub_orbit_radius = sub_orbit_radius
            self.sub_orbit_tau = sub_orbit_tau

        @ti.func
        def get_pos(self, t: float):
            rel_pos = ti.Vector([self.orbit_radius + t * self.orbit_radius_vel, 0.0, 0.0])
            rot_mat = ti.math.rot_by_axis(ti.Vector([0.0, 1.0, 0.0]), self.orbit_init_radian + t * self.orbit_tau)[
                :3, :3
            ]
            rel_pos = rot_mat @ rel_pos
            return rel_pos

        @ti.func
        def get_factor(self, i: int, j: int, k: int, dx: float, t: float):
            rel_pos = self.get_pos(t)
            tan_dir = self.get_tan_dir(t)
            ijk = ti.Vector([i, j, k], dt=gs.ti_float) * dx
            dist = 2 * self.jet_radius
            for q in ti.static(range(self.num_sub_jets)):
                jet_pos = ti.Vector([0.0, self.sub_orbit_radius, 0.0])
                rot_mat = ti.math.rot_by_axis(tan_dir, self.sub_orbit_radian_delta * q + self.sub_orbit_tau * t)[:3, :3]
                jet_pos = (rot_mat @ jet_pos) + self.world_center + rel_pos
                dist_q = (ijk - jet_pos).norm(gs.EPS)
                if dist_q < dist:
                    dist = dist_q
            factor = 0.0
            if dist < self.jet_radius:
                factor = 1.0
            return factor

        @ti.func
        def get_inward_dir(self, t: float):
            neg_pos = -self.get_pos(t)
            return neg_pos.normalized(gs.EPS)

        @ti.func
        def get_tan_dir(self, t: float):
            inward_dir = self.get_inward_dir(t)
            tan_rot_mat = ti.math.rot_by_axis(ti.Vector([0.0, 1.0, 0.0]), 0.0)[:3, :3]
            return tan_rot_mat @ inward_dir

    jet = [
        Jet(
            world_center=[0.5, 0.5, 0.5],
            orbit_radius=orbit_radius,
            orbit_radius_vel=orbit_radius_vel,
            orbit_init_degree=orbit_init_degree,
            orbit_tau=orbit_tau,
            sub_orbit_radius=sub_orbit_radius,
            jet_radius=jet_radius,
            sub_orbit_tau=sub_orbit_tau,
        )
        for orbit_init_degree in np.linspace(0, 360, 3, endpoint=False)
    ]
    scene.sim.solvers[-1].set_jets(jet)
    scene.build()
    scene.step()
