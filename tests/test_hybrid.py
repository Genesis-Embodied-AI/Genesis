import numpy as np
import pytest

import genesis as gs

from .utils import assert_allclose


pytestmark = [
    pytest.mark.field_only,
]


@pytest.mark.required
def test_rigid_mpm_muscle(show_viewer):
    ball_pos_init = (0.8, 0.6, 0.12)

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=3e-3,
            substeps=10,
        ),
        rigid_options=gs.options.RigidOptions(
            gravity=(0, 0, -9.8),
            constraint_timeconst=0.02,
        ),
        mpm_options=gs.options.MPMOptions(
            lower_bound=(0.0, 0.0, -0.2),
            upper_bound=(1.0, 1.0, 1.0),
            gravity=(0.0, 0.0, 0.0),  # mimic gravity compensation
            enable_CPIC=True,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(1.5, 1.3, 0.5),
            camera_lookat=(0.0, 0.0, 0.0),
            camera_fov=40,
        ),
        show_viewer=show_viewer,
        show_FPS=False,
    )

    scene.add_entity(morph=gs.morphs.Plane())
    robot = scene.add_entity(
        morph=gs.morphs.URDF(
            file="urdf/simple/two_link_arm.urdf",
            pos=(0.5, 0.5, 0.3),
            euler=(0.0, 0.0, 0.0),
            scale=0.2,
            fixed=True,
        ),
        material=gs.materials.Hybrid(
            mat_rigid=gs.materials.Rigid(
                gravity_compensation=1.0,
            ),
            mat_soft=gs.materials.MPM.Muscle(
                E=1e4,
                nu=0.45,
                rho=1000.0,
                model="neohooken",
            ),
            thickness=0.05,
            damping=1000.0,
        ),
    )
    ball = scene.add_entity(
        morph=gs.morphs.Sphere(
            pos=ball_pos_init,
            radius=0.12,
        ),
        material=gs.materials.Rigid(rho=1000, friction=0.5),
    )
    scene.build()

    scene.reset()
    for i in range(370):
        robot.control_dofs_velocity(np.array([1.0 * np.sin(2 * np.pi * i * 0.001)] * robot.n_dofs))
        scene.step()

    with np.testing.assert_raises(AssertionError):
        assert_allclose(ball.get_pos(), ball_pos_init, atol=1e-2)


@pytest.mark.required
@pytest.mark.parametrize(
    "material_type",
    [
        gs.materials.PBD.Liquid,
        gs.materials.SPH.Liquid,
        gs.materials.MPM.Liquid,
        gs.materials.MPM.Sand,
        gs.materials.MPM.Snow,
        gs.materials.MPM.Elastic,  # This makes little sense but nothing prevents doing this
    ],
)
def test_fluid_emitter(material_type, show_viewer):
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=4e-3,
            substeps=10,
        ),
        mpm_options=gs.options.MPMOptions(
            lower_bound=(0.0, -1.5, 0.0),
            upper_bound=(1.0, 1.5, 4.0),
        ),
        sph_options=gs.options.SPHOptions(
            particle_size=0.02,
        ),
        pbd_options=gs.options.PBDOptions(
            particle_size=0.02,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(5.5, 6.5, 3.2),
            camera_lookat=(0.5, 1.5, 1.5),
            camera_fov=35,
            max_FPS=120,
        ),
        vis_options=gs.options.VisOptions(
            rendered_envs_idx=[0],
        ),
        show_viewer=show_viewer,
    )
    plane = scene.add_entity(gs.morphs.Plane())
    wheel = scene.add_entity(
        morph=gs.morphs.URDF(
            file="urdf/wheel/fancy_wheel.urdf",
            pos=(0.5, 0.25, 1.6),
            euler=(0, 0, 0),
            fixed=True,
            convexify=False,
        ),
    )
    emitter = scene.add_emitter(
        material=material_type(),
        max_particles=100000,
        surface=gs.surfaces.Glass(
            color=(0.7, 0.85, 1.0, 0.7),
        ),
    )
    scene.build(n_envs=2)
    scene.step()
