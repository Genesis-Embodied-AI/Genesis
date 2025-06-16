import numpy as np
import pytest

import genesis as gs
from .utils import assert_allclose


@pytest.fixture(scope="session")
def fem_material_linear():
    """Fixture for common FEM linear material properties"""
    return gs.materials.FEM.Elastic()


@pytest.mark.parametrize("backend", [gs.cpu], "eps", [1e-7])
def test_sap_coupler_fem_floor(fem_material_linear, show_viewer):
    rot = R.from_euler("xyz", [30, 0, 0], degrees=True).as_matrix()

    camera_pos = np.array([3.5, 0.0, 0.0], dtype=np.float32)
    camera_lookat = (0, 0, 0.0)
    camera_fov = 40
    camera_up = np.array([0, 0, 1], dtype=np.float32) @ rot.T

    gravity = np.array([0, 0, -9.81], dtype=np.float32) @ rot.T

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=1 / 60,
            substeps=3,
            gravity=gravity,
        ),
        coupler_options=gs.options.SAPCouplerOptions(),
        fem_options=gs.options.FEMOptions(
            use_implicit_solver=True,
            n_newton_iterations=1,
            n_pcg_iterations=200,
            pcg_threshold=1e-6,
            damping_alpha=0.5,
            damping_beta=0.05,
        ),
        show_viewer=True,
        viewer_options=gs.options.ViewerOptions(
            camera_pos=camera_pos,
            camera_lookat=camera_lookat,
            camera_fov=camera_fov,
            camera_up=camera_up,
        ),
    )

    cam = scene.add_camera(
        res=(1280, 960),
        pos=camera_pos,
        lookat=camera_lookat,
        fov=camera_fov,
        GUI=False,
        up=camera_up,
    )

    scene.add_entity(
        material=gs.materials.FEM.Elastic(friction_mu=0.6),
        morph=gs.morphs.Mesh(
            file="./cube8.obj",
            scale=0.5,
            pos=np.array([0, 0, 0.55], dtype=np.float32),
            euler=(0, 0, 0),
        ),
        # morph=gs.morphs.Box(
        #     pos=(0.0, 0.0, 0.55),
        #     size=(0.5, 0.5, 0.5),
        #     euler=(0, 0, 0),
        # ),
    )
    # plane = scene.add_entity(
    #     material=gs.materials.Rigid(),
    #     morph=gs.morphs.URDF(
    #         file="urdf/plane/plane.urdf",
    #         fixed=True,
    #     ),
    # )
    scene.build(n_envs=1)

    cam.start_recording()
    for i in range(500):
        scene.step()
        cam.render()

    cam.stop_recording(save_to_filename="cube8_collision.mp4", fps=60)

    for entity in scene.entities:
        state = entity.get_state()
        vel = state.vel.detach().cpu().numpy()
        assert_allclose(vel, 0.0, atol=2e-3), f"Entity {entity.uid} velocity is not near zero."
        pos = state.pos.detach().cpu().numpy()
        min_pos_z = np.min(pos[..., 2])
        assert_allclose(
            min_pos_z, 0.0, atol=5e-2
        ), f"Entity {entity.uid} minimum Z position {min_pos_z} is not close to 0.0."
