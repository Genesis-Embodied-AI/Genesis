import numpy as np
import pytest
import torch
import igl

import genesis as gs
from genesis.utils.misc import tensor_to_array

from .utils import assert_allclose, get_hf_dataset


def test_multiple_rigid_entities(show_viewer):
    """Test adding multiple rigid entities to the scene"""
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=1 / 60,
            substeps=2,
        ),
        coupler_options=gs.options.SAPCouplerOptions(
            pcg_threshold=1e-10,
            sap_convergence_atol=1e-10,
            sap_convergence_rtol=1e-10,
            linesearch_ftol=1e-10,
        ),
        show_viewer=show_viewer,
        show_FPS=False,
    )

    scale = 0.4
    box = scene.add_entity(
        morph=gs.morphs.Box(
            size=(scale, scale, scale),
            pos=(0.0, 0.0, 0.2),
        ),
        material=gs.materials.Rigid(),
    )

    asset_path = get_hf_dataset(pattern="heavy_three_joint_link.xml")
    link = scene.add_entity(
        gs.morphs.MJCF(
            file=f"{asset_path}/heavy_three_joint_link.xml",
            scale=0.5,
            pos=(-0.5, -0.5, 0.4),
        ),
    )

    # Build the scene
    scene.build()

    # Run simulation
    for _ in range(150):
        scene.step()

    box_center = box.get_dofs_position()[:3]
    assert_allclose(
        box_center,
        np.array([0.0, 0.0, 0.2], dtype=np.float32),
        atol=2e-1,
        err_msg=f"Box center {box_center} moved too much from initial position [0.0, 0.0, 0.2].",
    )
    assert_allclose(
        box_center[2],
        0.2,
        atol=1e-4,
        err_msg=f"Box center Z {box_center[2]} should be close to 0.2.",
    )
    link_center = link.get_dofs_position()[:3]
    assert_allclose(
        link_center,
        np.array([-0.5, -0.5, 0.039], dtype=np.float32),
        atol=5e-1,
        err_msg=f"Link center {link_center} moved too much from initial position [-0.5, -0.5, 0.039].",
    )
    assert_allclose(
        link_center[2],
        0.039,
        atol=1e-3,
        err_msg=f"Link center Z {link_center[2]} should be close to 0.039.",
    )


@pytest.mark.parametrize("precision", ["64"])
@pytest.mark.parametrize("backend", [gs.gpu])
def test_franka_panda_grasp_rigid_cube(show_viewer):
    """
    Test if the Franka Panda can successfully grasp the rigid cube.
    """
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=1.0 / 60,
            substeps=2,
        ),
        coupler_options=gs.options.SAPCouplerOptions(
            pcg_threshold=1e-10,
            sap_convergence_atol=1e-10,
            sap_convergence_rtol=1e-10,
            linesearch_ftol=1e-10,
        ),
        show_viewer=show_viewer,
        show_FPS=False,
    )

    friction = 1.0
    force = 1.0
    franka = scene.add_entity(
        gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"),
        material=gs.materials.Rigid(coup_friction=friction, friction=friction),
    )
    # Only allow finger contact to accelerate
    for geom in franka.geoms:
        if "finger" not in geom.link.name:
            geom._contype = 0
            geom._conaffinity = 0
    asset_path = get_hf_dataset(pattern="meshes/cube8.obj")
    cube = scene.add_entity(
        morph=gs.morphs.Mesh(
            file=f"{asset_path}/meshes/cube8.obj",
            scale=0.02,
            pos=(0.65, 0.0, 0.02),
        ),
        material=gs.materials.Rigid(rho=1000.0, friction=friction, coup_friction=friction),
    )

    scene.build()
    motors_dof = np.arange(7)
    fingers_dof = np.arange(7, 9)
    qpos = np.array([-1.0119, 1.5576, 1.3673, -1.6867, -1.5812, 1.7745, 1.4598, 0.04, 0.04])
    franka.set_qpos(qpos)
    scene.step()

    end_effector = franka.get_link("hand")
    qpos = franka.inverse_kinematics(
        link=end_effector,
        pos=np.array([0.65, 0.0, 0.135]),
        quat=np.array([0, 1, 0, 0]),
    )

    # grasp
    for i in range(15):
        franka.control_dofs_position(qpos[:-2], motors_dof)
        franka.control_dofs_force(np.array([-force, -force]), fingers_dof)
        scene.step()

    # lift
    qpos = franka.inverse_kinematics(
        link=end_effector,
        pos=np.array([0.65, 0.0, 0.185]),
        quat=np.array([0, 1, 0, 0]),
    )
    for i in range(50):
        franka.control_dofs_position(qpos[:-2], motors_dof)
        franka.control_dofs_force(np.array([-force, -force]), fingers_dof)
        scene.step()

    # hold
    for i in range(10):
        franka.control_dofs_position(qpos[:-2], motors_dof)
        franka.control_dofs_force(np.array([-force, -force]), fingers_dof)
        scene.step()
        if i == 0:
            old_pos = cube.get_dofs_position()[:3]
        if i == 9:
            new_pos = cube.get_dofs_position()[:3]
    assert_allclose(
        new_pos,
        old_pos,
        atol=1e-4,
        err_msg=f"Cube should be not moving much. Old pos: {old_pos}, new pos: {new_pos}.",
    )
