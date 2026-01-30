import pytest

import genesis as gs

from .utils import get_hf_dataset

try:
    import uipc
except ImportError:
    pytest.skip("IPC Coupler is not supported because 'uipc' module is not available.", allow_module_level=True)


@pytest.mark.required
@pytest.mark.parametrize("n_envs", [0, 2])
def test_ipc_cloth(n_envs, show_viewer):
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=2e-3,
        ),
        coupler_options=gs.options.IPCCouplerOptions(
            contact_d_hat=0.01,
            contact_friction_mu=0.3,
            IPC_self_contact=False,
            two_way_coupling=True,
            disable_genesis_contact=True,
            enable_ipc_gui=False,
        ),
        show_viewer=show_viewer,
    )

    asset_path = get_hf_dataset(pattern="grid20x20.obj")
    scene.add_entity(
        morph=gs.morphs.Mesh(
            file=f"{asset_path}/grid20x20.obj",
            scale=2.0,
            pos=(0.0, 0.0, 1.5),
            euler=(0, 0, 0),
        ),
        material=gs.materials.FEM.Cloth(
            E=1e6,
            nu=0.499,
            rho=200,
            thickness=0.001,
            bending_stiffness=50.0,
        ),
        surface=gs.surfaces.Plastic(
            color=(0.3, 0.5, 0.8, 1.0),
            double_sided=True,
        ),
    )
    box = scene.add_entity(
        morph=gs.morphs.Box(
            size=(0.2, 0.2, 0.2),
            pos=(0, 0, 0.3),
        ),
        material=gs.materials.Rigid(
            rho=500,
            friction=0.3,
        ),
        surface=gs.surfaces.Plastic(
            color=(0.8, 0.3, 0.2, 0.8),
        ),
    )
    scene.sim.coupler.set_entity_coupling_type(
        entity=box,
        coupling_type="two_way_soft_constraint",
    )
    soft_ball = scene.add_entity(
        morph=gs.morphs.Sphere(
            radius=0.08,
            pos=(0.5, 0.0, 0.1),
        ),
        material=gs.materials.FEM.Elastic(
            E=1.0e3,
            nu=0.3,
            rho=1000.0,
            model="stable_neohookean",
        ),
        surface=gs.surfaces.Plastic(
            color=(0.2, 0.8, 0.3, 0.8),
        ),
    )

    scene.build(n_envs=n_envs)

    scene.step()


@pytest.mark.required
@pytest.mark.parametrize("n_envs", [0, 2])
def test_ipc_two_way_revolute(n_envs, show_viewer):
    """Test two-way coupling with revolute joint."""
    import numpy as np

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=1e-3,
        ),
        coupler_options=gs.options.IPCCouplerOptions(
            contact_d_hat=0.01,
            contact_friction_mu=0.3,
            IPC_self_contact=False,
            two_way_coupling=True,
            disable_genesis_contact=True,
            enable_ipc_gui=False,
        ),
        show_viewer=show_viewer,
    )

    # Add ground plane
    scene.add_entity(gs.morphs.Plane())

    # Add simple two-cube robot with revolute joint
    robot = scene.add_entity(
        morph=gs.morphs.URDF(
            file="urdf/simple/two_cube_revolute.urdf",
            pos=(0, 0, 0.2),
            fixed=True,
        ),
    )

    # Set two-way coupling for the robot
    scene.sim.coupler.set_entity_coupling_type(
        entity=robot,
        coupling_type="two_way_soft_constraint",
    )

    scene.build(n_envs=n_envs)

    # Run simulation with oscillating motion
    max_steps = 100
    omega = 2.0 * np.pi  # 1 Hz oscillation
    dt = scene.sim_options.dt

    for i in range(max_steps):
        t = i * dt
        # Apply sinusoidal target position to revolute joint
        target_qpos = 0.5 * np.sin(omega * t)
        robot.set_dofs_position([target_qpos], zero_velocity=False)

        scene.step()

        # After some warmup steps, check transform consistency
        if i > 50:
            # Get transforms from abd_data_by_link
            link_idx = 1  # cube link (moving part)
            env_idx = 0
            if (
                hasattr(scene.sim.coupler, "abd_data_by_link")
                and link_idx in scene.sim.coupler.abd_data_by_link
                and env_idx in scene.sim.coupler.abd_data_by_link[link_idx]
            ):
                abd_data = scene.sim.coupler.abd_data_by_link[link_idx][env_idx]
                genesis_transform = abd_data["aim_transform"]
                ipc_transform = abd_data["transform"]

                if genesis_transform is not None and ipc_transform is not None:
                    genesis_pos = genesis_transform[:3, 3]
                    ipc_pos = ipc_transform[:3, 3]

                    # Compare positions
                    pos_diff = np.linalg.norm(genesis_pos - ipc_pos)
                    assert pos_diff < 0.1, f"Position difference too large: {pos_diff}"


@pytest.mark.required
@pytest.mark.parametrize("n_envs", [0, 2])
def test_ipc_two_way_prismatic(n_envs, show_viewer):
    """Test two-way coupling with prismatic joint."""
    import numpy as np

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=1e-3,
        ),
        coupler_options=gs.options.IPCCouplerOptions(
            contact_d_hat=0.01,
            contact_friction_mu=0.3,
            IPC_self_contact=False,
            two_way_coupling=True,
            disable_genesis_contact=True,
            enable_ipc_gui=False,
        ),
        show_viewer=show_viewer,
    )

    # Add ground plane
    scene.add_entity(gs.morphs.Plane())

    # Add simple two-cube robot with prismatic joint
    robot = scene.add_entity(
        morph=gs.morphs.URDF(
            file="urdf/simple/two_cube_prismatic.urdf",
            pos=(0, 0, 0.2),
            fixed=True,
        ),
    )

    # Set two-way coupling for the robot
    scene.sim.coupler.set_entity_coupling_type(
        entity=robot,
        coupling_type="two_way_soft_constraint",
    )

    scene.build(n_envs=n_envs)

    # Run simulation with oscillating vertical motion on slider
    max_steps = 100
    omega = 2.0 * np.pi  # 1 Hz oscillation
    dt = scene.sim_options.dt

    for i in range(max_steps):
        t = i * dt
        # Apply sinusoidal target position to prismatic joint
        target_qpos = 0.15 + 0.1 * np.sin(omega * t)  # Oscillate between 0.05 and 0.25
        robot.set_dofs_position([target_qpos], zero_velocity=False)

        scene.step()

        # After some warmup steps, check transform consistency
        if i > 50:
            # Get transforms from abd_data_by_link for slider
            link_idx = 1  # slider link (moving part)
            env_idx = 0
            if (
                hasattr(scene.sim.coupler, "abd_data_by_link")
                and link_idx in scene.sim.coupler.abd_data_by_link
                and env_idx in scene.sim.coupler.abd_data_by_link[link_idx]
            ):
                abd_data = scene.sim.coupler.abd_data_by_link[link_idx][env_idx]
                genesis_transform = abd_data["aim_transform"]
                ipc_transform = abd_data["transform"]

                if genesis_transform is not None and ipc_transform is not None:
                    genesis_pos = genesis_transform[:3, 3]
                    ipc_pos = ipc_transform[:3, 3]

                    # Compare positions (mainly check z-axis for prismatic)
                    pos_diff = np.linalg.norm(genesis_pos - ipc_pos)
                    assert pos_diff < 0.1, f"Position difference too large: {pos_diff}"
