import numpy as np
import pytest

import genesis as gs
from genesis.utils.geom import R_to_xyz

from .utils import assert_allclose, get_hf_dataset

try:
    import uipc
except ImportError:
    pytest.skip("IPC Coupler is not supported because 'uipc' module is not available.", allow_module_level=True)

from uipc.backend import SceneVisitor
from uipc.geometry import SimplicialComplexSlot, apply_transform, merge


def get_cloth_vertex_positions(scene):
    """Extract cloth vertex positions from IPC scene.

    Returns an (N, 3) array of cloth vertex positions, or None if no cloth geometry is found.
    """
    visitor = SceneVisitor(scene.sim.coupler._ipc_scene)
    for geo_slot in visitor.geometries():
        if isinstance(geo_slot, SimplicialComplexSlot):
            geo = geo_slot.geometry()
            if geo.dim() == 2:  # Cloth is 2D shell
                proc_geo = geo
                if geo.instances().size() >= 1:
                    proc_geo = merge(apply_transform(geo))
                positions = proc_geo.positions().view().reshape(-1, 3)
                return positions  # Shape: (num_vertices, 3)
    return None


@pytest.mark.required
@pytest.mark.parametrize("n_envs", [0, 2])
def test_ipc_cloth(n_envs, show_viewer):
    """Test IPC cloth simulation with gravity physics validation.

    This test validates:
    1. Basic cloth, rigid, and soft body coupling
    2. Free fall physics using kinematic equations:
       - dx_{n+1} = v_n * dt + g * dt^2
       - v_{n+1} = (x_{n+1} - x_n) / dt
    """
    dt = 2e-3
    g = 9.8

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=dt,
            gravity=(0.0, 0.0, -g),
        ),
        coupler_options=gs.options.IPCCouplerOptions(
            contact_d_hat=0.01,
        ),
        show_viewer=show_viewer,
    )

    asset_path = get_hf_dataset(pattern="IPC/grid20x20.obj")
    scene.add_entity(
        morph=gs.morphs.Mesh(
            file=f"{asset_path}/IPC/grid20x20.obj",
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
            coupling_mode="two_way_soft_constraint",
        ),
        surface=gs.surfaces.Plastic(
            color=(0.8, 0.3, 0.2, 0.8),
        ),
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

    # Get initial state (vertex 0 of cloth)
    x_n = get_cloth_vertex_positions(scene)
    assert x_n is not None, "Could not retrieve cloth vertex positions"
    x_n = x_n[0, 2]  # Z position of vertex 0
    v_n = 0.0  # Initial velocity is zero

    # Run simulation and validate kinematic equations at each step
    num_validation_steps = 10

    for step in range(num_validation_steps):
        scene.step()

        # Get new position
        x_next = get_cloth_vertex_positions(scene)
        assert x_next is not None
        x_next = x_next[0, 2]

        # Expected displacement: dx = v_n * dt + 0.5 * g * dt^2
        expected_dx = v_n * dt - g * dt * dt  # Negative because gravity is -g

        # Validate displacement
        assert_allclose(x_next - x_n, expected_dx, rtol=0.01)

        # Calculate velocity: v_{n+1} = (x_{n+1} - x_n) / dt
        v_next = (x_next - x_n) / dt

        # Expected velocity: v_{n+1} = v_n + g * dt
        expected_v_next = v_n - g * dt  # Negative because gravity is -g
        assert_allclose(v_next, expected_v_next, rtol=0.01)

        # Update for next iteration
        x_n = x_next
        v_n = v_next


@pytest.mark.required
@pytest.mark.parametrize("n_envs", [0])
@pytest.mark.parametrize("coupling_type", ["two_way_soft_constraint", "external_articulation"])
def test_ipc_two_way_revolute(n_envs, coupling_type, show_viewer):
    """Test two-way coupling with revolute joint.

    Tests both coupling types:
    - two_way_soft_constraint: Soft constraint coupling for rigid links
    - external_articulation: Joint-level coupling with ExternalArticulationConstraint
    """
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=1e-3,
        ),
        coupler_options=gs.options.IPCCouplerOptions(
            contact_d_hat=0.01,
        ),
        show_viewer=show_viewer,
    )

    # Add simple two-cube robot with revolute joint
    robot = scene.add_entity(
        morph=gs.morphs.URDF(
            file="urdf/simple/two_cube_revolute.urdf",
            pos=(0, 0, 0.2),
            fixed=True,
        ),
        material=gs.materials.Rigid(coupling_mode=coupling_type),
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
                    # Compare positions
                    assert_allclose(genesis_transform[:3, 3], ipc_transform[:3, 3], atol=1e-3)

                    # Compare rotations
                    assert_allclose(R_to_xyz(genesis_transform[:3, :3]), R_to_xyz(ipc_transform[:3, :3]), atol=0.1)


@pytest.mark.required
@pytest.mark.parametrize("n_envs", [0])
@pytest.mark.parametrize("coupling_type", ["two_way_soft_constraint", "external_articulation"])
def test_ipc_two_way_prismatic(n_envs, coupling_type, show_viewer):
    """Test two-way coupling with prismatic joint.

    Tests both coupling types:
    - two_way_soft_constraint: Soft constraint coupling for rigid links
    - external_articulation: Joint-level coupling with ExternalArticulationConstraint
    """
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=1e-3,
        ),
        coupler_options=gs.options.IPCCouplerOptions(
            contact_d_hat=0.01,
        ),
        show_viewer=show_viewer,
    )

    # Add simple two-cube robot with prismatic joint
    robot = scene.add_entity(
        morph=gs.morphs.URDF(
            file="urdf/simple/two_cube_prismatic.urdf",
            pos=(0, 0, 0.2),
            fixed=True,
        ),
        material=gs.materials.Rigid(coupling_mode=coupling_type),
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
                    # Compare positions
                    assert_allclose(genesis_transform[:3, 3], ipc_transform[:3, 3], atol=1e-3)

                    # Compare rotations
                    assert_allclose(R_to_xyz(genesis_transform[:3, :3]), R_to_xyz(ipc_transform[:3, :3]), atol=0.1)


@pytest.mark.required
@pytest.mark.parametrize("n_envs", [0])
def test_ipc_cloth_gravity_freefall(n_envs, show_viewer):
    """Test cloth free fall physics validation.

    This test validates that cloth entities correctly follow free fall physics
    under gravity by checking the kinematic equation: displacement = 0.5 * g * t²

    The test tracks vertex 0 position and validates within 1% tolerance.
    """
    # Physics parameters
    dt = 2e-3  # 2ms timestep
    g = 9.8  # Gravity magnitude (m/s²)
    z0 = 2.0  # Initial height (m)
    num_steps = 50  # Total simulation steps (0.1s total)

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=dt,
            gravity=(0.0, 0.0, -g),
        ),
        coupler_options=gs.options.IPCCouplerOptions(
            contact_d_hat=0.01,
        ),
        show_viewer=show_viewer,
    )

    # NO ground plane - pure free fall test

    # Get cloth mesh asset
    asset_path = get_hf_dataset(pattern="IPC/grid20x20.obj")

    # Create cloth at initial height
    cloth = scene.add_entity(
        morph=gs.morphs.Mesh(
            file=f"{asset_path}/IPC/grid20x20.obj",
            scale=2.0,
            pos=(0.0, 0.0, z0),
            euler=(0, 0, 0),
        ),
        material=gs.materials.FEM.Cloth(
            E=1e6,  # Young's modulus (Pa)
            nu=0.499,  # Poisson's ratio
            rho=200,  # Density (kg/m³)
            thickness=0.001,  # Shell thickness (m)
            bending_stiffness=50.0,  # Bending resistance
        ),
        surface=gs.surfaces.Plastic(
            color=(0.3, 0.5, 0.8, 1.0),
        ),
    )

    scene.build(n_envs=n_envs)

    # Get initial position of vertex 0
    initial_positions = get_cloth_vertex_positions(scene)
    assert initial_positions is not None, "Could not retrieve initial cloth vertex positions"
    z_initial = initial_positions[0, 2]  # Z-coordinate of vertex 0

    # Run simulation
    for _ in range(num_steps):
        scene.step()

    # Get final position of vertex 0
    final_positions = get_cloth_vertex_positions(scene)
    assert final_positions is not None, "Could not retrieve final cloth vertex positions"
    z_final = final_positions[0, 2]

    # Validate displacement: 0.5 * g * t * (t + dt)
    t_total = num_steps * dt
    actual_displacement = z_initial - z_final
    expected_displacement = 0.5 * g * t_total * (t_total + dt)

    assert_allclose(
        actual_displacement,
        expected_displacement,
        rtol=0.01,
        err_msg=f"Free fall validation: z_initial={z_initial:.6f}, z_final={z_final:.6f}",
    )
