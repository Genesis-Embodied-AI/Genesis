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
    """Test IPC cloth simulation with gravity physics validation.

    This test validates:
    1. Basic cloth, rigid, and soft body coupling
    2. Free fall physics using kinematic equations:
       - dx_{n+1} = v_n * dt + g * dt^2
       - v_{n+1} = (x_{n+1} - x_n) / dt
    """
    import numpy as np
    from uipc.backend import SceneVisitor
    from uipc.geometry import SimplicialComplexSlot, apply_transform, merge

    def get_cloth_vertex_positions(scene):
        """Extract cloth vertex positions from IPC scene."""
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

    dt = 2e-3
    g = 9.8

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=dt,
            gravity=(0.0, 0.0, -g),
        ),
        coupler_options=gs.options.IPCCouplerOptions(
            dt=dt,
            gravity=(0.0, 0.0, -g),
            contact_d_hat=0.01,
            contact_friction_mu=0.3,
            IPC_self_contact=False,
            two_way_coupling=True,
            disable_genesis_contact=True,
            enable_ipc_gui=False,
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

    # Get initial state (vertex 0 of cloth)
    x_n = get_cloth_vertex_positions(scene)
    assert x_n is not None, "Could not retrieve cloth vertex positions"
    x_n = x_n[0, 2]  # Z position of vertex 0
    v_n = 0.0  # Initial velocity is zero

    # Run simulation and validate kinematic equations at each step
    num_validation_steps = 10
    tolerance = 0.01  # 1% tolerance for kinematic validation

    for step in range(num_validation_steps):
        scene.step()

        # Get new position
        x_next = get_cloth_vertex_positions(scene)
        assert x_next is not None
        x_next = x_next[0, 2]

        # Expected displacement: dx = v_n * dt + 0.5 * g * dt^2
        expected_dx = v_n * dt - g * dt * dt  # Negative because gravity is -g
        expected_x_next = x_n + expected_dx

        # Validate position
        pos_error = abs(x_next - expected_x_next) / abs(expected_dx) if abs(expected_dx) > 1e-6 else 0
        assert pos_error < tolerance, f"Step {step}: Position error {pos_error * 100:.2f}% exceeds tolerance"

        # Calculate velocity: v_{n+1} = (x_{n+1} - x_n) / dt
        v_next = (x_next - x_n) / dt

        # Expected velocity: v_{n+1} = v_n + g * dt
        expected_v_next = v_n - g * dt  # Negative because gravity is -g

        # Validate velocity
        vel_error = abs(v_next - expected_v_next) / abs(expected_v_next) if abs(expected_v_next) > 1e-6 else 0
        assert vel_error < tolerance, f"Step {step}: Velocity error {vel_error * 100:.2f}% exceeds tolerance"

        # Update for next iteration
        x_n = x_next
        v_n = v_next


@pytest.mark.required
@pytest.mark.parametrize("n_envs", [0])
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
                    assert pos_diff < 0.001, f"Position difference too large: {pos_diff}"

                    # Compare rotation (using rotation matrix to euler angles)
                    def rotmat_to_euler(R):
                        sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
                        singular = sy < 1e-6
                        if not singular:
                            x = np.arctan2(R[2, 1], R[2, 2])
                            y = np.arctan2(-R[2, 0], sy)
                            z = np.arctan2(R[1, 0], R[0, 0])
                        else:
                            x = np.arctan2(-R[1, 2], R[1, 1])
                            y = np.arctan2(-R[2, 0], sy)
                            z = 0
                        return np.array([x, y, z])

                    genesis_rot = genesis_transform[:3, :3]
                    ipc_rot = ipc_transform[:3, :3]
                    genesis_euler = rotmat_to_euler(genesis_rot)
                    ipc_euler = rotmat_to_euler(ipc_rot)
                    rot_diff = np.linalg.norm(genesis_euler - ipc_euler)
                    print(f"Step {i}: Rotation difference (rad) = {rot_diff:.6f}")
                    assert rot_diff < 0.1, f"Rotation difference too large: {rot_diff}"


@pytest.mark.required
@pytest.mark.parametrize("n_envs", [0])
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
                    assert pos_diff < 0.001, f"Position difference too large: {pos_diff}"

                    # Compare rotation (using rotation matrix to euler angles)
                    def rotmat_to_euler(R):
                        sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
                        singular = sy < 1e-6
                        if not singular:
                            x = np.arctan2(R[2, 1], R[2, 2])
                            y = np.arctan2(-R[2, 0], sy)
                            z = np.arctan2(R[1, 0], R[0, 0])
                        else:
                            x = np.arctan2(-R[1, 2], R[1, 1])
                            y = np.arctan2(-R[2, 0], sy)
                            z = 0
                        return np.array([x, y, z])

                    genesis_rot = genesis_transform[:3, :3]
                    ipc_rot = ipc_transform[:3, :3]
                    genesis_euler = rotmat_to_euler(genesis_rot)
                    ipc_euler = rotmat_to_euler(ipc_rot)
                    rot_diff = np.linalg.norm(genesis_euler - ipc_euler)
                    print(f"Step {i}: Rotation difference (rad) = {rot_diff:.6f}")
                    assert rot_diff < 0.1, f"Rotation difference too large: {rot_diff}"


@pytest.mark.required
@pytest.mark.parametrize("n_envs", [0])
def test_ipc_cloth_gravity_freefall(n_envs, show_viewer):
    """Test cloth free fall physics validation.

    This test validates that cloth entities correctly follow free fall physics
    under gravity by checking the kinematic equation: displacement = 0.5 * g * t²

    The test tracks vertex 0 position and validates within 1% tolerance.
    """
    import numpy as np
    from uipc.backend import SceneVisitor
    from uipc.geometry import SimplicialComplexSlot, apply_transform, merge

    def get_cloth_vertex_positions(scene):
        """Extract cloth vertex positions from IPC scene."""
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
            dt=dt,
            gravity=(0.0, 0.0, -g),
            contact_d_hat=0.01,
            contact_friction_mu=0.3,
            IPC_self_contact=False,
            two_way_coupling=True,
            disable_genesis_contact=True,
            enable_ipc_gui=False,
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
            double_sided=True,
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

    # Calculate displacement
    t_total = num_steps * dt
    actual_displacement = z_initial - z_final
    expected_displacement = 0.5 * g * t_total * (t_total + dt)

    # Calculate relative error
    relative_error = abs(actual_displacement - expected_displacement) / expected_displacement

    # Validation with 1% tolerance
    tolerance = 0.01
    print("\nFree fall validation:")
    print("  Initial Z:              {z_initial:.6f} m")
    print("  Final Z:                {z_final:.6f} m")
    print("  Actual displacement:    {actual_displacement:.6f} m")
    print("  Expected displacement:  {expected_displacement:.6f} m")
    print("  Relative error:         {relative_error * 100:.4f}%")

    assert relative_error < tolerance, (
        f"Physics validation failed: {relative_error * 100:.4f}% error (tolerance: {tolerance * 100}%)"
    )
