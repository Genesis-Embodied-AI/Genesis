import argparse
import logging

import numpy as np
import matplotlib.pyplot as plt

import genesis as gs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", default=False)
    args = parser.parse_args()

    # ==== Simulation Parameters (Configurable) ====
    dt = 1e-3
    test_time = 0.20  # seconds
    initial_cube_velocity = np.array([10.0, 0.0, 0.0, 100.0, 100.0, 100.0])  # [vx, vy, vz, wx, wy, wz]
    initial_momentum_reference_time = 0.02  # seconds - when to capture reference momentum for error calculation

    # Object types: 'rigid' or 'fem'
    cube_type = "rigid"  # 'rigid' or 'fem'
    blob_type = "rigid"  # 'rigid' or 'fem'
    # =============================================

    gs.init(backend=gs.gpu, logging_level=logging.DEBUG, performance_mode=True)
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=dt, gravity=(0.0, 0.0, 0.0)),
        coupler_options=gs.options.IPCCouplerOptions(
            dt=dt,
            gravity=(0.0, 0.0, 0.0),
            ipc_constraint_strength=(1, 1),  # (translation, rotation) strength ratios
            use_contact_proxy=False,
            enable_ipc_gui=args.vis,
        ),
        show_viewer=args.vis,
    )

    blob_rho = 500.0
    cube_rho = 500.0
    blob_radius = 0.1
    cube_size = (0.1, 0.1, 0.1)

    # Blob (sphere) - configurable type
    if blob_type == "rigid":
        blob = scene.add_entity(
            morph=gs.morphs.Sphere(pos=(0.6, 0.00, 0.00), radius=blob_radius),
            material=gs.materials.Rigid(rho=blob_rho, friction=0.3),
            surface=gs.surfaces.Plastic(color=(0.2, 0.2, 0.8, 0.8)),
        )
    else:  # fem
        blob = scene.add_entity(
            morph=gs.morphs.Sphere(pos=(0.6, 0.00, 0.00), radius=blob_radius),
            material=gs.materials.FEM.Elastic(E=1.0e5, nu=0.45, rho=blob_rho, model="stable_neohookean"),
            surface=gs.surfaces.Plastic(color=(0.2, 0.2, 0.8, 0.8)),
        )

    # Cube - configurable type
    if cube_type == "rigid":
        cube = scene.add_entity(
            morph=gs.morphs.Box(pos=(0.0, 0.03, 0.02), size=cube_size, euler=(0, 0, 0)),
            material=gs.materials.Rigid(rho=cube_rho, friction=0.3),
            surface=gs.surfaces.Plastic(color=(0.8, 0.2, 0.2, 0.8)),
        )
    else:  # fem
        cube = scene.add_entity(
            morph=gs.morphs.Box(pos=(0.0, 0.03, 0.02), size=cube_size, euler=(0, 0, 0)),
            material=gs.materials.FEM.Elastic(E=1.0e5, nu=0.45, rho=cube_rho, model="stable_neohookean"),
            surface=gs.surfaces.Plastic(color=(0.8, 0.2, 0.2, 0.8)),
        )

    scene.build(n_envs=1)

    # Show IPC GUI for debugging
    print("Scene built successfully!")
    print("Launching IPC debug GUI...")

    # Set initial velocity and angular velocity for cube
    if cube_type == "rigid":
        cube.set_dofs_velocity(tuple(initial_cube_velocity))

    # Get masses and inertias based on object types
    rigid_solver = scene.sim.rigid_solver

    # Determine link indices for rigid bodies
    # No plane, so rigid bodies start from index 0
    blob_link_idx = None
    cube_link_idx = None

    # Track rigid body index (starting from 0)
    next_rigid_idx = 0

    # Blob is created first
    if blob_type == "rigid":
        blob_link_idx = next_rigid_idx
        next_rigid_idx += 1

    # Cube is created second
    if cube_type == "rigid":
        cube_link_idx = next_rigid_idx
        next_rigid_idx += 1

    # Get masses and inertias for rigid objects
    blob_mass = None
    cube_mass = None
    blob_inertia = None
    cube_inertia = None

    # Debug: print total number of links
    print(f"Debug: Total rigid links in solver: {rigid_solver.n_links}")
    print(f"Debug: blob_link_idx={blob_link_idx}, cube_link_idx={cube_link_idx}")

    if blob_type == "rigid":
        if blob_link_idx is not None and blob_link_idx < rigid_solver.n_links:
            blob_mass = rigid_solver.links_info.inertial_mass[blob_link_idx] * 2  # *2 for IPC coupling
            blob_inertia = rigid_solver.links_info.inertial_i[blob_link_idx].to_numpy()
        else:
            raise ValueError(f"blob_link_idx {blob_link_idx} out of range (n_links={rigid_solver.n_links})")
    else:  # FEM
        # For FEM sphere: mass = (4/3) * π * r³ * ρ
        blob_mass = (4.0 / 3.0) * np.pi * (blob_radius**3) * blob_rho

    if cube_type == "rigid":
        if cube_link_idx is not None and cube_link_idx < rigid_solver.n_links:
            cube_mass = rigid_solver.links_info.inertial_mass[cube_link_idx] * 2  # *2 for IPC coupling
            cube_inertia = rigid_solver.links_info.inertial_i[cube_link_idx].to_numpy()
        else:
            raise ValueError(f"cube_link_idx {cube_link_idx} out of range (n_links={rigid_solver.n_links})")
    else:  # FEM
        # For FEM cube: mass = volume * ρ
        cube_size = cube_size[0]  # assuming cube, so all sides equal
        cube_mass = (cube_size**3) * cube_rho

    # Storage for FEM previous positions (for velocity computation via finite difference)
    fem_prev_positions = {"blob": None, "cube": None}

    # Helper function to compute FEM momentum from IPC
    def compute_fem_momentum_from_ipc(entity_name):
        """Compute momentum for FEM entity using IPC vertex data."""
        from uipc.backend import SceneVisitor
        from uipc.geometry import SimplicialComplexSlot, apply_transform, merge
        from uipc import builtin

        visitor = SceneVisitor(scene.sim.coupler._ipc_scene)

        for geo_slot in visitor.geometries():
            if isinstance(geo_slot, SimplicialComplexSlot):
                geo = geo_slot.geometry()
                if geo.dim() == 3:  # FEM is 3D
                    try:
                        meta_attrs = geo.meta()
                        solver_type_attr = meta_attrs.find("solver_type")
                        entity_idx_attr = meta_attrs.find("entity_idx")

                        if solver_type_attr and entity_idx_attr:
                            solver_type = str(solver_type_attr.view()[0])
                            entity_idx = int(str(entity_idx_attr.view()[0]))

                            # Match entity by index: blob is entity 0, cube is entity 1
                            expected_entity_idx = 0 if entity_name == "blob" else 1

                            if solver_type == "fem" and entity_idx == expected_entity_idx:
                                # Found matching FEM geometry
                                proc_geo = geo
                                if geo.instances().size() >= 1:
                                    proc_geo = merge(apply_transform(geo))

                                # Get vertex positions (current frame)
                                positions = proc_geo.positions().view().reshape(-1, 3)

                                # Get vertex masses
                                volume_attr = proc_geo.vertices().find(builtin.volume)
                                mass_density_attr = proc_geo.vertices().find(builtin.mass_density)
                                if volume_attr and mass_density_attr:
                                    volumes = volume_attr.view().reshape(-1)
                                    mass_densities = mass_density_attr.view().reshape(-1)
                                    vertex_masses = volumes * mass_densities
                                else:
                                    # Fallback: uniform mass distribution
                                    if entity_name == "blob":
                                        total_mass = blob_mass
                                    else:
                                        total_mass = cube_mass
                                    n_vertices = len(positions)
                                    vertex_masses = np.ones(n_vertices) * (total_mass / n_vertices)

                                # Compute velocities using finite difference
                                if fem_prev_positions[entity_name] is not None:
                                    vertex_velocities = (positions - fem_prev_positions[entity_name]) / dt
                                else:
                                    # First frame: zero velocity
                                    vertex_velocities = np.zeros_like(positions)

                                # Store current positions for next frame
                                fem_prev_positions[entity_name] = positions.copy()

                                # Compute linear momentum: P = sum(m_i * v_i)
                                linear_momentum = np.sum(vertex_masses[:, np.newaxis] * vertex_velocities, axis=0)

                                # Compute COM and COM velocity
                                total_mass_actual = np.sum(vertex_masses)
                                com = np.sum(vertex_masses[:, np.newaxis] * positions, axis=0) / total_mass_actual
                                vel = linear_momentum / total_mass_actual if total_mass_actual > 0 else np.zeros(3)

                                # Compute angular momentum: L = L_orbital + L_intrinsic
                                # L_intrinsic = sum(r_i × m_i*v_i) where r_i is position relative to COM
                                angular_momentum_intrinsic = np.zeros(3)
                                for i in range(len(positions)):
                                    r_i = positions[i] - com
                                    p_i = vertex_masses[i] * vertex_velocities[i]
                                    angular_momentum_intrinsic += np.cross(r_i, p_i)

                                # L_orbital = r_com × p_total (angular momentum due to COM motion)
                                angular_momentum_orbital = np.cross(com, linear_momentum)

                                # Total angular momentum relative to origin
                                angular_momentum = angular_momentum_orbital + angular_momentum_intrinsic

                                omega = np.zeros(3)  # Not computing rigid-body omega for FEM

                                return com, vel, omega, linear_momentum, angular_momentum

                    except Exception as e:
                        continue

        # Fallback: return zeros
        return np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3)

    # Calculate total linear and angular momentum
    def compute_total_momentum():
        """Compute total linear and angular momentum of the system.

        Angular momentum is computed relative to the origin:
        L_total = Σ(r_i × p_i + L_i^body)
        where r_i is COM position, p_i is linear momentum, L_i^body is intrinsic angular momentum
        """
        rigid_solver = scene.sim.rigid_solver

        # === Cube momentum ===
        if cube_type == "rigid":
            # Get cube COM position
            cube_pos = rigid_solver.get_links_pos(links_idx=cube_link_idx, ref="link_com").detach().cpu().numpy()
            cube_pos = cube_pos.flatten()[:3]

            # Get cube velocity
            cube_vel = rigid_solver.get_links_vel(links_idx=cube_link_idx, ref="link_com").detach().cpu().numpy()
            cube_vel = cube_vel.flatten()[:3]
            cube_linear_momentum = cube_mass * cube_vel

            # Get cube angular velocity
            cube_omega = rigid_solver.get_links_ang(links_idx=cube_link_idx).detach().cpu().numpy()
            cube_omega = cube_omega.flatten()[:3]

            # Angular momentum
            cube_L_body = cube_inertia @ cube_omega
            cube_L_orbital = np.cross(cube_pos, cube_linear_momentum)
            cube_angular_momentum = cube_L_orbital + cube_L_body
        else:  # FEM
            cube_pos, cube_vel, cube_omega, cube_linear_momentum, cube_angular_momentum = compute_fem_momentum_from_ipc(
                "cube"
            )

        # === Blob momentum ===
        if blob_type == "rigid":
            # Get blob COM position
            blob_pos = rigid_solver.get_links_pos(links_idx=blob_link_idx, ref="link_com").detach().cpu().numpy()
            blob_pos = blob_pos.flatten()[:3]

            # Get blob velocity
            blob_vel = rigid_solver.get_links_vel(links_idx=blob_link_idx, ref="link_com").detach().cpu().numpy()
            blob_vel = blob_vel.flatten()[:3]
            blob_linear_momentum = blob_mass * blob_vel

            # Get blob angular velocity
            blob_omega = rigid_solver.get_links_ang(links_idx=blob_link_idx).detach().cpu().numpy()
            blob_omega = blob_omega.flatten()[:3]

            # Angular momentum
            blob_L_body = blob_inertia @ blob_omega
            blob_L_orbital = np.cross(blob_pos, blob_linear_momentum)
            blob_angular_momentum = blob_L_orbital + blob_L_body
        else:  # FEM
            blob_pos, blob_vel, blob_omega, blob_linear_momentum, blob_angular_momentum = compute_fem_momentum_from_ipc(
                "blob"
            )

        # === Total momentum ===
        total_linear_momentum = cube_linear_momentum + blob_linear_momentum
        total_angular_momentum = cube_angular_momentum + blob_angular_momentum

        return (
            total_linear_momentum,
            total_angular_momentum,
            cube_linear_momentum,
            blob_linear_momentum,
            cube_angular_momentum,
            blob_angular_momentum,
            cube_vel,
            blob_vel,
            cube_omega,
            blob_omega,
            cube_mass,
            blob_mass,
        )

    print("\n=== Momentum Conservation Test (Zero Gravity) ===")
    print(f"Cube type: {cube_type.upper()}, mass: {cube_mass:.4f} kg")
    print(f"Blob type: {blob_type.upper()}, mass: {blob_mass:.4f} kg")
    print(f"Cube initial velocity: {initial_cube_velocity.tolist()} [vx, vy, vz, wx, wy, wz]")
    print(f"Blob initial velocity: [0, 0, 0, 0, 0, 0]")
    if cube_type == "rigid":
        print(f"Expected total linear momentum: [{initial_cube_velocity[0] * cube_mass:.4f}, 0, 0] kg·m/s")
    else:
        print(f"Note: FEM cube - initial velocity not directly settable")

    # Storage for plotting
    time_history = []
    cube_p_history = []
    blob_p_history = []
    total_p_history = []
    cube_L_history = []
    blob_L_history = []
    total_L_history = []
    cube_v_history = []
    blob_v_history = []
    cube_omega_history = []
    blob_omega_history = []

    # Storage for initial momentum (computed at specified reference time)
    initial_linear_momentum = None
    initial_angular_momentum = None

    n_steps = int(test_time / dt)

    for i_step in range(n_steps):
        # Compute momentum at every step
        (
            total_p,
            total_L,
            cube_p,
            blob_p,
            cube_L,
            blob_L,
            cube_v,
            blob_v,
            cube_omega,
            blob_omega,
            cube_m,
            blob_m,
        ) = compute_total_momentum()

        # Save data for plotting
        time_history.append(i_step * dt)
        cube_p_history.append(np.asarray(cube_p).flatten().copy())
        blob_p_history.append(np.asarray(blob_p).flatten().copy())
        total_p_history.append(np.asarray(total_p).flatten().copy())
        cube_L_history.append(np.asarray(cube_L).flatten().copy())
        blob_L_history.append(np.asarray(blob_L).flatten().copy())
        total_L_history.append(np.asarray(total_L).flatten().copy())
        cube_v_history.append(np.asarray(cube_v).flatten().copy())
        blob_v_history.append(np.asarray(blob_v).flatten().copy())
        cube_omega_history.append(np.asarray(cube_omega).flatten().copy())
        blob_omega_history.append(np.asarray(blob_omega).flatten().copy())

        # Capture initial momentum at specified reference time
        if initial_linear_momentum is None and i_step * dt >= initial_momentum_reference_time:
            initial_linear_momentum = np.asarray(total_p).flatten().copy()
            initial_angular_momentum = np.asarray(total_L).flatten().copy()
            print(f"\nCapturing initial momentum at t={i_step * dt:.4f}s:")
            print(f"  Linear:  {initial_linear_momentum}")
            print(f"  Angular: {initial_angular_momentum}")

        # Print every 100 steps
        if i_step % (n_steps // 10) == 0:
            # Ensure all are numpy arrays
            total_p = np.asarray(total_p).flatten()
            total_L = np.asarray(total_L).flatten()
            cube_p = np.asarray(cube_p).flatten()
            blob_p = np.asarray(blob_p).flatten()
            cube_L = np.asarray(cube_L).flatten()
            blob_L = np.asarray(blob_L).flatten()
            cube_v = np.asarray(cube_v).flatten()
            blob_v = np.asarray(blob_v).flatten()

            print(f"\n{'='*70}")
            print(f"Step {i_step:4d}: t = {i_step * dt:.3f}s")
            print(f"{'-'*70}")
            print(f"Cube   mass: {cube_m:8.4f} kg")
            print(f"Cube   vel:  [{cube_v[0]:9.5f}, {cube_v[1]:9.5f}, {cube_v[2]:9.5f}] m/s")
            print(f"Cube   p:    [{cube_p[0]:9.5f}, {cube_p[1]:9.5f}, {cube_p[2]:9.5f}] kg·m/s")
            print(f"Cube   L:    [{cube_L[0]:9.5f}, {cube_L[1]:9.5f}, {cube_L[2]:9.5f}] kg·m²/s")
            print(f"")
            print(f"Blob   mass: {blob_m:8.4f} kg")
            print(f"Blob   vel:  [{blob_v[0]:9.5f}, {blob_v[1]:9.5f}, {blob_v[2]:9.5f}] m/s")
            print(f"Blob   p:    [{blob_p[0]:9.5f}, {blob_p[1]:9.5f}, {blob_p[2]:9.5f}] kg·m/s")
            print(f"Blob   L:    [{blob_L[0]:9.5f}, {blob_L[1]:9.5f}, {blob_L[2]:9.5f}] kg·m²/s")
            print(f"")
            print(f"TOTAL  p:    [{total_p[0]:9.5f}, {total_p[1]:9.5f}, {total_p[2]:9.5f}] kg·m/s")
            print(f"TOTAL  L:    [{total_L[0]:9.5f}, {total_L[1]:9.5f}, {total_L[2]:9.5f}] kg·m²/s")
            print(f"|p_total|: {np.linalg.norm(total_p):.6f} kg·m/s")
            print(f"|L_total|: {np.linalg.norm(total_L):.6f} kg·m²/s")

        scene.step()

    # Convert lists to numpy arrays for plotting
    time_history = np.array(time_history)
    cube_p_history = np.array(cube_p_history)  # Shape: (n_steps, 3)
    blob_p_history = np.array(blob_p_history)  # Shape: (n_steps, 3)
    total_p_history = np.array(total_p_history)  # Shape: (n_steps, 3)
    cube_L_history = np.array(cube_L_history)  # Shape: (n_steps, 3)
    blob_L_history = np.array(blob_L_history)  # Shape: (n_steps, 3)
    total_L_history = np.array(total_L_history)  # Shape: (n_steps, 3)
    cube_v_history = np.array(cube_v_history)  # Shape: (n_steps, 3)
    blob_v_history = np.array(blob_v_history)  # Shape: (n_steps, 3)

    # Create plots
    fig, axes = plt.subplots(5, 2, figsize=(14, 18))
    fig.suptitle("Momentum Conservation Test (Zero Gravity)", fontsize=16)

    # Plot 1: Linear Momentum X-component
    ax = axes[0, 0]
    ax.plot(time_history, cube_p_history[:, 0], "r-", label="Cube px", linewidth=2)
    ax.plot(time_history, blob_p_history[:, 0], "b-", label="Blob px", linewidth=2)
    ax.plot(time_history, total_p_history[:, 0], "k--", label="Total px", linewidth=2)
    ax.axhline(
        y=initial_cube_velocity[0] * cube_mass,
        color="g",
        linestyle=":",
        label=f"Expected: {initial_cube_velocity[0] * cube_mass:.4f}",
    )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Momentum (kg·m/s)")
    ax.set_title("Linear Momentum X-component")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Linear Momentum Y-component
    ax = axes[0, 1]
    ax.plot(time_history, cube_p_history[:, 1], "r-", label="Cube py", linewidth=2)
    ax.plot(time_history, blob_p_history[:, 1], "b-", label="Blob py", linewidth=2)
    ax.plot(time_history, total_p_history[:, 1], "k--", label="Total py", linewidth=2)
    ax.axhline(y=0.0, color="g", linestyle=":", label="Expected: 0.0")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Momentum (kg·m/s)")
    ax.set_title("Linear Momentum Y-component")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Linear Momentum Z-component
    ax = axes[1, 0]
    ax.plot(time_history, cube_p_history[:, 2], "r-", label="Cube pz", linewidth=2)
    ax.plot(time_history, blob_p_history[:, 2], "b-", label="Blob pz", linewidth=2)
    ax.plot(time_history, total_p_history[:, 2], "k--", label="Total pz", linewidth=2)
    ax.axhline(y=0.0, color="g", linestyle=":", label="Expected: 0.0")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Momentum (kg·m/s)")
    ax.set_title("Linear Momentum Z-component")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: Total Linear Momentum Magnitude
    ax = axes[1, 1]
    total_p_mag = np.linalg.norm(total_p_history, axis=1)
    ax.plot(time_history, total_p_mag, "k-", linewidth=2, label="|p_total|")
    ax.axhline(
        y=initial_cube_velocity[0] * cube_mass,
        color="r",
        linestyle="--",
        label=f"Expected: {initial_cube_velocity[0] * cube_mass:.4f}",
    )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("|p| (kg·m/s)")
    ax.set_title("Total Linear Momentum Magnitude")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 5: Angular Momentum X-component
    ax = axes[2, 0]
    ax.plot(time_history, cube_L_history[:, 0], "r-", label="Cube Lx", linewidth=2)
    ax.plot(time_history, blob_L_history[:, 0], "b-", label="Blob Lx", linewidth=2)
    ax.plot(time_history, total_L_history[:, 0], "k--", label="Total Lx", linewidth=2)
    if initial_angular_momentum is not None:
        ax.axhline(
            y=initial_angular_momentum[0], color="g", linestyle=":", label=f"Ref: {initial_angular_momentum[0]:.6f}"
        )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Angular Momentum (kg·m²/s)")
    ax.set_title("Angular Momentum X-component")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 6: Angular Momentum Y-component
    ax = axes[2, 1]
    ax.plot(time_history, cube_L_history[:, 1], "r-", label="Cube Ly", linewidth=2)
    ax.plot(time_history, blob_L_history[:, 1], "b-", label="Blob Ly", linewidth=2)
    ax.plot(time_history, total_L_history[:, 1], "k--", label="Total Ly", linewidth=2)
    if initial_angular_momentum is not None:
        ax.axhline(
            y=initial_angular_momentum[1], color="g", linestyle=":", label=f"Ref: {initial_angular_momentum[1]:.6f}"
        )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Angular Momentum (kg·m²/s)")
    ax.set_title("Angular Momentum Y-component")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 7: Angular Momentum Z-component
    ax = axes[3, 0]
    ax.plot(time_history, cube_L_history[:, 2], "r-", label="Cube Lz", linewidth=2)
    ax.plot(time_history, blob_L_history[:, 2], "b-", label="Blob Lz", linewidth=2)
    ax.plot(time_history, total_L_history[:, 2], "k--", label="Total Lz", linewidth=2)
    if initial_angular_momentum is not None:
        ax.axhline(
            y=initial_angular_momentum[2], color="g", linestyle=":", label=f"Ref: {initial_angular_momentum[2]:.6f}"
        )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Angular Momentum (kg·m²/s)")
    ax.set_title("Angular Momentum Z-component")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 8: Total Angular Momentum Magnitude
    ax = axes[3, 1]
    total_L_mag = np.linalg.norm(total_L_history, axis=1)
    ax.plot(time_history, total_L_mag, "k-", linewidth=2, label="|L_total|")
    if initial_angular_momentum is not None:
        ax.axhline(
            y=np.linalg.norm(initial_angular_momentum),
            color="r",
            linestyle="--",
            label=f"Initial: {np.linalg.norm(initial_angular_momentum):.6f}",
        )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("|L| (kg·m²/s)")
    ax.set_title("Total Angular Momentum Magnitude")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 9: Linear Momentum Conservation Error
    ax = axes[4, 0]
    if initial_linear_momentum is not None:
        linear_momentum_error = np.linalg.norm(total_p_history - initial_linear_momentum, axis=1)
        ax.plot(time_history, linear_momentum_error, "r-", linewidth=2, label="Linear momentum error")
        ax.axvline(
            x=initial_momentum_reference_time,
            color="g",
            linestyle=":",
            alpha=0.5,
            label=f"Reference: t={initial_momentum_reference_time}s",
        )
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Error (kg·m/s)")
        ax.set_title(f"Linear Momentum Error (ref: t={initial_momentum_reference_time}s)")
        ax.set_yscale("log")
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        # Fallback
        expected_momentum = np.array([initial_cube_velocity[0] * cube_mass, 0.0, 0.0])
        linear_momentum_error = np.linalg.norm(total_p_history - expected_momentum, axis=1)
        ax.plot(time_history, linear_momentum_error, "r-", linewidth=2)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Error (kg·m/s)")
        ax.set_title("Linear Momentum Error")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)

    # Plot 10: Angular Momentum Conservation Error
    ax = axes[4, 1]
    if initial_angular_momentum is not None:
        angular_momentum_error = np.linalg.norm(total_L_history - initial_angular_momentum, axis=1)
        ax.plot(time_history, angular_momentum_error, "b-", linewidth=2, label="Angular momentum error")
        ax.axvline(
            x=initial_momentum_reference_time,
            color="g",
            linestyle=":",
            alpha=0.5,
            label=f"Reference: t={initial_momentum_reference_time}s",
        )
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Error (kg·m²/s)")
        ax.set_title(f"Angular Momentum Error (ref: t={initial_momentum_reference_time}s)")
        ax.set_yscale("log")
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, "No initial angular momentum captured", ha="center", va="center", transform=ax.transAxes)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Error (kg·m²/s)")
        ax.set_title("Angular Momentum Error")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
