import genesis as gs
import logging
import numpy as np
import matplotlib.pyplot as plt

def main():
    gs.init(backend=gs.gpu, logging_level=logging.DEBUG)
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=1e-3, gravity=(0.0, 0.0, 0.0)),
        coupler_options=gs.options.IPCCouplerOptions(
            dt=1e-3,
            gravity=(0.0, 0.0, 0.0),
            ipc_constraint_strength=(1, 1),  # (translation, rotation) strength ratios
            IPC_self_contact=False,  # Disable rigid-rigid contact in IPC
        ),
        profiling_options=gs.options.ProfilingOptions(
            show_FPS=False,
        ),
        show_viewer=True,
    )

    # Both FEM and Rigid bodies will be added to IPC for unified contact simulation
    # FEM bodies use StableNeoHookean constitution, Rigid bodies use ABD constitution

    scene.add_entity(gs.morphs.Plane())
    SCENE_POS = (0.0, 0.0, 0.0)

    # FEM entities (added to IPC as deformable bodies)
    blob = scene.add_entity(
        morph=gs.morphs.Sphere(pos=tuple(map(sum, zip(SCENE_POS, (0.3, -0.0, 0.4)))), radius=0.1),
        material=gs.materials.FEM.Elastic(E=1.0e5, nu=0.45, rho=1000.0, model="stable_neohookean")
    )


    # Rigid bodies (added to both Genesis rigid solver AND IPC as ABD objects)
    # This enables contact between rigid bodies and FEM bodies through IPC
    rigid_cube = scene.add_entity(
        morph=gs.morphs.Box(
            pos=tuple(map(sum, zip(SCENE_POS, (0.0, 0, 0.4)))),
            size=(0.1, 0.1, 0.1),
            euler=(0, 0, 0)
        ),
        material=gs.materials.Rigid(rho=1000, friction=0.3),
        surface=gs.surfaces.Plastic(color=(0.8, 0.2, 0.2, 0.8)),
    )


    scene.build(n_envs=1)

    # Show IPC GUI for debugging
    print("Scene built successfully!")
    print("Launching IPC debug GUI...")

    # Set initial velocity
    rigid_cube.set_dofs_velocity((4.0, 0, 0, 0, 0, 0))  # Initial velocity in x direction

    # Storage for previous positions to compute velocity
    fem_prev_pos = None
    rigid_prev_pos = None
    dt = scene.sim_options.dt

    # Get FEM blob total mass from material properties
    # For a sphere: mass = (4/3) * π * r³ * ρ
    blob_radius = blob.morph.radius
    blob_rho = blob.material.rho
    fem_total_mass = (4.0/3.0) * np.pi * (blob_radius**3) * blob_rho

    # Calculate total linear momentum
    def compute_total_linear_momentum():
        """Compute total linear momentum of the system."""
        nonlocal fem_prev_pos, rigid_prev_pos

        rigid_solver = scene.sim.rigid_solver

        # === Rigid body linear momentum ===
        # For rigid cube (link_idx=1, skip plane at link_idx=0)
        link_idx = 1
        rigid_mass = rigid_solver.links_info.inertial_mass[link_idx] *2

        # Get rigid COM position
        rigid_pos = rigid_solver.get_links_pos(links_idx=link_idx, ref="link_com").detach().cpu().numpy()
        rigid_pos = rigid_pos.flatten()[:3]

        # Compute rigid velocity using finite difference
        if rigid_prev_pos is not None:
            rigid_vel = (rigid_pos - rigid_prev_pos) / dt
        else:
            # First step: use the set initial velocity
            rigid_vel = np.array([4.0, 0.0, 0.0])

        rigid_prev_pos = rigid_pos.copy()
        rigid_linear_momentum = rigid_mass * rigid_vel

        # === FEM body linear momentum ===
        # Get FEM blob COM position from entity
        fem_entity = blob  # Reference to blob entity
        fem_state = fem_entity.get_state()
        fem_pos = fem_state.pos.detach().cpu().numpy()

        # Ensure fem_pos is a 1D array of shape (3,)
        fem_pos = np.atleast_1d(fem_pos).flatten()[:3]

        # Compute FEM COM velocity using finite difference
        if fem_prev_pos is not None:
            fem_vel = (fem_pos - fem_prev_pos) / dt
        else:
            fem_vel = np.zeros(3)  # First step: zero velocity

        fem_prev_pos = fem_pos.copy()
        fem_linear_momentum = fem_total_mass * fem_vel

        # === Total momentum ===
        total_linear_momentum = rigid_linear_momentum + fem_linear_momentum

        return (total_linear_momentum, rigid_linear_momentum, fem_linear_momentum,
                rigid_vel, fem_vel, rigid_mass, fem_total_mass)

    print("\n=== Linear Momentum Conservation Test (Zero Gravity) ===")
    rigid_mass = scene.sim.rigid_solver.links_info.inertial_mass[1]
    print(f"Rigid cube mass: {rigid_mass:.4f} kg")
    print(f"FEM blob mass: {fem_total_mass:.4f} kg (estimated from V*ρ)")
    print(f"Rigid initial velocity: [4.0, 0, 0, 0, 0, 0] m/s")
    print(f"FEM initial velocity: [0, 0, 0] m/s")
    print(f"Expected total momentum: [{4.0 * rigid_mass:.4f}, 0, 0] kg·m/s")

    # Storage for plotting
    time_history = []
    rigid_p_history = []
    fem_p_history = []
    total_p_history = []
    rigid_v_history = []
    fem_v_history = []

    for i_step in range(300):
        # Compute momentum at every step
        (total_p, rigid_p, fem_p,
         rigid_v, fem_v, rigid_m, fem_m) = compute_total_linear_momentum()

        # Save data for plotting
        time_history.append(i_step * dt)
        rigid_p_history.append(np.asarray(rigid_p).flatten().copy())
        fem_p_history.append(np.asarray(fem_p).flatten().copy())
        total_p_history.append(np.asarray(total_p).flatten().copy())
        rigid_v_history.append(np.asarray(rigid_v).flatten().copy())
        fem_v_history.append(np.asarray(fem_v).flatten().copy())

        # Print every 100 steps
        if i_step % 100 == 0:
            # Ensure all are numpy arrays
            total_p = np.asarray(total_p).flatten()
            rigid_p = np.asarray(rigid_p).flatten()
            fem_p = np.asarray(fem_p).flatten()
            rigid_v = np.asarray(rigid_v).flatten()
            fem_v = np.asarray(fem_v).flatten()

            print(f"\n{'='*70}")
            print(f"Step {i_step:4d}: t = {i_step * dt:.3f}s")
            print(f"{'-'*70}")
            print(f"Rigid  mass: {rigid_m:8.4f} kg")
            print(f"Rigid  vel:  [{rigid_v[0]:9.5f}, {rigid_v[1]:9.5f}, {rigid_v[2]:9.5f}] m/s")
            print(f"Rigid  mom:  [{rigid_p[0]:9.5f}, {rigid_p[1]:9.5f}, {rigid_p[2]:9.5f}] kg·m/s")
            print(f"")
            print(f"FEM    mass: {fem_m:8.4f} kg")
            print(f"FEM    vel:  [{fem_v[0]:9.5f}, {fem_v[1]:9.5f}, {fem_v[2]:9.5f}] m/s")
            print(f"FEM    mom:  [{fem_p[0]:9.5f}, {fem_p[1]:9.5f}, {fem_p[2]:9.5f}] kg·m/s")
            print(f"")
            print(f"TOTAL  mom:  [{total_p[0]:9.5f}, {total_p[1]:9.5f}, {total_p[2]:9.5f}] kg·m/s")
            print(f"|p_total|: {np.linalg.norm(total_p):.6f} kg·m/s")

        scene.step()

    # Convert lists to numpy arrays for plotting
    time_history = np.array(time_history)
    rigid_p_history = np.array(rigid_p_history)  # Shape: (n_steps, 3)
    fem_p_history = np.array(fem_p_history)      # Shape: (n_steps, 3)
    total_p_history = np.array(total_p_history)  # Shape: (n_steps, 3)
    rigid_v_history = np.array(rigid_v_history)  # Shape: (n_steps, 3)
    fem_v_history = np.array(fem_v_history)      # Shape: (n_steps, 3)

    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle('Momentum Conservation Test (Zero Gravity)', fontsize=16)

    # Plot 1: Linear Momentum X-component
    ax = axes[0, 0]
    ax.plot(time_history, rigid_p_history[:, 0], 'r-', label='Rigid px', linewidth=2)
    ax.plot(time_history, fem_p_history[:, 0], 'b-', label='FEM px', linewidth=2)
    ax.plot(time_history, total_p_history[:, 0], 'k--', label='Total px', linewidth=2)
    ax.axhline(y=4.0 * rigid_mass, color='g', linestyle=':', label=f'Expected: {4.0 * rigid_mass:.4f}')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Momentum (kg·m/s)')
    ax.set_title('Linear Momentum X-component')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Total Momentum Magnitude
    ax = axes[0, 1]
    total_p_mag = np.linalg.norm(total_p_history, axis=1)
    ax.plot(time_history, total_p_mag, 'k-', linewidth=2, label='|p_total|')
    ax.axhline(y=4.0 * rigid_mass, color='r', linestyle='--', label=f'Expected: {4.0 * rigid_mass:.4f}')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('|p| (kg·m/s)')
    ax.set_title('Total Momentum Magnitude')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Velocity X-component
    ax = axes[1, 0]
    ax.plot(time_history, rigid_v_history[:, 0], 'r-', label='Rigid vx', linewidth=2)
    ax.plot(time_history, fem_v_history[:, 0], 'b-', label='FEM vx', linewidth=2)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Velocity (m/s)')
    ax.set_title('Velocity X-component')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: Momentum Conservation Error
    ax = axes[1, 1]
    expected_momentum = np.array([4.0 * rigid_mass, 0.0, 0.0])
    momentum_error = np.linalg.norm(total_p_history - expected_momentum, axis=1)
    ax.plot(time_history, momentum_error, 'r-', linewidth=2)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Error (kg·m/s)')
    ax.set_title('Momentum Conservation Error')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('momentum_conservation_test.png', dpi=150)
    print(f"\n{'='*70}")
    print("Plot saved to: momentum_conservation_test.png")
    print(f"{'='*70}")
    plt.show()



if __name__ == "__main__":
    main()