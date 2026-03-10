"""Demo: Two ABD rigid cubes colliding head-on (pure rigid-rigid via IPC).

Tests momentum conservation and kinetic energy recovery with restitution.
With e=0 (inelastic), cubes stop after collision; with e=1 (elastic), cubes
bounce back. Momentum is conserved in both cases because restitution is
applied symmetrically to both ABD bodies.

Supports equal-mass (default) and asymmetric-mass (--mass-ratio) collisions.
"""

import argparse
import os

import numpy as np
import matplotlib.pyplot as plt

import genesis as gs
from genesis.utils.misc import tensor_to_array


def analytical_solution(m_a, m_b, v_a0, v_b0, e):
    """Compute analytical post-collision velocities."""
    p = m_a * v_a0 + m_b * v_b0
    v_cm = p / (m_a + m_b)
    v_rel = v_a0 - v_b0
    v_a_final = v_cm - e * m_b / (m_a + m_b) * v_rel
    v_b_final = v_cm + e * m_a / (m_a + m_b) * v_rel
    return v_a_final, v_b_final


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", default=False)
    parser.add_argument("-e", "--restitution", type=float, default=1.0)
    parser.add_argument(
        "-m",
        "--mass-ratio",
        type=float,
        default=1.0,
        help="Mass ratio m_b/m_a (default 1.0 = equal mass). E.g. 3.0 means cube B is 3x heavier.",
    )
    args = parser.parse_args()

    gs.init(backend=gs.cpu)

    rho_a = 1000
    rho_b = rho_a * args.mass_ratio

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=0.001,
            gravity=(0.0, 0.0, 0.0),
        ),
        coupler_options=gs.options.IPCCouplerOptions(
            contact_d_hat=0.001,
            enable_rigid_rigid_contact=True,
            enable_rigid_ground_contact=False,
            restitution=args.restitution,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.0, 0.8, 0.4),
            camera_lookat=(0.0, 0.0, 0.4),
        ),
        show_viewer=args.vis,
    )

    cube_a = scene.add_entity(
        morph=gs.morphs.Box(
            pos=(-0.2, 0.0, 0.4),
            size=(0.05, 0.05, 0.05),
        ),
        material=gs.materials.Rigid(
            rho=rho_a,
            friction=0.01,
            coup_type="two_way_soft_constraint",
            coup_stiffness=(1.0, 1.0),
        ),
        surface=gs.surfaces.Plastic(
            color=(0.8, 0.2, 0.2, 1.0),
        ),
    )

    cube_b = scene.add_entity(
        morph=gs.morphs.Box(
            pos=(0.2, 0.0, 0.4),
            size=(0.05, 0.05, 0.05),
        ),
        material=gs.materials.Rigid(
            rho=rho_b,
            friction=0.01,
            coup_type="two_way_soft_constraint",
            coup_stiffness=(1.0, 1.0),
        ),
        surface=gs.surfaces.Plastic(
            color=(0.2, 0.2, 0.8, 1.0),
        ),
    )
    scene.build()

    v_a0, v_b0 = 2.0, -2.0

    # Head-on collision: A moves right, B moves left
    cube_a.set_dofs_velocity((v_a0, 0, 0, 0, 0, 0))
    cube_b.set_dofs_velocity((v_b0, 0, 0, 0, 0, 0))

    mass_a = cube_a.get_mass()
    mass_b = cube_b.get_mass()
    initial_p = mass_a * v_a0 + mass_b * v_b0
    initial_ke = 0.5 * mass_a * v_a0**2 + 0.5 * mass_b * v_b0**2

    va_expected, vb_expected = analytical_solution(mass_a, mass_b, v_a0, v_b0, args.restitution)

    print(f"\n=== Two ABD Cubes Head-On Collision (e={args.restitution}, mass ratio={args.mass_ratio}) ===")
    print(f"Cube A: mass={mass_a:.4f} kg, rho={rho_a}, v0=[{v_a0:+.1f}, 0, 0] m/s")
    print(f"Cube B: mass={mass_b:.4f} kg, rho={rho_b:.0f}, v0=[{v_b0:+.1f}, 0, 0] m/s")
    print(f"Initial momentum: {initial_p:+.4f} kg·m/s")
    print(f"Initial KE: {initial_ke:.4f} J")
    print(f"Expected final: v_a={va_expected:+.4f}, v_b={vb_expected:+.4f}")

    # Storage for plotting
    time_history = []
    va_history = []
    vb_history = []
    pa_history = []
    pb_history = []
    total_p_history = []
    ke_history = []

    test_time = 0.50
    n_steps = int(test_time / scene.sim_options.dt) if "PYTEST_VERSION" not in os.environ else 5
    for i_step in range(n_steps):
        vel_a = tensor_to_array(cube_a.get_links_vel(links_idx_local=0, ref="link_com")[..., 0, :])
        vel_b = tensor_to_array(cube_b.get_links_vel(links_idx_local=0, ref="link_com")[..., 0, :])

        pa = mass_a * vel_a
        pb = mass_b * vel_b
        total_p = pa + pb
        ke = float(0.5 * mass_a * np.sum(vel_a**2) + 0.5 * mass_b * np.sum(vel_b**2))

        t = i_step * scene.sim_options.dt
        time_history.append(t)
        va_history.append(vel_a.copy())
        vb_history.append(vel_b.copy())
        pa_history.append(pa.copy())
        pb_history.append(pb.copy())
        total_p_history.append(total_p.copy())
        ke_history.append(ke)

        if i_step % (n_steps // 10) == 0:
            print(f"\nStep {i_step:4d}: t = {t:.3f}s")
            print(f"  A: vel_x={float(vel_a[0]):+.5f}  px={float(pa[0]):+.5f}")
            print(f"  B: vel_x={float(vel_b[0]):+.5f}  px={float(pb[0]):+.5f}")
            print(f"  Total px={float(total_p[0]):+.6f}  KE={ke:.4f}  KE/KE0={ke / initial_ke:.3f}")

        scene.step()

    # Convert to numpy arrays
    time_history = np.array(time_history)
    va_history = np.stack(va_history, axis=0)
    vb_history = np.stack(vb_history, axis=0)
    pa_history = np.stack(pa_history, axis=0)
    pb_history = np.stack(pb_history, axis=0)
    total_p_history = np.stack(total_p_history, axis=0)
    ke_history = np.array(ke_history)

    # Print final comparison
    vel_a_final = va_history[-1, 0]
    vel_b_final = vb_history[-1, 0]
    print(f"\n{'=' * 60}")
    print("Final vs Expected:")
    print(f"  v_a: {vel_a_final:+.4f}  (expected {va_expected:+.4f}, err {abs(vel_a_final - va_expected):.4f})")
    print(f"  v_b: {vel_b_final:+.4f}  (expected {vb_expected:+.4f}, err {abs(vel_b_final - vb_expected):.4f})")
    print(f"  p:   {float(total_p_history[-1, 0]):+.4f}  (expected {initial_p:+.4f})")

    # Create plots
    mass_label = f", mass ratio={args.mass_ratio}" if args.mass_ratio != 1.0 else ""
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle(
        f"Two ABD Cubes Head-On Collision (e={args.restitution}{mass_label})",
        fontsize=16,
    )

    # Plot 1: Velocity X-component
    ax = axes[0, 0]
    ax.plot(time_history, va_history[:, 0], color="tab:red", linestyle="-", label="Cube A vx", linewidth=2)
    ax.plot(time_history, vb_history[:, 0], color="tab:blue", linestyle="-", label="Cube B vx", linewidth=2)
    ax.axhline(y=va_expected, color="tab:red", linestyle=":", alpha=0.5, label=f"A expected: {va_expected:+.2f}")
    ax.axhline(y=vb_expected, color="tab:blue", linestyle=":", alpha=0.5, label=f"B expected: {vb_expected:+.2f}")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Velocity (m/s)")
    ax.set_title("Velocity X-component")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Momentum X-component
    ax = axes[0, 1]
    ax.plot(time_history, pa_history[:, 0], color="tab:red", linestyle="-", label="Cube A px", linewidth=2)
    ax.plot(time_history, pb_history[:, 0], color="tab:blue", linestyle="-", label="Cube B px", linewidth=2)
    ax.plot(time_history, total_p_history[:, 0], "k", linestyle="--", label="Total px", linewidth=2)
    ax.axhline(y=initial_p, color="tab:green", linestyle=":", label=f"Expected: {initial_p:+.4f}")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Momentum (kg·m/s)")
    ax.set_title("Linear Momentum X-component")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Kinetic Energy
    ax = axes[1, 0]
    ax.plot(time_history, ke_history, color="tab:purple", linestyle="-", linewidth=2, label="KE")
    ax.axhline(y=initial_ke, color="tab:green", linestyle="--", label=f"Initial KE: {initial_ke:.4f}")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Kinetic Energy (J)")
    ax.set_title("Kinetic Energy")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: Conservation Errors
    ax = axes[1, 1]
    p_err = np.abs(total_p_history[:, 0] - initial_p)
    ke_ratio = ke_history / initial_ke
    ax.plot(time_history, p_err, color="tab:red", linestyle="-", linewidth=2, label="|p - p0|")
    ax2 = ax.twinx()
    ax2.plot(time_history, ke_ratio, color="tab:blue", linestyle="-", linewidth=2, label="KE/KE0")
    ax2.axhline(y=1.0, color="tab:blue", linestyle=":", alpha=0.3)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("|p - p0| (kg·m/s)", color="tab:red")
    ax2.set_ylabel("KE / KE0", color="tab:blue")
    ax.set_title("Conservation Metrics")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    mass_suffix = f"_m{args.mass_ratio:.0f}" if args.mass_ratio != 1.0 else ""
    fname = f"abd_collision_e{args.restitution:.1f}{mass_suffix}.png"
    plt.savefig(fname, dpi=150)
    print(f"\nPlot saved to {fname}")
    plt.show()


if __name__ == "__main__":
    main()
