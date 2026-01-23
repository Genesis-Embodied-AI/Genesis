"""
Example demonstrating the KinematicContactProbe for detecting contact with surfaces.

A sphere drops onto a fixed box that has a 5x5 grid of kinematic contact probes on its surface.
The probes detect when the falling sphere makes contact and report penetration along
the probe normal direction.
"""

import argparse

import genesis as gs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", default=False, help="Show visualization GUI")
    parser.add_argument("-c", "--cpu", action="store_true", help="Use CPU instead of GPU")
    parser.add_argument("-t", "--seconds", type=float, default=3.0, help="Number of seconds to simulate")
    args = parser.parse_args()

    ########################## init ##########################
    gs.init(backend=gs.cpu if args.cpu else gs.gpu, logging_level="info")

    ########################## scene setup ##########################
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=0.01,
            substeps=10,
            gravity=(0.0, 0.0, -9.81),
        ),
        vis_options=gs.options.VisOptions(
            show_world_frame=True,
        ),
        show_viewer=args.vis,
    )

    # Ground plane
    scene.add_entity(gs.morphs.Plane())

    # Fixed flat box as the platform
    box_size = 0.5
    box_height = 0.3
    box = scene.add_entity(
        gs.morphs.Box(
            size=(box_size, box_size, box_height),
            pos=(0.0, 0.0, box_height / 2),  # Box sitting on ground
            fixed=True,
        ),
    )

    # Create a 5x5 grid of probe positions on top of the box
    # Probes are evenly spaced across the box surface
    grid_size = 5
    probe_spacing = box_size / (grid_size + 1)  # Spacing between probes
    probe_positions = []
    probe_normals = []

    for i in range(grid_size):
        for j in range(grid_size):
            # Calculate position relative to box center
            x = (i - (grid_size - 1) / 2) * probe_spacing
            y = (j - (grid_size - 1) / 2) * probe_spacing
            z = box_height / 2  # On top surface of box
            probe_positions.append((x, y, z))
            probe_normals.append((0.0, 0.0, 1.0))  # All probes sense upward

    # Kinematic contact probe array on top of the box
    # Each probe has a smaller radius to fit the grid
    probe_radius = probe_spacing * 0.4  # Slightly smaller than half the spacing
    probe = scene.add_sensor(
        gs.sensors.KinematicContactProbe(
            entity_idx=box.idx,
            link_idx_local=0,
            probe_local_pos=probe_positions,
            probe_local_normal=probe_normals,
            radius=probe_radius,
            stiffness=5000.0,  # Contact stiffness for force calculation
            draw_debug=args.vis,  # Visualize the sensing spheres
        )
    )

    # Falling sphere - starts above the box and drops due to gravity
    sphere_radius = 0.1
    falling_sphere = scene.add_entity(
        gs.morphs.Sphere(
            radius=sphere_radius,
            pos=(0.0, 0.0, 0.8),  # Start above the probe grid
            fixed=False,  # Dynamic - will fall
        ),
        surface=gs.surfaces.Default(
            color=(1, 1, 1, 0.5),
        ),
    )

    scene.build(n_envs=0)

    ########################## simulation loop ##########################
    steps = int(args.seconds / 0.01)
    contact_detected = False
    n_probes = len(probe_positions)

    # give a sphere a small velocity in x
    falling_sphere.set_dofs_velocity((0.1, 0.0, 0.0, 0.0, 0.0, 0.0))

    print("\n=== KinematicContactProbe Example (5x5 Grid) ===")
    print(f"A sphere is falling onto a box with a {grid_size}x{grid_size} grid of contact probes.")
    print(f"Total probes: {n_probes}, Probe radius: {probe_radius:.4f}m")
    print("The probes measure penetration along their normal direction (+z).")
    print("Watch for contact detection when the sphere lands.\n")

    for i in range(steps):
        scene.step()

        # Read probe sensor data
        data = probe.read()

        # Check for any contact (penetration > 0 for any probe)
        # data.penetration has shape (n_probes,)
        penetrations = data.penetration
        # print the coordinates of the probes that are in contact
        for i in range(n_probes):
            if penetrations[i] > 0:
                print(f"Probe {i} is in contact at {probe_positions[i]}")
    print("\n=== Simulation Complete ===")


if __name__ == "__main__":
    main()
