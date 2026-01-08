"""
Test MPM to Rigid Link Attachment

This script tests the new feature for attaching MPM particles to rigid links
using soft constraints. The rigid box is controlled via DOF position control,
and the attached MPM particles follow its movement.
"""

import torch
import genesis as gs

gs.init(backend=gs.gpu)

# Create scene with MPM and rigid solvers
scene = gs.Scene(
    sim_options=gs.options.SimOptions(
        dt=2e-3,
        substeps=20,
    ),
    mpm_options=gs.options.MPMOptions(
        lower_bound=(-1.0, -1.0, 0.0),
        upper_bound=(1.0, 1.0, 1.5),
        grid_density=64,
        enable_particle_constraints=True,
    ),
    viewer_options=gs.options.ViewerOptions(
        camera_pos=(1.5, 0.0, 0.8),
        camera_lookat=(0.0, 0.0, 0.4),
    ),
    show_viewer=True,
)

# Add ground plane
scene.add_entity(gs.morphs.Plane())

# Add a free rigid box that we will control via DOF
rigid_box = scene.add_entity(
    gs.morphs.Box(
        pos=(0.0, 0.0, 0.55),  # Positioned above the MPM cube
        size=(0.12, 0.12, 0.05),
        fixed=False,  # Free body
    ),
)

# Add an MPM elastic cube below the rigid box (not overlapping)
mpm_cube = scene.add_entity(
    material=gs.materials.MPM.Elastic(E=5e4, nu=0.3, rho=1000),
    morph=gs.morphs.Box(
        pos=(0.0, 0.0, 0.35),  # Below the rigid box with small gap
        size=(0.15, 0.15, 0.15),
    ),
)

# Build the scene
scene.build()

# Get particles in the top region of the MPM cube (close to rigid box)
bbox_min = (-0.08, -0.08, 0.41)
bbox_max = (0.08, 0.08, 0.44)
attached_particles = mpm_cube.get_particles_in_bbox(bbox_min, bbox_max)
print(f"Number of particles to attach: {len(attached_particles)}")

# Get the link of the rigid box
link = rigid_box.links[0]

# Attach particles to the rigid link
mpm_cube.set_particle_constraints(
    particles_idx_local=attached_particles,
    link=link,
    stiffness=1e5,
)
print("Particles attached to rigid link")

# Run simulation with direct qpos control
n_steps = 500
initial_z = 0.55

for i in range(n_steps):
    # Oscillate the box up and down using qpos
    # Free box has 7 qpos: 3 translation + 4 quaternion
    z_offset = 0.15 * (1 - abs((i % 200) - 100) / 100.0)

    # Set qpos directly: [x, y, z, qw, qx, qy, qz]
    target_qpos = torch.tensor([0.0, 0.0, initial_z + z_offset, 1.0, 0.0, 0.0, 0.0], device=gs.device)
    rigid_box.set_qpos(target_qpos)

    scene.step()

    if i % 100 == 0:
        box_pos = rigid_box.get_pos()
        print(f"Step {i}: Box z={box_pos[2].item():.3f}")

print("Simulation complete!")
