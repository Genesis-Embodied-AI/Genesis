"""
Test external articulation coupling with IPC for a four-cube chain.

Chain structure:
- Cube 0 (base): Fixed to world
- Cube 1: Connected to base via revolute joint (joint1, rotates around Z-axis)
- Cube 2: Connected to cube1 via FIXED joint (no joint element, should be merged)
- Cube 3: Connected to cube2 via prismatic joint (joint3, slides along X-axis)

This test verifies that:
1. Fixed joints (cube1-cube2) are properly merged in IPC
2. The IPC coupler correctly handles the merged body
3. External articulation constraint works with merged bodies
"""

import genesis as gs
import numpy as np


def main():
    # Initialize Genesis
    gs.init(backend=gs.gpu, logging_level="info", performance_mode=True)

    # Create scene
    dt = 0.01

    # Configure IPC coupler with external_articulation strategy
    coupler_options = gs.options.IPCCouplerOptions(
        dt=dt,
        gravity=(0.0, 0.0, -9.8),
        coupling_strategy="external_articulation",
        contact_friction_mu=0.5,
        IPC_self_contact=False,
        enable_ipc_gui=True,
    )

    # Disable rigid collision when using IPC
    rigid_options = gs.options.RigidOptions(
        enable_collision=False,
    )

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=dt,
            gravity=(0.0, 0.0, -9.8),
        ),
        rigid_options=rigid_options,
        # coupler_options=coupler_options,
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(1.5, 0.0, 1.0),
            camera_lookat=(0.0, 0.0, 0.5),
            camera_fov=40,
        ),
        show_viewer=True,
    )

    # Add the four-cube chain from MJCF
    robot = scene.add_entity(
        gs.morphs.MJCF(
            file="examples/IPC_Solver/assets/four_cube_chain.xml",
        ),
    )
    # robot = scene.add_entity(gs.morphs.URDF(file="urdf/simple/three_cube_chain.urdf", pos=(0.1, 0.0, 0.5), fixed=True))

    # Build scene
    print("Building scene...")
    scene.build()
    print("Scene built!")

    # Print joint information
    print("\n=== Joint Information ===")
    print(f"Total joints: {len(robot.joints)}")
    for i, joint in enumerate(robot.joints):
        print(f"\nJoint {i}: {joint.name}")
        print(f"  Type: {joint.type}")
        print(f"  Link: {joint.link.name}")
        if hasattr(joint.link, "parent_idx") and joint.link.parent_idx >= 0:
            parent_link = robot.links[joint.link.parent_idx - robot.link_start]
            print(f"  Parent link: {parent_link.name}")
        if joint.type == gs.JOINT_TYPE.REVOLUTE:
            print(f"  Rotation axis (local): {joint.dofs_motion_ang[0]}")
        elif joint.type == gs.JOINT_TYPE.PRISMATIC:
            print(f"  Translation axis (local): {joint.dofs_motion_vel[0]}")

    # Print link information
    print("\n=== Link Information ===")
    for i, link in enumerate(robot.links):
        print(f"\nLink {i}: {link.name}")
        print(f"  pos: {link.pos}")
        print(f"  quat: {link.quat}")
        print(f"  parent_idx: {link.parent_idx}")
        print(f"  n_joints: {len(link.joints)}")
        if len(link.joints) > 0:
            print(f"  joints: {[j.name for j in link.joints]}")
            for joint in link.joints:
                print(f"    - {joint.name}: type={joint.type}")
        else:
            print(f"  joints: [] (FIXED JOINT - should be merged in IPC)")

    # Check IPC coupling
    print("\n=== IPC Coupling Information ===")
    if hasattr(scene, "_coupler") and hasattr(scene._coupler, "_articulated_entities"):
        print(f"Articulated entities: {len(scene._coupler._articulated_entities)}")
        for entity_idx, art_data in scene._coupler._articulated_entities.items():
            print(f"\nEntity {entity_idx}:")
            print(f"  n_joints: {art_data['n_joints']}")
            print(f"  revolute_joints: {[j.name for j in art_data['revolute_joints']]}")
            print(f"  prismatic_joints: {[j.name for j in art_data['prismatic_joints']]}")

    # Print ABD slot mappings
    if hasattr(scene, "_coupler") and hasattr(scene._coupler, "_link_to_abd_slot"):
        print("\n=== ABD Slot Mappings ===")
        for (entity_idx, link_idx), abd_slot in scene._coupler._link_to_abd_slot.items():
            link = robot.links[link_idx - robot.link_start]
            print(f"Entity {entity_idx}, Link {link_idx} ({link.name}): ABD slot {abd_slot}")

    # Get initial qpos
    qpos_init = robot.get_qpos().cpu().numpy()
    print(f"\n=== Initial State ===")
    print(f"Initial qpos: {qpos_init}")
    print(f"Number of DOFs: {robot.n_dofs}")
    print(f"Number of links: {robot.n_links}")

    # Set PD control gains
    print("\n=== Setting PD Control ===")
    robot.set_dofs_kp(np.array([100.0, 100.0]))
    robot.set_dofs_kv(np.array([10.0, 10.0]))
    print("PD gains set: kp=[100.0, 100.0], kv=[10.0, 10.0]")

    # Phase 1: Hold at zero position (settle)
    print("\n=== Phase 1: Settling at zero (50 steps) ===")
    target_zero = np.zeros(robot.n_dofs)
    for i in range(50):
        robot.control_dofs_position(target_zero)
        scene.step()

        if i % 10 == 0:
            current_qpos = robot.get_qpos().cpu().numpy()
            print(f"Step {i:3d}: qpos = {current_qpos}")

    # Phase 2: Oscillating motion (back and forth)
    print("\n=== Phase 2: Oscillating motion ===")

    # Parameters for oscillation
    amplitude1 = 1  # radians for joint1 (Z-axis rotation)
    amplitude2 = 1  # radians for joint2 (Y-axis rotation)
    period = 500  # steps per cycle

    step = 0
    while scene.viewer.is_alive():
        phase = 2.0 * np.pi * step / period

        target = np.zeros(robot.n_dofs)
        if robot.n_dofs >= 1:
            target[0] = amplitude1 * np.sin(phase)
        if robot.n_dofs >= 2:
            target[1] = amplitude2 * np.sin(phase)

        robot.control_dofs_position(target)
        scene.step()
        step += 1


if __name__ == "__main__":
    main()
