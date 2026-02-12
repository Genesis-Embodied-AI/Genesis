"""
Unit test comparing analytical capsule-capsule contact detection with GJK.

This test creates a modified version of narrowphase.py in a temporary file that
forces capsule-capsule and sphere-capsule collisions to use GJK instead of 
analytical methods, allowing direct comparison between the two approaches.
"""

import gstaichi as ti
import os
import sys
import tempfile
import xml.etree.ElementTree as ET

import numpy as np
import pytest

import genesis as gs


def create_capsule_mjcf(name, pos, euler, radius, half_length):
    """Helper function to create an MJCF file with a single capsule."""
    mjcf = ET.Element("mujoco", model=name)
    ET.SubElement(mjcf, "compiler", angle="degree")
    ET.SubElement(mjcf, "option", timestep="0.01")
    worldbody = ET.SubElement(mjcf, "worldbody")
    body = ET.SubElement(
        worldbody,
        "body",
        name=name,
        pos=f"{pos[0]} {pos[1]} {pos[2]}",
        euler=f"{euler[0]} {euler[1]} {euler[2]}",
    )
    ET.SubElement(body, "geom", type="capsule", size=f"{radius} {half_length}")
    ET.SubElement(body, "joint", name=f"{name}_joint", type="free")
    return mjcf


def create_modified_narrowphase_file():
    """
    Create a modified version of narrowphase.py that forces capsule collisions to use GJK.
    
    Returns:
        str: Path to the temporary modified narrowphase.py file
    """
    import random
    
    # Find the original narrowphase.py file
    import genesis.engine.solvers.rigid.collider.narrowphase as narrowphase_module
    narrowphase_path = narrowphase_module.__file__
    
    # Read the original file
    with open(narrowphase_path, 'r') as f:
        content = f.read()
    
    # Replace relative imports with absolute imports
    content = content.replace('from .', 'from genesis.engine.solvers.rigid.collider.')
    
    # Replace the capsule-capsule contact call with GJK
    content = content.replace(
        'is_col, normal, contact_pos, penetration = capsule_contact.func_capsule_capsule_contact(',
        '# MODIFIED: Use GJK instead of analytical\n' +
        '                    # is_col, normal, contact_pos, penetration = capsule_contact.func_capsule_capsule_contact(\n' +
        '                    errno[i_b] |= 1 << 16  # Mark that we forced GJK for capsule-capsule\n' +
        '                    prefer_gjk = True  # Force GJK for capsule-capsule\n' +
        '                    if False:  # Skip analytical path\n' +
        '                        is_col, normal, contact_pos, penetration = capsule_contact.func_capsule_capsule_contact('
    )
    
    # For sphere-capsule, replace the call
    content = content.replace(
        'is_col, normal, contact_pos, penetration = capsule_contact.func_sphere_capsule_contact(',
        '# MODIFIED: Use GJK instead of analytical\n' +
        '                    # is_col, normal, contact_pos, penetration = capsule_contact.func_sphere_capsule_contact(\n' +
        '                    errno[i_b] |= 1 << 17  # Mark that we forced GJK for sphere-capsule\n' +
        '                    prefer_gjk = True  # Force GJK for sphere-capsule\n' +
        '                    if False:  # Skip analytical path\n' +
        '                        is_col, normal, contact_pos, penetration = capsule_contact.func_sphere_capsule_contact('
    )
    
    # Write to /tmp with random integer
    randint = random.randint(0, 1000000)
    temp_narrowphase_path = f'/tmp/narrow_{randint}.py'
    
    with open(temp_narrowphase_path, 'w') as f:
        f.write(content)
    
    print(f"Modified narrowphase written to: {temp_narrowphase_path}")
    
    return temp_narrowphase_path


@pytest.mark.parametrize("backend", [gs.cpu])
@pytest.mark.parametrize(
    "pos1,euler1,pos2,euler2,should_collide,description",
    [
        # Test 1: Vertical and horizontal capsules intersecting
        # Capsule 1: vertical at origin, Capsule 2: horizontal at x=0.15
        # Distance between axes: 0.15, sum of radii: 0.2 → should collide
        ((0, 0, 0), (0, 0, 0), (0.15, 0, 0), (0, 90, 0), True, "perpendicular_close"),
        # Test 2: Parallel capsules far apart
        # Distance: 1.0, sum of radii: 0.2 → no collision
        ((0, 0, 0), (0, 0, 0), (1.0, 0, 0), (0, 0, 0), False, "parallel_far"),
        # Test 3: Parallel capsules exactly touching
        # Distance: 0.2, sum of radii: 0.2 → edge case, may or may not detect
        ((0, 0, 0), (0, 0, 0), (0.19, 0, 0), (0, 0, 0), True, "parallel_touching"),
        # Test 4: Perpendicular capsules at same position (definitely colliding)
        # Axes intersect at center → should collide
        ((0, 0, 0), (0, 0, 0), (0, 0, 0), (90, 0, 0), True, "perpendicular_center"),
        # Test 5: Capsules offset diagonally (not touching)
        # Distance ~0.42, sum of radii: 0.2 → no collision
        ((0, 0, 0), (0, 0, 0), (0.3, 0.3, 0), (0, 0, 0), False, "diagonal_separated"),
    ],
)
def test_capsule_capsule_vs_gjk(backend, pos1, euler1, pos2, euler2, should_collide, description, monkeypatch):
    """
    Compare analytical capsule-capsule collision with GJK by monkey-patching narrowphase.
    """
    radius = 0.1
    half_length = 0.25
    
    # Scene 1: Using ORIGINAL analytical capsule-capsule detection (before any monkey-patching)
    scene_analytical = gs.Scene(
        show_viewer=False,
        rigid_options=gs.options.RigidOptions(
            dt=0.01,
            gravity=(0, 0, 0),
            use_gjk_collision=False,  # Use analytical methods
        ),
    )

    with tempfile.TemporaryDirectory() as tmpdir_mjcf:
        mjcf1 = create_capsule_mjcf("capsule1", pos1, euler1, radius, half_length)
        mjcf1_path = os.path.join(tmpdir_mjcf, "capsule1_analytical.xml")
        ET.ElementTree(mjcf1).write(mjcf1_path)
        scene_analytical.add_entity(gs.morphs.MJCF(file=mjcf1_path))

        mjcf2 = create_capsule_mjcf("capsule2", pos2, euler2, radius, half_length)
        mjcf2_path = os.path.join(tmpdir_mjcf, "capsule2_analytical.xml")
        ET.ElementTree(mjcf2).write(mjcf2_path)
        scene_analytical.add_entity(gs.morphs.MJCF(file=mjcf2_path))

        scene_analytical.build()
    
    # NOW monkey-patch for the GJK scene
    temp_narrowphase_path = create_modified_narrowphase_file()
    
    # Import the modified module and monkey-patch func_convex_convex_contact
    import importlib.util
    spec = importlib.util.spec_from_file_location("narrowphase_modified", temp_narrowphase_path)
    narrowphase_modified = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(narrowphase_modified)
    
    # Monkey-patch the narrowphase module to use modified function
    from genesis.engine.solvers.rigid.collider import narrowphase
    monkeypatch.setattr(
        narrowphase,
        "func_convex_convex_contact",
        narrowphase_modified.func_convex_convex_contact
    )

    # Scene 2: Force GJK for capsules (using modified narrowphase)
    scene_gjk = gs.Scene(
        show_viewer=False,
        rigid_options=gs.options.RigidOptions(
            dt=0.01,
            gravity=(0, 0, 0),
            use_gjk_collision=False,  # Still use_gjk_collision=False, but capsules will use GJK due to our patch
        ),
    )

    with tempfile.TemporaryDirectory() as tmpdir_mjcf:
        # Add same capsules to GJK scene
        mjcf1 = create_capsule_mjcf("capsule1", pos1, euler1, radius, half_length)
        mjcf1_path = os.path.join(tmpdir_mjcf, "capsule1_gjk.xml")
        ET.ElementTree(mjcf1).write(mjcf1_path)
        scene_gjk.add_entity(gs.morphs.MJCF(file=mjcf1_path))

        mjcf2 = create_capsule_mjcf("capsule2", pos2, euler2, radius, half_length)
        mjcf2_path = os.path.join(tmpdir_mjcf, "capsule2_gjk.xml")
        ET.ElementTree(mjcf2).write(mjcf2_path)
        scene_gjk.add_entity(gs.morphs.MJCF(file=mjcf2_path))

        scene_gjk.build()

    scene_analytical.step()
    scene_gjk.step()

    # Print errno values to see which code path was used
    print(f"\nTest: {description}")
    print(f"errno analytical: {scene_analytical._sim.rigid_solver._errno}")
    print(f"errno gjk: {scene_gjk._sim.rigid_solver._errno}")
    
    # Check if capsule-capsule path was used (bit 16)
    analytical_used_modified_capsule = (scene_analytical._sim.rigid_solver._errno[0] & (1 << 16)) != 0
    gjk_used_modified_capsule = (scene_gjk._sim.rigid_solver._errno[0] & (1 << 16)) != 0
    
    # Check if sphere-capsule path was used (bit 17)
    analytical_used_modified_sphere = (scene_analytical._sim.rigid_solver._errno[0] & (1 << 17)) != 0
    gjk_used_modified_sphere = (scene_gjk._sim.rigid_solver._errno[0] & (1 << 17)) != 0
    
    print(f"Analytical scene used modified capsule-capsule path (bit 16): {analytical_used_modified_capsule}")
    print(f"Analytical scene used modified sphere-capsule path (bit 17): {analytical_used_modified_sphere}")
    print(f"GJK scene used modified capsule-capsule path (bit 16): {gjk_used_modified_capsule}")
    print(f"GJK scene used modified sphere-capsule path (bit 17): {gjk_used_modified_sphere}")

    contacts_analytical = scene_analytical.rigid_solver.collider.get_contacts(as_tensor=False)
    contacts_gjk = scene_gjk.rigid_solver.collider.get_contacts(as_tensor=False)

    has_collision_analytical = contacts_analytical is not None and len(contacts_analytical["geom_a"]) > 0
    has_collision_gjk = contacts_gjk is not None and len(contacts_gjk["geom_a"]) > 0

    assert has_collision_analytical == has_collision_gjk, (
        f"Collision detection mismatch! Analytical: {has_collision_analytical}, GJK: {has_collision_gjk}"
    )

    if has_collision_analytical and has_collision_gjk:
        # Get first contact from each (may have multiple due to multi-contact)
        pen_analytical = contacts_analytical["penetration"][0]
        pen_gjk = contacts_gjk["penetration"][0]

        normal_analytical = np.array(contacts_analytical["normal"][0])
        normal_gjk = np.array(contacts_gjk["normal"][0])

        pos_analytical = np.array(contacts_analytical["position"][0])
        pos_gjk = np.array(contacts_gjk["position"][0])

        pen_tol = max(0.01, 0.1 * max(pen_analytical, pen_gjk))
        assert abs(pen_analytical - pen_gjk) < pen_tol, (
            f"Penetration mismatch! Analytical: {pen_analytical:.6f}, GJK: {pen_gjk:.6f}, diff: {abs(pen_analytical - pen_gjk):.6f}"
        )

        normal_agreement = abs(np.dot(normal_analytical, normal_gjk))
        assert normal_agreement > 0.95, (
            f"Normal direction mismatch! Analytical: {normal_analytical}, GJK: {normal_gjk}, agreement: {normal_agreement:.4f}"
        )

        pos_diff = np.linalg.norm(pos_analytical - pos_gjk)
        assert pos_diff < 0.05, f"Contact position mismatch! Diff: {pos_diff:.6f}"


@pytest.mark.parametrize("backend", [gs.cpu])
def test_capsule_analytical_accuracy(backend):
    """
    Test that analytical capsule-capsule gives exact results for simple cases.
    """
    # Simple test case: two vertical capsules offset horizontally
    # Capsule 1: center at origin, radius=0.1, half_length=0.25
    # Capsule 2: center at (0.15, 0, 0), same size
    # Line segments are both vertical, closest points are at centers
    # Distance between segments: 0.15
    # Sum of radii: 0.2
    # Expected penetration: 0.2 - 0.15 = 0.05

    scene = gs.Scene(
        show_viewer=False,
        rigid_options=gs.options.RigidOptions(
            dt=0.01,
            gravity=(0, 0, 0),
        ),
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        mjcf1 = create_capsule_mjcf("capsule1", (0, 0, 0), (0, 0, 0), 0.1, 0.25)
        mjcf1_path = os.path.join(tmpdir, "capsule1.xml")
        ET.ElementTree(mjcf1).write(mjcf1_path)
        scene.add_entity(gs.morphs.MJCF(file=mjcf1_path))

        mjcf2 = create_capsule_mjcf("capsule2", (0.15, 0, 0), (0, 0, 0), 0.1, 0.25)
        mjcf2_path = os.path.join(tmpdir, "capsule2.xml")
        ET.ElementTree(mjcf2).write(mjcf2_path)
        scene.add_entity(gs.morphs.MJCF(file=mjcf2_path))

        scene.build()

    scene.step()

    contacts = scene.rigid_solver.collider.get_contacts(as_tensor=False)

    assert contacts is not None and len(contacts["geom_a"]) > 0

    penetration = contacts["penetration"][0]
    expected_pen = 0.05

    assert abs(penetration - expected_pen) < 1e-5, (
        f"Analytical solution not exact! Expected: {expected_pen}, Got: {penetration:.6f}"
    )

    normal = np.array(contacts["normal"][0])

    # Check normal is along X axis
    assert abs(abs(normal[0]) - 1.0) < 1e-5, f"Normal should be along X axis, got {normal}"
    assert abs(normal[1]) < 1e-5 and abs(normal[2]) < 1e-5, f"Normal should be along X axis, got {normal}"
