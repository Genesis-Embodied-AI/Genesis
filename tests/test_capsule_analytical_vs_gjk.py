"""
Unit test comparing analytical capsule-capsule contact detection with GJK.

NOTE: These tests should ideally be run with debug mode enabled (pytest --dev)
to verify that the correct collision detection code paths are being used.

Debug mode enables collision path tracking counters in the narrowphase kernel that
empirically verify which algorithm (analytical vs GJK) was actually executed, not just
which was configured. Without debug mode, the test can only verify configuration flags,
not actual execution paths, making the test less robust.
"""

import os
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


@pytest.mark.debug(True)
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
def test_capsule_capsule_vs_gjk(backend, pos1, euler1, pos2, euler2, should_collide, description):
    """
    Compare analytical capsule-capsule collision with GJK.
    
    This test creates two scenes with identical capsule configurations:
    - One using analytical capsule-capsule detection (default)
    - One forcing GJK for all collisions
    
    Then compares the collision results and VERIFIES which code path was used
    by checking debug counters.
    """
    radius = 0.1
    half_length = 0.25
    
    # Scene 1: Using analytical capsule-capsule detection
    scene_analytical = gs.Scene(
        show_viewer=False,
        rigid_options=gs.options.RigidOptions(
            dt=0.01,
            gravity=(0, 0, 0),
            use_gjk_collision=False,  # Use analytical when available
        ),
    )
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Add capsules to analytical scene
        mjcf1 = create_capsule_mjcf("capsule1", pos1, euler1, radius, half_length)
        mjcf1_path = os.path.join(tmpdir, "capsule1_analytical.xml")
        ET.ElementTree(mjcf1).write(mjcf1_path)
        scene_analytical.add_entity(gs.morphs.MJCF(file=mjcf1_path))
        
        mjcf2 = create_capsule_mjcf("capsule2", pos2, euler2, radius, half_length)
        mjcf2_path = os.path.join(tmpdir, "capsule2_analytical.xml")
        ET.ElementTree(mjcf2).write(mjcf2_path)
        scene_analytical.add_entity(gs.morphs.MJCF(file=mjcf2_path))
        
        scene_analytical.build()
    
    # Scene 2: Force GJK for comparison (GJK is more accurate than MPR)
    scene_gjk = gs.Scene(
        show_viewer=False,
        rigid_options=gs.options.RigidOptions(
            dt=0.01,
            gravity=(0, 0, 0),
            use_gjk_collision=True,  # Force GJK for reference
        ),
    )
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Add same capsules to GJK scene
        mjcf1 = create_capsule_mjcf("capsule1", pos1, euler1, radius, half_length)
        mjcf1_path = os.path.join(tmpdir, "capsule1_gjk.xml")
        ET.ElementTree(mjcf1).write(mjcf1_path)
        scene_gjk.add_entity(gs.morphs.MJCF(file=mjcf1_path))
        
        mjcf2 = create_capsule_mjcf("capsule2", pos2, euler2, radius, half_length)
        mjcf2_path = os.path.join(tmpdir, "capsule2_gjk.xml")
        ET.ElementTree(mjcf2).write(mjcf2_path)
        scene_gjk.add_entity(gs.morphs.MJCF(file=mjcf2_path))
        
        scene_gjk.build()
    
    # Run one step to detect collisions
    scene_analytical.step()
    scene_gjk.step()
    
    # VERIFY: Check debug counters to ensure correct paths were taken (only in debug mode)
    # In debug mode, __debug__ is True and counters are active
    import gstaichi as ti
    if hasattr(ti.lang._template_mapper.__builtins__, '__debug__') and ti.lang._template_mapper.__builtins__['__debug__']:
        analytical_capsule_count = scene_analytical.rigid_solver.collider.collider_state.debug_analytical_capsule_count[0]
        analytical_gjk_count = scene_analytical.rigid_solver.collider.collider_state.debug_gjk_count[0]
        gjk_scene_capsule_count = scene_gjk.rigid_solver.collider.collider_state.debug_analytical_capsule_count[0]
        gjk_scene_gjk_count = scene_gjk.rigid_solver.collider.collider_state.debug_gjk_count[0]
        
        # Scene 1 (analytical) should use analytical path, NOT GJK
        assert analytical_capsule_count > 0, \
            f"Scene 1 should have used analytical capsule path (count={analytical_capsule_count})"
        assert analytical_gjk_count == 0, \
            f"Scene 1 should NOT have used GJK path (count={analytical_gjk_count})"
        
        # Scene 2 (GJK) should use GJK path, NOT analytical
        assert gjk_scene_gjk_count > 0, \
            f"Scene 2 should have used GJK path (count={gjk_scene_gjk_count})"
        assert gjk_scene_capsule_count == 0, \
            f"Scene 2 should NOT have used analytical capsule path (count={gjk_scene_capsule_count})"
    
    # Get contacts from both methods
    contacts_analytical = scene_analytical.rigid_solver.collider.get_contacts(as_tensor=False)
    contacts_gjk = scene_gjk.rigid_solver.collider.get_contacts(as_tensor=False)
    
    # Check if collision detection agrees
    has_collision_analytical = contacts_analytical is not None and len(contacts_analytical['geom_a']) > 0
    has_collision_gjk = contacts_gjk is not None and len(contacts_gjk['geom_a']) > 0
    
    # Both should agree on whether collision exists
    assert has_collision_analytical == has_collision_gjk, \
        f"Collision detection mismatch! Analytical: {has_collision_analytical}, GJK: {has_collision_gjk}"
    
    # If there is a collision, compare the details
    if has_collision_analytical and has_collision_gjk:
        # Get first contact from each (may have multiple due to multi-contact)
        pen_analytical = contacts_analytical['penetration'][0]
        pen_gjk = contacts_gjk['penetration'][0]
        
        normal_analytical = np.array(contacts_analytical['normal'][0])
        normal_gjk = np.array(contacts_gjk['normal'][0])
        
        pos_analytical = np.array(contacts_analytical['position'][0])
        pos_gjk = np.array(contacts_gjk['position'][0])
        
        # Check that penetration depths are similar (within 10% or 0.01 units)
        pen_tol = max(0.01, 0.1 * max(pen_analytical, pen_gjk))
        assert abs(pen_analytical - pen_gjk) < pen_tol, \
            f"Penetration mismatch! Analytical: {pen_analytical:.6f}, GJK: {pen_gjk:.6f}, diff: {abs(pen_analytical - pen_gjk):.6f}"
        
        # Normals should be aligned (dot product > 0.95)
        normal_agreement = abs(np.dot(normal_analytical, normal_gjk))
        assert normal_agreement > 0.95, \
            f"Normal direction mismatch! Analytical: {normal_analytical}, GJK: {normal_gjk}, agreement: {normal_agreement:.4f}"
        
        # Contact positions should be close (within 0.05 units)
        pos_diff = np.linalg.norm(pos_analytical - pos_gjk)
        assert pos_diff < 0.05, \
            f"Contact position mismatch! Diff: {pos_diff:.6f}"


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
    
    assert contacts is not None and len(contacts['geom_a']) > 0
    
    # Check penetration is correct (should be 0.05)
    penetration = contacts['penetration'][0]
    expected_pen = 0.05
    
    # Analytical solution should be exact (within numerical precision)
    assert abs(penetration - expected_pen) < 1e-5, \
        f"Analytical solution not exact! Expected: {expected_pen}, Got: {penetration:.6f}"
    
    # Normal should point in X direction [1, 0, 0] or [-1, 0, 0]
    normal = np.array(contacts['normal'][0])
    
    # Check normal is along X axis
    assert abs(abs(normal[0]) - 1.0) < 1e-5, f"Normal should be along X axis, got {normal}"
    assert abs(normal[1]) < 1e-5 and abs(normal[2]) < 1e-5, f"Normal should be along X axis, got {normal}"
