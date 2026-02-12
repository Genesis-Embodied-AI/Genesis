"""
Unit test comparing analytical capsule-capsule contact detection with GJK.

This test creates a modified version of narrowphase.py in a temporary file that
forces capsule-capsule and sphere-capsule collisions to use GJK instead of
analytical methods, allowing direct comparison between the two approaches.
"""

# import gstaichi as ti
import os
import gstaichi.lang.impl as impl
# import sys
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


def find_and_disable_condition(lines, function_name):
    """Find function call, look back for if/elif, and disable the entire multi-line condition."""
    # Find the line with the function call
    call_line_idx = None
    for i, line in enumerate(lines):
        if function_name in line and "(" in line:
            call_line_idx = i
            break

    if call_line_idx is None:
        raise ValueError(f"Could not find function call: {function_name}")

    # Look backwards to find the if or elif line
    condition_line_idx = None
    for i in range(call_line_idx - 1, -1, -1):
        stripped = lines[i].strip()
        if stripped.startswith("if ") or stripped.startswith("elif "):
            condition_line_idx = i
            break
        # Stop if we hit another major control structure
        if stripped.startswith("else:"):
            break

    if condition_line_idx is None:
        raise ValueError(f"Could not find if/elif for {function_name}")

    # Find the end of the condition (look for the : that ends it)
    condition_end_idx = condition_line_idx
    for i in range(condition_line_idx, call_line_idx):
        if ":" in lines[i]:
            condition_end_idx = i
            break

    # Modify the condition to wrap entire thing in False and (...)
    original_line = lines[condition_line_idx]
    indent = len(original_line) - len(original_line.lstrip())
    indent_str = original_line[:indent]

    # Extract the condition part (after if/elif and before :)
    if original_line.strip().startswith("if "):
        prefix = "if "
        rest = original_line.strip()[3:]  # Remove 'if '
    elif original_line.strip().startswith("elif "):
        prefix = "elif "
        rest = original_line.strip()[5:]  # Remove 'elif '
    else:
        raise ValueError(f"Expected if/elif but got: {original_line}")

    # If single-line condition
    if condition_end_idx == condition_line_idx:
        # Simple case: add False and
        modified_line = f"{indent_str}{prefix}False and {rest}"
        lines[condition_line_idx] = modified_line
    else:
        # Multi-line condition: wrap in False and (...)
        rest_no_colon = rest.rstrip(":").rstrip()
        lines[condition_line_idx] = f"{indent_str}{prefix}False and ({rest_no_colon}"
        
        # Add closing ) before the : on the last line
        last_line = lines[condition_end_idx]
        if ":" in last_line:
            # Insert ) before the :
            lines[condition_end_idx] = last_line.replace(":", "):", 1)

    return lines


def insert_errno_before_call(lines, function_call_pattern, errno_bit, comment):
    """Insert errno marker on the line before a function call."""
    call_line_idx = None
    for i, line in enumerate(lines):
        if function_call_pattern in line:
            # Find the position of the pattern in the line
            idx = line.find(function_call_pattern)
            if idx != -1:
                # Make sure it's not part of a longer identifier
                # Check that the character before the pattern (if any) is not alphanumeric or underscore
                if idx == 0 or not (line[idx - 1].isalnum() or line[idx - 1] == "_"):
                    call_line_idx = i
                    break

    if call_line_idx is None:
        raise ValueError(f"Could not find function call: {function_call_pattern}")

    # Get indentation from the call line
    indent = len(lines[call_line_idx]) - len(lines[call_line_idx].lstrip())
    indent_str = lines[call_line_idx][:indent]

    # Insert errno marker on the line before the call
    errno_line = f"{indent_str}errno[i_b] |= 1 << {errno_bit}  # {comment}"
    lines.insert(call_line_idx, errno_line)

    return lines


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

    with open(narrowphase_path, "r") as f:
        content = f.read()

    # remove relative imports
    content = content.replace("from . import ", "from genesis.engine.solvers.rigid.collider import ")
    content = content.replace("from .", "from genesis.engine.solvers.rigid.collider.")

    # disable fastcache
    content = content.replace("@ti.kernel(fastcache=gs.use_fastcache)", "@ti.kernel()")

    lines = content.split("\n")

    # Disable capsule-capsule analytical path
    lines = find_and_disable_condition(lines, "capsule_contact.func_capsule_capsule_contact")

    # Disable sphere-capsule analytical path
    lines = find_and_disable_condition(lines, "capsule_contact.func_sphere_capsule_contact")

    # Insert errno before GJK calls
    lines = insert_errno_before_call(
        lines, "diff_gjk.func_gjk_contact(", 16, "MODIFIED: GJK called for collision detection"
    )
    lines = insert_errno_before_call(lines, "gjk.func_gjk_contact(", 16, "MODIFIED: GJK called for collision detection")

    content = "\n".join(lines)

    randint = random.randint(0, 1000000)
    temp_narrowphase_path = f"/tmp/narrow_{randint}.py"

    with open(temp_narrowphase_path, "w") as f:
        f.write(content)

    print(f"Modified narrowphase written to: {temp_narrowphase_path}")

    return temp_narrowphase_path


@pytest.mark.parametrize("backend", [gs.cpu, gs.gpu])
@pytest.mark.parametrize(
    "pos1,euler1,pos2,euler2,should_collide,description",
    [
        # Test 1: Vertical and horizontal capsules intersecting
        # Capsule 1: vertical at origin, Capsule 2: horizontal at x=0.15
        # Distance between axes: 0.15, sum of radii: 0.2 → should collide
        ((0, 0, 0), (0, 0, 0), (0.15, 0, 0), (0, 90, 0), True, "perpendicular_close"),
        # Test 2: Parallel capsules with light contact
        # Distance: 0.18, sum of radii: 0.2 → penetration = 0.02 (light contact)
        ((0, 0, 0), (0, 0, 0), (0.18, 0, 0), (0, 0, 0), True, "parallel_light"),
        # Test 3: Diagonal capsule near vertical capsule (AABBs overlap, no collision)
        # Capsule 1 vertical at origin, Capsule 2 rotated 60° at X=0.4
        # 60° rotation creates AABB X: [-0.317, 0.317], so at X=0.4, AABB: [0.083, 0.717]
        # This overlaps with vertical capsule AABB [-0.1, 0.1], but distance > 0.2 (no collision)
        ((0, 0, 0), (0, 0, 0), (0.4, 0, 0), (0, 60, 0), False, "diagonal_near"),
        # Test 4: Parallel capsules with deep penetration (for multicontact)
        # Distance: 0.15, sum of radii: 0.2 → penetration = 0.05 (deeper for multicontact)
        ((0, 0, 0), (0, 0, 0), (0.15, 0, 0), (0, 0, 0), True, "parallel_deep"),
        # Test 5: Perpendicular capsules at same position (definitely colliding)
        # Axes intersect at center → should collide
        ((0, 0, 0), (0, 0, 0), (0, 0, 0), (90, 0, 0), True, "perpendicular_center"),
        # Test 6: Diagonal penetration - one rotated 45 degrees
        # Rotation creates larger AABB, ensuring collision
        ((0, 0, 0), (0, 0, 0), (0.15, 0, 0), (0, 45, 0), True, "diagonal_rotated"),
    ],
)
def test_capsule_capsule_vs_gjk(pos1, euler1, pos2, euler2, should_collide, description, monkeypatch):
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

    monkeypatch.setattr(narrowphase, "func_convex_convex_contact", narrowphase_modified.func_convex_convex_contact)

    # CRITICAL: Clear materialized kernel cache to force recompilation with monkey-patched function
    # The narrowphase.func_convex_convex_contact is a Kernel object with a materialized_kernels cache
    # that maps (func, template_slot_locations, autodiff_mode) -> compiled kernel
    # We must clear this cache to force Taichi to recompile with the patched function
    import gstaichi.lang.impl as impl
    if hasattr(narrowphase.func_convex_convex_contact, 'materialized_kernels'):
        narrowphase.func_convex_convex_contact.materialized_kernels.clear()

    # Scene 2: Force GJK for capsules (using modified narrowphase)
    scene_gjk = gs.Scene(
        show_viewer=False,
        rigid_options=gs.options.RigidOptions(
            dt=0.01,
            gravity=(0, 0, 0),
            use_gjk_collision=True,
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

    # Verify errno values to ensure correct code path was used
    print(f"\nTest: {description}")

    # Check if GJK was used (bit 16)
    analytical_used_gjk = (scene_analytical._sim.rigid_solver._errno[0] & (1 << 16)) != 0
    gjk_used_gjk = (scene_gjk._sim.rigid_solver._errno[0] & (1 << 16)) != 0

    # Verify that analytical scene did NOT use GJK, and GJK scene DID use GJK
    assert not analytical_used_gjk, (
        f"Analytical scene should not use GJK (errno={scene_analytical._sim.rigid_solver._errno[0]})"
    )
    assert gjk_used_gjk, f"GJK scene should use GJK (errno={scene_gjk._sim.rigid_solver._errno[0]})"

    contacts_analytical = scene_analytical.rigid_solver.collider.get_contacts(as_tensor=False, to_torch=False)
    contacts_gjk = scene_gjk.rigid_solver.collider.get_contacts(as_tensor=False, to_torch=False)

    has_collision_analytical = contacts_analytical is not None and len(contacts_analytical["geom_a"]) > 0
    has_collision_gjk = contacts_gjk is not None and len(contacts_gjk["geom_a"]) > 0

    # All test cases should result in collision
    assert has_collision_analytical, f"Analytical scene should detect collision for test '{description}'"
    assert has_collision_gjk, f"GJK scene should detect collision for test '{description}'"

    # First check that both methods agree on whether there's a collision
    assert has_collision_analytical == has_collision_gjk, (
        f"Collision detection mismatch! Analytical: {has_collision_analytical}, GJK: {has_collision_gjk}"
    )
    assert has_collision_analytical == should_collide

    # If both detected a collision, compare the contact details
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

        # For parallel/near-parallel capsules, the contact position is ambiguous
        # (any point along the overlapping region is equally valid). Instead of checking
        # exact position match, verify both contact points lie on the line connecting the surfaces.
        pos_diff = np.linalg.norm(pos_analytical - pos_gjk)
        if description in ["parallel_light", "parallel_deep"]:
            # For parallel capsules with sufficient penetration, multicontact may generate multiple contacts
            n_analytical = len(contacts_analytical["geom_a"])
            n_gjk = len(contacts_gjk["geom_a"])

            # If we have multiple contacts, validate they're consistent
            if n_analytical >= 2 or n_gjk >= 2:
                # Check that each analytical contact is near at least one GJK contact
                all_analytical_positions = np.array([contacts_analytical["position"][i] for i in range(n_analytical)])
                all_gjk_positions = np.array([contacts_gjk["position"][i] for i in range(n_gjk)])

                for i, pos_a in enumerate(all_analytical_positions):
                    # Find closest GJK contact to this analytical contact
                    min_dist = min(np.linalg.norm(pos_a - pos_g) for pos_g in all_gjk_positions)
                    assert min_dist < 0.1, (
                        f"Analytical contact {i} at {pos_a} not matched by any GJK contact (min_dist={min_dist:.6f})"
                    )

                # For parallel vertical capsules, verify contacts are on the line between axes
                # All contacts should have same X,Y but can have different Z
                expected_xy = np.array([0.075, 0.0])  # Midpoint between x=0 and x=0.15
                for i in range(n_analytical):
                    pos_a = all_analytical_positions[i]
                    xy_dist = np.linalg.norm(pos_a[:2] - expected_xy)
                    assert xy_dist < 0.02, f"Analytical contact {i} X,Y={pos_a[:2]} should be near {expected_xy}"
                    assert -0.26 < pos_a[2] < 0.26, f"Analytical contact {i} Z={pos_a[2]:.3f} outside capsule range"

                for i in range(n_gjk):
                    pos_g = all_gjk_positions[i]
                    xy_dist = np.linalg.norm(pos_g[:2] - expected_xy)
                    assert xy_dist < 0.02, f"GJK contact {i} X,Y={pos_g[:2]} should be near {expected_xy}"
                    assert -0.26 < pos_g[2] < 0.26, f"GJK contact {i} Z={pos_g[2]:.3f} outside capsule range"
            else:
                # Single contact case - just verify position is reasonable
                assert pos_diff < 0.1, f"Contact position mismatch for parallel capsules! Diff: {pos_diff:.6f}"
        else:
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


def create_sphere_mjcf(name, pos, radius):
    """Helper function to create an MJCF file with a single sphere."""
    mjcf = ET.Element("mujoco", model=name)
    ET.SubElement(mjcf, "compiler", angle="degree")
    ET.SubElement(mjcf, "option", timestep="0.01")
    worldbody = ET.SubElement(mjcf, "worldbody")
    body = ET.SubElement(
        worldbody,
        "body",
        name=name,
        pos=f"{pos[0]} {pos[1]} {pos[2]}",
    )
    ET.SubElement(body, "geom", type="sphere", size=f"{radius}")
    ET.SubElement(body, "joint", name=f"{name}_joint", type="free")
    return mjcf


@pytest.mark.parametrize("backend", [gs.cpu])
@pytest.mark.parametrize(
    "sphere_pos,capsule_pos,capsule_euler,should_collide,description",
    [
        # Test 1: Sphere above vertical capsule, touching the top
        # Sphere at (0, 0, 0.4), capsule vertical at origin
        # Distance from sphere center to capsule top: 0.4 - 0.25 = 0.15
        # Sum of radii: 0.1 + 0.1 = 0.2 → should collide (penetration = 0.05)
        ((0, 0, 0.4), (0, 0, 0), (0, 0, 0), True, "sphere_above_capsule_top"),
        # Test 2: Sphere close to capsule (light contact)
        # Sphere at (0.18, 0, 0), capsule vertical at origin
        # Distance to axis: 0.18, sum of radii: 0.2 → light penetration
        ((0.18, 0, 0), (0, 0, 0), (0, 0, 0), True, "sphere_close_to_capsule"),
        # Test 3: Sphere near cylinder but not touching (AABBs overlap)
        # Sphere at 45° in XY plane, distance = r1+r2+4*EPS = 0.24
        # Position: (0.17, 0.17, 0), AABBs overlap but no collision
        ((0.17, 0.17, 0), (0, 0, 0), (0, 0, 0), False, "sphere_near_cylinder"),
        # Test 4: Sphere near spherical cap but not touching (AABBs overlap)
        # Capsule rotated 45° around Y, sphere beyond top cap along axis
        # Cap at (0.177, 0, 0.177), sphere at (0.35, 0, 0.35), distance = r1+r2+4*EPS
        # Rotation creates larger AABB ensuring overlap, but no collision
        ((0.35, 0, 0.35), (0, 0, 0), (0, 45, 0), False, "sphere_near_cap"),
        # Test 5: Sphere touching capsule cylindrical surface
        # Sphere at (0.15, 0, 0), capsule vertical at origin
        # Distance to axis: 0.15, sum of radii: 0.2 → good penetration
        ((0.15, 0, 0), (0, 0, 0), (0, 0, 0), True, "sphere_touching_cylinder"),
        # Test 5: Sphere at capsule center (deep penetration)
        # Sphere at origin, capsule vertical at origin
        # Maximum penetration
        ((0, 0, 0), (0, 0, 0), (0, 0, 0), True, "sphere_at_capsule_center"),
        # Test 6: Sphere near capsule endpoint (hemispherical cap)
        # Sphere at (0.15, 0, 0.3), capsule vertical at origin
        # Tests collision with the rounded cap of the capsule
        ((0.15, 0, 0.3), (0, 0, 0), (0, 0, 0), True, "sphere_near_capsule_cap"),
        # Test 7: Sphere with horizontal capsule
        # Sphere at (0, 0.15, 0), capsule horizontal (rotated 90° around Y)
        # Distance to axis: 0.15, sum of radii: 0.2 → should collide
        ((0, 0.15, 0), (0, 0, 0), (0, 90, 0), True, "sphere_horizontal_capsule"),
    ],
)
def test_sphere_capsule_vs_gjk(
    backend, sphere_pos, capsule_pos, capsule_euler, should_collide, description, monkeypatch
):
    """
    Compare analytical sphere-capsule collision with GJK by monkey-patching narrowphase.
    """
    sphere_radius = 0.1
    capsule_radius = 0.1
    capsule_half_length = 0.25

    # Scene 1: Using ORIGINAL analytical sphere-capsule detection (before any monkey-patching)
    scene_analytical = gs.Scene(
        show_viewer=False,
        rigid_options=gs.options.RigidOptions(
            dt=0.01,
            gravity=(0, 0, 0),
        ),
    )

    with tempfile.TemporaryDirectory() as tmpdir_mjcf:
        sphere_mjcf = create_sphere_mjcf("sphere", sphere_pos, sphere_radius)
        sphere_path = os.path.join(tmpdir_mjcf, "sphere_analytical.xml")
        ET.ElementTree(sphere_mjcf).write(sphere_path)
        scene_analytical.add_entity(gs.morphs.MJCF(file=sphere_path))

        capsule_mjcf = create_capsule_mjcf("capsule", capsule_pos, capsule_euler, capsule_radius, capsule_half_length)
        capsule_path = os.path.join(tmpdir_mjcf, "capsule_analytical.xml")
        ET.ElementTree(capsule_mjcf).write(capsule_path)
        scene_analytical.add_entity(gs.morphs.MJCF(file=capsule_path))

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

    monkeypatch.setattr(narrowphase, "func_convex_convex_contact", narrowphase_modified.func_convex_convex_contact)

    # CRITICAL: Clear materialized kernel cache to force recompilation with monkey-patched function
    # The narrowphase.func_convex_convex_contact is a Kernel object with a materialized_kernels cache
    # that maps (func, template_slot_locations, autodiff_mode) -> compiled kernel
    # We must clear this cache to force Taichi to recompile with the patched function
    import gstaichi.lang.impl as impl
    if hasattr(narrowphase.func_convex_convex_contact, 'materialized_kernels'):
        narrowphase.func_convex_convex_contact.materialized_kernels.clear()

    # Scene 2: Force GJK for sphere-capsule (using modified narrowphase)
    scene_gjk = gs.Scene(
        show_viewer=False,
        rigid_options=gs.options.RigidOptions(
            dt=0.01,
            gravity=(0, 0, 0),
            use_gjk_collision=True,
        ),
    )

    with tempfile.TemporaryDirectory() as tmpdir_mjcf:
        # Add same objects to GJK scene
        sphere_mjcf = create_sphere_mjcf("sphere", sphere_pos, sphere_radius)
        sphere_path = os.path.join(tmpdir_mjcf, "sphere_gjk.xml")
        ET.ElementTree(sphere_mjcf).write(sphere_path)
        scene_gjk.add_entity(gs.morphs.MJCF(file=sphere_path))

        capsule_mjcf = create_capsule_mjcf("capsule", capsule_pos, capsule_euler, capsule_radius, capsule_half_length)
        capsule_path = os.path.join(tmpdir_mjcf, "capsule_gjk.xml")
        ET.ElementTree(capsule_mjcf).write(capsule_path)
        scene_gjk.add_entity(gs.morphs.MJCF(file=capsule_path))

        scene_gjk.build()

    scene_analytical.step()
    scene_gjk.step()

    # Verify errno values to ensure correct code path was used
    print(f"\nTest: {description}")

    # Check if GJK was used (bit 16)
    analytical_used_gjk = (scene_analytical._sim.rigid_solver._errno[0] & (1 << 16)) != 0
    gjk_used_gjk = (scene_gjk._sim.rigid_solver._errno[0] & (1 << 16)) != 0

    # Verify that analytical scene did NOT use GJK, and GJK scene DID use GJK
    assert not analytical_used_gjk, (
        f"Analytical scene should not use GJK (errno={scene_analytical._sim.rigid_solver._errno[0]})"
    )
    assert gjk_used_gjk, f"GJK scene should use GJK (errno={scene_gjk._sim.rigid_solver._errno[0]})"

    contacts_analytical = scene_analytical.rigid_solver.collider.get_contacts(as_tensor=False)
    contacts_gjk = scene_gjk.rigid_solver.collider.get_contacts(as_tensor=False)

    has_collision_analytical = contacts_analytical is not None and len(contacts_analytical["geom_a"]) > 0
    has_collision_gjk = contacts_gjk is not None and len(contacts_gjk["geom_a"]) > 0

    # First check that both methods agree on whether there's a collision
    assert has_collision_analytical == has_collision_gjk, (
        f"Collision detection mismatch! Analytical: {has_collision_analytical}, GJK: {has_collision_gjk}"
    )

    # If both detected a collision, compare the contact details
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
        # For degenerate cases (sphere at capsule center), the normal is ambiguous
        # Relax the normal agreement threshold for these cases
        normal_threshold = 0.5 if description == "sphere_at_capsule_center" else 0.95
        assert normal_agreement > normal_threshold, (
            f"Normal direction mismatch! Analytical: {normal_analytical}, GJK: {normal_gjk}, agreement: {normal_agreement:.4f}"
        )

        pos_diff = np.linalg.norm(pos_analytical - pos_gjk)
        assert pos_diff < 0.05, f"Contact position mismatch! Diff: {pos_diff:.6f}"
