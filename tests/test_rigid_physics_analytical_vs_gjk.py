"""
Unit test comparing analytical capsule-capsule contact detection with GJK.

This test creates a modified version of narrowphase.py in a temporary file that
forces capsule-capsule and sphere-capsule collisions to use GJK instead of
analytical methods, allowing direct comparison between the two approaches.
"""

import os
import importlib.util
import random
import tempfile
from typing import Callable
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
    # Find the original narrowphase.py file
    import genesis.engine.solvers.rigid.collider.narrowphase as narrowphase_module

    narrowphase_path = narrowphase_module.__file__

    with open(narrowphase_path, "r") as f:
        content = f.read()

    # remove relative imports
    content = content.replace("from . import ", "from genesis.engine.solvers.rigid.collider import ")
    content = content.replace("from .", "from genesis.engine.solvers.rigid.collider.")

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

    # Debug: Check if errno was actually inserted
    errno_count = content.count("errno[i_b] |= 1 << 16")
    assert errno_count >= 1

    randint = random.randint(0, 1000000)
    temp_narrowphase_path = f"/tmp/narrow_{randint}.py"

    with open(temp_narrowphase_path, "w") as f:
        f.write(content)

    print(f"Modified narrowphase written to: {temp_narrowphase_path}")

    return temp_narrowphase_path


@pytest.mark.parametrize("backend", [gs.cpu, gs.gpu])
def test_capsule_capsule_vs_gjk(backend, monkeypatch):
    """
    Compare analytical capsule-capsule collision with GJK by monkey-patching narrowphase.
    Tests multiple configurations with a single scene build (moving objects between tests).
    """
    # Define all test cases
    test_cases = [
        # (pos1, euler1, pos2, euler2, should_collide, description)
        ((0, 0, 0), (0, 0, 0), (0.15, 0, 0), (0, 90, 0), True, "perpendicular_close"),
        ((0, 0, 0), (0, 0, 0), (0.18, 0, 0), (0, 0, 0), True, "parallel_light"),
        ((0, 0, 0), (0, 90, 0), (0, 0.17, 0.17), (0, 90, 0), False, "horizontal_displaced"),
        ((0, 0, 0), (0, 0, 0), (0.15, 0, 0), (0, 0, 0), True, "parallel_deep"),
        ((0, 0, 0), (0, 0, 0), (0, 0, 0), (90, 0, 0), True, "perpendicular_center"),
        ((0, 0, 0), (0, 0, 0), (0.15, 0, 0), (0, 45, 0), True, "diagonal_rotated"),
    ]

    radius = 0.1
    half_length = 0.25

    # Build scenes once with initial configuration
    def build_scene(scene: gs.Scene):
        with tempfile.TemporaryDirectory() as tmpdir_mjcf:
            mjcf1 = create_capsule_mjcf("capsule1", (0, 0, 0), (0, 0, 0), radius, half_length)
            mjcf1_path = os.path.join(tmpdir_mjcf, "capsule1.xml")
            ET.ElementTree(mjcf1).write(mjcf1_path)
            entity1 = scene.add_entity(gs.morphs.MJCF(file=mjcf1_path))

            mjcf2 = create_capsule_mjcf("capsule2", (0, 0, 0), (0, 0, 0), radius, half_length)
            mjcf2_path = os.path.join(tmpdir_mjcf, "capsule2.xml")
            ET.ElementTree(mjcf2).write(mjcf2_path)
            entity2 = scene.add_entity(gs.morphs.MJCF(file=mjcf2_path))

            scene.build()
            return [entity1, entity2]

    scene_creator = AnalyticalVsGJKSceneCreator(monkeypatch=monkeypatch, build_scene=build_scene)
    scene_analytical, scene_gjk = scene_creator.setup_scenes_before()
    entities_analytical = scene_creator.entities
    entities_gjk = scene_creator.entities_gjk

    # Run all test cases
    for pos1, euler1, pos2, euler2, should_collide, description in test_cases:
        print(f"\nTest: {description}")

        try:
            # Set positions and orientations
            quat1 = gs.utils.geom.xyz_to_quat(xyz=np.array(euler1), degrees=True)
            quat2 = gs.utils.geom.xyz_to_quat(xyz=np.array(euler2), degrees=True)

            entities_analytical[0].set_qpos(np.array([*pos1, *quat1]))
            entities_analytical[1].set_qpos(np.array([*pos2, *quat2]))
            entities_gjk[0].set_qpos(np.array([*pos1, *quat1]))
            entities_gjk[1].set_qpos(np.array([*pos2, *quat2]))

            # Zero out velocities to prevent motion during step
            zero_vel = np.zeros(6)  # 3 linear + 3 angular
            entities_analytical[0].set_dofs_velocity(zero_vel)
            entities_analytical[1].set_dofs_velocity(zero_vel)
            entities_gjk[0].set_dofs_velocity(zero_vel)
            entities_gjk[1].set_dofs_velocity(zero_vel)

            scene_analytical.step()
            scene_gjk.step()

            scene_creator.checks_after()

            contacts_analytical = scene_analytical.rigid_solver.collider.get_contacts(as_tensor=False, to_torch=False)
            contacts_gjk = scene_gjk.rigid_solver.collider.get_contacts(as_tensor=False, to_torch=False)

            has_collision_analytical = contacts_analytical is not None and len(contacts_analytical["geom_a"]) > 0
            has_collision_gjk = contacts_gjk is not None and len(contacts_gjk["geom_a"]) > 0

            assert has_collision_analytical == has_collision_gjk, (
                f"Collision detection mismatch! Analytical: {has_collision_analytical}, GJK: {has_collision_gjk}"
            )
            assert has_collision_analytical == should_collide

            # If both detected a collision, compare the contact details
            if has_collision_analytical and has_collision_gjk:
                pen_analytical = contacts_analytical["penetration"][0]
                pen_gjk = contacts_gjk["penetration"][0]

                normal_analytical = np.array(contacts_analytical["normal"][0])
                normal_gjk = np.array(contacts_gjk["normal"][0])

                pos_analytical = np.array(contacts_analytical["position"][0])
                pos_gjk = np.array(contacts_gjk["position"][0])

                pen_tol = max(0.01, 0.1 * max(pen_analytical, pen_gjk))
                assert abs(pen_analytical - pen_gjk) < pen_tol, (
                    f"Penetration mismatch! Analytical: {pen_analytical:.6f}, GJK: {pen_gjk:.6f}"
                )

                normal_agreement = abs(np.dot(normal_analytical, normal_gjk))
                assert normal_agreement > 0.95, f"Normal mismatch! agreement: {normal_agreement:.4f}"

                pos_diff = np.linalg.norm(pos_analytical - pos_gjk)
                if description in ["parallel_light", "parallel_deep"]:
                    n_analytical = len(contacts_analytical["geom_a"])
                    n_gjk = len(contacts_gjk["geom_a"])

                    if n_analytical >= 2 or n_gjk >= 2:
                        all_analytical_positions = np.array(
                            [contacts_analytical["position"][i] for i in range(n_analytical)]
                        )
                        all_gjk_positions = np.array([contacts_gjk["position"][i] for i in range(n_gjk)])

                        for i, pos_a in enumerate(all_analytical_positions):
                            min_dist = min(np.linalg.norm(pos_a - pos_g) for pos_g in all_gjk_positions)
                            assert min_dist < 0.1

                        # For parallel vertical capsules, verify contacts are on the line between axes
                        if euler1 == (0, 0, 0) and euler2 == (0, 0, 0):  # Both vertical
                            expected_xy = np.array([pos2[0] / 2, 0.0])  # Midpoint between capsules
                            for pos_a in all_analytical_positions:
                                assert np.linalg.norm(pos_a[:2] - expected_xy) < 0.02
                                assert -0.26 < pos_a[2] < 0.26
                            for pos_g in all_gjk_positions:
                                assert np.linalg.norm(pos_g[:2] - expected_xy) < 0.02
                                assert -0.26 < pos_g[2] < 0.26
                    else:
                        assert pos_diff < 0.1
                else:
                    assert pos_diff < 0.05
        except Exception as e:
            print(f"\n{'=' * 80}")
            print(f"FAILED TEST SCENARIO: {description}")
            print(f"{'=' * 80}")
            print(f"Capsule 1: pos={pos1}, euler={euler1}")
            print(f"Capsule 2: pos={pos2}, euler={euler2}")
            print(f"Expected collision: {should_collide}")
            print(f"Backend: {backend}")
            print(f"Radius: {radius}, Half-length: {half_length}")
            print(f"{'=' * 80}")
            raise


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


class AnalyticalVsGJKSceneCreator:
    def __init__(self, monkeypatch, build_scene: Callable) -> None:
        self.monkeypatch = monkeypatch
        self.build_scene = build_scene

    def setup_scenes_before(self) -> tuple[gs.Scene, gs.Scene]:
        # Scene 1: Using ORIGINAL analytical sphere-capsule detection (before any monkey-patching)
        self.scene_analytical = gs.Scene(
            show_viewer=False,
            rigid_options=gs.options.RigidOptions(
                dt=0.01,
                gravity=(0, 0, 0),
            ),
        )
        self.entities = self.build_scene(self.scene_analytical)

        # NOW monkey-patch for the GJK scene
        temp_narrowphase_path = create_modified_narrowphase_file()
        spec = importlib.util.spec_from_file_location("narrowphase_modified", temp_narrowphase_path)
        narrowphase_modified = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(narrowphase_modified)
        from genesis.engine.solvers.rigid.collider import narrowphase

        self.monkeypatch.setattr(
            narrowphase, "func_convex_convex_contact", narrowphase_modified.func_convex_convex_contact
        )

        # Scene 2: Force GJK for sphere-capsule (using modified narrowphase)
        self.scene_gjk = gs.Scene(
            show_viewer=False,
            rigid_options=gs.options.RigidOptions(
                dt=0.01,
                gravity=(0, 0, 0),
                use_gjk_collision=True,
            ),
        )
        self.entities_gjk = self.build_scene(self.scene_gjk)

        return self.scene_analytical, self.scene_gjk

    def checks_after(self):
        # Check if GJK was used (bit 16)
        analytical_used_gjk = (self.scene_analytical._sim.rigid_solver._errno[0] & (1 << 16)) != 0
        gjk_used_gjk = (self.scene_gjk._sim.rigid_solver._errno[0] & (1 << 16)) != 0

        # Verify that analytical scene did NOT use GJK, and GJK scene DID use GJK
        assert not analytical_used_gjk, (
            f"Analytical scene should not use GJK (errno={self.scene_analytical._sim.rigid_solver._errno[0]})"
        )
        assert gjk_used_gjk, f"GJK scene should use GJK (errno={self.scene_gjk._sim.rigid_solver._errno[0]})"


@pytest.mark.parametrize("backend", [gs.cpu, gs.gpu])
def test_sphere_capsule_vs_gjk(backend, monkeypatch):
    """
    Compare analytical sphere-capsule collision with GJK by monkey-patching narrowphase.
    Tests multiple configurations with a single scene build (moving objects between tests).
    """
    # Define all test cases
    test_cases = [
        # (sphere_pos, capsule_pos, capsule_euler, should_collide, description, skip_gpu)
        ((0, 0, 0.4), (0, 0, 0), (0, 0, 0), True, "sphere_above_capsule_top", False),
        ((0.18, 0, 0), (0, 0, 0), (0, 0, 0), True, "sphere_close_to_capsule", False),
        ((0.17, 0.17, 0), (0, 0, 0), (0, 0, 0), False, "sphere_near_cylinder", False),
        ((0.35, 0, 0.35), (0, 0, 0), (0, 45, 0), False, "sphere_near_cap", False),
        ((0.15, 0, 0), (0, 0, 0), (0, 0, 0), True, "sphere_touching_cylinder", False),
        ((0, 0, 0), (0, 0, 0), (0, 0, 0), True, "sphere_at_capsule_center", False),
        ((0.15, 0, 0.3), (0, 0, 0), (0, 0, 0), True, "sphere_near_capsule_cap", True),
        ((0, 0.15, 0), (0, 0, 0), (0, 90, 0), True, "sphere_horizontal_capsule", False),
    ]

    sphere_radius = 0.1
    capsule_radius = 0.1
    capsule_half_length = 0.25

    # Build scenes once with initial configuration
    def build_scene(scene: gs.Scene):
        with tempfile.TemporaryDirectory() as tmpdir_mjcf:
            sphere_mjcf = create_sphere_mjcf("sphere", (0, 0, 0), sphere_radius)
            sphere_path = os.path.join(tmpdir_mjcf, "sphere.xml")
            ET.ElementTree(sphere_mjcf).write(sphere_path)
            entity_sphere = scene.add_entity(gs.morphs.MJCF(file=sphere_path))

            capsule_mjcf = create_capsule_mjcf("capsule", (0, 0, 0), (0, 0, 0), capsule_radius, capsule_half_length)
            capsule_path = os.path.join(tmpdir_mjcf, "capsule.xml")
            ET.ElementTree(capsule_mjcf).write(capsule_path)
            entity_capsule = scene.add_entity(gs.morphs.MJCF(file=capsule_path))

            scene.build()
            return [entity_sphere, entity_capsule]

    scene_creator = AnalyticalVsGJKSceneCreator(monkeypatch=monkeypatch, build_scene=build_scene)
    scene_analytical, scene_gjk = scene_creator.setup_scenes_before()
    entities_analytical = scene_creator.entities
    entities_gjk = scene_creator.entities_gjk

    # Run all test cases
    for sphere_pos, capsule_pos, capsule_euler, should_collide, description, skip_gpu in test_cases:
        # Skip on GPU if requested (for known GJK issues)
        if skip_gpu and backend == gs.gpu:
            print(f"\nTest: {description} - SKIPPED on GPU")
            continue

        print(f"\nTest: {description}")

        try:
            # Set positions and orientations
            capsule_quat = gs.utils.geom.xyz_to_quat(xyz=np.array(capsule_euler), degrees=True)
            sphere_quat = gs.utils.geom.xyz_to_quat(xyz=np.array([0, 0, 0]), degrees=True)

            entities_analytical[0].set_qpos(np.array([*sphere_pos, *sphere_quat]))
            entities_analytical[1].set_qpos(np.array([*capsule_pos, *capsule_quat]))
            entities_gjk[0].set_qpos(np.array([*sphere_pos, *sphere_quat]))
            entities_gjk[1].set_qpos(np.array([*capsule_pos, *capsule_quat]))

            # Zero out velocities to prevent motion during step
            zero_vel = np.zeros(6)  # 3 linear + 3 angular
            entities_analytical[0].set_dofs_velocity(zero_vel)
            entities_analytical[1].set_dofs_velocity(zero_vel)
            entities_gjk[0].set_dofs_velocity(zero_vel)
            entities_gjk[1].set_dofs_velocity(zero_vel)

            scene_analytical.step()
            scene_gjk.step()

            scene_creator.checks_after()

            contacts_analytical = scene_analytical.rigid_solver.collider.get_contacts(as_tensor=False, to_torch=False)
            contacts_gjk = scene_gjk.rigid_solver.collider.get_contacts(as_tensor=False, to_torch=False)

            has_collision_analytical = contacts_analytical is not None and len(contacts_analytical["geom_a"]) > 0
            has_collision_gjk = contacts_gjk is not None and len(contacts_gjk["geom_a"]) > 0

            assert has_collision_analytical == has_collision_gjk, (
                f"Collision detection mismatch! Analytical: {has_collision_analytical}, GJK: {has_collision_gjk}"
            )
            assert has_collision_analytical == should_collide

            # If both detected a collision, compare the contact details
            if has_collision_analytical and has_collision_gjk:
                pen_analytical = contacts_analytical["penetration"][0]
                pen_gjk = contacts_gjk["penetration"][0]

                normal_analytical = np.array(contacts_analytical["normal"][0])
                normal_gjk = np.array(contacts_gjk["normal"][0])

                pos_analytical = np.array(contacts_analytical["position"][0])
                pos_gjk = np.array(contacts_gjk["position"][0])

                pen_tol = max(0.01, 0.1 * max(pen_analytical, pen_gjk))
                assert abs(pen_analytical - pen_gjk) < pen_tol, (
                    f"Penetration mismatch! Analytical: {pen_analytical:.6f}, GJK: {pen_gjk:.6f}"
                )

                normal_agreement = abs(np.dot(normal_analytical, normal_gjk))
                normal_tol = 0.5 if description == "sphere_at_capsule_center" else 0.95
                assert normal_agreement > normal_tol, f"Normal mismatch! agreement: {normal_agreement:.4f}"

                pos_diff = np.linalg.norm(pos_analytical - pos_gjk)
                assert pos_diff < 0.05, f"Position mismatch! Diff: {pos_diff:.6f}"
        except Exception as e:
            print(f"\n{'=' * 80}")
            print(f"FAILED TEST SCENARIO: {description}")
            print(f"{'=' * 80}")
            print(f"Sphere: pos={sphere_pos}")
            print(f"Capsule: pos={capsule_pos}, euler={capsule_euler}")
            print(f"Expected collision: {should_collide}")
            print(f"Backend: {backend}")
            print(f"Sphere radius: {sphere_radius}")
            print(f"Capsule radius: {capsule_radius}, Half-length: {capsule_half_length}")
            print(f"{'=' * 80}")
            raise
