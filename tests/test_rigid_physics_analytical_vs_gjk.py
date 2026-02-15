"""
Unit test comparing analytical capsule-capsule contact detection with GJK.

This test creates a modified version of narrowphase.py in a temporary file that
forces capsule-capsule and sphere-capsule collisions to use GJK instead of
analytical methods, allowing direct comparison between the two approaches.

# errno

We abuse errno in this test, because it is considerably easier, and needs much less code, than
attempting to add a new tensor into one of the existing structures, and have that work for both
ndarray and field, via monkey-patching.

errno is NOT designed for how we use it. Nevertheless with a couple of reasonable-ish assumptions
we can work with it.

Assumption 1: when code runs normally and correctly, nothing in Genesis production code (not including
test code) will ever set bit 16 of errno to any value except 0.
Assumption 2: when taking a step, nothing in Genesis production code will set bit 16 of errno to any value
at all - including 0 - when running normally.

Both of these assumptions are implicitly tested by our code, in that should Genesis code violate them,
our tests will almost certainly fail.

Note that as part of our use of errno, we take full responsibilty ourselves for resetting it to 0 before each
test scenario. We do not assume - nor require - any existing Genesis code to handle this for us, for example
by setting errno to 0 in set_qpos.

Note that, for completeness, Genesis code does handle resetting errno to 0, inside set_qpos, but for simplicity,
we make resetting errno explicit in this test.
"""

import copy
import importlib.util
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import TYPE_CHECKING, Callable, cast

import numpy as np
import pytest

import genesis as gs
from .utils import assert_allclose

if TYPE_CHECKING:
    from genesis.engine.entities.rigid_entity import RigidGeom


ERRNO_CALLED_GJK = 1 << 16
POS_TOL = 1e-2  # otherwise tests fail


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
        pos=" ".join(map(str, pos)),
        euler=" ".join(map(str, euler)),
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


def insert_errno_before_call(lines, function_call_pattern, errno_value, comment):
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
    else:
        raise ValueError(f"Could not find function call: {function_call_pattern}")

    # Get indentation from the call line
    indent_size = len(lines[call_line_idx]) - len(lines[call_line_idx].lstrip())

    # Insert errno marker on the line before the call
    errno_line = f"{' ' * indent_size}errno[i_b] |= {errno_value}  # {comment}"
    lines.insert(call_line_idx, errno_line)

    return lines


def create_modified_narrowphase_file(tmp_path: Path):
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
        lines, "diff_gjk.func_gjk_contact(", ERRNO_CALLED_GJK, "MODIFIED: GJK called for collision detection"
    )
    lines = insert_errno_before_call(
        lines, "gjk.func_gjk_contact(", ERRNO_CALLED_GJK, "MODIFIED: GJK called for collision detection"
    )

    content = "\n".join(lines)

    # Debug: Check if errno was actually inserted
    errno_count = content.count(f"errno[i_b] |= {ERRNO_CALLED_GJK}")
    assert errno_count >= 1

    temp_narrowphase_path = tmp_path / "narrow.py"
    with open(temp_narrowphase_path, "w") as f:
        f.write(content)

    return temp_narrowphase_path


def scene_add_sphere(tmp_path: Path, scene: gs.Scene, radius: float) -> "RigidGeom":
    sphere_mjcf = create_sphere_mjcf("sphere", (0, 0, 0), radius)
    sphere_path = tmp_path / "sphere.xml"
    ET.ElementTree(sphere_mjcf).write(sphere_path)
    entity_sphere = cast("RigidGeom", scene.add_entity(gs.morphs.MJCF(file=sphere_path)))
    return entity_sphere


def scene_add_capsule(tmp_path: Path, scene: gs.Scene, half_length: float, radius: float) -> "RigidGeom":
    capsule_mjcf = create_capsule_mjcf("capsule", (0, 0, 0), (0, 0, 0), radius, half_length)
    capsule_path = tmp_path / "sphere.xml"
    ET.ElementTree(capsule_mjcf).write(capsule_path)
    entity_capsule = cast("RigidGeom", scene.add_entity(gs.morphs.MJCF(file=capsule_path)))
    return entity_capsule


class AnalyticalVsGJKSceneCreator:
    def __init__(self, monkeypatch, build_scene: Callable, tmp_path: Path, show_viewer: bool) -> None:
        self.monkeypatch = monkeypatch
        self.build_scene = build_scene
        self.tmp_path = tmp_path
        self.scene_analytical: gs.Scene
        self.scene_gjk: gs.Scene
        self.entities_analytical = []
        self.entities_gjk = []
        self.show_viewer = show_viewer

    def setup_scenes(self) -> tuple[gs.Scene, gs.Scene]:
        """Build both scenes WITHOUT any monkey-patching."""
        # Scene 1: Using ORIGINAL analytical collision detection
        self.scene_analytical = gs.Scene(show_viewer=self.show_viewer)
        self.build_scene(scene=self.scene_analytical, tmp_path=self.tmp_path, entities=self.entities_analytical)

        # Scene 2: Will use GJK after monkey-patching (built now with use_gjk_collision=True)
        self.scene_gjk = gs.Scene(
            show_viewer=self.show_viewer,
            rigid_options=gs.options.RigidOptions(use_gjk_collision=True),
        )
        self.build_scene(scene=self.scene_gjk, tmp_path=self.tmp_path, entities=self.entities_gjk)

        return self.scene_analytical, self.scene_gjk

    def apply_gjk_patch(self) -> None:
        """
        Monkey-patch the @ti.kernel for narrowphase with the modified version from a tmp file.

        This replaces the entire kernel object so that:
        - The new kernel has its own empty materialized_kernels cache
        - Fastcache sees a different filepath in the cache key (the tmp file),
          so it won't find a stale on-disk cache hit
        """
        temp_narrowphase_path = create_modified_narrowphase_file(tmp_path=self.tmp_path)
        spec = importlib.util.spec_from_file_location("narrowphase_modified", temp_narrowphase_path)
        narrowphase_modified = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(narrowphase_modified)
        from genesis.engine.solvers.rigid.collider import narrowphase

        self.monkeypatch.setattr(
            narrowphase,
            "func_narrow_phase_convex_vs_convex",
            narrowphase_modified.func_narrow_phase_convex_vs_convex,
        )

    def update_pos_quat_analytical(self, entity_idx: int, pos, euler) -> None:
        quat = gs.utils.geom.xyz_to_quat(xyz=np.array(euler, dtype=gs.np_float), degrees=True)
        self.entities_analytical[entity_idx].set_qpos((*pos, *quat))
        self.entities_analytical[entity_idx].zero_all_dofs_velocity()

    def update_pos_quat_gjk(self, entity_idx: int, pos, euler) -> None:
        quat = gs.utils.geom.xyz_to_quat(xyz=np.array(euler, dtype=gs.np_float), degrees=True)
        self.entities_gjk[entity_idx].set_qpos((*pos, *quat))
        self.entities_gjk[entity_idx].zero_all_dofs_velocity()

    def step_analytical(self):
        # see section '# errno' above for discussion on our abusing errno, and the assumptions which we make.
        self.scene_analytical._sim.rigid_solver._errno.fill(0)
        self.scene_analytical.step()
        errno_val = self.scene_analytical._sim.rigid_solver._errno[0]
        assert (errno_val & (ERRNO_CALLED_GJK)) == 0, f"Analytical scene should not use GJK (errno={errno_val})"

    def step_gjk(self):
        # see section '# errno' above for discussion on our abusing errno, and the assumptions which we make.
        self.scene_gjk._sim.rigid_solver._errno.fill(0)
        self.scene_gjk.step()
        errno_val = self.scene_gjk._sim.rigid_solver._errno[0]
        assert (errno_val & (ERRNO_CALLED_GJK)) != 0, f"GJK scene should use GJK (errno={errno_val})"


@pytest.mark.required
@pytest.mark.parametrize("backend", [gs.cpu, gs.gpu])
def test_capsule_capsule_vs_gjk(backend, monkeypatch, tmp_path: Path, show_viewer: bool, tol: float) -> None:
    """
    Compare analytical capsule-capsule collision with GJK by monkey-patching narrowphase.
    Tests multiple configurations with a single scene build (moving objects between tests).

    Two-phase approach to avoid kernel caching interference:
    1. Run ALL analytical scenarios first (original kernel)
    2. Apply monkey-patch (replaces the @ti.kernel with a new object from a tmp file)
    3. Run ALL GJK scenarios (patched kernel with its own empty cache)
    """
    test_cases = [
        # (pos0, euler0, pos1, euler1, should_collide, description)
        ((0, 0, 0), (0, 0, 0), (0.15, 0, 0), (0, 90, 0), True, "perpendicular_close"),
        ((0, 0, 0), (0, 0, 0), (0.18, 0, 0), (0, 0, 0), True, "parallel_light"),
        ((0, 0, 0), (0, 90, 0), (0, 0.17, 0.17), (0, 90, 0), False, "horizontal_displaced"),
        ((0, 0, 0), (0, 0, 0), (0.15, 0, 0), (0, 0, 0), True, "parallel_deep"),
        ((0, 0, 0), (0, 0, 0), (0, 0, 0), (90, 0, 0), True, "perpendicular_center"),
        ((0, 0, 0), (0, 0, 0), (0.15, 0, 0), (0, 45, 0), True, "diagonal_rotated"),
    ]

    radius = 0.1
    half_length = 0.25

    def build_scene(scene: gs.Scene, tmp_path: Path, entities: list):
        entities.append(scene_add_capsule(tmp_path, scene, half_length=half_length, radius=radius))
        entities.append(scene_add_capsule(tmp_path, scene, half_length=half_length, radius=radius))
        scene.build()

    scene_creator = AnalyticalVsGJKSceneCreator(
        monkeypatch=monkeypatch, build_scene=build_scene, tmp_path=tmp_path, show_viewer=show_viewer
    )
    scene_analytical, scene_gjk = scene_creator.setup_scenes()

    # Phase 1: Run all analytical scenarios (original, unpatched kernel)
    analytical_results = {}
    for pos0, euler0, pos1, euler1, should_collide, description in test_cases:
        try:
            scene_creator.update_pos_quat_analytical(entity_idx=0, pos=pos0, euler=euler0)
            scene_creator.update_pos_quat_analytical(entity_idx=1, pos=pos1, euler=euler1)
            scene_creator.step_analytical()

            contacts = scene_analytical.rigid_solver.collider.get_contacts(as_tensor=False, to_torch=False)
            has_collision = len(contacts["geom_a"]) > 0
            assert has_collision == should_collide, (
                f"Analytical collision mismatch! Got: {has_collision}, Expected: {should_collide}"
            )
            # Deep-copy so subsequent steps can't corrupt stored data
            analytical_results[description] = copy.deepcopy(contacts)
        except AssertionError as e:
            raise AssertionError(
                f"\nFAILED TEST SCENARIO (analytical phase): {description}\n"
                f"Capsule 0: pos={pos0}, euler={euler0}\n"
                f"Capsule 1: pos={pos1}, euler={euler1}\n"
                f"Expected collision: {should_collide}\n"
                f"Backend: {backend}\n"
                f"Radius: {radius}, Half-length: {half_length}\n"
            ) from e

    # Phase 2: Apply monkey-patch (replace @ti.kernel with version from tmp file)
    scene_creator.apply_gjk_patch()

    # Phase 3: Run all GJK scenarios (patched kernel, fresh cache)
    for pos0, euler0, pos1, euler1, should_collide, description in test_cases:
        try:
            scene_creator.update_pos_quat_gjk(entity_idx=0, pos=pos0, euler=euler0)
            scene_creator.update_pos_quat_gjk(entity_idx=1, pos=pos1, euler=euler1)
            scene_creator.step_gjk()

            contacts_gjk = scene_gjk.rigid_solver.collider.get_contacts(as_tensor=False, to_torch=False)
            contacts_analytical = analytical_results[description]

            has_collision_analytical = contacts_analytical is not None and len(contacts_analytical["geom_a"]) > 0
            has_collision_gjk = contacts_gjk is not None and len(contacts_gjk["geom_a"]) > 0

            assert has_collision_analytical == has_collision_gjk, (
                f"Collision detection mismatch! Analytical: {has_collision_analytical}, GJK: {has_collision_gjk}"
            )
            assert has_collision_gjk == should_collide

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

                    # When GJK has multicontact, verify analytical also generates sufficient contacts
                    if n_gjk >= 2:
                        assert n_analytical >= 2, (
                            f"GJK found {n_gjk} contacts, but analytical only found {n_analytical} "
                            f"(expected at least 2)"
                        )
                        assert n_analytical >= (n_gjk - 1), (
                            f"GJK found {n_gjk} contacts, but analytical only found {n_analytical} "
                            f"(expected at least {n_gjk - 1})"
                        )

                    if n_analytical >= 2 or n_gjk >= 2:
                        all_analytical_positions = np.array(
                            [contacts_analytical["position"][i] for i in range(n_analytical)]
                        )
                        all_gjk_positions = np.array([contacts_gjk["position"][i] for i in range(n_gjk)])

                        for i, pos_a in enumerate(all_analytical_positions):
                            min_dist = min(np.linalg.norm(pos_a - pos_g) for pos_g in all_gjk_positions)
                            assert min_dist < POS_TOL

                        # For parallel vertical capsules, verify contacts are on the line between axes
                        if euler0 == (0, 0, 0) and euler1 == (0, 0, 0):  # Both vertical
                            expected_xy = np.array([pos1[0] / 2, 0.0])  # Midpoint between capsules
                            for pos_a in all_analytical_positions:
                                assert np.linalg.norm(pos_a[:2] - expected_xy) < POS_TOL
                                assert -0.26 < pos_a[2] < 0.26
                            for pos_g in all_gjk_positions:
                                assert np.linalg.norm(pos_g[:2] - expected_xy) < POS_TOL
                                assert -0.26 < pos_g[2] < 0.26
                    else:
                        assert pos_diff < POS_TOL
                else:
                    assert pos_diff < POS_TOL
        except AssertionError as e:
            raise AssertionError(
                f"\nFAILED TEST SCENARIO (GJK phase): {description}\n"
                f"Capsule 0: pos={pos0}, euler={euler0}\n"
                f"Capsule 1: pos={pos1}, euler={euler1}\n"
                f"Expected collision: {should_collide}\n"
                f"Backend: {backend}\n"
                f"Radius: {radius}, Half-length: {half_length}\n"
            ) from e


@pytest.mark.required
@pytest.mark.parametrize("backend", [gs.cpu, gs.gpu])
def test_capsule_analytical_accuracy(tmp_path: Path, show_viewer: bool, tol: float):
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

    scene = gs.Scene(show_viewer=show_viewer)

    _cap1 = scene_add_capsule(tmp_path=tmp_path, scene=scene, half_length=0.25, radius=0.1)
    cap2 = scene_add_capsule(tmp_path=tmp_path, scene=scene, half_length=0.25, radius=0.1)

    scene.build()
    cap2.set_qpos(np.array([*(0.15, 0, 0), *(1, 0, 0, 0)], dtype=gs.np_float))
    scene.step()

    contacts = scene.rigid_solver.collider.get_contacts(as_tensor=False, to_torch=False)
    assert len(contacts["geom_a"]) > 0

    penetration = contacts["penetration"][0]
    expected_pen = 0.05

    assert abs(penetration - expected_pen) < POS_TOL, (
        f"Analytical solution not exact! Expected: {expected_pen}, Got: {penetration:.6f}"
    )

    assert_allclose(contacts["normal"][0], (-1.0, 0.0, 0.0), tol=tol)


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


@pytest.mark.required
@pytest.mark.parametrize("backend", [gs.cpu, gs.gpu])
def test_sphere_capsule_vs_gjk(backend, monkeypatch, tmp_path: Path, show_viewer: bool, tol: float) -> None:
    """
    Compare analytical sphere-capsule collision with GJK by monkey-patching narrowphase.
    Tests multiple configurations with a single scene build (moving objects between tests).

    Two-phase approach to avoid kernel caching interference:
    1. Run ALL analytical scenarios first (original kernel)
    2. Apply monkey-patch (replaces the @ti.kernel with a new object from a tmp file)
    3. Run ALL GJK scenarios (patched kernel with its own empty cache)
    """
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

    def build_scene(scene: gs.Scene, tmp_path: Path, entities: list) -> None:
        entities.append(scene_add_sphere(tmp_path, scene, radius=sphere_radius))
        entities.append(scene_add_capsule(tmp_path, scene, half_length=capsule_half_length, radius=capsule_radius))
        scene.build()

    scene_creator = AnalyticalVsGJKSceneCreator(
        monkeypatch=monkeypatch,
        build_scene=build_scene,
        tmp_path=tmp_path,
        show_viewer=show_viewer,
    )
    scene_analytical, scene_gjk = scene_creator.setup_scenes()

    # Phase 1: Run all analytical scenarios (original, unpatched kernel)
    analytical_results = {}
    for sphere_pos, capsule_pos, capsule_euler, should_collide, description, skip_gpu in test_cases:
        if skip_gpu and backend == gs.gpu:
            pytest.xfail(
                reason="gjk broken on gpu for this condition currently. "
                "(fails to provide contact, when we can see on paper that there should be one)."
            )

        try:
            scene_creator.update_pos_quat_analytical(entity_idx=0, pos=sphere_pos, euler=[0, 0, 0])
            scene_creator.update_pos_quat_analytical(entity_idx=1, pos=capsule_pos, euler=capsule_euler)
            scene_creator.step_analytical()

            contacts = scene_analytical.rigid_solver.collider.get_contacts(as_tensor=False, to_torch=False)
            has_collision = len(contacts["geom_a"]) > 0
            assert has_collision == should_collide, (
                f"Analytical collision mismatch! Got: {has_collision}, Expected: {should_collide}"
            )
            # Deep-copy so subsequent steps can't corrupt stored data
            analytical_results[description] = copy.deepcopy(contacts)
        except AssertionError as e:
            raise AssertionError(
                f"\nFAILED TEST SCENARIO (analytical phase): {description}\n"
                f"Sphere: pos={sphere_pos}\n"
                f"Capsule: pos={capsule_pos}, euler={capsule_euler}\n"
                f"Expected collision: {should_collide}\n"
                f"Backend: {backend}\n"
                f"Sphere radius: {sphere_radius}\n"
                f"Capsule radius: {capsule_radius}, Half-length: {capsule_half_length}\n"
            ) from e

    # Phase 2: Apply monkey-patch (replace @ti.kernel with version from tmp file)
    scene_creator.apply_gjk_patch()

    # Phase 3: Run all GJK scenarios (patched kernel, fresh cache)
    for sphere_pos, capsule_pos, capsule_euler, should_collide, description, skip_gpu in test_cases:
        if skip_gpu and backend == gs.gpu:
            continue

        try:
            scene_creator.update_pos_quat_gjk(entity_idx=0, pos=sphere_pos, euler=[0, 0, 0])
            scene_creator.update_pos_quat_gjk(entity_idx=1, pos=capsule_pos, euler=capsule_euler)
            scene_creator.step_gjk()

            contacts_gjk = scene_gjk.rigid_solver.collider.get_contacts(as_tensor=False, to_torch=False)
            contacts_analytical = analytical_results[description]

            has_collision_analytical = len(contacts_analytical["geom_a"]) > 0
            has_collision_gjk = len(contacts_gjk["geom_a"]) > 0

            assert has_collision_analytical == has_collision_gjk, (
                f"Collision detection mismatch! Analytical: {has_collision_analytical}, GJK: {has_collision_gjk}"
            )
            assert has_collision_gjk == should_collide

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

                assert_allclose(pos_analytical, pos_gjk, tol=POS_TOL)
        except AssertionError as e:
            raise AssertionError(
                f"\nFAILED TEST SCENARIO (GJK phase): {description}\n"
                f"Sphere: pos={sphere_pos}\n"
                f"Capsule: pos={capsule_pos}, euler={capsule_euler}\n"
                f"Expected collision: {should_collide}\n"
                f"Backend: {backend}\n"
                f"Sphere radius: {sphere_radius}\n"
                f"Capsule radius: {capsule_radius}, Half-length: {capsule_half_length}\n"
            ) from e
