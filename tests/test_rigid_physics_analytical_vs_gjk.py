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
from .conftest import TOL_SINGLE

if TYPE_CHECKING:
    from genesis.engine.entities.rigid_entity import RigidGeom


ERRNO_CALLED_GJK = 1 << 16
ERRNO_CALLED_CAPSULE_CAPSULE = 1 << 17
ERRNO_CALLED_SPHERE_CAPSULE = 1 << 18
ERRNO_CALLED_SPHERE_SPHERE = 1 << 19
ERRNO_CALLED_CYLINDER_SPHERE = 1 << 20
ERRNO_CALLED_CYLINDER_CYLINDER = 1 << 21

# All analytical specializations live in func_convex_convex_contact
# (inlined into func_narrow_phase_convex_vs_convex).
ANALYTICAL_ERRNO_BITS = {
    "capsule_contact.func_capsule_capsule_contact": ERRNO_CALLED_CAPSULE_CAPSULE,
    "capsule_contact.func_sphere_capsule_contact": ERRNO_CALLED_SPHERE_CAPSULE,
    "cylinder_contact.func_sphere_sphere_contact": ERRNO_CALLED_SPHERE_SPHERE,
    "cylinder_contact.func_cylinder_sphere_contact": ERRNO_CALLED_CYLINDER_SPHERE,
    "cylinder_contact.func_cylinder_cylinder_contact": ERRNO_CALLED_CYLINDER_CYLINDER,
}
POS_TOL = 3e-2  # analytical vs GJK perturbation multi-contact needs slack

# Tolerances for checking results against hand-computed expected values.
# Analytical solutions should be near-exact; GJK needs more slack; reason unclear.
#
# Penetration tolerance: absolute error in metres.
# Normal tolerance: maximum allowed value of (1 - |dot(actual, expected)|).
#   e.g. 1e-5 means the normal must agree to within ~0.26 degrees,
#        1e-2 means within ~8 degrees.
ANALYTICAL_PEN_TOL = TOL_SINGLE
ANALYTICAL_NORMAL_TOL = TOL_SINGLE
GJK_PEN_TOL = 1e-2
GJK_NORMAL_TOL = 1e-2


def _find_line_endpoints(positions):
    """Find the two contact points with maximum pairwise distance.

    Returns (i_a, i_b) indices into positions, or (0, 0) if only one point.
    """
    n = len(positions)
    if n <= 1:
        return 0, 0
    best_dist = -1.0
    best_i, best_j = 0, 0
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(positions[i] - positions[j])
            if d > best_dist:
                best_dist = d
                best_i, best_j = i, j
    return best_i, best_j


def _point_line_distance(point, line_a, line_b):
    """Perpendicular distance from point to the line through line_a and line_b."""
    ab = line_b - line_a
    ab_len = np.linalg.norm(ab)
    if ab_len < 1e-12:
        return np.linalg.norm(point - line_a)
    t = np.dot(point - line_a, ab) / (ab_len * ab_len)
    projection = line_a + t * ab
    return np.linalg.norm(point - projection)


def _check_expected_values(contacts, description, exp_pen, exp_normal, method_name, pen_tol, normal_tol):
    """Check that contacts match the expected penetration and/or normal, when provided.

    Parameters
    ----------
    pen_tol : float
        Maximum absolute penetration error (metres).
    normal_tol : float
        Maximum allowed ``1 - |dot(actual, expected)|``.
    """
    if not contacts or len(contacts["geom_a"]) == 0:
        return

    if exp_pen is not None:
        pen = contacts["penetration"][0]
        assert abs(pen - exp_pen) < pen_tol, (
            f"[{method_name}] {description}: penetration {pen:.6f} != expected {exp_pen:.6f} (tol={pen_tol})"
        )

    if exp_normal is not None:
        normal = np.array(contacts["normal"][0])
        exp_n = np.array(exp_normal, dtype=float)
        exp_n_len = np.linalg.norm(exp_n)
        assert gs.EPS is not None
        if exp_n_len > gs.EPS:
            dot_err = 1.0 - abs(np.dot(normal, exp_n / exp_n_len))
            assert dot_err < normal_tol, (
                f"[{method_name}] {description}: normal {normal} vs expected {exp_n / exp_n_len}, "
                f"1-|dot|={dot_err:.6e} >= {normal_tol}"
            )


def create_capsule_mjcf(name, pos, euler, radius, half_length, fixed=False):
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
    if not fixed:
        ET.SubElement(body, "joint", name=f"{name}_joint", type="free")
    return mjcf


def create_cylinder_mjcf(name, pos, euler, radius, half_height, fixed=False):
    """Helper function to create an MJCF file with a single cylinder.

    MuJoCo cylinder size is (radius, half_height).
    """
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
    ET.SubElement(body, "geom", type="cylinder", size=f"{radius} {half_height}")
    if not fixed:
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
    """Insert errno marker before every occurrence of a function call.

    Handles the case where the call is on a continuation line (e.g. after
    ``result = (\\n    func(...)``).  Walks backwards to find the statement
    start and inserts the errno line before it.
    """
    call_line_indices = []
    for i, line in enumerate(lines):
        if function_call_pattern in line:
            idx = line.find(function_call_pattern)
            if idx != -1:
                if idx == 0 or not (line[idx - 1].isalnum() or line[idx - 1] == "_"):
                    call_line_indices.append(i)

    if not call_line_indices:
        raise ValueError(f"Could not find function call: {function_call_pattern}")

    offset = 0
    for call_line_idx in call_line_indices:
        call_line_idx += offset

        insert_idx = call_line_idx
        paren_depth = 0
        for j in range(call_line_idx - 1, -1, -1):
            for ch in lines[j]:
                if ch == "(":
                    paren_depth += 1
                elif ch == ")":
                    paren_depth -= 1
            if paren_depth > 0:
                insert_idx = j
            break

        indent_size = len(lines[insert_idx]) - len(lines[insert_idx].lstrip())
        errno_line = f"{' ' * indent_size}errno[i_b] |= {errno_value}  # {comment}"
        lines.insert(insert_idx, errno_line)
        offset += 1

    return lines


def _read_narrowphase_source():
    """Read and fix imports in the original narrowphase source."""
    import genesis.engine.solvers.rigid.collider.narrowphase as narrowphase_module

    with open(narrowphase_module.__file__, "r") as f:
        content = f.read()

    content = content.replace("from . import ", "from genesis.engine.solvers.rigid.collider import ")
    content = content.replace("from .", "from genesis.engine.solvers.rigid.collider.")
    return content


def create_modified_narrowphase_file(tmp_path: Path):
    """
    Create a modified version of narrowphase.py that forces all primitive collisions to use GJK
    by disabling every analytical specialization branch.  Inserts errno markers before GJK calls.

    All analytical specializations are in the main kernel's func_convex_convex_contact and are
    gated by ``use_analytical``.  Setting it to False forces everything through GJK/MPR.
    """
    lines = _read_narrowphase_source().split("\n")

    # Disable all analytical specializations by forcing use_analytical = False.
    USE_ANALYTICAL = "use_analytical = is_cylinder_or_sphere_pair or is_capsule_pair or is_sphere_capsule_pair"
    found_analytical = False
    for i, line in enumerate(lines):
        if USE_ANALYTICAL in line:
            lines[i] = line.replace(USE_ANALYTICAL, "use_analytical = False")
            found_analytical = True
            break
    assert found_analytical, f"Could not find: {USE_ANALYTICAL}"

    lines = insert_errno_before_call(
        lines, "diff_gjk.func_gjk_contact(", ERRNO_CALLED_GJK, "MODIFIED: GJK called for collision detection"
    )
    lines = insert_errno_before_call(
        lines, "gjk.func_gjk_contact(", ERRNO_CALLED_GJK, "MODIFIED: GJK called for collision detection"
    )

    content = "\n".join(lines)
    assert content.count(f"errno[i_b] |= {ERRNO_CALLED_GJK}") >= 1

    temp_narrowphase_path = tmp_path / "narrow.py"
    with open(temp_narrowphase_path, "w") as f:
        f.write(content)
    return temp_narrowphase_path


def create_instrumented_narrowphase_file(tmp_path: Path):
    """
    Create a version of narrowphase.py with the analytical paths intact but instrumented
    with errno bits so we can verify which specializations were actually called.

    Also instruments the GJK path so we can verify the instrumented kernel is active.
    """
    lines = _read_narrowphase_source().split("\n")

    for func_call_pattern, errno_bit in ANALYTICAL_ERRNO_BITS.items():
        lines = insert_errno_before_call(
            lines,
            func_call_pattern + "(",
            errno_bit,
            f"INSTRUMENTED: {func_call_pattern} called",
        )

    lines = insert_errno_before_call(
        lines, "gjk.func_gjk_contact(", ERRNO_CALLED_GJK, "INSTRUMENTED: GJK called"
    )

    content = "\n".join(lines)
    for func_name, errno_bit in ANALYTICAL_ERRNO_BITS.items():
        assert content.count(f"errno[i_b] |= {errno_bit}") >= 1, f"Failed to insert errno for {func_name}"

    temp_path = tmp_path / "narrow_instrumented.py"
    with open(temp_path, "w") as f:
        f.write(content)
    return temp_path


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
        Monkey-patch the @qd.kernel for narrowphase with the modified version from a tmp file.

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
        assert (errno_val & ERRNO_CALLED_GJK) == 0, "Analytical scene should not use GJK."

    def step_gjk(self):
        # see section '# errno' above for discussion on our abusing errno, and the assumptions which we make.
        self.scene_gjk._sim.rigid_solver._errno.fill(0)
        self.scene_gjk.step()
        errno_val = self.scene_gjk._sim.rigid_solver._errno[0]
        assert (errno_val & ERRNO_CALLED_GJK) != 0, "GJK scene should use GJK."


@pytest.mark.required
@pytest.mark.parametrize("backend", [gs.cpu, gs.gpu])
def test_capsule_capsule_vs_gjk(backend, monkeypatch, tmp_path: Path, show_viewer: bool, tol: float) -> None:
    """
    Compare analytical capsule-capsule collision with GJK by monkey-patching narrowphase.
    Tests multiple configurations with a single scene build (moving objects between tests).

    Two-phase approach to avoid kernel caching interference:
    1. Run ALL analytical scenarios first (original kernel)
    2. Apply monkey-patch (replaces the @qd.kernel with a new object from a tmp file)
    3. Run ALL GJK scenarios (patched kernel with its own empty cache)
    """
    test_cases = [
        # (pos0, euler0, pos1, euler1, should_collide, description, exp_pen, exp_normal)
        # Segments cross at origin (distance=0), pen = sum of radii, normal is degenerate
        ((0, 0, 0), (0, 0, 0), (0.15, 0, 0), (0, 90, 0), True, "perpendicular_close", 0.2, None),
        # Parallel vertical, seg distance = 0.18, pen = 0.2 - 0.18 = 0.02
        ((0, 0, 0), (0, 0, 0), (0.18, 0, 0), (0, 0, 0), True, "parallel_light", 0.02, (-1, 0, 0)),
        ((0, 0, 0), (0, 90, 0), (0, 0.17, 0.17), (0, 90, 0), False, "horizontal_displaced", None, None),
        # Parallel vertical, seg distance = 0.15, pen = 0.2 - 0.15 = 0.05
        ((0, 0, 0), (0, 0, 0), (0.15, 0, 0), (0, 0, 0), True, "parallel_deep", 0.05, (-1, 0, 0)),
        # Segments cross at origin (distance=0), pen = sum of radii, normal is degenerate
        ((0, 0, 0), (0, 0, 0), (0, 0, 0), (90, 0, 0), True, "perpendicular_center", 0.2, None),
        # 45° capsule segment crosses the vertical segment at (0, 0, -0.15), so dist=0, pen = sum of radii
        ((0, 0, 0), (0, 0, 0), (0.15, 0, 0), (0, 45, 0), True, "diagonal_rotated", 0.2, None),
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
    for pos0, euler0, pos1, euler1, should_collide, description, exp_pen, exp_normal in test_cases:
        try:
            scene_creator.update_pos_quat_analytical(entity_idx=0, pos=pos0, euler=euler0)
            scene_creator.update_pos_quat_analytical(entity_idx=1, pos=pos1, euler=euler1)
            scene_creator.step_analytical()

            contacts = scene_analytical.rigid_solver.collider.get_contacts(as_tensor=False, to_torch=False)
            has_collision = len(contacts["geom_a"]) > 0
            assert has_collision == should_collide, "Analytical collision mismatch!"
            _check_expected_values(
                contacts, description, exp_pen, exp_normal, "analytical", ANALYTICAL_PEN_TOL, ANALYTICAL_NORMAL_TOL
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

    # Phase 2: Apply monkey-patch (replace @qd.kernel with version from tmp file)
    scene_creator.apply_gjk_patch()

    # Phase 3: Run all GJK scenarios (patched kernel, fresh cache)
    for pos0, euler0, pos1, euler1, should_collide, description, exp_pen, exp_normal in test_cases:
        try:
            scene_creator.update_pos_quat_gjk(entity_idx=0, pos=pos0, euler=euler0)
            scene_creator.update_pos_quat_gjk(entity_idx=1, pos=pos1, euler=euler1)
            scene_creator.step_gjk()

            contacts_gjk = scene_gjk.rigid_solver.collider.get_contacts(as_tensor=False, to_torch=False)
            contacts_analytical = analytical_results[description]

            has_collision_analytical = contacts_analytical is not None and len(contacts_analytical["geom_a"]) > 0
            has_collision_gjk = contacts_gjk is not None and len(contacts_gjk["geom_a"]) > 0

            assert has_collision_analytical == has_collision_gjk, "Collision detection mismatch!"
            assert has_collision_gjk == should_collide

            _check_expected_values(contacts_gjk, description, exp_pen, exp_normal, "GJK", GJK_PEN_TOL, GJK_NORMAL_TOL)

            # If both detected a collision, compare the contact details
            if has_collision_analytical and has_collision_gjk:
                pen_analytical = contacts_analytical["penetration"][0]
                pen_gjk = contacts_gjk["penetration"][0]

                normal_analytical = np.array(contacts_analytical["normal"][0])
                normal_gjk = np.array(contacts_gjk["normal"][0])

                pos_analytical = np.array(contacts_analytical["position"][0])
                pos_gjk = np.array(contacts_gjk["position"][0])
                assert_allclose(pen_analytical, pen_gjk, atol=POS_TOL, rtol=0.1, err_msg="Penetration mismatch!")

                normal_agreement = abs(np.dot(normal_analytical, normal_gjk))
                assert normal_agreement > 0.95, "Normal mismatch!"

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

                        for pos_a in all_analytical_positions:
                            min_dist = min(np.linalg.norm(pos_a - pos_g) for pos_g in all_gjk_positions)
                            assert min_dist < POS_TOL

                        # For parallel vertical capsules, verify contacts are on the line between axes
                        if euler0 == (0, 0, 0) and euler1 == (0, 0, 0):  # Both vertical
                            expected_xy = np.array([pos1[0] / 2, 0.0])  # Midpoint between capsules
                            for pos_a in all_analytical_positions:
                                assert_allclose(pos_a[:2], expected_xy, tol=POS_TOL)
                                assert_allclose(pos_a[2], 0.0, tol=0.26)
                            for pos_g in all_gjk_positions:
                                assert_allclose(pos_g[:2], expected_xy, tol=POS_TOL)
                                assert -0.26 < pos_g[2] < 0.26
                    else:
                        assert_allclose(pos_analytical, pos_gjk, tol=POS_TOL)
                else:
                    assert_allclose(pos_analytical, pos_gjk, tol=POS_TOL)
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
    assert_allclose(penetration, expected_pen, tol=POS_TOL, err_msg="Analytical solution not exact!")

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
    2. Apply monkey-patch (replaces the @qd.kernel with a new object from a tmp file)
    3. Run ALL GJK scenarios (patched kernel with its own empty cache)

    Note that these can be visualized, for verification purposes, using the script at:
    https://github.com/Genesis-Embodied-AI/perso_hugh/blob/main/genesis/visualize_sphere_capsule.py
    (note: only accessible internally)
    """
    test_cases = [
        # (sphere_pos, capsule_pos, capsule_euler, should_collide, description, exp_pen, exp_normal)
        # Sphere above top cap: dist to segment endpoint (0,0,0.25) = 0.15, pen = 0.05
        ((0, 0, 0.4), (0, 0, 0), (0, 0, 0), True, "sphere_above_capsule_top", 0.05, (0, 0, 1)),
        # Sphere beside cylinder: dist to axis = 0.18, pen = 0.02
        ((0.18, 0, 0), (0, 0, 0), (0, 0, 0), True, "sphere_close_to_capsule", 0.02, (1, 0, 0)),
        # dist to axis = sqrt(0.17^2+0.17^2) ≈ 0.24 > 0.2, no collision
        ((0.17, 0.17, 0), (0, 0, 0), (0, 0, 0), False, "sphere_near_cylinder", None, None),
        ((0.35, 0, 0.35), (0, 0, 0), (0, 45, 0), False, "sphere_near_cap", None, None),
        # Sphere beside cylinder: dist to axis = 0.15, pen = 0.05
        ((0.15, 0, 0), (0, 0, 0), (0, 0, 0), True, "sphere_touching_cylinder", 0.05, (1, 0, 0)),
        # Sphere at capsule centre: dist = 0, pen = sum of radii = 0.2, normal is degenerate
        ((0, 0, 0), (0, 0, 0), (0, 0, 0), True, "sphere_at_capsule_center", 0.2, None),
        # Sphere near top cap: nearest segment pt = (0,0,0.25), dist = sqrt(0.15²+0.05²) ≈ 0.1581
        # pen = 0.2 - sqrt(0.025) ≈ 0.041886, normal along (3, 0, 1)
        ((0.15, 0, 0.3), (0, 0, 0), (0, 0, 0), True, "sphere_near_capsule_cap", 0.041886, (3, 0, 1)),
        # Horizontal capsule (axis along X after 90° Y rotation), sphere offset in Y: pen = 0.05
        ((0, 0.15, 0), (0, 0, 0), (0, 90, 0), True, "sphere_horizontal_capsule", 0.05, (0, 1, 0)),
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
    for sphere_pos, capsule_pos, capsule_euler, should_collide, description, exp_pen, exp_normal in test_cases:
        try:
            scene_creator.update_pos_quat_analytical(entity_idx=0, pos=sphere_pos, euler=[0, 0, 0])
            scene_creator.update_pos_quat_analytical(entity_idx=1, pos=capsule_pos, euler=capsule_euler)
            scene_creator.step_analytical()

            contacts = scene_analytical.rigid_solver.collider.get_contacts(as_tensor=False, to_torch=False)
            has_collision = len(contacts["geom_a"]) > 0
            assert has_collision == should_collide, "Analytical collision mismatch"
            _check_expected_values(
                contacts, description, exp_pen, exp_normal, "analytical", ANALYTICAL_PEN_TOL, ANALYTICAL_NORMAL_TOL
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

    # Phase 2: Apply monkey-patch (replace @qd.kernel with version from tmp file)
    scene_creator.apply_gjk_patch()

    # Phase 3: Run all GJK scenarios (patched kernel, fresh cache)
    for sphere_pos, capsule_pos, capsule_euler, should_collide, description, exp_pen, exp_normal in test_cases:
        try:
            scene_creator.update_pos_quat_gjk(entity_idx=0, pos=sphere_pos, euler=[0, 0, 0])
            scene_creator.update_pos_quat_gjk(entity_idx=1, pos=capsule_pos, euler=capsule_euler)
            scene_creator.step_gjk()

            contacts_gjk = scene_gjk.rigid_solver.collider.get_contacts(as_tensor=False, to_torch=False)
            contacts_analytical = analytical_results[description]

            has_collision_analytical = len(contacts_analytical["geom_a"]) > 0
            has_collision_gjk = len(contacts_gjk["geom_a"]) > 0

            assert has_collision_analytical == has_collision_gjk, "Collision detection mismatch!"
            assert has_collision_gjk == should_collide

            _check_expected_values(contacts_gjk, description, exp_pen, exp_normal, "GJK", GJK_PEN_TOL, GJK_NORMAL_TOL)

            # If both detected a collision, compare the contact details
            if has_collision_analytical and has_collision_gjk:
                pen_analytical = contacts_analytical["penetration"][0]
                pen_gjk = contacts_gjk["penetration"][0]

                normal_analytical = np.array(contacts_analytical["normal"][0])
                normal_gjk = np.array(contacts_gjk["normal"][0])

                pos_analytical = np.array(contacts_analytical["position"][0])
                pos_gjk = np.array(contacts_gjk["position"][0])
                assert_allclose(pen_analytical, pen_gjk, atol=POS_TOL, rtol=0.1, err_msg="Penetration mismatch!")

                normal_agreement = abs(np.dot(normal_analytical, normal_gjk))
                normal_tol = 0.5 if description == "sphere_at_capsule_center" else 0.95
                assert normal_agreement > normal_tol, "Normal mismatch!"

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


def _snapshot_contacts(scene):
    """Return a dict with per-contact arrays for an n_envs=0 scene."""
    contacts = scene.rigid_solver.collider.get_contacts(as_tensor=False, to_torch=False)
    if contacts is None or len(contacts["geom_a"]) == 0:
        return {
            "n": 0,
            "geom_a": np.array([], dtype=int),
            "geom_b": np.array([], dtype=int),
            "position": np.zeros((0, 3)),
            "normal": np.zeros((0, 3)),
            "penetration": np.array([]),
        }
    n = len(contacts["geom_a"])
    return {
        "n": n,
        "geom_a": np.array(contacts["geom_a"]),
        "geom_b": np.array(contacts["geom_b"]),
        "position": np.array([contacts["position"][i] for i in range(n)]),
        "normal": np.array([contacts["normal"][i] for i in range(n)]),
        "penetration": np.array([contacts["penetration"][i] for i in range(n)]),
    }


def _build_cylinder_arena(scene, entities, tmp_path: Path):
    """Build an arena of fixed cylinder walls with two free cylinders inside.

    Uses MJCF to ensure geoms get GEOM_TYPE.CYLINDER (gs.morphs.Cylinder
    produces GEOM_TYPE.MESH which bypasses analytical specializations).
    """
    ARENA_HALF = 0.5
    WALL_R = 0.08
    WALL_HALF_H = 1.0
    SPACING = 0.20
    cyl_idx = 0

    positions = [i * SPACING for i in range(-int(ARENA_HALF / SPACING), int(ARENA_HALF / SPACING) + 1)]

    def _add_cyl(pos, euler, radius, half_height, fixed):
        nonlocal cyl_idx
        name = f"cyl_{cyl_idx}"
        cyl_idx += 1
        mjcf = create_cylinder_mjcf(name, pos, euler, radius, half_height, fixed=fixed)
        path = tmp_path / f"{name}.xml"
        ET.ElementTree(mjcf).write(path)
        return scene.add_entity(gs.morphs.MJCF(file=str(path)))

    # Floor / ceiling: horizontal cylinders along X, spaced along Y
    for y in positions:
        _add_cyl((0, y, -ARENA_HALF), (0, 90, 0), WALL_R, WALL_HALF_H, fixed=True)
        _add_cyl((0, y, +ARENA_HALF), (0, 90, 0), WALL_R, WALL_HALF_H, fixed=True)

    # Left / right walls: vertical cylinders along Z, spaced along Y
    for y in positions:
        _add_cyl((-ARENA_HALF, y, 0), (0, 0, 0), WALL_R, WALL_HALF_H, fixed=True)
        _add_cyl((+ARENA_HALF, y, 0), (0, 0, 0), WALL_R, WALL_HALF_H, fixed=True)

    # Front / back walls: vertical cylinders along Z, spaced along X
    for x in positions:
        _add_cyl((x, -ARENA_HALF, 0), (0, 0, 0), WALL_R, WALL_HALF_H, fixed=True)
        _add_cyl((x, +ARENA_HALF, 0), (0, 0, 0), WALL_R, WALL_HALF_H, fixed=True)

    # Two free cylinders
    entities.append(_add_cyl((0.1, 0.05, 0.0), (0, 0, 0), 0.10, 0.20, fixed=False))
    entities.append(_add_cyl((-0.1, -0.05, 0.1), (30, 45, 0), 0.12, 0.175, fixed=False))

    scene.build()


@pytest.mark.required
@pytest.mark.parametrize("backend", [gs.gpu])
def test_cylinder_arena_analytical_vs_gjk(backend, monkeypatch, tmp_path: Path, show_viewer: bool, tol: float):
    """
    Fuzz test: cylinder-only arena guaranteeing cylinder-cylinder collisions.

    All walls and free objects are cylinders, so every collision exercises the
    cylinder_cylinder analytical specialization.  Two phases:
      1. GJK (reference): run forward with GJK-forced kernel, record contacts.
      2. Analytical (instrumented): replay same qpos+forces with analytical
         kernel, verify ERRNO bits, compare contacts against GJK reference.
    """
    import torch

    N_STEPS = 200
    BOUNCE_FORCE = 600.0

    rigid_opts = gs.options.RigidOptions(
        dt=0.005,
        gravity=(0, 0, 0),
        enable_collision=True,
        use_gjk_collision=True,
        enable_multi_contact=True,
    )

    from genesis.engine.solvers.rigid.collider import narrowphase

    # --- Build both scenes BEFORE any patching ---
    scene_gjk = gs.Scene(show_viewer=show_viewer, rigid_options=rigid_opts)
    entities_gjk = []
    gjk_dir = tmp_path / "gjk_mjcf"
    gjk_dir.mkdir()
    _build_cylinder_arena(scene_gjk, entities_gjk, gjk_dir)

    scene_ana = gs.Scene(show_viewer=False, rigid_options=rigid_opts)
    entities_ana = []
    ana_dir = tmp_path / "ana_mjcf"
    ana_dir.mkdir()
    _build_cylinder_arena(scene_ana, entities_ana, ana_dir)

    for body in entities_gjk:
        body.set_dofs_damping(np.array([15.0, 15.0, 15.0], dtype=gs.np_float), dofs_idx_local=[3, 4, 5])
    for body in entities_ana:
        body.set_dofs_damping(np.array([15.0, 15.0, 15.0], dtype=gs.np_float), dofs_idx_local=[3, 4, 5])

    # --- Phase 1: GJK reference run ---
    gjk_path = create_modified_narrowphase_file(tmp_path=tmp_path)
    spec_gjk = importlib.util.spec_from_file_location("narrowphase_cyl_gjk", gjk_path)
    narrowphase_gjk = importlib.util.module_from_spec(spec_gjk)
    spec_gjk.loader.exec_module(narrowphase_gjk)

    monkeypatch.setattr(
        narrowphase,
        "func_narrow_phase_convex_vs_convex",
        narrowphase_gjk.func_narrow_phase_convex_vs_convex,
    )

    torch.manual_seed(0)
    for body in entities_gjk:
        v = (torch.rand(body.n_dofs, dtype=gs.tc_float, device=gs.device) * 2 - 1) * 3.0
        body.set_dofs_velocity(v)

    recorded_qpos = []
    recorded_contacts = []
    recorded_forces = []
    gjk_confirmed = False

    torch.manual_seed(42)
    for step in range(N_STEPS):
        step_qpos = []
        for body in entities_gjk:
            q = body.get_qpos()
            step_qpos.append(q.cpu().numpy().copy() if hasattr(q, "cpu") else np.array(q).copy())
        recorded_qpos.append(step_qpos)

        scene_gjk._sim.rigid_solver._errno.fill(0)
        scene_gjk.step()
        recorded_contacts.append(_snapshot_contacts(scene_gjk))

        errno_val = int(scene_gjk._sim.rigid_solver._errno[0])
        if recorded_contacts[-1]["n"] > 0:
            assert (errno_val & ERRNO_CALLED_GJK) != 0, (
                f"Step {step}: GJK not called despite {recorded_contacts[-1]['n']} contacts."
            )
            gjk_confirmed = True

        nc = recorded_contacts[-1]["n"]
        step_forces = []
        if nc > 0:
            geom_a = recorded_contacts[-1]["geom_a"]
            geom_b = recorded_contacts[-1]["geom_b"]
            hit_geoms = set(geom_a.tolist()) | set(geom_b.tolist())
        else:
            hit_geoms = set()

        for body in entities_gjk:
            body_geoms = set(range(body.geom_start, body.geom_end))
            if body_geoms & hit_geoms:
                f = torch.zeros(body.n_dofs, dtype=gs.tc_float, device=gs.device)
                f[0:3] = (torch.rand(3, dtype=gs.tc_float, device=gs.device) * 2 - 1) * BOUNCE_FORCE
            else:
                f = torch.zeros(body.n_dofs, dtype=gs.tc_float, device=gs.device)
            step_forces.append(f.clone())
            body.control_dofs_force(f)
        recorded_forces.append(step_forces)

    assert gjk_confirmed, "Phase 1: GJK was never confirmed via errno."

    total_gjk = sum(c["n"] for c in recorded_contacts)
    print(f"Phase 1 (GJK): {total_gjk} total contacts over {N_STEPS} steps.")

    # --- Phase 2: Analytical replay ---
    monkeypatch.undo()

    instrumented_path = create_instrumented_narrowphase_file(tmp_path=tmp_path)
    spec_inst = importlib.util.spec_from_file_location("narrowphase_cyl_inst", instrumented_path)
    narrowphase_instrumented = importlib.util.module_from_spec(spec_inst)
    spec_inst.loader.exec_module(narrowphase_instrumented)

    monkeypatch.setattr(
        narrowphase,
        "func_narrow_phase_convex_vs_convex",
        narrowphase_instrumented.func_narrow_phase_convex_vs_convex,
    )

    mismatches = []
    total_contacts_gjk = 0
    total_contacts_ana = 0
    steps_with_contacts = 0
    max_pos_err_seen = 0.0
    max_pen_err_seen = 0.0
    max_collinear_err_seen = 0.0
    min_dot_seen = 1.0
    count_diff_steps = 0
    analytical_errno_seen = 0

    for step in range(N_STEPS):
        for i, body in enumerate(entities_ana):
            body.set_qpos(recorded_qpos[step][i])
            body.zero_all_dofs_velocity()

        for i, body in enumerate(entities_ana):
            body.control_dofs_force(recorded_forces[step][i])

        scene_ana._sim.rigid_solver._errno.fill(0)
        scene_ana.step()
        errno_val = int(scene_ana._sim.rigid_solver._errno[0])
        analytical_errno_seen |= errno_val

        contacts_ana = _snapshot_contacts(scene_ana)
        contacts_gjk = recorded_contacts[step]

        total_contacts_gjk += contacts_gjk["n"]
        total_contacts_ana += contacts_ana["n"]

        if contacts_gjk["n"] == 0 and contacts_ana["n"] == 0:
            continue
        steps_with_contacts += 1

        if contacts_gjk["n"] != contacts_ana["n"]:
            count_diff_steps += 1

        gjk_pairs = (
            set(zip(contacts_gjk["geom_a"].tolist(), contacts_gjk["geom_b"].tolist()))
            if contacts_gjk["n"] > 0
            else set()
        )
        ana_pairs = (
            set(zip(contacts_ana["geom_a"].tolist(), contacts_ana["geom_b"].tolist()))
            if contacts_ana["n"] > 0
            else set()
        )
        all_pairs = gjk_pairs | ana_pairs
        print(f"  step {step}: gjk={contacts_gjk['n']} ana={contacts_ana['n']}  pairs={sorted(all_pairs)}")

        only_ana = ana_pairs - gjk_pairs
        only_gjk = gjk_pairs - ana_pairs
        if only_ana:
            print(f"    step {step}: analytical-only pairs: {only_ana}")
        if only_gjk:
            mismatches.append(f"step {step}: analytical missed GJK pairs: {only_gjk}")

        common_pairs = gjk_pairs & ana_pairs
        for ga, gb in common_pairs:
            gjk_mask = (contacts_gjk["geom_a"] == ga) & (contacts_gjk["geom_b"] == gb)
            ana_mask = (contacts_ana["geom_a"] == ga) & (contacts_ana["geom_b"] == gb)

            gjk_pen = contacts_gjk["penetration"][gjk_mask]
            ana_pen = contacts_ana["penetration"][ana_mask]
            gjk_nrm = contacts_gjk["normal"][gjk_mask]
            ana_nrm = contacts_ana["normal"][ana_mask]
            gjk_pos = contacts_gjk["position"][gjk_mask]
            ana_pos = contacts_ana["position"][ana_mask]

            n_gjk = len(gjk_pen)
            n_ana = len(ana_pen)

            for j in range(n_gjk):
                print(f"      gjk[{j}] pos={gjk_pos[j]} pen={gjk_pen[j]:.6f} nrm={gjk_nrm[j]}")
            for j in range(n_ana):
                print(f"      ana[{j}] pos={ana_pos[j]} pen={ana_pen[j]:.6f} nrm={ana_nrm[j]}")

            if n_gjk >= 2 and n_ana < 2:
                mismatches.append(
                    f"step {step} pair ({ga},{gb}): GJK has {n_gjk} contacts "
                    f"but analytical only has {n_ana} (expected >= 2)"
                )

            if n_gjk == 1 and n_ana == 1:
                pos_err = np.linalg.norm(gjk_pos[0] - ana_pos[0])
                pen_err = abs(float(gjk_pen[0]) - float(ana_pen[0]))
                dot = np.dot(gjk_nrm[0], ana_nrm[0])
                max_pos_err_seen = max(max_pos_err_seen, pos_err)
                max_pen_err_seen = max(max_pen_err_seen, pen_err)
                min_dot_seen = min(min_dot_seen, dot)
                if pos_err > POS_TOL:
                    mismatches.append(f"step {step} pair ({ga},{gb}): position err={pos_err:.6f}")
                if pen_err > POS_TOL:
                    mismatches.append(f"step {step} pair ({ga},{gb}): penetration err={pen_err:.6f}")
                if dot < 0.95:
                    mismatches.append(f"step {step} pair ({ga},{gb}): normal dot={dot:.4f}")
                continue

            for lbl, pos, pen, nrm, n in [
                ("gjk", gjk_pos, gjk_pen, gjk_nrm, n_gjk),
                ("ana", ana_pos, ana_pen, ana_nrm, n_ana),
            ]:
                if n >= 3:
                    ei, ej = _find_line_endpoints(pos)
                    for k in range(n):
                        if k == ei or k == ej:
                            continue
                        d = _point_line_distance(pos[k], pos[ei], pos[ej])
                        max_collinear_err_seen = max(max_collinear_err_seen, d)
                        if d > POS_TOL:
                            mismatches.append(
                                f"step {step} pair ({ga},{gb}): {lbl} contact {k} not collinear (dist={d:.6f})"
                            )

            if n_gjk < 2 or n_ana < 2:
                continue

            gjk_ei, gjk_ej = _find_line_endpoints(gjk_pos)
            ana_ei, ana_ej = _find_line_endpoints(ana_pos)
            gjk_ends = np.array([gjk_pos[gjk_ei], gjk_pos[gjk_ej]])
            ana_ends = np.array([ana_pos[ana_ei], ana_pos[ana_ej]])

            err_same = np.linalg.norm(gjk_ends[0] - ana_ends[0]) + np.linalg.norm(gjk_ends[1] - ana_ends[1])
            err_swap = np.linalg.norm(gjk_ends[0] - ana_ends[1]) + np.linalg.norm(gjk_ends[1] - ana_ends[0])
            if err_swap < err_same:
                ana_ends = ana_ends[::-1]

            for idx, (ge, ae) in enumerate(zip(gjk_ends, ana_ends)):
                pos_err = np.linalg.norm(ge - ae)
                max_pos_err_seen = max(max_pos_err_seen, pos_err)
                if pos_err > POS_TOL:
                    mismatches.append(
                        f"step {step} pair ({ga},{gb}): endpoint {idx} position err={pos_err:.6f} (gjk={ge}, ana={ae})"
                    )

            best_gjk = int(np.argmax(gjk_pen))
            best_ana = int(np.argmax(ana_pen))
            dot = np.dot(gjk_nrm[best_gjk], ana_nrm[best_ana])
            min_dot_seen = min(min_dot_seen, dot)
            if dot < 0.95:
                mismatches.append(
                    f"step {step} pair ({ga},{gb}): normal dot={dot:.4f} "
                    f"(gjk={gjk_nrm[best_gjk]}, ana={ana_nrm[best_ana]})"
                )

            pen_err = abs(float(gjk_pen[best_gjk]) - float(ana_pen[best_ana]))
            max_pen_err_seen = max(max_pen_err_seen, pen_err)
            if pen_err > POS_TOL:
                mismatches.append(
                    f"step {step} pair ({ga},{gb}): deepest penetration err={pen_err:.6f} "
                    f"(gjk={gjk_pen[best_gjk]:.6f}, ana={ana_pen[best_ana]:.6f})"
                )

    # Verify cylinder-cylinder specialization was called
    expected_bits = {
        "cylinder_contact.func_cylinder_cylinder_contact": ERRNO_CALLED_CYLINDER_CYLINDER,
    }
    missing = [name for name, bit in expected_bits.items() if not (analytical_errno_seen & bit)]
    if missing:
        seen = [name for name, bit in ANALYTICAL_ERRNO_BITS.items() if analytical_errno_seen & bit]
        pytest.fail(
            f"Phase 2: expected specializations never called: {missing}\n"
            f"Specializations called: {seen}\n"
            f"errno bits seen: {analytical_errno_seen:#010x}"
        )

    print("\n=== CYLINDER ARENA DIAGNOSTICS ===")
    print(f"Steps with contacts: {steps_with_contacts}/{N_STEPS}")
    print(f"Total contacts — gjk: {total_contacts_gjk}, analytical: {total_contacts_ana}")
    print(f"Steps with different contact counts: {count_diff_steps}")
    print(f"Max endpoint position error: {max_pos_err_seen:.8f}  (threshold={POS_TOL})")
    print(f"Max penetration error: {max_pen_err_seen:.8f}  (threshold={POS_TOL})")
    print(f"Max collinearity error: {max_collinear_err_seen:.8f}  (threshold={POS_TOL})")
    print(f"Min normal dot: {min_dot_seen:.6f}  (threshold=0.95)")
    print(f"Mismatches (failures): {len(mismatches)}")
    print("==================================\n")

    if mismatches:
        msg = f"Analytical vs GJK cylinder fuzz: {len(mismatches)} mismatches:\n" + "\n".join(mismatches[:20])
        if len(mismatches) > 20:
            msg += f"\n... and {len(mismatches) - 20} more"
        pytest.fail(msg)


# ---------------------------------------------------------------------------
# Per-pair fuzz tests
# ---------------------------------------------------------------------------


def _add_mjcf_entity(scene, tmp_path, geom_type, name, pos, euler, radius, half_length, fixed=False):
    """Create and add a single MJCF primitive (cylinder, capsule, or sphere)."""
    if geom_type == "cylinder":
        mjcf = create_cylinder_mjcf(name, pos, euler, radius, half_length, fixed=fixed)
    elif geom_type == "capsule":
        mjcf = create_capsule_mjcf(name, pos, euler, radius, half_length, fixed=fixed)
    elif geom_type == "sphere":
        mjcf = create_sphere_mjcf(name, pos, radius)
    else:
        raise ValueError(f"Unknown geom_type: {geom_type}")
    path = tmp_path / f"{name}.xml"
    ET.ElementTree(mjcf).write(path)
    return scene.add_entity(gs.morphs.MJCF(file=str(path)))


def _build_wall_arena(scene, tmp_path, wall_type, free_bodies, arena_half=0.5):
    """Build a cage of fixed wall primitives with free bodies inside.

    Parameters
    ----------
    wall_type : str
        ``"cylinder"``, ``"capsule"``, or ``"box"`` — the primitive used for
        the walls.  ``"box"`` uses plain ``gs.morphs.Box`` (no analytical
        specialization) so only the free-body pair triggers analytical paths.
    free_bodies : list[dict]
        Each dict has keys: ``type``, ``pos``, ``euler``, ``radius``, ``half_length``.
    arena_half : float
        Half-width of the cubic arena (default 0.5).
    Returns the list of free-body entities.
    """
    ARENA_HALF = arena_half
    WALL_THICKNESS = 2

    if wall_type == "box":
        arena = ARENA_HALF + WALL_THICKNESS / 2
        wall_defs = [
            ((0, 0, -arena), (2 * arena, 2 * arena, WALL_THICKNESS)),
            ((0, 0, +arena), (2 * arena, 2 * arena, WALL_THICKNESS)),
            ((-arena, 0, 0), (WALL_THICKNESS, 2 * arena, 2 * arena)),
            ((+arena, 0, 0), (WALL_THICKNESS, 2 * arena, 2 * arena)),
            ((0, -arena, 0), (2 * arena, WALL_THICKNESS, 2 * arena)),
            ((0, +arena, 0), (2 * arena, WALL_THICKNESS, 2 * arena)),
        ]
        for pos, size in wall_defs:
            scene.add_entity(gs.morphs.Box(pos=pos, size=size, fixed=True))
    else:
        WALL_R = 0.08
        WALL_HALF_H = 1.0
        SPACING = 0.20
        idx = 0

        positions = [i * SPACING for i in range(-int(ARENA_HALF / SPACING), int(ARENA_HALF / SPACING) + 1)]

        def _add_wall(pos, euler):
            nonlocal idx
            name = f"wall_{idx}"
            idx += 1
            _add_mjcf_entity(scene, tmp_path, wall_type, name, pos, euler, WALL_R, WALL_HALF_H, fixed=True)

        for y in positions:
            _add_wall((0, y, -ARENA_HALF), (0, 90, 0))
            _add_wall((0, y, +ARENA_HALF), (0, 90, 0))
        for y in positions:
            _add_wall((-ARENA_HALF, y, 0), (0, 0, 0))
            _add_wall((+ARENA_HALF, y, 0), (0, 0, 0))
        for x in positions:
            _add_wall((x, -ARENA_HALF, 0), (0, 0, 0))
            _add_wall((x, +ARENA_HALF, 0), (0, 0, 0))

    entities = []
    for i, body_def in enumerate(free_bodies):
        name = f"free_{i}"
        ent = _add_mjcf_entity(
            scene,
            tmp_path,
            body_def["type"],
            name,
            body_def["pos"],
            body_def.get("euler", (0, 0, 0)),
            body_def["radius"],
            body_def.get("half_length", 0.0),
        )
        entities.append(ent)

    scene.build()
    return entities


def _run_pair_fuzz(
    monkeypatch,
    tmp_path,
    show_viewer,
    wall_type,
    free_bodies,
    expected_errno_bits,
    label,
    arena_half=0.5,
):
    """Reusable two-phase GJK-vs-analytical fuzz for a specific collision pair.

    Phase 1 (GJK reference): Run with GJK-forced kernel (known-good), record
        qpos, forces, and contacts.
    Phase 2 (Analytical replay): Replay same qpos+forces with instrumented
        analytical kernel, verify errno bits, compare contacts against GJK.
    """
    import torch

    N_STEPS = 200
    BOUNCE_FORCE = 600.0

    rigid_opts = gs.options.RigidOptions(
        dt=0.005,
        gravity=(0, 0, 0),
        enable_collision=True,
        use_gjk_collision=True,
        enable_multi_contact=True,
    )

    from genesis.engine.solvers.rigid.collider import narrowphase

    # --- Build both scenes BEFORE any patching ---
    gjk_dir = tmp_path / "gjk_mjcf"
    gjk_dir.mkdir()
    scene_gjk = gs.Scene(show_viewer=show_viewer, rigid_options=rigid_opts)
    entities_gjk = _build_wall_arena(scene_gjk, gjk_dir, wall_type, free_bodies)

    ana_dir = tmp_path / "ana_mjcf"
    ana_dir.mkdir()
    scene_ana = gs.Scene(show_viewer=False, rigid_options=rigid_opts)
    entities_ana = _build_wall_arena(scene_ana, ana_dir, wall_type, free_bodies)

    for body in entities_gjk:
        body.set_dofs_damping(np.array([15.0, 15.0, 15.0], dtype=gs.np_float), dofs_idx_local=[3, 4, 5])
    for body in entities_ana:
        body.set_dofs_damping(np.array([15.0, 15.0, 15.0], dtype=gs.np_float), dofs_idx_local=[3, 4, 5])

    # --- Phase 1: GJK reference run ---
    gjk_path = create_modified_narrowphase_file(tmp_path=tmp_path)
    spec_gjk = importlib.util.spec_from_file_location(f"narrowphase_{label}_gjk", gjk_path)
    narrowphase_gjk = importlib.util.module_from_spec(spec_gjk)
    spec_gjk.loader.exec_module(narrowphase_gjk)

    monkeypatch.setattr(
        narrowphase,
        "func_narrow_phase_convex_vs_convex",
        narrowphase_gjk.func_narrow_phase_convex_vs_convex,
    )

    torch.manual_seed(0)
    for body in entities_gjk:
        v = (torch.rand(body.n_dofs, dtype=gs.tc_float, device=gs.device) * 2 - 1) * 3.0
        body.set_dofs_velocity(v)

    recorded_qpos = []
    recorded_contacts = []
    recorded_forces = []
    gjk_confirmed = False

    torch.manual_seed(42)
    for step in range(N_STEPS):
        step_qpos = []
        for body in entities_gjk:
            q = body.get_qpos()
            step_qpos.append(q.cpu().numpy().copy() if hasattr(q, "cpu") else np.array(q).copy())
        recorded_qpos.append(step_qpos)

        scene_gjk._sim.rigid_solver._errno.fill(0)
        scene_gjk.step()
        recorded_contacts.append(_snapshot_contacts(scene_gjk))

        errno_val = int(scene_gjk._sim.rigid_solver._errno[0])
        if recorded_contacts[-1]["n"] > 0:
            assert (errno_val & ERRNO_CALLED_GJK) != 0, (
                f"[{label}] Step {step}: GJK not called despite {recorded_contacts[-1]['n']} contacts."
            )
            gjk_confirmed = True

        nc = recorded_contacts[-1]["n"]
        step_forces = []
        if nc > 0:
            hit_geoms = set(recorded_contacts[-1]["geom_a"].tolist()) | set(
                recorded_contacts[-1]["geom_b"].tolist()
            )
        else:
            hit_geoms = set()

        for body in entities_gjk:
            body_geoms = set(range(body.geom_start, body.geom_end))
            if body_geoms & hit_geoms:
                f = torch.zeros(body.n_dofs, dtype=gs.tc_float, device=gs.device)
                f[0:3] = (torch.rand(3, dtype=gs.tc_float, device=gs.device) * 2 - 1) * BOUNCE_FORCE
            else:
                f = torch.zeros(body.n_dofs, dtype=gs.tc_float, device=gs.device)
            step_forces.append(f.clone())
            body.control_dofs_force(f)
        recorded_forces.append(step_forces)

    assert gjk_confirmed, f"[{label}] Phase 1: GJK was never confirmed via errno."

    total_gjk = sum(c["n"] for c in recorded_contacts)
    all_pairs_seen = set()
    for c in recorded_contacts:
        if c["n"] > 0:
            all_pairs_seen.update(zip(c["geom_a"].tolist(), c["geom_b"].tolist()))
    print(f"[{label}] Phase 1 (GJK) OK. {total_gjk} contacts, pairs seen: {sorted(all_pairs_seen)}")

    # --- Phase 2: Analytical replay ---
    monkeypatch.undo()

    instrumented_path = create_instrumented_narrowphase_file(tmp_path=tmp_path)
    spec_inst = importlib.util.spec_from_file_location(f"narrowphase_{label}_inst", instrumented_path)
    narrowphase_instrumented = importlib.util.module_from_spec(spec_inst)
    spec_inst.loader.exec_module(narrowphase_instrumented)

    monkeypatch.setattr(
        narrowphase,
        "func_narrow_phase_convex_vs_convex",
        narrowphase_instrumented.func_narrow_phase_convex_vs_convex,
    )

    assert narrowphase.func_narrow_phase_convex_vs_convex is narrowphase_instrumented.func_narrow_phase_convex_vs_convex, (
        "monkeypatch failed: instrumented kernel not set"
    )

    mismatches = []
    total_contacts_gjk = 0
    total_contacts_ana = 0
    steps_with_contacts = 0
    max_pos_err_seen = 0.0
    max_pen_err_seen = 0.0
    min_dot_seen = 1.0
    analytical_errno_seen = 0

    for step in range(N_STEPS):
        for i, body in enumerate(entities_ana):
            body.set_qpos(recorded_qpos[step][i])
            body.zero_all_dofs_velocity()
        for i, body in enumerate(entities_ana):
            body.control_dofs_force(recorded_forces[step][i])

        scene_ana._sim.rigid_solver._errno.fill(0)
        scene_ana.step()
        errno_val = int(scene_ana._sim.rigid_solver._errno[0])
        analytical_errno_seen |= errno_val

        contacts_ana = _snapshot_contacts(scene_ana)
        contacts_gjk = recorded_contacts[step]

        total_contacts_gjk += contacts_gjk["n"]
        total_contacts_ana += contacts_ana["n"]

        if contacts_gjk["n"] == 0 and contacts_ana["n"] == 0:
            continue
        steps_with_contacts += 1

        gjk_pairs = (
            set(zip(contacts_gjk["geom_a"].tolist(), contacts_gjk["geom_b"].tolist()))
            if contacts_gjk["n"] > 0
            else set()
        )
        ana_pairs = (
            set(zip(contacts_ana["geom_a"].tolist(), contacts_ana["geom_b"].tolist()))
            if contacts_ana["n"] > 0
            else set()
        )

        only_gjk = gjk_pairs - ana_pairs
        if only_gjk:
            mismatches.append(f"step {step}: analytical missed GJK pairs: {only_gjk}")

        common_pairs = gjk_pairs & ana_pairs
        for ga, gb in common_pairs:
            gjk_mask = (contacts_gjk["geom_a"] == ga) & (contacts_gjk["geom_b"] == gb)
            ana_mask = (contacts_ana["geom_a"] == ga) & (contacts_ana["geom_b"] == gb)

            gjk_pen = contacts_gjk["penetration"][gjk_mask]
            ana_pen = contacts_ana["penetration"][ana_mask]
            gjk_nrm = contacts_gjk["normal"][gjk_mask]
            ana_nrm = contacts_ana["normal"][ana_mask]
            gjk_pos_arr = contacts_gjk["position"][gjk_mask]
            ana_pos_arr = contacts_ana["position"][ana_mask]

            if len(gjk_pen) == 1 and len(ana_pen) == 1:
                pos_err = np.linalg.norm(gjk_pos_arr[0] - ana_pos_arr[0])
                pen_err = abs(float(gjk_pen[0]) - float(ana_pen[0]))
                dot = np.dot(gjk_nrm[0], ana_nrm[0])
                max_pos_err_seen = max(max_pos_err_seen, pos_err)
                max_pen_err_seen = max(max_pen_err_seen, pen_err)
                min_dot_seen = min(min_dot_seen, dot)
                if pos_err > POS_TOL:
                    mismatches.append(f"step {step} ({ga},{gb}): pos err={pos_err:.6f}")
                if pen_err > POS_TOL:
                    mismatches.append(f"step {step} ({ga},{gb}): pen err={pen_err:.6f}")
                if dot < 0.95:
                    mismatches.append(f"step {step} ({ga},{gb}): normal dot={dot:.4f}")
            elif len(gjk_pen) >= 2 and len(ana_pen) >= 2:
                best_gjk = int(np.argmax(gjk_pen))
                best_ana = int(np.argmax(ana_pen))
                pen_err = abs(float(gjk_pen[best_gjk]) - float(ana_pen[best_ana]))
                max_pen_err_seen = max(max_pen_err_seen, pen_err)
                if pen_err > POS_TOL:
                    mismatches.append(f"step {step} ({ga},{gb}): pen err={pen_err:.6f}")
                dot = np.dot(gjk_nrm[best_gjk], ana_nrm[best_ana])
                min_dot_seen = min(min_dot_seen, dot)
                if dot < 0.95:
                    mismatches.append(f"step {step} ({ga},{gb}): normal dot={dot:.4f}")

    print(f"[{label}] Phase 2: total analytical contacts={total_contacts_ana}, errno={analytical_errno_seen:#010x}")

    missing = [name for name, bit in expected_errno_bits.items() if not (analytical_errno_seen & bit)]
    if missing:
        seen = [name for name, bit in ANALYTICAL_ERRNO_BITS.items() if analytical_errno_seen & bit]
        pytest.fail(
            f"[{label}] Phase 2: expected analytical specializations never called: {missing}\n"
            f"Specializations called: {seen}\n"
            f"errno bits seen: {analytical_errno_seen:#010x}"
        )

    print(f"\n=== {label} DIAGNOSTICS ===")
    print(f"Steps with contacts: {steps_with_contacts}/{N_STEPS}")
    print(f"Total contacts — gjk: {total_contacts_gjk}, analytical: {total_contacts_ana}")
    print(f"Max position error: {max_pos_err_seen:.8f}  (threshold={POS_TOL})")
    print(f"Max penetration error: {max_pen_err_seen:.8f}  (threshold={POS_TOL})")
    print(f"Min normal dot: {min_dot_seen:.6f}  (threshold=0.95)")
    print(f"Mismatches: {len(mismatches)}")

    if mismatches:
        msg = f"[{label}] {len(mismatches)} mismatches:\n" + "\n".join(mismatches[:20])
        if len(mismatches) > 20:
            msg += f"\n... and {len(mismatches) - 20} more"
        pytest.fail(msg)


@pytest.mark.required
@pytest.mark.parametrize("backend", [gs.gpu])
def test_sphere_sphere_fuzz(backend, monkeypatch, tmp_path, show_viewer, tol):
    """Fuzz: two free spheres in a box-wall arena.

    Box walls avoid cylinder-sphere wall collisions which have large contact
    position differences, keeping the test focused on sphere-sphere only.
    """
    _run_pair_fuzz(
        monkeypatch,
        tmp_path,
        show_viewer,
        wall_type="box",
        free_bodies=[
            {"type": "sphere", "pos": (0.05, 0.0, 0.0), "radius": 0.12, "half_length": 0},
            {"type": "sphere", "pos": (-0.05, 0.1, 0.0), "radius": 0.10, "half_length": 0},
        ],
        expected_errno_bits={
            "cylinder_contact.func_sphere_sphere_contact": ERRNO_CALLED_SPHERE_SPHERE,
        },
        label="sphere_sphere",
    )


@pytest.mark.required
@pytest.mark.parametrize("backend", [gs.gpu])
def test_cylinder_sphere_fuzz(backend, monkeypatch, tmp_path, show_viewer, tol):
    """Fuzz: one free cylinder + one free sphere in a box-wall arena.

    Bodies start close together so they collide quickly.  Box walls avoid
    interfering cylinder-cylinder wall collisions.
    """
    _run_pair_fuzz(
        monkeypatch,
        tmp_path,
        show_viewer,
        wall_type="box",
        free_bodies=[
            {"type": "cylinder", "pos": (0.0, 0.0, 0.0), "euler": (0, 0, 0), "radius": 0.10, "half_length": 0.20},
            {"type": "sphere", "pos": (0.15, 0.0, 0.0), "radius": 0.10, "half_length": 0},
        ],
        expected_errno_bits={
            "cylinder_contact.func_cylinder_sphere_contact": ERRNO_CALLED_CYLINDER_SPHERE,
        },
        label="cylinder_sphere",
    )


@pytest.mark.required
@pytest.mark.parametrize("backend", [gs.gpu])
def test_capsule_capsule_fuzz(backend, monkeypatch, tmp_path, show_viewer, tol):
    """Fuzz: two free capsules in a capsule-wall arena."""
    _run_pair_fuzz(
        monkeypatch,
        tmp_path,
        show_viewer,
        wall_type="capsule",
        free_bodies=[
            {"type": "capsule", "pos": (0.05, 0.0, 0.0), "euler": (0, 0, 0), "radius": 0.10, "half_length": 0.20},
            {
                "type": "capsule",
                "pos": (-0.05, 0.1, 0.05),
                "euler": (30, 45, 0),
                "radius": 0.09,
                "half_length": 0.18,
            },
        ],
        expected_errno_bits={
            "capsule_contact.func_capsule_capsule_contact": ERRNO_CALLED_CAPSULE_CAPSULE,
        },
        label="capsule_capsule",
    )


@pytest.mark.required
@pytest.mark.parametrize("backend", [gs.gpu])
def test_sphere_capsule_fuzz(backend, monkeypatch, tmp_path, show_viewer, tol):
    """Fuzz: one free sphere + one free capsule in a capsule-wall arena."""
    _run_pair_fuzz(
        monkeypatch,
        tmp_path,
        show_viewer,
        wall_type="capsule",
        free_bodies=[
            {"type": "sphere", "pos": (0.05, 0.0, 0.0), "radius": 0.10, "half_length": 0},
            {
                "type": "capsule",
                "pos": (-0.05, 0.1, 0.05),
                "euler": (0, 0, 0),
                "radius": 0.10,
                "half_length": 0.22,
            },
        ],
        expected_errno_bits={
            "capsule_contact.func_sphere_capsule_contact": ERRNO_CALLED_SPHERE_CAPSULE,
        },
        label="sphere_capsule",
    )
