"""QIPC coupler alignment tests: verify coupler output matches standalone QIPC on identical URDFs."""
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

try:
    import quadrants as qd
    from qipc import Scene as QIPCScene
except ImportError:
    pytest.skip("QIPC coupler requires 'quadrants' and 'qipc' packages.", allow_module_level=True)

import trimesh

import genesis as gs

from .conftest import TOL_DOUBLE


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SIMPLE_REVOLUTE_URDF = """\
<?xml version="1.0"?>
<robot name="test_two_link">
  <link name="base">
    <inertial><mass value="1.0"/><origin xyz="0 0 0" rpy="0 0 0"/>
      <inertia ixx="0.00167" ixy="0" ixz="0" iyy="0.00167" iyz="0" izz="0.00167"/></inertial>
    <visual><geometry><mesh filename="base.stl"/></geometry></visual>
    <collision><geometry><mesh filename="base.stl"/></geometry></collision>
  </link>
  <link name="child">
    <inertial><mass value="0.5"/><origin xyz="0 0 0" rpy="0 0 0"/>
      <inertia ixx="0.00083" ixy="0" ixz="0" iyy="0.00083" iyz="0" izz="0.00083"/></inertial>
    <visual><geometry><mesh filename="child.stl"/></geometry></visual>
    <collision><geometry><mesh filename="child.stl"/></geometry></collision>
  </link>
  <joint name="j1" type="revolute">
    <parent link="base"/><child link="child"/>
    <origin xyz="0 0 0.2" rpy="0 0 0"/><axis xyz="0 0 1"/>
    <limit lower="-1.57" upper="1.57"/>
  </joint>
</robot>
"""

FIXED_JOINT_URDF = """\
<?xml version="1.0"?>
<robot name="test_fixed_merge">
  <link name="base">
    <inertial><mass value="2.0"/><origin xyz="0 0 0" rpy="0 0 0"/>
      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/></inertial>
    <visual><geometry><mesh filename="base.stl"/></geometry></visual>
    <collision><geometry><mesh filename="base.stl"/></geometry></collision>
  </link>
  <link name="arm">
    <inertial><mass value="1.0"/><origin xyz="0 0 0" rpy="0 0 0"/>
      <inertia ixx="0.005" ixy="0" ixz="0" iyy="0.005" iyz="0" izz="0.005"/></inertial>
    <visual><geometry><mesh filename="child.stl"/></geometry></visual>
    <collision><geometry><mesh filename="child.stl"/></geometry></collision>
  </link>
  <link name="hand">
    <inertial><mass value="0.5"/><origin xyz="0 0 0" rpy="0 0 0"/>
      <inertia ixx="0.002" ixy="0" ixz="0" iyy="0.002" iyz="0" izz="0.002"/></inertial>
    <visual><geometry><mesh filename="child.stl"/></geometry></visual>
    <collision><geometry><mesh filename="child.stl"/></geometry></collision>
  </link>
  <link name="tool">
    <inertial><mass value="0.3"/><origin xyz="0 0 0" rpy="0 0 0"/>
      <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/></inertial>
    <visual><geometry><mesh filename="child.stl"/></geometry></visual>
    <collision><geometry><mesh filename="child.stl"/></geometry></collision>
  </link>
  <joint name="rev1" type="revolute">
    <parent link="base"/><child link="arm"/>
    <origin xyz="0 0 0.15" rpy="0 0 0"/><axis xyz="0 0 1"/>
    <limit lower="-1.57" upper="1.57"/>
  </joint>
  <joint name="fixed_arm_hand" type="fixed">
    <parent link="arm"/><child link="hand"/>
    <origin xyz="0 0 0.15" rpy="0 0 0.7854"/>
  </joint>
  <joint name="fixed_hand_tool" type="fixed">
    <parent link="hand"/><child link="tool"/>
    <origin xyz="0.05 0 0" rpy="0 0 0"/>
  </joint>
</robot>
"""


def _write_box_stl(path, extents=(0.1, 0.1, 0.1)):
    box = trimesh.creation.box(extents=list(extents))
    box.export(str(path), file_type="stl")


@pytest.fixture
def simple_revolute_dir(tmp_path):
    (tmp_path / "robot.urdf").write_text(SIMPLE_REVOLUTE_URDF)
    _write_box_stl(tmp_path / "base.stl")
    _write_box_stl(tmp_path / "child.stl")
    return tmp_path


@pytest.fixture
def fixed_joint_dir(tmp_path):
    (tmp_path / "robot.urdf").write_text(FIXED_JOINT_URDF)
    _write_box_stl(tmp_path / "base.stl")
    _write_box_stl(tmp_path / "child.stl")
    return tmp_path


# ---------------------------------------------------------------------------
# Scene builders
# ---------------------------------------------------------------------------


def build_genesis_scene(urdf_path, position=(0.0, 0.0, 0.0), is_fixed=True, kappa=1e8, kp=200.0, kv=50.0,
                        kappa_pivot=1e8, kappa_axis=1e8):
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=0.01,
            gravity=(0.0, 0.0, -9.81),
        ),
        coupler_options=gs.options.QIPCCouplerOptions(
            contact_enable=False,
            debug_viewer=False,
        ),
        show_viewer=False,
    )
    scene.add_entity(
        morph=gs.morphs.URDF(
            file=str(urdf_path),
            pos=position,
            fixed=is_fixed,
        ),
        material=gs.materials.Rigid(
            qipc_abd_kappa=kappa,
            qipc_kappa_pivot=kappa_pivot,
            qipc_kappa_axis=kappa_axis,
            qipc_default_kp=kp,
            qipc_default_kv=kv,
        ),
    )
    scene.build()
    return scene


def build_standalone_scene(urdf_path, position=(0.0, 0.0, 0.0), is_fixed=True, kappa=1e8, kp=200.0, kv=50.0,
                           kappa_pivot=1e8, kappa_axis=1e8):
    scene = QIPCScene(
        dt=0.01,
        gravity=(0.0, 0.0, -9.81),
        **{"contact/enable": False},
    )
    model = scene.add_urdf(
        urdf_path,
        kappa=kappa,
        fix_base=is_fixed,
        position=position,
        kappa_pivot=kappa_pivot,
        kappa_axis=kappa_axis,
        enable_controller=True,
        default_kp=kp,
        default_kv=kv,
    )
    scene.init()
    return scene, model


def _get_constitution_data(scene, cls_name):
    for cls, data in scene._constitution_data.items():
        name = cls.__name__ if isinstance(cls, type) else str(cls)
        if name == cls_name:
            return data
    return None


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


SCENE_PARAMS = dict(
    position=(0.0, 0.0, 0.3),
    is_fixed=True,
    kappa=1e8,
    kp=200.0,
    kv=50.0,
    kappa_pivot=1e8,
    kappa_axis=1e8,
)


class TestQIPCAlignment:
    """Verify that Genesis+QIPCCoupler produces identical solver state as standalone QIPC."""

    def test_init_state_revolute(self, simple_revolute_dir):
        """ABD q, joint parameters, and controller gains match at init for a revolute joint."""
        urdf_path = simple_revolute_dir / "robot.urdf"
        standalone, _ = build_standalone_scene(urdf_path, **SCENE_PARAMS)
        genesis = build_genesis_scene(urdf_path, **SCENE_PARAMS)
        coupler_scene = genesis.sim._coupler._scene

        assert coupler_scene.affine_body.q.shape == standalone.affine_body.q.shape
        torch.testing.assert_close(coupler_scene.affine_body.q, standalone.affine_body.q, atol=TOL_DOUBLE, rtol=0)

        rev_a = _get_constitution_data(coupler_scene, "AffineBodyRevoluteJoint")
        rev_b = _get_constitution_data(standalone, "AffineBodyRevoluteJoint")
        assert rev_a is not None and rev_b is not None
        for key in ("anchor_left", "anchor_right", "axis_left", "axis_right", "kappa_pivot", "kappa_axis"):
            torch.testing.assert_close(rev_a[key], rev_b[key], atol=TOL_DOUBLE, rtol=0)

        torch.testing.assert_close(
            coupler_scene.joint_system.kp, standalone.joint_system.kp, atol=TOL_DOUBLE, rtol=0
        )
        torch.testing.assert_close(
            coupler_scene.joint_system.kv, standalone.joint_system.kv, atol=TOL_DOUBLE, rtol=0
        )

    def test_init_state_fixed_joint_merge(self, fixed_joint_dir):
        """Fixed joints cause link merging; merged body count and transforms must match standalone."""
        urdf_path = fixed_joint_dir / "robot.urdf"
        standalone, _ = build_standalone_scene(urdf_path, **SCENE_PARAMS)
        genesis = build_genesis_scene(urdf_path, **SCENE_PARAMS)
        coupler_scene = genesis.sim._coupler._scene

        assert coupler_scene.affine_body.q.shape[0] == standalone.affine_body.q.shape[0]
        torch.testing.assert_close(coupler_scene.affine_body.q, standalone.affine_body.q, atol=TOL_DOUBLE, rtol=0)

    def test_step_alignment(self, simple_revolute_dir):
        """After stepping under gravity with no control, ABD q and theta stay identical."""
        urdf_path = simple_revolute_dir / "robot.urdf"
        standalone, _ = build_standalone_scene(urdf_path, **SCENE_PARAMS)
        genesis = build_genesis_scene(urdf_path, **SCENE_PARAMS)
        coupler = genesis.sim._coupler
        coupler_scene = coupler._scene

        for _ in range(5):
            standalone.step()
            coupler._skip_first_step = False
            coupler._substep_count = coupler._substeps_per_step - 1
            coupler.couple(0)

        torch.testing.assert_close(coupler_scene.affine_body.q, standalone.affine_body.q, atol=TOL_DOUBLE, rtol=0)
        torch.testing.assert_close(
            coupler_scene.joint_system.theta, standalone.joint_system.theta, atol=TOL_DOUBLE, rtol=0,
        )

    def test_control_alignment(self, simple_revolute_dir):
        """Applying the same target and stepping produces matching theta trajectories."""
        urdf_path = simple_revolute_dir / "robot.urdf"
        standalone, model = build_standalone_scene(urdf_path, **SCENE_PARAMS)
        genesis = build_genesis_scene(urdf_path, **SCENE_PARAMS)
        coupler = genesis.sim._coupler

        target = torch.tensor([0.5], dtype=torch.float64, device="cuda")
        model.control_dofs_position(target)
        coupler._jc.control_dofs_position(target)

        for _ in range(10):
            standalone.step()
            coupler._skip_first_step = False
            coupler._substep_count = coupler._substeps_per_step - 1
            coupler.couple(0)

        torch.testing.assert_close(
            coupler._scene.joint_system.theta, standalone.joint_system.theta, atol=TOL_DOUBLE, rtol=0,
        )


# ---------------------------------------------------------------------------
# Ground contact and multi-entity tests
# ---------------------------------------------------------------------------


class TestGroundContact:
    """IPC ground contact: strict no-penetration via half-plane barrier."""

    def test_ground_contact(self, simple_revolute_dir, show_viewer):
        """A free ABD box falls onto ground and stops above z=0 (IPC no-penetration)."""

        scene = gs.Scene(
            sim_options=gs.options.SimOptions(
                dt=0.01,
                gravity=(0.0, 0.0, -9.81),
            ),
            coupler_options=gs.options.QIPCCouplerOptions(
                contact_enable=True,
                contact_d_hat=0.01,
                init_collision_pair_capacity=1000,
            ),
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(0.5, -0.5, 0.7),
                camera_lookat=(0.0, 0.0, 0.3),
            ),
            show_viewer=show_viewer,
        )

        scene.add_entity(gs.morphs.Plane())

        drop_height = 0.5
        urdf_path = simple_revolute_dir / "robot.urdf"
        box = scene.add_entity(
            morph=gs.morphs.URDF(
                file=str(urdf_path),
                pos=(0, 0, drop_height),
                fixed=False,
            ),
            material=gs.materials.Rigid(
                qipc_abd_kappa=1e8,
                qipc_kappa_pivot=1e5,
                qipc_kappa_axis=1e5,
            ),
        )

        scene.build()

        for _ in range(100):
            scene.step()

        pos = box.get_pos()
        z = float(pos[2]) if pos.dim() == 1 else float(pos[0, 2])
        assert z > 0, f"IPC no-penetration violated: z={z}"

    def test_multi_entity(self, simple_revolute_dir, show_viewer):
        """Two entities + ground: each entity's state is independently correct."""

        scene = gs.Scene(
            sim_options=gs.options.SimOptions(
                dt=0.01,
                gravity=(0.0, 0.0, -9.81),
            ),
            coupler_options=gs.options.QIPCCouplerOptions(
                contact_enable=True,
                contact_d_hat=0.01,
                init_collision_pair_capacity=2000,
            ),
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(1.0, -1.0, 0.8),
                camera_lookat=(0.5, 0.0, 0.3),
            ),
            show_viewer=show_viewer,
        )

        scene.add_entity(gs.morphs.Plane())

        urdf_path = simple_revolute_dir / "robot.urdf"

        robot_a = scene.add_entity(
            morph=gs.morphs.URDF(
                file=str(urdf_path),
                pos=(0, 0, 0.3),
                fixed=True,
            ),
            material=gs.materials.Rigid(
                qipc_abd_kappa=1e8,
                qipc_kappa_pivot=1e5,
                qipc_kappa_axis=1e5,
                qipc_default_kp=500.0,
                qipc_default_kv=50.0,
            ),
        )

        robot_b = scene.add_entity(
            morph=gs.morphs.URDF(
                file=str(urdf_path),
                pos=(1, 0, 0.3),
                fixed=True,
            ),
            material=gs.materials.Rigid(
                qipc_abd_kappa=1e8,
                qipc_kappa_pivot=1e5,
                qipc_kappa_axis=1e5,
                qipc_default_kp=500.0,
                qipc_default_kv=50.0,
            ),
        )

        scene.build()

        robot_a.control_dofs_position(0.5)
        robot_b.control_dofs_position(-0.5)

        for _ in range(50):
            scene.step()

        qa = float(robot_a.get_dofs_position()[0])
        qb = float(robot_b.get_dofs_position()[0])

        # Verify global DOF index mapping: each robot received its own target,
        # not the other's. If indices are crossed, qa would be negative or zero.
        # Measured values: qa ~ 0.497, qb ~ -0.497 (target 0.5, PD tracking error < 1%)
        assert qa > 0.45, f"robot_a should track positive target, got qa={qa}"
        assert qb < -0.45, f"robot_b should track negative target, got qb={qb}"
        # Symmetric setup (same kp/kv/dt/geometry) -> exactly symmetric response
        assert abs(qa + qb) < 1e-6, f"robot_a and robot_b should be symmetric, sum={qa + qb}"


# ---------------------------------------------------------------------------
# Joint limits and stacked free-base tests
# ---------------------------------------------------------------------------


def _build_two_cube_joint_mjcf(joint_type: str, joint_limits: tuple[float, float], *, fixed: bool = True) -> str:
    """Build a two-cube MJCF with a revolute or prismatic joint.

    Ported from tests/ipc/test_rigid.py::build_two_cube_joint_mjcf (commit d9c147c).
    """
    import xml.etree.ElementTree as ET

    mjcf = ET.Element("mujoco", model=f"two_cube_{joint_type}")
    ET.SubElement(mjcf, "compiler", angle="radian")
    worldbody = ET.SubElement(mjcf, "worldbody")
    base = ET.SubElement(worldbody, "body", name="base")
    if not fixed:
        ET.SubElement(base, "freejoint", name="root")
    ET.SubElement(base, "geom", type="box", size="0.05 0.05 0.05")
    ET.SubElement(base, "inertial", mass="1.0", pos="0 0 0", diaginertia="0.00667 0.00667 0.00667")
    child = ET.SubElement(base, "body", name="moving", pos="0.1 0 0")
    ET.SubElement(child, "geom", type="box", size="0.05 0.05 0.05", pos="0.1 0 0")
    ET.SubElement(child, "inertial", mass="1.0", pos="0 0 0", diaginertia="0.00667 0.00667 0.00667")
    mj_type = "hinge" if joint_type == "revolute" else "slide"
    axis = "0 1 0" if joint_type == "revolute" else "1 0 0"
    lo, hi = joint_limits
    ET.SubElement(child, "joint", name="joint1", type=mj_type, axis=axis, range=f"{lo} {hi}")
    return ET.tostring(mjcf, encoding="unicode")


class TestJointLimits:
    """Joint position limits via bang-bang velocity control.

    Ported from tests/ipc/test_rigid.py::test_joint_position_limits_bang_bang
    (commit d9c147c, 51d3a84 by duburcqa). Parameters and tolerance (0.05) match
    the IPC version; scene setup adapted for QIPCCoupler.
    """

    @pytest.mark.parametrize("joint_type", ["revolute", "prismatic"])
    def test_joint_position_limits_bang_bang(self, joint_type, show_viewer):
        """Bang-bang velocity control respects joint theta limits."""
        DT = 0.01
        V_MAX = 2.0
        HALF_PERIOD = 60
        NUM_OSCILLATIONS = 2

        if joint_type == "revolute":
            limits = (-0.5, 0.5)
        else:
            limits = (-0.3, 0.3)

        mjcf_content = _build_two_cube_joint_mjcf(joint_type, limits, fixed=True)

        scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=DT, gravity=(0.0, 0.0, 0.0)),
            coupler_options=gs.options.QIPCCouplerOptions(
                contact_enable=False,
            ),
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(0.3, -0.3, 0.6),
                camera_lookat=(0.1, 0.0, 0.5),
            ),
            show_viewer=show_viewer,
        )

        robot = scene.add_entity(
            morph=gs.morphs.MJCF(file=mjcf_content, pos=(0, 0, 0.5)),
            material=gs.materials.Rigid(
                qipc_abd_kappa=1e8,
                qipc_kappa_pivot=1e5,
                qipc_kappa_axis=1e5,
                qipc_default_kp=0.0,
                qipc_default_kv=100.0,
            ),
        )

        scene.build()

        pos_history = []
        total_steps = 2 * HALF_PERIOD * NUM_OSCILLATIONS
        for step in range(total_steps):
            phase = (step // HALF_PERIOD) % 2
            vel_target = V_MAX if phase == 0 else -V_MAX
            robot.control_dofs_velocity(vel_target, dofs_idx_local=-1)
            scene.step()
            pos_history.append(float(robot.get_dofs_position(dofs_idx_local=-1)[0]))

        pos_arr = np.array(pos_history)
        lower, upper = limits
        tolerance = 0.05

        assert pos_arr.min() >= lower - tolerance, (
            f"Joint violated lower limit: min={pos_arr.min():.4f}, limit={lower}"
        )
        assert pos_arr.max() <= upper + tolerance, (
            f"Joint violated upper limit: max={pos_arr.max():.4f}, limit={upper}"
        )
        assert pos_arr.max() > 0.1, (
            f"Joint didn't reach positive excursion: max={pos_arr.max():.4f}"
        )
        assert pos_arr.min() < -0.1, (
            f"Joint didn't reach negative excursion: min={pos_arr.min():.4f}"
        )


class TestStackedFreeBodies:
    """Multiple free-base entities stacking on ground with IPC contact.

    Ported from tests/ipc/test_rigid.py::test_stacked_revolute_pairs_collision
    (commit d9c147c by duburcqa). The IPC version is xfail (external_articulation
    does not support non-fixed base); QIPC passes because free-base is just a
    free ABD body.
    """

    def test_stacked_free_base_collision(self, show_viewer):
        """Free-base cubes stacked on ground maintain order and no penetration."""
        DT = 0.01
        CONTACT_D_HAT = 0.01
        NUM_SETTLE = 100

        mjcf_content = _build_two_cube_joint_mjcf("revolute", (-1.57, 1.57), fixed=False)

        scene = gs.Scene(
            sim_options=gs.options.SimOptions(
                dt=DT,
                gravity=(0.0, 0.0, -9.81),
            ),
            coupler_options=gs.options.QIPCCouplerOptions(
                contact_enable=True,
                contact_d_hat=CONTACT_D_HAT,
                init_collision_pair_capacity=5000,
            ),
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(0.5, -0.5, 1.0),
                camera_lookat=(0.0, 0.0, 0.3),
            ),
            show_viewer=show_viewer,
        )

        scene.add_entity(gs.morphs.Plane())

        heights = [0.15, 0.40, 0.65]
        robots = []
        for h in heights:
            r = scene.add_entity(
                morph=gs.morphs.MJCF(file=mjcf_content, pos=(0, 0, h)),
                material=gs.materials.Rigid(qipc_abd_kappa=1e8),
            )
            robots.append(r)

        scene.build()

        for _ in range(NUM_SETTLE):
            scene.step()

        base_z = []
        for r in robots:
            pos = r.get_pos()
            z = float(pos[2]) if pos.dim() == 1 else float(pos[0, 2])
            base_z.append(z)

        for i, z in enumerate(base_z):
            assert z > 0, f"Robot {i} penetrated ground: z={z:.4f}"

        for i in range(len(base_z) - 1):
            assert base_z[i] < base_z[i + 1], (
                f"Stacking order violated: robot {i} z={base_z[i]:.4f} >= robot {i+1} z={base_z[i+1]:.4f}"
            )
