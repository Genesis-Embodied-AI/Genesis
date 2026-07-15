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
            rigid_abd_kappa=kappa,
            joint_kappa_pivot=kappa_pivot,
            joint_kappa_axis=kappa_axis,
            default_kp=kp,
            default_kv=kv,
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
        torch.testing.assert_close(coupler_scene.affine_body.q, standalone.affine_body.q, atol=1e-12, rtol=0)

        rev_a = _get_constitution_data(coupler_scene, "AffineBodyRevoluteJoint")
        rev_b = _get_constitution_data(standalone, "AffineBodyRevoluteJoint")
        assert rev_a is not None and rev_b is not None
        for key in ("anchor_left", "anchor_right", "axis_left", "axis_right", "kappa_pivot", "kappa_axis"):
            torch.testing.assert_close(rev_a[key], rev_b[key], atol=1e-12, rtol=0)

        torch.testing.assert_close(coupler_scene.solver._joint_kp, standalone.solver._joint_kp, atol=1e-12, rtol=0)
        torch.testing.assert_close(coupler_scene.solver._joint_kv, standalone.solver._joint_kv, atol=1e-12, rtol=0)

    def test_init_state_fixed_joint_merge(self, fixed_joint_dir):
        """Fixed joints cause link merging; merged body count and transforms must match standalone."""
        urdf_path = fixed_joint_dir / "robot.urdf"
        standalone, _ = build_standalone_scene(urdf_path, **SCENE_PARAMS)
        genesis = build_genesis_scene(urdf_path, **SCENE_PARAMS)
        coupler_scene = genesis.sim._coupler._scene

        assert coupler_scene.affine_body.q.shape[0] == standalone.affine_body.q.shape[0]
        torch.testing.assert_close(coupler_scene.affine_body.q, standalone.affine_body.q, atol=1e-12, rtol=0)

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

        torch.testing.assert_close(coupler_scene.affine_body.q, standalone.affine_body.q, atol=1e-6, rtol=0)
        torch.testing.assert_close(
            coupler_scene.solver._joint_theta, standalone.solver._joint_theta, atol=1e-6, rtol=0,
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
            coupler._scene.solver._joint_theta, standalone.solver._joint_theta, atol=1e-6, rtol=0,
        )
