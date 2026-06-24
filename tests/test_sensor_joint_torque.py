"""
Tests for JointTorqueSensor against analytical solutions on a simple pendulum.
"""

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

import genesis as gs

from .utils import assert_allclose

# ── pendulum parameters ──────────────────────────────────────────────────────
PENDULUM_M = 1.0                         # link mass, kg
PENDULUM_L = 1.0                         # pivot-to-mass distance, m
PENDULUM_I = PENDULUM_M * PENDULUM_L**2  # moment of inertia around pivot, kg⋅m²
G = 9.81                                 # gravity, m/s²
DT = 0.002                               # simulation timestep, s

# ── tolerances ───────────────────────────────────────────────────────────────
# Algebraic equality (tau_sensor = tau_ctrl when armature = frictionloss = 0).
# Symplectic Euler makes velocity update exact, so the only error is fp rounding.
EXACT_TOL = 1e-3  # N⋅m, generous for float32 accumulation

# Analytical check using numerically-differentiated θ̈.
# Under symplectic Euler: qacc_num = Δvel / dt is exact; residual error comes
# from evaluating the gravity term at θ_before vs the midpoint (O(dt) effect).
PHYS_TOL = 0.05  # N⋅m


# ── helpers ──────────────────────────────────────────────────────────────────


def _gravity_torque(theta: float) -> float:
    """Gravity contribution to the physical joint torque at angle *theta* (rad).

    Q_grav = −m·g·L·sin θ (restoring).  The physical joint torque is
    M_link·θ̈ − Q_grav = M_link·θ̈ + m·g·L·sin θ, so this returns
    m·g·L·sin θ — the gravity part alone.
    """
    return float(PENDULUM_M * G * PENDULUM_L * np.sin(theta))


def _make_pendulum_xml(
    tmp_path: Path,
    *,
    armature: float = 0.0,
    frictionloss: float = 0.0,
) -> Path:
    """Write a single-link pendulum MJCF and return the file path."""
    mjcf = ET.Element("mujoco", model="pendulum")
    ET.SubElement(mjcf, "compiler", angle="radian")
    ET.SubElement(mjcf, "option", timestep=str(DT), gravity=f"0 0 -{G}")
    worldbody = ET.SubElement(mjcf, "worldbody")
    arm = ET.SubElement(worldbody, "body", name="arm")
    ET.SubElement(
        arm,
        "joint",
        name="hinge",
        type="hinge",
        axis="0 1 0",
        armature=str(armature),
        frictionloss=str(frictionloss),
    )
    mass_body = ET.SubElement(arm, "body", name="mass", pos=f"0 0 -{PENDULUM_L}")
    ET.SubElement(mass_body, "geom", type="sphere", size="0.05", mass=str(PENDULUM_M))
    path = tmp_path / "pendulum.xml"
    ET.ElementTree(mjcf).write(path)
    return path


def _build_scene(
    tmp_path: Path,
    show_viewer: bool,
    *,
    armature: float = 0.0,
    frictionloss: float = 0.0,
    wall: bool = False,
):
    """Build a pendulum scene with an optional fixed wall obstacle.

    The wall (when *wall=True*) is a fixed box placed in the path of the mass
    as it swings from θ = 0 toward positive angles.  Its face is at roughly
    x = 0.72, which the sphere (r = 0.05) at (L·sin θ, 0, −L·cos θ) reaches
    near θ ≈ 45°.

    Entities must be added before scene.build(); frictionloss rank for the
    sensor is also fixed at build time, so frictionloss must be in the MJCF.
    """
    xml_path = _make_pendulum_xml(tmp_path, armature=armature, frictionloss=frictionloss)
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=DT),
        show_viewer=show_viewer,
    )
    pendulum = scene.add_entity(gs.morphs.MJCF(file=str(xml_path), align=False))
    if wall:
        # Fixed box positioned so the pendulum mass (sphere r=0.05) at θ=π/4
        # is already in contact: mass centre = (0.707, 0, −0.707), box face at x=0.75
        # (half-size 0.1, centre x=0.85) -> gap = 0.75−0.707 = 0.043 < 0.05 = radius.
        # Any positive torque presses the mass into the wall from the very first step.
        scene.add_entity(
            gs.morphs.Box(pos=(0.85, 0.0, -0.70), size=(0.2, 0.5, 0.2), fixed=True)
        )
    sensor = scene.add_sensor(
        gs.options.sensors.JointTorqueSensor(
            entity_idx=pendulum.idx,
            dofs_idx_local=(0,),
        )
    )
    scene.build()
    return scene, pendulum, sensor


# ── tests ─────────────────────────────────────────────────────────────────────


@pytest.mark.required
@pytest.mark.parametrize("backend", [gs.cpu, gs.gpu])
def test_joint_torque_sensor_no_armature_no_friction(backend, tmp_path, show_viewer):
    """tau_sensor = tau_control at every step when armature = frictionloss = 0.

    Also verifies the analytical identity tau_control = M_link·θ̈ + m·g·L·sin(θ)
    via numerical differentiation of the joint velocity.
    """
    scene, pendulum, sensor = _build_scene(tmp_path, show_viewer)

    # Non-zero initial angle so gravity contributes from the first step.
    pendulum.set_qpos([np.pi / 6])

    tau_c = 2.5  # N⋅m constant applied torque

    vel_prev = pendulum.get_dofs_velocity()[0].item()

    for _ in range(20):
        theta_before = pendulum.get_dofs_position()[0].item()
        pendulum.control_dofs_force([tau_c], [0])
        scene.step()

        vel_after = pendulum.get_dofs_velocity()[0].item()
        qacc_num = (vel_after - vel_prev) / DT

        tau_s = sensor.read()[0].item()
        tau_ctrl = pendulum.get_dofs_control_force([0]).item()

        # tau_sensor = tau_control (no armature, no friction -> exact algebraic equality)
        assert_allclose(
            tau_s, tau_ctrl, tol=EXACT_TOL,
            err_msg="tau_sensor != tau_control (no armature, no friction)",
        )
        assert_allclose(
            tau_s, tau_c, tol=EXACT_TOL,
            err_msg="tau_sensor != applied force tau_c",
        )

        # tau_sensor = M_link·θ̈ + gravity_torque(θ_before)
        # (gravity is evaluated at the state used to compute forces for this step)
        tau_phys = PENDULUM_I * qacc_num + _gravity_torque(theta_before)
        assert_allclose(
            tau_s, tau_phys, tol=PHYS_TOL,
            err_msg="tau_sensor != physical joint torque (no armature, no friction)",
        )

        vel_prev = vel_after


@pytest.mark.required
@pytest.mark.parametrize("backend", [gs.cpu, gs.gpu])
def test_joint_torque_sensor_with_armature(backend, tmp_path, show_viewer):
    """With armature I_arm: tau_sensor = tau_control − I_arm·θ̈ = physical joint torque.

    The armature inertia is consumed on the motor side of the gearbox; what
    reaches the link is the control force minus the armature's share of the
    acceleration load.
    """
    I_arm = 0.5  # kg⋅m²
    scene, pendulum, sensor = _build_scene(tmp_path, show_viewer, armature=I_arm)

    # Start at θ = 0 so gravity is zero for the first step, giving a clean
    # analytical check: qacc_0 = tau_c / (M_link + I_arm).
    tau_c = 3.0  # N⋅m

    vel_prev = pendulum.get_dofs_velocity()[0].item()

    for _ in range(20):
        theta_before = pendulum.get_dofs_position()[0].item()
        pendulum.control_dofs_force([tau_c], [0])
        scene.step()

        vel_after = pendulum.get_dofs_velocity()[0].item()
        qacc_num = (vel_after - vel_prev) / DT

        tau_s = sensor.read()[0].item()
        tau_ctrl = pendulum.get_dofs_control_force([0]).item()

        # Physical joint torque: what the link actually receives.
        tau_phys = PENDULUM_I * qacc_num + _gravity_torque(theta_before)

        # The sensor must match the physical torque (not the motor torque).
        assert_allclose(
            tau_s, tau_phys, tol=PHYS_TOL,
            err_msg="tau_sensor != physical joint torque (with armature)",
        )

        # Equivalently: tau_sensor − tau_control = −I_arm · θ̈
        # (armature absorbs its share of the acceleration)
        correction = -I_arm * qacc_num
        assert_allclose(
            tau_s - tau_ctrl, correction, tol=PHYS_TOL,
            err_msg="tau_sensor − tau_control != −I_arm · θ̈",
        )

        # Sanity: sensor must be strictly less than control when qacc > 0.
        if abs(qacc_num) > 0.1:
            assert tau_s < tau_ctrl, (
                f"Expected tau_sensor < tau_control when accelerating with armature, "
                f"got tau_sensor={tau_s:.4f} tau_control={tau_ctrl:.4f} θ̈={qacc_num:.4f}"
            )

        vel_prev = vel_after


@pytest.mark.required
@pytest.mark.parametrize("backend", [gs.cpu, gs.gpu])
def test_joint_torque_sensor_with_friction(backend, tmp_path, show_viewer):
    """With frictionloss: tau_sensor = M_link·θ̈ + gravity_torque (physical torque), not tau_control.

    Friction is dissipated in the gearbox; the sensor on the output shaft reads
    the torque that actually reaches the link, which is less than tau_control when
    the joint is moving.
    """
    frictionloss = 0.4  # N⋅m
    scene, pendulum, sensor = _build_scene(tmp_path, show_viewer, frictionloss=frictionloss)

    # Start at 30° with enough torque to keep moving against friction.
    pendulum.set_qpos([np.pi / 6])
    tau_c = 5.0  # N⋅m (well above frictionloss so the pendulum actually moves)

    vel_prev = pendulum.get_dofs_velocity()[0].item()

    for _ in range(20):
        theta_before = pendulum.get_dofs_position()[0].item()
        pendulum.control_dofs_force([tau_c], [0])
        scene.step()

        vel_after = pendulum.get_dofs_velocity()[0].item()
        qacc_num = (vel_after - vel_prev) / DT

        tau_s = sensor.read()[0].item()
        tau_ctrl = pendulum.get_dofs_control_force([0]).item()

        # Physical joint torque is always the Newton ground truth.
        tau_phys = PENDULUM_I * qacc_num + _gravity_torque(theta_before)
        assert_allclose(
            tau_s, tau_phys, tol=PHYS_TOL,
            err_msg="tau_sensor != physical joint torque (with frictionloss)",
        )

        # When the joint is actively moving, friction dissipates part of the
        # control force, so tau_sensor < tau_control.
        if abs(vel_after) > 0.05:  # only check when clearly in motion
            assert tau_s < tau_ctrl, (
                f"Expected tau_sensor < tau_control when moving with friction, "
                f"got tau_sensor={tau_s:.4f} tau_control={tau_ctrl:.4f} vel={vel_after:.4f}"
            )

        vel_prev = vel_after


@pytest.mark.required
@pytest.mark.parametrize("backend", [gs.cpu, gs.gpu])
def test_joint_torque_sensor_with_obstacle(backend, tmp_path, show_viewer):
    """tau_sensor = tau_control at every step (no armature, no friction), even during contact.

    Contact forces change θ̈ but do not appear explicitly in the sensor formula;
    they flow through the EOM and are already captured by tau_control (Newton 3rd
    law at the joint).  The formula tau_sensor = tau_control must therefore hold
    before, during, and after impact.
    """
    scene, pendulum, sensor = _build_scene(tmp_path, show_viewer, wall=True)

    # Start at θ = π/4 (45°).  The wall face is at x = 0.75; the mass centre at
    # this angle is x = 0.707, so the gap (0.043) is less than sphere radius (0.05):
    # contact is active from step 1.  Positive torque keeps pressing into the wall.
    pendulum.set_qpos([np.pi / 4])
    tau_c = 10.0  # N⋅m

    for _ in range(30):
        pendulum.control_dofs_force([tau_c], [0])
        scene.step()

        tau_s = sensor.read()[0].item()
        tau_ctrl = pendulum.get_dofs_control_force([0]).item()

        # tau_sensor = tau_control regardless of contact (no armature, no friction).
        assert_allclose(
            tau_s, tau_ctrl, tol=EXACT_TOL,
            err_msg="tau_sensor != tau_control in contact scenario",
        )
        assert_allclose(
            tau_s, tau_c, tol=EXACT_TOL,
            err_msg="tau_sensor != applied force tau_c in contact scenario",
        )
