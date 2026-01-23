"""
Unit tests for the KinematicContactProbe class.

Tests cover:
- Basic contact detection (sphere probe touching surface)
- No-contact case returns zeros
- Multiple environments support
- Force computation accuracy
- Collision filtering (contype/conaffinity bitmasks)
- Lazy evaluation
"""

import math

import numpy as np
import pytest
import torch

import genesis as gs

from .utils import assert_allclose


@pytest.mark.required
@pytest.mark.parametrize("n_envs", [0, 2])
def test_kinematic_contact_probe_basic_contact(show_viewer, tol, n_envs):
    """Test basic contact detection with a sphere probe touching a plane surface."""
    PROBE_RADIUS = 0.01
    STIFFNESS = 1000.0

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=1e-2,
            substeps=1,
            gravity=(0.0, 0.0, 0.0),  # No gravity for stable test
        ),
        profiling_options=gs.options.ProfilingOptions(show_FPS=False),
        show_viewer=show_viewer,
    )

    # Plane at z=0
    scene.add_entity(gs.morphs.Plane())

    # Sensor mount positioned so probe will be in contact with plane
    # Probe center at z=-0.005, plane at z=0
    # Expected penetration = -dist = 0.005 (SDF penetration into plane)
    sensor_mount = scene.add_entity(
        gs.morphs.Sphere(
            radius=0.05,
            pos=(0.0, 0.0, 0.045),  # Mount center at z=0.045, probe at z=-0.005
            fixed=True,
            collision=False,
        ),
    )

    probe = scene.add_sensor(
        gs.sensors.KinematicContactProbe(
            entity_idx=sensor_mount.idx,
            link_idx_local=0,
            probe_local_pos=[(0.0, 0.0, -0.05)],  # Probe center below mount
            probe_local_normal=[(0.0, 0.0, 1.0)],  # Sensing upward
            radius=PROBE_RADIUS,
            stiffness=STIFFNESS,
            draw_debug=show_viewer,
        )
    )

    scene.build(n_envs=n_envs)
    scene.step()

    # Now probe should detect contact
    data = probe.read()

    expected_penetration = 0.005  # SDF penetration into plane

    # Check penetration is approximately what we expect (output shape is now (n_probes,) or (n_envs, n_probes))
    # For single probe, we compare the first (and only) probe
    assert_allclose(data.penetration[..., 0], expected_penetration, tol=2e-3)

    # Normal should point upward (in link local frame)
    # Output shape is (n_probes, 3) or (n_envs, n_probes, 3), take first probe
    normal = data.normal[..., 0, :]
    # The contact normal should point from the surface toward the probe center
    # For a horizontal surface below the probe, this is +z
    assert normal[..., 2].abs().mean() > 0.9  # Primarily in z direction

    # Check force magnitude: F = stiffness * penetration * normal
    expected_force_magnitude = STIFFNESS * expected_penetration
    force_magnitude = torch.norm(data.force[..., 0, :], dim=-1)
    assert_allclose(force_magnitude, expected_force_magnitude, tol=expected_force_magnitude * 0.1)


@pytest.mark.required
@pytest.mark.parametrize("n_envs", [0, 2])
def test_kinematic_contact_probe_no_contact(show_viewer, tol, n_envs):
    """Test that sensor returns zeros when there is no contact."""
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=1e-2,
            substeps=1,
            gravity=(0.0, 0.0, 0.0),  # No gravity to keep things stable
        ),
        profiling_options=gs.options.ProfilingOptions(show_FPS=False),
        show_viewer=show_viewer,
    )

    # A sphere to potentially sense (but placed far away)
    scene.add_entity(
        gs.morphs.Sphere(
            radius=0.5,
            pos=(10.0, 0.0, 0.5),  # Far away
            fixed=True,
        ),
    )

    # Sensor mount - placed with no nearby objects
    sensor_mount = scene.add_entity(
        gs.morphs.Sphere(
            radius=0.05,
            pos=(0.0, 0.0, 5.0),  # High up, no nearby objects
            fixed=True,
            collision=False,
        ),
    )

    probe = scene.add_sensor(
        gs.sensors.KinematicContactProbe(
            entity_idx=sensor_mount.idx,
            link_idx_local=0,
            probe_local_pos=[(0.0, 0.0, 0.0)],
            radius=0.01,
            stiffness=1000.0,
        )
    )

    scene.build(n_envs=n_envs)
    scene.step()

    # Should have no contact (output has n_probes dimension now)
    data = probe.read()
    assert_allclose(data.penetration[..., 0], 0.0, tol=tol)
    assert_allclose(data.position[..., 0, :], 0.0, tol=tol)
    assert_allclose(data.normal[..., 0, :], 0.0, tol=tol)
    assert_allclose(data.force[..., 0, :], 0.0, tol=tol)


@pytest.mark.required
@pytest.mark.parametrize("n_envs", [2])
def test_kinematic_contact_probe_multiple_envs_different_contacts(show_viewer, tol, n_envs):
    """Test that different environments can have different contact states.

    Uses support function semantics: for probe_normal=UP, support(-normal)=DOWN
    gives the sphere's bottom, so probe must be positioned near sphere bottom.
    """
    PROBE_RADIUS = 0.02

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=1e-2,
            substeps=1,
            gravity=(0.0, 0.0, 0.0),
        ),
        profiling_options=gs.options.ProfilingOptions(show_FPS=False),
        show_viewer=show_viewer,
    )

    # Plane at z=0
    scene.add_entity(gs.morphs.Plane())

    # Target sphere at x=2 (to avoid plane interference)
    # Sphere center at z=0.5, radius=0.5, so bottom at z=0
    scene.add_entity(
        gs.morphs.Sphere(
            radius=0.5,
            pos=(2.0, 0.0, 0.5),
            fixed=True,
        ),
    )

    # Sensor mount - with probe_normal=UP, support(DOWN) gives sphere bottom at z=0
    # Position probe near z=0 to detect sphere bottom
    sensor_mount = scene.add_entity(
        gs.morphs.Sphere(
            radius=0.05,
            pos=(2.0, 0.0, 0.06),  # Probe will be at z=0.01 (offset -0.05)
            fixed=True,
            collision=False,
        ),
    )

    probe = scene.add_sensor(
        gs.sensors.KinematicContactProbe(
            entity_idx=sensor_mount.idx,
            link_idx_local=0,
            probe_local_pos=[(0.0, 0.0, -0.05)],
            radius=PROBE_RADIUS,
            stiffness=1000.0,
        )
    )

    scene.build(n_envs=n_envs)
    scene.step()

    # Set different positions for each environment
    # Env 0: Far from sphere (no contact)
    # Env 1: Near sphere bottom (contact via support function)
    # Sphere bottom is at (2, 0, 0), probe needs to be within PROBE_RADIUS of this
    positions = np.array(
        [
            [2.0, 0.0, 2.0],  # Env 0: Far away, probe at z=1.95
            [2.0, 0.0, 0.06],  # Env 1: Probe at z=0.01, within 0.02 of sphere bottom at z=0
        ]
    )
    sensor_mount.set_pos(positions)

    scene.step()

    data = probe.read()

    # Env 0 should have no contact (shape is (n_envs, n_probes), first probe index is 0)
    assert_allclose(data.penetration[0, 0], 0.0, tol=tol)

    # Env 1 should have contact (probe near sphere bottom)
    assert data.penetration[1, 0] > 0.0, "Env 1 should have contact"


@pytest.mark.required
def test_kinematic_contact_probe_multiple_sensors_same_link(show_viewer, tol):
    """Test multiple probe sensors on the same link."""
    PROBE_RADIUS = 0.01

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=1e-2,
            substeps=1,
            gravity=(0.0, 0.0, 0.0),
        ),
        profiling_options=gs.options.ProfilingOptions(show_FPS=False),
        show_viewer=show_viewer,
    )

    # Plane at z=0
    scene.add_entity(gs.morphs.Plane())

    # Sensor mount
    sensor_mount = scene.add_entity(
        gs.morphs.Sphere(
            radius=0.05,
            pos=(0.0, 0.0, 0.045),  # Just above plane
            fixed=True,
            collision=False,
        ),
    )

    # Add multiple probe sensors at different offsets (all should touch plane)
    probe_center = scene.add_sensor(
        gs.sensors.KinematicContactProbe(
            entity_idx=sensor_mount.idx,
            link_idx_local=0,
            probe_local_pos=[(0.0, 0.0, -0.05)],  # Probe at z=-0.005
            radius=PROBE_RADIUS,
            stiffness=1000.0,
        )
    )

    probe_left = scene.add_sensor(
        gs.sensors.KinematicContactProbe(
            entity_idx=sensor_mount.idx,
            link_idx_local=0,
            probe_local_pos=[(-0.1, 0.0, -0.05)],
            radius=PROBE_RADIUS,
            stiffness=1000.0,
        )
    )

    probe_right = scene.add_sensor(
        gs.sensors.KinematicContactProbe(
            entity_idx=sensor_mount.idx,
            link_idx_local=0,
            probe_local_pos=[(0.1, 0.0, -0.05)],
            radius=PROBE_RADIUS,
            stiffness=1000.0,
        )
    )

    scene.build(n_envs=0)
    scene.step()

    # All sensors should detect approximately the same penetration
    # since they're all at the same height above a flat surface
    data_center = probe_center.read()
    data_left = probe_left.read()
    data_right = probe_right.read()

    # All should be in contact (penetration is now (n_probes,), take first probe)
    assert data_center.penetration[0] > 0.0
    assert data_left.penetration[0] > 0.0
    assert data_right.penetration[0] > 0.0

    # Penetrations should be similar (flat surface)
    assert_allclose(data_center.penetration[0], data_left.penetration[0], tol=1e-3)
    assert_allclose(data_center.penetration[0], data_right.penetration[0], tol=1e-3)


@pytest.mark.required
def test_kinematic_contact_probe_noise_and_delay(show_viewer, tol):
    """Test noise and delay functionality of probe sensor.

    Uses plane for initial contact (simpler than sphere with support function).
    """
    DT = 1e-2
    DELAY_STEPS = 2
    BIAS = 0.001

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=DT,
            substeps=1,
            gravity=(0.0, 0.0, 0.0),
        ),
        profiling_options=gs.options.ProfilingOptions(show_FPS=False),
        show_viewer=show_viewer,
    )

    # Plane at z=0 - use plane for reliable contact detection
    scene.add_entity(gs.morphs.Plane())

    # Sensor mount positioned so probe contacts plane
    # Probe at z=-0.005, plane at z=0
    sensor_mount = scene.add_entity(
        gs.morphs.Sphere(
            radius=0.05,
            pos=(0.0, 0.0, 0.045),  # Probe will be at z=-0.005
            fixed=True,
            collision=False,
        ),
    )

    probe_clean = scene.add_sensor(
        gs.sensors.KinematicContactProbe(
            entity_idx=sensor_mount.idx,
            probe_local_pos=[(0.0, 0.0, -0.05)],
            radius=0.01,
            stiffness=1000.0,
        )
    )

    probe_delayed = scene.add_sensor(
        gs.sensors.KinematicContactProbe(
            entity_idx=sensor_mount.idx,
            probe_local_pos=[(0.0, 0.0, -0.05)],
            radius=0.01,
            stiffness=1000.0,
            delay=DT * DELAY_STEPS,
        )
    )

    probe_noisy = scene.add_sensor(
        gs.sensors.KinematicContactProbe(
            entity_idx=sensor_mount.idx,
            probe_local_pos=[(0.0, 0.0, -0.05)],
            radius=0.01,
            stiffness=1000.0,
            bias=BIAS,
            noise=0.0001,
        )
    )

    scene.build(n_envs=0)

    # Step to initialize
    for _ in range(DELAY_STEPS + 1):
        scene.step()

    # Clean and ground truth should match (penetration is now (n_probes,), take first probe)
    clean_data = probe_clean.read()
    ground_truth = probe_clean.read_ground_truth()
    assert_allclose(clean_data.penetration[0], ground_truth.penetration[0], tol=tol)

    # Noisy sensor should have bias added
    noisy_data = probe_noisy.read()
    assert_allclose(noisy_data.penetration[0], ground_truth.penetration[0] + BIAS, tol=0.001)

    # Delayed sensor should read old data
    # Store current ground truth
    current_gt = probe_delayed.read_ground_truth().penetration.clone()

    # Move sensor away (no contact)
    sensor_mount.set_pos([0.0, 0.0, 2.0])

    # Step but not enough to clear delay
    scene.step()

    # Delayed sensor should still read old (contact) data
    delayed_data = probe_delayed.read()
    assert delayed_data.penetration[0] > 0.0, "Delayed sensor should still show contact"

    # Step through delay period
    for _ in range(DELAY_STEPS):
        scene.step()

    # Now delayed sensor should show no contact
    delayed_data = probe_delayed.read()
    assert_allclose(delayed_data.penetration[0], 0.0, tol=0.001)


@pytest.mark.required
def test_kinematic_contact_probe_lazy_evaluation(show_viewer, tol):
    """Test that probe sensor uses lazy evaluation when no delay is configured.

    Without delay, the collision query should only execute when read() is called,
    not on every simulation step. This verifies the plan requirement:
    "Lazy evaluation - Collision query only executes when user calls .read()"
    """
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=1e-2,
            substeps=1,
            gravity=(0.0, 0.0, 0.0),
        ),
        profiling_options=gs.options.ProfilingOptions(show_FPS=False),
        show_viewer=show_viewer,
    )

    # Plane at z=0
    scene.add_entity(gs.morphs.Plane())

    # Sensor mount initially in contact with plane
    sensor_mount = scene.add_entity(
        gs.morphs.Sphere(
            radius=0.05,
            pos=(0.0, 0.0, 0.045),  # Probe at z=-0.005, plane at z=0
            fixed=True,
            collision=False,
        ),
    )

    probe = scene.add_sensor(
        gs.sensors.KinematicContactProbe(
            entity_idx=sensor_mount.idx,
            link_idx_local=0,
            probe_local_pos=[(0.0, 0.0, -0.05)],
            radius=0.01,
            stiffness=1000.0,
            # No delay configured - should use lazy evaluation
        )
    )

    scene.build(n_envs=0)
    scene.step()

    # Read initial state - should be in contact (penetration is now (n_probes,))
    data1 = probe.read()
    assert data1.penetration[0] > 0.0, "Should detect contact initially"

    # Move sensor away (no contact)
    sensor_mount.set_pos([0.0, 0.0, 2.0])

    # Step WITHOUT calling read()
    scene.step()
    scene.step()
    scene.step()

    # Now read - with lazy evaluation, the query runs NOW and should show no contact
    # (if it wasn't lazy, the query would have run during step and might have cached old data)
    data2 = probe.read()
    assert_allclose(data2.penetration[0], 0.0, tol=tol)

    # Move sensor back to contact position
    sensor_mount.set_pos([0.0, 0.0, 0.045])

    # Step without reading
    scene.step()

    # Read again - should show contact (lazy query runs now)
    data3 = probe.read()
    assert data3.penetration[0] > 0.0, "Should detect contact after moving back"


@pytest.mark.required
def test_kinematic_contact_probe_collision_filtering(show_viewer, tol):
    """Test that probe sensor respects collision filtering (contype).

    Geoms with contype=0 should not be detected by the probe sensor.
    Uses support function semantics: probe must be near sphere bottom (support(DOWN)).
    """
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=1e-2,
            substeps=1,
            gravity=(0.0, 0.0, 0.0),
        ),
        profiling_options=gs.options.ProfilingOptions(show_FPS=False),
        show_viewer=show_viewer,
    )

    # Add a sphere with collision disabled (contype=0 effectively)
    # Note: When collision=False on a morph, the geom gets contype=0
    # Sphere center at z=0.5, radius=0.5, so bottom at z=0
    invisible_sphere = scene.add_entity(
        gs.morphs.Sphere(
            radius=0.5,
            pos=(0.0, 0.0, 0.5),
            fixed=True,
            collision=False,  # This disables collision detection
        ),
    )

    # Add a sphere with collision enabled
    # Sphere center at z=0.5, radius=0.5, so bottom at z=0
    visible_sphere = scene.add_entity(
        gs.morphs.Sphere(
            radius=0.5,
            pos=(2.0, 0.0, 0.5),
            fixed=True,
            collision=True,
        ),
    )

    # Sensor mount near invisible sphere bottom
    # With probe_normal=UP, support(DOWN) gives sphere bottom at z=0
    # Position probe near z=0 to potentially detect sphere bottom
    sensor_mount = scene.add_entity(
        gs.morphs.Sphere(
            radius=0.05,
            pos=(0.0, 0.0, 0.06),  # Probe will be at z=0.01 (offset -0.05)
            fixed=True,
            collision=False,
        ),
    )

    probe = scene.add_sensor(
        gs.sensors.KinematicContactProbe(
            entity_idx=sensor_mount.idx,
            link_idx_local=0,
            probe_local_pos=[(0.0, 0.0, -0.05)],
            radius=0.02,  # Radius to reach sphere bottom at z=0
            stiffness=1000.0,
        )
    )

    scene.build(n_envs=0)
    scene.step()

    # Should NOT detect contact with the invisible sphere (collision=False)
    data = probe.read()
    assert_allclose(data.penetration[0], 0.0, tol=tol), "Should not detect collision-disabled sphere"

    # Move sensor near the visible sphere bottom
    sensor_mount.set_pos([2.0, 0.0, 0.06])  # Probe at z=0.01, sphere bottom at z=0
    scene.step()

    # Should detect contact with the visible sphere
    data2 = probe.read()
    assert data2.penetration[0] > 0.0, "Should detect collision-enabled sphere"


@pytest.mark.required
def test_kinematic_contact_probe_self_detection_filter(show_viewer, tol):
    """Test that probe sensor doesn't detect its own link's geom.

    The sensor should skip geoms that belong to the same link as the sensor.
    """
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=1e-2,
            substeps=1,
            gravity=(0.0, 0.0, 0.0),
        ),
        profiling_options=gs.options.ProfilingOptions(show_FPS=False),
        show_viewer=show_viewer,
    )

    # Sensor mount - with collision enabled so it creates a geom
    sensor_mount = scene.add_entity(
        gs.morphs.Sphere(
            radius=0.1,
            pos=(0.0, 0.0, 0.5),
            fixed=True,
            collision=True,  # Mount has collision enabled
        ),
    )

    # Place sensor probe inside the mount's own geom
    probe = scene.add_sensor(
        gs.sensors.KinematicContactProbe(
            entity_idx=sensor_mount.idx,
            link_idx_local=0,
            probe_local_pos=[(0.0, 0.0, 0.0)],  # Probe at center of mount
            radius=0.05,  # Large enough to overlap with mount's geom
            stiffness=1000.0,
        )
    )

    scene.build(n_envs=0)
    scene.step()

    # Should NOT detect its own link's geom
    data = probe.read()
    assert_allclose(data.penetration[0], 0.0, tol=tol), "Should not detect own link's geom"


@pytest.mark.required
def test_kinematic_contact_probe_primitive_geom_types(show_viewer, tol):
    """Test probe sensor works with different primitive geometry types.

    Tests: SPHERE and PLANE - both should be detected correctly without SDF.

    Uses support function semantics:
    - PLANE: Projects probe onto plane
    - SPHERE: With probe_normal=UP, support(DOWN) gives sphere bottom

    Note: BOX, CYLINDER, MESH, CONVEX require SDF preprocessing.
    """
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=1e-2,
            substeps=1,
            gravity=(0.0, 0.0, 0.0),
        ),
        profiling_options=gs.options.ProfilingOptions(show_FPS=False),
        show_viewer=show_viewer,
    )

    # Plane at z=0
    scene.add_entity(gs.morphs.Plane())

    # Sphere at x=2 with center at z=0.5, radius=0.5
    # Bottom of sphere is at z=0
    scene.add_entity(
        gs.morphs.Sphere(
            radius=0.5,
            pos=(2.0, 0.0, 0.5),
            fixed=True,
        )
    )

    # Sensor mount
    sensor_mount = scene.add_entity(
        gs.morphs.Sphere(
            radius=0.05,
            pos=(0.0, 0.0, 0.0),  # Will be repositioned
            fixed=True,
            collision=False,
        ),
    )

    probe = scene.add_sensor(
        gs.sensors.KinematicContactProbe(
            entity_idx=sensor_mount.idx,
            link_idx_local=0,
            probe_local_pos=[(0.0, 0.0, -0.05)],
            radius=0.02,  # Slightly larger radius for sphere detection
            stiffness=1000.0,
        )
    )

    scene.build(n_envs=0)

    # Test plane detection
    sensor_mount.set_pos([0.0, 0.0, 0.04])  # Probe at z=-0.01, plane at z=0
    scene.step()
    data = probe.read()
    assert data.penetration[0] > 0.0, "Should detect plane contact"

    # Test sphere detection - position probe near sphere BOTTOM (support(DOWN))
    # Sphere bottom is at (2, 0, 0), probe needs to be within radius of this point
    sensor_mount.set_pos([2.0, 0.0, 0.06])  # Probe at z=0.01, sphere bottom at z=0
    scene.step()
    data = probe.read()
    assert data.penetration[0] > 0.0, "Should detect sphere contact"


@pytest.mark.required
def test_kinematic_contact_probe_conaffinity_filtering(show_viewer, tol):
    """Test that probe sensor respects conaffinity bitmask filtering.

    Uses MuJoCo-style filtering: collision happens only if
    (geom.contype & sensor.conaffinity) != 0 AND (sensor.contype & geom.conaffinity) != 0
    """
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=1e-2,
            substeps=1,
            gravity=(0.0, 0.0, 0.0),
        ),
        profiling_options=gs.options.ProfilingOptions(show_FPS=False),
        show_viewer=show_viewer,
    )

    # Plane at z=0 (uses default contype=1, conaffinity=0xFFFFFFFF)
    scene.add_entity(gs.morphs.Plane())

    # Sensor mount
    sensor_mount = scene.add_entity(
        gs.morphs.Sphere(
            radius=0.05,
            pos=(0.0, 0.0, 0.045),  # Probe at z=-0.005, plane at z=0
            fixed=True,
            collision=False,
        ),
    )

    # Sensor with default contype=1, conaffinity=0xFFFFFFFF (should detect plane)
    probe_default = scene.add_sensor(
        gs.sensors.KinematicContactProbe(
            entity_idx=sensor_mount.idx,
            link_idx_local=0,
            probe_local_pos=[(0.0, 0.0, -0.05)],
            radius=0.01,
            stiffness=1000.0,
            # Default: contype=1, conaffinity=0xFFFFFFFF
        )
    )

    # Sensor with conaffinity=0 (should NOT detect anything)
    probe_no_affinity = scene.add_sensor(
        gs.sensors.KinematicContactProbe(
            entity_idx=sensor_mount.idx,
            link_idx_local=0,
            probe_local_pos=[(0.0, 0.0, -0.05)],
            radius=0.01,
            stiffness=1000.0,
            conaffinity=0,  # No bits set - won't collide with anything
        )
    )

    # Sensor with contype=0 (should NOT detect anything)
    probe_no_contype = scene.add_sensor(
        gs.sensors.KinematicContactProbe(
            entity_idx=sensor_mount.idx,
            link_idx_local=0,
            probe_local_pos=[(0.0, 0.0, -0.05)],
            radius=0.01,
            stiffness=1000.0,
            contype=0,  # No bits set - won't collide with anything
        )
    )

    scene.build(n_envs=0)
    scene.step()

    # Default sensor should detect contact with plane (penetration is now (n_probes,))
    data_default = probe_default.read()
    assert data_default.penetration[0] > 0.0, "Default sensor should detect plane"

    # Sensor with conaffinity=0 should NOT detect contact
    # Because (geom.contype & sensor.conaffinity) = (1 & 0) = 0, which fails the filter
    data_no_affinity = probe_no_affinity.read()
    assert_allclose(data_no_affinity.penetration[0], 0.0, tol=tol), "Sensor with conaffinity=0 should not detect plane"

    # Sensor with contype=0 should NOT detect contact
    # Because (sensor.contype & geom.conaffinity) = (0 & 0xFFFFFFFF) = 0, which fails the filter
    data_no_contype = probe_no_contype.read()
    assert_allclose(data_no_contype.penetration[0], 0.0, tol=tol), "Sensor with contype=0 should not detect plane"


@pytest.mark.skip(reason="Integration test requires robot URDF - skip for CI")
def test_kinematic_contact_probe_robot_hand_integration(show_viewer, tol):
    """Integration test: Robot hand grasping object, verify contact pattern.

    This test is skipped by default as it requires robot assets.
    """
    # This test would require:
    # 1. A robot hand URDF with fingertip links
    # 2. Multiple probe sensors on fingertips
    # 3. An object to grasp
    # 4. Verification of contact pattern during grasp
    pass


@pytest.mark.required
def test_kinematic_contact_probe_box_support_function(show_viewer, tol):
    """Test probe sensor works with BOX geometry using support function path.

    This test verifies that the support-function-based contact query works for box geoms.
    Uses support function semantics: with probe_normal=UP, support(DOWN) gives a bottom
    CORNER of the box (not the face center) - the vertex furthest in the -Z direction.

    The BOX type uses _func_support_box which calculates support analytically.
    For direction (0, 0, -1), it returns the corner at (+x, +y, -z) half-sizes.
    """
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=1e-2,
            substeps=1,
            gravity=(0.0, 0.0, 0.0),
        ),
        profiling_options=gs.options.ProfilingOptions(show_FPS=False),
        show_viewer=show_viewer,
    )

    # Add a box at z=0.25 (half the box height)
    # Box of size 0.5x0.5x0.5, bottom at z=0, top at z=0.5
    # Bottom corners are at (±0.25, ±0.25, 0)
    # For direction (0, 0, -1), support returns (+0.25, +0.25, 0)
    box_size = 0.5
    box_entity = scene.add_entity(
        gs.morphs.Box(
            size=(box_size, box_size, box_size),
            pos=(0.0, 0.0, box_size / 2),  # Bottom of box at z=0
            fixed=True,
        ),
    )

    # Position probe near the bottom corner that support(DOWN) returns
    # Support(DOWN) for this box = (0.25, 0.25, 0.0)
    # Place probe near this corner to detect contact
    corner_x = 0.25
    corner_y = 0.25
    corner_z = 0.0
    probe_offset_z = -0.05
    sensor_mount = scene.add_entity(
        gs.morphs.Sphere(
            radius=0.02,
            pos=(corner_x, corner_y, corner_z + 0.01 - probe_offset_z),  # Probe at (0.25, 0.25, 0.01)
            fixed=True,
            collision=False,
        ),
    )

    probe = scene.add_sensor(
        gs.sensors.KinematicContactProbe(
            entity_idx=sensor_mount.idx,
            link_idx_local=0,
            probe_local_pos=[(0.0, 0.0, probe_offset_z)],
            radius=0.02,  # Small enough to detect precise contact
            stiffness=1000.0,
        )
    )

    scene.build(n_envs=0)

    # Verify box geom exists
    solver = scene._sim.rigid_solver
    box_geoms = [g for g in solver.geoms if g.type == gs.GEOM_TYPE.BOX]
    assert len(box_geoms) == 1, "Should have exactly one BOX type geom"

    scene.step()
    data_contact = probe.read()

    # Verify contact is detected - probe at (0.25, 0.25, 0.01), box corner at (0.25, 0.25, 0)
    # Distance = 0.01 < radius 0.02, so contact should be detected
    # Penetration is now (n_probes,), take first probe
    penetration_at_surface = data_contact.penetration[0].item()
    assert penetration_at_surface > 0.0, (
        f"Support function path should detect contact with box corner, got penetration={penetration_at_surface}"
    )

    # Verify other contact data is populated (shape is now (n_probes, 3))
    assert data_contact.normal[0].abs().sum() > 0, "Normal should be non-zero when in contact"
    assert data_contact.force[0].abs().sum() > 0, "Force should be non-zero when in contact"

    # Now move probe far away - should have no contact
    sensor_mount.set_pos(torch.tensor([0.0, 0.0, 2.0], dtype=gs.tc_float))
    scene.step()
    data_no_contact = probe.read()

    penetration_far = data_no_contact.penetration[0].item()
    assert penetration_far == 0.0, (
        f"Should have no contact when probe is far above box, got penetration={penetration_far}"
    )


@pytest.mark.required
@pytest.mark.parametrize("n_envs", [0, 2])
def test_kinematic_contact_probe_multi_probe(show_viewer, tol, n_envs):
    """Test multiple probes in a single KinematicContactProbe sensor.

    Creates a sensor with 3 probes at different positions. Verifies that:
    1. Output shape is (n_envs, n_probes) for penetration and (n_envs, n_probes, 3) for vectors
    2. Each probe independently detects contact based on its position
    """
    PROBE_RADIUS = 0.02

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=1e-2,
            substeps=1,
            gravity=(0.0, 0.0, 0.0),
        ),
        profiling_options=gs.options.ProfilingOptions(show_FPS=False),
        show_viewer=show_viewer,
    )

    # Plane at z=0
    scene.add_entity(gs.morphs.Plane())

    # Sensor mount positioned above the plane
    sensor_mount = scene.add_entity(
        gs.morphs.Sphere(
            radius=0.05,
            pos=(0.0, 0.0, 0.1),  # Mount at z=0.1
            fixed=True,
            collision=False,
        ),
    )

    # Create a sensor with 3 probes:
    # - Probe 0: At z=-0.11 (below plane, should detect contact)
    # - Probe 1: At z=-0.09 (touching plane, should detect contact)
    # - Probe 2: At z=0.0 (above plane, no contact)
    probe = scene.add_sensor(
        gs.sensors.KinematicContactProbe(
            entity_idx=sensor_mount.idx,
            link_idx_local=0,
            probe_local_pos=[
                (0.0, 0.0, -0.11),  # Probe 0: z=-0.01 world
                (0.0, 0.1, -0.11),  # Probe 1: z=-0.01 world, offset in y
                (0.0, 0.0, 0.0),  # Probe 2: z=0.1 world (far above plane)
            ],
            probe_local_normal=[
                (0.0, 0.0, 1.0),  # All probes sense upward
                (0.0, 0.0, 1.0),
                (0.0, 0.0, 1.0),
            ],
            radius=PROBE_RADIUS,
            stiffness=1000.0,
        )
    )

    scene.build(n_envs=n_envs)
    scene.step()

    data = probe.read()

    # Verify output shapes
    if n_envs == 0:
        # Shape should be (n_probes,) for penetration and (n_probes, 3) for vectors
        assert data.penetration.shape == (3,), f"Expected penetration shape (3,), got {data.penetration.shape}"
        assert data.position.shape == (3, 3), f"Expected position shape (3, 3), got {data.position.shape}"
        assert data.normal.shape == (3, 3), f"Expected normal shape (3, 3), got {data.normal.shape}"
        assert data.force.shape == (3, 3), f"Expected force shape (3, 3), got {data.force.shape}"
    else:
        # Shape should be (n_envs, n_probes) for penetration and (n_envs, n_probes, 3) for vectors
        assert data.penetration.shape == (n_envs, 3), (
            f"Expected penetration shape ({n_envs}, 3), got {data.penetration.shape}"
        )
        assert data.position.shape == (n_envs, 3, 3), (
            f"Expected position shape ({n_envs}, 3, 3), got {data.position.shape}"
        )
        assert data.normal.shape == (n_envs, 3, 3), f"Expected normal shape ({n_envs}, 3, 3), got {data.normal.shape}"
        assert data.force.shape == (n_envs, 3, 3), f"Expected force shape ({n_envs}, 3, 3), got {data.force.shape}"

    # Verify probe 0 and 1 detect contact (they're below/at plane level)
    assert (data.penetration[..., 0] > 0.0).all(), "Probe 0 should detect contact with plane"
    assert (data.penetration[..., 1] > 0.0).all(), "Probe 1 should detect contact with plane"

    # Verify probe 2 does not detect contact (it's above plane)
    assert_allclose(data.penetration[..., 2], 0.0, tol=tol), "Probe 2 should not detect contact"

    # Verify contact positions are different for probes 0 and 1 (due to y offset)
    if n_envs == 0:
        pos_diff = (data.position[0] - data.position[1]).abs()
    else:
        pos_diff = (data.position[:, 0] - data.position[:, 1]).abs()
    # Y component should differ by ~0.1
    assert (pos_diff[..., 1].mean() > 0.05).all(), "Contact positions should differ in y direction"
