import importlib
import sys
import tempfile
import textwrap
from pathlib import Path

import numpy as np
import pytest
import torch

import genesis as gs
import genesis.utils.geom as gu
from genesis.utils.misc import tensor_to_array

from .utils import assert_allclose, assert_equal

# ------------------------------------------------------------------------------------------
# -------------------------------- Lazy Sensor Discovery -----------------------------------
# ------------------------------------------------------------------------------------------


@pytest.mark.required
def test_lazy_sensor_discovery(show_viewer, tmp_path):
    """Test that add_sensor auto-discovers sensor classes from the options class's sibling modules."""
    from genesis.engine.sensors.camera import RasterizerCameraSensor
    from genesis.engine.sensors.contact_force import ContactSensor
    from genesis.engine.sensors.depth_camera import DepthCameraSensor
    from genesis.engine.sensors.imu import IMUSensor
    from genesis.engine.sensors.sensor_manager import SensorManager

    # Verify built-in registrations resolve to the exact sensor classes
    assert SensorManager.SENSOR_TYPES_MAP[gs.sensors.Contact] is ContactSensor
    assert SensorManager.SENSOR_TYPES_MAP[gs.sensors.IMU] is IMUSensor
    assert SensorManager.SENSOR_TYPES_MAP[gs.sensors.RasterizerCameraOptions] is RasterizerCameraSensor
    # DepthCamera inherits from Raycaster without re-parameterizing, registered only by sensor side
    assert SensorManager.SENSOR_TYPES_MAP[gs.sensors.DepthCamera] is DepthCameraSensor

    # Create a fake plugin package in a temp directory
    pkg_dir = tmp_path / "fake_sensor_plugin"
    pkg_dir.mkdir()

    (pkg_dir / "__init__.py").write_text("")

    (pkg_dir / "options.py").write_text(
        textwrap.dedent(
            """\
        from genesis.options.sensors.options import SensorOptions

        class FakeSensorOptions(SensorOptions["FakeSensor"]):
            pass
        """
        )
    )

    (pkg_dir / "sensor.py").write_text(
        textwrap.dedent(
            """\
        from dataclasses import dataclass

        import genesis as gs
        import torch
        from genesis.engine.sensors.base_sensor import Sensor, SharedSensorMetadata

        from .options import FakeSensorOptions


        @dataclass
        class FakeSensorMetadata(SharedSensorMetadata):
            pass


        class FakeSensor(Sensor[FakeSensorOptions, FakeSensorMetadata]):
            def _get_return_format(self):
                return (1,)

            @classmethod
            def _get_cache_dtype(cls):
                return gs.tc_float

            @classmethod
            def _update_shared_cache(
                cls, metadata, gt_cache, ground_truth_data_timeline, measured_data_timeline, intermediate_cache,
            ):
                pass

            @classmethod
            def reset(cls, metadata, shared_ground_truth_cache, envs_idx):
                pass

            def build(self):
                pass
        """
        )
    )

    sys.path.insert(0, str(tmp_path))
    try:
        # Import ONLY the options module (not the sensor module)
        options_mod = importlib.import_module("fake_sensor_plugin.options")
        FakeSensorOptions = options_mod.FakeSensorOptions

        # Verify it's not yet registered
        assert FakeSensorOptions not in SensorManager.SENSOR_TYPES_MAP

        # Trigger lazy discovery via resolve
        sensor_cls = SensorManager._resolve_sensor_cls(FakeSensorOptions)
        assert sensor_cls.__name__ == "FakeSensor"

        # Now it should be registered
        assert SensorManager.SENSOR_TYPES_MAP[FakeSensorOptions] is sensor_cls

        # Verify it works end-to-end with a scene
        scene = gs.Scene(show_viewer=show_viewer)
        scene.add_entity(gs.morphs.Plane())
        sensor = scene.add_sensor(FakeSensorOptions())
        scene.build()
        scene.step()
        data = sensor.read()
        assert data.shape[-1] == 1
    finally:
        sys.path.remove(str(tmp_path))
        for mod_name in list(sys.modules):
            if mod_name.startswith("fake_sensor_plugin"):
                del sys.modules[mod_name]
        SensorManager.SENSOR_TYPES_MAP.pop(FakeSensorOptions, None)


@pytest.mark.required
def test_post_process_requires_intermediate_override():
    # Strict-override rule: a subclass overriding `_post_process` without also overriding `_get_intermediate_format`
    # or `_get_intermediate_dtype` must raise TypeError at class-definition time. The intermediate buffer is
    # structurally distinct from the return buffer (timeline ring is in intermediate space); the explicit override
    # forces the author to declare it - even a no-op override is acceptable when shape/dtype coincide with return.
    # Local import: importing `genesis.engine.sensors.base_sensor` at module top triggers the sensors package
    # `__init__.py`, which transitively loads `genesis.utils.sdf` and dereferences `gs.qd_float`. That attribute is
    # only set by `gs.init(...)`, which runs in the autouse conftest fixture after pytest collection. Defer here.
    from genesis.engine.sensors.base_sensor import Sensor

    with pytest.raises(TypeError, match="_get_intermediate"):

        class BadSensor(Sensor):
            def _get_return_format(self):
                return (1,)

            @classmethod
            def _get_cache_dtype(cls):
                return gs.tc_float

            @classmethod
            def _post_process(cls, shared_metadata, tensor):
                return tensor * 2


@pytest.mark.required
def test_pipeline_contract(tol):
    # Two synthetic sensor families share a single scene/build:
    #   * `FakePipelineSensor` is a vector sensor whose components each take a different
    #     (physics_imp, measured_only_imp, transform_alpha, hardware_imp) path, so GT-cleanliness, physics
    #     propagation through transform recurrence, the `is_measured` gate on `_apply_transform`, and HW non-
    #     compounding are all verified in one batched pass.
    #   * `FakeSimpleSensor` instances cover the three return-space ring allocation paths: no-ring (delay=0,
    #     history=0), history-only ring, and delay+history ring. All four instances of `FakeSimpleSensor` share
    #     the same per-class step counter, so they all see the same raw value at each step and the expected
    #     outputs are simple shifts / windows of that sequence.
    from dataclasses import dataclass

    from genesis.engine.sensors.base_sensor import SimpleSensor, SimpleSensorMetadata
    from genesis.options.sensors.options import SimpleSensorOptions

    @dataclass
    class FakeMetadata(SimpleSensorMetadata):
        # Per-component knob vectors, shape `(1, vec_size)` so they broadcast over the batch dim of slot 0.
        step_counter: int = 0
        physics_imp: torch.Tensor = None
        measured_only_imp: torch.Tensor = None
        transform_alpha: torch.Tensor = None
        hardware_imp: torch.Tensor = None

    class FakeOptions(SimpleSensorOptions["FakePipelineSensor"]):
        physics_imp: tuple[float, ...] = (0.0,)
        measured_only_imp: tuple[float, ...] = (0.0,)
        transform_alpha: tuple[float, ...] = (0.0,)
        hardware_imp: tuple[float, ...] = (0.0,)

    class FakePipelineSensor(SimpleSensor[FakeOptions, FakeMetadata]):
        def _get_return_format(self):
            return (len(self._options.physics_imp),)

        @classmethod
        def _get_cache_dtype(cls):
            return gs.tc_float

        def build(self):
            super().build()
            self._shared_metadata.physics_imp = torch.tensor(
                [self._options.physics_imp], device=gs.device, dtype=gs.tc_float
            )
            self._shared_metadata.measured_only_imp = torch.tensor(
                [self._options.measured_only_imp], device=gs.device, dtype=gs.tc_float
            )
            self._shared_metadata.transform_alpha = torch.tensor(
                [self._options.transform_alpha], device=gs.device, dtype=gs.tc_float
            )
            self._shared_metadata.hardware_imp = torch.tensor(
                [self._options.hardware_imp], device=gs.device, dtype=gs.tc_float
            )

        @classmethod
        def reset(cls, shared_metadata, ground_truth_cache, envs_idx):
            super().reset(shared_metadata, ground_truth_cache, envs_idx)
            shared_metadata.step_counter = 0

        @classmethod
        def _update_raw_data(cls, metadata, raw_data_T):
            # Same scalar raw value across all components and envs; per-component divergence is introduced by the
            # downstream hook vectors. 1-indexed step.
            metadata.step_counter += 1
            raw_data_T.fill_(float(metadata.step_counter))

        @classmethod
        def _apply_physics_imperfections(cls, metadata, slot_0, timeline):
            slot_0.add_(metadata.physics_imp)

        @classmethod
        def _apply_transform(cls, metadata, data, timeline, *, is_measured):
            # Measured-only pre-acquisition contribution: exercises the `is_measured` gate.
            if is_measured:
                data.add_(metadata.measured_only_imp)
            # Stateful linear recurrence per component, branch-symmetric. `timeline.at(1)` is the previous step's
            # post-transform value on this branch (clean of hardware noise - the load-bearing invariant under test).
            data.add_(timeline.at(1) * metadata.transform_alpha)

        @classmethod
        def _apply_hardware_imperfections(cls, metadata, working_buf):
            working_buf.add_(metadata.hardware_imp)

    # Each row is one (physics_imp, measured_only_imp, transform_alpha, hardware_imp) tuple. Components are
    # independent.
    paths = [
        (0.0, 0.0, 0.0, 0.0),  # identity pipeline
        (0.0, 0.0, 0.0, 100.0),  # hardware only: GT must stay clean, measured = raw + H
        (0.0, 0.0, 1.0, 0.0),  # stateful transform on both branches
        (0.0, 0.0, 1.0, 100.0),  # stateful transform + large H: HW must NOT compound through recurrence
        (5.0, 0.0, 0.0, 0.0),  # physics imperfection measured-only, no transform
        (0.0, 5.0, 0.0, 0.0),  # measured-only pre-acquisition (transform with is_measured)
        (5.0, 0.0, 1.0, 0.0),  # physics imperfection compounds through transform recurrence
        (5.0, 5.0, 1.0, 100.0),  # all four together
    ]
    P = np.array([row[0] for row in paths], dtype=np.float32)
    M = np.array([row[1] for row in paths], dtype=np.float32)
    A = np.array([row[2] for row in paths], dtype=np.float32)
    H = np.array([row[3] for row in paths], dtype=np.float32)

    # Companion simple sensor for the ring-allocation paths. No knobs, no overrides beyond raw write - the read
    # just echoes the shared per-class step counter. Four instances cover (delay=0, history=0), history-only,
    # delay-only, and (delay + history).
    @dataclass
    class FakeSimpleMetadata(SimpleSensorMetadata):
        step_counter: int = 0

    class FakeSimpleOptions(SimpleSensorOptions["FakeSimpleSensor"]):
        pass

    class FakeSimpleSensor(SimpleSensor[FakeSimpleOptions, FakeSimpleMetadata]):
        def _get_return_format(self):
            return (1,)

        @classmethod
        def _get_cache_dtype(cls):
            return gs.tc_float

        @classmethod
        def reset(cls, shared_metadata, ground_truth_cache, envs_idx):
            super().reset(shared_metadata, ground_truth_cache, envs_idx)
            shared_metadata.step_counter = 0

        @classmethod
        def _update_raw_data(cls, metadata, raw_data_T):
            metadata.step_counter += 1
            raw_data_T.fill_(float(metadata.step_counter))

    DT = 1e-2
    DELAY_STEPS = 2
    HISTORY_LEN = 3
    scene = gs.Scene(sim_options=gs.options.SimOptions(dt=DT), show_viewer=False)
    scene.add_entity(gs.morphs.Plane())  # minimum scene; the sensors do not depend on any physics.
    sensor = scene.add_sensor(
        FakeOptions(
            physics_imp=tuple(P.tolist()),
            measured_only_imp=tuple(M.tolist()),
            transform_alpha=tuple(A.tolist()),
            hardware_imp=tuple(H.tolist()),
        )
    )
    s_baseline = scene.add_sensor(FakeSimpleOptions())
    s_history = scene.add_sensor(FakeSimpleOptions(history_length=HISTORY_LEN))
    s_delay = scene.add_sensor(FakeSimpleOptions(delay=DELAY_STEPS * DT))
    s_both = scene.add_sensor(FakeSimpleOptions(history_length=HISTORY_LEN, delay=DELAY_STEPS * DT))
    scene.build()
    scene.reset()  # zero the build-warmup counter increment so step 1 sees raw = 1.

    n_steps = 8
    gt_observed = np.zeros((n_steps, len(paths)), dtype=np.float32)
    measured_observed = np.zeros((n_steps, len(paths)), dtype=np.float32)
    baseline_observed = np.zeros(n_steps, dtype=np.float32)
    history_observed = np.zeros((n_steps, HISTORY_LEN), dtype=np.float32)
    delay_observed = np.zeros(n_steps, dtype=np.float32)
    both_observed = np.zeros((n_steps, HISTORY_LEN), dtype=np.float32)
    for i in range(n_steps):
        scene.step()
        gt_observed[i] = tensor_to_array(sensor.read_ground_truth()).reshape(-1)
        measured_observed[i] = tensor_to_array(sensor.read()).reshape(-1)
        baseline_observed[i] = tensor_to_array(s_baseline.read()).item()
        history_observed[i] = tensor_to_array(s_history.read()).reshape(-1)
        delay_observed[i] = tensor_to_array(s_delay.read()).item()
        both_observed[i] = tensor_to_array(s_both.read()).reshape(-1)

    # Analytical expectation for the vector sensor, per component. Let raw[k] = k, and (P, M, A, H) be the per-
    # component vectors.
    # GT ring:    gt[k]  = k + A * gt[k-1]                       (raw -> transform with is_measured=False)
    # Meas ring:  m[k]   = (k + P + M) + A * m[k-1]              (raw -> physics_imp -> transform is_measured=True)
    # Measured:   meas[k] = m[k] + H                             (working buffer adds H; no compounding into m)
    gt_expected = np.zeros_like(gt_observed)
    measured_expected = np.zeros_like(measured_observed)
    gt_prev = np.zeros(len(paths), dtype=np.float32)
    m_prev = np.zeros(len(paths), dtype=np.float32)
    for k in range(1, n_steps + 1):
        gt_k = k + A * gt_prev
        m_k = (k + P + M) + A * m_prev
        gt_expected[k - 1] = gt_k
        measured_expected[k - 1] = m_k + H
        gt_prev, m_prev = gt_k, m_k

    assert_allclose(gt_observed, gt_expected, tol=tol)
    assert_allclose(measured_observed, measured_expected, tol=tol)

    # Ring-allocation paths. raw[k] = k for every FakeSimpleSensor instance (shared step counter); delayed reads
    # before slot D has been filled return zero (ring initialized to zero on reset). History reads source slots
    # `at(0..H-1)` of the return-space ring directly - i.e. the last H post-`_post_process` snapshots - without
    # additional delay shift. A sensor that configures both `delay > 0` and `history_length > 0` therefore sees
    # undelayed history alongside a delayed non-history read; this matches the implementation and is what the
    # combined test asserts.
    raw = np.arange(1, n_steps + 1, dtype=np.float32)
    expected_baseline = raw
    expected_delay = np.where(raw - DELAY_STEPS >= 1, raw - DELAY_STEPS, 0.0)
    expected_history = np.zeros((n_steps, HISTORY_LEN), dtype=np.float32)
    for k in range(1, n_steps + 1):
        for h in range(HISTORY_LEN):
            past_step = k - h
            expected_history[k - 1, h] = past_step if past_step >= 1 else 0.0

    assert_allclose(baseline_observed, expected_baseline, tol=tol)
    assert_allclose(delay_observed, expected_delay, tol=tol)
    assert_allclose(history_observed, expected_history, tol=tol)
    # The combined delay + history sensor returns the same history as the history-only sensor (delay is bypassed
    # by the ring-gather history path); verify they match.
    assert_allclose(both_observed, expected_history, tol=tol)


@pytest.mark.required
def test_pipeline_contract_uint8_delay(tol):
    # ZOH delay sampling must work on non-float return dtypes. A sensor whose `_post_process` casts a float
    # intermediate to a `uint8` return stores `uint8` snapshots in the per-class return-space ring; delay
    # sampling reads those slots verbatim (the dtype-safe ZOH default). Verifies the slot is correctly typed
    # and the delayed values match the cast of `raw[k - delay]`.
    from dataclasses import dataclass

    from genesis.engine.sensors.base_sensor import SimpleSensor, SimpleSensorMetadata
    from genesis.options.sensors.options import SimpleSensorOptions

    @dataclass
    class FakeQuantizedMetadata(SimpleSensorMetadata):
        step_counter: int = 0

    class FakeQuantizedOptions(SimpleSensorOptions["FakeQuantizedSensor"]):
        pass

    class FakeQuantizedSensor(SimpleSensor[FakeQuantizedOptions, FakeQuantizedMetadata]):
        def _get_return_format(self):
            return (1,)

        @classmethod
        def _get_cache_dtype(cls):
            return torch.uint8

        @classmethod
        def _get_intermediate_dtype(cls):
            return gs.tc_float

        @classmethod
        def reset(cls, shared_metadata, ground_truth_cache, envs_idx):
            super().reset(shared_metadata, ground_truth_cache, envs_idx)
            shared_metadata.step_counter = 0

        @classmethod
        def _update_raw_data(cls, metadata, raw_data_T):
            metadata.step_counter += 1
            raw_data_T.fill_(float(metadata.step_counter))

        @classmethod
        def _post_process(cls, shared_metadata, tensor, timeline, *, is_measured):
            return tensor.clamp(0, 255).to(torch.uint8)

    DT = 1e-2
    DELAY_STEPS = 2
    scene = gs.Scene(sim_options=gs.options.SimOptions(dt=DT), show_viewer=False)
    scene.add_entity(gs.morphs.Plane())
    sensor = scene.add_sensor(FakeQuantizedOptions(delay=DELAY_STEPS * DT))
    scene.build()
    scene.reset()

    n_steps = 8
    observed = np.zeros(n_steps, dtype=np.uint8)
    for i in range(n_steps):
        scene.step()
        observed[i] = tensor_to_array(sensor.read()).item()

    raw = np.arange(1, n_steps + 1, dtype=np.float32)
    expected = np.where(raw - DELAY_STEPS >= 1, raw - DELAY_STEPS, 0.0).astype(np.uint8)
    assert observed.dtype == np.uint8
    assert_equal(observed, expected)


@pytest.mark.required
def test_add_and_read_all_registered_sensors():
    """Add all sensors into scene and read them, verifying SensorManager cache and tensor contiguity"""
    from genesis.engine.sensors.sensor_manager import SensorManager

    scene = gs.Scene(
        show_viewer=False,
    )
    scene.add_entity(gs.morphs.Plane())
    box = scene.add_entity(
        gs.morphs.Box(
            size=(0.2, 0.2, 0.2),
            pos=(0.0, 0.0, 0.1),
        )
    )
    sphere = scene.add_entity(
        gs.morphs.Sphere(
            radius=0.1,
            pos=(0.2, 0.0, 0.1),
        )
    )

    sensors = []

    for option_cls in SensorManager.SENSOR_TYPES_MAP.keys():
        sensor_kwargs = {}
        if issubclass(option_cls, gs.sensors.BaseCameraOptions):
            continue  # skip camera options
        if issubclass(option_cls, gs.sensors.RigidSensorOptionsMixin):
            sensor_kwargs.update(
                entity_idx=box.idx,
            )
        if issubclass(option_cls, gs.sensors.Raycaster):
            sensor_kwargs.update(
                pattern=gs.sensors.raycaster.DepthCameraPattern(),
            )
        if issubclass(
            option_cls,
            (gs.sensors.SurfaceDistanceProbe, gs.sensors.ProximityTaxel, gs.sensors.ElastomerTaxel),
        ):
            sensor_kwargs.update(
                track_link_idx=(sphere.base_link_idx,),
            )
        if issubclass(option_cls, gs.sensors.TemperatureGrid):
            sensor_kwargs.update(
                properties_dict={
                    -1: gs.sensors.TemperatureProperties(),
                },
            )

        sensor = scene.add_sensor(option_cls(**sensor_kwargs))
        sensors.append(sensor)

    scene.build(n_envs=2)

    scene.step()
    for sensor in sensors:
        sensor.read()


# ------------------------------------------------------------------------------------------
# -------------------------------------- IMU Sensors ---------------------------------------
# ------------------------------------------------------------------------------------------


@pytest.mark.required
@pytest.mark.parametrize("n_envs", [0, 2])
def test_imu_sensor(show_viewer, tol, n_envs):
    """Test if the IMU sensor returns the correct data."""
    GRAVITY = -10.0
    DT = 1e-2
    BIAS = (0.1, 0.2, 0.3)
    DELAY_STEPS = 2
    MAG_FIELD = (0.3, 0.1, 0.5)  # arbitrary world magnetic field

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=DT,
            substeps=1,
            gravity=(0.0, 0.0, GRAVITY),
        ),
        profiling_options=gs.options.ProfilingOptions(
            show_FPS=False,
        ),
        show_viewer=show_viewer,
    )

    scene.add_entity(gs.morphs.Plane())

    box = scene.add_entity(
        morph=gs.morphs.Box(
            size=(0.1, 0.1, 0.1),
            pos=(0.0, 0.0, 0.2),
        ),
    )

    imu = scene.add_sensor(
        gs.sensors.IMU(
            entity_idx=box.idx,
            magnetic_field=MAG_FIELD,
        )
    )
    imu_delayed = scene.add_sensor(
        gs.sensors.IMU(
            entity_idx=box.idx,
            delay=DT * DELAY_STEPS,
            magnetic_field=MAG_FIELD,
        )
    )
    imu_noisy = scene.add_sensor(
        gs.sensors.IMU(
            entity_idx=box.idx,
            acc_cross_axis_coupling=0.01,
            gyro_cross_axis_coupling=(0.02, 0.03, 0.04),
            mag_cross_axis_coupling=0.01,
            acc_noise=(0.01, 0.01, 0.01),
            gyro_noise=(0.01, 0.01, 0.01),
            mag_noise=(0.01, 0.01, 0.01),
            acc_random_walk=(0.001, 0.001, 0.001),
            gyro_random_walk=(0.001, 0.001, 0.001),
            mag_random_walk=(0.001, 0.001, 0.001),
            delay=DT,
            magnetic_field=MAG_FIELD,
            jitter=DT * 0.1,
        )
    )

    scene.build(n_envs=n_envs)

    # box is in freefall
    for _ in range(10):
        scene.step()

    # IMU should calculate "classical linear acceleration" using the local frame without accounting for gravity
    # acc_classical_lin_z = - theta_dot ** 2 - cos(theta) * g
    assert_allclose(imu.read().lin_acc, 0.0, tol=tol)
    assert_allclose(imu.read().ang_vel, 0.0, tol=tol)
    assert_allclose(imu.read().mag, MAG_FIELD, tol=tol)
    assert_allclose(imu_noisy.read().lin_acc, 0.0, tol=1e-1)
    assert_allclose(imu_noisy.read().ang_vel, 0.0, tol=1e-1)
    assert_allclose(imu_noisy.read().mag, MAG_FIELD, tol=1e-1)

    # shift COM to induce angular velocity
    box.set_COM_shift([0.05, 0.05, 0.05])

    # update noise and bias for accelerometer, gyroscope and magnetometer
    imu_noisy.set_noise((0.01, 0.01, 0.01, 0.02, 0.02, 0.02, 0.05, 0.05, 0.05))
    imu_noisy.set_bias((0.01, 0.01, 0.01, 0.02, 0.02, 0.02, 0.05, 0.05, 0.05))
    imu_noisy.set_jitter(0.001)

    for _ in range(10 - DELAY_STEPS):
        scene.step()

    true_imu_delayed_reading = imu_delayed.read_ground_truth()

    for _ in range(DELAY_STEPS):
        scene.step()

    assert_equal(imu_delayed.read().lin_acc, true_imu_delayed_reading.lin_acc)
    assert_equal(imu_delayed.read().ang_vel, true_imu_delayed_reading.ang_vel)
    assert_equal(imu_delayed.read().mag, true_imu_delayed_reading.mag)

    # check that position offset affects linear acceleration
    imu.set_pos_offset((0.5, 0.0, 0.0))
    lin_acc_no_offset = imu.read().lin_acc
    scene.step()
    lin_acc_with_offset = imu.read().lin_acc
    with np.testing.assert_raises(AssertionError):
        assert_allclose(lin_acc_no_offset, lin_acc_with_offset, atol=0.2)
    imu.set_pos_offset((0.0, 0.0, 0.0))

    # let box collide with ground
    for _ in range(20):
        scene.step()

    assert_equal(imu.read_ground_truth().lin_acc, imu_delayed.read_ground_truth().lin_acc)
    assert_equal(imu.read_ground_truth().ang_vel, imu_delayed.read_ground_truth().ang_vel)
    assert_equal(imu.read_ground_truth().mag, imu_delayed.read_ground_truth().mag)

    with np.testing.assert_raises(AssertionError, msg="Angular velocity should not be zero due to COM shift"):
        assert_allclose(imu.read_ground_truth().ang_vel, 0.0, tol=tol)

    with np.testing.assert_raises(AssertionError, msg="Delayed accl data should not be equal to the ground truth data"):
        assert_equal(imu_delayed.read().lin_acc - imu_delayed.read_ground_truth().lin_acc, 0.0)

    with np.testing.assert_raises(AssertionError, msg="Delayed mag data should not be equal to the ground truth data"):
        assert_equal(imu_delayed.read().mag - imu_delayed.read_ground_truth().mag, 0.0)

    box.set_COM_shift((0.0, 0.0, 0.0))
    box.set_quat((0.0, 0.0, 0.0, 1.0))  # pi rotation around z-axis

    # wait for the box to be stationary on ground
    for _ in range(50):
        scene.step()

    assert_allclose(imu.read().lin_acc, (0.0, 0.0, -GRAVITY), tol=5e-6)
    assert_allclose(imu.read().ang_vel, (0.0, 0.0, 0.0), tol=1e-5)
    assert_allclose(imu.read().mag, (-MAG_FIELD[0], -MAG_FIELD[1], MAG_FIELD[2]), tol=tol)

    # rotate IMU 90 deg around x axis means gravity should be along -y axis
    imu.set_quat_offset(gu.euler_to_quat((90.0, 0.0, 0.0)))
    scene.step()
    assert_allclose(imu.read().lin_acc, (0.0, GRAVITY, 0.0), tol=5e-6)
    assert_allclose(imu.read().mag, (-MAG_FIELD[0], -MAG_FIELD[2], -MAG_FIELD[1]), tol=tol)

    imu.set_acc_cross_axis_coupling((0.0, 1.0, 0.0))
    scene.step()
    assert_allclose(imu.read().lin_acc, GRAVITY, tol=5e-6)

    scene.reset()
    box.set_dofs_velocity((1.0, 2.0, 3.0), dofs_idx_local=slice(3, None))
    scene.step()
    assert_allclose(imu.read_ground_truth().ang_vel, (1.0, 3.0, -2.0), tol=0.1)

    imu.set_quat_offset((1.0, 0.0, 0.0, 0.0))
    imu.set_acc_cross_axis_coupling((0.0, 0.0, 0.0))
    scene.reset()

    assert_allclose(imu.read().lin_acc, 0.0, tol=gs.EPS)  # biased, but cache hasn't been updated yet
    assert_allclose(imu_delayed.read().lin_acc, 0.0, tol=gs.EPS)
    assert_allclose(imu_noisy.read().ang_vel, 0.0, tol=gs.EPS)
    assert_allclose(imu_noisy.read().mag, 0.0, tol=gs.EPS)  # biased

    imu.set_bias(BIAS + 2 * (0.0, 0.0, 0.0))
    scene.step()
    assert_allclose(imu.read().lin_acc, BIAS, tol=tol)
    assert_allclose(imu.read().mag, MAG_FIELD, tol=tol)


@pytest.mark.required
@pytest.mark.parametrize("n_envs", [0, 2])
def test_sensor_history_length_contact_and_imu(show_viewer, tol, n_envs):
    """history_length stacks recent frames from ring snapshot buffers (Contact + IMU)."""
    GRAVITY = -10.0
    DT = 1e-2
    HISTORY_LEN = 4

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=DT,
            gravity=(0.0, 0.0, GRAVITY),
        ),
        profiling_options=gs.options.ProfilingOptions(show_FPS=False),
        show_viewer=show_viewer,
    )
    scene.add_entity(gs.morphs.Plane())
    box = scene.add_entity(
        morph=gs.morphs.Box(
            size=(0.1, 0.1, 0.1),
            pos=(0.0, 0.0, 0.2),
        ),
    )

    contact_h = scene.add_sensor(
        gs.sensors.Contact(
            entity_idx=box.idx,
            history_length=HISTORY_LEN,
        )
    )
    imu_h = scene.add_sensor(
        gs.sensors.IMU(
            entity_idx=box.idx,
            history_length=HISTORY_LEN,
        )
    )
    imu_ref = scene.add_sensor(
        gs.sensors.IMU(
            entity_idx=box.idx,
        )
    )

    scene.build(n_envs=n_envs)

    def _expected_shape_with_history(shape: tuple[int, ...]):
        return (HISTORY_LEN, *shape) if n_envs == 0 else (n_envs, HISTORY_LEN, *shape)

    prev_c = None
    prev_i = None
    for _ in range(HISTORY_LEN * 2):
        scene.step()
        cg = contact_h.read_ground_truth()
        assert cg.shape == _expected_shape_with_history((1,))
        ig = imu_h.read_ground_truth()
        assert ig.lin_acc.shape == _expected_shape_with_history((3,))
        assert ig.ang_vel.shape == _expected_shape_with_history((3,))
        assert ig.mag.shape == _expected_shape_with_history((3,))

        assert_equal(contact_h.read(), cg)

        batch_shape = () if n_envs == 0 else (slice(None),)
        cur_slice = (*batch_shape, 0)
        prev_slice = (*batch_shape, 1)
        assert_allclose(ig.lin_acc[cur_slice], imu_ref.read_ground_truth().lin_acc, tol=tol)
        assert_allclose(ig.ang_vel[cur_slice], imu_ref.read_ground_truth().ang_vel, tol=tol)
        assert_allclose(ig.mag[cur_slice], imu_ref.read_ground_truth().mag, tol=tol)

        if prev_c is not None:
            assert_equal(cg[prev_slice], prev_c[cur_slice])
        if prev_i is not None:
            assert_allclose(ig.lin_acc[prev_slice], prev_i.lin_acc[cur_slice], tol=gs.EPS)
            assert_allclose(ig.ang_vel[prev_slice], prev_i.ang_vel[cur_slice], tol=gs.EPS)
            assert_allclose(ig.mag[prev_slice], prev_i.mag[cur_slice], tol=gs.EPS)
        prev_c = cg
        prev_i = ig


# ------------------------------------------------------------------------------------------
# ------------------------------------ Contact Sensors -------------------------------------
# ------------------------------------------------------------------------------------------


@pytest.mark.required
@pytest.mark.parametrize("n_envs", [0, 2])
def test_contact_sensors_gravity_force(n_envs, show_viewer, tol):
    """Test if the sensor will detect the correct forces being applied on a falling box."""
    GRAVITY = -10.0
    BIAS = (0.1, 0.2, 0.3)
    NOISE = 0.01
    DT = 1e-2
    DELAY_STEPS = 2

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            gravity=(0.0, 0.0, GRAVITY),
            dt=DT,
        ),
        profiling_options=gs.options.ProfilingOptions(
            show_FPS=False,
        ),
        show_viewer=show_viewer,
    )

    floor = scene.add_entity(morph=gs.morphs.Plane())

    # Add duck (with convex decomposition enabled) to offset geom index vs link index
    scene.add_entity(
        morph=gs.morphs.Mesh(
            file="meshes/duck.obj",
            scale=0.04,
            pos=(0.0, 1.0, 0.2),
            euler=(90, 0, 90),
        ),
    )

    box = scene.add_entity(
        morph=gs.morphs.Box(
            size=(1.0, 1.0, 1.0),  # volume = 1 m^3
            pos=(0.0, 0.0, 0.55),
        ),
        material=gs.materials.Rigid(
            rho=1.0,  # mass = 1.0 kg
        ),
        surface=gs.surfaces.Default(
            color=(1.0, 0.0, 0.0, 1.0),
        ),
    )
    box_2 = scene.add_entity(
        morph=gs.morphs.Box(
            size=(0.2, 0.2, 0.2),  # volume = 0.008 m^3
            pos=(1.0, 0.0, 0.4),
        ),
        material=gs.materials.Rigid(
            rho=100.0,  # mass = 0.8 kg
        ),
        surface=gs.surfaces.Default(
            color=(0.0, 1.0, 0.0, 1.0),
        ),
    )
    box_3 = scene.add_entity(
        morph=gs.morphs.Box(
            size=(0.2, 0.2, 0.2),  # volume = 0.008 m^3
            pos=(1.0, 0.0, 0.61),
        ),
        material=gs.materials.Rigid(
            rho=25.0,  # mass = 0.2 kg
        ),
        surface=gs.surfaces.Default(
            color=(0.0, 0.0, 1.0, 1.0),
        ),
    )

    bool_sensor_floor = scene.add_sensor(
        gs.sensors.Contact(
            entity_idx=floor.idx,
        )
    )
    bool_sensor_box_2 = scene.add_sensor(
        gs.sensors.Contact(
            entity_idx=box_2.idx,
        )
    )
    force_sensor = scene.add_sensor(
        gs.sensors.ContactForce(
            entity_idx=box.idx,
        )
    )
    force_sensor_box_2 = scene.add_sensor(
        gs.sensors.ContactForce(
            entity_idx=box_2.idx,
        )
    )
    force_sensor_noisy = scene.add_sensor(
        gs.sensors.ContactForce(
            entity_idx=box.idx,
            min_force=0.01,
            max_force=(10.0, 20.0, -GRAVITY / 2),
            noise=NOISE,
            bias=BIAS,
            random_walk=(NOISE * 0.01, NOISE * 0.02, NOISE * 0.03),
            delay=DT * DELAY_STEPS,
            jitter=0.01,
        )
    )
    # Adding extra sensor sharing same dtype to force discontinuous memory layout for ground truth when batched
    scene.add_sensor(
        gs.sensors.IMU(
            entity_idx=box.idx,
        )
    )

    scene.build(n_envs=n_envs)

    # Move CoM to get unbalanced forces on each contact points
    box_com_offset = (0.3, 0.1, 0.0)
    box.set_COM_shift(box_com_offset)

    # Rotate the box make sure the force is correctly computed in local frame
    box_2.set_dofs_position((np.pi / 2, np.pi / 4, np.pi / 2), dofs_idx_local=slice(3, None))

    # Add another cube on top of it make sure the forces are correctly aggregated
    box_3.set_dofs_position((-np.pi / 2, -np.pi / 4, -np.pi / 2), dofs_idx_local=slice(3, None))

    # Note that it is necessary to do a first step, because the initial state right after reset is not valid
    for _ in range(DELAY_STEPS + 1):
        scene.step()

    # Make sure that box CoM is valid
    assert_allclose(box.get_links_pos(ref="root_com")[..., :2], box_com_offset[:2], tol=tol)

    assert not bool_sensor_floor.read().any(), "ContactSensor for floor should not detect any contact yet."
    assert not bool_sensor_box_2.read().any(), "ContactSensor for box_2 should not detect any contact yet."
    assert_allclose(force_sensor_noisy.read_ground_truth(), 0.0, tol=gs.EPS)
    assert_allclose(force_sensor.read(), force_sensor_noisy.read_ground_truth(), tol=gs.EPS)
    assert_allclose(force_sensor_noisy.read(), BIAS, tol=NOISE * 3)

    for _ in range(20):
        scene.step()

    assert bool_sensor_floor.read().all(), "ContactSensor for floor should detect contact with the ground"
    assert not bool_sensor_box_2.read().any(), "ContactSensor for box_2 should not detect any contact yet."
    assert_allclose(force_sensor_noisy.read(), force_sensor_noisy.read(), tol=gs.EPS)

    for _ in range(80):
        scene.step()

    assert bool_sensor_box_2.read().all(), "ContactSensor for box_2 should detect contact with the ground"

    # Moving force back in world frame because box is not perfectly flat on the ground due to CoM offset
    with np.testing.assert_raises(AssertionError):
        assert_allclose(box.get_quat(), 0.0, atol=tol)
    # Unsaturated GT physics check uses force_sensor (no max_force). force_sensor_noisy clamps in _post_process,
    # which applies uniformly to read() and read_ground_truth().
    assert_allclose(
        gu.transform_by_quat(force_sensor.read_ground_truth(), box.get_quat()), (0.0, 0.0, -GRAVITY), tol=tol
    )

    # FIXME: Adding CoM offset on box is disturbing contact force computations on box_2 for some reason...
    assert_allclose(force_sensor_box_2.read_ground_truth(), (-0.8 * GRAVITY, 0.0, 0.0), tol=1e-2)

    assert_allclose(force_sensor_noisy.read()[..., :2], BIAS[:2], tol=NOISE * 3)
    assert_allclose(force_sensor_noisy.read()[..., 2], -GRAVITY / 2, tol=gs.EPS)


@pytest.mark.required
def test_contact_sensor_filter_link_idx(show_viewer):
    """Contact sensor filter_link_idx ignores contacts whose other participant is a listed link."""
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            gravity=(0.0, 0.0, -10.0),
        ),
        profiling_options=gs.options.ProfilingOptions(show_FPS=False),
        show_viewer=show_viewer,
    )
    floor = scene.add_entity(morph=gs.morphs.Plane())
    box_on_floor = scene.add_entity(
        morph=gs.morphs.Box(
            size=(0.2, 0.2, 0.2),
            pos=(0.0, 0.0, 0.1),
        ),
    )
    box = scene.add_entity(
        morph=gs.morphs.Box(
            size=(0.2, 0.2, 0.2),
            pos=(0.0, 0.5, 0.1),
        ),
    )
    sensor = scene.add_sensor(
        gs.sensors.Contact(
            entity_idx=box_on_floor.idx,
        )
    )
    sensor_filtered = scene.add_sensor(
        gs.sensors.Contact(
            entity_idx=box_on_floor.idx,
            filter_link_idx=(floor.link_start,),
        )
    )
    scene.build(n_envs=2)
    box.set_pos(
        (
            (0.0, 0.5, 0.1),  # box not touching box_on_floor
            (0.0, 0.0, 0.3),  # box on top of box_on_floor
        )
    )
    for _ in range(20):  # make sure the boxes are stably resting
        scene.step()
    data = sensor.read()
    filtered_data = sensor_filtered.read()
    assert data[0], "Contact sensor should detect contact with the floor"
    assert not filtered_data[0], "Contact sensor with filter_link_idx should filter out contact with the floor"
    assert data[1], "Contact sensor should detect contact with the box"
    assert filtered_data[1], "Contact sensor with filter_link_idx should still detect contact with the box"


# ------------------------------------------------------------------------------------------
# ------------------------------------ Raycast Sensors -------------------------------------
# ------------------------------------------------------------------------------------------


@pytest.mark.required
@pytest.mark.parametrize("n_envs", [0, 2])
def test_raycaster_hits(show_viewer, n_envs):
    """Test if the Raycaster sensor with GridPattern rays pointing to ground returns the correct distance."""
    NUM_RAYS_XY = (3, 5)
    SPHERE_POS = (2.5, 0.5, 1.0)
    BOX_SIZE = 0.05
    RAYCAST_BOX_SIZE = 0.1
    RAYCAST_GRID_SIZE_X = 1.0
    RAYCAST_HEIGHT = 1.0

    scene = gs.Scene(
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(-3.0, RAYCAST_GRID_SIZE_X * (NUM_RAYS_XY[1] / NUM_RAYS_XY[0]), 2 * RAYCAST_HEIGHT),
            camera_lookat=(1.5, RAYCAST_GRID_SIZE_X * (NUM_RAYS_XY[1] / NUM_RAYS_XY[0]), RAYCAST_HEIGHT),
        ),
        vis_options=gs.options.VisOptions(
            rendered_envs_idx=(0,),
            env_separate_rigid=False,
        ),
        profiling_options=gs.options.ProfilingOptions(
            show_FPS=False,
        ),
        show_viewer=show_viewer,
    )
    scene.add_entity(gs.morphs.Plane())

    spherical_sensor = scene.add_entity(
        gs.morphs.Sphere(
            radius=RAYCAST_HEIGHT,
            pos=SPHERE_POS,
            fixed=True,
        ),
    )
    spherical_raycaster = scene.add_sensor(
        gs.sensors.Raycaster(
            pattern=gs.sensors.raycaster.SphericalPattern(
                n_points=NUM_RAYS_XY,
            ),
            entity_idx=spherical_sensor.idx,
            return_world_frame=False,
            draw_debug=show_viewer,
            debug_ray_start_color=(0.0, 0.0, 0.0, 0.0),
            debug_ray_hit_color=(1.0, 0.0, 0.0, 1.0),
        )
    )

    grid_sensor = scene.add_entity(
        gs.morphs.Box(
            size=(RAYCAST_BOX_SIZE, RAYCAST_BOX_SIZE, RAYCAST_BOX_SIZE),
            pos=(0.0, 0.0, RAYCAST_HEIGHT + 0.5 * RAYCAST_BOX_SIZE),
            collision=False,
            fixed=False,
        ),
    )
    grid_res = RAYCAST_GRID_SIZE_X / (NUM_RAYS_XY[0] - 1)
    grid_size_y = grid_res * (NUM_RAYS_XY[1] - 1)
    grid_raycaster = scene.add_sensor(
        gs.sensors.Raycaster(
            pattern=gs.sensors.raycaster.GridPattern(
                resolution=grid_res,
                size=(RAYCAST_GRID_SIZE_X, grid_size_y),
                direction=(0.0, 0.0, -1.0),  # pointing downwards to ground
            ),
            entity_idx=grid_sensor.idx,
            pos_offset=(0.0, 0.0, -0.5 * RAYCAST_BOX_SIZE),
            return_world_frame=True,
            draw_debug=show_viewer,
            debug_ray_start_color=(0.0, 0.0, 0.0, 0.0),
            debug_ray_hit_color=(0.0, 1.0, 0.0, 1.0),
        )
    )
    depth_camera = scene.add_sensor(
        gs.sensors.DepthCamera(
            pattern=gs.sensors.raycaster.DepthCameraPattern(
                res=NUM_RAYS_XY[::-1],
            ),
            entity_idx=spherical_sensor.idx,
            draw_debug=show_viewer,
            debug_ray_start_color=(0.0, 0.0, 0.0, 0.0),
            debug_ray_hit_color=(0.0, 0.0, 1.0, 1.0),
        ),
    )

    obstacle_1 = scene.add_entity(
        gs.morphs.Box(
            size=(BOX_SIZE, BOX_SIZE, BOX_SIZE),
            pos=(grid_res, grid_res, 0.5 * BOX_SIZE),
        ),
    )
    obstacle_2 = scene.add_entity(
        gs.morphs.Box(
            size=(BOX_SIZE, BOX_SIZE, BOX_SIZE),
            pos=(RAYCAST_GRID_SIZE_X, grid_size_y, RAYCAST_HEIGHT + RAYCAST_BOX_SIZE + BOX_SIZE),
            fixed=True,
        ),
    )

    # Build the simulation and do one step
    scene.build(n_envs=n_envs)
    batch_shape = (n_envs,) if n_envs > 0 else ()

    # Validate grid raycast
    for obstacle_pos, sensor_pos, hit_ij in (
        (None, None, (-1, -2)),
        ((grid_res, grid_res, BOX_SIZE), None, (-1, -2)),
        (None, (*(grid_res * (e - 2) for e in NUM_RAYS_XY), RAYCAST_HEIGHT + 0.5 * RAYCAST_BOX_SIZE), (1, 0)),
    ):
        # Update obstacle and/or sensor position if necessary
        if obstacle_pos is not None:
            obstacle_1.set_pos(np.tile(obstacle_pos, (*batch_shape, 1)))
        obstacle_pos = obstacle_1.get_pos()
        if sensor_pos is not None:
            grid_sensor.set_pos(np.tile(sensor_pos, (*batch_shape, 1)))
        scene.sim._sensor_manager.step()
        if show_viewer:
            scene.visualizer.update(force=True)

        # Fetch updated sensor data
        grid_hits = grid_raycaster.read().points
        grid_distances = grid_raycaster.read().distances
        assert grid_distances.shape == (*batch_shape, *NUM_RAYS_XY)

        # Check hits
        grid_sensor_origin = grid_sensor.get_pos()
        x = torch.linspace(-0.5, 0.5, NUM_RAYS_XY[0]) * RAYCAST_GRID_SIZE_X + grid_sensor_origin[..., [0]]
        y = torch.linspace(-0.5, 0.5, NUM_RAYS_XY[1]) * grid_size_y + grid_sensor_origin[..., [1]]
        # xg, yg = torch.meshgrid(x, y, indexing="ij")
        xg = x.unsqueeze(-1).expand((*batch_shape, -1, NUM_RAYS_XY[1]))
        yg = y.unsqueeze(-2).expand((*batch_shape, NUM_RAYS_XY[0], -1))
        zg = torch.zeros((*batch_shape, *NUM_RAYS_XY))
        zg[(..., *hit_ij)] = obstacle_pos[..., 2] + 0.5 * BOX_SIZE
        grid_hits_ref = torch.stack([xg, yg, zg], dim=-1)
        assert_allclose(grid_hits, grid_hits_ref, tol=gs.EPS)

        # Check distances
        grid_distances_ref = torch.full((*batch_shape, *NUM_RAYS_XY), RAYCAST_HEIGHT)
        grid_distances_ref[(..., *hit_ij)] = RAYCAST_HEIGHT - obstacle_pos[..., 2] - 0.5 * BOX_SIZE
        assert_allclose(grid_distances, grid_distances_ref, tol=gs.EPS)

    # Validate spherical raycast
    spherical_distances = spherical_raycaster.read().distances
    assert spherical_distances.shape == (*batch_shape, *NUM_RAYS_XY)
    # Note that the tolerance must be large because the sphere geometry is discretized
    assert_allclose(spherical_distances, RAYCAST_HEIGHT, tol=5e-3)

    # Check that we can read image from depth camera
    assert_equal(depth_camera.read_image().shape, batch_shape + NUM_RAYS_XY)
    # Note that the tolerance must be large because the sphere geometry is discretized
    assert_allclose(depth_camera.read_image(), RAYCAST_HEIGHT, tol=5e-3)

    # Simulate for a while and check again that the ray is casted properly
    offset = torch.from_numpy(np.random.rand(*batch_shape, 3)).to(dtype=gs.tc_float, device=gs.device)
    for entity in (grid_sensor, obstacle_1, obstacle_2):
        pos = entity.get_pos() + offset
        if entity is obstacle_2:
            pos[..., 2] = BOX_SIZE / 2
        entity.set_pos(pos)
    if show_viewer:
        scene.visualizer.update(force=True)
    grid_sensor_pos = grid_sensor.get_pos()
    for _ in range(60):
        scene.step()
    grid_sensor.set_pos(grid_sensor_pos)
    scene.sim._sensor_manager.step()
    if show_viewer:
        scene.visualizer.update(force=True)

    grid_distances = grid_raycaster.read().distances
    grid_distances_ref = torch.full((*batch_shape, *NUM_RAYS_XY), RAYCAST_HEIGHT)
    grid_distances_ref[(..., -1, -2)] = RAYCAST_HEIGHT - BOX_SIZE
    grid_distances_ref[(..., *hit_ij)] = RAYCAST_HEIGHT - BOX_SIZE
    grid_distances_ref += offset[..., 2].reshape((*(-1 for e in batch_shape), 1, 1))
    assert_allclose(grid_distances, grid_distances_ref, tol=1e-3)


@pytest.mark.required
@pytest.mark.parametrize("n_envs", [0, 2])
@pytest.mark.parametrize("kin_raycastable", [True, False])
def test_raycaster_against_visual(tmp_path, show_viewer, n_envs, kin_raycastable):
    # Two depth cameras, one per entity:
    #   - cam_kin -> KinematicEntity sphere. When use_visual_raycasting=True the depth camera reads the entity's
    #     visual mesh (including set_vverts overrides, which survive step() until set_vverts(None) hands control
    #     back to FK). When False the kinematic entity is completely ignored by the raycaster.
    #   - cam_rigid -> RigidEntity whose visual mesh (sphere radius 0.2) is intentionally different from its collision
    #     mesh (capsule radius 0.05). With use_visual_raycasting=True the depth must match the visual sphere.
    urdf_path = tmp_path / "vis_diff.urdf"
    urdf_path.write_text(
        textwrap.dedent(
            """
            <robot name="vis_diff">
                <link name="root">
                    <visual>
                        <origin rpy="0 0 0" xyz="0 0 0"/>
                        <geometry>
                            <sphere radius="0.2"/>
                        </geometry>
                    </visual>
                    <collision>
                        <origin rpy="0 0 0" xyz="0 0 0"/>
                        <geometry>
                            <capsule radius="0.05" length="0.05"/>
                        </geometry>
                    </collision>
                </link>
            </robot>
            """
        )
    )

    scene = gs.Scene(
        profiling_options=gs.options.ProfilingOptions(
            show_FPS=False,
        ),
        show_viewer=show_viewer,
    )
    plane = scene.add_entity(gs.morphs.Plane())
    kin_sphere = scene.add_entity(
        morph=gs.morphs.Mesh(
            file="meshes/sphere.obj",
            scale=0.2,
            pos=(0.0, 0.0, 0.5),
            fixed=True,
            enable_custom_vverts=True,
        ),
        material=gs.materials.Kinematic(use_visual_raycasting=kin_raycastable),
    )
    scene.add_entity(
        morph=gs.morphs.URDF(
            file=str(urdf_path),
            pos=(0.0, 0.0, 1.5),
            fixed=True,
        ),
        material=gs.materials.Rigid(use_visual_raycasting=True),
    )
    cam_kin = scene.add_sensor(
        gs.sensors.DepthCamera(
            pattern=gs.sensors.DepthCameraPattern(
                res=(40, 30),
                fov_horizontal=30.0,
            ),
            entity_idx=plane.idx,
            link_idx_local=0,
            pos_offset=(-1.0, 0.0, 0.5),
            euler_offset=(0.0, 0.0, 0.0),
            max_range=5.0,
            return_world_frame=True,
        ),
    )
    cam_rigid = scene.add_sensor(
        gs.sensors.DepthCamera(
            pattern=gs.sensors.DepthCameraPattern(
                res=(40, 30),
                fov_horizontal=30.0,
            ),
            entity_idx=plane.idx,
            link_idx_local=0,
            pos_offset=(-1.0, 0.0, 1.5),
            euler_offset=(0.0, 0.0, 0.0),
            max_range=5.0,
            return_world_frame=True,
        ),
    )
    if n_envs > 0:
        scene.build(n_envs=n_envs)
    else:
        scene.build()
    scene.step()

    # Each camera at x=-1 along its own z-row looks along +x. The center pixel hits the closest point of its target
    # sphere at x=-0.2 -> depth 0.8. For cam_rigid this comes from the visual BVH (not the collision capsule). When
    # the kinematic entity opts out of raycasting, cam_kin sees nothing and returns the no_hit_value (max_range=5.0).
    NO_HIT = 5.0  # max_range
    kin_at_origin = 0.8 if kin_raycastable else NO_HIT
    kin_scaled = 0.6 if kin_raycastable else NO_HIT
    assert_allclose(cam_kin.read_image()[..., 15, 20], kin_at_origin, tol=1e-2)
    assert_allclose(cam_rigid.read_image()[..., 15, 20], 0.8, tol=1e-2)

    # Scale the kinematic sphere by 2x around its center via per-vertex set_vverts. The new radius is 0.4, so the
    # closest point becomes x=-0.4 and the depth at the center pixel drops to 0.6. Scaling perturbs each vvert by a
    # different amount, so only the correct vvert-to-state mapping yields 0.6. cam_rigid is unaffected.
    fk_vverts = tensor_to_array(kin_sphere.get_vverts())
    center = np.array([0.0, 0.0, 0.5], dtype=np.float32)
    kin_sphere.set_vverts((fk_vverts - center) * 2.0 + center)
    scene.step()
    assert_allclose(cam_kin.read_image()[..., 15, 20], kin_scaled, tol=1e-2)
    assert_allclose(cam_rigid.read_image()[..., 15, 20], 0.8, tol=1e-2)

    # Push the kinematic sphere far away. cam_kin should report no_hit_value at the center pixel; cam_rigid still sees
    # the rigid visual sphere.
    kin_sphere.set_vverts((100.0, 100.0, 100.0))
    scene.step()
    assert_allclose(cam_kin.read_image()[..., 15, 20], NO_HIT, tol=gs.EPS)
    assert_allclose(cam_rigid.read_image()[..., 15, 20], 0.8, tol=1e-2)

    # Restoring FK control returns the original hit distance on cam_kin; cam_rigid stays put.
    kin_sphere.set_vverts(None)
    scene.step()
    assert_allclose(cam_kin.read_image()[..., 15, 20], kin_at_origin, tol=1e-2)
    assert_allclose(cam_rigid.read_image()[..., 15, 20], 0.8, tol=1e-2)


@pytest.mark.required
def test_lidar_bvh_parallel_env(show_viewer, tol):
    """Verify each environment receives a different lidar distance when geometries differ."""
    scene = gs.Scene(
        vis_options=gs.options.VisOptions(
            rendered_envs_idx=(1,),
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(1, -5, 3),
            camera_lookat=(1, 0.5, 0),
        ),
        show_viewer=show_viewer,
    )
    scene.add_entity(gs.morphs.Plane())

    sensor_mount = scene.add_entity(
        gs.morphs.Box(
            size=(0.1, 0.1, 0.1),
            pos=(0.0, 0.0, 0.5),
            fixed=True,
            collision=False,
        )
    )
    obstacle_1 = scene.add_entity(
        gs.morphs.Box(
            size=(0.2, 0.2, 0.2),
            pos=(1.0, 0.0, 0.5),
            fixed=True,
        ),
    )
    obstacle_2 = scene.add_entity(
        gs.morphs.Box(
            size=(0.05, 0.4, 0.4),
            pos=(1.0, 0.0, 0.5),
            fixed=True,
        ),
    )

    lidar = scene.add_sensor(
        gs.sensors.Lidar(
            entity_idx=sensor_mount.idx,
            pattern=gs.options.sensors.SphericalPattern(
                n_points=(1, 1),
                fov=(0.0, 0.0),
            ),
            max_range=5.0,
            draw_debug=show_viewer,
            debug_ray_start_color=(0.0, 0.0, 0.0, 0.0),
            debug_ray_hit_color=(1.0, 0.0, 0.0, 1.0),
        )
    )

    scene.build(n_envs=2)

    sensor_positions = np.array([[0.0, 0.0, 0.5], [0.0, 1.0, 0.5]], dtype=gs.np_float)
    obstacle_1_positions = np.array([[1.1, 0.0, 0.5], [2.5, 1.0, 0.5]], dtype=gs.np_float)
    obstacle_2_positions = np.array([[1.4, 0.0, 0.5], [2.2, 1.0, 0.5]], dtype=gs.np_float)
    sensor_mount.set_pos(sensor_positions)
    obstacle_1.set_pos(obstacle_1_positions)
    obstacle_2.set_pos(obstacle_2_positions)

    scene.step()

    distances = lidar.read().distances
    assert distances.shape == (2, 1, 1)
    lidar_distances = distances[:, 0, 0]

    front_positions = np.minimum(obstacle_1_positions[:, 0] - 0.1, obstacle_2_positions[:, 0] - 0.025)
    expected_distances = front_positions - sensor_positions[:, 0]
    assert_allclose(lidar_distances, expected_distances, tol=tol)


@pytest.mark.required
def test_lidar_cache_offset_parallel_env(show_viewer, tol):
    scene = gs.Scene(
        show_viewer=show_viewer,
    )

    scene.add_entity(
        morph=gs.morphs.Plane(),
    )
    cube = scene.add_entity(
        morph=gs.morphs.Box(
            size=(0.1, 0.1, 1.0),
            pos=(0.0, 0.0, 0.5),
        ),
    )

    sensors = [
        scene.add_sensor(
            gs.sensors.Raycaster(
                pattern=gs.sensors.raycaster.SphericalPattern(
                    n_points=(2, 2),
                ),
                entity_idx=cube.idx,
                return_world_frame=False,
            )
        ),
        scene.add_sensor(
            gs.sensors.Raycaster(
                pattern=gs.sensors.raycaster.SphericalPattern(
                    n_points=(2, 2),
                ),
                entity_idx=cube.idx,
                return_world_frame=False,
            )
        ),
        scene.add_sensor(
            gs.sensors.Raycaster(
                pattern=gs.sensors.raycaster.SphericalPattern(
                    n_points=(2, 2),
                ),
                entity_idx=cube.idx,
                return_world_frame=False,
            )
        ),
    ]

    scene.build()

    scene.step()
    for sensor in sensors:
        sensor_data = sensor.read()
        assert (sensor_data.distances > gs.EPS).any()
        assert (sensor_data.points.abs() > gs.EPS).any()


# ------------------------------------------------------------------------------------------
# -------------------------------------- Kinematic Tactile Sensors ---------------------------------------
# ------------------------------------------------------------------------------------------


@pytest.mark.required
@pytest.mark.parametrize("n_envs", [0, 2])
def test_temperature_grid_sensor_contact_and_reset(show_viewer, tol, n_envs):
    """After build, grid is at base temp. Hot box on center heats center above corner; cold box cools it. Move away -> near base; reset -> exactly base."""
    BOX_SIZE = 0.06
    PLATFORM_SIZE = 0.2
    FAR_POS = (PLATFORM_SIZE * 1.5, PLATFORM_SIZE * 1.5, PLATFORM_SIZE * 1.5)
    GRID_SIZE = (3, 3, 1)
    GRID_CENTER = (GRID_SIZE[0] // 2, GRID_SIZE[1] // 2, GRID_SIZE[2] // 2)
    BASE_TEMP = 22.0
    DIFF_TEMP = 0.5

    scene = gs.Scene(show_viewer=show_viewer)
    scene.add_entity(gs.morphs.Plane())
    platform = scene.add_entity(
        gs.morphs.Box(
            size=(PLATFORM_SIZE, PLATFORM_SIZE, PLATFORM_SIZE),
            pos=(0.0, 0.0, PLATFORM_SIZE / 2),
            fixed=True,
        )
    )
    hot_box = scene.add_entity(
        gs.morphs.Box(
            size=(BOX_SIZE, BOX_SIZE, BOX_SIZE),
            pos=(0.0, 0.0, PLATFORM_SIZE + BOX_SIZE / 2),
        )
    )
    cold_box = scene.add_entity(
        gs.morphs.Box(
            size=(BOX_SIZE, BOX_SIZE, BOX_SIZE),
            pos=FAR_POS,
        ),
    )
    TemperatureProperties = gs.sensors.TemperatureProperties
    sensor = scene.add_sensor(
        gs.sensors.TemperatureGrid(
            ambient_temperature=BASE_TEMP,
            convection_coefficient=0.0,
            simulate_all_link_temperatures=False,
            entity_idx=platform.idx,
            grid_size=GRID_SIZE,
            properties_dict={
                platform.base_link_idx: TemperatureProperties(
                    base_temperature=BASE_TEMP,
                    conductivity=400.0,
                    density=2000.0,
                    specific_heat=1.0,
                    emissivity=0.95,
                ),
                hot_box.base_link_idx: TemperatureProperties(
                    base_temperature=BASE_TEMP + 100.0,
                    conductivity=200.0,
                    density=3000.0,
                    specific_heat=1.0,
                    emissivity=0.1,
                ),
                # default properties; should apply to the cold box
                -1: TemperatureProperties(
                    base_temperature=BASE_TEMP - 100.0,
                    conductivity=150.0,
                    density=8000.0,
                    specific_heat=1.0,
                    emissivity=0.2,
                ),
            },
        )
    )
    scene.build(n_envs=n_envs)

    # After build, all cells at base temperature
    assert_allclose(sensor.read_ground_truth(), BASE_TEMP, tol=tol)

    # Hot box on center
    hot_box.set_pos((0.0, 0.0, PLATFORM_SIZE + BOX_SIZE / 2))
    for _ in range(50):
        scene.step()
    data = sensor.read()
    assert (data > BASE_TEMP + DIFF_TEMP).all(), f"Hot box should have heated the grid by at least {DIFF_TEMP}°C"
    assert (data[..., GRID_CENTER[0], GRID_CENTER[1], GRID_CENTER[2]] > data[0, 0, 0]).all(), (
        "Center cell should be hotter than corner"
    )

    # Reset: exactly base temperature everywhere
    scene.reset()
    assert_allclose(sensor.read_ground_truth(), BASE_TEMP, tol=tol)

    # Cold box on center
    hot_box.set_pos(FAR_POS)
    cold_box.set_pos((0.0, 0.0, PLATFORM_SIZE + BOX_SIZE / 2))
    for _ in range(50):
        scene.step()
    data = sensor.read()
    assert (data < BASE_TEMP - DIFF_TEMP).all(), f"Cold box should have cooled the grid by at least {DIFF_TEMP}°C"
    assert (data[..., GRID_CENTER[0], GRID_CENTER[1], GRID_CENTER[2]] < data[0, 0, 0]).all(), (
        "Center cell should be colder than corner"
    )

    # Move both away; step until grid returns near base
    hot_box.set_pos(FAR_POS)
    cold_box.set_pos((-FAR_POS[0], -FAR_POS[1], FAR_POS[2]))
    for _ in range(150):
        scene.step()
    data = sensor.read()
    assert_allclose(data, BASE_TEMP, tol=5e-2)


@pytest.mark.required
@pytest.mark.parametrize("n_envs", [0, 2])
def test_temperature_grid_simulate_all_link_temps(show_viewer, tol, n_envs):
    """With simulate_all_link_temperatures=True, two boxes in contact exchange heat."""
    BOX_SIZE = 0.06
    BASE_TEMP = 22.0
    HOT_BASE = BASE_TEMP + 80.0
    COLD_BASE = BASE_TEMP - 80.0

    scene = gs.Scene(show_viewer=show_viewer)
    scene.add_entity(gs.morphs.Plane())
    hot_box = scene.add_entity(
        gs.morphs.Box(
            size=(BOX_SIZE, BOX_SIZE, BOX_SIZE),
            pos=(0.0, 0.0, BOX_SIZE),
        )
    )
    cold_box = scene.add_entity(
        gs.morphs.Box(
            size=(BOX_SIZE, BOX_SIZE, BOX_SIZE),
            pos=(0.0, 0.0, BOX_SIZE * 2 + 0.001),
        )
    )
    hot_link_idx = hot_box.base_link_idx
    cold_link_idx = cold_box.base_link_idx
    sensor1 = scene.add_sensor(
        gs.sensors.TemperatureGrid(
            entity_idx=hot_box.idx,
            grid_size=(1, 1, 1),
            ambient_temperature=BASE_TEMP,
            properties_dict={
                hot_link_idx: gs.sensors.TemperatureProperties(
                    base_temperature=HOT_BASE,
                    conductivity=200.0,
                    density=2000.0,
                    specific_heat=1.0,
                    emissivity=0.1,
                ),
                cold_link_idx: gs.sensors.TemperatureProperties(
                    base_temperature=COLD_BASE,
                    conductivity=200.0,
                    density=2000.0,
                    specific_heat=1.0,
                    emissivity=0.1,
                ),
            },
            simulate_all_link_temperatures=True,
        )
    )
    sensor2 = scene.add_sensor(
        gs.sensors.TemperatureGrid(
            entity_idx=cold_box.idx,
            grid_size=(1, 1, 1),
        )
    )
    scene.build(n_envs=n_envs)

    link_temps = sensor1.link_temperatures  # (n_envs, n_links)

    assert_equal(link_temps[..., hot_link_idx], HOT_BASE)
    assert_equal(link_temps[..., cold_link_idx], COLD_BASE)

    cold_box.set_pos((0.0, 0.0, BOX_SIZE / 2))
    for _ in range(100):
        scene.step()

    assert_equal(sensor1.link_temperatures, sensor2.link_temperatures)

    assert (link_temps[..., hot_link_idx] < HOT_BASE - 1.0).all(), "Hot box link should have cooled"
    assert (link_temps[..., cold_link_idx] > COLD_BASE + 1.0).all(), "Cold box link should have heated up"

    assert_allclose(torch.mean(sensor1.read()), link_temps[..., hot_link_idx], tol=2e-2)
    assert_allclose(torch.mean(sensor2.read()), link_temps[..., cold_link_idx], tol=2e-2)


@pytest.mark.required
@pytest.mark.parametrize("n_envs", [0, 2])
def test_surface_distance_sensor_box_sphere(show_viewer, tol, n_envs):
    SPHERE_RADIUS = 0.05
    DISTANCE = 0.15
    MAX_RANGE = 10.0
    BOX_PROBE_POS = [(0.0, 0.0, 0.0), (0.0, 0.0, 0.05)]
    SPHERE_PROBE_POS = [(0.0, 0.0, SPHERE_RADIUS)]

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            gravity=(0.0, 0.0, 0.0),
        ),
        profiling_options=gs.options.ProfilingOptions(
            show_FPS=False,
        ),
        show_viewer=show_viewer,
    )
    box = scene.add_entity(
        gs.morphs.Box(
            size=(0.1, 0.1, 0.1),
            pos=(0.0, 0.0, 0.0),
        ),
    )
    # Tracked objects
    sphere1 = scene.add_entity(
        gs.morphs.Sphere(
            radius=SPHERE_RADIUS,
            pos=(0.0, 0.0, DISTANCE),
        ),
    )
    sphere2 = scene.add_entity(
        gs.morphs.Sphere(
            radius=SPHERE_RADIUS,
            pos=(0.0, 0.0, DISTANCE * 2.0),
        ),
    )
    # Not tracked objects
    sphere3 = scene.add_entity(
        gs.morphs.Sphere(
            radius=SPHERE_RADIUS,
            pos=(0.0, DISTANCE / 2.0, 0.0),
        ),
    )

    box_to_spheres_dist_sensor = scene.add_sensor(
        gs.sensors.SurfaceDistanceProbe(
            entity_idx=box.idx,
            probe_local_pos=BOX_PROBE_POS,
            probe_radius=MAX_RANGE,
            track_link_idx=(sphere1.base_link_idx, sphere2.base_link_idx),
        )
    )
    sphere_to_box_dist_sensor = scene.add_sensor(
        gs.sensors.SurfaceDistanceProbe(
            entity_idx=sphere1.idx,
            probe_local_pos=SPHERE_PROBE_POS,
            probe_radius=MAX_RANGE,
            track_link_idx=(box.base_link_idx,),
            resolution=0.001,
            bias=0.1,
            noise=0.01,
            random_walk=0.01,
        )
    )
    scene.build(n_envs=n_envs)

    scene.step()

    box_prox_data = box_to_spheres_dist_sensor.read()
    sphere_prox_noisy_data = sphere_to_box_dist_sensor.read()
    sphere_prox_data = sphere_to_box_dist_sensor.read_ground_truth()

    for i in range(len(BOX_PROBE_POS)):
        assert_allclose(box_prox_data[..., i], DISTANCE - SPHERE_RADIUS - BOX_PROBE_POS[i][2], tol=tol)
    assert_allclose(box_to_spheres_dist_sensor.nearest_points, (0.0, 0.0, DISTANCE - SPHERE_RADIUS), tol=tol)
    assert_allclose(sphere_prox_data, DISTANCE, tol=tol)

    with np.testing.assert_raises(AssertionError):
        assert_allclose(sphere_prox_noisy_data, sphere_prox_data, tol=tol)

    sphere1_pos = np.array((0.0, 0.0, DISTANCE * 3.0))
    sphere1.set_pos(sphere1_pos)

    scene.step()

    box_prox_data = box_to_spheres_dist_sensor.read()
    sphere_prox_data = sphere_to_box_dist_sensor.read_ground_truth()

    assert_allclose(box_prox_data[..., 0], DISTANCE * 2.0 - SPHERE_RADIUS, tol=tol)
    assert_allclose(box_prox_data[..., 1], DISTANCE * 2.0 - SPHERE_RADIUS - 0.05, tol=tol)
    assert_allclose(sphere_prox_data, DISTANCE * 3.0, tol=tol)

    box_pos = np.array((0.0, 0.0, -MAX_RANGE))
    box.set_pos(box_pos)
    scene.step()

    box_prox_data = box_to_spheres_dist_sensor.read()
    sphere_prox_data = sphere_to_box_dist_sensor.read_ground_truth()

    assert_allclose(box_prox_data, MAX_RANGE, tol=tol)
    assert_allclose(sphere_prox_data, MAX_RANGE, tol=tol)
    for i in range(len(BOX_PROBE_POS)):
        assert_allclose(
            box_to_spheres_dist_sensor.nearest_points[..., i, :],
            np.array(BOX_PROBE_POS[i]) + box_pos,
            tol=tol,
            err_msg="When out of range, points should be the probe position in world frame",
        )
    assert_allclose(
        sphere_to_box_dist_sensor.nearest_points,
        np.array(SPHERE_PROBE_POS) + sphere1_pos,
        tol=tol,
        err_msg="When out of range, points should be the probe position in world frame",
    )


def _as_env_batch(data, n_envs: int) -> torch.Tensor:
    data = torch.as_tensor(data, device=gs.device)
    return data.unsqueeze(0) if n_envs == 0 else data


# ------------------------------------------------------------------------------------------
# ----------------------------------- Tactile Sensors --------------------------------------
# ------------------------------------------------------------------------------------------


@pytest.mark.required
@pytest.mark.parametrize("n_envs", [0, 2])
def test_kinematic_contact_probe_box_sphere_support(show_viewer, tol, n_envs):
    """Test ContactProbe, ContactDepthProbe, and KinematicTaxel on a box resting on ground with sphere on top."""
    BOX_SIZE = 0.5
    PROBE_RADIUS = 0.05
    PENETRATION = 0.02
    CONTACT_THRESHOLD = 0.002
    STIFFNESS = 100.0
    SPHERE_RADIUS = 0.1

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            gravity=(0.0, 0.0, 0.0),
        ),
        profiling_options=gs.options.ProfilingOptions(
            show_FPS=False,
        ),
        show_viewer=show_viewer,
    )
    scene.add_entity(gs.morphs.Plane())
    box = scene.add_entity(
        gs.morphs.Box(
            size=(BOX_SIZE, BOX_SIZE, BOX_SIZE),
            pos=(0.0, 0.0, BOX_SIZE / 2 - PENETRATION),  # box is penetrating ground plane
            fixed=False,  # probe will not detect fixed-fixed contact
        )
    )
    sphere = scene.add_entity(
        gs.morphs.Sphere(
            radius=SPHERE_RADIUS,
            pos=(0.0, 0.0, BOX_SIZE + SPHERE_RADIUS + 0.2),  # start with sphere above the box
            fixed=True,
        )
    )

    probe_local_pos = (
        (0.0, 0.0, BOX_SIZE / 2),
        (BOX_SIZE / 4, BOX_SIZE / 4, BOX_SIZE / 2),
        (-BOX_SIZE / 4, -BOX_SIZE / 4, BOX_SIZE / 2),
        (0.0, 0.0, -BOX_SIZE / 2),
    )
    probe_normals = (
        (0.0, 0.0, 1.0),
        (0.0, 0.0, 1.0),
        (0.0, 0.0, 1.0),
        (0.0, 0.0, -1.0),
    )
    probe_radii = (
        PROBE_RADIUS,
        PROBE_RADIUS / 10.0,
        BOX_SIZE / 3.0,
        PROBE_RADIUS,
    )
    common_kwargs = dict(
        entity_idx=box.idx,
        probe_local_pos=probe_local_pos,
        probe_radius=probe_radii,
        draw_debug=show_viewer,
    )
    contact_probe = scene.add_sensor(
        gs.sensors.ContactProbe(
            contact_threshold=CONTACT_THRESHOLD,
            **common_kwargs,
        )
    )
    depth_probe = scene.add_sensor(gs.sensors.ContactDepthProbe(**common_kwargs))
    noisy_radius_depth_probe = scene.add_sensor(
        gs.sensors.ContactDepthProbe(
            probe_radius_noise=0.25,
            **common_kwargs,
        )
    )
    taxel = scene.add_sensor(
        gs.sensors.KinematicTaxel(
            probe_local_normal=probe_normals,
            normal_stiffness=STIFFNESS,
            normal_damping=0.0,
            shear_scalar=0.0,
            twist_scalar=0.0,
            **common_kwargs,
        )
    )
    sphere_taxel = scene.add_sensor(
        gs.sensors.KinematicTaxel(
            entity_idx=sphere.idx,
            probe_local_pos=((0.0, 0.0, -SPHERE_RADIUS),),
            probe_local_normal=((0.0, 0.0, -1.0),),
            probe_radius=PROBE_RADIUS,
            normal_stiffness=STIFFNESS,
            normal_damping=0.0,
            shear_scalar=0.0,
            twist_scalar=0.0,
            draw_debug=show_viewer,
        )
    )

    scene.build(n_envs=n_envs)
    scene.step()

    depth = _as_env_batch(depth_probe.read_ground_truth(), n_envs)
    contact = _as_env_batch(contact_probe.read_ground_truth(), n_envs)
    force = _as_env_batch(taxel.read_ground_truth().force, n_envs)
    torque = _as_env_batch(taxel.read_ground_truth().torque, n_envs)

    assert_equal(contact, depth > CONTACT_THRESHOLD)
    assert _as_env_batch(noisy_radius_depth_probe.read(), n_envs).shape == depth.shape
    # Check that the box's bottom probe (idx 3) detects the ground.
    assert (depth[..., 3] > tol).all(), "Bottom probe should detect the ground."
    assert (force[..., 3, 2] > tol).all(), "Bottom taxel force should point upward."
    # Top probes should not detect anything yet.
    assert_allclose(depth[..., :3], 0.0, tol=gs.EPS)
    assert_allclose(force[..., :3, :], 0.0, tol=gs.EPS)
    assert_allclose(torque, 0.0, tol=gs.EPS)

    # Forces should be equivalent to the penetration * stiffness along normal vector.
    expected_normals = -torch.tensor(probe_normals, dtype=gs.tc_float, device=gs.device)
    assert_allclose(force, depth.unsqueeze(-1) * STIFFNESS * expected_normals, tol=tol)

    # Now position the sphere to penetrate the top of the box.
    box_top_z = BOX_SIZE - PENETRATION
    sphere.set_pos((0.0, 0.0, box_top_z + SPHERE_RADIUS - PENETRATION))
    scene.step()

    depth = _as_env_batch(depth_probe.read_ground_truth(), n_envs)
    contact = _as_env_batch(contact_probe.read_ground_truth(), n_envs)
    force = _as_env_batch(taxel.read_ground_truth().force, n_envs)
    sphere_force = _as_env_batch(sphere_taxel.read_ground_truth().force, n_envs)

    assert_equal(contact, depth > CONTACT_THRESHOLD)
    assert (depth[..., 0] > tol).all(), "Top center probe should detect the sphere."
    assert (force[..., 0, 2] < -tol).all(), "Top center taxel force should point downward."
    assert_allclose(depth[..., 1], 0.0, tol=gs.EPS)
    assert (depth[..., 2] > tol).all(), "Large offset probe should detect the nearby sphere."
    assert (sphere_force[..., 0, 2] > tol).all(), "Sphere taxel should see the box underneath."

    # Move sphere away and check no contact.
    sphere.set_pos((0.0, 0.0, box_top_z + SPHERE_RADIUS + PROBE_RADIUS + 0.2))
    scene.step()
    assert_allclose(sphere_taxel.read_ground_truth().force, 0.0, tol=gs.EPS)


@pytest.mark.required
@pytest.mark.parametrize("n_envs", [0, 2])
def test_elastomer_sensor_sphere_ground_dilate_shear(show_viewer, tol, n_envs):
    """ElastomerTaxel should separate dilation and shear on a dome-like sensor surface."""
    SPHERE_RADIUS = 0.2
    PROBE_RADIUS = 0.02
    PENETRATION = 0.01
    GROUND_THICKNESS = 0.08
    N_RINGS = 3
    LATERAL_SHIFT = 0.01
    SHEAR_SCALE = 100.0

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            gravity=(0.0, 0.0, 0.0),
        ),
        profiling_options=gs.options.ProfilingOptions(
            show_FPS=False,
        ),
        show_viewer=show_viewer,
    )

    ground = scene.add_entity(
        gs.morphs.Box(
            size=(0.8, 0.8, GROUND_THICKNESS),
            pos=(0.0, 0.0, -GROUND_THICKNESS / 2),
            fixed=True,
        )
    )
    # Sphere penetrating the ground (center below z=0 by PENETRATION).
    sphere_init_pos = (0.0, 0.0, SPHERE_RADIUS - PENETRATION)
    sphere = scene.add_entity(
        gs.morphs.Sphere(
            radius=SPHERE_RADIUS,
            pos=sphere_init_pos,
            fixed=False,
        )
    )

    probe_local_pos, probe_local_normal = gu.generate_ring_points_on_sphere(
        radius=SPHERE_RADIUS,
        cap_axis=(0.0, 0.0, -1.0),
        n_rings=N_RINGS,
        arc_spacing=2.0 * PROBE_RADIUS,
        return_normals=True,
    )
    normals = torch.as_tensor(probe_local_normal, dtype=gs.tc_float, device=gs.device)
    sensor_kwargs = dict(
        entity_idx=sphere.idx,
        probe_local_pos=probe_local_pos,
        probe_local_normal=probe_local_normal,
        probe_radius=PROBE_RADIUS,
        track_link_idx=(ground.base_link_idx,),
        n_sample_points=800,
        lambda_s=0.0,
        draw_debug=show_viewer,
    )
    dilate_sensor = scene.add_sensor(
        gs.sensors.ElastomerTaxel(
            dilate_scale=1.0,
            shear_scale=0.0,
            **sensor_kwargs,
        )
    )
    shear_sensor = scene.add_sensor(
        gs.sensors.ElastomerTaxel(
            dilate_scale=0.0,
            shear_scale=SHEAR_SCALE,
            **sensor_kwargs,
        )
    )
    combined_sensor = scene.add_sensor(
        gs.sensors.ElastomerTaxel(
            dilate_scale=1.0,
            shear_scale=SHEAR_SCALE,
            **sensor_kwargs,
        )
    )
    assert not dilate_sensor._is_grid and not dilate_sensor._use_grid_fft

    scene.build(n_envs=n_envs)
    scene.step()

    dilate_data = _as_env_batch(dilate_sensor.read_ground_truth(), n_envs)
    shear_data = _as_env_batch(shear_sensor.read_ground_truth(), n_envs)
    combined_data = _as_env_batch(combined_sensor.read_ground_truth(), n_envs)
    normal_projection = (dilate_data * normals).sum(dim=-1)
    assert (normal_projection[..., 0] > tol).all(), "Bottom marker should dilate along its outward normal."
    assert torch.linalg.norm(dilate_data, dim=-1).max() > tol
    assert_allclose(shear_data, 0.0, tol=tol)
    assert_allclose(combined_data, dilate_data, tol=tol)

    sphere.set_pos((LATERAL_SHIFT, 0.0, sphere_init_pos[2]))
    scene.step()

    dilate_data = _as_env_batch(dilate_sensor.read_ground_truth(), n_envs)
    shear_data = _as_env_batch(shear_sensor.read_ground_truth(), n_envs)
    combined_data = _as_env_batch(combined_sensor.read_ground_truth(), n_envs)
    shear_normal_projection = (shear_data * normals).sum(dim=-1)
    shear_tangent = shear_data - shear_normal_projection.unsqueeze(-1) * normals
    assert torch.linalg.norm(shear_tangent, dim=-1).max() > tol
    assert_allclose(shear_normal_projection, 0.0, tol=tol)
    assert_allclose(combined_data, dilate_data + shear_data, tol=5e-5)

    sphere.set_pos((0.0, 0.0, SPHERE_RADIUS + 0.05))
    scene.step()
    assert_equal(combined_sensor.read_ground_truth(), 0.0, err_msg="ElastomerTaxel should be zero with no contact.")


@pytest.mark.required
@pytest.mark.parametrize("n_envs", [0, 2])
def test_elastomer_sensor_grid_box_sphere(show_viewer, tol, n_envs):
    """ElastomerTaxel grid and flat probe layouts should agree on the same flat pad."""
    SPHERE_RADIUS = 0.1
    BOX_SIZE = 0.1
    PENETRATION = 0.01
    GRID_SIZE = (8, 8)
    LATERAL_SHIFT = 0.01
    SHEAR_SCALE = 100.0

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            gravity=(0.0, 0.0, 0.0),
        ),
        profiling_options=gs.options.ProfilingOptions(
            show_FPS=False,
        ),
        show_viewer=show_viewer,
    )
    sphere = scene.add_entity(
        gs.morphs.Sphere(
            radius=SPHERE_RADIUS,
            pos=(0.0, 0.0, SPHERE_RADIUS),
            fixed=True,
        )
    )
    box = scene.add_entity(
        gs.morphs.Box(
            size=(BOX_SIZE, BOX_SIZE, BOX_SIZE),
            pos=(0.0, 0.0, SPHERE_RADIUS * 2 + BOX_SIZE / 2 - PENETRATION),
            fixed=False,
        )
    )
    probe_local_pos = gu.generate_grid_points_on_plane(
        lo=(-BOX_SIZE / 2, -BOX_SIZE / 2, -BOX_SIZE / 2),
        hi=(BOX_SIZE / 2, BOX_SIZE / 2, -BOX_SIZE / 2),
        normal=(0.0, 0.0, -1.0),
        nx=GRID_SIZE[0],
        ny=GRID_SIZE[1],
    )
    sensor_kwargs = dict(
        entity_idx=box.idx,
        probe_local_normal=(0.0, 0.0, -1.0),
        probe_radius=0.02,
        track_link_idx=(sphere.base_link_idx,),
        n_sample_points=600,
        lambda_s=0.0,
        draw_debug=show_viewer,
    )
    elastomer_grid_sensor = scene.add_sensor(
        gs.sensors.ElastomerTaxel(
            probe_local_pos=probe_local_pos,
            dilate_scale=1.0,
            shear_scale=0.0,
            **sensor_kwargs,
        )
    )
    elastomer_sensor = scene.add_sensor(
        gs.sensors.ElastomerTaxel(
            probe_local_pos=probe_local_pos.reshape(-1, 3),
            dilate_scale=1.0,
            shear_scale=0.0,
            **sensor_kwargs,
        )
    )
    shear_sensor = scene.add_sensor(
        gs.sensors.ElastomerTaxel(
            probe_local_pos=probe_local_pos.reshape(-1, 3),
            dilate_scale=0.0,
            shear_scale=SHEAR_SCALE,
            **sensor_kwargs,
        )
    )
    combined_sensor = scene.add_sensor(
        gs.sensors.ElastomerTaxel(
            probe_local_pos=probe_local_pos.reshape(-1, 3),
            dilate_scale=1.0,
            shear_scale=SHEAR_SCALE,
            **sensor_kwargs,
        )
    )
    assert elastomer_grid_sensor._is_grid and elastomer_grid_sensor._use_grid_fft
    assert not elastomer_sensor._is_grid and not elastomer_sensor._use_grid_fft
    assert_allclose(elastomer_sensor.probe_local_pos, elastomer_grid_sensor.probe_local_pos, tol=gs.EPS)

    scene.build(n_envs=n_envs)
    scene.step()

    # Test dilate displacement: grid sensor should match the flat-layout sensor and detect contact magnitude.
    grid_data = elastomer_grid_sensor.read_ground_truth()
    flat_data = elastomer_sensor.read_ground_truth()
    assert_allclose(flat_data, grid_data, tol=tol)
    assert torch.linalg.norm(torch.as_tensor(grid_data, device=gs.device), dim=-1).max() > tol
    assert_allclose(shear_sensor.read_ground_truth(), 0.0, tol=tol)
    assert_allclose(combined_sensor.read_ground_truth(), flat_data, tol=tol)

    # Test combined displacement: dilate + shear contributions should add when the box slides laterally.
    box.set_pos((LATERAL_SHIFT, 0.0, SPHERE_RADIUS * 2 + BOX_SIZE / 2 - PENETRATION))
    scene.step()
    dilate_data = elastomer_sensor.read_ground_truth()
    shear_data = shear_sensor.read_ground_truth()
    combined_data = combined_sensor.read_ground_truth()
    assert torch.linalg.norm(torch.as_tensor(shear_data, device=gs.device), dim=-1).max() > tol
    assert_allclose(combined_data, dilate_data + shear_data, tol=5e-5)

    # Move box away and check no contact.
    box.set_pos((0.0, 0.0, BOX_SIZE + SPHERE_RADIUS * 2 + 0.05))
    scene.step()
    assert_equal(elastomer_grid_sensor.read_ground_truth(), 0.0, err_msg="ElastomerTaxel should be zero in air.")
    assert_equal(combined_sensor.read_ground_truth(), 0.0, err_msg="ElastomerTaxel should be zero in air.")


@pytest.mark.required
@pytest.mark.parametrize("n_envs", [0, 2])
def test_proximity_sensor_box_on_box(show_viewer, tol, n_envs):
    """ProximityTaxel reports a nonzero point-cloud force in contact and near-zero force in air."""
    BOX_SIZE = 0.2
    PENETRATION = 0.01

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            gravity=(0.0, 0.0, 0.0),
        ),
        profiling_options=gs.options.ProfilingOptions(
            show_FPS=False,
        ),
        show_viewer=show_viewer,
    )
    support = scene.add_entity(
        gs.morphs.Box(
            size=(BOX_SIZE, BOX_SIZE, BOX_SIZE),
            pos=(0.0, 0.0, BOX_SIZE / 2),
            fixed=True,
        )
    )
    taxel_box = scene.add_entity(
        gs.morphs.Box(
            size=(BOX_SIZE, BOX_SIZE, BOX_SIZE),
            pos=(0.0, 0.0, BOX_SIZE + BOX_SIZE / 2 - PENETRATION),
            fixed=False,
        )
    )
    sensor = scene.add_sensor(
        gs.sensors.ProximityTaxel(
            entity_idx=taxel_box.idx,
            probe_local_pos=((0.0, 0.0, -BOX_SIZE / 2), (BOX_SIZE / 4, 0.0, -BOX_SIZE / 2)),
            probe_local_normal=(0.0, 0.0, -1.0),
            probe_radius=0.06,
            probe_radius_noise=0.1,
            track_link_idx=(support.base_link_idx,),
            n_sample_points=600,
            stiffness=100.0,
            shear_coupling=0.0,
            draw_debug=show_viewer,
        )
    )

    scene.build(n_envs=n_envs)
    scene.step()

    force_norm = torch.linalg.norm(_as_env_batch(sensor.read_ground_truth().force, n_envs), dim=-1)
    assert (force_norm > tol).all()

    taxel_box.set_pos((0.0, 0.0, BOX_SIZE + BOX_SIZE / 2 + 0.2))
    scene.step()
    force_norm = torch.linalg.norm(_as_env_batch(sensor.read_ground_truth().force, n_envs), dim=-1)
    assert_allclose(force_norm, 0.0, tol=gs.EPS)


@pytest.mark.required
def test_tactile_sensors_heterogeneous_object(show_viewer, tol):
    """Heterogeneous active-env masks should keep tactile readings variant-specific."""
    PAD_SIZE = (0.4, 0.4, 0.1)
    PAD_TOP_Z = PAD_SIZE[2]
    OBJECT_Z_SIZE = 0.16
    BOX_XY_SIZE = 0.28
    SPHERE_RADIUS = 0.08
    PENETRATION = 0.01
    CENTER_PROBE = (0.0, 0.0, PAD_SIZE[2] / 2)
    OUTER_PROBE = (0.13, 0.0, PAD_SIZE[2] / 2)

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            gravity=(0.0, 0.0, 0.0),
        ),
        profiling_options=gs.options.ProfilingOptions(
            show_FPS=False,
        ),
        show_viewer=show_viewer,
    )
    pad = scene.add_entity(
        gs.morphs.Box(
            size=PAD_SIZE,
            pos=(0.0, 0.0, PAD_SIZE[2] / 2),
            fixed=True,
        )
    )
    obj = scene.add_entity(
        morph=[
            gs.morphs.Box(
                size=(BOX_XY_SIZE, BOX_XY_SIZE, OBJECT_Z_SIZE),
            ),
            gs.morphs.Sphere(
                radius=SPHERE_RADIUS,
            ),
        ],
        material=gs.materials.Rigid(
            friction=0.5,
        ),
    )

    probe_local_pos = (CENTER_PROBE, OUTER_PROBE)
    expected_contact = torch.tensor([[True, True], [True, False]], dtype=gs.tc_bool, device=gs.device)
    common = dict(
        entity_idx=pad.idx,
        probe_local_pos=probe_local_pos,
        probe_radius=0.025,
        draw_debug=show_viewer,
    )
    contact_probe = scene.add_sensor(
        gs.sensors.ContactProbe(
            contact_threshold=0.001,
            **common,
        )
    )
    depth_probe = scene.add_sensor(gs.sensors.ContactDepthProbe(**common))
    kinematic_taxel = scene.add_sensor(
        gs.sensors.KinematicTaxel(
            probe_local_normal=(0.0, 0.0, 1.0),
            normal_stiffness=100.0,
            normal_damping=0.0,
            shear_scalar=0.0,
            twist_scalar=0.0,
            **common,
        )
    )
    proximity_taxel = scene.add_sensor(
        gs.sensors.ProximityTaxel(
            probe_local_normal=(0.0, 0.0, 1.0),
            probe_radius=0.04,
            track_link_idx=(obj.base_link_idx,),
            n_sample_points=800,
            stiffness=100.0,
            shear_coupling=0.0,
            **{k: v for k, v in common.items() if k != "probe_radius"},
        )
    )
    elastomer_taxel = scene.add_sensor(
        gs.sensors.ElastomerTaxel(
            probe_local_normal=(0.0, 0.0, 1.0),
            track_link_idx=(obj.base_link_idx,),
            n_sample_points=800,
            **common,
        )
    )
    surface_probe = scene.add_sensor(
        gs.sensors.SurfaceDistanceProbe(
            probe_radius=1.0,
            track_link_idx=(obj.base_link_idx,),
            **{k: v for k, v in common.items() if k != "probe_radius"},
        )
    )

    scene.build(n_envs=2)
    obj.set_pos(
        [
            [0.0, 0.0, PAD_TOP_Z + OBJECT_Z_SIZE / 2 - PENETRATION],
            [0.0, 0.0, PAD_TOP_Z + SPHERE_RADIUS - PENETRATION],
        ]
    )
    scene.step()

    contact = contact_probe.read_ground_truth()
    depth = depth_probe.read_ground_truth()
    kinematic_norm = torch.linalg.norm(kinematic_taxel.read_ground_truth().force, dim=-1)
    proximity_norm = torch.linalg.norm(proximity_taxel.read_ground_truth().force, dim=-1)
    elastomer_norm = torch.linalg.norm(elastomer_taxel.read_ground_truth(), dim=-1)
    surface_distance = surface_probe.read_ground_truth()

    assert_equal(contact, expected_contact)
    assert_equal(depth > 0.001, expected_contact)
    assert_equal(kinematic_norm > tol, expected_contact)
    assert (proximity_norm[0, 0] > tol) and (proximity_norm[1, 0] > tol)
    assert proximity_norm[0, 1] > proximity_norm[1, 1] + tol
    assert (elastomer_norm[0, 0] > tol) and (elastomer_norm[1, 0] > tol)
    assert elastomer_norm[0, 1] > elastomer_norm[1, 1] + gs.EPS
    assert surface_distance[0, 1] < surface_distance[1, 1]


# ------------------------------------------------------------------------------------------
# ----------------------------------- Bulk read API ----------------------------------------
# ------------------------------------------------------------------------------------------


@pytest.mark.required
@pytest.mark.parametrize("n_envs", [0, 3])
def test_read_sensors_bulk_api(show_viewer, n_envs):
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            gravity=(0.0, 0.0, -10.0),
        ),
        profiling_options=gs.options.ProfilingOptions(
            show_FPS=False,
        ),
        show_viewer=show_viewer,
    )
    scene.add_entity(
        morph=gs.morphs.Plane(),
    )
    box_a = scene.add_entity(
        morph=gs.morphs.Box(
            size=(0.1, 0.1, 0.1),
            pos=(0.0, 0.0, 0.2),
        ),
    )
    box_b = scene.add_entity(
        morph=gs.morphs.Box(
            size=(0.1, 0.1, 0.1),
            pos=(0.5, 0.0, 0.2),
        ),
    )

    # Diverse sensor set covering multiple dtypes (float for IMU/ContactForce, bool for Contact, uint8 for the static
    # camera) and heterogeneous per-sensor cache sizes within the float dtype (9 cells for IMU vs 3 for ContactForce).
    # Two IMUs on box_a, one IMU on box_b. ContactForce and Contact sensors on both boxes. A static camera not attached
    # to any entity (entity_idx defaults to -1).
    imu_a1 = scene.add_sensor(
        gs.sensors.IMU(
            entity_idx=box_a.idx,
        ),
    )
    imu_a2 = scene.add_sensor(
        gs.sensors.IMU(
            entity_idx=box_a.idx,
        ),
    )
    imu_b = scene.add_sensor(
        gs.sensors.IMU(
            entity_idx=box_b.idx,
        ),
    )
    force_a = scene.add_sensor(
        gs.sensors.ContactForce(
            entity_idx=box_a.idx,
        ),
    )
    force_b = scene.add_sensor(
        gs.sensors.ContactForce(
            entity_idx=box_b.idx,
        ),
    )
    contact_a = scene.add_sensor(
        gs.sensors.Contact(
            entity_idx=box_a.idx,
        ),
    )
    contact_b = scene.add_sensor(
        gs.sensors.Contact(
            entity_idx=box_b.idx,
        ),
    )
    static_cam = scene.add_sensor(
        gs.sensors.RasterizerCameraOptions(
            res=(32, 32),
        ),
    )

    scene.build(n_envs=n_envs)
    for _ in range(5):
        scene.step()

    # Scene-wide read returns every sensor class. Per-entity reads restrict to classes present on that entity, so the
    # static camera class is excluded from both box_a and box_b reads. Each call allocates a fresh tensor per class.
    scene_data = scene.read_sensors()
    a_data = box_a.read_sensors()
    b_data = box_b.read_sensors()
    assert set(scene_data.keys()) == {
        gs.sensors.types.IMU,
        gs.sensors.types.ContactForce,
        gs.sensors.types.Contact,
        gs.sensors.types.RasterizerCameraOptions,
    }
    assert set(a_data.keys()) == {gs.sensors.types.IMU, gs.sensors.types.ContactForce, gs.sensors.types.Contact}
    assert set(b_data.keys()) == {gs.sensors.types.IMU, gs.sensors.types.ContactForce, gs.sensors.types.Contact}

    # Sensors within a class are sorted by entity_idx, so per-entity reads must match contiguous slices of the
    # scene-wide read.
    for type_tag, a_slice, b_slice in (
        (gs.sensors.types.IMU, slice(0, 18), slice(18, 27)),
        (gs.sensors.types.ContactForce, slice(0, 3), slice(3, 6)),
        (gs.sensors.types.Contact, slice(0, 1), slice(1, 2)),
    ):
        assert_equal(a_data[type_tag], scene_data[type_tag][..., a_slice])
        assert_equal(b_data[type_tag], scene_data[type_tag][..., b_slice])

    # Individual sensor reads must agree with bulk reads at both scene and entity levels.
    # IMU cache layout per sensor is 3 acc + 3 gyro + 3 mag in that order.
    for local_idx, imu in enumerate((imu_a1, imu_a2, imu_b)):
        base = local_idx * 9
        imu_data = imu.read()
        assert_equal(scene_data[gs.sensors.types.IMU][..., base : base + 3], imu_data.lin_acc)
        assert_equal(scene_data[gs.sensors.types.IMU][..., base + 3 : base + 6], imu_data.ang_vel)
        assert_equal(scene_data[gs.sensors.types.IMU][..., base + 6 : base + 9], imu_data.mag)
    for entity_local_idx, imu in enumerate((imu_a1, imu_a2)):
        base = entity_local_idx * 9
        imu_data = imu.read()
        assert_equal(a_data[gs.sensors.types.IMU][..., base : base + 3], imu_data.lin_acc)
        assert_equal(a_data[gs.sensors.types.IMU][..., base + 3 : base + 6], imu_data.ang_vel)
        assert_equal(a_data[gs.sensors.types.IMU][..., base + 6 : base + 9], imu_data.mag)
    # ContactForce returns a 3-vector per sensor.
    for local_idx, force in enumerate((force_a, force_b)):
        base = local_idx * 3
        assert_equal(scene_data[gs.sensors.types.ContactForce][..., base : base + 3], force.read())
    assert_equal(a_data[gs.sensors.types.ContactForce], force_a.read())
    assert_equal(b_data[gs.sensors.types.ContactForce], force_b.read())
    # Contact returns a bool per sensor.
    assert_equal(scene_data[gs.sensors.types.Contact][..., 0:1], contact_a.read())
    assert_equal(scene_data[gs.sensors.types.Contact][..., 1:2], contact_b.read())
    assert_equal(a_data[gs.sensors.types.Contact], contact_a.read())
    assert_equal(b_data[gs.sensors.types.Contact], contact_b.read())

    # `read_sensors` always returns a fresh tensor independent of internal sensor storage. Two successive calls (at
    # the scene level or the entity level) must back onto distinct storage. Verified on both IMU (identity
    # `_post_process`, intermediate-space ring) and Contact (overridden `_post_process`, per-class return-space ring).
    for type_tag in (gs.sensors.types.IMU, gs.sensors.types.Contact):
        scene_a = scene.read_sensors()[type_tag]
        scene_b = scene.read_sensors()[type_tag]
        entity_a = box_a.read_sensors()[type_tag]
        assert scene_a.untyped_storage().data_ptr() != scene_b.untyped_storage().data_ptr()
        assert scene_a.untyped_storage().data_ptr() != entity_a.untyped_storage().data_ptr()
        assert_equal(scene_a, scene_b)

    # Batching must be exercised end-to-end. For n_envs > 0, every per-env row of the bulk view must equal that env's
    # individual sensor read.
    if n_envs > 0:
        for env_idx in range(n_envs):
            assert_equal(scene.read_sensors()[gs.sensors.types.IMU][env_idx, 0:3], imu_a1.read().lin_acc[env_idx])
            assert_equal(scene.read_sensors()[gs.sensors.types.Contact][env_idx, 0:1], contact_a.read()[env_idx])


@pytest.mark.required
@pytest.mark.parametrize("n_envs", [0, 2])
def test_imu_resolution_only_quantizes(show_viewer, n_envs):
    # IMU with only `*_resolution` set (no other noise/delay) returns acceleration components quantized to that
    # resolution.
    RESOLUTION = 0.5
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            gravity=(0.0, 0.0, -10.0),
        ),
        profiling_options=gs.options.ProfilingOptions(
            show_FPS=False,
        ),
        show_viewer=show_viewer,
    )
    scene.add_entity(
        morph=gs.morphs.Plane(),
    )
    box = scene.add_entity(
        morph=gs.morphs.Box(
            size=(0.1, 0.1, 0.1),
            pos=(0.0, 0.0, 0.2),
        ),
    )
    imu = scene.add_sensor(
        gs.sensors.IMU(
            entity_idx=box.idx,
            acc_resolution=RESOLUTION,
        ),
    )
    scene.build(n_envs=n_envs)
    for _ in range(3):
        scene.step()

    measured = imu.read().lin_acc
    remainders = (measured / RESOLUTION) - torch.round(measured / RESOLUTION)
    assert_allclose(remainders, 0.0, tol=gs.EPS)
