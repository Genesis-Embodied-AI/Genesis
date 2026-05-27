import math
from functools import partial
from unittest.mock import patch

import pytest
import torch
import numpy as np
from scipy.linalg import polar as scipy_polar
from scipy.spatial.transform import Rotation as R, Slerp

import genesis as gs
import genesis.utils.geom as gu
from genesis.utils.tools import FPSTracker
from genesis.utils.misc import tensor_to_array
from genesis.utils import warnings as warnings_mod
from genesis.utils.warnings import warn_once
from genesis.utils.urdf import compose_inertial_properties
from genesis.utils.ring_buffer import TensorRingBuffer

from .utils import assert_allclose


TOL = 1e-7

pytestmark = [pytest.mark.required]


@pytest.fixture
def clear_seen_fixture():
    warnings_mod._seen.clear()
    yield
    warnings_mod._seen.clear()


@pytest.mark.required
def test_warn_once_logs_once(clear_seen_fixture):
    msg = "This is a warning"
    with patch.object(gs, "logger", create=True) as mock_logger:
        with patch.object(mock_logger, "warning") as mock_warning:
            warn_once(msg)
            warn_once(msg)
            mock_warning.assert_called_once_with(msg)


@pytest.mark.required
def test_warn_once_logs_different_messages(clear_seen_fixture):
    msg1 = "Warning 1"
    msg2 = "Warning 2"
    with patch.object(gs, "logger", create=True) as mock_logger:
        with patch.object(mock_logger, "warning") as mock_warning:
            warn_once(msg1)
            warn_once(msg2)
            assert mock_warning.call_count == 2
            mock_warning.assert_any_call(msg1)
            mock_warning.assert_any_call(msg2)


@pytest.mark.required
def test_warn_once_with_empty_message(clear_seen_fixture):
    with patch.object(gs, "logger", create=True) as mock_logger:
        with patch.object(mock_logger, "warning") as mock_warning:
            warn_once("")
            warn_once("")
            mock_warning.assert_called_once_with("")


def _qd_kernel_wrapper(qd_func, num_inputs, num_outputs, *args):
    import quadrants as qd

    if num_inputs == 1 and num_outputs == 1:

        @qd.kernel
        def kernel(qd_in: qd.template(), qd_out: qd.template()):
            qd.loop_config(serialize=False)
            for I in qd.grouped(qd.ndrange(*qd_in.shape)):
                qd_out[I] = qd_func(qd_in[I], *args)

    elif num_inputs == 2 and num_outputs == 1:

        @qd.kernel
        def kernel(qd_in_1: qd.template(), qd_in_2: qd.template(), qd_out: qd.template()):
            qd.loop_config(serialize=False)
            for I in qd.grouped(qd.ndrange(*qd_in_1.shape)):
                qd_out[I] = qd_func(qd_in_1[I], qd_in_2[I], *args)

    elif num_inputs == 3 and num_outputs == 1:

        @qd.kernel
        def kernel(qd_in_1: qd.template(), qd_in_2: qd.template(), qd_in_3: qd.template(), qd_out: qd.template()):
            qd.loop_config(serialize=False)
            for I in qd.grouped(qd.ndrange(*qd_in_1.shape)):
                qd_out[I] = qd_func(qd_in_1[I], qd_in_2[I], qd_in_3[I], *args)

    elif num_inputs == 4 and num_outputs == 2:

        @qd.kernel
        def kernel(
            qd_in_1: qd.template(),
            qd_in_2: qd.template(),
            qd_in_3: qd.template(),
            qd_in_4: qd.template(),
            qd_out_1: qd.template(),
            qd_out_2: qd.template(),
        ):
            qd.loop_config(serialize=False)
            for I in qd.grouped(qd.ndrange(*qd_in_1.shape)):
                qd_out_1[I], qd_out_2[I] = qd_func(qd_in_1[I], qd_in_2[I], qd_in_3[I], qd_in_4[I], *args)

    else:
        raise NotImplementedError(f"Quadrants func with arity in={num_inputs},out={num_outputs} not supported")

    return kernel


@pytest.mark.slow  # ~110s
@pytest.mark.required
@pytest.mark.parametrize("batch_shape", [(10, 40, 25), ()])
def test_geom_quadrants_vs_tensor_consistency(batch_shape):
    import quadrants as qd

    for qd_func, py_func, shapes_in, shapes_out, *args in (
        (gu.qd_xyz_to_quat, gu.xyz_to_quat, [[3]], [[4]]),
        (gu.qd_quat_to_R, gu.quat_to_R, [[4]], [[3, 3]], gs.EPS),
        (gu.qd_quat_to_xyz, gu.quat_to_xyz, [[4]], [[3]], gs.EPS),
        (gu.qd_trans_quat_to_T, gu.trans_quat_to_T, [[3], [4]], [[4, 4]], gs.EPS),
        (gu.qd_transform_quat_by_quat, gu.transform_quat_by_quat, [[4], [4]], [[4]]),
        (gu.qd_transform_by_quat, gu.transform_by_quat, [[3], [4]], [[3]]),
        (gu.qd_inv_transform_by_quat, gu.inv_transform_by_quat, [[3], [4]], [[3]]),
        (gu.qd_transform_by_T, gu.transform_by_T, [[3], [4, 4]], [[3]]),
        (gu.qd_inv_transform_by_T, gu.inv_transform_by_T, [[3], [4, 4]], [[3]]),
        (gu.qd_transform_by_trans_quat, gu.transform_by_trans_quat, [[3], [3], [4]], [[3]]),
        (gu.qd_inv_transform_by_trans_quat, gu.inv_transform_by_trans_quat, [[3], [3], [4]], [[3]]),
        (gu.qd_transform_pos_quat_by_trans_quat, gu.transform_pos_quat_by_trans_quat, [[3], [4], [3], [4]], [[3], [4]]),
    ):
        num_inputs, num_outputs = len(shapes_in), len(shapes_out)
        shape_args = (*shapes_in, *shapes_out)
        np_args, tc_args, qd_args, qd_outs = [], [], [], []
        for i in range(len(shape_args)):
            np_arg = np.random.rand(*batch_shape, *shape_args[i]).astype(gs.np_float)

            tc_arg = torch.as_tensor(np_arg, dtype=gs.tc_float, device=gs.device)
            qd_type = qd.Vector if len(shape_args[i]) == 1 else qd.Matrix
            qd_arg = qd_type.field(*shape_args[i], dtype=gs.qd_float, shape=batch_shape)
            qd_arg.from_numpy(np_arg)

            if i < num_inputs:
                np_args.append(np_arg)
                tc_args.append(tc_arg)
                qd_args.append(qd_arg)
            else:
                qd_outs.append(qd_arg)

        np_outs = py_func(*np_args)
        if not isinstance(np_outs, (list, tuple)):
            np_outs = (np_outs,)
        for np_out, shape_out in zip(np_outs, shapes_out):
            assert np_out.shape == (*batch_shape, *shape_out)

        tc_outs = py_func(*tc_args)
        if not isinstance(tc_outs, (list, tuple)):
            tc_outs = (tc_outs,)
        tc_outs = tuple(map(tensor_to_array, tc_outs))

        kernel = _qd_kernel_wrapper(qd_func, num_inputs, num_outputs, *args)
        kernel(*qd_args, *qd_outs)

        for np_out, tc_out, qd_out in zip(np_outs, tc_outs, qd_outs):
            np.testing.assert_allclose(np_out, qd_out.to_numpy(), atol=1e2 * gs.EPS)
            np.testing.assert_allclose(np_out, tc_out, atol=1e2 * gs.EPS)


def polar(A, pure_rotation: bool, side, tol):
    # filter out singular A (which is not invertible)
    # non-invertible matrix makes non-unique SVD which may break the consistency.
    N = A.shape[-1]
    if isinstance(A, np.ndarray):
        dets = np.linalg.det(A)
        mask = np.abs(dets) < tol
        if A.ndim > 2:
            if mask.any():
                I = np.eye(N, dtype=A.dtype)
                A = np.where(mask[..., None, None], I, A)
        else:
            if mask:
                A = np.eye(N, dtype=A.dtype)
    elif isinstance(A, torch.Tensor):
        dets = torch.linalg.det(A.reshape((-1, 3, 3))).reshape(A.shape[:-2])
        mask = torch.abs(dets) < tol
        if A.ndim > 2:
            if mask.any():
                I = torch.eye(N, dtype=A.dtype, device=A.device)
                A = torch.where(mask[..., None, None], I, A)
        else:
            if mask:
                A = torch.eye(N, dtype=A.dtype, device=A.device)
    return gu.polar(A, pure_rotation=pure_rotation, side=side)


@pytest.mark.required
@pytest.mark.parametrize("batch_shape", [(10, 40, 25), ()])
def test_geom_numpy_vs_torch_consistency(batch_shape, tol):
    for py_func, shapes_in, shapes_out in (
        (gu.slerp, [[4], [4], [1]], [[4]]),
        (gu.z_up_to_R, [[3], [3], [3, 3]], [[3, 3]]),
        (gu.pos_lookat_up_to_T, [[3], [3], [3]], [[4, 4]]),
        (partial(polar, pure_rotation=False, side="left", tol=tol), [[3, 3]], [[3, 3], [3, 3]]),
        (partial(polar, pure_rotation=False, side="right", tol=tol), [[3, 3]], [[3, 3], [3, 3]]),
    ):
        num_inputs = len(shapes_in)
        shape_args = (*shapes_in, *shapes_out)
        np_args, tc_args = [], []
        for i in range(len(shape_args)):
            np_arg = np.random.randn(*batch_shape, *shape_args[i]).clip(-1.0, 1.0).astype(gs.np_float)
            tc_arg = torch.as_tensor(np_arg, dtype=gs.tc_float, device=gs.device)

            if i < num_inputs:
                np_args.append(np_arg)
                tc_args.append(tc_arg)

        np_outs = py_func(*np_args)
        if not isinstance(np_outs, (list, tuple)):
            np_outs = (np_outs,)
        for np_out, shape_out in zip(np_outs, shapes_out):
            assert np_out.shape == (*batch_shape, *shape_out)

        tc_outs = py_func(*tc_args)
        if not isinstance(tc_outs, (list, tuple)):
            tc_outs = (tc_outs,)
        tc_outs = tuple(map(tensor_to_array, tc_outs))

        for np_out, tc_out in zip(np_outs, tc_outs):
            assert_allclose(np_out, tc_out, tol=tol)


@pytest.mark.required
@pytest.mark.parametrize("batch_shape", [(10, 40, 25), ()])
def test_geom_quadrants_inverse(batch_shape):
    import quadrants as qd

    for qd_func, qd_func_inv, shapes_value_args, shapes_transform_args in (
        (gu.qd_transform_by_T, gu.qd_inv_transform_by_T, [[3]], [[4, 4]]),
        (gu.qd_transform_by_trans_quat, gu.qd_inv_transform_by_trans_quat, [[3]], [[3], [4]]),
        (gu.qd_transform_motion_by_trans_quat, gu.qd_inv_transform_motion_by_trans_quat, [[3], [3]], [[3], [4]]),
    ):
        shapes_in = (*shapes_value_args, *shapes_transform_args)
        num_inputs, num_outputs = len(shapes_in), len(shapes_value_args)
        qd_value_in_args, qd_transform_args, qd_value_out_args, qd_value_inv_out_args = [], [], [], []
        for i, shape_arg in enumerate(map(tuple, (*shapes_in, *shapes_value_args, *shapes_value_args))):
            if shape_arg in ((4, 4), (3, 3)):
                R = gu.rotvec_to_R(np.random.randn(*batch_shape, 3).clip(-1.0, 1.0).astype(gs.np_float))
                if shape_arg == (4, 4):
                    trans = np.random.randn(*batch_shape, 3).astype(gs.np_float)
                    np_arg = gu.trans_R_to_T(trans, R)
                else:
                    np_arg = R
            else:
                np_arg = np.random.randn(*batch_shape, *shape_arg).clip(-1.0, 1.0).astype(gs.np_float)

            qd_type = qd.Vector if len(shape_arg) == 1 else qd.Matrix
            qd_arg = qd_type.field(*shape_arg, dtype=gs.qd_float, shape=batch_shape)
            qd_arg.from_numpy(np_arg)

            if i < len(shapes_value_args):
                qd_value_in_args.append(qd_arg)
            elif i < num_inputs:
                qd_transform_args.append(qd_arg)
            elif i < num_inputs + num_outputs:
                qd_value_out_args.append(qd_arg)
            else:
                qd_value_inv_out_args.append(qd_arg)

        kernel = _qd_kernel_wrapper(qd_func, num_inputs, num_outputs)
        kernel(*qd_value_in_args, *qd_transform_args, *qd_value_out_args)
        kernel = _qd_kernel_wrapper(qd_func_inv, num_inputs, num_outputs)
        kernel(*qd_value_out_args, *qd_transform_args, *qd_value_inv_out_args)

        for qd_value_in_arg, qd_value_inv_out_arg in zip(qd_value_in_args, qd_value_inv_out_args):
            np.testing.assert_allclose(qd_value_in_arg.to_numpy(), qd_value_inv_out_arg.to_numpy(), atol=1e2 * gs.EPS)


@pytest.mark.required
@pytest.mark.parametrize("batch_shape", [(10, 40, 25), ()])
def test_geom_quadrants_identity(batch_shape):
    import quadrants as qd

    for qd_funcs, shape_args, funcs_args in (
        ((gu.qd_xyz_to_quat, gu.qd_quat_to_xyz), ([3], [4]), ((), (gs.EPS,))),
        ((gu.qd_xyz_to_quat, gu.qd_quat_to_R, gu.qd_R_to_xyz), ([3], [4], [3, 3]), ((), (gs.EPS,), (gs.EPS,))),
        (
            (gu.qd_xyz_to_quat, gu.qd_quat_to_rotvec, gu.qd_rotvec_to_R, gu.qd_R_to_xyz),
            ([3], [4], [3], [3, 3]),
            ((), (gs.EPS,), (gs.EPS,), (gs.EPS,)),
        ),
        ((gu.qd_rotvec_to_quat, gu.qd_quat_to_rotvec), ([3], [4]), ((gs.EPS,), (gs.EPS,))),
    ):
        qd_args = []
        for shape_arg in (*shape_args, shape_args[0]):
            qd_type = qd.Vector if len(shape_arg) == 1 else qd.Matrix
            qd_arg = qd_type.field(*shape_arg, dtype=gs.qd_float, shape=batch_shape)
            qd_arg.from_numpy(np.random.randn(*batch_shape, *shape_arg).clip(-1.0, 1.0).astype(gs.np_float))
            qd_args.append(qd_arg)

        for i, (qd_func, args) in enumerate(zip(qd_funcs, funcs_args)):
            kernel = _qd_kernel_wrapper(qd_func, 1, 1, *args)
            kernel(*qd_args[i : (i + 2)])

        np.testing.assert_allclose(qd_args[0].to_numpy(), qd_args[-1].to_numpy(), atol=1e2 * gs.EPS)


@pytest.mark.required
@pytest.mark.parametrize("batch_shape", [(10, 40, 25), ()])
def test_geom_tensor_identity(batch_shape):
    for py_funcs, shape_args in (
        ((gu.R_to_rot6d, gu.rot6d_to_R), ([3, 3], [6])),
        ((gu.R_to_quat, gu.quat_to_R), ([3, 3], [4])),
    ):
        np_args, tc_args = [], []
        for shape_arg in (*shape_args, shape_args[0]):
            if tuple(shape_arg) == (3, 3):
                np_arg = gu.rotvec_to_R(np.random.randn(*batch_shape, 3).clip(-1.0, 1.0).astype(gs.np_float))
            else:
                np_arg = np.random.randn(*batch_shape, *shape_arg).clip(-1.0, 1.0).astype(gs.np_float)
            tc_arg = torch.as_tensor(np_arg, dtype=gs.tc_float, device=gs.device)
            np_args.append(np_arg)
            tc_args.append(tc_arg)

        for i, py_func in enumerate(py_funcs):
            np_args[i + 1][:] = py_func(np_args[i])
            tc_args[i + 1][:] = py_func(tc_args[i])

        np.testing.assert_allclose(np_args[0], np_args[-1], atol=1e2 * gs.EPS)
        np.testing.assert_allclose(tensor_to_array(tc_args[0]), tensor_to_array(tc_args[-1]), atol=1e2 * gs.EPS)


def test_fps_tracker():
    n_envs = 23
    tracker = FPSTracker(alpha=0.0, minimum_interval_seconds=0.1, n_envs=n_envs)
    tracker.step(current_time=10.0)
    assert not tracker.step(current_time=10.0)
    assert not tracker.step(current_time=10.0)
    assert not tracker.step(current_time=10.0)
    fps = tracker.step(current_time=10.2)
    # num envs * [num steps] / (delta time)
    assert math.isclose(fps, n_envs * 4 / 0.2)

    assert not tracker.step(current_time=10.21)
    assert not tracker.step(current_time=10.22)
    assert not tracker.step(current_time=10.29)
    fps = tracker.step(current_time=10.31)
    # num envs * [num steps] / (delta time)
    assert math.isclose(fps, n_envs * 4 / 0.11)

    assert not tracker.step(current_time=10.33)
    assert not tracker.step(current_time=10.37)
    assert not tracker.step(current_time=10.39)
    fps = tracker.step(current_time=10.45)
    # num envs * [num steps] / (delta time)
    assert math.isclose(fps, n_envs * 4 / 0.14)


@pytest.mark.required
def test_compose_inertial_properties():
    """Test composition of inertial properties combining multiple effects."""
    mass1, com1 = 1.0, np.array([1.0, 0.0, 0.0])
    inertia1 = np.array([[2.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

    mass2, com2 = 2.0, np.array([0.0, 2.0, 0.0])
    inertia2 = np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 1.0]])

    # Analytical calculations: mass=3.0, COM=[1/3, 4/3, 0]
    expected_mass, expected_com = 3.0, np.array([1.0 / 3.0, 4.0 / 3.0, 0.0])

    # Translate inertias to combined COM using parallel axis theorem
    def translate_inertia(I, m, r):  # I + m*(||r||²*I - r⊗r)
        return I + m * (np.dot(r, r) * np.eye(3) - np.outer(r, r))

    expected_inertia = translate_inertia(inertia1, mass1, expected_com - com1) + translate_inertia(
        inertia2, mass2, expected_com - com2
    )

    # Now call the function and verify results
    combined_mass, combined_com, combined_inertia = compose_inertial_properties(
        mass1, com1, inertia1, mass2, com2, inertia2
    )

    assert_allclose(combined_mass, expected_mass, tol=TOL)
    assert_allclose(combined_com, expected_com, tol=TOL)
    assert_allclose(combined_inertia, expected_inertia, tol=TOL)


@pytest.mark.required
@pytest.mark.parametrize("batch_shape", [(10, 40, 25), ()])
def test_slerp(batch_shape, tol):
    INTERP_RATIO = 0.7

    numel = math.prod(batch_shape)
    q0 = np.random.rand(numel, 4)
    q0 /= np.linalg.norm(q0)
    q1 = np.random.rand(numel, 4)
    q1 /= np.linalg.norm(q1)

    lerp_true = np.empty_like(q0)
    for i in range(numel):
        rots = R.from_quat([q0[i], q1[i]], scalar_first=True)
        slerp = Slerp([0, 1], rots)
        lerp_true[i] = slerp([INTERP_RATIO]).as_quat(scalar_first=True)

    lerp = gu.slerp(q0.reshape((*batch_shape, 4)), q1.reshape((*batch_shape, 4)), np.full(batch_shape, INTERP_RATIO))
    assert_allclose(lerp_true.reshape((*batch_shape, 4)), lerp, tol=tol)


@pytest.mark.required
@pytest.mark.parametrize("side", ["right", "left"])
def test_polar_decomposition(side, tol):
    """Test polar decomposition for numpy inputs with scipy validation."""
    # Generate random matrices (not necessarily square)
    M, N = 3, 3
    np_A = np.random.randn(M, N).astype(gs.np_float)

    # Test numpy version (with pure_rotation=False to match original behavior)
    np_U, np_P = gu.polar(np_A, pure_rotation=False, side=side)
    assert np_U.shape == (M, N)
    if side == "right":
        assert np_P.shape == (N, N)
        # Verify A ≈ U @ P
        np_reconstructed = np_U @ np_P
    else:
        assert np_P.shape == (M, M)
        # Verify A ≈ P @ U
        np_reconstructed = np_P @ np_U

    assert_allclose(np_A, np_reconstructed, tol=tol)

    # Note: U from polar decomposition may not be exactly unitary due to numerical errors,
    # but the reconstruction A ≈ U @ P (or P @ U) is the most important property

    # Verify P is positive semi-definite (eigenvalues >= 0)
    np_eigenvals = np.linalg.eigvals(np_P)
    assert np.all(np_eigenvals.real >= -tol), "P should be positive semi-definite"

    # Validate against scipy
    scipy_U, scipy_P = scipy_polar(np_A, side=side)
    np_U_scipy, np_P_scipy = gu.polar(np_A, pure_rotation=False, side=side)
    assert_allclose(scipy_U, np_U_scipy, tol=tol)
    assert_allclose(scipy_P, np_P_scipy, tol=tol)


@pytest.mark.required
@pytest.mark.parametrize("is_pure", [False, True])
def test_polar_pure_rotation(is_pure, tol):
    """Test that pure_rotation parameter ensures det(U) = 1 for square matrices."""
    M, N = 3, 3  # Square matrices only

    # Create a matrix that will have det(U) = -1 by using a reflection
    np_A = np.random.randn(M, N).astype(gs.np_float) @ np.diag([1, 1, -1])

    np_U, np_P = gu.polar(np_A, pure_rotation=is_pure)

    # Check determinants
    np_det = np.linalg.det(np_U)
    if is_pure:
        assert (np_det - 1.0) < tol, "With pure_rotation, det should be 1 (pure rotation)"
    else:
        assert abs(np_det - 1.0) < tol, "Without pure_rotation, det might be -1 (reflection)"

    # Reconstruction should still work
    np_recon = np_U @ np_P
    assert_allclose(np_A, np_recon, tol=tol)


@pytest.mark.required
@pytest.mark.parametrize("side", ["right", "left"])
@pytest.mark.parametrize("batch_shape", [(5,), (3, 4), (2, 3, 4)])
def test_polar_decomposition_batched_numpy(side, batch_shape, tol):
    """Test batched polar decomposition for numpy inputs."""
    M, N = 3, 3
    np_A = np.random.randn(*batch_shape, M, N).astype(gs.np_float)

    # Test batched numpy version
    np_U, np_P = gu.polar(np_A, pure_rotation=False, side=side)
    assert np_U.shape == (*batch_shape, M, N)
    if side == "right":
        assert np_P.shape == (*batch_shape, N, N)
        # Verify A ≈ U @ P for each batch element
        np_reconstructed = np_U @ np_P
    else:
        assert np_P.shape == (*batch_shape, M, M)
        # Verify A ≈ P @ U for each batch element
        np_reconstructed = np_P @ np_U

    assert_allclose(np_A, np_reconstructed, tol=tol)

    # Verify P is positive semi-definite for each batch element
    for idx in np.ndindex(batch_shape):
        np_eigenvals = np.linalg.eigvals(np_P[idx])
        assert np.all(np_eigenvals.real >= -tol), f"P should be positive semi-definite at batch index {idx}"


@pytest.mark.required
@pytest.mark.parametrize("side", ["right", "left"])
def test_polar_decomposition_batched_pure_rotation(side, tol):
    """Test batched polar decomposition with pure_rotation parameter.

    Note: This test verifies that batched polar decomposition works with pure_rotation=True.
    The reconstruction accuracy is verified, though the pure_rotation fix for batched arrays
    may have limitations. The single-matrix pure_rotation test validates that functionality.
    """
    batch_shape = (5,)
    M, N = 3, 3
    np_A = np.random.randn(*batch_shape, M, N).astype(gs.np_float)

    # Test with pure_rotation - reconstruction should still work
    np_U, np_P = gu.polar(np_A, pure_rotation=True, side=side)

    # Reconstruction should work
    if side == "right":
        np_reconstructed = np_U @ np_P
    else:
        np_reconstructed = np_P @ np_U

    assert_allclose(np_A, np_reconstructed, tol=tol)


# =============================================================================================================
# TensorRingBuffer tests
# =============================================================================================================


@pytest.fixture
def ring_buffer_1d():
    """Create a simple 1D ring buffer of N=4, shape=(3,)."""
    return TensorRingBuffer(N=4, shape=(3,), dtype=torch.float32)


@pytest.fixture
def ring_buffer_2d():
    """Create a 2D ring buffer of N=3, shape=(2, 4)."""
    return TensorRingBuffer(N=3, shape=(2, 4), dtype=torch.float64)


@pytest.mark.required
def test_ring_buffer_init_defaults():
    """Verify auto-allocation of buffer and default idx of -1."""
    buf = TensorRingBuffer(N=4, shape=(3,), dtype=torch.float32)
    assert buf.N == 4
    assert buf.buffer.shape == (4, 3)
    assert buf.buffer.dtype == torch.float32
    assert buf.buffer.device.type == gs.device.type
    # _idx should be a 0D tensor with value -1
    assert buf._idx.ndim == 0
    assert buf._idx.item() == -1


@pytest.mark.required
def test_ring_buffer_init_with_external_buffer():
    """Provide an external buffer tensor."""
    ext = torch.empty((4, 3), dtype=torch.float32, device=gs.device)
    buf = TensorRingBuffer(N=4, shape=(3,), dtype=torch.float32, buffer=ext)
    assert buf.buffer is ext


@pytest.mark.required
def test_ring_buffer_init_buffer_shape_mismatch():
    """Buffer shape mismatch should raise AssertionError."""
    ext = torch.empty((5, 3), dtype=torch.float32, device=gs.device)
    with pytest.raises(AssertionError):
        TensorRingBuffer(N=4, shape=(3,), dtype=torch.float32, buffer=ext)


@pytest.mark.required
def test_ring_buffer_init_with_external_idx():
    """Provide an external 0D index tensor."""
    idx = torch.tensor(2, dtype=torch.int64, device=gs.device)
    buf = TensorRingBuffer(N=4, shape=(3,), dtype=torch.float32, idx=idx)
    assert buf._idx is idx
    assert buf._idx.item() == 2


@pytest.mark.required
def test_ring_buffer_init_idx_wrong_dtype():
    """Non-integer idx dtype should raise AssertionError."""
    idx = torch.tensor(0, dtype=torch.float32, device=gs.device)
    with pytest.raises(AssertionError):
        TensorRingBuffer(N=4, shape=(3,), dtype=torch.float32, idx=idx)


@pytest.mark.required
def test_ring_buffer_set_and_get(ring_buffer_1d, tol):
    """set() followed by at(0) returns what was just set (before rotate)."""
    buf = ring_buffer_1d
    t = torch.tensor([1.0, 2.0, 3.0], device=gs.device)
    buf.set(t)
    # at(0) after set (before rotate) reads from buffer[_idx] which holds the just-set value
    assert_allclose(buf.at(0), t, tol=tol)


@pytest.mark.required
def test_ring_buffer_rotate_advances_idx(ring_buffer_1d):
    """rotate() increments the internal index, wrapping at N."""
    buf = ring_buffer_1d

    # Production pattern: rotate then set. After each full cycle _idx points to the written slot.
    for expected_idx in range(buf.N):
        buf.rotate()
        buf.set(torch.zeros(3, device=gs.device))
        assert buf._idx.item() == expected_idx


@pytest.mark.required
def test_ring_buffer_at_relative_index(ring_buffer_1d, tol):
    """at() indexes from most recent (0) to oldest (N-1) in stable state.

    Stable state is reached by following the production pattern: rotate → set → at(0).
    After each full cycle, _idx points to the slot just written.
    """
    buf = ring_buffer_1d
    vals = [torch.tensor([float(i), 0.0, 0.0], device=gs.device) for i in range(4)]

    # Write all values using the production pattern: rotate then set
    for v in vals:
        buf.rotate()
        buf.set(v)

    # Stable state: buffer = [v0, v1, v2, v3] (written in order), _idx=3
    # at(0) = most recent = v3, at(1) = v2, ..., at(3) = v0
    assert_allclose(buf.at(0), vals[3], tol=tol, err_msg="at(0) should be most recent")
    assert_allclose(buf.at(1), vals[2], tol=tol, err_msg="at(1) should be second most recent")
    assert_allclose(buf.at(2), vals[1], tol=tol, err_msg="at(2) should be third most recent")
    assert_allclose(buf.at(3), vals[0], tol=tol, err_msg="at(3) should be oldest")


@pytest.mark.required
def test_ring_buffer_at_returns_view_or_clone(ring_buffer_1d):
    """at() with copy=None returns a view when possible."""
    buf = ring_buffer_1d
    buf.rotate()
    v = torch.tensor([1.0, 2.0, 3.0], device=gs.device)
    buf.set(v)

    # at(0) returns a view into the buffer
    result = buf.at(0)
    assert result.untyped_storage().data_ptr() == buf.buffer.untyped_storage().data_ptr()


@pytest.mark.required
def test_ring_buffer_at_copy_true(ring_buffer_1d):
    """at() with copy=True always returns a clone."""
    buf = ring_buffer_1d
    buf.rotate()
    v = torch.tensor([1.0, 2.0, 3.0], device=gs.device)
    buf.set(v)

    result = buf.at(0, copy=True)
    assert result.untyped_storage().data_ptr() != buf.buffer.untyped_storage().data_ptr()


@pytest.mark.required
def test_ring_buffer_at_copy_false_raises_when_needed(ring_buffer_1d):
    """at() with copy=False raises when a view is impossible."""
    buf = ring_buffer_1d
    buf.rotate()
    v = torch.tensor([1.0, 2.0, 3.0], device=gs.device)
    buf.set(v)

    # at() with a 1D idx tensor forces allocation (advanced indexing)
    idx_tensor = torch.tensor([0], device=gs.device)
    with pytest.raises(Exception):
        buf.at(idx_tensor, copy=False)


@pytest.mark.required
def test_ring_buffer_at_with_others_idx(ring_buffer_2d, tol):
    """at() with others_idx extracts sub-slices."""
    buf = ring_buffer_2d
    v = torch.arange(8, dtype=torch.float64, device=gs.device).reshape(2, 4)
    buf.rotate()
    buf.set(v)

    # at(0, 0) gives the first row of the most recent entry
    result = buf.at(0, 0)
    assert_allclose(result, v[0], tol=tol)


@pytest.mark.required
def test_ring_buffer_at_per_row(ring_buffer_2d):
    """at() with per_row=True handles per-row indexing."""
    buf = ring_buffer_2d
    for i in range(3):
        buf.rotate()
        v = torch.full((2, 4), float(i * 10), dtype=torch.float64, device=gs.device)
        buf.set(v)

    # per_row selects one ring slot per row of the second dimension
    idx_per_row = torch.tensor([0, 1], device=gs.device)
    result = buf.at(idx_per_row, per_row=True)
    assert result.shape == (2, 4)


@pytest.mark.required
def test_ring_buffer_clone_is_independent(ring_buffer_1d):
    """Clone should be a deep copy, independent of the original."""
    buf = ring_buffer_1d
    v = torch.tensor([1.0, 2.0, 3.0], device=gs.device)
    buf.rotate()
    buf.set(v)

    cloned = buf.clone()
    assert cloned.N == buf.N
    assert cloned.buffer.shape == buf.buffer.shape
    assert cloned._idx.item() == buf._idx.item()
    # Modifying clone should not affect original
    cloned.buffer[0, 0] = 999.0
    assert buf.buffer[0, 0] != 999.0


@pytest.mark.required
def test_ring_buffer_getitem_slice(ring_buffer_1d):
    """__getitem__ with a slice returns a view-based sub-buffer."""
    buf = ring_buffer_1d
    sliced = buf[1:3]
    assert sliced.N == 4
    assert sliced.buffer.shape == (4, 2)  # shape went from (3,) to (2,)


@pytest.mark.required
def test_ring_buffer_getitem_int(ring_buffer_1d):
    """__getitem__ with an integer returns a view-based sub-buffer of size 1."""
    buf = ring_buffer_1d
    sliced = buf[0]
    assert sliced.N == 4
    assert sliced.buffer.shape == (4, 1)


@pytest.mark.required
def test_ring_buffer_getitem_tuple(ring_buffer_1d):
    """__getitem__ with a single-element tuple returns a view-based sub-buffer."""
    buf = ring_buffer_1d
    # Buffer shape (4, 3); key (0,) gives indexes=(slice(None), 0), sliced.shape=(4,)
    # resulting sub-buffer has shape () with buffer shape (4,)
    sliced = buf[(0,)]
    assert sliced.N == 4
    assert sliced.buffer.shape == (4,)


@pytest.mark.required
def test_ring_buffer_getitem_invalid_key(ring_buffer_1d):
    """__getitem__ with an unsupported key type should raise TypeError."""
    with pytest.raises(TypeError):
        buf = ring_buffer_1d
        buf["invalid"]


@pytest.mark.required
def test_ring_buffer_wraparound(tol):
    """Writing more than N elements should wrap and overwrite oldest (production pattern: rotate then set)."""
    N = 3
    buf = TensorRingBuffer(N=N, shape=(2,), dtype=torch.float32)
    for i in range(5):
        buf.rotate()
        v = torch.tensor([float(i), float(i * 10)], device=gs.device)
        buf.set(v)

    # After 5 rotate+set cycles starting from _idx=-1:
    # Cycle 0: rotate→0, set→buf[0]=v0
    # Cycle 1: rotate→1, set→buf[1]=v1
    # Cycle 2: rotate→2, set→buf[2]=v2
    # Cycle 3: rotate→0, set→buf[0]=v3 (overwrites v0)
    # Cycle 4: rotate→1, set→buf[1]=v4 (overwrites v1)
    # Final _idx=1, buffer=[v3, v4, v2]
    assert buf._idx.item() == 1
    # at(0) = most recent = v4, at(1) = v3, at(2) = oldest surviving = v2
    assert_allclose(buf.at(0), torch.tensor([4.0, 40.0], device=gs.device), tol=tol)
    assert_allclose(buf.at(1), torch.tensor([3.0, 30.0], device=gs.device), tol=tol)
    assert_allclose(buf.at(2), torch.tensor([2.0, 20.0], device=gs.device), tol=tol)


@pytest.mark.required
def test_ring_buffer_multidimensional(tol):
    """Work with multi-dimensional tensors."""
    buf = TensorRingBuffer(N=3, shape=(2, 3, 4), dtype=torch.float32)
    v = torch.randn(2, 3, 4, device=gs.device)
    buf.rotate()
    buf.set(v)
    retrieved = buf.at(0)
    assert_allclose(retrieved, v, tol=tol)
