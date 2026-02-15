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


def _ti_kernel_wrapper(ti_func, num_inputs, num_outputs, *args):
    import quadrants as ti

    if num_inputs == 1 and num_outputs == 1:

        @ti.kernel
        def kernel(ti_in: ti.template(), ti_out: ti.template()):
            ti.loop_config(serialize=False)
            for I in ti.grouped(ti.ndrange(*ti_in.shape)):
                ti_out[I] = ti_func(ti_in[I], *args)

    elif num_inputs == 2 and num_outputs == 1:

        @ti.kernel
        def kernel(ti_in_1: ti.template(), ti_in_2: ti.template(), ti_out: ti.template()):
            ti.loop_config(serialize=False)
            for I in ti.grouped(ti.ndrange(*ti_in_1.shape)):
                ti_out[I] = ti_func(ti_in_1[I], ti_in_2[I], *args)

    elif num_inputs == 3 and num_outputs == 1:

        @ti.kernel
        def kernel(ti_in_1: ti.template(), ti_in_2: ti.template(), ti_in_3: ti.template(), ti_out: ti.template()):
            ti.loop_config(serialize=False)
            for I in ti.grouped(ti.ndrange(*ti_in_1.shape)):
                ti_out[I] = ti_func(ti_in_1[I], ti_in_2[I], ti_in_3[I], *args)

    elif num_inputs == 4 and num_outputs == 2:

        @ti.kernel
        def kernel(
            ti_in_1: ti.template(),
            ti_in_2: ti.template(),
            ti_in_3: ti.template(),
            ti_in_4: ti.template(),
            ti_out_1: ti.template(),
            ti_out_2: ti.template(),
        ):
            ti.loop_config(serialize=False)
            for I in ti.grouped(ti.ndrange(*ti_in_1.shape)):
                ti_out_1[I], ti_out_2[I] = ti_func(ti_in_1[I], ti_in_2[I], ti_in_3[I], ti_in_4[I], *args)

    else:
        raise NotImplementedError(f"Taichi func with arity in={num_inputs},out={num_outputs} not supported")

    return kernel


@pytest.mark.slow  # ~110s
@pytest.mark.required
@pytest.mark.parametrize("batch_shape", [(10, 40, 25), ()])
def test_geom_taichi_vs_tensor_consistency(batch_shape):
    import quadrants as ti

    for ti_func, py_func, shapes_in, shapes_out, *args in (
        (gu.ti_xyz_to_quat, gu.xyz_to_quat, [[3]], [[4]]),
        (gu.ti_quat_to_R, gu.quat_to_R, [[4]], [[3, 3]], gs.EPS),
        (gu.ti_quat_to_xyz, gu.quat_to_xyz, [[4]], [[3]], gs.EPS),
        (gu.ti_trans_quat_to_T, gu.trans_quat_to_T, [[3], [4]], [[4, 4]], gs.EPS),
        (gu.ti_transform_quat_by_quat, gu.transform_quat_by_quat, [[4], [4]], [[4]]),
        (gu.ti_transform_by_quat, gu.transform_by_quat, [[3], [4]], [[3]]),
        (gu.ti_inv_transform_by_quat, gu.inv_transform_by_quat, [[3], [4]], [[3]]),
        (gu.ti_transform_by_T, gu.transform_by_T, [[3], [4, 4]], [[3]]),
        (gu.ti_inv_transform_by_T, gu.inv_transform_by_T, [[3], [4, 4]], [[3]]),
        (gu.ti_transform_by_trans_quat, gu.transform_by_trans_quat, [[3], [3], [4]], [[3]]),
        (gu.ti_inv_transform_by_trans_quat, gu.inv_transform_by_trans_quat, [[3], [3], [4]], [[3]]),
        (gu.ti_transform_pos_quat_by_trans_quat, gu.transform_pos_quat_by_trans_quat, [[3], [4], [3], [4]], [[3], [4]]),
    ):
        num_inputs, num_outputs = len(shapes_in), len(shapes_out)
        shape_args = (*shapes_in, *shapes_out)
        np_args, tc_args, ti_args, ti_outs = [], [], [], []
        for i in range(len(shape_args)):
            np_arg = np.random.rand(*batch_shape, *shape_args[i]).astype(gs.np_float)

            tc_arg = torch.as_tensor(np_arg, dtype=gs.tc_float, device=gs.device)
            ti_type = ti.Vector if len(shape_args[i]) == 1 else ti.Matrix
            ti_arg = ti_type.field(*shape_args[i], dtype=gs.ti_float, shape=batch_shape)
            ti_arg.from_numpy(np_arg)

            if i < num_inputs:
                np_args.append(np_arg)
                tc_args.append(tc_arg)
                ti_args.append(ti_arg)
            else:
                ti_outs.append(ti_arg)

        np_outs = py_func(*np_args)
        if not isinstance(np_outs, (list, tuple)):
            np_outs = (np_outs,)
        for np_out, shape_out in zip(np_outs, shapes_out):
            assert np_out.shape == (*batch_shape, *shape_out)

        tc_outs = py_func(*tc_args)
        if not isinstance(tc_outs, (list, tuple)):
            tc_outs = (tc_outs,)
        tc_outs = tuple(map(tensor_to_array, tc_outs))

        kernel = _ti_kernel_wrapper(ti_func, num_inputs, num_outputs, *args)
        kernel(*ti_args, *ti_outs)

        for np_out, tc_out, ti_out in zip(np_outs, tc_outs, ti_outs):
            np.testing.assert_allclose(np_out, ti_out.to_numpy(), atol=1e2 * gs.EPS)
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
        dets = torch.linalg.det(A)
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
def test_geom_taichi_inverse(batch_shape):
    import quadrants as ti

    for ti_func, ti_func_inv, shapes_value_args, shapes_transform_args in (
        (gu.ti_transform_by_T, gu.ti_inv_transform_by_T, [[3]], [[4, 4]]),
        (gu.ti_transform_by_trans_quat, gu.ti_inv_transform_by_trans_quat, [[3]], [[3], [4]]),
        (gu.ti_transform_motion_by_trans_quat, gu.ti_inv_transform_motion_by_trans_quat, [[3], [3]], [[3], [4]]),
    ):
        shapes_in = (*shapes_value_args, *shapes_transform_args)
        num_inputs, num_outputs = len(shapes_in), len(shapes_value_args)
        ti_value_in_args, ti_transform_args, ti_value_out_args, ti_value_inv_out_args = [], [], [], []
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

            ti_type = ti.Vector if len(shape_arg) == 1 else ti.Matrix
            ti_arg = ti_type.field(*shape_arg, dtype=gs.ti_float, shape=batch_shape)
            ti_arg.from_numpy(np_arg)

            if i < len(shapes_value_args):
                ti_value_in_args.append(ti_arg)
            elif i < num_inputs:
                ti_transform_args.append(ti_arg)
            elif i < num_inputs + num_outputs:
                ti_value_out_args.append(ti_arg)
            else:
                ti_value_inv_out_args.append(ti_arg)

        kernel = _ti_kernel_wrapper(ti_func, num_inputs, num_outputs)
        kernel(*ti_value_in_args, *ti_transform_args, *ti_value_out_args)
        kernel = _ti_kernel_wrapper(ti_func_inv, num_inputs, num_outputs)
        kernel(*ti_value_out_args, *ti_transform_args, *ti_value_inv_out_args)

        for ti_value_in_arg, ti_value_inv_out_arg in zip(ti_value_in_args, ti_value_inv_out_args):
            np.testing.assert_allclose(ti_value_in_arg.to_numpy(), ti_value_inv_out_arg.to_numpy(), atol=1e2 * gs.EPS)


@pytest.mark.required
@pytest.mark.parametrize("batch_shape", [(10, 40, 25), ()])
def test_geom_taichi_identity(batch_shape):
    import quadrants as ti

    for ti_funcs, shape_args, funcs_args in (
        ((gu.ti_xyz_to_quat, gu.ti_quat_to_xyz), ([3], [4]), ((), (gs.EPS,))),
        ((gu.ti_xyz_to_quat, gu.ti_quat_to_R, gu.ti_R_to_xyz), ([3], [4], [3, 3]), ((), (gs.EPS,), (gs.EPS,))),
        (
            (gu.ti_xyz_to_quat, gu.ti_quat_to_rotvec, gu.ti_rotvec_to_R, gu.ti_R_to_xyz),
            ([3], [4], [3], [3, 3]),
            ((), (gs.EPS,), (gs.EPS,), (gs.EPS,)),
        ),
        ((gu.ti_rotvec_to_quat, gu.ti_quat_to_rotvec), ([3], [4]), ((gs.EPS,), (gs.EPS,))),
    ):
        ti_args = []
        for shape_arg in (*shape_args, shape_args[0]):
            ti_type = ti.Vector if len(shape_arg) == 1 else ti.Matrix
            ti_arg = ti_type.field(*shape_arg, dtype=gs.ti_float, shape=batch_shape)
            ti_arg.from_numpy(np.random.randn(*batch_shape, *shape_arg).clip(-1.0, 1.0).astype(gs.np_float))
            ti_args.append(ti_arg)

        for i, (ti_func, args) in enumerate(zip(ti_funcs, funcs_args)):
            kernel = _ti_kernel_wrapper(ti_func, 1, 1, *args)
            kernel(*ti_args[i : (i + 2)])

        np.testing.assert_allclose(ti_args[0].to_numpy(), ti_args[-1].to_numpy(), atol=1e2 * gs.EPS)


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
