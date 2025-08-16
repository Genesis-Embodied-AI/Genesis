from unittest.mock import patch

import pytest
import torch
import numpy as np
from scipy.spatial.transform import Rotation

import genesis as gs
import genesis.utils.geom as gu
from genesis.utils.misc import tensor_to_array
from genesis.utils import warnings as warnings_mod
from genesis.utils.warnings import warn_once

from .utils import (
    assert_allclose,
)


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


def _ti_kernel_wrapper(ti_func, num_inputs, num_outputs):
    import gstaichi as ti

    if num_inputs == 1 and num_outputs == 1:

        @ti.kernel
        def kernel(ti_in: ti.template(), ti_out: ti.template()):
            ti.loop_config(serialize=False)
            for I in ti.grouped(ti.ndrange(*ti_in.shape)):
                ti_out[I] = ti_func(ti_in[I])

    elif num_inputs == 2 and num_outputs == 1:

        @ti.kernel
        def kernel(ti_in_1: ti.template(), ti_in_2: ti.template(), ti_out: ti.template()):
            ti.loop_config(serialize=False)
            for I in ti.grouped(ti.ndrange(*ti_in_1.shape)):
                ti_out[I] = ti_func(ti_in_1[I], ti_in_2[I])

    elif num_inputs == 3 and num_outputs == 1:

        @ti.kernel
        def kernel(ti_in_1: ti.template(), ti_in_2: ti.template(), ti_in_3: ti.template(), ti_out: ti.template()):
            ti.loop_config(serialize=False)
            for I in ti.grouped(ti.ndrange(*ti_in_1.shape)):
                ti_out[I] = ti_func(ti_in_1[I], ti_in_2[I], ti_in_3[I])

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
                ti_out_1[I], ti_out_2[I] = ti_func(ti_in_1[I], ti_in_2[I], ti_in_3[I], ti_in_4[I])

    else:
        raise NotImplementedError(f"Taichi func with arity in={num_inputs},out={num_outputs} not supported")

    return kernel


@pytest.mark.required
@pytest.mark.parametrize("batch_shape", [(10, 40, 25), ()])
def test_utils_geom_taichi_vs_tensor_consistency(batch_shape):
    import gstaichi as ti

    for ti_func, py_func, shapes_in, shapes_out in (
        (gu.ti_xyz_to_quat, gu.xyz_to_quat, [[3]], [[4]]),
        (gu.ti_quat_to_R, gu.quat_to_R, [[4]], [[3, 3]]),
        (gu.ti_quat_to_xyz, gu.quat_to_xyz, [[4]], [[3]]),
        (gu.ti_trans_quat_to_T, gu.trans_quat_to_T, [[3], [4]], [[4, 4]]),
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

        kernel = _ti_kernel_wrapper(ti_func, num_inputs, num_outputs)
        kernel(*ti_args, *ti_outs)

        for np_out, tc_out, ti_out in zip(np_outs, tc_outs, ti_outs):
            np.testing.assert_allclose(np_out, ti_out.to_numpy(), atol=1e2 * gs.EPS)
            np.testing.assert_allclose(np_out, tc_out, atol=1e2 * gs.EPS)


@pytest.mark.required
@pytest.mark.parametrize("batch_shape", [(10, 40, 25), ()])
def test_utils_geom_numpy_vs_tensor_consistency(batch_shape, tol):
    for py_func, shapes_in, shapes_out in (
        (gu.z_up_to_R, [[3], [3], [3, 3]], [[3, 3]]),
        (gu.pos_lookat_up_to_T, [[3], [3], [3]], [[4, 4]]),
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
def test_utils_geom_taichi_inverse(batch_shape):
    import gstaichi as ti

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
def test_utils_geom_taichi_identity(batch_shape):
    import gstaichi as ti

    for ti_funcs, shape_args in (
        ((gu.ti_xyz_to_quat, gu.ti_quat_to_xyz), ([3], [4])),
        ((gu.ti_xyz_to_quat, gu.ti_quat_to_R, gu.ti_R_to_xyz), ([3], [4], [3, 3])),
        ((gu.ti_xyz_to_quat, gu.ti_quat_to_rotvec, gu.ti_rotvec_to_R, gu.ti_R_to_xyz), ([3], [4], [3], [3, 3])),
        ((gu.ti_rotvec_to_quat, gu.ti_quat_to_rotvec), ([3], [4])),
    ):
        ti_args = []
        for shape_arg in (*shape_args, shape_args[0]):
            ti_type = ti.Vector if len(shape_arg) == 1 else ti.Matrix
            ti_arg = ti_type.field(*shape_arg, dtype=gs.ti_float, shape=batch_shape)
            ti_arg.from_numpy(np.random.randn(*batch_shape, *shape_arg).clip(-1.0, 1.0).astype(gs.np_float))
            ti_args.append(ti_arg)

        num_funcs = len(ti_funcs)
        for i, ti_func in enumerate(ti_funcs):
            kernel = _ti_kernel_wrapper(ti_func, 1, 1)
            kernel(*ti_args[i : (i + 2)])

        np.testing.assert_allclose(ti_args[0].to_numpy(), ti_args[-1].to_numpy(), atol=1e2 * gs.EPS)


@pytest.mark.required
@pytest.mark.parametrize("batch_shape", [(10, 40, 25), ()])
def test_utils_geom_tensor_identity(batch_shape):
    import gstaichi as ti

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

        num_funcs = len(py_funcs)
        for i, py_func in enumerate(py_funcs):
            np_args[i + 1][:] = py_func(np_args[i])
            tc_args[i + 1][:] = py_func(tc_args[i])

        np.testing.assert_allclose(np_args[0], np_args[-1], atol=1e2 * gs.EPS)
        np.testing.assert_allclose(tensor_to_array(tc_args[0]), tensor_to_array(tc_args[-1]), atol=1e2 * gs.EPS)
