import argparse
import gc
import os
import pathlib
import subprocess
import sys

import gstaichi as ti
import pytest
import numpy as np

import genesis as gs
from genesis.utils.misc import ti_to_torch

from .utils import assert_allclose


RET_SUCCESS = 42
RET_SKIP = 43

FILE_PATH = pathlib.Path(__file__)
MODULE_ROOT_DIR = FILE_PATH.parents[1]
MODULE = ".".join((FILE_PATH.parent.name, FILE_PATH.stem))


def _initialize_genesis(backend: gs.constants.backend | str):
    if isinstance(backend, str):
        backend = getattr(gs.constants.backend, backend)

    # Skip if requested backend is not available
    try:
        gs.utils.get_device(backend)
    except gs.GenesisException:
        print(f"Backend '{backend}' not available on this machine", file=sys.stderr)
        sys.exit(RET_SKIP)

    # Skip test if gstaichi ndarray mode is enabled but not supported by this specific test
    if sys.platform == "darwin" and backend != gs.cpu and os.environ.get("GS_USE_NDARRAY") == "1":
        print(
            "Using gstaichi ndarray on Mac OS with gpu backend is unreliable, because Apple Metal only supports up to "
            "31 kernel parameters, which is not enough for most solvers.",
            file=sys.stderr,
        )
        sys.exit(RET_SKIP)

    gs.init(backend=backend, precision="32")

    if backend != gs.cpu and gs.backend == gs.cpu:
        print("No GPU available on this machine", file=sys.stderr)
        sys.exit(RET_SKIP)


@pytest.mark.parametrize("batch_shape", [(2, 3, 5), ()])
@pytest.mark.parametrize(
    "ti_type_spec, arg_shape",
    [
        (("field", "scalar"), ()),
        (("field", "vector"), (7,)),
        (("field", "matrix"), (7, 1)),
        (("field", "matrix"), (7, 11)),
        (("ndarray", "scalar"), ()),
        (("ndarray", "vector"), (7,)),
        (("ndarray", "matrix"), (7, 1)),
        (("ndarray", "matrix"), (7, 11)),
    ],
)
def test_to_torch(ti_type_spec, batch_shape, arg_shape):
    import gstaichi as ti

    for _ in range(10):
        TI_TYPE_MAP = {
            ("field", "scalar"): ti.field,
            ("field", "vector"): ti.Vector.field,
            ("field", "matrix"): ti.Matrix.field,
            ("ndarray", "scalar"): ti.ndarray,
            ("ndarray", "vector"): ti.Vector.ndarray,
            ("ndarray", "matrix"): ti.Matrix.ndarray,
        }

        np_arg = np.asarray(np.random.rand(*batch_shape, *arg_shape), dtype=np.float32)
        ti_arg = TI_TYPE_MAP[ti_type_spec](*arg_shape, dtype=ti.f32, shape=batch_shape)
        ti_arg.from_numpy(np_arg)
        assert_allclose(ti_to_torch(ti_arg), ti_arg.to_numpy(), tol=gs.EPS)

        # Restart taichi runtime
        arch_idx = int(ti.cfg.arch)
        debug = ti.cfg.debug
        ti.reset()
        ti.init(arch=ti._lib.core.Arch(arch_idx), debug=debug)
        gc.collect()


def gs_static_child(args: list[str]):
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", type=str, choices=["cpu", "gpu"], default="cpu")
    parser.add_argument("--enable-multi-contact", action="store_true")
    parser.add_argument("--expected-num-contacts", type=int, required=True)
    parser.add_argument("--expected-use-src-ll-cache", type=int, required=True)
    parser.add_argument("--expected-src-ll-cache-hit", type=int, required=True)
    args = parser.parse_args(args)

    _initialize_genesis(backend=args.backend)

    scene = gs.Scene(
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(1, 1, 0.5),
            camera_lookat=(0.0, 0.0, 0.0),
        ),
        rigid_options=gs.options.RigidOptions(
            enable_multi_contact=args.enable_multi_contact,
        ),
        show_viewer=False,
    )
    scene.add_entity(
        gs.morphs.Plane(),
    )
    scene.add_entity(
        gs.morphs.Box(
            size=(0.4, 0.4, 0.4),
            pos=(0.0, 0.0, 0.18),
        )
    )
    scene.build()

    scene.rigid_solver.collider.detection()
    gs.ti.sync()
    actual_contacts = scene.rigid_solver.collider._collider_state.n_contacts.to_numpy()
    assert actual_contacts == args.expected_num_contacts
    from genesis.engine.solvers.rigid.collider_decomp import func_narrow_phase_convex_vs_convex

    assert (
        func_narrow_phase_convex_vs_convex._primal.src_ll_cache_observations.cache_key_generated
        == args.expected_use_src_ll_cache
    )
    assert (
        func_narrow_phase_convex_vs_convex._primal.src_ll_cache_observations.cache_validated
        == args.expected_src_ll_cache_hit
    )
    assert (
        func_narrow_phase_convex_vs_convex._primal.src_ll_cache_observations.cache_loaded
        == args.expected_src_ll_cache_hit
    )

    sys.exit(RET_SUCCESS)


@pytest.mark.required
@pytest.mark.parametrize("backend", [None])  # Disable genesis initialization at worker level
@pytest.mark.parametrize("test_backend", ["cpu", "gpu"])
@pytest.mark.parametrize("use_ndarray", [False, True])
@pytest.mark.parametrize("enable_pure", [False, True])
@pytest.mark.parametrize("enable_multicontact, expected_num_contacts", [(False, 1), (True, 4)])
def test_static(
    enable_multicontact: bool,
    expected_num_contacts: int,
    enable_pure: bool,
    test_backend: str,
    use_ndarray: bool,
    tmp_path: pathlib.Path,
) -> None:
    for it in range(3):
        # we iterate to make sure stuff is really being read from cache
        cmd_line = [
            sys.executable,
            "-m",
            MODULE,
            gs_static_child.__name__,
            "--expected-num-contacts",
            str(expected_num_contacts),
            "--expected-use-src-ll-cache",
            "1" if enable_pure and use_ndarray else "0",
            "--expected-src-ll-cache-hit",
            "1" if enable_pure and use_ndarray and it > 0 else "0",
            "--backend",
            test_backend,
        ]
        if enable_multicontact:
            cmd_line += ["--enable-multi-contact"]
        env_changes = {}
        env_changes["GS_BETA_PURE"] = "1" if enable_pure else "0"
        env_changes["TI_OFFLINE_CACHE_FILE_PATH"] = str(tmp_path)
        env_changes["GS_USE_NDARRAY"] = "1" if use_ndarray else "0"
        env = dict(os.environ)
        env.update(env_changes)

        proc = subprocess.run(cmd_line, capture_output=True, text=True, env=env, cwd=MODULE_ROOT_DIR)
        if proc.returncode == RET_SKIP:
            pytest.skip(proc.stderr)
        elif proc.returncode != RET_SUCCESS:
            print("============================")
            print("it", it)
            for k, v in env_changes.items():
                print(f"export {k}={v}")
            print(" ".join(cmd_line))
            print("stderr", proc.stderr)
            print("stdout", proc.stdout)
        assert proc.returncode == RET_SUCCESS


def gs_num_envs_child(args: list[str]):
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", type=str, choices=["cpu", "gpu"], default="cpu")
    parser.add_argument("--n_envs", type=int, required=True)
    parser.add_argument("--expected-use-src-ll-cache", action="store_true")
    parser.add_argument("--expected-src-ll-cache-hit", action="store_true")
    parser.add_argument("--expected-fe-ll-cache-hit", action="store_true")
    args = parser.parse_args(args)

    _initialize_genesis(backend=args.backend)

    scene = gs.Scene(
        show_viewer=False,
    )
    scene.add_entity(
        gs.morphs.Plane(),
    )
    scene.add_entity(
        gs.morphs.Box(
            size=(0.4, 0.4, 0.4),
            pos=(0.0, 0.0, 0.18),
        )
    )
    scene.build(n_envs=args.n_envs, env_spacing=(0.5, 0.5))

    scene.rigid_solver.collider.detection()
    gs.ti.sync()

    from genesis.engine.solvers.rigid.rigid_solver_decomp import kernel_step_1

    assert kernel_step_1._primal.fe_ll_cache_observations.cache_hit == args.expected_fe_ll_cache_hit
    assert kernel_step_1._primal.src_ll_cache_observations.cache_key_generated == args.expected_use_src_ll_cache
    assert kernel_step_1._primal.src_ll_cache_observations.cache_loaded == args.expected_src_ll_cache_hit

    sys.exit(RET_SUCCESS)


@pytest.mark.required
@pytest.mark.parametrize("backend", [None])  # Disable genesis initialization at worker level
@pytest.mark.parametrize("test_backend", ["cpu", "gpu"])
@pytest.mark.parametrize("enable_pure", [False, True])
@pytest.mark.parametrize("use_ndarray", [False, True])
def test_num_envs(use_ndarray: bool, enable_pure: bool, test_backend: str, tmp_path: pathlib.Path) -> None:
    # Change n_envs each time, and check effect on reading from cache
    for it, n_envs in enumerate([3, 5, 7]):
        cmd_line = [
            sys.executable,
            "-m",
            MODULE,
            gs_num_envs_child.__name__,
            "--backend",
            test_backend,
            "--n_envs",
            str(n_envs),
        ]
        env = dict(os.environ)
        env["GS_BETA_PURE"] = "1" if enable_pure else "0"
        env["GS_USE_NDARRAY"] = "1" if use_ndarray else "0"
        env["TI_OFFLINE_CACHE_FILE_PATH"] = str(tmp_path)
        # notes:
        # - if we use pure, we won't get as far as fe-ll-cache
        # - ndarray and pure therefore wont ever use fe-ll-cache (first time, nothing in cache; after that hit src-ll cache)
        # - not use ndarray will always try using the fe-ll-cache, but cache will be empty on first it
        #   but since we are changing num envs each time, using fields will never get a cache hit either
        # soooo we are left only with (not pure) and (ndarray) and (it > 0)
        expected_fe_ll_cache_hit = not enable_pure and use_ndarray and it > 0
        # fields are not supported by src-ll-cache currently
        expected_use_src_ll_cache = enable_pure and use_ndarray
        expected_src_ll_cache_hit = enable_pure and use_ndarray and it > 0
        if expected_fe_ll_cache_hit:
            cmd_line += ["--expected-fe-ll-cache-hit"]
        if expected_use_src_ll_cache:
            cmd_line += ["--expected-use-src-ll-cache"]
        if expected_src_ll_cache_hit:
            cmd_line += ["--expected-src-ll-cache-hit"]
        proc = subprocess.run(cmd_line, capture_output=True, text=True, env=env, cwd=MODULE_ROOT_DIR)
        if proc.returncode == RET_SKIP:
            pytest.skip(proc.stderr)
        elif proc.returncode != RET_SUCCESS:
            print("============================")
            print("it", it, "n_envs", n_envs)
            print("cmd_line", cmd_line)
            print("env", env)
            print("stderr", proc.stderr)
            print("stdout", proc.stdout)
        assert proc.returncode == RET_SUCCESS


def change_scene(args: list[str]):
    from genesis.engine.solvers.rigid.rigid_solver_decomp import kernel_step_1

    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", type=str, choices=["cpu", "gpu"], default="cpu")
    parser.add_argument("--n_objs", type=int, required=True)
    parser.add_argument("--n_envs", type=int, required=True)
    parser.add_argument("--expected-src-ll-cache-hit", type=int, required=True)
    args = parser.parse_args(args)

    _initialize_genesis(backend=args.backend)

    scene = gs.Scene(
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(1, 1, 0.5),
            camera_lookat=(0.0, 0.0, 0.0),
        ),
        profiling_options=gs.options.ProfilingOptions(
            show_FPS=False,
        ),
        show_viewer=False,
    )
    plane = scene.add_entity(
        gs.morphs.Plane(),
    )
    for i_obj in range(args.n_objs):
        cube = scene.add_entity(
            gs.morphs.Box(
                size=(0.4, 0.4, 0.4),
                pos=(0.0, 0.5 * i_obj, 0.8),
            )
        )
    scene.build(n_envs=args.n_envs)

    for i in range(500):
        scene.step()

    qpos = scene.sim.rigid_solver.get_qpos()
    if args.n_envs > 0:
        assert qpos.ndim == 2
        assert qpos.shape[0] == args.n_envs
    else:
        assert qpos.ndim == 1
    assert qpos.shape[-1] == args.n_objs * 7

    z = qpos.reshape((*qpos.shape[:-1], args.n_objs, 7))[..., 2]
    assert_allclose(z, 0.2, atol=1e-2, err_msg=f"zs {z} is not close to 0.2.")

    assert kernel_step_1._primal.src_ll_cache_observations.cache_validated == args.expected_src_ll_cache_hit
    assert kernel_step_1._primal.src_ll_cache_observations.cache_loaded == args.expected_src_ll_cache_hit

    sys.exit(RET_SUCCESS)


@pytest.mark.required
@pytest.mark.parametrize("backend", [None])  # Disable genesis initialization at worker level
@pytest.mark.parametrize("test_backend", ["cpu", "gpu"])
@pytest.mark.parametrize(
    "list_n_objs_n_envs",
    [
        [(1, 1), (2, 2), (3, 3)],
        # [(3, 0), (1, 1), (2, 2)],  # FIXME: This does not work with gpu, needs to investigate (cache key changes).
    ],
)
@pytest.mark.parametrize("enable_pure", [True])
def test_ndarray_no_compile(
    enable_pure: bool, list_n_objs_n_envs: list[tuple[int, int]], test_backend: str, tmp_path: pathlib.Path
) -> None:
    # Iterate to make sure stuff is really being read from cache
    for i, (n_objs, n_envs) in enumerate(list_n_objs_n_envs):
        cmd_line = [
            sys.executable,
            "-m",
            MODULE,
            change_scene.__name__,
            "--n_objs",
            str(n_objs),
            "--n_envs",
            str(n_envs),
            "--expected-src-ll-cache-hit",
            "1" if enable_pure and i > 0 else "0",
            "--backend",
            test_backend,
        ]
        env = dict(os.environ)
        env["GS_BETA_PURE"] = "1" if enable_pure else "0"
        env["GS_USE_NDARRAY"] = "1"
        env["TI_OFFLINE_CACHE_FILE_PATH"] = str(tmp_path)
        proc = subprocess.run(cmd_line, capture_output=True, text=True, env=env, cwd=MODULE_ROOT_DIR)

        # Display error message only in case of failure
        if proc.returncode == RET_SKIP:
            pytest.skip(proc.stderr)
        elif proc.returncode != RET_SUCCESS:
            print(proc.stdout)
            print("-" * 100)
            print(proc.stderr)
        assert proc.returncode == RET_SUCCESS


# The following lines are critical for the test to work
if __name__ == "__main__":
    globals()[sys.argv[1]](sys.argv[2:])
