import pathlib
import pytest
import subprocess
import sys
import os

import argparse
import genesis as gs


RET_SUCCESS = 42


def gs_static_child(args: list[str]):
    parser = argparse.ArgumentParser()
    parser.add_argument("--enable-multi-contact", action="store_true")
    parser.add_argument("--expected-num-contacts", type=int, required=True)
    parser.add_argument("--expected-use-src-ll-cache", type=int, required=True)
    parser.add_argument("--expected-src-ll-cache-hit", type=int, required=True)
    parser.add_argument("--test-backend", type=str, choices=["cpu", "gpu"], default="cpu")
    args = parser.parse_args(args)

    gs.init(
        backend=getattr(gs, args.test_backend),
        precision="32",
    )

    scene = gs.Scene(
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(1, 1, 0.5),
            camera_lookat=(0.0, 0.0, 0.0),
        ),
        rigid_options=gs.options.RigidOptions(enable_multi_contact=args.enable_multi_contact),
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
# should not affect expected_num_contacts
@pytest.mark.parametrize("use_ndarray", [False, True])
# should not affect expected_num_contacts
# note that using `backend` instead of `test_backend`, breaks genesis pytest...
@pytest.mark.parametrize("test_backend", ["cpu", "gpu"])
# should not affect expected_num_contacts
@pytest.mark.parametrize("enable_pure", [False, True])
@pytest.mark.parametrize(
    "enable_multicontact,expected_num_contacts",
    [
        (False, 1),
        (True, 4),
    ],
)
def test_gs_static(
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
            __file__,
            gs_static_child.__name__,
            "--expected-num-contacts",
            str(expected_num_contacts),
            "--expected-use-src-ll-cache",
            "1" if enable_pure and use_ndarray else "0",
            "--expected-src-ll-cache-hit",
            "1" if enable_pure and use_ndarray and it > 0 else "0",
            "--test-backend",
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

        proc = subprocess.run(cmd_line, capture_output=True, text=True, env=env)
        if proc.returncode != RET_SUCCESS:
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
    parser.add_argument("--test-backend", type=str, choices=["cpu", "gpu"], default="cpu")
    parser.add_argument("--expected-use-src-ll-cache", action="store_true")
    parser.add_argument("--expected-src-ll-cache-hit", action="store_true")
    parser.add_argument("--expected-fe-ll-cache-hit", action="store_true")
    parser.add_argument("--num-env", type=int, required=True)
    args = parser.parse_args(args)

    gs.init(backend=getattr(gs, args.test_backend), precision="32")

    scene = gs.Scene(show_viewer=False)
    scene.add_entity(
        gs.morphs.Plane(),
    )
    scene.add_entity(
        gs.morphs.Box(
            size=(0.4, 0.4, 0.4),
            pos=(0.0, 0.0, 0.18),
        )
    )

    scene.build(n_envs=args.num_env, env_spacing=(0.5, 0.5))
    scene.rigid_solver.collider.detection()
    gs.ti.sync()

    from genesis.engine.solvers.rigid.rigid_solver_decomp import kernel_step_1

    assert kernel_step_1._primal.fe_ll_cache_observations.cache_hit == args.expected_fe_ll_cache_hit
    assert kernel_step_1._primal.src_ll_cache_observations.cache_key_generated == args.expected_use_src_ll_cache
    assert kernel_step_1._primal.src_ll_cache_observations.cache_loaded == args.expected_src_ll_cache_hit

    sys.exit(RET_SUCCESS)


@pytest.mark.required
@pytest.mark.parametrize("enable_pure", [False, True])  # should not affect result
# note that using `backend` instead of `test_backend`, breaks genesis pytest...
@pytest.mark.parametrize("test_backend", ["cpu", "gpu"])  # should not affect result
@pytest.mark.parametrize("use_ndarray", [False, True])
def test_gs_num_envs(use_ndarray: bool, enable_pure: bool, test_backend: str, tmp_path: pathlib.Path) -> None:
    if sys.platform == "darwin" and test_backend == "gpu" and use_ndarray:
        pytest.skip(
            "fast cache not supported on mac gpus when using ndarray, because mac gpu only supports up to 31 kernel"
            " parameters, and we need more than that."
        )

    # change num_env each time, and check effect on reading from cache
    for it, num_env in enumerate([3, 5, 7]):
        cmd_line = [
            sys.executable,
            __file__,
            gs_num_envs_child.__name__,
            "--test-backend",
            test_backend,
            "--num-env",
            str(num_env),
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
        proc = subprocess.run(cmd_line, capture_output=True, text=True, env=env)
        if proc.returncode != RET_SUCCESS:
            print("============================")
            print("it", it, "num_env", num_env)
            print("cmd_line", cmd_line)
            print("env", env)
            print("stderr", proc.stderr)
            print("stdout", proc.stdout)
        assert proc.returncode == RET_SUCCESS


def change_scene(args: list[str]):
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_obj", type=int, required=True)
    parser.add_argument("--n_env", type=int, required=True)
    parser.add_argument("--expected-src-ll-cache-hit", type=int, required=True)
    parser.add_argument("--test-backend", type=str, choices=["cpu", "gpu"], default="cpu")
    args = parser.parse_args(args)

    gs.init(backend=getattr(gs, args.test_backend), precision="32")

    scene = gs.Scene(
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(1, 1, 0.5),
            camera_lookat=(0.0, 0.0, 0.0),
        ),
        rigid_options=gs.options.RigidOptions(),
        profiling_options=gs.options.ProfilingOptions(show_FPS=False),
        show_viewer=False,
    )
    plane = scene.add_entity(
        gs.morphs.Plane(),
    )

    for i_obj in range(args.n_obj):
        cube = scene.add_entity(
            gs.morphs.Box(
                size=(0.4, 0.4, 0.4),
                pos=(0.0, 0.5 * i_obj, 0.8),
            )
        )
    if args.n_env > 0:
        scene.build(n_envs=args.n_env)
    else:
        scene.build()

    for i in range(500):
        scene.step()
    # ti_field_to_torch does not work with ndarray now
    # qpos = scene.sim.rigid_solver.get_qpos()
    qpos = scene.sim.rigid_solver.qpos.to_numpy()

    z = qpos.reshape(args.n_obj, 7, max(1, args.n_env))[:, 2, :]

    # workaround to get current file's path, and import from .utils
    current_file_path = pathlib.Path(__file__)
    sys.path.append(current_file_path.parent)
    from utils import assert_allclose

    sys.path.remove(current_file_path.parent)

    assert_allclose(z, 0.2, atol=1e-2, err_msg=f"zs {z} is not close to 0.2.")

    assert qpos.shape[0] == args.n_obj * 7
    assert qpos.shape[1] == max(args.n_env, 1)
    from genesis.engine.solvers.rigid.rigid_solver_decomp import kernel_step_1

    assert kernel_step_1._primal.src_ll_cache_observations.cache_validated == args.expected_src_ll_cache_hit
    assert kernel_step_1._primal.src_ll_cache_observations.cache_loaded == args.expected_src_ll_cache_hit

    sys.exit(RET_SUCCESS)


@pytest.mark.required
@pytest.mark.parametrize(
    "list_n_objs_n_envs",
    [
        [(1, 1), (2, 2), (3, 3)],
        # [(3, 0), (1, 1), (2, 2)],  # FIXME:this does not work with gpu, needs to investigate. (cache key cahnges)
    ],
)
@pytest.mark.parametrize("enable_pure", [True])  # should not affect result
# note that using `backend` instead of `test_backend`, breaks genesis pytest...
@pytest.mark.parametrize("test_backend", ["cpu", "gpu"])  # should not affect result
def test_ndarray_no_compile(
    enable_pure: bool, list_n_objs_n_envs: list[tuple[int, int]], test_backend: str, tmp_path: pathlib.Path
) -> None:
    if sys.platform == "darwin" and test_backend == "gpu":
        pytest.skip(
            "fast cache not supported on mac gpus when using ndarray, because mac gpu only supports up to 31 kernel"
            " parameters, and we need more than that."
        )

    for it in range(len(list_n_objs_n_envs)):
        # we iterate to make sure stuff is really being read from cache
        n_objs, n_envs = list_n_objs_n_envs[it]
        cmd_line = [
            sys.executable,
            __file__,
            change_scene.__name__,
            "--n_obj",
            str(n_objs),
            "--n_env",
            str(n_envs),
            "--expected-src-ll-cache-hit",
            "1" if enable_pure and it > 0 else "0",
            "--test-backend",
            test_backend,
        ]
        env = dict(os.environ)
        env["GS_BETA_PURE"] = "1" if enable_pure else "0"
        env["GS_USE_NDARRAY"] = "1"  # test ndarray
        env["TI_OFFLINE_CACHE_FILE_PATH"] = str(tmp_path)
        proc = subprocess.run(cmd_line, capture_output=True, text=True, env=env)
        if proc.returncode != RET_SUCCESS:
            print(proc.stdout)  # needs to do this to see error messages
            print("-" * 100)
            print(proc.stderr)
        assert proc.returncode == RET_SUCCESS


# The following lines are critical for the test to work. If they are missing, the test will
# incorrectly pass, without doing anything.
if __name__ == "__main__":
    print("__main__")
    globals()[sys.argv[1]](sys.argv[2:])
