import pathlib
import pytest
import subprocess
import sys
import os

import argparse
import genesis as gs


def gs_static_child(args: list[str]):
    print("gs_static_child")
    parser = argparse.ArgumentParser()
    parser.add_argument("--enable-multi-contact", action="store_true")
    parser.add_argument("--expected-num-contacts", type=int, required=True)
    parser.add_argument("--expected-use-src-ll-cache", action="store_true")
    parser.add_argument("--expected-src-ll-cache-hit", action="store_true")
    parser.add_argument("--test-backend", type=str, choices=["cpu", "gpu"], default="cpu")
    args = parser.parse_args(args)

    print(f'backend {args.test_backend}', type(args.test_backend))

    gs.init(backend=getattr(gs, args.test_backend), precision="32")

    scene = gs.Scene(
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(1, 1, 0.5),
            camera_lookat=(0.0, 0.0, 0.0),
        ),
        rigid_options=gs.options.RigidOptions(enable_multi_contact=args.enable_multi_contact),
        show_viewer=False,
    )
    plane = scene.add_entity(
        gs.morphs.Plane(),
    )
    cube = scene.add_entity(
        gs.morphs.Box(
            size=(0.4, 0.4, 0.4),
            pos=(0.0, 0.0, 0.18),
        )
    )
    scene.build()

    scene.rigid_solver.collider.detection()
    gs.ti.sync()
    print("expected num contacts", args.expected_num_contacts)
    actual_contacts = scene.rigid_solver.collider._collider_state.n_contacts.to_numpy()
    print('actual', actual_contacts, type(actual_contacts))
    assert scene.rigid_solver.collider._collider_state.n_contacts.to_numpy() == args.expected_num_contacts
    from genesis.engine.solvers.rigid.collider_decomp import func_narrow_phase_convex_vs_convex
    assert func_narrow_phase_convex_vs_convex._primal.src_ll_cache_observations.cache_key_generated == args.expected_use_src_ll_cache
    assert func_narrow_phase_convex_vs_convex._primal.src_ll_cache_observations.cache_validated == args.expected_src_ll_cache_hit
    assert func_narrow_phase_convex_vs_convex._primal.src_ll_cache_observations.cache_loaded == args.expected_src_ll_cache_hit


@pytest.mark.parametrize(
    "enable_multicontact,expected_num_contacts", [
        (False, 1),
        (True, 4),
    ]
)
@pytest.mark.parametrize("enable_pure", [False, True])  # should not affect result
# note that using `backend` instead of `test_backend`, breaks genesis pytest...
@pytest.mark.parametrize("test_backend", ["cpu", "gpu"])  # should not affect result
def test_gs_static(enable_multicontact: bool, enable_pure: bool, test_backend: str, expected_num_contacts: int, tmp_path: pathlib.Path) -> None:
    print("test_gs_static")
    for it in range(3):
        # we iterate to make sure stuff is really being read from cache
        cmd_line = [
            sys.executable,
            __file__,
            gs_static_child.__name__,
            "--expected-num-contacts", str(expected_num_contacts),
            "--expected-use-src-ll-cache", "1" if enable_pure else "0",
            "--expected-src-ll-cache-hit", "1" if enable_pure and it > 0 else "0",
            "--test-backend", test_backend,
        ]
        if enable_multicontact:
            cmd_line += ["--enable-multi-contact"]
        env = dict(os.environ)
        env["GS_BETA_PURE"] = "1" if enable_pure else "0"
        env["TI_OFFLINE_CACHE_FILE_PATH"] = str(tmp_path)
        proc = subprocess.run(cmd_line, capture_output=True, text=True, env=env)
        print(proc.stdout)
        print(proc.stderr)
        assert proc.returncode == 0


if __name__ == '__main__':
    print("__main__")
    globals()[sys.argv[1]](sys.argv[2:])
