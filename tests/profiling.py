import os
import re
import sys

import quadrants as qd
import pytest


def parser_add_options(parser: pytest.Parser) -> None:
    parser.addoption(
        "--profile-ref",
        type=str,
        default="",
        help="Label prefix added to output filenames (e.g. 'branch' or 'main').",
    )
    parser.addoption(
        "--profile-wait",
        type=int,
        required=True,
        help="Number of steps to wait before profiling. Depends on what you want to view, since the profile will likely vary throughout the benchmark.",
    )
    parser.addoption(
        "--profile-warmup", type=int, default=0, help="Number of warmup steps for profiling (default 0 is ok)."
    )
    parser.addoption(
        "--profile-active",
        type=int,
        default=1,
        help="Number of active profiling steps. (default 1 is ok; more than 1 will create large trace files)",
    )
    parser.addoption(
        "--profile-repeat",
        type=int,
        default=1,
        help="Number of times to repeat profiling. (default 1 is ok, unless you want to profile at multiple points during the simulation)",
    )
    parser.addoption(
        "--profile-dir",
        type=str,
        default="",
        help="Directory to write trace/summary files to. Defaults to CWD.",
    )


def _sanitize_test_name(node_id: str) -> str:
    """Extract a filesystem-safe name from a pytest node ID.

    e.g. 'tests/test_rigid_benchmarks.py::test_speed[duck_in_box_easy-None-True-30000-gpu]'
    becomes 'duck_in_box_easy-None-True-30000-gpu'
    """
    name = node_id.split("::")[-1]
    bracket = name.find("[")
    if bracket != -1:
        name = name[bracket + 1 :].rstrip("]")
    return re.sub(r"[^\w\-.]", "_", name)


def pytorch_profiler(pytestconfig, request):
    """Per-test fixture providing a PyTorch profiler step function.

    Activated by env var GS_PROFILING=1. Yields a step_fn that must be called
    after each simulation step.

    The profiler uses a schedule so that only a window of steps is actively
    traced, keeping the overhead minimal. On exit, a Chrome trace and summary
    are written with the test name in the filename.
    """
    import torch
    from torch.profiler import ProfilerActivity

    test_name = _sanitize_test_name(request.node.nodeid)

    wait = pytestconfig.getoption("--profile-wait")
    warmup = pytestconfig.getoption("--profile-warmup")
    active = pytestconfig.getoption("--profile-active")
    repeat = pytestconfig.getoption("--profile-repeat")
    ref = pytestconfig.getoption("--profile-ref")
    profile_dir = pytestconfig.getoption("--profile-dir")

    label = f"{ref}_{test_name}" if ref else test_name

    if profile_dir:
        os.makedirs(profile_dir, exist_ok=True)

    def _path(filename: str) -> str:
        if profile_dir:
            return os.path.join(profile_dir, filename)
        return filename

    schedule = torch.profiler.schedule(wait=wait, warmup=warmup, active=active, repeat=repeat)

    activities = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)

    trace_counter = [0]

    def trace_handler(prof):
        trace_path = _path(f"trace_{label}_{trace_counter[0]}.json")
        prof.export_chrome_trace(trace_path)
        trace_counter[0] += 1

        sort_by = "cuda_time_total" if torch.cuda.is_available() else "cpu_time_total"
        table = prof.key_averages().table(sort_by=sort_by, row_limit=40)
        summary_path = _path(f"profile_summary_{label}_{trace_counter[0] - 1}.txt")
        with open(summary_path, "w") as f:
            f.write(f"=== Kernel Summary (sorted by {sort_by}) ===\n")
            f.write(table + "\n")

        print(f"Exported trace to {trace_path}, summary to {summary_path}")

    prof = torch.profiler.profile(
        activities=activities,
        schedule=schedule,
        record_shapes=False,
        profile_memory=False,
        with_stack=True,
        with_flops=False,
        on_trace_ready=trace_handler,
    )

    step_counter = [0]
    original_step = prof.step

    def counted_step():
        original_step()
        step_counter[0] += 1
        qd.sync()

    print(f"PyTorch profiling: {test_name} (wait={wait}, warmup={warmup}, active={active})")
    with prof:
        yield counted_step
    if trace_counter[0] == 0:
        print(
            f"WARNING: on_trace_ready never fired ({step_counter[0]} steps called). Exporting fallback.",
            file=sys.stderr,
        )
        trace_path = _path(f"trace_{label}_fallback.json")
        prof.export_chrome_trace(trace_path)

        sort_by = "cuda_time_total" if torch.cuda.is_available() else "cpu_time_total"
        table = prof.key_averages().table(sort_by=sort_by, row_limit=40)
        summary_path = _path(f"profile_summary_{label}_fallback.txt")
        with open(summary_path, "w") as f:
            f.write(f"=== Kernel Summary (sorted by {sort_by}) ===\n")
            f.write(table + "\n")
        print(f"Fallback summary written to {summary_path}")
