import pytest


def parser_add_options(parser: pytest.Parser) -> None:
    parser.addoption(
        "--profile-ref",
        type=str,
        default="",
        help="Added to output filename.",
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


def pytorch_profiler(pytestconfig):
    """Session-scoped fixture providing a PyTorch profiler context manager.

    Activated by env var GS_PROFILING=1. Yields a (profiler, step_fn) tuple where step_fn
    must be called after each simulation step.

    The profiler uses a schedule so that only a window of steps is actively
    traced, keeping the overhead minimal. On exit, a Chrome trace is written to
    ``profile_trace.json``.
    """
    import torch
    from torch.profiler import ProfilerActivity

    wait = pytestconfig.getoption("--profile-wait")
    warmup = pytestconfig.getoption("--profile-warmup")
    active = pytestconfig.getoption("--profile-active")
    repeat = pytestconfig.getoption("--profile-repeat")
    ref = pytestconfig.getoption("--profile-ref")

    schedule = torch.profiler.schedule(wait=wait, warmup=warmup, active=active, repeat=repeat)

    activities = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)

    trace_counter = [0]

    def trace_handler(prof):
        trace_path = f"trace_{ref}_{trace_counter[0]}.json"
        prof.export_chrome_trace(trace_path)
        trace_counter[0] += 1
        print(f"Exported trace cycle {trace_counter[0]} to {trace_path}")

    prof = torch.profiler.profile(
        activities=activities,
        schedule=schedule,
        record_shapes=False,
        profile_memory=False,
        with_stack=True,
        with_flops=False,
        on_trace_ready=trace_handler,
    )

    print(f"PyTorch profiling enabled (wait={wait}, warmup={warmup}, active={active})")
    with prof:
        yield prof.step
