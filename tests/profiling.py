import argparse
import pytest


def parser_add_options(parser: pytest.Parser) -> None:
    parser.addoption("--profile", action="store_true", default=False, help="Enable PyTorch profiling for benchmarks.")
    parser.addoption("--profile-wait", type=int, default=5, help="Number of steps to skip before profiling.")
    parser.addoption("--profile-warmup", type=int, default=0, help="Number of warmup steps for profiling.")
    parser.addoption("--profile-active", type=int, default=1, help="Number of active profiling steps.")
    parser.addoption("--profile-repeat", type=int, default=1, help="Number of times to repeat profiling.")


def pytorch_profiler(pytestconfig):
    """Session-scoped fixture providing a PyTorch profiler context manager.

    Activated by --profile. Yields a (profiler, step_fn) tuple where step_fn
    must be called after each simulation step. When --profile is not set, yields
    (None, noop).

    The profiler uses a schedule so that only a window of steps is actively
    traced, keeping the overhead minimal. On exit, a Chrome trace is written to
    ``profile_trace.json``.
    """
    if not pytestconfig.getoption("--profile"):
        noop = lambda: None  # noqa: E731
        yield None, noop
        return

    import torch
    from torch.profiler import ProfilerActivity

    wait = pytestconfig.getoption("--profile-wait")
    warmup = pytestconfig.getoption("--profile-warmup")
    active = pytestconfig.getoption("--profile-active")
    repeat = pytestconfig.getoption("--profile-repeat")

    schedule = torch.profiler.schedule(wait=wait, warmup=warmup, active=active, repeat=repeat)

    activities = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)

    prof = torch.profiler.profile(
        activities=activities,
        schedule=schedule,
        record_shapes=False,
        profile_memory=False,
        with_stack=True,
        with_flops=False,
    )

    print(f"PyTorch profiling enabled (wait={wait}, warmup={warmup}, active={active})")
    prof.__enter__()

    yield prof, prof.step

    prof.__exit__(None, None, None)

    trace_path = "profile_trace.json"
    prof.export_chrome_trace(str(trace_path))
    print(f"Chrome trace exported to: {trace_path}")
