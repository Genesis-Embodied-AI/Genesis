"""
Benchmark: bool mask reset vs nonzero() + int index reset.

Compares the two scene.reset() approaches:

  OLD (blocking):
      idx = reset_buf.nonzero(as_tuple=False).squeeze(-1)
      scene.reset(state=init_state, envs_idx=idx)

  NEW (non-blocking):
      scene.reset(state=init_state, envs_idx=reset_buf)

On GPU the nonzero() call forces a GPU→CPU sync. At high env counts this
becomes a significant bottleneck in RL training loops.
"""

import time

import torch
import genesis as gs


def bench_reset(scene, init_state, reset_buf, method, n_warmup=10, n_iters=100):
    """Benchmark a single reset method, return per-call time in ms."""

    if method == "bool_mask":

        def do_reset():
            scene.reset(state=init_state, envs_idx=reset_buf)
    elif method == "nonzero":

        def do_reset():
            idx = reset_buf.nonzero(as_tuple=False).squeeze(-1)
            scene.reset(state=init_state, envs_idx=idx)
    else:
        raise ValueError(method)

    # Warmup
    for _ in range(n_warmup):
        scene.step()
        do_reset()
    torch.cuda.synchronize()

    # Timed iterations
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iters):
        scene.step()
        do_reset()
    torch.cuda.synchronize()
    t1 = time.perf_counter()

    ms_per_call = (t1 - t0) / n_iters * 1000
    return ms_per_call


def main():
    if not torch.cuda.is_available():
        print("CUDA not available — this benchmark requires a GPU. Skipping.")
        return

    gs.init(backend=gs.gpu)

    n_envs = 4
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.01),
        show_viewer=False,
    )
    scene.add_entity(gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True))
    scene.add_entity(gs.morphs.Box(size=(0.1, 0.1, 0.1), pos=(0, 0, 0.5)))
    scene.build(n_envs=n_envs)

    init_state = scene.get_state()

    print(f"Environments : {n_envs}")
    print(f"Zerocopy     : {gs.use_zerocopy}")
    print(f"Device       : {gs.device}")
    print()

    # Test different reset fractions: 25%, 50%, 75%, 100%
    reset_fractions = [0.25, 0.5, 0.75, 1.0]

    header = f"{'Reset %':>8}  {'# resets':>8}  {'nonzero (ms)':>14}  {'bool mask (ms)':>15}  {'speedup':>8}"
    print(header)
    print("-" * len(header))

    for frac in reset_fractions:
        n_reset = int(n_envs * frac)
        # Deterministic mask: first n_reset envs are True
        reset_buf = torch.zeros(n_envs, dtype=torch.bool, device=gs.device)
        reset_buf[:n_reset] = True

        ms_nonzero = bench_reset(scene, init_state, reset_buf, "nonzero")
        ms_bool = bench_reset(scene, init_state, reset_buf, "bool_mask")
        speedup = ms_nonzero / ms_bool

        print(f"{frac * 100:7.0f}%  {n_reset:8d}  {ms_nonzero:14.2f}  {ms_bool:15.2f}  {speedup:7.2f}x")

    print()

    # Also benchmark with a random mask
    print("--- Random mask (50% reset probability per env) ---")
    torch.manual_seed(42)
    reset_buf = torch.rand(n_envs, device=gs.device) < 0.5
    n_reset = reset_buf.sum().item()

    ms_nonzero = bench_reset(scene, init_state, reset_buf, "nonzero")
    ms_bool = bench_reset(scene, init_state, reset_buf, "bool_mask")
    speedup = ms_nonzero / ms_bool

    print(
        f"  Random 50%  ({int(n_reset)} resets):  nonzero={ms_nonzero:.2f} ms  bool_mask={ms_bool:.2f} ms  speedup={speedup:.2f}x"
    )


if __name__ == "__main__":
    main()
