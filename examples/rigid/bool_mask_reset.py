"""
Zerocopy binary mask fast path for scene.reset().

In RL training loops, users typically convert a boolean reset buffer to integer
indices before calling scene.reset():

    reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)  # blocking GPU sync!
    scene.reset(envs_idx=reset_env_ids)

The nonzero() call forces a GPU-CPU synchronization that stalls the pipeline.

With the bool mask fast path, you can now pass the bool tensor directly:

    scene.reset(envs_idx=self.reset_buf)  # non-blocking, no GPU sync

This example demonstrates the feature by running a simple RL-style loop with
4 parallel environments, resetting environments that meet a termination condition
using a bool mask instead of int indices.
"""

import torch
import genesis as gs


def main():
    gs.init(backend=gs.gpu)

    # Build a simple scene with a falling box across 4 parallel environments
    n_envs = 4
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.01),
        show_viewer=False,
    )
    scene.add_entity(gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True))
    box = scene.add_entity(gs.morphs.Box(size=(0.1, 0.1, 0.1), pos=(0, 0, 0.5)))
    scene.build(n_envs=n_envs, env_spacing=(0.1, 0.1))

    # Save the initial state for resets
    init_state = scene.get_state()

    print(f"Running {n_envs} parallel environments...")
    print(f"Zerocopy mode: {gs.use_zerocopy}")
    print()

    # set varying box positions for each environment
    box_pos = torch.tensor([[0, 0, 0.25], [0, 0, 0.5], [0, 0, 0.75], [0, 0, 1.0]], dtype=gs.tc_float, device=gs.device)
    box.set_pos(box_pos)

    for step in range(500):
        scene.step()

        # Get the box height in each environment
        box_pos = box.get_pos()  # shape: (n_envs, 3)
        box_height = box_pos[:, 2]

        # RL-style termination: reset environments where box fell below threshold
        # This is a bool tensor on GPU — no sync needed!
        reset_buf = box_height < 0.05

        if reset_buf.any():
            # ======================================================
            # KEY: Pass the bool mask directly — no nonzero() needed!
            # ======================================================
            scene.reset(state=init_state, envs_idx=reset_buf)

            n_reset = reset_buf.sum().item()
            print(f"  Step {step:3d}: Reset {n_reset} env(s) | heights: {box_height.cpu().numpy().round(3)}")

    # Final state
    final_pos = box.get_pos()
    print(f"\nFinal box heights: {final_pos[:, 2].cpu().numpy().round(3)}")

    # Verify equivalence: compare bool mask reset with int index reset
    print("\n--- Equivalence check ---")
    for _ in range(50):
        scene.step()

    state_before = box.get_pos().clone()

    # Reset envs 0 and 2 with int indices
    scene.reset(state=init_state, envs_idx=torch.tensor([0, 2], dtype=gs.tc_int, device=gs.device))
    pos_after_int = box.get_pos().clone()

    # Restore and reset envs 0 and 2 with bool mask
    # First undo the reset by stepping back to diverged state
    scene.reset(state=init_state)
    for _ in range(50):
        scene.step()
    bool_mask = torch.tensor([True, False, True, False], dtype=torch.bool, device=gs.device)
    scene.reset(state=init_state, envs_idx=bool_mask)
    pos_after_bool = box.get_pos().clone()

    diff = (pos_after_int - pos_after_bool).abs().max().item()
    print(f"Max difference between int-index and bool-mask reset: {diff:.2e}")
    # Tolerance accounts for f32 precision (kernel_set_state uses Taichi kernels
    # while the bool-mask path uses PyTorch ops — both are f32-correct but may
    # differ at the ULP level after FK propagation).
    tol = 1e-4 if gs.np_float.__name__ == "float32" else 1e-9
    assert diff < tol, f"Results diverged! diff={diff}, tol={tol}"
    print(f"PASSED: Bool mask reset matches int index reset (tol={tol:.0e}).")


if __name__ == "__main__":
    main()
