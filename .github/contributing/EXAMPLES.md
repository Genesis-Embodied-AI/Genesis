# Examples Reference

Location: `examples/`, one folder per feature area. Every script lives in a folder; nothing sits at the root.

## Running Examples

```bash
uv run examples/tutorials/hello_genesis.py
uv run examples/rigid/single_franka.py
```

## Command-line convention

Every example parses its arguments the same way, so learning one script's CLI teaches all of them.

**Short flags are a closed, reserved set.** Each letter means exactly one thing across the whole tree. A flag
that is not in this table gets no short form.

| Short | Long | Type | Default |
|-------|------|------|---------|
| `-v` | `--vis` | `store_true` | off |
| `-c` | `--cpu` | `store_true` | off (GPU) |
| `-b` | `--num-envs` | `int` | per script |
| `-s` | `--steps` | `int` | per script |
| `-t` | `--seconds` | `float` | per script |
| `-e` | `--exp-name` | `str` | per script |
| `-r` | `--record` | `store_true` | off |
| `-o` | `--output-dir` | `str` | per script |
| `-d` | `--debug` | `store_true` | off |

Rules:

1. Short flag first, then long: `parser.add_argument("-v", "--vis", ...)`.
2. Long names use dashes between words (`--num-envs`); argparse exposes them as the underscored
   `args.num_envs`, so no `dest=` is needed.
3. Single-dash flags are one character.
4. Booleans are `store_true` and default off, which makes `default=False` redundant. Where the negative form
   reads better, a single `store_true` negative flag covers it (`--no-ipc`, `--no-force`).
5. Use `--steps` when the script's natural unit is iterations, `-t/--seconds` when it reasons in simulated
   time (that is, it also exposes `--dt` or compares against `scene.t`).
6. `--dt` is the only spelling for the timestep; it matches `SimOptions(dt=...)`.
7. Every flag carries a `help=`. Keyword order is `action=` / `type=`, then `default=`, then `help=`.
8. The parser is named `parser`, built as the first statements of `main()`, with
   `if __name__ == "__main__": main()` at the bottom.

A script whose physics only works on one backend hardcodes it and says why in a comment, rather than
exposing a flag with a single usable value (see `rigid/hibernation.py`).

Long-running scripts cut their horizon under `"PYTEST_VERSION" in os.environ` so the examples CI stays cheap.

## Style

Examples are read as reference code, so they are held to `CODING_GUIDELINES.md` like the rest of the tree.
Beyond it, two things matter most here:

- **Keep a script linear.** A helper function is warranted when it is called more than once, or is passed as
  a callback or thread target. A one-shot linear script reads top to bottom inside `main()`.
- **Model the sanctioned data access.** Reach for the public entity and solver getters first, and
  `qd_to_numpy` / `qd_to_torch` from `genesis.utils.misc` when no getter exists.

## Key Examples

### Getting Started
- `tutorials/hello_genesis.py` - Basic introduction
- `tutorials/control_your_robot.py` - Robot control basics
- `tutorials/parallel_simulation.py` - Batched environments

### Robotics
- `rigid/single_franka.py` - Single Franka arm
- `rigid/ik_franka.py` - Inverse kinematics
- `rigid/domain_randomization.py` - Domain randomization

### Multi-Physics
- `coupling/cloth_on_rigid.py` - Cloth-rigid coupling
- `coupling/sph_rigid.py` - Fluid-rigid coupling
- `coupling/sand_wheel.py` - Granular-rigid coupling

### Training
- `locomotion/go2_env.py` - Quadruped RL environment
- `manipulation/grasp_env.py` - Grasping RL environment
