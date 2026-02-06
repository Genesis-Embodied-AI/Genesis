# Examples Reference

Location: `examples/` (105 files total)

## Running Examples

```bash
uv run examples/tutorials/hello_genesis.py
uv run examples/rigid/single_franka.py
```

## Categories

| Directory | Files | Description |
|-----------|-------|-------------|
| `tutorials/` | 18 | Getting started guides |
| `rigid/` | 29 | Rigid body and robot examples |
| `coupling/` | 11 | Multi-physics coupling |
| `drone/` | 7 | Quadcopter simulations |
| `IPC_Solver/` | 5 | IPC contact examples |
| `locomotion/` | 4 | Quadruped training |
| `manipulation/` | 4 | Grasping and manipulation |
| `rendering/` | 4 | Rendering demos |
| `sensors/` | 4 | Sensor examples |
| `sap_coupling/` | 4 | SAP coupling examples |
| `collision/` | 3 | Collision demos |
| `speed_benchmark/` | 3 | Performance benchmarks |

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
