# QIPCCoupler

QIPC coupler integrates [cuda-graph-qipc](https://github.com/Genesis-Embodied-AI/cuda-graph-qipc) as the physics backend for rigid/articulated entities in Genesis.

## Installing cuda-graph-qipc

Clone the repo and build into the genesis-world venv:

```bash
git clone https://github.com/Genesis-Embodied-AI/cuda-graph-qipc.git
cd cuda-graph-qipc
python build.py --genesis
```

The `--genesis` flag installs the built package into the active genesis-world virtual environment. Rebuild after any changes to cuda-graph-qipc.

## Quick Start

```python
import genesis as gs

gs.init(precision="64")

scene = gs.Scene(
    sim_options=gs.options.SimOptions(
        dt=0.01,
        gravity=(0.0, 0.0, -9.81),
    ),
    coupler_options=gs.options.QIPCCouplerOptions(
        contact_enable=False,
    ),
)

robot = scene.add_entity(
    morph=gs.morphs.URDF(
        file="urdf/simple/two_cube_revolute.urdf",
        pos=(0, 0, 0.3),
        fixed=True,
    ),
    material=gs.materials.Rigid(
        qipc_abd_kappa=1e8,
        qipc_kappa_pivot=1e5,
        qipc_kappa_axis=1e5,
        qipc_default_kp=500.0,
        qipc_default_kv=50.0,
    ),
)

scene.build()

for _ in range(100):
    robot.control_dofs_position(0.5)
    scene.step()
```

See `examples/qipc/` for more examples.
