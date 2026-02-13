# Genesis Architecture

## Project Structure

```
Genesis/
├── genesis/                    # Main source code
│   ├── __init__.py            # Entry point, gs.init(), global state
│   ├── engine/                # Core simulation engine
│   │   ├── scene.py           # Scene class - main API entry point
│   │   ├── simulator.py       # Manages all solvers
│   │   ├── entities/          # Entity types (RigidEntity, MPMEntity, etc.)
│   │   ├── solvers/           # Physics solvers
│   │   │   ├── rigid/         # Rigid body solver
│   │   │   ├── mpm_solver.py  # Material Point Method
│   │   │   ├── sph_solver.py  # Smoothed Particle Hydrodynamics
│   │   │   ├── fem_solver.py  # Finite Element Method
│   │   │   ├── pbd_solver.py  # Position Based Dynamics
│   │   │   └── sf_solver.py   # Stable Fluid
│   │   ├── materials/         # Material definitions per solver
│   │   └── couplers/          # Inter-solver coupling
│   ├── options/               # Configuration classes (Pydantic models)
│   │   ├── morphs.py          # Shape definitions (Box, Mesh, URDF, MJCF)
│   │   ├── solvers.py         # Solver configuration options
│   │   └── surfaces.py        # Surface properties
│   ├── vis/                   # Visualization (Visualizer, Camera, Viewer)
│   ├── sensors/               # Sensor systems (camera, IMU, etc.)
│   └── utils/                 # Utilities (mesh, geometry, etc.)
├── tests/                     # Test files
├── examples/                  # Example scripts
└── genesis/assets/            # Built-in meshes, URDFs, textures
```

## Core Components Flow

```
gs.init() → Scene → Simulator → Solvers → Entities
                 ↓
            Visualizer → Viewer / Cameras
```

## Entities

Entities are physical objects in the simulation:

| Entity Type | Solver | Use Case |
|-------------|--------|----------|
| `RigidEntity` | Rigid | Robots, rigid objects |
| `MPMEntity` | MPM | Deformable solids, granular materials |
| `SPHEntity` | SPH | Liquids, fluids |
| `FEMEntity` | FEM | Finite element deformable bodies |
| `PBD2DEntity`, `PBD3DEntity` | PBD | Cloth, soft bodies |
| `DroneEntity` | Rigid | Quadcopters with aerodynamics |
| `ToolEntity` | Tool | Cutting/interaction tools |

Location: `genesis/engine/entities/`

## Morphs

Morphs define geometry and initial pose (solver-agnostic):

```python
# Primitives
gs.morphs.Box(size=(1, 1, 1), pos=(0, 0, 0.5))
gs.morphs.Sphere(radius=0.5, pos=(0, 0, 1))
gs.morphs.Plane()

# Robot descriptions
gs.morphs.URDF(file="path/to/robot.urdf", fixed=True)
gs.morphs.MJCF(file="path/to/robot.xml")
```

Location: `genesis/options/morphs.py`

## Materials

Materials define physical properties and determine which solver handles the entity:

```python
gs.materials.Rigid(rho=1000)
gs.materials.MPM.Elastic(E=1e5, nu=0.3)
gs.materials.SPH.Liquid(sampler="pbs")
gs.materials.PBD.Cloth(stretch_compliance=0.0)
```

Location: `genesis/engine/materials/`

## Solvers

| Solver | Options Class | Purpose |
|--------|--------------|---------|
| Rigid | `gs.options.RigidOptions` | Articulated rigid body dynamics |
| MPM | `gs.options.MPMOptions` | Continuum mechanics |
| SPH | `gs.options.SPHOptions` | Fluid simulation |
| FEM | `gs.options.FEMOptions` | Finite element deformation |
| PBD | `gs.options.PBDOptions` | Fast soft body simulation |
| SF | `gs.options.SFOptions` | Eulerian fluid/smoke |

Location: `genesis/engine/solvers/`

## Key Files Reference

| File | Purpose |
|------|---------|
| `genesis/__init__.py` | Package entry, `gs.init()`, global state |
| `genesis/engine/scene.py` | `Scene` class - main user interface |
| `genesis/engine/simulator.py` | `Simulator` - manages all solvers |
| `genesis/options/morphs.py` | Shape/geometry definitions |
| `genesis/options/solvers.py` | Solver option classes |
