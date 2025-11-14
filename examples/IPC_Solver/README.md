## IPC examples

This repo shows how to use IPC coupler. (Incremental Potential Contact) algorithm provides a unified framework for robust contact handling across different material types including cloth, deformable FEM objects, and rigid bodies.


**Prerequisites:**
1. Build and install libuipc from source: https://github.com/spiriMirror/libuipc
2. Ensure Genesis is in development mode: `pip install -e .`

**Test Cases:**

1. **Basic cloth simulation:**
   ```bash
   python examples/IPC_Solver/ipc_cloth.py -v
   ```
  Expected: A cloth sheet falls under gravity and collides with the ground plane

2. **Robotic grasping of deformables:**
   ```bash
   python examples/IPC_Solver/ipc_grasp.py -v --ipc
   ```
 Expected: Franka Panda arm grasps and manipulates a deformable object with IPC contact resolution

3. **Interactive cloth manipulation (requires trajectory data):**
   ```bash
   python examples/IPC_Solver/ipc_arm_cloth.py -v --ipc
   ```
    Expected: Playback of recorded trajectory showing arm-cloth interaction

   ```bash
   python examples/IPC_Solver/ipc_twist_cloth_band.py -v
   ```
    Expected: four rigid cubes hold and twist a cloth band
    

**Verification:**
- No interpenetration between objects during contact
- Cloth renders correctly (visible mesh with proper shading)
- Two-way coupling: Genesis rigid body motion affects IPC objects and vice versa
