## IPC examples

This repo shows how to use IPC coupler. (Incremental Potential Contact) algorithm provides a unified framework for robust contact handling across different material types including cloth, deformable FEM objects, and rigid bodies.

**Prerequisites:**
1. Install LibUIPC optional dependency: `pip install pyuipc`
2. Ensure Genesis is in development mode: `pip install -e .`

Beware only Linux/Windows x86 CPU & Nvidia GPU is supported for now.

**Test Cases:**

1. **Basic cloth simulation:**
   ```bash
   python examples/IPC_Solver/ipc_cloth_falling.py -v
   ```
  Expected: A cloth falls under gravity and collides with the ground plane clustered with objects

1. **Robotic grasping of deformables:**
   ```bash
   python examples/IPC_Solver/ipc_robot_cloth_grasp.py -v --ipc
   ```
 Expected: Franka Panda robot grasps and manipulates a deformable object with IPC contact resolution

1. **Interactive cloth manipulation:**
   ```bash
   python examples/IPC_Solver/ipc_robot_cloth_telop.py -v --ipc
   ```
    Expected: Interactive Franka Panda robot manipulation and manipulates two pieces of cloths


**Verification:**
- No interpenetration between objects during contact
- Cloth renders correctly (visible mesh with proper shading)
- Two-way coupling: Genesis rigid body motion affects IPC objects and vice versa
