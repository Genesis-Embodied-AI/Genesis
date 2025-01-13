# Drone Examples

This directory contains examples of drone simulations using the Genesis framework.

## Available Examples

### 1. Interactive Drone (`interactive_drone.py`)
A real-time interactive drone simulation where you can control the drone using keyboard inputs:
- ↑ (Up Arrow): Move Forward (North)
- ↓ (Down Arrow): Move Backward (South)
- ← (Left Arrow): Move Left (West)
- → (Right Arrow): Move Right (East)
- Space: Increase Thrust (Accelerate)
- Left Shift: Decrease Thrust (Decelerate)
- ESC: Quit

Run with:
```bash
python interactive_drone.py -v -m
```

### 2. Automated Flight (`fly.py`)
A pre-programmed drone flight simulation that follows a predefined trajectory stored in `fly_traj.pkl`.

Run with:
```bash
python fly.py -v -m
```

### 3. Hover Environment (`hover_env.py`, `hover_train.py`, `hover_eval.py`)

The hover environment (`hover_env.py`) is designed to train a drone to maintain a stable hover position by reaching randomly generated target points. The environment includes:

 - Initialization of the scene and entities (plane, drone and target).
 - Reward functions to provide feedback to the agent based on its performance in reaching the target points.
 - **Command resampling to generate new random target points** and environment reset functionalities to ensure continuous training.

**Acknowledgement**: The reward design is inspired by [Champion-level drone racing using deep
reinforcement learning (Nature 2023)](https://www.nature.com/articles/s41586-023-06419-4.pdf)

#### 3.0 Installation

At this stage, we have defined the environments. Now, we use the PPO implementation from `rsl-rl` to train the policy. Follow these installation steps:

```bash
# Install rsl_rl.
git clone https://github.com/leggedrobotics/rsl_rl
cd rsl_rl && git checkout v1.0.2 && pip install -e .

# Install tensorboard.
pip install tensorboard
```

#### 3.1 Training

Train the drone hovering policy using the `HoverEnv` environment.

Run with:

```bash
python hover_train.py -e drone-hovering -B 8192 --max_iterations 500
```

#### 3.2 Evaluation

Evaluate the trained drone hovering policy.

Run with:

```bash
python hover_eval.py -e drone-hovering --ckpt 500 --record
```

**Note**: If you experience slow performance or encounter other issues 
during evaluation, try removing the `--record` option.

## Technical Details

- The drone model used is the Crazyflie 2.X (`urdf/drones/cf2x.urdf`)
- Base hover RPM is approximately 14468
- Movement is achieved by varying individual rotor RPMs to create directional thrust
- The simulation uses realistic physics including gravity and aerodynamics
- Visualization is optimized for macOS using threaded rendering when run with `-m` flag

## Controls Implementation

The interactive drone uses differential RPM control:
- Forward/Backward movement: Adjusts front/back rotor pairs
- Left/Right movement: Adjusts left/right rotor pairs
- All movements maintain a stable hover while creating directional thrust
- RPM changes are automatically clipped to safe ranges (0-25000 RPM)