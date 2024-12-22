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