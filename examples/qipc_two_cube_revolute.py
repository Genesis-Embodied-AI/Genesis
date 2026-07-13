"""
QIPC Coupler MVP: two_cube_revolute.urdf with PD sinusoidal tracking.

Validates the full coupling pipeline:
  Genesis parses URDF → QIPCCoupler reads entity data → builds QIPC scene →
  PD control forwarded → QIPC steps → state written back to Genesis buffers.

Both Genesis viewer and QIPC viewer run simultaneously for visual debugging.
"""
import math

import genesis as gs

gs.init(precision="64", logging_level="info")

scene = gs.Scene(
    sim_options=gs.options.SimOptions(dt=0.01, gravity=(0.0, 0.0, -9.81)),
    coupler_options=gs.options.QIPCCouplerOptions(
        rigid_abd_kappa=1e8,
        joint_kappa_pivot=1e5,
        joint_kappa_axis=1e5,
        default_kp=500.0,
        default_kv=50.0,
        contact_enable=False,
        debug_viewer=True,
    ),
    show_viewer=True,
)

robot = scene.add_entity(
    morph=gs.morphs.URDF(
        file="urdf/simple/two_cube_revolute.urdf",
        pos=(0, 0, 0.3),
        fixed=True,
    ),
)

scene.build()


FREQ = 0.5
AMPLITUDE = 0.8
N_STEPS = 5000000

for i in range(N_STEPS):
    t = scene.sim.cur_t
    target = AMPLITUDE * math.sin(2 * math.pi * FREQ * t)
    robot.control_dofs_position(target)
    scene.step()

    if i % 50 == 0:
        qpos = robot.get_dofs_position()
        print(f"step {i:4d} | t={t:.3f}s | target={target:.4f} | qpos={float(qpos[0]):.4f}")

qpos = robot.get_dofs_position()
print(f"\nFinal joint angle: {float(qpos[0]):.4f} rad")
print("Visual check: Genesis viewer and QIPC viewer should show the same robot pose.")
