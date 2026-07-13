"""
QIPC Coupler: Franka Panda 7-DOF with PD position control.

Validates multi-joint pipeline: Genesis MJCF → QIPCCoupler (single JointCollection) → PD tracking.
"""
import math

import genesis as gs

gs.init(precision="64", logging_level="info")

scene = gs.Scene(
    sim_options=gs.options.SimOptions(dt=0.01, gravity=(0.0, 0.0, -9.81)),
    coupler_options=gs.options.QIPCCouplerOptions(
        rigid_abd_kappa=1e8,
        joint_kappa_pivot=1e8,
        joint_kappa_axis=1e8,
        default_kp=200.0,
        default_kv=1000.0,
        contact_enable=False,
        debug_viewer=False,
    ),
    show_viewer=True,
)

panda = scene.add_entity(
    morph=gs.morphs.MJCF(
        file="xml/franka_emika_panda/panda.xml",
        pos=(0, 0, 0),
    ),
)

scene.build()

FREQ = 0.3
viewer = scene.sim._coupler._scene.viewer
viewer.up_axis = "z"
i = 0

while viewer.show():
    if viewer.want_step:
        t = scene.sim.cur_t
        targets = [0.3 * math.sin(2 * math.pi * FREQ * t + j * 0.5) for j in range(panda.n_dofs)]
        panda.control_dofs_position(targets)
        scene.step()

        if i % 50 == 0:
            qpos = panda.get_dofs_position()
            print(f"step {i:4d} | t={t:.2f}s | qpos[0]={float(qpos[0]):.4f}")
        i += 1

qpos = panda.get_dofs_position()
print(f"\nFinal: {[f'{float(q):.3f}' for q in qpos]}")
