import genesis as gs

import numpy as np
gs.init(backend=gs.cpu, precision="64")

scene = gs.Scene(
    viewer_options=gs.options.ViewerOptions(
        camera_pos=(10, 0, 10),
        camera_lookat=(0.0, 0.0, 3),
        camera_fov=60,
    ),
    sim_options=gs.options.SimOptions(
        dt=1e-2,
        gravity=(0, 0, 0),
    ),
    rigid_options=gs.options.RigidOptions(
        constraint_solver=gs.constraint_solver.CG,
    ),
)

# plane = scene.add_entity(
#     gs.morphs.Plane(),
# )
franka = scene.add_entity(
    gs.morphs.MJCF(
        file="xml/four_bar_linkage.xml",
        # euler=(10, 10, 10),
        # pos=(0, 0, 0.5),
        # scale=0.05
    ),
)

scene.build()
rigid = scene.sim.rigid_solver
rigid.qpos.from_numpy(np.array([0.1, 0.1, 0.1])[:, None])
rigid._kernel_forward_kinematics_links_geoms()
import time
for i in range(10000):
    print("i-----------------", i)
    scene.step()
    if i >= 0:
        from IPython import embed; embed()
    # import ipdb; ipdb.set_trace()
