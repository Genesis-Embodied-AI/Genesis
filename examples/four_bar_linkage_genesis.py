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
    ),
    rigid_options=gs.options.RigidOptions(
        constraint_solver=gs.constraint_solver.Newton,
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
rigid.qpos.from_numpy(np.array([0.09920997, 0.04553844, 0.10750141])[:, None])
rigid.dofs_state.vel.from_numpy(np.array([-0.22096718, -1.19638867, -3.11377182])[:, None])
rigid._kernel_forward_kinematics_links_geoms()
import time
for i in range(10000):
    scene.step()

    # print("state", rigid.qpos.to_numpy().reshape(-1), rigid.dofs_state.acc.to_numpy().reshape(-1))
    # print("efc_force", rigid.constraint_solver.efc_force.to_numpy().reshape(-1))
    # print("qacc_warmstart", rigid.constraint_solver.qacc_ws.to_numpy().reshape(-1))
    # if i >= 0:
    #     from IPython import embed; embed()
    # # import ipdb; ipdb.set_trace()
