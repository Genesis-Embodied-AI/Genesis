import numpy as np

import genesis as gs

########################## init ##########################
gs.init(backend=gs.gpu)

########################## create a scene ##########################

scene = gs.Scene(
    show_viewer=False,
    rigid_options=gs.options.RigidOptions(
        dt=0.01,
        constraint_solver=gs.constraint_solver.Newton,
    ),
)

########################## entities ##########################
scene.add_entity(
    gs.morphs.Plane(),
)
robot = scene.add_entity(
    gs.morphs.URDF(
        file="urdf/anymal_c/urdf/anymal_c.urdf",
        pos=(0, 0, 0.8),
    ),
)
########################## build ##########################
n_envs = 30000
scene.build(n_envs=n_envs)

joint_names = [
    "RH_HAA",
    "LH_HAA",
    "RF_HAA",
    "LF_HAA",
    "RH_HFE",
    "LH_HFE",
    "RF_HFE",
    "LF_HFE",
    "RH_KFE",
    "LH_KFE",
    "RF_KFE",
    "LF_KFE",
]
motor_dofs = [robot.get_joint(name).dof_idx_local for name in joint_names]

robot.set_dofs_kp(np.full(12, 1000), motor_dofs)
robot.control_dofs_position(np.zeros((n_envs, 12)), motor_dofs)

# Speed: 14.4M FPS
for i in range(1000):
    scene.step()
