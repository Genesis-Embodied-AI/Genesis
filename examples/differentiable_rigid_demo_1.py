import torch
import genesis as gs

show_viewer = False

gs.init(precision="32", logging_level="info")

dt = 1e-2
horizon = 100
substeps = 1
goal_pos = gs.tensor([0.7, 1.0, 0.05])
goal_quat = gs.tensor([0.3, 0.2, 0.1, 0.9])
goal_quat = goal_quat / torch.norm(goal_quat, dim=-1, keepdim=True)

scene = gs.Scene(
    sim_options=gs.options.SimOptions(dt=dt, substeps=substeps, requires_grad=True, gravity=(0, 0, -1)),
    rigid_options=gs.options.RigidOptions(
        enable_collision=False,
        enable_self_collision=False,
        enable_joint_limit=False,
        disable_constraint=True,
        use_contact_island=False,
        use_hibernation=False,
    ),
    viewer_options=gs.options.ViewerOptions(
        camera_pos=(2.5, -0.15, 2.42),
        camera_lookat=(0.5, 0.5, 0.1),
    ),
    show_viewer=show_viewer,
)

box = scene.add_entity(
    gs.morphs.Box(
        pos=(0, 0, 0),
        size=(0.1, 0.1, 0.2),
    ),
    surface=gs.surfaces.Default(
        color=(0.9, 0.0, 0.0, 1.0),
    ),
)
if show_viewer:
    target = scene.add_entity(
        gs.morphs.Box(
            pos=goal_pos,
            quat=goal_quat,
            size=(0.1, 0.1, 0.2),
        ),
        surface=gs.surfaces.Default(
            color=(0.0, 0.9, 0.0, 0.5),
        ),
    )

scene.build()

num_iter = 200
lr = 1e-2

init_pos = gs.tensor([0.3, 0.1, 0.28], requires_grad=True)
init_quat = gs.tensor([1.0, 0.0, 0.0, 0.0], requires_grad=True)
optimizer = torch.optim.Adam([init_pos, init_quat], lr=lr)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_iter, eta_min=1e-3)

for iter in range(num_iter):
    scene.reset()

    box.set_pos(init_pos)
    box.set_quat(init_quat)

    loss = 0
    for i in range(horizon):
        scene.step()
        if show_viewer:
            target.set_pos(goal_pos)
            target.set_quat(goal_quat)

    box_state = box.get_state()
    box_pos = box_state.pos
    box_quat = box_state.quat
    loss = torch.abs(box_pos - goal_pos).sum() + torch.abs(box_quat - goal_quat).sum()

    optimizer.zero_grad()
    loss.backward()  # this lets gradient flow all the way back to tensor input
    optimizer.step()
    scheduler.step()

    with torch.no_grad():
        init_quat.data = init_quat / torch.norm(init_quat, dim=-1, keepdim=True)
        
    print("loss: ", loss.item())

# assert_allclose(loss, 0.0, atol=1e-2)
