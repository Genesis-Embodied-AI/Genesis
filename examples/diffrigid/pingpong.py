import torch
import genesis as gs

show_viewer = True

gs.init(precision="32", logging_level="warn", backend=gs.cpu)

dt = 1e-2
horizon = 200
substeps = 4
goal_pos = gs.tensor([0.0, 0.1, -0.1])

scene = gs.Scene(
    sim_options=gs.options.SimOptions(dt=dt, substeps=substeps, requires_grad=True),
    rigid_options=gs.options.RigidOptions(
        use_gjk_collision=True,
        enable_joint_limit=False,
    ),
    show_viewer=show_viewer,
)

ball = scene.add_entity(
    gs.morphs.Sphere(
        pos=(0, 0, 0.5),
        radius=0.1,
    ),
    surface=gs.surfaces.Default(
        color=(0.9, 0.0, 0.0, 1.0),
    ),
    material=gs.materials.Rigid(
        rho=0.001,
    )
)
# if show_viewer:
#     target = scene.add_entity(
#         gs.morphs.Sphere(
#             pos=goal_pos.cpu().numpy().tolist(),
#             radius=0.1,
#         ),
#         surface=gs.surfaces.Default(
#             color=(0.0, 0.9, 0.0, 0.5),
#         ),
#     )

racket = scene.add_entity(
    gs.morphs.Box(
        pos=(0, 0, 0),
        size=(5.0, 5.0, 0.01),
        #fixed=True,
    ),
    surface=gs.surfaces.Default(
        color=(0.0, 0.0, 0.9, 1.0),
    ),
    material=gs.materials.Rigid(
        gravity_compensation=1,
    )
)

scene.build()

num_iter = 200
lr = 1e-4

init_pos = gs.tensor([0.0, 0.0, 0.0], requires_grad=True)
init_quat = gs.tensor([1.0, 0.0, 0.0, 0.0], requires_grad=True)
optimizer = torch.optim.Adam([init_pos, init_quat], lr=lr)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_iter, eta_min=1e-3)

prev_loss = float('inf')
for iter in range(num_iter):
    scene.reset()

    racket.set_pos(init_pos)
    racket.set_quat(init_quat)
    #ball.set_dofs_velocity(gs.tensor([0, 0, -2.0, 0, 0, 0]))

    losses = []
    for i in range(horizon):
        scene.step()
        # ball_state = ball.get_state()
        # ball_pos = ball_state.pos
        # losses.append(torch.abs(ball_pos - goal_pos).sum())
        # if show_viewer:
        #     target.set_pos(goal_pos)

    ball_state = ball.get_state()
    ball_pos = ball_state.pos
    loss = torch.abs(ball_pos - goal_pos).sum()
    # loss = sum(losses) / len(losses)

    optimizer.zero_grad()
    loss.backward()  # this lets gradient flow all the way back to tensor input
    optimizer.step()
    scheduler.step()

    with torch.no_grad():
        init_quat.data = init_quat / torch.norm(init_quat, dim=-1, keepdim=True)
        
    print(f"Loss: {prev_loss:.6g} -> {loss.item():.6g}")
    prev_loss = loss.item()

# assert_allclose(loss, 0.0, atol=1e-2)
