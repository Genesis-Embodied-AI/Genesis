"""
One step optimization for the basic debugging of differentiable rigid simulation.
"""
import torch
import genesis as gs
import matplotlib.pyplot as plt

show_viewer = False

gs.init(precision="32", logging_level="warn", backend=gs.cpu)

dt = 1e-2
horizon = 1
substeps = 1
goal_pos = gs.tensor([0.0, 0.0, 1.0])

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
        pos=(0, 0, 0.109),  # small penetration with ground
        radius=0.1,
    ),
    surface=gs.surfaces.Default(
        color=(0.9, 0.0, 0.0, 1.0),
    ),
)

ground = scene.add_entity(
    gs.morphs.Box(
        pos=(0, 0, 0),
        size=(5.0, 5.0, 0.02),
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

num_iter = 400
lr = 1e-4

init_pos = gs.tensor([0.0, 0.0, 0.0], requires_grad=True)
#init_quat = gs.tensor([1.0, 0.0, 0.0, 0.0], requires_grad=True)
optimizer = torch.optim.Adam([init_pos], lr=lr)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_iter, eta_min=1e-3)

prev_loss = float('inf')
losses = []
for iter in range(num_iter):
    scene.reset()

    ground.set_pos(init_pos)
 #   ground.set_quat(init_quat)
    
    for i in range(horizon):
        scene.step()

    ball_state = ball.get_state()
    ball_pos = ball_state.pos
    loss = torch.abs(ball_pos - goal_pos).sum()
    
    optimizer.zero_grad()
    loss.backward()  # this lets gradient flow all the way back to tensor input
    optimizer.step()
    scheduler.step()

    grad_norm = torch.nn.utils.clip_grad_norm_(init_pos.grad, 1.0)

    # with torch.no_grad():
    #     init_quat.data = init_quat / torch.norm(init_quat, dim=-1, keepdim=True)
    # with torch.no_grad():
    #     init_pos.data[0] = 0.0
    #     init_pos.data[1] = 0.0
        
    print(f"Loss: {prev_loss:.6g} -> {loss.item():.6g}")
    prev_loss = loss.item()

    losses.append(loss.item())

    plt.plot(losses)
    plt.savefig("loss.png")
    plt.close()