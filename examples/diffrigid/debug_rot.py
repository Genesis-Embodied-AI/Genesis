"""
One step optimization for the basic debugging of differentiable rigid simulation.
"""
import torch
import genesis as gs
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
matplotlib.use("Agg")

show_viewer = False

gs.init(precision="32", logging_level="warn", backend=gs.cpu)

dt = 1e-2
horizon = 10
substeps = 1
grad_window = None
np.random.seed(0)
goal_quat = np.random.randn(4)
goal_quat = goal_quat / np.linalg.norm(goal_quat)
goal_quat = gs.tensor(goal_quat)

scene = gs.Scene(
    sim_options=gs.options.SimOptions(dt=dt, substeps=substeps, requires_grad=True, grad_window_steps=grad_window),
    rigid_options=gs.options.RigidOptions(
        use_gjk_collision=True,
        enable_joint_limit=False,
    ),
    show_viewer=show_viewer,
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
ball = scene.add_entity(
    gs.morphs.Sphere(
        pos=(0, 0, 0.109),  # small penetration with ground
        radius=0.1,
    ),
    surface=gs.surfaces.Default(
        color=(0.9, 0.0, 0.0, 1.0),
    ),
)
cam = scene.add_camera(
    pos=(3.5, 0.5, 2.5),
    lookat=(0.0, 0.0, 0.5),
    fov=40,
    GUI=False,
)

scene.build()

num_iter = 10000
lr = 1e-2

force = gs.zeros((horizon, 6), requires_grad=True)
with torch.no_grad():
    torch.manual_seed(0)
    force.data[:, 3:] = torch.randn_like(force.data[:, 3:])
optimizer = torch.optim.Adam([force], lr=lr)

render_every = 100
prev_loss = float('inf')
losses = []
for iter in range(num_iter):
    scene.reset()

    curr_losses = []
    if iter % render_every == 0: 
        cam.start_recording()   
    for i in range(horizon):
        curr_force = force[i]
        ground.control_dofs_force(curr_force)
        scene.step()

        box_state = ground.get_state()
        box_quat = box_state.quat
        curr_loss = (box_quat - goal_quat).abs().sum()
        curr_losses.append(curr_loss)
        
        if iter % render_every == 0: 
            cam.render()
    if iter % render_every == 0: 
        cam.stop_recording(save_to_filename=f"video_{iter:06d}.mp4", fps=30)
    
    loss = sum(curr_losses) / len(curr_losses)
    optimizer.zero_grad()
    loss.backward()  # this lets gradient flow all the way back to tensor input
    optimizer.step()
    grad_norm = torch.nn.utils.clip_grad_norm_(force.grad, 1.0)

    with torch.no_grad():
        force.data[:, :3] = 0.0
        
    print(f"Loss: {prev_loss:.6g} -> {loss.item():.6g} | Grad Norm: {grad_norm:.6g} | Force: {force.data.mean(dim=0).cpu().numpy().tolist()}")
    prev_loss = loss.item()

    losses.append(loss.item())

    plt.plot(losses)
    plt.savefig("loss.png")
    plt.close()