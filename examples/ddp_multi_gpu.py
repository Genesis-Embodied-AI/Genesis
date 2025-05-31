#!/usr/bin/env python3
"""
Multi-node / multi-GPU Genesis ✕ PyTorch DDP demo
=================================================

Single machine, 2 GPUs:
    torchrun --standalone --nnodes=1 --nproc_per_node=2 examples/ddp_multi_gpu.py

Expectation:
    - In nvidia-smi, you will see multiple GPUs are being used.
    - As you increase the number of GPUs, the gradient will be less noisy and the loss decreases faster.
"""

import os, argparse, random, numpy as np
import torch, torch.nn as nn, torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import genesis as gs


class TinyMLP(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, act_dim),
        )

    def forward(self, x):
        return self.net(x.float())


def run_worker(args: argparse.Namespace) -> None:
    # setup
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    os.environ["CUDA_VISIBLE_DEVICES"] = str(local_rank)
    os.environ["TI_VISIBLE_DEVICE"] = str(local_rank)
    # FIXME: Forcing rendering device is not working reliably on all machines
    # os.environ["EGL_DEVICE_ID"] = str(local_rank)
    gs.init(backend=gs.gpu, seed=local_rank)

    # sim
    scene = gs.Scene(
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(3.5, 0.0, 2.5),
            camera_lookat=(0.0, 0.0, 0.5),
            camera_fov=40,
        ),
        show_viewer=False,
        show_FPS=False,
    )
    scene.add_entity(gs.morphs.Plane())
    scene.add_entity(
        gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"),
        visualize_contact=True,
    )
    scene.build(n_envs=args.n_envs)

    # model
    gpu_id = 0
    torch.cuda.set_device(gpu_id)
    dist.init_process_group(backend="nccl", init_method="env://")
    device = torch.device("cuda", gpu_id)

    rigid = scene.sim.rigid_solver
    qpos = rigid.get_qpos()
    obs_dim = qpos.shape[1]
    act_dim = 1
    model = TinyMLP(obs_dim, act_dim).to(device)
    model = DDP(model, device_ids=[gpu_id])
    optim = torch.optim.Adam(model.parameters(), lr=3e-4)

    # train loop
    for step in range(args.steps):
        scene.step()
        qpos = rigid.get_qpos()

        obs = qpos + torch.randn_like(qpos)

        logits = model(obs)
        target = qpos.sum(dim=1, keepdim=True)
        loss = torch.nn.functional.mse_loss(logits, target)

        optim.zero_grad(set_to_none=True)
        loss.backward()  # DDP handles all-reduce, gradients are averaged
        optim.step()

        if local_rank == 0 and step % 100 == 0:
            print(f"[{step:04d}/{args.steps}] loss = {loss.item():.6f}")

    # cleanup
    dist.barrier()  # sync all ranks before shutting down NCCL
    dist.destroy_process_group()
    gs.destroy()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=1000, help="simulation / training steps")
    p.add_argument("--vis", action="store_true", help="open viewer on rank-0")
    p.add_argument("--n_envs", type=int, default=2048, help="number of environments")
    return p.parse_args()


if __name__ == "__main__":
    run_worker(parse_args())
