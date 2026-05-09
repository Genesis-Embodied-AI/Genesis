"""
Depth Camera with Deforming Kinematic Entity + Articulated Rigid Robot
======================================================================

End-to-end demonstration of multi-solver raycasting and per-sensor link
resolution.  The scene contains:

- **Rigid entities** (RigidSolver): ground plane + Go2 quadruped robot
- **Kinematic entities** (KinematicSolver): a deforming sphere (opted-in
  via ``use_visual_raycasting``) and a static box (NOT opted-in, invisible
  to raycasters — verifies phase-3 per-entity filtering).

Two depth cameras are attached to *different solvers*:

1. A camera on the Go2's ``base`` link (rigid) — looks forward.
2. A camera on the plane (rigid, world-fixed) — third-person overhead view.

Both cameras see the rigid robot AND the opted-in kinematic sphere, while
the non-opted-in kinematic box is invisible to rays (phase-3 filtering).

Depth frames are saved to ``/tmp/depth_out/`` as grayscale PNGs:
- ``cam_robot_XXXX.png`` — Go2-mounted camera
- ``cam_world_XXXX.png`` — world-fixed camera

Usage
-----
    python depth_camera_custom_vverts.py -v          # with Genesis 3D viewer
    python depth_camera_custom_vverts.py             # headless
    python depth_camera_custom_vverts.py -v -B 4     # batched
"""

import argparse
import os
import tempfile
from pathlib import Path

import numpy as np
import torch
import trimesh

import genesis as gs


# ----------------------------- helpers ---------------------------------


def create_sphere_mesh(radius=0.3, subdivisions=3):
    mesh = trimesh.creation.icosphere(subdivisions=subdivisions, radius=radius)
    return mesh.vertices.astype(np.float32), mesh.faces.astype(np.int32)


def wave_deform(verts, t, amplitude=0.08, freq=4.0):
    deformed = verts.copy()
    deformed[:, 2] += amplitude * np.sin(freq * verts[:, 0] + t) * np.cos(freq * verts[:, 1] + t * 0.7)
    deformed[:, 0] += amplitude * 0.5 * np.sin(freq * verts[:, 2] + t * 1.3)
    return deformed


def save_depth_png(depth_img, out_path, max_range):
    depth_np = depth_img.detach().cpu().numpy()
    valid = np.isfinite(depth_np) & (depth_np < max_range)
    normalized = np.where(valid, 1.0 - depth_np / max_range, 0.0)
    gray = (normalized * 255).astype(np.uint8)
    try:
        from PIL import Image

        img = Image.fromarray(gray).resize((gray.shape[1] * 4, gray.shape[0] * 4), Image.NEAREST)
        img.save(out_path)
    except ImportError:
        np.save(out_path.with_suffix(".npy"), gray)


# ----------------------------- main ------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Multi-solver depth camera demo")
    parser.add_argument("-v", "--vis", action="store_true", help="Open Genesis 3D viewer")
    parser.add_argument("-B", "--num_envs", type=int, default=0, help="Number of parallel envs (0 = unbatched)")
    parser.add_argument("--steps", type=int, default=300, help="Number of simulation steps")
    parser.add_argument("--save-every", type=int, default=10, help="Save depth PNG every N steps")
    parser.add_argument("--out-dir", default="/tmp/depth_out", help="Directory for saved depth PNGs")
    args = parser.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("cam_*.png"):
        old.unlink()

    gs.init(backend=gs.gpu)

    scene = gs.Scene(
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(3.0, -3.0, 2.0),
            camera_lookat=(0.0, 0.0, 0.3),
            camera_fov=45,
            max_FPS=60,
        ),
        rigid_options=gs.options.RigidOptions(dt=0.01, gravity=(0, 0, -9.81)),
        show_viewer=args.vis,
    )

    # =====================================================================
    # Rigid entities (RigidSolver)
    # =====================================================================
    plane = scene.add_entity(gs.morphs.Plane())

    go2 = scene.add_entity(
        gs.morphs.URDF(
            file="urdf/go2/urdf/go2.urdf",
            pos=(0.0, 0.0, 0.42),
        ),
    )

    # =====================================================================
    # Kinematic entities (KinematicSolver)
    # =====================================================================

    # 1) Deforming sphere — opted-IN via use_visual_raycasting.
    #    Visible to both depth cameras.
    sphere_verts, sphere_faces = create_sphere_mesh(radius=0.25, subdivisions=3)
    sphere_verts[:, 2] += 0.5
    sphere_verts[:, 0] += 1.0  # place to the right of the robot

    mesh_sphere = trimesh.Trimesh(vertices=sphere_verts, faces=sphere_faces, process=False)
    tmp_sphere = tempfile.NamedTemporaryFile(suffix=".obj", delete=False)
    mesh_sphere.export(tmp_sphere.name)

    kin_sphere = scene.add_entity(
        morph=gs.morphs.Mesh(file=tmp_sphere.name, pos=(0, 0, 0), fixed=True),
        material=gs.materials.Kinematic(use_visual_raycasting=True),
        surface=gs.surfaces.Default(color=(0.2, 0.8, 0.4)),
    )

    # 2) Static box — NOT opted-in (use_visual_raycasting=False, default).
    #    Visible in the 3D viewer but INVISIBLE to both depth cameras.
    #    This verifies phase-3 per-entity filtering.
    box_tri = trimesh.creation.box(extents=(0.3, 0.3, 0.3))
    box_verts = box_tri.vertices.astype(np.float32)
    box_verts[:, 2] += 0.15
    box_verts[:, 1] += 1.0  # place behind the robot
    box_faces = box_tri.faces.astype(np.int32)

    mesh_box = trimesh.Trimesh(vertices=box_verts, faces=box_faces, process=False)
    tmp_box = tempfile.NamedTemporaryFile(suffix=".obj", delete=False)
    mesh_box.export(tmp_box.name)

    kin_box_no_raycast = scene.add_entity(
        morph=gs.morphs.Mesh(file=tmp_box.name, pos=(0, 0, 0), fixed=True),
        material=gs.materials.Kinematic(),  # use_visual_raycasting=False (default)
        surface=gs.surfaces.Default(color=(1.0, 0.3, 0.3)),
    )

    # =====================================================================
    # Depth camera sensors — on different solvers
    # =====================================================================
    max_range = 5.0
    cam_res = (96, 72)

    # Camera 1: mounted on Go2 base link (RigidSolver entity)
    cam_robot = scene.add_sensor(
        gs.sensors.DepthCamera(
            pattern=gs.sensors.DepthCameraPattern(res=cam_res, fov_horizontal=90.0),
            entity_idx=go2.idx,
            link_idx_local=0,  # base link
            pos_offset=(0.3, 0.0, 0.1),
            euler_offset=(0.0, 0.0, 0.0),
            max_range=max_range,
            return_world_frame=True,
        ),
    )

    # Camera 2: mounted on the plane (RigidSolver, world-fixed, third-person)
    cam_world = scene.add_sensor(
        gs.sensors.DepthCamera(
            pattern=gs.sensors.DepthCameraPattern(res=cam_res, fov_horizontal=60.0),
            entity_idx=plane.idx,
            link_idx_local=0,
            pos_offset=(-1.5, 0.0, 1.5),
            euler_offset=(0.0, 45.0, 0.0),
            max_range=max_range,
            return_world_frame=True,
        ),
    )

    # =====================================================================
    # Build
    # =====================================================================
    if args.num_envs > 0:
        scene.build(n_envs=args.num_envs)
    else:
        scene.build()

    os.unlink(tmp_sphere.name)
    os.unlink(tmp_box.name)

    # Set Go2 to a standing pose
    joint_names = [
        "FR_hip_joint",
        "FR_thigh_joint",
        "FR_calf_joint",
        "FL_hip_joint",
        "FL_thigh_joint",
        "FL_calf_joint",
        "RR_hip_joint",
        "RR_thigh_joint",
        "RR_calf_joint",
        "RL_hip_joint",
        "RL_thigh_joint",
        "RL_calf_joint",
    ]
    standing_angles = [0.0, 0.8, -1.5, 0.0, 0.8, -1.5, 0.0, 1.0, -1.5, 0.0, 1.0, -1.5]
    dofs_idx = [go2.get_joint(name).dofs_idx_local[0] for name in joint_names]
    go2.set_dofs_position(standing_angles, dofs_idx)

    B = max(1, args.num_envs)
    print("=" * 65)
    print("Scene entities:")
    print(f"  [RIGID]      plane       idx={plane.idx}")
    print(f"  [RIGID]      go2         idx={go2.idx}  (articulated, {go2.n_links} links)")
    print(f"  [KINEMATIC]  kin_sphere   idx={kin_sphere.idx}  use_visual_raycasting=True")
    print(f"  [KINEMATIC]  kin_box      idx={kin_box_no_raycast.idx}  use_visual_raycasting=False")
    print()
    print("Depth cameras:")
    print(f"  cam_robot : on go2 base link (entity_idx={go2.idx})")
    print(f"  cam_world : on plane          (entity_idx={plane.idx})")
    print()
    print(f"Output: {out_dir}/cam_robot_XXXX.png, cam_world_XXXX.png")
    print(f"  (view live:  feh --reload 0.1 {out_dir})")
    print("=" * 65)
    print()

    try:
        for step in range(args.steps):
            t = step * 0.05

            # Deform the kinematic sphere each frame
            if args.num_envs > 0:
                all_verts = np.stack([wave_deform(sphere_verts, t + b * 0.5) for b in range(B)], axis=0)
                kin_sphere.set_vverts(all_verts)
            else:
                kin_sphere.set_vverts(wave_deform(sphere_verts, t))

            scene.step()

            # Read both depth cameras
            depth_robot = cam_robot.read_image()
            depth_world = cam_world.read_image()

            img_robot = depth_robot[0] if args.num_envs > 0 else depth_robot
            img_world = depth_world[0] if args.num_envs > 0 else depth_world

            if step % args.save_every == 0:
                save_depth_png(img_robot, out_dir / f"cam_robot_{step:04d}.png", max_range)
                save_depth_png(img_world, out_dir / f"cam_world_{step:04d}.png", max_range)

            if step % 50 == 0:
                valid_r = torch.isfinite(img_robot) & (img_robot < max_range)
                valid_w = torch.isfinite(img_world) & (img_world < max_range)
                print(
                    f"  step {step:4d}:  cam_robot hits={valid_r.sum().item():5d}  "
                    f"cam_world hits={valid_w.sum().item():5d}",
                    flush=True,
                )

    except KeyboardInterrupt:
        print("\nInterrupted.", flush=True)

    print(f"\nDone! Depth PNGs at: {out_dir}")


if __name__ == "__main__":
    main()
