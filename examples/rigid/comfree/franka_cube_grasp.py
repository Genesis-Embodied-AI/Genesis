"""Franka Panda cube grasping demo using the ComFree constraint solver.

Replicates the grasp-robustness benchmark from ComFree-Sim (arXiv:2603.12185)
using the Genesis ComFree solver. The Franka arm approaches a cube on a pedestal,
grasps it, lifts it, and optionally applies perturbation forces.

Usage:
    uv run python examples/rigid/comfree/franka_cube_grasp.py
    uv run python examples/rigid/comfree/franka_cube_grasp.py -v
    uv run python examples/rigid/comfree/franka_cube_grasp.py --engine newton

Reference: references/comfree_warp/test_local/test_franka_grasp.py
Genesis equivalent of: examples/rigid/franka_cube.py
"""

import argparse
import math
import time

import numpy as np

import genesis as gs


# ── Constants ──────────────────────────────────────────────────────────────────
ARM_DOF = 7
CUBE_SIZE = 0.04  # full extent of the cube (Genesis Box uses full size)
CUBE_HALF = CUBE_SIZE / 2
PEDESTAL_TOP = 0.12  # top surface of the pedestal
LIFTED_THRESHOLD = 0.18


def smoothstep(alpha: float) -> float:
    alpha = min(1.0, max(0.0, alpha))
    return alpha * alpha * (3.0 - 2.0 * alpha)


def build_scene(engine="comfree", vis=False, dt=0.002, stiffness=0.2, damping=0.001, num_envs=1):
    """Create the Franka + cube scene (matching examples/rigid/franka_cube.py)."""

    gs.init(backend=gs.cpu, precision="32", logging_level="info", performance_mode=True)

    if engine == "comfree":
        constraint_solver = gs.constraint_solver.ComFree
    else:
        constraint_solver = gs.constraint_solver.Newton

    scene = gs.Scene(
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(3, -1, 1.5),
            camera_lookat=(0.0, 0.0, 0.5),
            camera_fov=30,
            res=(960, 640),
            max_FPS=60,
        ),
        sim_options=gs.options.SimOptions(dt=dt),
        rigid_options=gs.options.RigidOptions(
            dt=dt,
            constraint_solver=constraint_solver,
            comfree_stiffness=stiffness,
            comfree_damping=damping,
            # box_box_detection not required - MPR/GJK handles box-box pairs
        ),
        show_viewer=vis,
    )

    scene.add_entity(gs.morphs.Plane())
    franka = scene.add_entity(gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"))
    cube = scene.add_entity(
        gs.morphs.Box(size=(CUBE_SIZE, CUBE_SIZE, CUBE_SIZE), pos=(0.65, 0.0, CUBE_HALF)),
    )
    scene.build(n_envs=num_envs)
    return scene, franka, cube


def run_grasp_trial(
    engine="comfree",
    vis=False,
    dt=0.002,
    stiffness=0.2,
    damping=0.001,
    steps_scale=1.0,
    perturb=True,
    num_envs=1,
):
    """Run the full grasp trial following examples/rigid/franka_cube.py style.

    Phases (matching reference test_franka_grasp.py):
      1. approach  - IK to position above cube, hold
      2. grasp     - close fingers
      3. lift      - IK to raised position, hold while grasping
      4. hold      - hold lifted position
      5. perturb   - apply sinusoidal forces to test grasp robustness (optional)
    """
    scene, franka, cube = build_scene(
        engine=engine, vis=vis, dt=dt, stiffness=stiffness, damping=damping, num_envs=num_envs
    )

    motors_dof = np.arange(ARM_DOF)
    fingers_dof = np.arange(ARM_DOF, ARM_DOF + 2)

    # ── Initial configuration (same as franka_cube.py) ────────────────────────
    qpos = np.array([-1.0124, 1.5559, 1.3662, -1.6878, -1.5799, 1.7757, 1.4602, 0.04, 0.04])
    franka.set_qpos(qpos)
    scene.step()

    end_effector = franka.get_link("hand")

    # ── Compute IK targets ────────────────────────────────────────────────────
    # When n_envs >= 1, IK expects pos shape (n_envs, 3) and quat shape (n_envs, 4)
    approach_pos = np.tile([0.65, 0.0, 0.135], (num_envs, 1))
    lift_pos = np.tile([0.65, 0.0, 0.4], (num_envs, 1))
    ik_quat = np.tile([0, 1, 0, 0], (num_envs, 1))

    # Approach: above cube (same as franka_cube.py)
    approach_q = franka.inverse_kinematics(
        link=end_effector,
        pos=approach_pos,
        quat=ik_quat,
    ).numpy()

    # Lift: raised position (higher target for ComFree's softer contacts)
    lift_q = franka.inverse_kinematics(
        link=end_effector,
        pos=lift_pos,
        quat=ik_quat,
    ).numpy()

    # ── Phase definitions ─────────────────────────────────────────────────────
    # (name, steps, arm_qpos, finger_pos, force_scale)
    open_fingers = 0.04
    closed_fingers = 0.0  # fully closed in Genesis finger control

    # Phase step counts are calibrated for dt=0.002; scale proportionally for other dt values
    dt_scale = 0.002 / dt
    approach_arm = approach_q[:, :ARM_DOF]
    lift_arm = lift_q[:, :ARM_DOF]

    phases = [
        ("approach", int(500 * steps_scale * dt_scale), approach_arm, open_fingers, 0.0),
        ("grasp", int(500 * steps_scale * dt_scale), approach_arm, closed_fingers, 0.0),
        ("lift", int(1000 * steps_scale * dt_scale), lift_arm, closed_fingers, 0.0),
        ("hold", int(1000 * steps_scale * dt_scale), lift_arm, closed_fingers, 0.0),
    ]
    if perturb:
        phases.append(("perturb", int(2500 * steps_scale * dt_scale), lift_arm, closed_fingers, 1.5))

    # ── Simulation loop ───────────────────────────────────────────────────────
    min_cube_z = float("inf")
    max_cube_z = float("-inf")
    total_steps = 0
    wall_time_start = time.perf_counter()

    for phase_name, phase_steps, arm_target, finger_target, force_scale in phases:
        phase_steps = max(1, phase_steps)
        for local_step in range(phase_steps):
            franka.control_dofs_position(arm_target, motors_dof)
            franka.control_dofs_position(np.array([finger_target, finger_target]), fingers_dof)

            # Apply perturbation forces to hand link
            if force_scale > 0.0:
                t = local_step * dt
                fx = force_scale * math.sin(14.0 * t)
                fy = 0.75 * force_scale * math.cos(9.0 * t)
                hand_link = franka.get_link("hand")
                scene.sim.rigid_solver.apply_links_external_force(
                    force=np.array([[fx, fy, 0.0]]),
                    links_idx=[hand_link.idx_local],
                )

            scene.step()

            cube_z = cube.get_pos().numpy().flatten()[2]
            min_cube_z = min(min_cube_z, cube_z)
            max_cube_z = max(max_cube_z, cube_z)
            total_steps += 1

            if total_steps % 100 == 0:
                print(f"  [{phase_name}] step {local_step}/{phase_steps}, cube_z={cube_z:.4f}")

    wall_time = time.perf_counter() - wall_time_start
    final_cube_z = cube.get_pos().numpy().flatten()[2]
    success = final_cube_z > LIFTED_THRESHOLD

    results = {
        "engine": engine,
        "total_steps": total_steps,
        "final_cube_z": final_cube_z,
        "min_cube_z": min_cube_z,
        "max_cube_z": max_cube_z,
        "success": success,
        "wall_time": wall_time,
        "steps_per_sec": total_steps / wall_time if wall_time > 0 else 0,
        "num_envs": num_envs,
    }
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-v", "--vis", action="store_true", default=False, help="Launch viewer.")
    parser.add_argument("--engine", choices=("comfree", "newton"), default="comfree", help="Constraint solver backend.")
    parser.add_argument(
        "--dt", type=float, default=0.002, help="Simulation timestep (ComFree benefits from smaller dt)."
    )
    parser.add_argument("--stiffness", type=float, default=0.2, help="ComFree stiffness (k_user).")
    parser.add_argument("--damping", type=float, default=0.001, help="ComFree damping (d_user).")
    parser.add_argument("--steps-scale", type=float, default=1.0, help="Scale phase durations.")
    parser.add_argument("--no-perturb", action="store_true", help="Skip perturbation phase.")
    parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to run in parallel.")
    args = parser.parse_args()

    results = run_grasp_trial(
        engine=args.engine,
        vis=args.vis,
        dt=args.dt,
        stiffness=args.stiffness,
        damping=args.damping,
        steps_scale=args.steps_scale,
        perturb=not args.no_perturb,
        num_envs=args.num_envs,
    )

    print("\n" + "=" * 60)
    print(f"Engine:           {results['engine']}")
    print(f"Number of environments: {results['num_envs']}")
    print(f"Total steps:      {results['total_steps']}")
    print(f"Final cube z:     {results['final_cube_z']:.4f}")
    print(f"Min cube z:       {results['min_cube_z']:.4f}")
    print(f"Max cube z:       {results['max_cube_z']:.4f}")
    print(f"Wall time:        {results['wall_time']:.2f}s")
    print(f"Steps/sec:        {results['steps_per_sec']:.1f}")
    print(f"Success:          {results['success']}")
    print("=" * 60)

    if not results["success"]:
        print("\nGrasp trial FAILED - cube was not lifted.")
        raise SystemExit(1)
    else:
        print("\nGrasp trial PASSED - cube successfully lifted!")


if __name__ == "__main__":
    main()
