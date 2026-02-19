"""
Debug script for two-cube IPC joint coupling regressions.

This script mirrors the logic in IPC unit tests for:
- two_cube_revolute.urdf
- two_cube_prismatic.urdf

It prints per-step diagnostics for:
- Genesis vs IPC transform mismatch on link `moving`
- base link z motion (for fixed or free base)
- whether the moving link was actually added to IPC metadata

Example runs:
python examples/IPC_Solver/ipc_two_cube_joint_debug.py -v --vis_ipc --joint prismatic --coupling_type external_articulation --free-base
python examples/IPC_Solver/ipc_two_cube_joint_debug.py -v --vis_ipc --joint revolute --coupling_type two_way_soft_constraint --free-base
"""

import argparse

import numpy as np

import genesis as gs

try:
    from uipc.backend import SceneVisitor
    from uipc.geometry import SimplicialComplexSlot
except ImportError as exc:
    raise RuntimeError("This debug script requires `uipc`.") from exc


def _read_ipc_geometry_metadata(geo):
    try:
        meta_attrs = geo.meta()

        solver_type_attr = meta_attrs.find("solver_type")
        if not solver_type_attr or solver_type_attr.name() != "solver_type":
            return None
        solver_type_view = solver_type_attr.view()
        if len(solver_type_view) == 0:
            return None
        solver_type = str(solver_type_view[0])

        env_idx_attr = meta_attrs.find("env_idx")
        if not env_idx_attr:
            return None
        env_idx = int(str(env_idx_attr.view()[0]))

        if solver_type == "rigid":
            idx_attr = meta_attrs.find("link_idx")
        elif solver_type in ("fem", "cloth"):
            idx_attr = meta_attrs.find("entity_idx")
        else:
            return None

        if not idx_attr:
            return None
        idx = int(str(idx_attr.view()[0]))
        return (solver_type, env_idx, idx)
    except Exception:
        return None


def _get_ipc_rigid_links(scene, env_idx=0):
    visitor = SceneVisitor(scene.sim.coupler._ipc_scene)
    links = set()
    for geo_slot in visitor.geometries():
        if not isinstance(geo_slot, SimplicialComplexSlot):
            continue
        meta = _read_ipc_geometry_metadata(geo_slot.geometry())
        if meta is None:
            continue
        solver_type, meta_env_idx, idx = meta
        if solver_type == "rigid" and meta_env_idx == env_idx:
            links.add(idx)
    return links


def _entity_base_z(entity):
    pos = np.asarray(entity.get_pos().detach().cpu().numpy())
    if pos.ndim == 1:
        return float(pos[2])
    return float(pos.reshape(-1, 3)[0, 2])


def _rotmat_to_euler(rot_mat):
    sy = np.sqrt(rot_mat[0, 0] ** 2 + rot_mat[1, 0] ** 2)
    singular = sy < 1e-6
    if not singular:
        x = np.arctan2(rot_mat[2, 1], rot_mat[2, 2])
        y = np.arctan2(-rot_mat[2, 0], sy)
        z = np.arctan2(rot_mat[1, 0], rot_mat[0, 0])
    else:
        x = np.arctan2(-rot_mat[1, 2], rot_mat[1, 1])
        y = np.arctan2(-rot_mat[2, 0], sy)
        z = 0.0
    return np.array([x, y, z])


def _build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", default=False)
    parser.add_argument("--vis_ipc", action="store_true", default=False)
    parser.add_argument("--joint", choices=["revolute", "prismatic"], default="revolute")
    parser.add_argument(
        "--coupling_type",
        type=str,
        default="external_articulation",
        choices=["two_way_soft_constraint", "external_articulation"],
    )
    parser.add_argument("--fixed-base", dest="fixed_base", action="store_true")
    parser.add_argument("--free-base", dest="fixed_base", action="store_false")
    parser.set_defaults(fixed_base=False)
    parser.add_argument("--dt", type=float, default=1e-2)
    parser.add_argument("--robot_z", type=float, default=0.5)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--check_from", type=int, default=50)
    parser.add_argument("--print_every", type=int, default=10)
    parser.add_argument("--ground_z_threshold", type=float, default=0.08)
    parser.add_argument("--settle_steps", type=int, default=10)
    parser.add_argument(
        "--control_mode",
        choices=["control", "set"],
        default="control",
        help="control: use PD targets (same style as ipc_test_joint.py); set: use set_dofs_position (same style as unit tests).",
    )
    parser.add_argument("--hold", action="store_true", default=False)
    return parser


def _target_qpos(joint, step_idx, dt):
    t = step_idx * dt
    omega = 2.0 * np.pi
    if joint == "revolute":
        return 0.5 * np.sin(omega * t)
    return 0.15 + 0.1 * np.sin(omega * t)


def main():
    parser = _build_parser()
    args = parser.parse_args()

    gs.init(backend=gs.gpu, logging_level="debug")

    dt = args.dt
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=dt,
            gravity=(0.0, 0.0, -9.8),
        ),
        rigid_options=gs.options.RigidOptions(enable_collision=False),
        coupler_options=gs.options.IPCCouplerOptions(
            dt=dt,
            gravity=(0.0, 0.0, -9.8),
            contact_friction_mu=0.5,
            ipc_constraint_strength=(1, 1),
            IPC_self_contact=False,
            two_way_coupling=True,
            disable_ipc_ground_contact=False,
            disable_ipc_logging=True,
            newton_velocity_tol=1e-2,
            newton_transrate_tol=1e-2,
            linear_system_tol_rate=1e-3,
            sync_dof_enable=False,
            newton_semi_implicit_enable=False,
            enable_ipc_gui=args.vis_ipc,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(1.5, 0.0, 1.0),
            camera_lookat=(0.0, 0.0, 0.5),
            camera_fov=40,
        ),
        show_viewer=(args.vis or args.vis_ipc),
    )

    scene.add_entity(gs.morphs.Plane())

    urdf_file = (
        "urdf/simple/two_cube_revolute.urdf" if args.joint == "revolute" else "urdf/simple/two_cube_prismatic.urdf"
    )
    robot = scene.add_entity(
        morph=gs.morphs.URDF(
            file=urdf_file,
            pos=(0.0, 0.0, args.robot_z),
            fixed=args.fixed_base,
        ),
    )
    scene.sim.coupler.set_entity_coupling_type(entity=robot, coupling_type=args.coupling_type)
    scene.build(n_envs=0)

    moving_link_idx = robot.get_link("moving").idx
    base_link_idx = robot.get_link("base").idx
    ipc_links = _get_ipc_rigid_links(scene, env_idx=0)
    coupler = scene.sim.coupler

    print("\n=== IPC Joint Debug Config ===")
    print(f"joint={args.joint}")
    print(f"coupling_type={args.coupling_type}")
    print(f"fixed_base={args.fixed_base}")
    print(f"dt={dt}")
    print(f"control_mode={args.control_mode}")
    print(f"steps={args.steps}")
    print(f"moving_link_idx={moving_link_idx}, base_link_idx={base_link_idx}")
    print(f"moving in IPC metadata: {moving_link_idx in ipc_links}")
    print(f"moving in _link_to_abd_slot: {(0, moving_link_idx) in coupler._link_to_abd_slot}")
    print(f"base in IPC metadata: {base_link_idx in ipc_links}")
    print(f"base in _link_to_abd_slot: {(0, base_link_idx) in coupler._link_to_abd_slot}")

    initial_base_z = _entity_base_z(robot)
    print(f"initial base z={initial_base_z:.6f}")

    max_pos_diff = 0.0
    max_rot_diff = 0.0
    worst_pos_step = -1
    worst_rot_step = -1
    min_base_z = initial_base_z

    if args.settle_steps > 0:
        zero_target = np.zeros(robot.n_dofs, dtype=np.float32)
        for _ in range(args.settle_steps):
            robot.control_dofs_position(zero_target)
            scene.step()

    for i in range(args.steps):
        qpos = _target_qpos(args.joint, i, dt)
        if args.control_mode == "control":
            robot.control_dofs_position([qpos], [robot.n_dofs - 1])
        else:
            robot.set_dofs_position([qpos], zero_velocity=False)
        scene.step()

        pos_diff = np.nan
        rot_diff = np.nan
        if (
            i >= args.check_from
            and hasattr(coupler, "abd_data_by_link")
            and moving_link_idx in coupler.abd_data_by_link
            and 0 in coupler.abd_data_by_link[moving_link_idx]
        ):
            abd_data = coupler.abd_data_by_link[moving_link_idx][0]
            genesis_transform = abd_data["aim_transform"]
            ipc_transform = abd_data["transform"]
            if genesis_transform is not None and ipc_transform is not None:
                genesis_pos = genesis_transform[:3, 3]
                ipc_pos = ipc_transform[:3, 3]
                pos_diff = float(np.linalg.norm(genesis_pos - ipc_pos))
                if pos_diff > max_pos_diff:
                    max_pos_diff = pos_diff
                    worst_pos_step = i

                genesis_euler = _rotmat_to_euler(genesis_transform[:3, :3])
                ipc_euler = _rotmat_to_euler(ipc_transform[:3, :3])
                rot_diff = float(np.linalg.norm(genesis_euler - ipc_euler))
                if rot_diff > max_rot_diff:
                    max_rot_diff = rot_diff
                    worst_rot_step = i

        if i % args.print_every == 0:
            base_z = _entity_base_z(robot)
            min_base_z = min(min_base_z, base_z)
            q_now = robot.get_qpos().detach().cpu().numpy().reshape(-1)[-1]
            print(
                f"step={i:4d} target={qpos:+.4f} q={q_now:+.4f} "
                f"pos_diff={pos_diff:.6f} rot_diff={rot_diff:.6f} base_z={base_z:.6f}"
            )
        else:
            min_base_z = min(min_base_z, _entity_base_z(robot))

    final_base_z = _entity_base_z(robot)
    base_drop = initial_base_z - final_base_z
    min_base_z = min(min_base_z, final_base_z)
    touched_ground = min_base_z <= args.ground_z_threshold

    print("\n=== Summary ===")
    print(f"max position diff={max_pos_diff:.6f} at step {worst_pos_step} (unit-test target < 0.001)")
    print(f"max rotation diff={max_rot_diff:.6f} at step {worst_rot_step} (unit-test target < 0.1)")
    print(f"final base z={final_base_z:.6f}, base drop={base_drop:.6f} (free-base target >= 0.03)")
    print(f"min base z={min_base_z:.6f}, touched_ground={touched_ground} (threshold={args.ground_z_threshold:.3f})")

    if args.hold:
        viewer = getattr(scene, "viewer", None)
        if viewer is not None:
            print("Hold mode enabled. Close viewer window to exit.")
            while viewer.is_alive():
                scene.step()


if __name__ == "__main__":
    main()
