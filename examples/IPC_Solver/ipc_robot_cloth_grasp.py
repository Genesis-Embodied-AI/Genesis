import argparse
import os

import numpy as np

import genesis as gs


def main():
    gs.init(backend=gs.gpu, logging_level="debug")

    parser = argparse.ArgumentParser()
    parser.add_argument("--ipc", action="store_true", default=False)
    parser.add_argument("--vis_ipc", action="store_true", default=False)
    parser.add_argument("-v", "--vis", action="store_true", default=False)
    parser.add_argument(
        "--coupling_type",
        type=str,
        default="external_articulation",
        choices=["two_way_soft_constraint", "external_articulation"],
    )
    args = parser.parse_args()

    dt = 1e-2

    coupler_options = (
        gs.options.IPCCouplerOptions(
            dt=dt,
            gravity=(0.0, 0.0, -9.8),
            ipc_constraint_strength=(100, 100),  # (translation, rotation) strength ratios,
            # coupling_strategy="external_articulation",
            disable_ipc_ground_contact=True,
            disable_ipc_logging=True,
            IPC_self_contact=False,
            contact_friction_mu=0.8,
            enable_ipc_gui=args.vis_ipc,
            newton_transrate_tol=10,
        )
        if args.ipc
        else None
    )
    args.vis = args.vis or args.vis_ipc

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=dt, gravity=(0.0, 0.0, -9.8)),
        rigid_options=None,
        coupler_options=coupler_options,
        show_viewer=args.vis,
    )

    # Both FEM and Rigid bodies will be added to IPC for unified contact simulation
    # FEM bodies use StableNeoHookean constitution, Rigid bodies use ABD constitution

    scene.add_entity(gs.morphs.Plane())

    franka = scene.add_entity(
        gs.morphs.MJCF(file="xml/franka_emika_panda/panda_non_overlap.xml"),
    )

    if args.ipc:
        scene.sim.coupler.set_entity_coupling_type(
            entity=franka,
            coupling_type=args.coupling_type,
        )
        scene.sim.coupler.set_ipc_coupling_link_filter(
            entity=franka,
            link_names=["left_finger", "right_finger"],
        )

    material = (
        gs.materials.FEM.Elastic(E=5.0e4, nu=0.45, rho=1000.0, model="stable_neohookean")
        if args.ipc
        else gs.materials.Rigid()
    )

    cube = scene.add_entity(
        morph=gs.morphs.Box(pos=(0.65, 0.0, 0.03), size=(0.05, 0.05, 0.05)),
        material=material,
        surface=gs.surfaces.Plastic(color=(0.2, 0.8, 0.2, 0.5)),
    )

    scene.build()
    print("Scene built successfully!")

    motors_dof = np.arange(7)
    # qpos = np.array([-1.0124, 1.5559, 1.3662, -1.6878, -1.5799, 1.7757, 1.4602, 0.04, 0.04])
    fingers_dof = np.arange(7, 9)
    current_kp = franka.get_dofs_kp()
    new_kp = current_kp
    new_kp[fingers_dof] = current_kp[fingers_dof] * 5.0
    franka.set_dofs_kp(new_kp)
    end_effector = franka.get_link("hand")
    qpos = franka.inverse_kinematics(
        link=end_effector,
        pos=np.array([0.65, 0.0, 0.4]),
        quat=np.array([0, 1, 0, 0]),
    )

    # hold
    for _ in range(int(2 / dt) if "PYTEST_VERSION" not in os.environ else 1):
        franka.control_dofs_position(qpos[:-2], motors_dof)
        franka.control_dofs_position(np.array([0.04, 0.04]), fingers_dof)
        scene.step()
    qpos = franka.inverse_kinematics(
        link=end_effector,
        pos=np.array([0.65, 0.0, 0.25]),
        quat=np.array([0, 1, 0, 0]),
    )

    # hold
    for _ in range(int(1 / dt) if "PYTEST_VERSION" not in os.environ else 1):
        franka.control_dofs_position(qpos[:-2], motors_dof)
        franka.control_dofs_position(np.array([0.04, 0.04]), fingers_dof)
        scene.step()
    qpos = franka.inverse_kinematics(
        link=end_effector,
        pos=np.array([0.65, 0.0, 0.135]),
        quat=np.array([0, 1, 0, 0]),
    )

    # hold
    for _ in range(int(0.5 / dt) if "PYTEST_VERSION" not in os.environ else 1):
        franka.control_dofs_position(qpos[:-2], motors_dof)
        franka.control_dofs_position(np.array([0.04, 0.04]), fingers_dof)
        scene.step()

    # grasp
    finder_pos = 0.0
    for _ in range(int(0.1 / dt) if "PYTEST_VERSION" not in os.environ else 1):
        franka.control_dofs_position(qpos[:-2], motors_dof)
        franka.control_dofs_position(np.array([finder_pos, finder_pos]), fingers_dof)
        scene.step()

    # lift
    qpos = franka.inverse_kinematics(
        link=end_effector,
        pos=np.array([0.65, 0.0, 0.4]),
        quat=np.array([0, 1, 0, 0]),
    )

    for _ in range(int(0.2 / dt) if "PYTEST_VERSION" not in os.environ else 1):
        franka.control_dofs_position(qpos[:-2], motors_dof)
        franka.control_dofs_position(np.array([finder_pos, finder_pos]), fingers_dof)
        scene.step()


if __name__ == "__main__":
    main()
