import genesis as gs
import logging
import argparse

import numpy as np


def main():
    gs.init(backend=gs.gpu, logging_level=logging.DEBUG, performance_mode=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--ipc", action="store_true", default=False)
    parser.add_argument("--vis_ipc", action="store_true", default=False)
    parser.add_argument("-v", "--vis", action="store_true", default=False)
    args = parser.parse_args()

    dt = 1e-2

    coupler_options = (
        gs.options.IPCCouplerOptions(
            dt=dt,
            gravity=(0.0, 0.0, -9.8),
            ipc_constraint_strength=(100, 100),  # (translation, rotation) strength ratios,
            contact_friction_mu=0.8,
            IPC_self_contact=False,  # Disable rigid-rigid contact in IPC
            two_way_coupling=True,  # Enable two-way coupling (forces from IPC to Genesis rigid bodies)
            enable_ipc_gui=args.vis_ipc,
        )
        if args.ipc
        else None
    )
    args.vis = args.vis or args.vis_ipc

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=dt, gravity=(0.0, 0.0, -9.8)),
        coupler_options=coupler_options,
        show_viewer=args.vis,
    )

    # Both FEM and Rigid bodies will be added to IPC for unified contact simulation
    # FEM bodies use StableNeoHookean constitution, Rigid bodies use ABD constitution

    scene.add_entity(gs.morphs.Plane())

    franka = scene.add_entity(
        gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"),
    )

    scene.sim.coupler.set_ipc_link_filter(
        entity=franka,
        link_names=["left_finger", "right_finger"],
    )

    material = (
        gs.materials.FEM.Elastic(E=5.0e3, nu=0.45, rho=1000.0, model="stable_neohookean")
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
    fingers_dof = np.arange(7, 9)
    qpos = np.array([-1.0124, 1.5559, 1.3662, -1.6878, -1.5799, 1.7757, 1.4602, 0.04, 0.04])
    franka.set_qpos(qpos)
    scene.step()
    end_effector = franka.get_link("hand")
    qpos = franka.inverse_kinematics(
        link=end_effector,
        pos=np.array([0.65, 0.0, 0.135]),
        quat=np.array([0, 1, 0, 0]),
    )
    franka.control_dofs_position(qpos[:-2], motors_dof)
    # hold
    for i in range(int(0.1 / dt)):
        franka.set_qpos(qpos)
        scene.step()

    current_kp = franka.get_dofs_kp()
    new_kp = current_kp
    new_kp[fingers_dof] = current_kp[fingers_dof] * 10
    franka.set_dofs_kp(new_kp)

    print(f"New kp: {franka.get_dofs_kp()}")
    # grasp
    finder_pos = 0.0
    for i in range(int(0.1 / dt)):
        franka.control_dofs_position(qpos[:-2], motors_dof)
        franka.control_dofs_position(np.array([finder_pos, finder_pos]), fingers_dof)
        scene.step()
    # lift
    qpos = franka.inverse_kinematics(
        link=end_effector,
        pos=np.array([0.65, 0.0, 0.3]),
        quat=np.array([0, 1, 0, 0]),
    )

    for i in range(int(0.2 / dt)):
        franka.control_dofs_position(qpos[:-2], motors_dof)
        franka.control_dofs_position(np.array([finder_pos, finder_pos]), fingers_dof)
        scene.step()


if __name__ == "__main__":
    main()
