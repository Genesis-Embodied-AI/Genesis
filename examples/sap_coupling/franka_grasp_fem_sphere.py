import argparse
import sys
import numpy as np
import genesis as gs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--cpu", action="store_true", default=(sys.platform == "darwin"))
    parser.add_argument("-v", "--vis", action="store_true", default=False)
    args = parser.parse_args()

    ########################## init ##########################

    gs.init(backend=gs.cpu if args.cpu else gs.gpu, precision="64", performance_mode=True)

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=1.0 / 60,
            substeps=2,
        ),
        rigid_options=gs.options.RigidOptions(
            enable_self_collision=False,
        ),
        fem_options=gs.options.FEMOptions(
            use_implicit_solver=True,
            pcg_threshold=1e-10,
        ),
        coupler_options=gs.options.SAPCouplerOptions(
            pcg_threshold=1e-10,
            sap_convergence_atol=1e-10,
            sap_convergence_rtol=1e-10,
            linesearch_ftol=1e-10,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(1.3, 0.0, 0.15),
            camera_lookat=(0.65, 0.0, 0.15),
            max_FPS=60,
        ),
        show_viewer=args.vis,
    )

    ########################## entities ##########################

    franka = scene.add_entity(
        gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"),
        material=gs.materials.Rigid(
            coup_friction=1.0,
            friction=1.0,
        ),
    )
    sphere = scene.add_entity(
        morph=gs.morphs.Sphere(
            radius=0.02,
            pos=(0.65, 0.0, 0.02),
        ),
        material=gs.materials.FEM.Elastic(
            model="linear_corotated",
            friction_mu=1.0,
            E=1e5,
            nu=0.4,
        ),
    )

    ########################## build ##########################

    scene.build()

    motors_dof = np.arange(7)
    fingers_dof = np.arange(7, 9)
    end_effector = franka.get_link("hand")

    ########################## simulate ##########################

    # init
    franka.set_qpos((-1.0124, 1.5559, 1.3662, -1.6878, -1.5799, 1.7757, 1.4602, 0.04, 0.04))

    # hold
    qpos = franka.inverse_kinematics(link=end_effector, pos=(0.65, 0.0, 0.13), quat=(0, 1, 0, 0))
    franka.control_dofs_position(qpos[motors_dof], motors_dof)
    for i in range(15):
        scene.step()

    # grasp
    for i in range(10):
        franka.control_dofs_force(np.array([-1.0, -1.0]), fingers_dof)
        scene.step()

    # lift
    qpos = franka.inverse_kinematics(link=end_effector, pos=(0.65, 0.0, 0.3), quat=(0, 1, 0, 0))
    franka.control_dofs_position(qpos[motors_dof], motors_dof)
    for i in range(40):
        franka.control_dofs_force(np.array([-1.0, -1.0]), fingers_dof)
        scene.step()


if __name__ == "__main__":
    main()
