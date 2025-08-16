import numpy as np
import argparse
import genesis as gs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", default=False)
    args = parser.parse_args()
    gs.init(backend=gs.gpu, precision="64")
    show_viewer = args.vis

    camera_pos = (1.5, 0.0, 0.5)
    camera_lookat = (0.65, 0.0, 0.1)
    camera_up = (0, 0, 1)
    camera_fov = 30
    res = (1920, 1080)
    scene = gs.Scene(
        viewer_options=gs.options.ViewerOptions(
            camera_pos=camera_pos,
            camera_lookat=camera_lookat,
            camera_fov=camera_fov,
            camera_up=camera_up,
            res=res,
            max_FPS=60,
        ),
        sim_options=gs.options.SimOptions(
            dt=1.0 / 60,
            substeps=2,
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
        show_viewer=show_viewer,
    )

    ########################## entities ##########################
    friction = 1.0
    force = 1.0
    franka = scene.add_entity(
        gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"),
        material=gs.materials.Rigid(coup_friction=friction, friction=friction),
    )
    fem_material_linear_corotated_soft = gs.materials.FEM.Elastic(
        model="linear_corotated", friction_mu=friction, E=1e5, nu=0.4
    )
    sphere = scene.add_entity(
        morph=gs.morphs.Sphere(
            radius=0.02,
            pos=np.array([0.65, 0.0, 0.02], dtype=np.float32),
        ),
        material=fem_material_linear_corotated_soft,
    )
    ########################## build ##########################
    scene.build()

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
    for i in range(10):
        print("hold", i)
        scene.step()
    # grasp
    for i in range(30):
        print("grasp", i)
        franka.control_dofs_position(qpos[:-2], motors_dof)
        franka.control_dofs_force(np.array([-force, -force]), fingers_dof)
        scene.step()

    # lift
    qpos = franka.inverse_kinematics(
        link=end_effector,
        pos=np.array([0.65, 0.0, 0.3]),
        quat=np.array([0, 1, 0, 0]),
    )
    for i in range(100):
        print("lift", i)
        franka.control_dofs_position(qpos[:-2], motors_dof)
        franka.control_dofs_force(np.array([-force, -force]), fingers_dof)
        scene.step()


if __name__ == "__main__":
    main()
