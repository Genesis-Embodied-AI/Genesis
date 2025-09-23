import argparse
import numpy as np
import genesis as gs


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", default=False)
    args = parser.parse_args()
    gs.init(backend=gs.gpu, precision="64")
    show_viewer = args.vis

    camera_pos = np.array([1.5, -1.5, 1.5], dtype=np.float32)
    camera_lookat = (0, 0, 0.0)
    camera_fov = 40
    camera_up = np.array([0, 0, 1], dtype=np.float32)
    res = (1920, 1080)

    fem_material_linear_corotated = gs.materials.FEM.Elastic(
        model="linear_corotated",
    )
    fem_material_linear_corotated_soft = gs.materials.FEM.Elastic(model="linear_corotated", E=1e5, nu=0.4)

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=1 / 60,
            substeps=2,
        ),
        fem_options=gs.options.FEMOptions(
            use_implicit_solver=True,
        ),
        coupler_options=gs.options.SAPCouplerOptions(),
        show_viewer=show_viewer,
        viewer_options=gs.options.ViewerOptions(
            camera_pos=camera_pos,
            camera_lookat=camera_lookat,
            camera_fov=camera_fov,
            camera_up=camera_up,
            res=res,
            max_FPS=60,
        ),
    )

    sphere = scene.add_entity(
        morph=gs.morphs.Sphere(
            pos=(0.0, 0.0, 0.1),
            radius=0.1,
        ),
        material=fem_material_linear_corotated_soft,
    )

    scale = 0.1
    cube = scene.add_entity(
        morph=gs.morphs.Mesh(
            file=f"meshes/cube8.obj",
            scale=scale,
            pos=(0.0, 0.0, scale * 4.0),
        ),
        material=fem_material_linear_corotated,
    )

    # Build the scene
    scene.build()

    # Run simulation
    for _ in range(200):
        scene.step()


if __name__ == "__main__":
    main()
