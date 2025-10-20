"""
IPC Cloth Simulation Example

This example demonstrates cloth simulation using IPC (Incremental Potential Contact)
with Genesis. The cloth is simulated using NeoHookeanShell constitution.
"""

import argparse

import genesis as gs
import logging


def main():
    gs.init(backend=gs.gpu, logging_level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", default=False)
    parser.add_argument("--vis_ipc", action="store_true", default=False)
    args = parser.parse_args()

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=2e-3, gravity=(0.0, 0.0, -9.8)),
        coupler_options=gs.options.IPCCouplerOptions(
            dt=2e-3,  
            gravity=(0.0, 0.0, -9.8),
            contact_d_hat=0.01,  # Contact barrier distance (10mm) - must be appropriate for mesh resolution
            contact_friction_mu=0.3,  # Friction coefficient
            IPC_self_contact=False,  # Enable cloth self-collision
            two_way_coupling=True,  # Enable two-way coupling (forces from IPC to Genesis rigid bodies
            enable_ipc_gui=args.vis_ipc,
        ),
        show_viewer=args.vis,
    )
    args.vis = args.vis or args.vis_ipc

    # Ground plane
    scene.add_entity(gs.morphs.Plane())

    SCENE_POS = (0.0, 0.0, 0.0)

    # Cloth using Cloth material
    # Note: Using coarse grid mesh to avoid IPC thickness violations
    # The built-in cloth.obj is too dense for IPC's contact detection
    cloth = scene.add_entity(
        morph=gs.morphs.Mesh(
            file="meshes/grid20x20.obj",
            scale=2.0, 
            pos=tuple(map(sum, zip(SCENE_POS, (0.0, 0.0, 0.3)))),
            euler=(90, 0, 0),
        ),
        material=gs.materials.Cloth(
            E=10e5,  # Young's modulus (Pa) - soft cloth (10 kPa)
            nu=0.499,  # Poisson's ratio - nearly incompressible
            rho=200,  # Density (kg/m³) - typical fabric
            thickness=0.001,  # Shell thickness (m) - 1mm
            bending_stiffness=100.0,  # Bending resistance
        ),
        surface=gs.surfaces.Plastic(color=(0.3, 0.5, 0.8, 1.0), double_sided=True),
    )

    # Add 16 rigid cubes uniformly distributed under the cloth (4x4 grid)
    cube_size = 0.2
    cube_height = 0.3  # Height below cloth
    grid_spacing = 0.6  # Spacing between cubes

    for i in range(4):
        for j in range(4):
            x = (i - 1.5) * grid_spacing  # Center the grid
            y = (j - 1.5) * grid_spacing
            scene.add_entity(
                morph=gs.morphs.Box(
                    pos=tuple(map(sum, zip(SCENE_POS, (x, y, cube_height)))),
                    size=(cube_size, cube_size, cube_size),
                ),
                material=gs.materials.Rigid(rho=500, friction=0.3),
                surface=gs.surfaces.Plastic(color=(0.8, 0.3, 0.2, 0.8)),
            )

    # Optional: Add another FEM volume object
    # soft_ball = scene.add_entity(
    #     morph=gs.morphs.Sphere(
    #         pos=tuple(map(sum, zip(SCENE_POS, (0.2, 0.0, 0.6)))),
    #         radius=0.08
    #     ),
    #     material=gs.materials.FEM.Elastic(
    #         E=1.0e5, nu=0.45, rho=1000.0, model="stable_neohookean"
    #     ),
    #     surface=gs.surfaces.Plastic(color=(0.2, 0.8, 0.3, 0.8)),
    # )

    gs.logger.info("Building scene...")
    scene.build(n_envs=1)
    gs.logger.info("Scene built successfully!")

    # Simulation loop
    print("\nRunning simulation...")
    horizon = 1000
    for i in range(horizon):
        scene.step()
        if i % 100 == 0:
            print(f"  Step {i}/{horizon}")

    print("\nSimulation completed successfully!")


if __name__ == "__main__":
    main()
