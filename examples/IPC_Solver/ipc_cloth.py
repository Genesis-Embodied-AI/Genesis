"""
IPC Cloth Simulation Example

This example demonstrates cloth simulation using IPC (Incremental Potential Contact)
with Genesis. The cloth is simulated using NeoHookeanShell constitution.
"""

import argparse
import logging

from huggingface_hub import snapshot_download

import genesis as gs


def main():
    gs.init(backend=gs.gpu, logging_level=logging.INFO, performance_mode=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", default=False)
    parser.add_argument("--vis_ipc", action="store_true", default=False)
    args = parser.parse_args()

    dt = 2e-3
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=dt, gravity=(0.0, 0.0, -9.8)),
        coupler_options=gs.options.IPCCouplerOptions(
            dt=dt,
            gravity=(0.0, 0.0, -9.8),
            contact_d_hat=0.01,  # Contact barrier distance (10mm) - must be appropriate for mesh resolution
            contact_friction_mu=0.3,  # Friction coefficient
            IPC_self_contact=False,  # Disable rigid self-contact in IPC
            two_way_coupling=True,  # Enable two-way coupling (forces from IPC to Genesis rigid bodies)
            disable_genesis_ground_contact=True,  # Disable Genesis ground contact to avoid double contact handling
            enable_ipc_gui=args.vis_ipc,
        ),
        show_viewer=args.vis,
    )
    args.vis = args.vis or args.vis_ipc

    # Ground plane
    scene.add_entity(gs.morphs.Plane())

    # Cloth using Cloth material
    # Note: Using coarse grid mesh to avoid IPC thickness violations
    # The built-in cloth.obj is too dense for IPC's contact detection
    asset_path = snapshot_download(
        repo_type="dataset",
        repo_id="Genesis-Intelligence/assets",
        revision="72b04f7125e21df1bebd54a7f7b39d1cd832331c",
        allow_patterns="grid20x20.obj",
        max_workers=1,
    )
    cloth = scene.add_entity(
        morph=gs.morphs.Mesh(
            file=f"{asset_path}/grid20x20.obj",
            scale=2.0,
            pos=(0.0, 0.0, 1.5),
            euler=(0, 0, 0),
        ),
        material=gs.materials.FEM.Cloth(
            E=10e5,  # Young's modulus (Pa) - soft cloth (10 kPa)
            nu=0.499,  # Poisson's ratio - nearly incompressible
            rho=200,  # Density (kg/m³)
            thickness=0.001,  # Shell thickness (m) - 1mm
            bending_stiffness=50.0,  # Bending resistance
        ),
        surface=gs.surfaces.Plastic(color=(0.3, 0.5, 0.8, 1.0), double_sided=True),
    )

    cube_size = 0.2
    cube_height = 0.3  # Height below cloth

    scene.add_entity(
        morph=gs.morphs.Box(
            pos=(0, 0, cube_height),
            size=(cube_size, cube_size, cube_size),
        ),
        material=gs.materials.Rigid(rho=500, friction=0.3),
        surface=gs.surfaces.Plastic(color=(0.8, 0.3, 0.2, 0.8)),
    )

    # Optional: Add another FEM volume object
    soft_ball = scene.add_entity(
        morph=gs.morphs.Sphere(pos=(0.5, 0.0, 0.1), radius=0.08),
        material=gs.materials.FEM.Elastic(E=1.0e3, nu=0.3, rho=1000.0, model="stable_neohookean"),
        surface=gs.surfaces.Plastic(color=(0.2, 0.8, 0.3, 0.8)),
    )

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


if __name__ == "__main__":
    main()
