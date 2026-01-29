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

    dt = 1e-3

    coupler_options = (
        gs.options.IPCCouplerOptions(
            dt=dt,
            gravity=(0.0, 0.0, -9.8),
            ipc_constraint_strength=(3, 3),  # (translation, rotation) strength ratios,
            contact_friction_mu=0.8,
            use_contact_proxy=True,
            # disable_ipc_logging   = False,
            enable_ipc_gui=args.vis_ipc,
        )
        if args.ipc
        else None
    )
    args.vis = args.vis or args.vis_ipc
    rigid_options = gs.options.RigidOptions(
        enable_collision=False,  # Disable rigid collision when using IPC
    )
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=dt, gravity=(0.0, 0.0, -9.8)),
        rigid_options=rigid_options,
        coupler_options=coupler_options,
        show_viewer=args.vis,
    )

    scene.add_entity(gs.morphs.Plane())

    material = (
        gs.materials.FEM.Elastic(E=5.0e3, nu=0.45, rho=1000.0, model="stable_neohookean")
        if args.ipc
        else gs.materials.Rigid()
    )
    material = gs.materials.Rigid()
    cube = scene.add_entity(
        morph=gs.morphs.Box(pos=(0.65, 0.0, 0.1), size=(0.05, 0.05, 0.05)),
        material=material,
        surface=gs.surfaces.Plastic(color=(0.2, 0.8, 0.2, 0.5)),
    )

    scene.build()
    print("Scene built successfully!")

    for i in range(int(10 / dt)):

        scene.step()


if __name__ == "__main__":
    main()
