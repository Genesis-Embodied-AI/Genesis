import numpy as np
import genesis as gs
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pile_type", type=str, default="falling", choices=("static", "falling"))
    parser.add_argument("--num_cubes", type=int, default=5, choices=(5, 6, 7, 8, 9, 10))
    parser.add_argument("--cpu", action="store_true", help="Use CPU backend instead of GPU", default=True)
    parser.add_argument("--steps", type=int, default=150)
    parser.add_argument("-v", "--vis", action="store_true", default=False)
    args = parser.parse_args()

    gs.init(backend=gs.cpu if args.cpu else gs.gpu, precision="32")

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=0.01,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0, -5.5, 2.5),
            camera_lookat=(0, 0.0, 1.5),
            max_FPS=60,
        ),
        rigid_options=gs.options.RigidOptions(
            use_contact_island=True,
            use_hibernation=True,
            # Relaxed thresholds for stacked structures - contact forces create
            # constant small accelerations that prevent hibernation with defaults
            hibernation_thresh_vel=1e-2,
            hibernation_thresh_acc=0.2,
        ),
        show_viewer=args.vis,
    )

    plane = scene.add_entity(gs.morphs.Plane())

    # create pyramid of boxes
    box_size = 0.25
    box_spacing = (1.0 - 1e-3 + 0.1 * (args.pile_type == "static")) * box_size
    box_pos_offset = (-0.5, 1, 0.0) + 0.5 * np.array([box_size, box_size, box_size])
    boxes = {}
    for i in range(args.num_cubes):
        for j in range(args.num_cubes - i):
            box = scene.add_entity(
                gs.morphs.Box(
                    size=[box_size, box_size, box_size],
                    pos=box_pos_offset + box_spacing * np.array([i + 0.5 * j, 0, j]),
                ),
                # visualize_contact=True,
            )

    scene.build()

    solver = scene.sim.rigid_solver
    for i in range(args.steps):
        scene.step()
        # hibernated = solver.entities_state.hibernated.to_numpy().mean()
        # vel = solver.dofs_state.vel.to_numpy().mean()
        # acc = solver.dofs_state.acc.to_numpy().mean()
        # print(f"Acc: {np.abs(acc).max()}")
        # print(f"Vel: {np.abs(vel).max()}")
        # if hibernated > 0.0:
        #     print(f"Hibernated: {hibernated}")
        #     from IPython import embed; embed()
        #     break


if __name__ == "__main__":
    main()
