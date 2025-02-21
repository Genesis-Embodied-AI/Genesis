import genesis as gs

import argparse

import numpy as np

def run_sim(scene, enable_vis):
    from time import time

    t_prev = time()
    i = 0
    while True:
        i += 1

        scene.step()

        t_now = time()
        print(1 / (t_now - t_prev), "FPS")
        t_prev = t_now
        if i > 200:
            break

    if enable_vis:

        scene.viewer.stop()

########################## init ##########################



########################## create a scene ##########################
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", default=False)
    args = parser.parse_args()

    ########################## init ##########################
    gs.init(
        backend=gs.cpu,  # Use 8 CPU threads
        seed=0,
        precision="32",
        logging_level="debug"
    )

    scene = gs.Scene(
    sim_options=gs.options.SimOptions(
        dt=4e-3,
        substeps=10,
        ),
        sph_options=gs.options.SPHOptions(
        lower_bound=(-0.5, -0.5, 0.0),
        upper_bound=(0.5, 0.5, 1),
        particle_size=0.03,
    ),
    vis_options=gs.options.VisOptions(
        visualize_sph_boundary=True,
    ),
    viewer_options=gs.options.ViewerOptions(
    
        res = (2560,1664),
        camera_pos=(3.5, 5.0, 2.5),  # Move the camera back further
        camera_lookat=(0.0, 2.0, 1.5),
        camera_fov=40,
    ),
    show_viewer=args.vis,
)
    
########################## entities ##########################
    plane = scene.add_entity(
        morph=gs.morphs.Plane(),
    )

    liquid = scene.add_entity(
    # viscous liquid
    # material=gs.materials.SPH.Liquid(mu=0.02, gamma=0.02),
        material=gs.materials.SPH.Liquid(),
        morph=gs.morphs.Box(
            pos=(0.0, 0.0, 0.65),
            size=(0.3, 0.3, 0.3),
        ),
        surface=gs.surfaces.Default(
            color=(0.4, 0.8, 1.0),
            vis_mode="particle",
        ),
    )

########################## build ##########################
    scene.build()

    gs.tools.run_in_another_thread(fn=run_sim, args=(scene, args.vis))
    if args.vis:
        scene.viewer.start()

# get particle positions
    particles = liquid.get_particles()

if __name__ == "__main__":
    main()   
