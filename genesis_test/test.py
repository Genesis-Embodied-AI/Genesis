import argparse
import genesis as gs
from time import time

def run_sim(scene, enable_vis):
    t_prev = time()
    i = 0
    while True:
        i += 1
        scene.step()
        
        t_now = time()
        print(1 / (t_now - t_prev), "FPS")
        t_prev = t_now
        
        if i > 1000:  # You can adjust this number
            break

    if enable_vis:
        scene.viewer.stop()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", default=False)
    args = parser.parse_args()

    gs.init(backend=gs.cpu)

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(3.5, 0.0, 2.5),
            camera_lookat=(0.0, 0.0, 0.5),
            camera_fov=40,
        ),
        show_viewer=args.vis,
    )
    
    _ = scene.add_entity(gs.morphs.Plane())
    scene.build()

    gs.tools.run_in_another_thread(fn=run_sim, args=(scene, args.vis))
    if args.vis:
        scene.viewer.start()

if __name__ == "__main__":
    main()