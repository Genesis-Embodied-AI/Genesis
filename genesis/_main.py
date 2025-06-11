import argparse
import multiprocessing
import os
import threading
from functools import partial

import tkinter as tk
from tkinter import ttk

import numpy as np
import torch
from taichi._lib import core as _ti_core
from taichi.lang import impl

import genesis as gs


FPS = 60


class JointControlGUI:
    def __init__(self, master, motors_name, motors_position_limit, motors_position):
        self.master = master
        self.master.title("Joint Controller")  # Set the window title
        self.motors_name = motors_name
        self.motors_position_limit = motors_position_limit
        self.motors_position = motors_position
        self.motors_default_position = np.clip(
            np.zeros(len(motors_name)), motors_position_limit[:, 0], motors_position_limit[:, 1]
        )
        self.sliders = []
        self.values_label = []
        self.create_widgets()
        self.reset_motors_position()

    def create_widgets(self):
        for i_m, name in enumerate(self.motors_name):
            self.update_joint_position(i_m, self.motors_default_position[i_m])
            min_limit, max_limit = map(float, self.motors_position_limit[i_m])
            frame = tk.Frame(self.master)
            frame.pack(pady=5, padx=10, fill=tk.X)

            tk.Label(frame, text=f"{name}", font=("Arial", 12), width=20).pack(side=tk.LEFT)

            slider = ttk.Scale(
                frame,
                from_=min_limit,
                to=max_limit,
                orient=tk.HORIZONTAL,
                length=300,
                command=partial(self.update_joint_position, i_m),
            )
            slider.pack(side=tk.LEFT, padx=5)
            self.sliders.append(slider)

            value_label = tk.Label(frame, text=f"{slider.get():.2f}", font=("Arial", 12))
            value_label.pack(side=tk.LEFT, padx=5)
            self.values_label.append(value_label)

            # Update label dynamically
            def update_label(s=slider, l=value_label):
                def callback(event):
                    l.config(text=f"{s.get():.2f}")

                return callback

            slider.bind("<Motion>", update_label())

        tk.Button(self.master, text="Reset", font=("Arial", 12), command=self.reset_motors_position).pack(pady=20)

    def update_joint_position(self, idx, val):
        self.motors_position[idx] = float(val)

    def reset_motors_position(self):
        for i_m, slider in enumerate(self.sliders):
            slider.set(self.motors_default_position[i_m])
            self.values_label[i_m].config(text=f"{self.motors_default_position[i_m]:.2f}")
            self.motors_position[i_m] = self.motors_default_position[i_m]


def get_motors_info(robot):
    motors_dof_idx = []
    motors_dof_name = []
    for joint in robot.joints:
        if joint.type == gs.JOINT_TYPE.FREE:
            continue
        elif joint.type == gs.JOINT_TYPE.FIXED:
            continue
        dofs_idx_local = robot.get_joint(joint.name).dofs_idx_local
        if dofs_idx_local:
            if len(dofs_idx_local) == 1:
                dofs_name = [joint.name]
            else:
                dofs_name = [f"{joint.name}_{i_d}" for i_d in range(dofs_idx_local)]
            motors_dof_idx += dofs_idx_local
            motors_dof_name += dofs_name
    return motors_dof_idx, motors_dof_name


def clean():
    print("Cleaned up all genesis and taichi cache files...")
    gs.utils.misc.clean_cache_files()
    _ti_core.clean_offline_cache_files(os.path.abspath(impl.default_cfg().offline_cache_file_path))


def _start_gui(motors_name, motors_position_limit, motors_position, stop_event):
    def on_close():
        nonlocal after_id
        if after_id is not None:
            root.after_cancel(after_id)
            after_id = None
        stop_event.set()
        root.destroy()
        root.quit()

    root = tk.Tk()
    app = JointControlGUI(root, motors_name, motors_position_limit, motors_position)
    root.protocol("WM_DELETE_WINDOW", on_close)

    def check_event():
        nonlocal after_id
        if stop_event.is_set():
            on_close()
        elif root.winfo_exists():
            after_id = root.after(100, check_event)

    after_id = root.after(100, check_event)
    root.mainloop()


def view(filename, collision, rotate, scale=1.0, show_link_frame=False):
    gs.init(backend=gs.cpu)

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            gravity=(0.0, 0.0, 0.0),
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(3.5, 0.0, 2.5),
            camera_lookat=(0.0, 0.0, 0.5),
            camera_fov=40,
            max_FPS=FPS,
        ),
        vis_options=gs.options.VisOptions(
            show_link_frame=show_link_frame,
        ),
        show_viewer=True,
    )

    if filename.endswith(".urdf"):
        morph_cls = gs.morphs.URDF
    elif filename.endswith(".xml"):
        morph_cls = gs.morphs.MJCF
    else:
        morph_cls = gs.morphs.Mesh
    entity = scene.add_entity(
        morph_cls(file=filename, collision=collision, scale=scale),
        surface=gs.surfaces.Default(
            vis_mode="visual" if not collision else "collision",
        ),
    )
    scene.build(compile_kernels=False)

    # Get motor info
    motors_dof_idx, motors_name = get_motors_info(entity)

    # Get motor position limits.
    # Makes sure that all joints are bounded, included revolute joints.
    if motors_dof_idx:
        motors_position_limit = torch.stack(entity.get_dofs_limit(motors_dof_idx), dim=1).numpy()
        motors_position_limit[motors_position_limit == -np.inf] = -np.pi
        motors_position_limit[motors_position_limit == +np.inf] = +np.pi

        # Start the GUI process
        manager = multiprocessing.Manager()
        motors_position = manager.list([0.0 for _ in motors_dof_idx])
        stop_event = multiprocessing.Event()
        gui_process = multiprocessing.Process(
            target=_start_gui, args=(motors_name, motors_position_limit, motors_position, stop_event), daemon=True
        )
        gui_process.start()
    else:
        stop_event = multiprocessing.Event()

    t = 0
    while scene.viewer.is_alive() and not stop_event.is_set():
        # Rotate entity if requested
        if rotate:
            t += 1 / FPS
            entity.set_quat(gs.utils.geom.xyz_to_quat(np.array([0, 0, t * 50]), rpy=True, degrees=True))

        if motors_dof_idx:
            entity.set_dofs_position(
                position=torch.tensor(motors_position),
                dofs_idx_local=motors_dof_idx,
                zero_velocity=True,
            )
        scene.visualizer.update(force=True)
    stop_event.set()
    if motors_dof_idx:
        gui_process.join()


def animate(filename_pattern, fps):
    import glob

    from PIL import Image

    gs.init()
    files = sorted(glob.glob(filename_pattern))
    imgs = []
    for file in files:
        print(f"Loading {file}")
        imgs.append(np.array(Image.open(file)))
    gs.tools.animate(imgs, "video.mp4", fps)


def main():
    parser = argparse.ArgumentParser(description="Genesis CLI")
    subparsers = parser.add_subparsers(dest="command")

    parser_clean = subparsers.add_parser("clean", help="Clean all the files cached by genesis and taichi")

    parser_view = subparsers.add_parser("view", help="Visualize a given asset (mesh/URDF/MJCF)")
    parser_view.add_argument("filename", type=str, help="File to visualize")
    parser_view.add_argument(
        "-c", "--collision", action="store_true", default=False, help="Whether to visualize collision geometry"
    )
    parser_view.add_argument("-r", "--rotate", action="store_true", default=False, help="Whether to rotate the entity")
    parser_view.add_argument("-s", "--scale", type=float, default=1.0, help="Scale of the entity")
    parser_view.add_argument("-l", "--link_frame", action="store_true", default=False, help="Show link frame")

    parser_animate = subparsers.add_parser("animate", help="Compile a list of image files into a video")
    parser_animate.add_argument("filename_pattern", type=str, help="Image files, via glob pattern")
    parser_animate.add_argument("--fps", type=int, default=30, help="FPS of the output video")

    args = parser.parse_args()

    if args.command == "clean":
        clean()
    elif args.command == "view":
        view(args.filename, args.collision, args.rotate, args.scale, args.link_frame)
    elif args.command == "animate":
        animate(args.filename_pattern, args.fps)
    elif args.command == None:
        parser.print_help()


if __name__ == "__main__":
    main()
