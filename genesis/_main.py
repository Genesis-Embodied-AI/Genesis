import argparse
import multiprocessing
import os
import threading
import tkinter as tk
from tkinter import ttk

import numpy as np
import torch
from taichi._lib import core as _ti_core
from taichi.lang import impl

import genesis as gs


class JointControlGUI:
    def __init__(self, master, motor_names, dof_pos_limits, gui_joint_positions):
        self.master = master
        self.master.title("Joint Controller")  # Set the window title
        self.motor_names = motor_names
        self.dof_pos_limits = dof_pos_limits
        self.gui_joint_positions = gui_joint_positions
        self.default_joint_positions = np.clip(np.zeros(len(motor_names)), dof_pos_limits[:, 0], dof_pos_limits[:, 1])
        self.sliders = []
        self.value_labels = []
        self.create_widgets()
        self.reset_joint_position()

    def create_widgets(self):
        for i, name in enumerate(self.motor_names):
            self.update_joint_position(i, self.default_joint_positions[i])
            min_limit, max_limit = self.dof_pos_limits[i]
            if min_limit == torch.tensor(-np.inf):
                min_limit = -np.pi
            if max_limit == torch.tensor(np.inf):
                max_limit = np.pi
            frame = tk.Frame(self.master)
            frame.pack(pady=5, padx=10, fill=tk.X)

            tk.Label(frame, text=f"{name}", font=("Arial", 12), width=20).pack(side=tk.LEFT)

            slider = ttk.Scale(
                frame,
                from_=float(min_limit),
                to=float(max_limit),
                orient=tk.HORIZONTAL,
                length=300,
                command=lambda val, idx=i: self.update_joint_position(idx, val),
            )
            slider.pack(side=tk.LEFT, padx=5)
            self.sliders.append(slider)

            value_label = tk.Label(frame, text=f"{slider.get():.2f}", font=("Arial", 12))
            value_label.pack(side=tk.LEFT, padx=5)
            self.value_labels.append(value_label)

            # Update label dynamically
            def update_label(s=slider, l=value_label):
                def callback(event):
                    l.config(text=f"{s.get():.2f}")

                return callback

            slider.bind("<Motion>", update_label())

        tk.Button(self.master, text="Reset", font=("Arial", 12), command=self.reset_joint_position).pack(pady=20)

    def update_joint_position(self, idx, val):
        self.gui_joint_positions[idx] = float(val)

    def reset_joint_position(self):
        for i, slider in enumerate(self.sliders):
            slider.set(self.default_joint_positions[i])
            self.value_labels[i].config(text=f"{self.default_joint_positions[i]:.2f}")
            self.gui_joint_positions[i] = self.default_joint_positions[i]


def get_movable_dofs(robot):
    motor_dofs = []
    motor_dof_names = []
    for joint in robot.joints:
        if joint.type == gs.JOINT_TYPE.FREE:
            continue
        elif joint.type == gs.JOINT_TYPE.FIXED:
            continue
        motor_dofs.append(robot.get_joint(joint.name).dof_idx_local)
        motor_dof_names.append(joint.name)
    return motor_dofs, motor_dof_names


def clean():
    gs.utils.misc.clean_cache_files()
    _ti_core.clean_offline_cache_files(os.path.abspath(impl.default_cfg().offline_cache_file_path))
    print("Cleaned up all genesis and taichi cache files.")


def _start_gui_mac(motor_names, dof_pos_limits, gui_joint_positions, stop_event):
    def on_close():
        stop_event.set()
        root.destroy()

    root = tk.Tk()
    app = JointControlGUI(root, motor_names, dof_pos_limits, gui_joint_positions)
    root.protocol("WM_DELETE_WINDOW", on_close)

    root.mainloop()


def view(filename, collision, rotate, scale=1.0):
    gs.init(backend=gs.cpu)
    FPS = 60
    dt = 1 / FPS
    scene = gs.Scene(
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(3.5, 0.0, 2.5),
            camera_lookat=(0.0, 0.0, 0.5),
            camera_fov=40,
            max_FPS=FPS,
        ),
        sim_options=gs.options.SimOptions(
            gravity=(0, 0, 0),
        ),
    )

    if filename.endswith(".urdf"):
        morph = gs.morphs.URDF(file=filename, collision=collision, scale=scale)
    elif filename.endswith(".xml"):
        morph = gs.morphs.MJCF(file=filename, collision=collision, scale=scale)
    else:
        morph = gs.morphs.Mesh(file=filename, collision=collision, scale=scale)

    entity = scene.add_entity(
        morph,
        surface=gs.surfaces.Default(
            vis_mode="visual" if not collision else "collision",
        ),
    )
    scene.build(compile_kernels=False)

    motor_dofs, motor_names = get_movable_dofs(entity)
    dof_pos_limits = torch.stack(entity.get_dofs_limit(motor_dofs), dim=1).numpy()

    if gs.platform == "Linux":
        # Shared positions list between GUI and simulation
        gui_joint_positions = np.zeros(len(motor_dofs))

        # Start GUI in a separate thread
        stop_event = threading.Event()
        is_gui_closed = [False]

        def on_close():
            is_gui_closed[0] = True
            stop_event.set()

        def start_gui():
            root = tk.Tk()
            app = JointControlGUI(root, motor_names, dof_pos_limits, gui_joint_positions)
            root.protocol("WM_DELETE_WINDOW", on_close)
            root.mainloop()

        if len(motor_names) > 0:
            gui_thread = threading.Thread(target=start_gui, daemon=True)
            gui_thread.start()

        t = 0
        while scene.viewer.is_alive() and not is_gui_closed[0]:
            # rotate entity
            t += dt
            if rotate:
                entity.set_quat(gs.utils.geom.xyz_to_quat(np.array([0, 0, t * 50])))

            entity.set_dofs_position(
                position=torch.tensor(gui_joint_positions),
                dofs_idx_local=motor_dofs,
                zero_velocity=True,
            )
            scene.visualizer.update(force=True)

    elif gs.platform == "macOS":
        manager = multiprocessing.Manager()
        gui_joint_positions = manager.list([0.0] * len(motor_dofs))
        stop_event = multiprocessing.Event()
        # Start the GUI process
        gui_process = multiprocessing.Process(
            target=_start_gui_mac, args=(motor_names, dof_pos_limits, gui_joint_positions, stop_event), daemon=True
        )
        gui_process.start()

        def update_scene(scene, stop_event, gui_joint_positions, motor_dofs, rotate, entity):
            t = 0
            while scene.viewer.is_alive() and not stop_event.is_set():
                # rotate entity
                t += dt
                if rotate:
                    entity.set_quat(gs.utils.geom.xyz_to_quat(np.array([0, 0, t * 50])))

                entity.set_dofs_position(
                    # position=torch.tensor(gui_positions),
                    position=torch.tensor(gui_joint_positions),
                    dofs_idx_local=motor_dofs,
                    zero_velocity=True,
                )
                scene.visualizer.update(force=True)
            scene.viewer.stop()

        gs.tools.run_in_another_thread(
            fn=update_scene, args=(scene, stop_event, gui_joint_positions, motor_dofs, rotate, entity)
        )
        scene.viewer.start()

    else:
        raise NotImplementedError(f"Platform {gs.platform} is not supported.")


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

    parser_animate = subparsers.add_parser("animate", help="Compile a list of image files into a video")
    parser_animate.add_argument("filename_pattern", type=str, help="Image files, via glob pattern")
    parser_animate.add_argument("--fps", type=int, default=30, help="FPS of the output video")

    args = parser.parse_args()

    if args.command == "clean":
        clean()
    elif args.command == "view":
        view(args.filename, args.collision, args.rotate, args.scale)
    elif args.command == "animate":
        animate(args.filename_pattern, args.fps)
    elif args.command == None:
        parser.print_help()


if __name__ == "__main__":
    main()
