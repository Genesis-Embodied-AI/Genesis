import argparse
import multiprocessing
from functools import partial

import tkinter as tk
from tkinter import ttk

import numpy as np
import torch

import genesis as gs


FPS = 60


class JointControlGUI:
    def __init__(self, master, display_items, motors_position_limit, motors_position):
        self.master = master
        self.master.title("Joint Controller")  # Set the window title
        self.display_items = display_items
        self.motors_position_limit = motors_position_limit
        self.motors_position = motors_position
        n_dofs = len(motors_position_limit)
        self.motors_default_position = np.clip(
            np.zeros(n_dofs), motors_position_limit[:, 0], motors_position_limit[:, 1]
        )
        self.sliders = []
        self.values_label = []
        self.create_widgets()
        self.reset_motors_position()

    def create_widgets(self):
        container = tk.Frame(self.master)
        container.pack(fill=tk.BOTH, expand=True)

        canvas = tk.Canvas(container, borderwidth=0, highlightthickness=0)
        scrollbar = ttk.Scrollbar(container, orient=tk.VERTICAL)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        def on_yscrollcommand(*args):
            canvas.update_idletasks()
            top, bot = canvas.yview()
            if top < 0:
                canvas.yview_moveto(0)
            scrollbar.set(*canvas.yview())

        canvas.configure(yscrollcommand=on_yscrollcommand)
        scrollbar.configure(command=canvas.yview)

        scrollable_frame = tk.Frame(canvas)
        window_id = canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")

        def update_scroll_region_and_bar():
            bbox = canvas.bbox("all")
            if bbox:
                w = max(bbox[2] - bbox[0], 1)
                h = max(bbox[3] - bbox[1], 1)
                canvas.configure(scrollregion=(0, 0, w, h))
                content_h = h
                canvas.update_idletasks()
                canvas_h = canvas.winfo_height()
                if content_h > canvas_h:
                    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
                else:
                    scrollbar.pack_forget()
                    canvas.yview_moveto(0)

        def on_frame_configure(event):
            update_scroll_region_and_bar()

        def on_canvas_configure(event):
            canvas.itemconfig(window_id, width=event.width)
            update_scroll_region_and_bar()

        scrollable_frame.bind("<Configure>", on_frame_configure)
        canvas.bind("<Configure>", on_canvas_configure)

        def on_mousewheel(event):
            if not event.delta:
                return
            canvas.update_idletasks()
            if not scrollbar.winfo_ismapped():
                return
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
            if canvas.yview()[0] < 0:
                canvas.yview_moveto(0)

        def on_linux_scroll(event):
            canvas.update_idletasks()
            if not scrollbar.winfo_ismapped():
                return
            if event.num == 4:
                canvas.yview_scroll(-1, "units")
            elif event.num == 5:
                canvas.yview_scroll(1, "units")
            if canvas.yview()[0] < 0:
                canvas.yview_moveto(0)

        canvas.bind_all("<MouseWheel>", on_mousewheel)
        canvas.bind_all("<Button-4>", on_linux_scroll)
        canvas.bind_all("<Button-5>", on_linux_scroll)

        slider_idx = 0
        for label, is_delimiter in self.display_items:
            frame = tk.Frame(scrollable_frame)
            if is_delimiter:
                frame.pack(pady=(12, 4), padx=10, fill=tk.X)
                sep = ttk.Separator(scrollable_frame, orient=tk.HORIZONTAL)
                sep.pack(fill=tk.X, padx=10, pady=(0, 2))
                tk.Label(frame, text=label, font=("Arial", 12, "bold")).pack()
                continue
            frame.pack(pady=5, padx=10, fill=tk.X)
            self.update_joint_position(slider_idx, self.motors_default_position[slider_idx])
            min_limit, max_limit = map(float, self.motors_position_limit[slider_idx])
            tk.Label(frame, text=label, font=("Arial", 12), anchor=tk.W).pack(side=tk.LEFT)
            value_label = tk.Label(frame, text="0.00", font=("Arial", 12))
            value_label.pack(side=tk.RIGHT, padx=(5, 0))
            slider = ttk.Scale(
                frame,
                from_=min_limit,
                to=max_limit,
                orient=tk.HORIZONTAL,
                command=partial(self.update_joint_position, slider_idx),
            )
            slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
            slider.set(self.motors_default_position[slider_idx])
            value_label.config(text=f"{slider.get():.2f}")
            self.sliders.append(slider)
            self.values_label.append(value_label)

            # Update label dynamically
            def update_label(s=slider, l=value_label):
                def callback(event):
                    l.config(text=f"{s.get():.2f}")

                return callback

            slider.bind("<Motion>", update_label())
            slider_idx += 1

        tk.Button(scrollable_frame, text="Reset", font=("Arial", 12), command=self.reset_motors_position).pack(pady=20)

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
                dofs_name = [f"{joint.name}_{i_d}" for i_d in dofs_idx_local]
            motors_dof_idx += dofs_idx_local
            motors_dof_name += dofs_name
    return motors_dof_idx, motors_dof_name


def get_motors_info_for_view(entities):
    if not hasattr(entities, "__iter__") or hasattr(entities, "joints"):
        entities = [entities]
    entity_specs = []
    for entity in entities:
        motors_dof_idx, motors_dof_name = get_motors_info(entity)
        if motors_dof_idx:
            entity_specs.append((entity, motors_dof_idx, motors_dof_name))
    if not entity_specs:
        return [], [], np.zeros((0, 2), dtype=np.float64)

    display_items = []
    all_limits = []
    entity_dof_specs = []
    n_entities_in_scene = len(entities)
    for i, (entity, dofs_idx, names) in enumerate(entity_specs):
        if n_entities_in_scene > 1:
            display_items.append((f"——— {entity.name} ———", True))
        for name in names:
            display_items.append((name, False))
        entity_dof_specs.append((entity, dofs_idx))
        limits = torch.stack(entity.get_dofs_limit(dofs_idx), dim=1).numpy()
        limits[limits == -np.inf] = -np.pi
        limits[limits == np.inf] = np.pi
        all_limits.append(limits)
    motors_position_limit = np.vstack(all_limits)
    return display_items, entity_dof_specs, motors_position_limit


def _start_gui(display_items, motors_position_limit, motors_position, stop_event):
    def on_close():
        nonlocal after_id
        if after_id is not None:
            root.after_cancel(after_id)
            after_id = None
        stop_event.set()
        root.destroy()
        root.quit()

    root = tk.Tk()
    root.minsize(520, 400)

    # Size window so content fits without vertical scroll when possible
    row_heights = [50 if is_delimiter else 36 for _, is_delimiter in display_items]
    content_h = sum(row_heights) + 100  # + reset button and padding
    screen_h = root.winfo_screenheight()
    height = min(content_h, max(400, screen_h - 120))
    root.geometry(f"560x{height}")

    # Store joint control gui to make sure it does not get garbage collected, just in case, because it may break tkinter
    _app = JointControlGUI(root, display_items, motors_position_limit, motors_position)

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
            show_world_frame=True,
        ),
        show_viewer=True,
    )

    filename_lower = filename.lower()
    morphs = gs.options.morphs
    surface = gs.surfaces.Default(vis_mode="visual" if not collision else "collision")

    if filename_lower.endswith(morphs.USD_FORMATS):
        morph = gs.morphs.USD(file=filename, collision=collision, scale=scale)
        entities = scene.add_stage(morph=morph, vis_mode=surface.vis_mode)
    elif filename_lower.endswith(morphs.URDF_FORMAT):
        morph_cls = gs.morphs.URDF
        entities = [
            scene.add_entity(
                morph_cls(file=filename, collision=collision, scale=scale),
                surface=surface,
            )
        ]
    elif filename_lower.endswith(morphs.MJCF_FORMAT):
        morph_cls = gs.morphs.MJCF
        entities = [
            scene.add_entity(
                morph_cls(file=filename, collision=collision, scale=scale),
                surface=surface,
            )
        ]
    elif filename_lower.endswith(morphs.MESH_FORMATS):
        morph_cls = gs.morphs.Mesh
        entities = [
            scene.add_entity(
                morph_cls(file=filename, collision=collision, scale=scale),
                surface=surface,
            )
        ]
    else:
        gs.raise_exception(
            f"Unsupported file format for 'gs view'. Expected {morphs.URDF_FORMAT}, "
            f"{morphs.MJCF_FORMAT}, {morphs.MESH_FORMATS}, or {morphs.USD_FORMATS}."
        )

    scene.build(compile_kernels=False)

    display_items, entity_dof_specs, motors_position_limit = get_motors_info_for_view(entities)
    total_dofs = len(motors_position_limit)

    # Start the GUI process
    if total_dofs > 0:
        manager = multiprocessing.Manager()
        motors_position = manager.list([0.0] * total_dofs)
        stop_event = multiprocessing.Event()
        gui_process = multiprocessing.Process(
            target=_start_gui,
            args=(display_items, motors_position_limit, motors_position, stop_event),
            daemon=True,
        )
        gui_process.start()
    else:
        stop_event = multiprocessing.Event()

    t = 0
    while scene.viewer.is_alive() and not stop_event.is_set():
        # Rotate entity if requested
        if rotate:
            t += 1 / FPS
            quat = gs.utils.geom.xyz_to_quat(np.array([0, 0, t * 50]), rpy=True, degrees=True)
            for entity in entities:
                entity.set_quat(quat)

        if total_dofs > 0:
            offset = 0
            for entity, dofs_idx in entity_dof_specs:
                n = len(dofs_idx)
                entity.set_dofs_position(
                    position=torch.tensor(motors_position[offset : offset + n]),
                    dofs_idx_local=dofs_idx,
                    zero_velocity=True,
                )
                offset += n
        scene.visualizer.update(force=True)
    stop_event.set()
    if total_dofs > 0:
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

    parser_view = subparsers.add_parser("view", help="Visualize a given asset (Mesh/URDF/MJCF/USD)")
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

    if args.command == "view":
        view(args.filename, args.collision, args.rotate, args.scale, args.link_frame)
    elif args.command == "animate":
        animate(args.filename_pattern, args.fps)
    elif args.command is None:
        parser.print_help()


if __name__ == "__main__":
    main()
