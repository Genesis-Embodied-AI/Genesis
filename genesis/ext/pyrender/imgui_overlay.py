"""
ImGui overlay plugin for joint control and simulation controls.

Requires: pip install imgui-bundle
"""

import os
import time
from typing import TYPE_CHECKING

import numpy as np

import genesis as gs
from genesis.vis.scene_ops import (
    FREE_JOINT_POS_LIMIT,
    QUATERNION_COMPONENT_LIMIT,
    build_entity_joint_data,
    refresh_visual_transforms,
    set_entity_wireframe as _shared_set_entity_wireframe,
    switch_entity_vis_mode as _shared_switch_entity_vis_mode,
)
from genesis.vis.viewer_plugins import ViewerPlugin, EVENT_HANDLED, EVENT_HANDLE_STATE

if TYPE_CHECKING:
    from genesis.engine.scene import Scene
    from genesis.ext.pyrender.viewer import Viewer

_FPS_HISTORY_SIZE = 30
_MORPH_TYPES = ["URDF", "MJCF", "Mesh", "Box", "Sphere", "Cylinder", "Plane"]


class ImGuiOverlayPlugin(ViewerPlugin):
    """
    ViewerPlugin that adds an ImGui control panel for simulation and joint control.

    Features:
    - Simulation controls: play/pause, step, reset
    - Joint sliders for each entity (editable only when paused)
    - FPS display with rolling average
    - Multi-step support
    - Custom panel registration API

    Limitations:
    - Only controls environment 0 in batched simulations

    Usage:
        scene.build()
        plugin = ImGuiOverlayPlugin()
        scene.viewer._pyrender_viewer.register_plugin(plugin)

        while scene.viewer.is_alive():
            if plugin.should_step():
                scene.step()
    """

    def __init__(
        self,
        show_sim_controls=True,
        show_entity_browser=True,
        show_visualization=True,
        show_camera_controls=True,
        rebuild_fn=None,
    ):
        super().__init__()
        self._imgui = None
        self._impl = None
        self._io = None
        self._available = False
        self._init_attempted = False
        self._last_time = None
        self.paused = False
        self._step_requested = False
        self._steps_remaining = 0
        self._step_count = 1
        self._entity_cache = {}
        self._user_panels = []
        self._fps_history = []

        # Section visibility flags
        self.show_sim_controls = show_sim_controls
        self.show_entity_browser = show_entity_browser
        self.show_visualization = show_visualization
        self.show_camera_controls = show_camera_controls

        # Scene rebuild support
        self._rebuild_fn = rebuild_fn
        self._rebuild_requested = False
        self._specs_dirty = False
        self._entity_specs = []  # populated at build time
        self._add_entity_file = ""
        self._add_entity_morph_type = 0  # index into _MORPH_TYPES
        self._add_entity_pos = [0.0, 0.0, 0.0]
        self._add_entity_scale = 1.0
        # Type-specific geometry params
        self._add_box_size = [0.2, 0.2, 0.2]
        self._add_sphere_radius = 0.1
        self._add_cylinder_radius = 0.05
        self._add_cylinder_height = 0.2
        self._add_entity_fixed = True
        # File browser state
        self._file_browser_open = False
        self._file_browser_dir = os.getcwd()
        self._file_browser_selected = -1
        # Gizmo state
        self._gizmo = None  # imguizmo.im_guizmo module (lazy loaded)
        self._gizmo_operation = None  # gizmo.OPERATION.translate
        self._gizmo_mode = None  # gizmo.MODE.world
        self._gizmo_entity_idx = -1  # which free-joint entity is selected for gizmo
        self._gizmo_cached_matrix = None  # cached 4x4 object matrix while dragging (avoids qpos round-trip jitter)
        # Per-entity euler/quat mode: entity_idx -> "euler" or "quat"
        self._rotation_mode = {}
        # Per-entity wireframe state: entity_idx -> bool
        self._wireframe_state = {}

    def register_panel(self, callback, section="side"):
        """Register custom UI panel. callback(imgui) called each frame.

        Thread-safe: uses copy-on-write list.

        Args:
            callback: Function taking imgui module as argument, called each frame.
            section: "side" adds to main panel, "overlay" creates floating window.
        """
        new_list = list(self._user_panels) + [(callback, section)]
        self._user_panels = new_list  # Atomic reference swap

    def build(self, viewer: "Viewer", camera, scene: "Scene"):
        """Store references; ImGui initialization is deferred to on_draw (viewer thread)."""
        super().build(viewer, camera, scene)
        # Reset ImGui state so it re-initializes in the new viewer thread
        # (needed after scene rebuild creates a new viewer/OpenGL context)
        # Don't destroy the old context here — it belonged to the old viewer
        # thread and is already invalid after scene.destroy().
        if self._init_attempted:
            self._impl = None
            self._io = None
            self._available = False
            self._init_attempted = False
            self._last_time = None
        # Cache entity data now (doesn't require OpenGL)
        self._cache_entity_data()
        self._capture_entity_specs()

    def _init_imgui(self):
        """Initialize ImGui. Must be called from the viewer thread (e.g., in on_draw)."""
        if self._init_attempted:
            return
        self._init_attempted = True

        try:
            from imgui_bundle import imgui
            from imgui_bundle.python_backends import pyglet_backend

            self._imgui = imgui
            imgui.create_context()
            # Load default font at larger size before renderer builds the atlas
            io = imgui.get_io()
            io.fonts.clear()
            font_cfg = imgui.ImFontConfig()
            font_cfg.size_pixels = 18.0
            io.fonts.add_font_default(font_cfg)
            self._impl = pyglet_backend.create_renderer(self.viewer, attach_callbacks=False)
            # Fix: Set window reference for cursor handling (not set when attach_callbacks=False)
            self._impl._window = self.viewer
            self._io = imgui.get_io()
            self._io.set_ini_filename("")  # Don't persist window positions
            # Set up clipboard (pyglet backend doesn't do this by default)
            # Pyglet caches _clipboard_str and only clears it on SelectionClear
            # events, which may not be dispatched in time. Invalidate the cache
            # before each read so we always get fresh system clipboard content.
            window_ref = self.viewer

            def _get_clipboard(_ctx):
                try:
                    window_ref._clipboard_str = None
                    text = window_ref.get_clipboard_text()
                    return text if text else ""
                except Exception:
                    return ""

            def _set_clipboard(_ctx, text):
                try:
                    window_ref.set_clipboard_text(text)
                except Exception:
                    pass

            platform_io = imgui.get_platform_io()
            platform_io.platform_get_clipboard_text_fn = _get_clipboard
            platform_io.platform_set_clipboard_text_fn = _set_clipboard
            self._setup_style()
            self._available = True
            # Try to load ImGuizmo for 3D gizmos
            try:
                from imgui_bundle import imguizmo

                self._gizmo = imguizmo.im_guizmo
                self._gizmo_operation = self._gizmo.OPERATION.translate
                self._gizmo_mode = self._gizmo.MODE.world
                self._gizmo.set_gizmo_size_clip_space(0.15)
                self._gizmo.allow_axis_flip(False)
            except ImportError:
                pass
        except ImportError:
            print("ImGuiOverlayPlugin: imgui-bundle not found. Install with: pip install imgui-bundle")
        except Exception as e:
            print(f"ImGuiOverlayPlugin: Failed to initialize ImGui: {e}")

    def _setup_style(self):
        """Apply modern rounded dark theme."""
        imgui = self._imgui
        imgui.style_colors_dark()
        style = imgui.get_style()
        Col_ = imgui.Col_
        sc = style.set_color_

        # Geometry - modern rounded, borderless
        style.window_rounding = 12.0
        style.frame_rounding = 8.0
        style.child_rounding = 10.0
        style.popup_rounding = 10.0
        style.scrollbar_rounding = 8.0
        style.grab_rounding = 6.0
        style.tab_rounding = 8.0
        style.window_border_size = 0.0
        style.frame_border_size = 0.0

        # Spacing
        style.window_padding = (12.0, 10.0)
        style.frame_padding = (8.0, 5.0)
        style.item_spacing = (8.0, 6.0)
        style.item_inner_spacing = (6.0, 4.0)
        style.scrollbar_size = 10.0
        style.grab_min_size = 10.0

        # Semi-transparent backgrounds
        sc(Col_.window_bg, (0.11, 0.11, 0.14, 0.92))
        sc(Col_.child_bg, (0.13, 0.13, 0.16, 0.60))
        sc(Col_.popup_bg, (0.11, 0.11, 0.14, 0.96))

        # Text
        sc(Col_.text, (0.93, 0.94, 0.96, 1.0))
        sc(Col_.text_disabled, (0.45, 0.47, 0.52, 1.0))

        # Borders - subtle
        sc(Col_.border, (0.25, 0.26, 0.30, 0.35))

        # Frames (sliders, input fields) - frosted
        sc(Col_.frame_bg, (0.18, 0.18, 0.22, 0.75))
        sc(Col_.frame_bg_hovered, (0.24, 0.24, 0.30, 0.85))
        sc(Col_.frame_bg_active, (0.28, 0.28, 0.36, 0.95))

        # Title bar
        sc(Col_.title_bg, (0.09, 0.09, 0.12, 0.95))
        sc(Col_.title_bg_active, (0.12, 0.12, 0.16, 1.0))
        sc(Col_.title_bg_collapsed, (0.09, 0.09, 0.12, 0.70))

        # Buttons - accent blue with soft edges
        sc(Col_.button, (0.22, 0.38, 0.58, 0.80))
        sc(Col_.button_hovered, (0.28, 0.48, 0.70, 0.90))
        sc(Col_.button_active, (0.20, 0.34, 0.52, 1.0))

        # Headers (collapsing headers) - subtle highlight
        sc(Col_.header, (0.18, 0.18, 0.24, 0.65))
        sc(Col_.header_hovered, (0.26, 0.40, 0.58, 0.75))
        sc(Col_.header_active, (0.24, 0.38, 0.56, 0.90))

        # Interactive accents - bright blue
        sc(Col_.check_mark, (0.45, 0.72, 0.95, 1.0))
        sc(Col_.slider_grab, (0.38, 0.62, 0.88, 0.90))
        sc(Col_.slider_grab_active, (0.45, 0.72, 0.95, 1.0))

        # Scrollbar - minimal
        sc(Col_.scrollbar_bg, (0.08, 0.08, 0.10, 0.30))
        sc(Col_.scrollbar_grab, (0.30, 0.32, 0.38, 0.50))
        sc(Col_.scrollbar_grab_hovered, (0.40, 0.42, 0.50, 0.70))
        sc(Col_.scrollbar_grab_active, (0.48, 0.50, 0.58, 0.90))

        # Tabs
        sc(Col_.tab, (0.14, 0.14, 0.18, 0.70))
        sc(Col_.tab_hovered, (0.28, 0.46, 0.66, 0.85))
        sc(Col_.tab_selected, (0.22, 0.38, 0.58, 0.90))

        # Separators - very subtle
        sc(Col_.separator, (0.28, 0.30, 0.36, 0.30))
        sc(Col_.separator_hovered, (0.38, 0.56, 0.78, 0.60))
        sc(Col_.separator_active, (0.42, 0.64, 0.88, 0.85))

        # Resize grip
        sc(Col_.resize_grip, (0.28, 0.40, 0.58, 0.20))
        sc(Col_.resize_grip_hovered, (0.35, 0.55, 0.78, 0.50))
        sc(Col_.resize_grip_active, (0.40, 0.65, 0.90, 0.75))

    def _get_entity_name(self, entity, idx: int) -> str:
        """Extract a human-readable name for an entity, with index for disambiguation."""
        return getattr(entity, "name", None) or f"Entity_{idx}"

    def _cache_entity_data(self):
        """Cache static joint metadata from all rigid entities."""
        self._entity_cache.clear()

        if not hasattr(self.scene, "rigid_solver") or self.scene.rigid_solver is None:
            return

        for entity in self.scene.rigid_solver.entities:
            if entity.n_dofs == 0:
                # Still include for vis_mode toggle, but no joint data
                self._entity_cache[entity.idx] = {
                    "entity": entity,
                    "name": self._get_entity_name(entity, entity.idx),
                    "q_names": [],
                    "q_limits": ([], []),
                    "q_is_quaternion": [],
                    "quat_groups": [],
                    "has_free_joint": False,
                    "free_joint_q_start": -1,
                    "n_qs": 0,
                    "n_dofs": 0,
                }
                continue

            jdata = build_entity_joint_data(entity)
            if jdata["q_names"]:
                self._entity_cache[entity.idx] = {
                    "entity": entity,
                    "name": self._get_entity_name(entity, entity.idx),
                    "q_names": jdata["q_names"],
                    "q_limits": (jdata["q_limits_lower"], jdata["q_limits_upper"]),
                    "q_is_quaternion": jdata["q_is_quaternion"],
                    "quat_groups": jdata["quat_groups"],
                    "has_free_joint": jdata["has_free_joint"],
                    "free_joint_q_start": jdata["free_joint_q_start"],
                    "n_qs": len(jdata["q_names"]),
                    "n_dofs": entity.n_dofs,
                }

    def _capture_entity_specs(self):
        """Capture current entity specs for rebuild support."""
        self._entity_specs = []
        if not hasattr(self.scene, "sim"):
            return
        for entity in self.scene.sim.entities:
            morph = entity.morph
            spec = {
                "morph": morph,
                "material": entity.material,
                "surface": entity.surface,
                "visualize_contact": getattr(entity, "visualize_contact", False),
                "scale": getattr(morph, "scale", 1.0),
            }
            self._entity_specs.append(spec)

    @property
    def entity_specs(self):
        """Current entity specs list (read by rebuild_fn)."""
        return self._entity_specs

    @property
    def rebuild_requested(self):
        """True if the user clicked Rebuild. Check this in your main loop."""
        if self._rebuild_requested:
            self._rebuild_requested = False
            return True
        return False

    def _apply_qpos_update(self, entity, new_qpos, is_multi_env: bool) -> None:
        """Apply qpos update to entity, handling single-env vs multi-env correctly.

        Args:
            entity: The RigidEntity to update.
            new_qpos: Array-like of new joint positions.
            is_multi_env: If True, pass envs_idx=0 to set_qpos. If False, omit envs_idx.
        """
        qpos_array = np.asarray(new_qpos)
        if is_multi_env:
            entity.set_qpos(qpos_array, envs_idx=0)
        else:
            entity.set_qpos(qpos_array)

        refresh_visual_transforms(self.scene, self.viewer.gs_context)

    def _switch_entity_vis_mode(self, entity, new_mode):
        """Switch entity visualization between 'visual' and 'collision' at runtime."""
        _shared_switch_entity_vis_mode(self.scene, self.viewer.gs_context, entity, new_mode)

    def _set_entity_wireframe(self, entity, wireframe):
        """Toggle wireframe rendering for all geom nodes of an entity."""
        _shared_set_entity_wireframe(self.viewer.gs_context, entity, wireframe)

    def _is_capturing(self) -> bool:
        """Check if ImGui or gizmo wants mouse/keyboard input."""
        if not self._available:
            return False
        return self._io.want_capture_mouse or self._io.want_capture_keyboard or self._is_gizmo_active()

    # Event handlers - forward input to ImGui and block when capturing
    def on_mouse_press(self, x, y, button, modifiers) -> EVENT_HANDLE_STATE:
        if self._available:
            self._impl.on_mouse_press(x, y, button, modifiers)
        return EVENT_HANDLED if self._is_capturing() else None

    def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers) -> EVENT_HANDLE_STATE:
        if self._available:
            self._impl.on_mouse_drag(x, y, dx, dy, buttons, modifiers)
        return EVENT_HANDLED if self._is_capturing() else None

    def on_mouse_release(self, x, y, button, modifiers) -> EVENT_HANDLE_STATE:
        if self._available:
            self._impl.on_mouse_release(x, y, button, modifiers)
        return EVENT_HANDLED if self._is_capturing() else None

    def on_mouse_scroll(self, x, y, dx, dy) -> EVENT_HANDLE_STATE:
        if self._available:
            # imgui backend expects: on_mouse_scroll(x, y, mods, scroll)
            self._impl.on_mouse_scroll(x, y, 0, dy)
        return EVENT_HANDLED if self._is_capturing() else None

    def on_mouse_motion(self, x, y, dx, dy) -> EVENT_HANDLE_STATE:
        if self._available:
            self._impl.on_mouse_motion(x, y, dx, dy)
        return EVENT_HANDLED if self._is_capturing() else None

    def on_key_press(self, symbol, modifiers) -> EVENT_HANDLE_STATE:
        if self._available:
            self._impl.on_key_press(symbol, modifiers)
        return EVENT_HANDLED if self._is_capturing() else None

    def on_key_release(self, symbol, modifiers) -> EVENT_HANDLE_STATE:
        if self._available:
            self._impl.on_key_release(symbol, modifiers)
        return EVENT_HANDLED if self._is_capturing() else None

    def on_text(self, text) -> EVENT_HANDLE_STATE:
        if self._available:
            self._impl.on_text(text)
        return EVENT_HANDLED if self._is_capturing() else None

    def on_resize(self, width, height) -> EVENT_HANDLE_STATE:
        if self._available:
            self._impl.on_resize(width, height)
        return None

    def on_draw(self) -> None:
        """Render ImGui overlay after scene is drawn."""
        # Lazy initialization: must happen in viewer thread (which owns OpenGL context)
        if not self._init_attempted:
            self._init_imgui()

        if not self._available:
            return

        # Update delta time manually (avoid calling pyglet.clock.tick() which conflicts with viewer loop)
        current_time = time.perf_counter()
        if self._last_time is not None:
            self._io.delta_time = current_time - self._last_time
        else:
            self._io.delta_time = 1.0 / 60.0
        if self._io.delta_time <= 0.0:
            self._io.delta_time = 1.0 / 1000.0
        self._last_time = current_time

        # Track FPS history
        if self._io.delta_time > 0:
            self._fps_history.append(1.0 / self._io.delta_time)
            if len(self._fps_history) > _FPS_HISTORY_SIZE:
                self._fps_history = self._fps_history[-_FPS_HISTORY_SIZE:]

        self._imgui.new_frame()

        # Initialize ImGuizmo for this frame
        if self._gizmo is not None:
            self._gizmo.begin_frame()
            io = self._io
            self._gizmo.set_rect(0, 0, io.display_size.x, io.display_size.y)
            self._gizmo.set_orthographic(not self.viewer.viewer_flags["use_perspective_cam"])

        self._render_control_panel()

        # Render 3D gizmos for selected free-joint entity
        if self._gizmo is not None and self._gizmo_entity_idx >= 0:
            self._render_gizmo()

        self._imgui.render()
        self._impl.render(self._imgui.get_draw_data())

    def _render_control_panel(self):
        """Render unified control panel with all sections."""
        imgui = self._imgui
        imgui.begin("Genesis Control Panel", flags=imgui.WindowFlags_.always_auto_resize)

        if self.show_sim_controls:
            self._render_sim_controls()

        if imgui.begin_tab_bar("##main_tabs"):
            if self.show_entity_browser:
                if imgui.begin_tab_item("Entities")[0]:
                    self._render_entity_browser()
                    imgui.end_tab_item()

            if self.show_visualization:
                if imgui.begin_tab_item("Visualization")[0]:
                    self._render_visualization()
                    imgui.end_tab_item()

            if self.show_camera_controls:
                if imgui.begin_tab_item("Camera")[0]:
                    self._render_camera_controls()
                    imgui.end_tab_item()

            if self._rebuild_fn is not None:
                if imgui.begin_tab_item("Scene")[0]:
                    self._render_scene_editor()
                    imgui.end_tab_item()

            imgui.end_tab_bar()

        # Render user callback panels (side panels)
        for callback, section in self._user_panels:
            if section == "side":
                callback(imgui)

        imgui.end()

        # Render overlay panels as separate windows
        for callback, section in self._user_panels:
            if section == "overlay":
                callback(imgui)

    def _render_sim_controls(self):
        """Render simulation control buttons, time display, and FPS."""
        imgui = self._imgui

        # State label
        if self.paused:
            imgui.text_colored((1.0, 0.7, 0.0, 1.0), "Paused")
        else:
            imgui.text_colored((0.4, 0.9, 0.4, 1.0), "Running")

        # Play/Pause and Reset (always visible), Step (only when paused)
        if imgui.button("Pause" if not self.paused else "Play", size=(60, 0)):
            self.paused = not self.paused
        if self.paused:
            imgui.same_line()
            if imgui.button("Step", size=(50, 0)):
                self._steps_remaining = self._step_count
        imgui.same_line()
        if imgui.button("Reset", size=(50, 0)):
            with self.viewer.render_lock:
                self.scene.reset()
                # Clear contact arrows from previous timesteps
                self.viewer.gs_context.clear_dynamic_nodes(only_outdated=False)

        # Time display (frame count * dt = simulation time)
        if hasattr(self.scene, "t"):
            sim_time = self.scene.t * self.scene.sim.dt
            imgui.text(f"Time: {sim_time:.3f}s  Step: {self.scene.t}")

        # FPS display
        if self._fps_history:
            avg_fps = sum(self._fps_history) / len(self._fps_history)
            imgui.same_line()
            imgui.text(f"  FPS: {avg_fps:.0f}")

        if hasattr(self.scene, "n_envs") and self.scene.n_envs > 1:
            imgui.text_colored((1.0, 0.7, 0.0, 1.0), f"Note: Controlling env 0 of {self.scene.n_envs}")

        imgui.separator()

    def _render_visualization(self):
        """Render visualization toggle controls."""
        imgui = self._imgui
        render_flags = self.viewer.render_flags
        gs_context = self.viewer.gs_context

        # Shadows
        changed, new_val = imgui.checkbox("Shadows", render_flags["shadows"])
        if changed:
            render_flags["shadows"] = new_val

        # World Frame
        changed, new_val = imgui.checkbox("World Frame", gs_context.world_frame_shown)
        if changed:
            if new_val:
                gs_context.on_world_frame()
            else:
                gs_context.off_world_frame()

        # Link Frame
        changed, new_val = imgui.checkbox("Link Frame", gs_context.link_frame_shown)
        if changed:
            if new_val:
                gs_context.on_link_frame()
            else:
                gs_context.off_link_frame()

        # Link Frame Size slider
        link_size = gs_context.link_frame_size
        changed_size, new_size = imgui.slider_float("Frame Size##link_frame_size", link_size, 0.02, 0.5, "%.2f")
        if changed_size and gs_context.link_frame_size > 0:
            scale = new_size / gs_context.link_frame_size
            gs_context.link_frame_mesh.vertices *= scale
            gs_context.link_frame_size = new_size
            if gs_context.link_frame_shown:
                gs_context.off_link_frame()
                gs_context.on_link_frame()

        # Camera Frustum
        changed, new_val = imgui.checkbox("Camera Frustum", gs_context.camera_frustum_shown)
        if changed:
            if new_val:
                gs_context.on_camera_frustum()
            else:
                gs_context.off_camera_frustum()

        # Face Normals
        changed, new_val = imgui.checkbox("Face Normals", render_flags["face_normals"])
        if changed:
            render_flags["face_normals"] = new_val

        # Vertex Normals
        changed, new_val = imgui.checkbox("Vertex Normals", render_flags["vertex_normals"])
        if changed:
            render_flags["vertex_normals"] = new_val

        imgui.separator()

        # Orthographic Camera
        is_ortho = not self.viewer.viewer_flags["use_perspective_cam"]
        changed, new_ortho = imgui.checkbox("Orthographic Camera", is_ortho)
        if changed:
            self.viewer.viewer_flags["use_perspective_cam"] = not new_ortho
            if new_ortho:
                self.viewer._camera_node.camera = self.viewer._default_orth_cam
            else:
                self.viewer._camera_node.camera = self.viewer._default_persp_cam

    def _render_gizmo(self):
        """Render 3D manipulation gizmo for the selected free-joint entity."""
        gizmo = self._gizmo
        Matrix16 = gizmo.Matrix16

        data = self._entity_cache.get(self._gizmo_entity_idx)
        if data is None or not data.get("has_free_joint"):
            return

        entity = data["entity"]
        qs = data["free_joint_q_start"]

        from scipy.spatial.transform import Rotation as R

        # While actively dragging, use the cached matrix to avoid qpos round-trip jitter.
        # Only read from qpos when not dragging (to pick up external changes).
        if gizmo.is_using() and self._gizmo_cached_matrix is not None:
            obj_mat = self._gizmo_cached_matrix
        else:
            # Get current qpos
            qpos_raw = entity.get_qpos()
            qpos_np = qpos_raw.cpu().numpy() if hasattr(qpos_raw, "cpu") else np.asarray(qpos_raw)
            is_multi_env = qpos_np.ndim == 2
            qpos = qpos_np[0] if is_multi_env else qpos_np.flatten()

            # Extract position and quaternion from qpos
            pos = qpos[qs : qs + 3]
            quat_wxyz = qpos[qs + 3 : qs + 7]  # w, x, y, z

            rot = R.from_quat([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])  # scipy uses x,y,z,w

            obj_mat = np.eye(4)
            obj_mat[:3, :3] = rot.as_matrix()
            obj_mat[:3, 3] = pos

        # ImGuizmo expects column-major (transpose for row-major numpy)
        object_matrix = Matrix16(obj_mat.T.flatten().tolist())

        # Get view matrix (inverse of camera pose)
        cam_pose = self.viewer._trackball._n_pose.copy()
        view_mat = np.linalg.inv(cam_pose)
        camera_view = Matrix16(view_mat.T.flatten().tolist())

        # Get projection matrix
        w, h = int(self._io.display_size.x), int(self._io.display_size.y)
        if w > 0 and h > 0:
            proj = self.camera.camera.get_projection_matrix(width=w, height=h)
            camera_proj = Matrix16(proj.T.flatten().tolist())
        else:
            return

        # Draw gizmo
        modified = gizmo.manipulate(
            camera_view,
            camera_proj,
            self._gizmo_operation,
            self._gizmo_mode,
            object_matrix,
        )

        if modified:
            # Extract new transform from modified matrix (column-major → row-major)
            new_mat = np.array(object_matrix.values).reshape(4, 4).T
            # Cache the matrix for next frame to avoid qpos round-trip jitter
            self._gizmo_cached_matrix = new_mat.copy()

            new_pos = new_mat[:3, 3]
            new_rot = R.from_matrix(new_mat[:3, :3])
            new_quat_xyzw = new_rot.as_quat()  # scipy: x,y,z,w
            new_quat_wxyz = [new_quat_xyzw[3], new_quat_xyzw[0], new_quat_xyzw[1], new_quat_xyzw[2]]

            # Read current qpos for non-free-joint DOFs
            qpos_raw = entity.get_qpos()
            qpos_np = qpos_raw.cpu().numpy() if hasattr(qpos_raw, "cpu") else np.asarray(qpos_raw)
            is_multi_env = qpos_np.ndim == 2
            qpos = qpos_np[0] if is_multi_env else qpos_np.flatten()

            # Update only the free-joint DOFs
            new_qpos = list(qpos)
            new_qpos[qs : qs + 3] = new_pos.tolist()
            new_qpos[qs + 3 : qs + 7] = new_quat_wxyz

            # Auto-pause on gizmo edit
            self.paused = True

            with self.viewer.render_lock:
                self._apply_qpos_update(entity, new_qpos, is_multi_env)
        elif not gizmo.is_using():
            # Clear cache when drag ends so next interaction reads fresh qpos
            self._gizmo_cached_matrix = None

    def _is_gizmo_active(self):
        """Check if the gizmo is being used (for input blocking)."""
        if self._gizmo is not None:
            return self._gizmo.is_using() or self._gizmo.is_over()
        return False

    def _render_camera_controls(self):
        """Render camera position, lookat, FOV controls."""
        imgui = self._imgui
        trackball = self.viewer._trackball

        # Read current camera state from trackball
        pose = trackball._n_pose
        pos = [float(pose[0, 3]), float(pose[1, 3]), float(pose[2, 3])]
        # Use trackball's actual orbit center as lookat (not derived from z-axis)
        target = trackball._n_target
        lookat = [float(target[0]), float(target[1]), float(target[2])]

        # Position drag
        changed_pos, new_pos = imgui.drag_float3("Position##cam_pos", pos, 0.05, -100.0, 100.0, "%.2f")

        # Lookat drag
        changed_lookat, new_lookat = imgui.drag_float3("Lookat##cam_lookat", lookat, 0.05, -100.0, 100.0, "%.2f")

        if changed_pos or changed_lookat:
            cam_pos = np.array(list(new_pos)) if changed_pos else np.array(pos)
            cam_lookat = np.array(list(new_lookat)) if changed_lookat else np.array(lookat)
            # Build pose with fixed world-up to prevent unintuitive roll
            from genesis.utils import geom as gu

            world_up = np.array([0.0, 0.0, 1.0])
            cam_pose = gu.pos_lookat_up_to_T(cam_pos, cam_lookat, world_up)
            self.scene.viewer._camera_up = cam_pose[:3, 1].copy()
            trackball.set_camera_pose(cam_pose)
            # Sync trackball orbit center so mouse orbiting works correctly after
            trackball._n_target = cam_lookat.copy()
            trackball._target = cam_lookat.copy()

        # FOV slider
        fov_deg = float(self.camera.camera.yfov * 180.0 / np.pi)
        changed_fov, new_fov = imgui.slider_float("FOV##cam_fov", fov_deg, 15.0, 120.0, "%.1f")
        if changed_fov:
            self.camera.camera.yfov = new_fov * np.pi / 180.0

        # Reset Camera button
        if imgui.button("Reset Camera", size=(120, 0)):
            self.viewer._reset_view()

    _FILE_EXTENSIONS = {
        "URDF": {".urdf"},
        "MJCF": {".xml"},
        "Mesh": {".obj", ".stl", ".ply", ".dae", ".glb", ".gltf"},
    }

    def _render_file_browser(self, morph_type):
        """Render a file browser popup for selecting asset files."""
        imgui = self._imgui
        if not self._file_browser_open:
            return

        imgui.open_popup("File Browser##file_popup")
        imgui.set_next_window_size((500, 400))
        if imgui.begin_popup_modal("File Browser##file_popup")[0]:
            # Current directory display with parent navigation
            if imgui.button("^##parent_dir"):
                parent = os.path.dirname(self._file_browser_dir)
                if parent != self._file_browser_dir:
                    self._file_browser_dir = parent
                    self._file_browser_selected = -1
            imgui.same_line()
            imgui.text(self._file_browser_dir)
            imgui.separator()

            # List directory contents
            valid_exts = self._FILE_EXTENSIONS.get(morph_type, set())
            try:
                entries = sorted(os.listdir(self._file_browser_dir))
            except OSError:
                entries = []

            dirs = [
                e for e in entries if os.path.isdir(os.path.join(self._file_browser_dir, e)) and not e.startswith(".")
            ]
            files = [
                e
                for e in entries
                if os.path.isfile(os.path.join(self._file_browser_dir, e))
                and (not valid_exts or os.path.splitext(e)[1].lower() in valid_exts)
            ]
            items = [d + "/" for d in dirs] + files

            if imgui.begin_child("file_list", size=(0, -30)):
                for idx, item in enumerate(items):
                    is_dir = item.endswith("/")
                    selected = idx == self._file_browser_selected
                    if imgui.selectable(item, selected)[0]:
                        if is_dir:
                            self._file_browser_dir = os.path.join(self._file_browser_dir, item[:-1])
                            self._file_browser_selected = -1
                        else:
                            self._file_browser_selected = idx
                    # Double-click on file to confirm
                    if not is_dir and imgui.is_item_hovered() and imgui.is_mouse_double_clicked(0):
                        self._add_entity_file = os.path.join(self._file_browser_dir, item)
                        self._file_browser_open = False
                        imgui.close_current_popup()
                imgui.end_child()

            # OK / Cancel buttons
            can_select = self._file_browser_selected >= 0 and self._file_browser_selected >= len(dirs)
            if imgui.button("OK", size=(80, 0)) and can_select:
                file_name = files[self._file_browser_selected - len(dirs)]
                self._add_entity_file = os.path.join(self._file_browser_dir, file_name)
                self._file_browser_open = False
                imgui.close_current_popup()
            imgui.same_line()
            if imgui.button("Cancel", size=(80, 0)):
                self._file_browser_open = False
                imgui.close_current_popup()

            imgui.end_popup()
        else:
            # Popup was closed (e.g. clicking outside)
            self._file_browser_open = False

    def _render_scene_editor(self):
        """Render scene editing controls (entity scale, add entity, rebuild)."""
        imgui = self._imgui

        # Per-entity scale editing
        for i, spec in enumerate(self._entity_specs):
            morph = spec["morph"]
            morph_name = type(morph).__name__
            file_name = getattr(morph, "file", "")

            imgui.text(f"{morph_name}: {file_name or '(builtin)'}")

            # Scale editing
            current_scale = spec["scale"]
            if isinstance(current_scale, (list, tuple, np.ndarray)):
                scale_val = float(current_scale[0]) if len(current_scale) > 0 else 1.0
            else:
                scale_val = float(current_scale)
            changed, new_scale = imgui.drag_float(f"Scale##scale_{i}", scale_val, 0.01, 0.01, 100.0, "%.3f")
            if changed:
                spec["scale"] = new_scale
                self._specs_dirty = True

            # Remove button
            imgui.same_line()
            if imgui.button(f"X##remove_{i}"):
                self._entity_specs.pop(i)
                self._specs_dirty = True
                break

            imgui.separator()

        # Add entity section
        if imgui.collapsing_header("Add Entity##add_entity"):
            imgui.indent()
            changed_type, self._add_entity_morph_type = imgui.combo(
                "Type##add_type", self._add_entity_morph_type, _MORPH_TYPES
            )

            morph_type = _MORPH_TYPES[self._add_entity_morph_type]
            # Default fixed=True for Plane when type changes
            if changed_type and morph_type == "Plane":
                self._add_entity_fixed = True

            # File path for file-based morphs
            if morph_type in ("URDF", "MJCF", "Mesh"):
                _, self._add_entity_file = imgui.input_text("File##add_file", self._add_entity_file, 256)
                imgui.same_line()
                if imgui.button("Browse##add_browse"):
                    self._file_browser_open = True
                    self._file_browser_selected = -1
                    # Start browsing from current file's directory if set
                    if self._add_entity_file:
                        parent = os.path.dirname(self._add_entity_file)
                        if os.path.isdir(parent):
                            self._file_browser_dir = parent

                self._render_file_browser(morph_type)

                _, self._add_entity_scale = imgui.drag_float(
                    "Scale##add_scale", self._add_entity_scale, 0.01, 0.01, 100.0, "%.3f"
                )

            # Type-specific geometry params
            if morph_type == "Box":
                _, self._add_box_size = imgui.drag_float3(
                    "Size##add_box_size", self._add_box_size, 0.01, 0.01, 100.0, "%.3f"
                )
            elif morph_type == "Sphere":
                _, self._add_sphere_radius = imgui.drag_float(
                    "Radius##add_sphere_r", self._add_sphere_radius, 0.01, 0.01, 100.0, "%.3f"
                )
            elif morph_type == "Cylinder":
                _, self._add_cylinder_radius = imgui.drag_float(
                    "Radius##add_cyl_r", self._add_cylinder_radius, 0.01, 0.01, 100.0, "%.3f"
                )
                _, self._add_cylinder_height = imgui.drag_float(
                    "Height##add_cyl_h", self._add_cylinder_height, 0.01, 0.01, 100.0, "%.3f"
                )

            # Position (all types except Plane)
            if morph_type != "Plane":
                _, self._add_entity_pos = imgui.drag_float3(
                    "Position##add_pos", self._add_entity_pos, 0.05, -100.0, 100.0, "%.2f"
                )

            # Fixed checkbox
            _, self._add_entity_fixed = imgui.checkbox("Fixed##add_fixed", self._add_entity_fixed)

            if imgui.button("Add##add_btn"):
                pos = tuple(self._add_entity_pos)
                scale = self._add_entity_scale
                fixed = self._add_entity_fixed
                box_size = tuple(self._add_box_size)
                morph_cls_map = {
                    "URDF": lambda: gs.morphs.URDF(file=self._add_entity_file, pos=pos, scale=scale, fixed=fixed),
                    "MJCF": lambda: gs.morphs.MJCF(file=self._add_entity_file, pos=pos, scale=scale, fixed=fixed),
                    "Mesh": lambda: gs.morphs.Mesh(file=self._add_entity_file, pos=pos, scale=scale, fixed=fixed),
                    "Box": lambda: gs.morphs.Box(pos=pos, size=box_size, fixed=fixed),
                    "Sphere": lambda: gs.morphs.Sphere(pos=pos, radius=self._add_sphere_radius, fixed=fixed),
                    "Cylinder": lambda: gs.morphs.Cylinder(
                        pos=pos, radius=self._add_cylinder_radius, height=self._add_cylinder_height, fixed=fixed
                    ),
                    "Plane": lambda: gs.morphs.Plane(),
                }
                new_morph = morph_cls_map[morph_type]()
                self._entity_specs.append(
                    {
                        "morph": new_morph,
                        "material": None,
                        "surface": None,
                        "visualize_contact": False,
                        "scale": scale,
                    }
                )
                self._specs_dirty = True
            imgui.unindent()

        # Rebuild button
        if self._specs_dirty:
            imgui.text_colored((1.0, 0.7, 0.0, 1.0), "Changes pending")
        if imgui.button("Rebuild Scene", size=(150, 0)):
            # Update morph scale values before rebuild
            for spec in self._entity_specs:
                morph = spec["morph"]
                if hasattr(morph, "scale"):
                    morph.scale = spec["scale"]
            # Signal rebuild to main thread (don't call _rebuild_fn from viewer thread)
            self._rebuild_requested = True
            self._specs_dirty = False

    def _render_entity_browser(self):
        """Render entity list with joint sliders."""
        imgui = self._imgui

        if not self._entity_cache:
            imgui.text("No controllable entities")
            return

        for entity_idx, data in self._entity_cache.items():
            entity = data["entity"]
            expanded = imgui.collapsing_header(
                f"{data['name']}##entity_{entity_idx}", flags=imgui.TreeNodeFlags_.default_open
            )
            if not expanded:
                continue

            imgui.indent()

            # DOF count display
            imgui.text(f"DOFs: {data['n_dofs']}")

            # Vis mode combo
            vis_modes = ["visual", "collision"]
            current_mode = entity.surface.vis_mode
            current_mode_idx = vis_modes.index(current_mode) if current_mode in vis_modes else 0
            changed_mode, new_mode_idx = imgui.combo(f"Vis Mode##vis_{entity_idx}", current_mode_idx, vis_modes)
            if changed_mode:
                self._switch_entity_vis_mode(entity, vis_modes[new_mode_idx])

            # Per-entity wireframe toggle
            is_wireframe = self._wireframe_state.get(entity_idx, False)
            changed_wf, new_wf = imgui.checkbox(f"Wireframe##wf_{entity_idx}", is_wireframe)
            if changed_wf:
                self._wireframe_state[entity_idx] = new_wf
                self._set_entity_wireframe(entity, new_wf)

            # Visualize contact toggle
            show_contact = entity.visualize_contact
            changed_contact, new_contact = imgui.checkbox(f"Show Contacts##contact_{entity_idx}", show_contact)
            if changed_contact:
                entity._visualize_contact = new_contact
                for link in entity.links:
                    link._visualize_contact = new_contact

            # Gizmo toggle for free-joint entities
            if data.get("has_free_joint") and self._gizmo is not None:
                gizmo_active = self._gizmo_entity_idx == entity_idx
                changed_gizmo, new_gizmo = imgui.checkbox(f"Gizmo##gizmo_{entity_idx}", gizmo_active)
                if changed_gizmo:
                    self._gizmo_entity_idx = entity_idx if new_gizmo else -1
                if gizmo_active:
                    imgui.same_line()
                    gizmo = self._gizmo
                    if imgui.radio_button(
                        f"Translate##gop_{entity_idx}", self._gizmo_operation == gizmo.OPERATION.translate
                    ):
                        self._gizmo_operation = gizmo.OPERATION.translate
                    imgui.same_line()
                    if imgui.radio_button(f"Rotate##gop_{entity_idx}", self._gizmo_operation == gizmo.OPERATION.rotate):
                        self._gizmo_operation = gizmo.OPERATION.rotate

            # Joint sections only for entities with DOFs
            if data["n_dofs"] > 0:
                # Get qpos - handle multi-env case by using only env 0
                qpos_raw = entity.get_qpos()
                qpos_np = qpos_raw.cpu().numpy() if hasattr(qpos_raw, "cpu") else np.asarray(qpos_raw)

                # If multi-env (2D tensor with shape [n_envs, n_qs]), use only env 0
                is_multi_env = qpos_np.ndim == 2
                if is_multi_env:
                    qpos = qpos_np[0]
                else:
                    qpos = qpos_np.flatten()

                changed_any = False
                new_qpos = list(qpos)

                # Joint control section
                if imgui.collapsing_header(f"Joint Control##joints_{entity_idx}"):
                    imgui.indent()

                    # Euler/Quat toggle for free-joint entities
                    use_euler = False
                    if data.get("has_free_joint"):
                        rot_mode = self._rotation_mode.get(entity_idx, "quat")
                        if imgui.radio_button(f"Quaternion##rotmode_{entity_idx}", rot_mode == "quat"):
                            self._rotation_mode[entity_idx] = "quat"
                            rot_mode = "quat"
                        imgui.same_line()
                        if imgui.radio_button(f"Euler##rotmode_{entity_idx}", rot_mode == "euler"):
                            self._rotation_mode[entity_idx] = "euler"
                            rot_mode = "euler"
                        use_euler = rot_mode == "euler"

                    if use_euler:
                        # Euler mode: show position + euler angles from get_dofs_position
                        changed_any = self._render_joints_euler_mode(
                            entity, data, entity_idx, is_multi_env, qpos, new_qpos
                        )
                    else:
                        # Quat mode: show all qpos components
                        lower, upper = data["q_limits"]
                        for i, (name, val, lo, hi, is_quat) in enumerate(
                            zip(data["q_names"], qpos, lower, upper, data["q_is_quaternion"])
                        ):
                            if is_quat:
                                changed, new_val = imgui.drag_float(
                                    f"{name}##{entity_idx}_{i}", float(val), 0.01, float(lo), float(hi), "%.4f"
                                )
                            else:
                                changed, new_val = imgui.slider_float(
                                    f"{name}##{entity_idx}_{i}", float(val), float(lo), float(hi), "%.3f"
                                )
                            if changed:
                                new_qpos[i] = new_val
                                changed_any = True
                    imgui.unindent()

                if changed_any:
                    # Auto-pause when user edits joints
                    self.paused = True
                    if not (data.get("has_free_joint") and self._rotation_mode.get(entity_idx) == "euler"):
                        # Normalize any edited quaternion groups (quat mode only)
                        for qstart, qend in data["quat_groups"]:
                            q = np.array(new_qpos[qstart:qend])
                            norm = np.linalg.norm(q)
                            if norm > 1e-8:
                                q /= norm
                                new_qpos[qstart:qend] = q.tolist()
                    with self.viewer.render_lock:
                        self._apply_qpos_update(entity, new_qpos, is_multi_env)

            imgui.unindent()

    def _render_joints_euler_mode(self, entity, data, entity_idx, is_multi_env, qpos, new_qpos):
        """Render free joint as position + euler angles, plus remaining joints normally.

        Free joint edits are applied immediately via set_dofs_position.
        Non-free joint edits update new_qpos for the caller to apply.
        Returns True if any non-free-joint value changed (needing set_qpos).
        """
        imgui = self._imgui
        non_free_changed = False
        qs = data["free_joint_q_start"]

        # Get dofs_position for euler angles
        dofs_raw = entity.get_dofs_position()
        dofs_np = dofs_raw.cpu().numpy() if hasattr(dofs_raw, "cpu") else np.asarray(dofs_raw)
        dofs = dofs_np[0] if dofs_np.ndim == 2 else dofs_np.flatten()

        # Position (first 3 dofs = same as first 3 qpos for free joint)
        pos = [float(dofs[0]), float(dofs[1]), float(dofs[2])]
        changed_pos, new_pos = imgui.drag_float3(
            f"Position##euler_pos_{entity_idx}", pos, 0.05, -FREE_JOINT_POS_LIMIT, FREE_JOINT_POS_LIMIT, "%.3f"
        )

        # Euler angles (dofs 3-5, in radians, display as degrees)
        euler_rad = [float(dofs[3]), float(dofs[4]), float(dofs[5])]
        euler_deg = [np.degrees(e) for e in euler_rad]
        changed_rot, new_euler_deg = imgui.drag_float3(
            f"Euler (deg)##euler_rot_{entity_idx}", euler_deg, 0.5, -360.0, 360.0, "%.1f"
        )

        if changed_pos or changed_rot:
            self.paused = True
            new_dofs = list(dofs)
            if changed_pos:
                new_dofs[0], new_dofs[1], new_dofs[2] = new_pos
            if changed_rot:
                new_dofs[3] = np.radians(new_euler_deg[0])
                new_dofs[4] = np.radians(new_euler_deg[1])
                new_dofs[5] = np.radians(new_euler_deg[2])

            # Use set_dofs_position for the whole entity (handles euler->quat internally)
            dofs_array = np.asarray(new_dofs)
            with self.viewer.render_lock:
                if is_multi_env:
                    entity.set_dofs_position(dofs_array, envs_idx=0)
                else:
                    entity.set_dofs_position(dofs_array)
                refresh_visual_transforms(self.scene, self.viewer.gs_context)

            # Refresh new_qpos with updated free joint qpos (euler->quat conversion happened)
            fresh_raw = entity.get_qpos()
            fresh_qpos = fresh_raw.cpu().numpy() if hasattr(fresh_raw, "cpu") else np.asarray(fresh_raw)
            fresh = fresh_qpos[0] if fresh_qpos.ndim == 2 else fresh_qpos.flatten()
            for i in range(qs, qs + 7):
                new_qpos[i] = float(fresh[i])

        # Render remaining (non-free) joints normally
        lower, upper = data["q_limits"]
        free_end = qs + 7  # free joint takes 7 qpos slots
        for i, (name, val, lo, hi, is_quat) in enumerate(
            zip(data["q_names"], qpos, lower, upper, data["q_is_quaternion"])
        ):
            if qs <= i < free_end:
                continue  # Skip free joint components (handled above)
            if is_quat:
                changed, new_val = imgui.drag_float(
                    f"{name}##{entity_idx}_{i}", float(val), 0.01, float(lo), float(hi), "%.4f"
                )
            else:
                changed, new_val = imgui.slider_float(
                    f"{name}##{entity_idx}_{i}", float(val), float(lo), float(hi), "%.3f"
                )
            if changed:
                new_qpos[i] = new_val
                non_free_changed = True

        return non_free_changed

    def should_step(self) -> bool:
        """Check if simulation should advance this frame."""
        if self._steps_remaining > 0:
            self._steps_remaining -= 1
            return True
        # Legacy single-step support
        if self._step_requested:
            self._step_requested = False
            return True
        return not self.paused

    def on_close(self) -> None:
        """Clean up ImGui resources."""
        if self._available and self._impl:
            self._impl.shutdown()
        if self._imgui:
            self._imgui.destroy_context()
