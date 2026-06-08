"""Nyx headless-rendering diagnosis / fix verification.

Background
----------
Nyx's native renderer (``nyx_py_renderer``) links GLFW and calls
``glfwInit()`` + ``glfwGetRequiredInstanceExtensions()`` unconditionally at
startup, even when the camera is offscreen (``openWindow=False``). On a
display-less Hopper box those required instance extensions are the X11/Wayland
WSI surface extensions; enabling them in the Vulkan instance drives the NVIDIA
driver into its windowing (``libnvidia-eglcore``) path, which segfaults when
there is no display server.

The fix tested here: give GLFW a real (virtual) X11 display via Xvfb. Run:

    xvfb-run -a -s "-screen 0 640x480x24" python3 nyx_headless_test.py

If that survives ``scene.build()`` and writes an image, the segfault was the
missing-display WSI probe, and Xvfb is the headless workaround.

Plain ``python3 nyx_headless_test.py`` (no Xvfb) reproduces the crash.
"""
import faulthandler
import os
import sys

faulthandler.enable()

print(f"[env] DISPLAY={os.environ.get('DISPLAY')!r} "
      f"XDG_RUNTIME_DIR={os.environ.get('XDG_RUNTIME_DIR')!r}", flush=True)

import numpy as np
import genesis as gs
import gs_nyx.nyx_py_renderer as npr
from gs_nyx_plugin.nyx_camera_options import NyxCameraOptions

gs.init()
scene = gs.Scene(sim_options=gs.options.SimOptions(dt=0.01), show_viewer=False)
scene.add_entity(morph=gs.morphs.Plane(plane_size=(10.0, 10.0)))
scene.add_entity(morph=gs.morphs.Sphere(radius=0.3, pos=(0.0, 0.0, 0.5)))
cam = scene.add_sensor(NyxCameraOptions(
    res=(320, 240), pos=(-1.2, 1.2, 1.2), lookat=(0.0, 0.0, 0.3),
    fov=30.0, spp=8, render_mode=npr.ERenderMode.FastPathTracer,
))

print("[step] scene.build(n_envs=1) ...", flush=True)
scene.build(n_envs=1)
print("[ok]   scene.build survived (no segfault)", flush=True)

scene.step()
img = cam.read()
print(f"[read] type={type(img)}", flush=True)

# img may be torch CUDA tensor (N,H,W,3) or numpy
if hasattr(img, "cpu"):
    img = img.cpu().numpy()
img = np.asarray(img)
print(f"[read] shape={img.shape} dtype={img.dtype} mean={float(img.mean()):.2f}",
      flush=True)

# squeeze env dim if present
while img.ndim > 3:
    img = img[0]
if img.dtype != np.uint8:
    img = np.clip(img, 0, 255).astype(np.uint8)

out = "/workspace/Uni-Genesis/docker-genesis/render_out/nyx_headless.png"
os.makedirs(os.path.dirname(out), exist_ok=True)
try:
    from PIL import Image
    Image.fromarray(img).save(out)
    print(f"[save] wrote {out}", flush=True)
except Exception as e:
    print(f"[save] PIL unavailable ({e}); skipping write", flush=True)

print("[done] Nyx headless render succeeded", flush=True)
