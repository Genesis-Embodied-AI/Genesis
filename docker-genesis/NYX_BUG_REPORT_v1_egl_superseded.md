# Nyx renderer segfaults in `libnvidia-eglcore` on H20-3e (SM90) / driver 570.133.20

## Summary

`gs_nyx` (Nyx renderer) crashes with **SIGSEGV inside `libnvidia-eglcore.so`**
during `scene.build()` when a `NyxCameraOptions` sensor is present. The crash is
in NVIDIA's EGL core library, reached from `nyx_py_renderer.abi3.so` during the
renderer's native `startup()`. It is reproducible with the official
`genesis-nyx/examples/01_hello_nyx.py` and a minimal 2-line scene. The rest of
the stack (rigid sim, pyrender, Madrona, Vulkan device enumeration) works on the
same machine, so this is specific to Nyx's EGL-based context creation.

## Environment

| | |
|---|---|
| GPU | 8× NVIDIA **H20-3e** (Hopper, SM 90, 143 GB), headless (no display) |
| Driver | **570.133.20**, CUDA 12.8 |
| Host OS | Alibaba Cloud Linux 3 (glibc 2.32) |
| Container | `pytorch/pytorch:2.11.0-cuda12.8-cudnn9-devel` (Ubuntu 24.04, glibc 2.39) |
| Run | `docker run --gpus all -e NVIDIA_DRIVER_CAPABILITIES=all` |
| Python | 3.12 |
| genesis-world | 1.0.0 |
| quadrants | 1.0.0 |
| gs-nyx | 0.1.1 |
| gs-nyx-plugin | 0.1.2 |

## Reproduction

```python
import faulthandler; faulthandler.enable()
import genesis as gs
import gs_nyx.nyx_py_renderer as npr
from gs_nyx_plugin.nyx_camera_options import NyxCameraOptions
gs.init()
scene = gs.Scene(sim_options=gs.options.SimOptions(dt=0.01), show_viewer=False)
scene.add_entity(morph=gs.morphs.Plane(plane_size=(10.0, 10.0)))
cam = scene.add_sensor(NyxCameraOptions(
    res=(320, 240), pos=(-1.0, 1.0, 1.2), lookat=(0, 0, 0.1),
    fov=20.0, spp=4, render_mode=npr.ERenderMode.FastPathTracer,
))
scene.build(n_envs=1)   # <-- SIGSEGV here
```

`examples/01_hello_nyx.py` from genesis-nyx crashes identically.

## Crash signature

Python `faulthandler` points at the Nyx renderer startup:

```
Fatal Python error: Segmentation fault
Current thread:
  File ".../gs_nyx_plugin/nyx_renderer.py", line 445 in build   # self._renderer.startup(startup_params)
  File ".../gs_nyx_plugin/nyx_camera_sensor.py", line 373 in build
  File ".../genesis/engine/sensors/sensor_manager.py", line 248 in build
  File ".../genesis/engine/simulator.py", line 218 in build
  File ".../genesis/engine/scene.py", line 891 in build
```

`gdb` native backtrace — the fault is **inside NVIDIA's EGL core**, called from
the Nyx renderer `.so`:

```
Thread 1 "python3" received signal SIGSEGV, Segmentation fault.
0x...58cf in ?? () from /lib/x86_64-linux-gnu/libnvidia-eglcore.so.570.133.20
#0  ...  libnvidia-eglcore.so.570.133.20
#1  ...  libnvidia-eglcore.so.570.133.20
#2  ...  gs_nyx/nyx_py_renderer.abi3.so
#3  ...  gs_nyx/nyx_py_renderer.abi3.so
...
#12 ...  gs_nyx/nyx_py_renderer.abi3.so
#13 PyObject_Vectorcall ()
```

## What works on the same machine (rules out a general GPU/driver problem)

- Rigid-body sim on GPU (with `quadrants==1.0.0`).
- **pyrender** rasterizer offscreen render (EGL via `PYOPENGL_PLATFORM=egl`).
- **Madrona** `BatchRenderer`.
- `vulkaninfo` enumerates all 8 H20-3e devices (`VK_LAYER_NV_optimus`, GPU id 0–7).
- `libcuda.so.1`, `libEGL_nvidia`, `libGLX_nvidia`, `libvulkan.so.1` all present
  and the NVIDIA Vulkan ICD (`/usr/share/vulkan/icd.d/nvidia_icd.json`) is read.

## Notes / suspected cause

- `strings nyx_py_renderer.abi3.so` shows the renderer links **GLFW** and uses
  `glfwCreateWindow` / `glfwCreateWindowSurface` / `VK_KHR_swapchain`, plus a
  `"Vulkan API: GLFW does not support Vulkan."` string. The startup path appears
  to create a GLFW window / EGL context even though the example sets
  `show_viewer=False` and the camera is offscreen.
- On a **headless datacenter GPU with no display output** (H20-3e), NVIDIA's
  `libnvidia-eglcore` context creation segfaults. There was no `DISPLAY` and
  `XDG_RUNTIME_DIR` was unset (setting `XDG_RUNTIME_DIR` did not help).
- Request: a fully headless / surfaceless render path (EGL surfaceless or pure
  Vulkan offscreen, no GLFW window) for Nyx, or guidance on how to force one.

## Ask

Is there a supported way to run Nyx fully headless (no window/surface) on a
display-less Hopper GPU? If not, this looks like a bug in Nyx's startup creating
a window-backed context that NVIDIA's EGL core cannot satisfy on H20-3e.
