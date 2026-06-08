# Nyx renderer crashes during `scene.build()` on a headless H20-3e: zero-size external-memory allocation (unchecked) + divide-by-zero in renderer resource setup

## Summary

With a **version-matched** stack (`genesis-world` 1.x + `gs-nyx` 0.1.2 + `gs-nyx-plugin` 0.1.3),
`scene.build()` crashes inside the **Nyx native renderer** (`nyx_py_renderer.abi3.so`)
on a display-less NVIDIA H20-3e. Step-by-step debugging shows the crash is **not**
an EGL/display/driver problem (an earlier `libnvidia-eglcore` backtrace was a
*secondary* symptom; see `NYX_BUG_REPORT_v1_egl_superseded.md`). The real defects
are two unchecked degenerate-scene assumptions in the renderer's Vulkan resource
setup:

1. **Unchecked zero-size external-memory allocation.** During build, Nyx calls
   `vkAllocateMemory` with `allocationSize == 0` (with a `pNext` external-memory
   chain — the CUDA↔Vulkan interop buffer). `allocationSize == 0` is invalid
   Vulkan usage; the NVIDIA driver returns `VK_ERROR_OUT_OF_DEVICE_MEMORY (-2)`
   and writes a **NULL** `VkDeviceMemory`. Nyx **does not check the result** and
   passes the NULL handle straight into `vkBindBufferMemory`, which the driver
   dereferences → **SIGSEGV** (inside `libnvidia-eglcore`, called from
   `vkBindBufferMemory`).

2. **Divide-by-zero in the same resource-setup path.** If the zero-size
   allocation is forced to succeed (work-around: bump size to a non-zero floor),
   the renderer proceeds and then hits a hardware **SIGFPE** — an integer
   `div` by a zero divisor (`div %rsi`, `rsi = 0`) in the same call chain.
   Both faults share one root cause: **some element count is `0`** and is used
   unchecked as an allocation size and as a divisor.

Both are reproducible with the official `examples/01_hello_nyx.py` and with a
2-entity minimal scene; scene richness (extra geometry, env map) does not change
the outcome.

## Environment

| | |
|---|---|
| GPU | NVIDIA **H20-3e** (Hopper, SM 90, 143 GB), headless (no display) |
| Driver | **570.133.20**, CUDA 12.8, Vulkan 1.4.303 |
| Host OS | Alibaba Cloud Linux 3 (glibc 2.32) |
| Container | `pytorch/pytorch:2.11.0-cuda12.8-cudnn9-devel` (Ubuntu 24.04, glibc 2.39) |
| Python | 3.12 |
| genesis-world | 1.0.0 (mounted editable) |
| gs-nyx | **0.1.2** (cp312-abi3 wheel) |
| gs-nyx-plugin | **0.1.3** |
| quadrants | 1.0.0 |

> Note on versions: `gs-nyx-plugin` **0.1.2** is built against the *pre-1.0*
> Genesis sensor API (`Sensor[Options, Metadata, Data]`, 3 generic params;
> `__init__(self, options, idx, manager)`). On genesis-world ≥ 1.0 it fails at
> `add_sensor` with `TypeError: NyxCameraData.__new__() missing 1 required
> positional argument: 'rgb'` before ever reaching `build()`. **Plugin 0.1.3**
> fixes the sensor API (`Sensor[Options, Context, Metadata, Data]`, 4 params;
> 5-arg `__init__`) and is required for genesis-world ≥ 1.0. All findings below
> are with 0.1.3 + gs-nyx 0.1.2.

## Reproduction

```python
import genesis as gs
import gs_nyx.nyx_py_renderer as npr
from gs_nyx_plugin.nyx_camera_options import NyxCameraOptions

gs.init()
scene = gs.Scene(sim_options=gs.options.SimOptions(dt=0.01), show_viewer=False)
scene.add_entity(morph=gs.morphs.Plane(plane_size=(10.0, 10.0)))
scene.add_entity(morph=gs.morphs.Sphere(radius=0.3, pos=(0, 0, 0.5)))
cam = scene.add_sensor(NyxCameraOptions(
    res=(320, 240), pos=(-1, 1, 1.2), lookat=(0, 0, 0.1),
    fov=20.0, spp=4, render_mode=npr.ERenderMode.FastPathTracer,
))
scene.build(n_envs=1)   # <-- SIGSEGV (bug #1)
```

`examples/01_hello_nyx.py` crashes identically.

## Evidence

### Bug #1 — unchecked zero-size external-memory allocation → NULL bind → SIGSEGV

`gdb` backtrace (symbols in the renderer are stripped; frame #2 is the call
site, disassembled):

```
Thread 1 "python3" received signal SIGSEGV, Segmentation fault.
#0  ...58cf in libnvidia-eglcore.so.570.133.20      ; mov 0xa0(%rbp),%r12  (rax=0, rbx=0)
#1  ...181f2 in libnvidia-eglcore.so.570.133.20
#2  ...c59fe in gs_nyx/nyx_py_renderer.abi3.so      ; instruction AFTER:
                                                    ;   call vkBindBufferMemory@plt
                                                    ;   test %eax,%eax
#3..#12  gs_nyx/nyx_py_renderer.abi3.so
#13 PyObject_Vectorcall
```

Intercepting `vkAllocateMemory` / `vkBindBufferMemory` (via `LD_PRELOAD`) shows
every prior allocation succeeds, then the failing one:

```
vkAllocateMemory size=32768 typeIdx=1 pNext=(nil)        -> r=0  mem=0x...    (ok)
vkAllocateMemory size=0     typeIdx=2 pNext=0x7fff...     -> r=-2 mem=0x0      <-- size 0, external-mem chain
vkBindBufferMemory ... mem=0x0                                                 <-- binds NULL -> driver SIGSEGV
```

- `r = -2` is `VK_ERROR_OUT_OF_DEVICE_MEMORY` (the driver rejecting
  `allocationSize == 0`, which is invalid per the Vulkan spec:
  *"allocationSize must be greater than 0"*).
- The `pNext != NULL` only on this allocation indicates an external-memory
  struct (`VkExportMemoryAllocateInfo` / `VkImportMemoryFdInfoKHR`), i.e. the
  CUDA-interop buffer.
- The pure-Vulkan control (below) proves the driver is healthy; the NULL bind is
  Nyx feeding the driver an invalid handle.

### Bug #2 — divide-by-zero in the same path → SIGFPE

Forcing the size-0 allocation to succeed (bump to any non-zero floor; tested
256 / 4096 / 65536 — identical result) advances past the NULL bind, then:

```
Thread 1 "python3" received signal SIGFPE, Arithmetic exception.
#0  ...c84b0 in gs_nyx/nyx_py_renderer.abi3.so      ; => div %rsi   (rsi = 0)
#1..#9  gs_nyx/nyx_py_renderer.abi3.so              ; same functions as the SIGSEGV chain
#10 PyObject_Vectorcall
```

The SIGFPE call chain is the **same renderer functions** as the SIGSEGV chain
(same offsets, different load base), confirming a shared root cause: a count
that is `0` is used both as an allocation size and as an integer divisor without
a guard. The divisor is independent of the work-around's bumped value.

### Control — the driver and headless Vulkan are fine

A ~20-line pure-Vulkan program on the same node:

```
vkCreateInstance           -> VK_SUCCESS
vkEnumeratePhysicalDevices -> 2 (NVIDIA H20-3e)
vkCreateDevice (0 ext)     -> VK_SUCCESS
vkDestroyDevice / vkDestroyInstance -> clean exit, no crash
```

`vulkaninfo --summary` enumerates both GPUs (`apiVersion 1.4.303`, driver
`NVIDIA`). So `vkCreateInstance` / `vkCreateDevice` / WSI selection are **not**
the problem — ruling out the headless-EGL theory. (For completeness: forcing
GLFW's Null platform and stripping all WSI instance/device extensions
— `VK_KHR_swapchain`, surface extensions — does **not** change the crash; the
fault is purely in the memory-allocation path.)

## Root cause

In the Nyx native renderer's per-build resource setup, an element count can be
`0` for a (trivial or interop) resource, and that count is used **unchecked**:

- as `VkMemoryAllocateInfo::allocationSize` → invalid `vkAllocateMemory(size=0)`,
  whose `VK_ERROR_OUT_OF_DEVICE_MEMORY` return and NULL `VkDeviceMemory` are not
  checked before `vkBindBufferMemory` → NULL deref in the driver (SIGSEGV);
- as an integer divisor → `div` by zero (SIGFPE) when the allocation is made to
  succeed.

## Suggested fixes (upstream, in `gs-nyx`)

1. Never call `vkAllocateMemory` with `allocationSize == 0`; clamp to a minimum
   (or skip the resource) when the underlying count is 0.
2. **Check the result of `vkAllocateMemory`** (and all fallible Vulkan calls)
   before using the handle; never bind a NULL `VkDeviceMemory`.
3. Guard the divisor in the same resource-setup routine against 0.

## Work-around used locally (not a fix)

An `LD_PRELOAD` shim (`nyx_headless_shim.c`, attached) that:
- forces GLFW's headless Null platform (so `glfwInit()` succeeds with no DISPLAY),
- bumps `vkAllocateMemory(size=0)` to a non-zero floor,
- returns `VK_ERROR_OUT_OF_DEVICE_MEMORY` instead of crashing on a NULL
  `vkBindBufferMemory`.

This converts the SIGSEGV into a clean error and exposes Bug #2 (SIGFPE), which
the shim cannot paper over (hardware divide fault). **A real fix must be in
`gs-nyx`.**

## How to reproduce the debugging

```bash
# In the genesis-dev container, with genesis mounted:
pip install --no-deps --force-reinstall gs_nyx-0.1.2-cp312-abi3-*.whl gs_nyx_plugin-0.1.3-*.whl
gcc -shared -fPIC -O2 nyx_headless_shim.c -o nyx_headless_shim.so -ldl

# Bug #1 (SIGSEGV) — without the shim:
python3 -c "import genesis ...; scene.build(...)"        # SIGSEGV at vkBindBufferMemory

# See the failing allocation + bug #2 (SIGFPE) — with the shim:
LD_PRELOAD=./nyx_headless_shim.so python3 examples/01_hello_nyx.py   # logs size=0 alloc, then SIGFPE
```

## Attachments

- `nyx_headless_shim.c` — diagnostic/work-around LD_PRELOAD shim.
- `nyx_probe_counts.py` — scene-variant probe (plane / plane+sphere / +envmap).
- gdb backtraces: SIGSEGV (`nyx_upgrade/bt4.txt`, `nyx_upgrade/gdb_bt.txt`),
  SIGFPE (`nyx_upgrade/fpe.txt`).
- `NYX_BUG_REPORT_v1_egl_superseded.md` — the earlier (incorrect) EGL-focused
  report, kept to document why the EGL theory was ruled out.
