python3 -m pip install --user --no-deps --no-build-isolation -e /workspace/Uni-Genesis/genesis >/dev/null 2>&1
python3 -m pip install --no-deps --force-reinstall \
  /workspace/Uni-Genesis/docker-genesis/nyx_upgrade/gs_nyx-0.1.2-*.whl \
  /workspace/Uni-Genesis/docker-genesis/nyx_upgrade/gs_nyx_plugin-0.1.3-*.whl >/dev/null 2>&1
gcc -shared -fPIC -O2 /workspace/Uni-Genesis/docker-genesis/nyx_headless_shim.c -o /tmp/nyx_headless_shim.so -ldl
cat > /tmp/repro.py <<'PY'
import genesis as gs
import gs_nyx.nyx_py_renderer as npr
from gs_nyx_plugin.nyx_camera_options import NyxCameraOptions
gs.init()
scene = gs.Scene(sim_options=gs.options.SimOptions(dt=0.01), show_viewer=False)
scene.add_entity(morph=gs.morphs.Plane(plane_size=(10.0,10.0)))
cam = scene.add_sensor(NyxCameraOptions(res=(320,240), pos=(-1,1,1.2), lookat=(0,0,0.1),
    fov=20.0, spp=4, render_mode=npr.ERenderMode.FastPathTracer))
scene.build(n_envs=1)
scene.step()
img = cam.read().rgb[0].cpu().numpy()
print(">>> RENDER OK shape", img.shape, "mean", float(img.mean()), flush=True)
PY

run() {
  local name="$1"; shift
  env "$@" NYX_HEADLESS_SHIM_QUIET=1 LD_PRELOAD=/tmp/nyx_headless_shim.so \
      python3 /tmp/repro.py > /tmp/run.log 2>&1
  local rc=$?
  local verdict="FAIL (rc=$rc)"
  grep -q "RENDER OK" /tmp/run.log && verdict="RENDER OK"
  [ $rc -eq 139 ] && verdict="SIGSEGV (rc=139)"
  printf '%-30s -> %s\n' "$name" "$verdict"
}

cd /workspace/Uni-Genesis/genesis-nyx
run "baseline (shim only)"
run "surfaceless EGL"          EGL_PLATFORM=surfaceless
run "GLX vendor nvidia"        __GLX_VENDOR_LIBRARY_NAME=nvidia
run "egl vendor pinned"        __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
run "cuda dev 0 only"          CUDA_VISIBLE_DEVICES=0
