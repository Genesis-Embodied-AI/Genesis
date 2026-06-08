# Find which scene element count triggers the size=0 alloc / div-by-zero.
# We vary one factor at a time and report how far Nyx build() gets.
import os, sys
import genesis as gs
import gs_nyx.nyx_py_renderer as npr
import gs_nyx.nyx_py_sdk as nps
from gs_nyx_plugin.nyx_camera_options import NyxCameraOptions

variant = sys.argv[1] if len(sys.argv) > 1 else "plane_only"
gs.init()
scene = gs.Scene(sim_options=gs.options.SimOptions(dt=0.01), show_viewer=False)

if variant == "plane_only":
    scene.add_entity(morph=gs.morphs.Plane(plane_size=(10.0,10.0)))
elif variant == "plane_sphere":
    scene.add_entity(morph=gs.morphs.Plane(plane_size=(10.0,10.0)))
    scene.add_entity(morph=gs.morphs.Sphere(radius=0.3, pos=(0,0,0.5)))
elif variant == "with_envmap":
    scene.add_entity(morph=gs.morphs.Plane(plane_size=(10.0,10.0)))
    scene.add_entity(morph=gs.morphs.Sphere(radius=0.3, pos=(0,0,0.5)))
    em = nps.EnvironmentMapAsset()
    em.texture = "/workspace/Uni-Genesis/genesis-nyx/examples/assets/kloppenheim_07_puresky_4k.hdr"
    em.layout = nps.EEnvMapLayout.LongLat; em.multiplier = 8
    cam = scene.add_sensor(NyxCameraOptions(res=(320,240), pos=(-1,1,1.2), lookat=(0,0,0.1),
        fov=20.0, spp=4, render_mode=npr.ERenderMode.FastPathTracer, env_maps=(em,)))
    scene.build(n_envs=1); scene.step()
    img = cam.read().rgb[0].cpu().numpy()
    print(f">>> {variant}: RENDER OK shape={img.shape} mean={float(img.mean()):.1f}")
    sys.exit(0)

cam = scene.add_sensor(NyxCameraOptions(res=(320,240), pos=(-1,1,1.2), lookat=(0,0,0.1),
    fov=20.0, spp=4, render_mode=npr.ERenderMode.FastPathTracer))
scene.build(n_envs=1); scene.step()
img = cam.read().rgb[0].cpu().numpy()
print(f">>> {variant}: RENDER OK shape={img.shape} mean={float(img.mean()):.1f}")
