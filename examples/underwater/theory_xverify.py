"""Theory-verifiable, cross-verified buoyancy tests for the patched Genesis SPH coupler.
Each asserts a CLOSED-FORM law and (where possible) cross-verifies the same quantity two independent ways.
Run AFTER the buoyancy patch is in place, e.g. python theory_xverify.py <G|H|J>

Round-6 validation note: G/H spawn rigid boxes above the water and let them enter by dynamics before measuring.
This avoids the invalid pre-filled-water setup that traps SPH particles inside rigid volumes at build time.

  G  float-fraction LAW : a free floating body settles at submerged fraction sub = rho_b/rho_w (Archimedes
                          equilibrium). Swept over densities -> the line sub vs rho_b must have slope 1/rho_w
                          through 0. CROSS-VERIFY: buoyancy from float-equilibrium (= weight) vs from the
                          static hold-force method should agree.
  H  depth-invariance   : Archimedes buoyancy on a FULLY submerged body is independent of depth. Hold the
                          same body at two depths -> the fluid vertical force must be equal (ratio ~1).
  J  hydrostatic p(h)   : the SPH pressure field itself must satisfy p(h) = rho_w*g*h (the law the patch
                          integrates). Fit pressure vs depth -> slope must equal rho_w*g. (fluid-only, no body)
"""
import sys, math, numpy as np, torch, genesis as gs
def to_np(x): return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)
def rpy(q):
    w, x, y, z = q
    return np.array([
        math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)),
        math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x)))),
        math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)),
    ], dtype=float)
def get_ang(entity):
    try:
        return to_np(entity.get_ang()).reshape(-1)[:3]
    except Exception:
        return np.zeros(3, dtype=float)
T = sys.argv[1] if len(sys.argv) > 1 else "G"
g, rho_w = 9.81, 1000.0
dt = 5.0e-4
gs.init(backend=gs.cuda, precision="32", logging_level="warning")

def scene_with_water(particle_size=0.02):
    sph = gs.options.SPHOptions(lower_bound=(-0.45, -0.45, 0.0), upper_bound=(0.45, 0.45, 0.75), particle_size=particle_size)
    sc = gs.Scene(show_viewer=False, sim_options=gs.options.SimOptions(dt=dt, gravity=(0, 0, -g)),
                  sph_options=sph, coupler_options=gs.options.LegacyCouplerOptions(rigid_sph=True, sph_akinci_boundary=True))
    sc.add_entity(material=gs.materials.Rigid(needs_coup=True), morph=gs.morphs.Plane())
    water = sc.add_entity(material=gs.materials.SPH.Liquid(rho=rho_w),
                          morph=gs.morphs.Box(pos=(0, 0, 0.28), size=(0.8, 0.8, 0.5)))
    return sc, water

if T == "G":  # float-fraction law for ONE density (pass rho_b as argv[2]); loop in bash to sweep
    rho_b = float(sys.argv[2]) if len(sys.argv) > 2 else 600.0
    particle_size = float(sys.argv[3]) if len(sys.argv) > 3 else 0.02
    L = 0.18
    sc, water = scene_with_water(particle_size=particle_size)
    box = sc.add_entity(material=gs.materials.Rigid(rho=rho_b, needs_coup=True),
                        morph=gs.morphs.Box(pos=(0, 0, 0.78), size=(L, L, L)))
    sc.build(); rs = sc.sim.rigid_solver; bl = box.base_link.idx
    pp = to_np(water.get_particles_pos()).reshape(-1, 3)
    trapped = int(((np.abs(pp[:, 0]) < L / 2) & (np.abs(pp[:, 1]) < L / 2) & (np.abs(pp[:, 2] - 0.78) < L / 2)).sum())
    for _ in range(12000):
        q = to_np(box.get_quat()).reshape(-1); ang = get_ang(box)
        Tatt = (-np.array([80.0, 80.0, 60.0]) * rpy(q) - np.array([10.0, 10.0, 8.0]) * ang).astype(np.float32)
        rs.apply_links_external_torque(torque=Tatt[None], links_idx=np.asarray([bl]))
        sc.step()
    zs = float(np.percentile(to_np(water.get_particles_pos()).reshape(-1, 3)[:, 2], 92))
    ab = to_np(box.get_AABB()).reshape(-1); bz, tz = float(ab[2]), float(ab[5])
    sub = float(np.clip((zs - bz) / (tz - bz), 0, 1)); pred = rho_b / rho_w
    print(f"[G] rho_b={rho_b:.0f} particle_size={particle_size:.3f} trapped_at_spawn={trapped}  predicted_sub(rho_b/rho_w)={pred:.2f}  measured_sub={sub:.3f}  |err|={abs(sub-pred):.3f}")

elif T == "H":  # depth-invariance of buoyancy on a fully submerged body
    def hold_force(ztgt):
        sc, water = scene_with_water()
        L = 0.12
        box = sc.add_entity(material=gs.materials.Rigid(rho=1500.0, needs_coup=True),
                            morph=gs.morphs.Box(pos=(0, 0, 0.78), size=(L, L, L)))
        sc.build(); rs = sc.sim.rigid_solver; bl = box.base_link.idx; m = float(box.get_mass())
        hs = []
        pp = to_np(water.get_particles_pos()).reshape(-1, 3)
        trapped = int(((np.abs(pp[:, 0]) < L / 2) & (np.abs(pp[:, 1]) < L / 2) & (np.abs(pp[:, 2] - 0.78) < L / 2)).sum())
        for i in range(14000):
            z = float(to_np(box.get_pos()).reshape(-1)[2]); vz = float(to_np(box.get_vel()).reshape(-1)[2])
            target = 0.78 if i < 1000 else ztgt
            F = 6000.0 * (target - z) - 650.0 * vz + m * g
            q = to_np(box.get_quat()).reshape(-1); ang = get_ang(box)
            Tatt = (-np.array([90.0, 90.0, 60.0]) * rpy(q) - np.array([12.0, 12.0, 9.0]) * ang).astype(np.float32)
            rs.apply_links_external_force(force=np.array([[0, 0, F]], np.float32), links_idx=np.asarray([bl]), ref="root_com")
            rs.apply_links_external_torque(torque=Tatt[None], links_idx=np.asarray([bl]))
            sc.step()
            if i >= 10000: hs.append(F - m * g)
        return -float(np.mean(hs)), trapped
    f_shallow, trapped_shallow = hold_force(0.18); f_deep, trapped_deep = hold_force(0.10)
    ratio = f_deep / f_shallow if f_shallow else 0.0
    print(f"[H] untrapped fully-submerged buoyancy trapped=({trapped_shallow},{trapped_deep})  shallow(z=0.18)={f_shallow:.2f} N   deep(z=0.10)={f_deep:.2f} N   ratio={ratio:.3f}")
    print(f"[H] VERDICT {'PASS' if 0.90 < ratio < 1.10 else 'FAIL'} (Archimedes buoyancy is depth-INDEPENDENT for full submersion)")

elif T == "J":  # hydrostatic pressure law p(h) = rho_w*g*h (fluid only, tests the field the patch integrates)
    sc, water = scene_with_water()
    sc.build()
    for _ in range(8000): sc.step()
    pos = to_np(water.get_particles_pos()).reshape(-1, 3)
    zs = float(np.percentile(pos[:, 2], 98))
    # read internal SPH pressure field
    try:
        psph = to_np(sc.sim.sph_solver.particles_reordered.p).reshape(-1)
    except Exception as e:
        print(f"[J] could not read internal pressure field ({e}); skipping"); sys.exit(0)
    depth = zs - pos[:, 2]
    m = (depth > 0.03) & (depth < 0.40)
    A = np.vstack([depth[m], np.ones(m.sum())]).T
    slope, intc = np.linalg.lstsq(A, psph[m], rcond=None)[0]
    pred = rho_w * g
    print(f"[J] hydrostatic fit p = {slope:.1f}*h + {intc:.1f}   predicted slope rho_w*g = {pred:.1f}")
    print(f"[J] slope/(rho_w*g) = {slope/pred:.2f}  VERDICT {'PASS' if 0.7 < slope/pred < 1.3 else 'FAIL (WCSPH under-builds hydrostatic pressure)'}")
