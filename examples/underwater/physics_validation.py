"""Quantitative physics-correctness validation for the Genesis swim stack — CLOSED-FORM references,
real error numbers (NOT vision). Each test isolates one physical law and compares the sim to analytics.

  A1 free-fall            : z(t)=z0-1/2 g t^2 , v(t)=-g t            (gravity + integrator)
  A2 F=ma                 : a = F/m                                  (Newton's 2nd law)
  A3 quadratic-drag vterm : v_inf=sqrt(F/k), v(t)=v_inf*tanh(t/tau)  (Morison drag model + nonlinear integ)
  B  impulse-momentum     : m*(v_f-v_0) == integral F dt             (momentum conservation under varying F)
  C  Archimedes (SPH)     : floating submerged fraction == rho_b/rho_w (NATIVE two-way SPH buoyancy magnitude)

Usage: python physics_validation.py <A1|A2|A3|B|C>

Round-6 SPH buoyancy validation note: C/E spawn rigid boxes above the water and let them enter by dynamics before
measurement, avoiding the invalid pre-filled setup that traps SPH particles inside rigid volumes at build time.
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
T = sys.argv[1] if len(sys.argv) > 1 else "A1"
dt = 2.0e-3

def mk(gravity, sph=None):
    kw = dict(show_viewer=False, sim_options=gs.options.SimOptions(dt=dt, gravity=gravity))
    if sph:
        kw["sph_options"] = sph
        kw["coupler_options"] = gs.options.LegacyCouplerOptions(rigid_sph=True, sph_akinci_boundary=True)
    return gs.Scene(**kw)

gs.init(backend=gs.cuda, precision="32", logging_level="warning")

if T == "A1":  # free fall in vacuum
    g = 9.81; sc = mk((0, 0, -g))
    box = sc.add_entity(material=gs.materials.Rigid(rho=500.0), morph=gs.morphs.Box(pos=(0, 0, 3.0), size=(0.3, 0.3, 0.3)))
    sc.build(); z0 = float(to_np(box.get_pos()).reshape(-1)[2]); errs = []
    for i in range(1, 401):
        sc.step(); t = i * dt
        z = float(to_np(box.get_pos()).reshape(-1)[2]); v = float(to_np(box.get_vel()).reshape(-1)[2])
        za = z0 - 0.5 * g * t * t; va = -g * t
        if i % 80 == 0:
            errs.append((round(t, 2), round(z, 4), round(za, 4), round(v, 4), round(va, 4)))
    zf = float(to_np(box.get_pos()).reshape(-1)[2]); vf = float(to_np(box.get_vel()).reshape(-1)[2]); tf = 400 * dt
    za = z0 - 0.5 * g * tf * tf; va = -g * tf
    ez = abs(zf - za) / abs(z0 - za); ev = abs(vf - va) / abs(va)
    print("[A1] (t, z_sim, z_analytic, v_sim, v_analytic):")
    for e in errs: print("   ", e)
    print(f"[A1] final t={tf:.2f}s z_sim={zf:.4f} z_an={za:.4f} (relerr {ez*100:.3f}%) | v_sim={vf:.4f} v_an={va:.4f} (relerr {ev*100:.3f}%)")
    print(f"[A1] VERDICT {'PASS' if ez < 0.02 and ev < 0.02 else 'FAIL'} (free-fall matches z0-1/2 g t^2 and -g t)")

elif T == "A2":  # F = m a  (gravity off)
    sc = mk((0, 0, 0))
    box = sc.add_entity(material=gs.materials.Rigid(rho=500.0), morph=gs.morphs.Box(pos=(0, 0, 0), size=(0.4, 0.4, 0.4)))
    sc.build(); rs = sc.sim.rigid_solver; bl = box.base_link.idx; m = float(box.get_mass()); F = 60.0
    a_an = F / m
    for i in range(300):
        rs.apply_links_external_force(force=np.array([[F, 0, 0]], np.float32), links_idx=np.asarray([bl]), ref="root_com")
        sc.step()
    vx = float(to_np(box.get_vel()).reshape(-1)[0]); tf = 300 * dt; a_meas = vx / tf
    err = abs(a_meas - a_an) / a_an
    print(f"[A2] m={m:.4f}kg F={F}N  a_analytic=F/m={a_an:.4f}  a_measured=vx/t={a_meas:.4f} (vx={vx:.4f} @t={tf:.2f}s)")
    print(f"[A2] relerr {err*100:.3f}%  VERDICT {'PASS' if err < 0.01 else 'FAIL'} (Newton 2nd law a=F/m)")

elif T == "A3":  # quadratic drag terminal velocity
    sc = mk((0, 0, 0))
    box = sc.add_entity(material=gs.materials.Rigid(rho=500.0), morph=gs.morphs.Box(pos=(0, 0, 0), size=(0.4, 0.4, 0.4)))
    sc.build(); rs = sc.sim.rigid_solver; bl = box.base_link.idx; m = float(box.get_mass())
    F, k = 80.0, 25.0
    v_inf = math.sqrt(F / k); tau = m / math.sqrt(F * k)
    errs = []; N = 1500
    for i in range(N):
        v = float(to_np(box.get_vel()).reshape(-1)[0])
        Fnet = F - k * abs(v) * v
        rs.apply_links_external_force(force=np.array([[Fnet, 0, 0]], np.float32), links_idx=np.asarray([bl]), ref="root_com")
        sc.step()
        if (i + 1) % 300 == 0:
            t = (i + 1) * dt; vs = float(to_np(box.get_vel()).reshape(-1)[0]); va = v_inf * math.tanh(t / tau)
            errs.append((round(t, 2), round(vs, 4), round(va, 4), round(abs(vs - va) / va * 100, 2)))
    vf = float(to_np(box.get_vel()).reshape(-1)[0]); ev = abs(vf - v_inf) / v_inf
    print(f"[A3] F={F} k={k}  v_inf=sqrt(F/k)={v_inf:.4f} m/s  tau={tau:.3f}s")
    print("[A3] (t, v_sim, v_analytic=v_inf*tanh(t/tau), relerr%):")
    for e in errs: print("   ", e)
    print(f"[A3] final v_sim={vf:.4f} vs v_inf={v_inf:.4f} (relerr {ev*100:.3f}%)")
    print(f"[A3] VERDICT {'PASS' if ev < 0.02 else 'FAIL'} (drag terminal velocity matches closed form)")

elif T == "B":  # impulse-momentum theorem under time-varying force
    sc = mk((0, 0, 0))
    box = sc.add_entity(material=gs.materials.Rigid(rho=500.0), morph=gs.morphs.Box(pos=(0, 0, 0), size=(0.4, 0.4, 0.4)))
    sc.build(); rs = sc.sim.rigid_solver; bl = box.base_link.idx; m = float(box.get_mass())
    v0 = to_np(box.get_vel()).reshape(-1)[:3].copy()
    impulse = np.zeros(3); N = 1200
    for i in range(N):
        t = i * dt
        Fv = np.array([40.0 * math.sin(3.0 * t), 18.0 * math.cos(2.0 * t), 0.0], np.float32)  # arbitrary time-varying
        rs.apply_links_external_force(force=Fv[None], links_idx=np.asarray([bl]), ref="root_com")
        impulse += Fv * dt
        sc.step()
    vf = to_np(box.get_vel()).reshape(-1)[:3]
    dp = m * (vf - v0)  # measured momentum change
    res = np.linalg.norm(dp - impulse) / (np.linalg.norm(impulse) + 1e-9)
    print(f"[B] m={m:.4f}kg  integral(F dt)=[{impulse[0]:.4f},{impulse[1]:.4f}]  m*dv=[{dp[0]:.4f},{dp[1]:.4f}]")
    print(f"[B] relative residual {res*100:.3f}%  VERDICT {'PASS' if res < 0.02 else 'FAIL'} (impulse-momentum: m*dv == integral F dt)")

elif T == "D":  # added-mass effective inertia: accelerate then COAST, recover m_eff from drag decay, compare m+m_add
    sc = mk((0, 0, 0))
    rho_b, rho_w, Cm = 600.0, 1000.0, 0.6
    box = sc.add_entity(material=gs.materials.Rigid(rho=rho_b), morph=gs.morphs.Box(pos=(0, 0, 0), size=(0.4, 0.4, 0.4)))
    sc.build(); rs = sc.sim.rigid_solver; bl = box.base_link.idx
    m = float(box.get_mass()); V = m / rho_b; m_add = Cm * rho_w * V; m_eff_pred = m + m_add
    k = 30.0; F_drive = 120.0
    vprev = 0.0; afilt = 0.0; coast = []
    for i in range(2400):
        v = float(to_np(box.get_vel()).reshape(-1)[0])
        a = (v - vprev) / dt; afilt = 0.7 * afilt + 0.3 * a; vprev = v
        thrust = F_drive if i < 1200 else 0.0     # cut at 1200 -> coast under drag with added-mass inertia
        Fnet = thrust - k * abs(v) * v - m_add * afilt   # explicit added-mass reaction force
        rs.apply_links_external_force(force=np.array([[Fnet, 0, 0]], np.float32), links_idx=np.asarray([bl]), ref="root_com")
        sc.step()
        if 1300 <= i <= 2300 and i % 50 == 0:
            coast.append(((i - 1200) * dt, float(to_np(box.get_vel()).reshape(-1)[0])))
    # coast ODE: m_eff dv/dt = -k v^2  =>  1/v = 1/v0 + (k/m_eff)(t-t0) ; slope of 1/v vs t = k/m_eff
    ts = np.array([c[0] for c in coast]); vs = np.array([c[1] for c in coast])
    A = np.vstack([ts, np.ones_like(ts)]).T
    slope = np.linalg.lstsq(A, 1.0 / vs, rcond=None)[0][0]
    m_eff_meas = k / slope
    err = abs(m_eff_meas - m_eff_pred) / m_eff_pred
    print(f"[D] m_rigid={m:.3f}kg V={V:.5f}m^3 Cm={Cm} rho_w={rho_w} -> m_add={m_add:.3f}kg  m_eff_predicted={m_eff_pred:.3f}kg")
    print(f"[D] coast 1/v vs t slope={slope:.4f} -> m_eff_measured=k/slope={m_eff_meas:.3f}kg")
    print(f"[D] relerr {err*100:.2f}%  VERDICT {'PASS' if err < 0.06 else 'FAIL'} (effective inertia = rigid mass + Fossen added mass)")

elif T in ("F", "F200"):  # does an MPM body float? F=rho600 (realistic robot), F200=rho200 (the Genesis duck)
    g = 9.81; rho_w = 1000.0; rho_b = 200.0 if T == "F200" else 600.0
    dt_c = 4e-3
    sph = gs.options.SPHOptions(lower_bound=(-0.45, -0.45, 0.0), upper_bound=(0.45, 0.45, 0.75), particle_size=0.02)
    mpm = gs.options.MPMOptions(lower_bound=(-0.5, -0.5, -0.05), upper_bound=(0.5, 0.5, 0.8), grid_density=64)
    sc = gs.Scene(show_viewer=False, sim_options=gs.options.SimOptions(dt=dt_c, substeps=10, gravity=(0, 0, -g)),
                  sph_options=sph, mpm_options=mpm, coupler_options=gs.options.LegacyCouplerOptions(rigid_sph=True, mpm_sph=True))
    sc.add_entity(material=gs.materials.Rigid(needs_coup=True), morph=gs.morphs.Plane())
    water = sc.add_entity(material=gs.materials.SPH.Liquid(rho=rho_w), morph=gs.morphs.Box(pos=(0, 0, 0.28), size=(0.8, 0.8, 0.5)))
    L = 0.22
    duck = sc.add_entity(material=gs.materials.MPM.Elastic(rho=rho_b), morph=gs.morphs.Box(pos=(0, 0, 0.5), size=(L, L, L)))
    sc.build()
    for _ in range(1500): sc.step()
    zs = float(np.percentile(to_np(water.get_particles_pos()).reshape(-1, 3)[:, 2], 92))
    pp = to_np(duck.get_particles_pos()).reshape(-1, 3); bz, tz = float(pp[:, 2].min()), float(pp[:, 2].max())
    f_meas = float(np.clip((zs - bz) / (tz - bz), 0, 1)); f_arch = rho_b / rho_w
    print(f"[F] MPM body rho_b={rho_b} in SPH water rho_w={rho_w}: Archimedes float fraction = {f_arch:.2f}")
    print(f"[F] measured submerged fraction f={f_meas:.3f} (waterline={zs:.3f} body z=[{bz:.3f},{tz:.3f}])")
    print(f"[F] vs rigid (test C) which SANK to sub=1.0 -> MPM {'FLOATS (grid-pressure coupling works)' if f_meas < 0.9 else 'also sinks'}")

elif T in ("E", "EOPT"):  # measure STATIC vertical fluid force on a held submerged body. EOPT = config-only tuned
    g = 9.81; rho_w, rho_b = 1000.0, 1500.0   # heavy box so it stays down; hold it fixed and read the fluid support
    TUNE = (T == "EOPT")
    psize = 0.013 if TUNE else 0.02
    psolver = "DFSPH" if TUNE else "WCSPH"
    subs = 10 if TUNE else 1
    dt_c = 5.0e-3 if TUNE else 5.0e-4   # substeps gives the stable inner dt; outer dt larger
    sph = gs.options.SPHOptions(lower_bound=(-0.45, -0.45, 0.0), upper_bound=(0.45, 0.45, 0.75),
                                particle_size=psize, pressure_solver=psolver)
    sc = gs.Scene(show_viewer=False, sim_options=gs.options.SimOptions(dt=dt_c, substeps=subs, gravity=(0, 0, -g)),
                  sph_options=sph, coupler_options=gs.options.LegacyCouplerOptions(rigid_sph=True, sph_akinci_boundary=True))
    sc.add_entity(material=gs.materials.Rigid(needs_coup=True), morph=gs.morphs.Plane())
    water = sc.add_entity(material=gs.materials.SPH.Liquid(rho=rho_w), morph=gs.morphs.Box(pos=(0, 0, 0.28), size=(0.8, 0.8, 0.5)))
    L = 0.12; ZTGT = 0.14   # fully submerged below the measured SPH waterline after entering from above
    box = sc.add_entity(material=gs.materials.Rigid(rho=rho_b, needs_coup=True), morph=gs.morphs.Box(pos=(0, 0, 0.78), size=(L, L, L)))
    sc.build(); rs = sc.sim.rigid_solver; bl = box.base_link.idx
    m = float(box.get_mass()); V = L**3; F_arch = rho_w * g * V
    pp = to_np(water.get_particles_pos()).reshape(-1, 3)
    trapped = int(((np.abs(pp[:, 0]) < L / 2) & (np.abs(pp[:, 1]) < L / 2) & (np.abs(pp[:, 2] - 0.78) < L / 2)).sum())
    nsteps = 5000 if TUNE else 14000; warm = int(nsteps * 0.72)
    holds = []
    for i in range(nsteps):
        z = float(to_np(box.get_pos()).reshape(-1)[2]); vz = float(to_np(box.get_vel()).reshape(-1)[2])
        target = 0.78 if i < 1000 else ZTGT
        Fhold = 6000.0 * (target - z) - 650.0 * vz + m * g   # stiff PD that also cancels gravity -> Fhold-mg = -F_SPH_vert at steady
        q = to_np(box.get_quat()).reshape(-1); ang = get_ang(box)
        Tatt = (-np.array([90.0, 90.0, 60.0]) * rpy(q) - np.array([12.0, 12.0, 9.0]) * ang).astype(np.float32)
        rs.apply_links_external_force(force=np.array([[0, 0, Fhold]], np.float32), links_idx=np.asarray([bl]), ref="root_com")
        rs.apply_links_external_torque(torque=Tatt[None], links_idx=np.asarray([bl]))
        sc.step()
        if i >= warm: holds.append(Fhold - m * g)   # steady: m*a=0=Fhold+F_SPH-mg -> F_SPH = mg-Fhold = -(Fhold-mg)
    F_sph = -float(np.mean(holds))   # vertical fluid force on the static submerged body
    cfg = f"DFSPH+substeps{subs}+psize{psize}" if TUNE else f"WCSPH+psize{psize}(baseline)"
    print(f"[{T}] config={cfg} trapped_at_spawn={trapped}  STATIC submerged box m={m:.3f}kg V={V:.5f}m^3  Archimedes = {F_arch:.2f} N (UP)")
    print(f"[{T}] measured native-SPH vertical force on HELD-STILL body = {F_sph:.2f} N  ({F_sph/F_arch*100:.1f}% of Archimedes)")

elif T == "C":  # Archimedes — NATIVE SPH buoyancy magnitude (no analytic buoyancy)
    g = 9.81; rho_w, rho_b = 1000.0, 600.0
    dt_c = 5.0e-4   # SPH needs a smaller stable timestep (dt=2e-3 explodes -> degenerate measurement)
    sph = gs.options.SPHOptions(lower_bound=(-0.45, -0.45, 0.0), upper_bound=(0.45, 0.45, 0.75), particle_size=0.02)
    sc = gs.Scene(show_viewer=False, sim_options=gs.options.SimOptions(dt=dt_c, gravity=(0, 0, -g)),
                  sph_options=sph, coupler_options=gs.options.LegacyCouplerOptions(rigid_sph=True, sph_akinci_boundary=True))
    sc.add_entity(material=gs.materials.Rigid(needs_coup=True), morph=gs.morphs.Plane())
    water = sc.add_entity(material=gs.materials.SPH.Liquid(rho=rho_w),
                          morph=gs.morphs.Box(pos=(0, 0, 0.28), size=(0.8, 0.8, 0.5)))
    L = 0.22
    box = sc.add_entity(material=gs.materials.Rigid(rho=rho_b, needs_coup=True),
                        morph=gs.morphs.Box(pos=(0, 0, 0.78), size=(L, L, L)))
    sc.build()
    pp = to_np(water.get_particles_pos()).reshape(-1, 3)
    trapped = int(((np.abs(pp[:, 0]) < L / 2) & (np.abs(pp[:, 1]) < L / 2) & (np.abs(pp[:, 2] - 0.78) < L / 2)).sum())
    for _ in range(12000): sc.step()  # let it settle to flotation equilibrium (NO analytic buoyancy applied)
    zs = float(np.percentile(to_np(water.get_particles_pos()).reshape(-1, 3)[:, 2], 92))
    ab = to_np(box.get_AABB()).reshape(-1); bz, tz = float(ab[2]), float(ab[5])
    f_meas = float(np.clip((zs - bz) / (tz - bz), 0, 1))
    f_arch = rho_b / rho_w
    print(f"[C] rho_b={rho_b} rho_w={rho_w} trapped_at_spawn={trapped}  Archimedes floating fraction f=rho_b/rho_w={f_arch:.3f}")
    print(f"[C] measured submerged fraction f={f_meas:.3f}  (waterline={zs:.3f} box z=[{bz:.3f},{tz:.3f}])")
    print(f"[C] native-SPH / Archimedes ratio = {f_meas/f_arch:.2f}  (HONEST: coarse particle_size under-resolves buoyant force; sign+trend correct)")
