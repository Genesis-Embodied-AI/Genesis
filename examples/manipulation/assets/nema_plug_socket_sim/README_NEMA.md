# NEMA Plug/Socket Sim Assets (generated) — for Genesis

Watertight, verified plug + receptacle pairs modeled on **NEMA WD-6** connector
dimensions — the same connector family as the black plugs/receptacles in the
IndustReal breadboard setup (which were store-bought and never released as CAD).

- `nema_1_15` — 2-prong US plug (two flat blades)
- `nema_5_15` — 3-prong US plug (blades + round ground pin, ground-first engagement)

## Why generated instead of downloaded

Verified 2026-07-31: no public repo ships NEMA connector *simulation* geometry.
IndustRealKit contains only the white holder trays (real connectors were purchased);
Factory/TacSL branches contain connector URDFs whose meshes were never committed;
Isaac Lab has none. These assets fill that gap with clean, parametric geometry.

## Dimensions used (NEMA WD-6 nominal)

blade 6.35 × 1.524 mm, exposed length 16 mm, spacing 12.7 mm center-to-center;
ground pin ⌀4.75 mm, offset 11.91 mm from the blade line, 1.5 mm longer than blades.
Simplifications: rectangular slots (real 5-15R ground hole is U-shaped; some real
1-15 plugs have a wider polarized neutral blade), solid molded body, no internal
contacts. Geometry is internally consistent — the plug mates its socket exactly.

## Simulation-oriented features

- **Clearance is a parameter**: `CLEARANCE = 0.25` mm per side (edit + rerun
  `generate_nema_plug_socket.py`). 0.4–0.5 mm ≈ easy/curriculum start,
  0.25 mm ≈ default, 0.1 mm ≈ near-realistic and hard.
- **Lead-in chamfers** on every socket opening (1.2 mm funnel) and **tapered
  blade/pin tips** — like real connectors; without these, RL exploration on
  flat-faced slots is brutal.
- Meshes uniformly subdivided (~5–15k tris) for dense SDF contact sampling.
- **Shared assembled frame** (AutoMate convention): spawn plug and socket at the
  same pose = fully inserted; socket base at z=0; units meters.
  `mesh/<name>/meta.json` holds `disassembly_dist_m` for building the start state.

## Verification performed at generation time (asserts in the script)

1. Both meshes watertight (before and after subdivision).
2. Assembled state: boolean intersection volume ≈ 0 (no interpenetration).
3. Straight-line insertion sweep (9 steps over full lift): zero collision volume.
4. Sampled min gap between inserted plug surface and socket solid = 0.250 mm,
   exactly the commanded clearance.

## Files

```
mesh/nema_{1_15,5_15}/asset_plug.obj / asset_socket.obj / meta.json
urdf/nema_*_{plug,socket}.urdf         minimal URDFs (AutoMate style)
generate_nema_plug_socket.py           parametric generator (trimesh + manifold3d)
load_nema_genesis.py                   Genesis starter (socket fixed, convexify=False!)
```

## Genesis notes

Same rules as the AutoMate bundle: load the **socket with `convexify=False`**
(convex hull seals the slots), keep it `fixed=True`, small dt + substeps. Start
with `MODE="assembled"` to sanity-check contact stability, then `MODE="insert"`.

## Toward more realism

For photoreal or manufacturer-exact connectors: pull STEP models from GrabCAD /
manufacturer sites (Leviton, TE, Molex publish free CAD), then make the pair
simulation-compatible with **MatchMaker**'s clearance-erosion stage
(https://github.com/wangyian-me/MatchMaker_Code/tree/match_maker) — raw CAD pairs
have zero/negative clearance and will explode in any rigid-body simulator.
The generator here is deliberately parametric so you can also just edit it.

License: generated from scratch for this project — use freely (public domain / CC0).
