# Underwater examples

Opt-in Akinci boundary particles (`sph_akinci_boundary=True`). Stock flotation is #2857; see `../../UNDERWATER.md`.

```bash
python examples/underwater/buoyancy_demo.py
python examples/underwater/buoyancy_demo.py --compare
python examples/underwater/physics_validation.py A3
python examples/underwater/theory_xverify.py H
```

| Script | Purpose |
|--------|---------|
| `buoyancy_demo.py` | Light floats / heavy sinks; optional on vs off |
| `physics_validation.py` | Closed-form / Archimedes probes |
| `theory_xverify.py` | Depth-invariance and related checks |

Use `particle_size ≈ body_size / 7` and `dt ≤ 5e-4` for stable WCSPH.
