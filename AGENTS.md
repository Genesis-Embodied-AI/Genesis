# Genesis Agent Guidelines

`CODING_GUIDELINES.md` is the authority on how code is written, tested, and reviewed here, and it is imported below so it applies in full: naming, formatting, comments, docstrings, typing, data access, kernels, API design, error handling, testing, and git and pull-request conventions. The reference docs under `.github/contributing/` cover the rest - ARCHITECTURE, TESTING, CODING_CONVENTIONS, EXAMPLES, PULL_REQUESTS, USD_PARSER. Ask when two of them conflict.

@CODING_GUIDELINES.md

## Running the test suite

- **Never filter or truncate test output inline** (no `pytest ... | tail`, no `| grep`). Redirect the full output to a log file and extract from the file afterwards: a filter between pytest and disk destroys the only copy of the failure names and masks the exit code.
- Example tests need `-m examples` to override the default marker filter in `pyproject.toml`.
- Lint and format with ruff (check and format, line length 120) through pre-commit; install the hooks once with `pre-commit install` and they run on every commit.

## Reproducing Apple software renderer failures locally

All GitHub Apple Silicon macOS runners are VMs whose virtualized GPU has no OpenGL, so rendering falls back to the Apple Software Renderer. That renderer is half-broken; macOS-only rendering failures come from tests tripping one of its failure modes, so be careful about which rendering features unit tests rely on. To force it locally on any Mac, hijack the two pixel-format attributes that pyglet unconditionally appends, before any GL context is created:

```python
# force_sw.py - import before any GL context is created, e.g. 'pytest -p force_sw' with PYTHONPATH set
from pyglet.libs.darwin import cocoapy

cocoapy.NSOpenGLPFAAllRenderers = 70  # NSOpenGLPFARendererID
cocoapy.NSOpenGLPFAMaximumPolicy = 0x00020400  # kCGLRendererGenericFloatID
```

- Run with the same flags as macOS CI: `PYTHONPATH=<dir-with-force_sw.py> GS_TORCH_FORCE_CPU_DEVICE=1 pytest -p force_sw --dev --logical --backend cpu --forked <tests>` (CI additionally selects `-m 'required and not slow'`).
- Verify it took effect: Genesis logs "Software rendering context detected" and `scene.visualizer.is_software` is True.
- Known failure mode: any geometry with vertices outside the camera frustum is misrasterized, breaking pixel comparisons. In practice this bites through ground planes, whose default `plane_size` is effectively infinite (1 km x 1 km): give them a finite size and a position that puts every vertex inside the view.
- Known failure mode: shadow mapping is forcibly disabled on software rendering backends for performance reasons, so snapshot scenes must disable shadows explicitly (`shadow=False` for the rasterizer); a snapshot generated on hardware GL with shadows enabled can never match.
