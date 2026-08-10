"""Isolated CoACD entrypoint.

Must not import ``torch`` or ``genesis``: ROCm/HIP torch loads system ``libgomp`` into the
parent process while the CoACD wheel vendors its own ``libgomp``. Two OpenMP runtimes in one
process SIGSEGV inside CoACD's parallel regions (see SarahWeiii/CoACD#104). This module is
executed in a fresh interpreter via ``subprocess`` so only CoACD's OpenMP is mapped.
"""

from __future__ import annotations

import json
import sys


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) != 3:
        print("usage: _coacd_worker.py IN.npz OUT.npz KWARGS.json", file=sys.stderr)
        return 2

    in_path, out_path, kwargs_path = argv

    # Local imports keep module import side-effect free for discovery tools.
    import coacd
    import numpy as np

    data = np.load(in_path)
    vertices = np.ascontiguousarray(data["vertices"], dtype=np.float64)
    faces = np.ascontiguousarray(data["faces"], dtype=np.int32)
    with open(kwargs_path, "r", encoding="utf-8") as f:
        kwargs = json.load(f)

    result = coacd.run_coacd(coacd.Mesh(vertices, faces), **kwargs)

    # Variable-length hull list → save as object arrays of contiguous buffers.
    parts_v = []
    parts_f = []
    for vs, fs in result:
        parts_v.append(np.ascontiguousarray(vs, dtype=np.float64))
        parts_f.append(np.ascontiguousarray(fs, dtype=np.int32))
    np.savez_compressed(
        out_path,
        n_parts=np.asarray([len(parts_v)], dtype=np.int32),
        **{f"v{i}": parts_v[i] for i in range(len(parts_v))},
        **{f"f{i}": parts_f[i] for i in range(len(parts_f))},
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
