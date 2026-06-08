#!/usr/bin/env bash
# Upgrade Nyx stack to the genesis-1.0/1.1-compatible versions and run the
# official Nyx example headless. Run INSIDE the genesis-dev:h20 container.
#
#   gs-nyx        0.1.1 -> 0.1.2
#   gs-nyx-plugin 0.1.2 -> 0.1.3   (4-arg Sensor generic, 5-arg __init__ — matches genesis 1.0+)
#
# We install with --no-deps so the plugin's `genesis-world>=1.1.0` pin does NOT
# drag in / overwrite the mounted editable genesis. We first test against the
# mounted genesis 1.0.0; the two contracts that broke 0.1.2 (generic arity,
# __init__ arity) are both satisfied by 1.0.0, so this should reach build().
set -uo pipefail

WHL_DIR=/workspace/Uni-Genesis/docker-genesis/nyx_upgrade

echo "=== editable genesis (mounted) ==="
python3 -m pip install --user --no-deps --no-build-isolation -e /workspace/Uni-Genesis/genesis >/dev/null 2>&1
python3 -c "from genesis.version import __version__; print('genesis', __version__)" 2>&1 | tail -1

echo "=== upgrade gs-nyx 0.1.2 + gs-nyx-plugin 0.1.3 (--no-deps) ==="
python3 -m pip install --no-deps --force-reinstall \
    "$WHL_DIR"/gs_nyx-0.1.2-*.whl \
    "$WHL_DIR"/gs_nyx_plugin-0.1.3-*.whl 2>&1 | grep -iE "Successfully|error|ERROR" | tail

python3 - <<'PY'
import importlib.metadata as md
for p in ("gs-nyx","gs-nyx-plugin"):
    print(p, md.version(p))
PY

echo "=== run official 01_hello_nyx.py (headless, no Xvfb first) ==="
cd /workspace/Uni-Genesis/genesis-nyx
set +e
python3 examples/01_hello_nyx.py 2>&1 | grep -vE '\[Genesis\].*(Adding|created|version|backend|Running on|theme|seed)' | grep -viE 'CoACD.*Processing' | tail -45
echo "=== exit: $? ==="
