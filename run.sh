#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export SOFA_ROOT="${SOFA_ROOT:-$PWD/.sofa/SOFA_v25.12.00_Linux}"
if [[ ! -x "$SOFA_ROOT/bin/runSofa" ]]; then
    echo "Run ./setup.sh first, or set SOFA_ROOT to the SOFA 25.12 installation." >&2
    exit 1
fi
export PYTHONPATH="$SOFA_ROOT/plugins/SofaPython3/lib/python3/site-packages${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$SOFA_ROOT/lib:$SOFA_ROOT/plugins/SofaPython3/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
if [[ ${1:-} == --train ]]; then
    shift
    exec python3.12 train.py "$@"
fi
if [[ ${1:-} == --check ]]; then
    exec python3.12 check_scene.py
fi
scene_file="$PWD/scene.py"
if [[ ${1:-} == --replay ]]; then
    export POINTING_CHECKPOINT="${2:-$PWD/runs/touch.npz}"
    if [[ ! -f "$POINTING_CHECKPOINT" ]]; then
        echo "Missing $POINTING_CHECKPOINT. Run ./run.sh --train first." >&2
        exit 1
    fi
    shift
    if (( $# )); then shift; fi
    scene_file="$PWD/replay.py"
fi
exec "$SOFA_ROOT/bin/runSofa" -l SofaPython3 -g imgui -a "$@" "$scene_file"
