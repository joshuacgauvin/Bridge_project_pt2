#!/usr/bin/env bash
# One-command cloud run (Linux/macOS): builds a venv beside this script, installs the
# pinned requirements, and starts the search detached so it survives an SSH disconnect.
#
#   bash run_tuning.sh                 # full search (any tune_rsf.py flags pass through)
#   bash run_tuning.sh --smoke         # quick end-to-end check first
#   tail -f tuning.log                 # follow progress
#
# Rerunning the same command after a crash or a stopped VM resumes where it left off.
set -euo pipefail
cd "$(dirname "$0")"

[ -f rsf_tuning_train.csv ] || { echo "rsf_tuning_train.csv is missing from $(pwd)"; exit 1; }

# The pins only have prebuilt wheels for Python 3.12 on x86-64 (numpy 2.5 dropped 3.11,
# ecos has no 3.13 wheel, scikit-survival has no ARM wheel) — anything else compiles from source.
[ "$(uname -m)" = "x86_64" ] || {
    echo "This needs an x86-64 (Intel/AMD) machine; $(uname -m) has no scikit-survival wheel."
    exit 1
}
PY="${PYTHON:-python3}"
"$PY" -c 'import sys; sys.exit(sys.version_info[:2] != (3, 12))' || {
    echo "Python 3.12 is required. Point PYTHON at it, e.g. PYTHON=python3.12 bash run_tuning.sh"
    exit 1
}
if [ ! -x .venv/bin/python ]; then
    "$PY" -m venv .venv
    .venv/bin/python -m pip install --upgrade pip
    .venv/bin/python -m pip install -r requirements.txt
fi

nohup .venv/bin/python -u tune_rsf.py "$@" >> tuning.log 2>&1 &
echo "started tune_rsf.py (PID $!) — follow with: tail -f $(pwd)/tuning.log"
