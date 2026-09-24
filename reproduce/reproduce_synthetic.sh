#!/bin/bash
# Clean-environment, end-to-end reproduction of the synthetic reconstruction study (Section 5, Appendix F).
#   bash reproduce/reproduce_synthetic.sh [WORKDIR]
# Creates a fresh Python 3.9 env with only the pinned requirements, clones the repository at the record revision,
# runs the unit tests, regenerates every saved record, and compares them with the archived ones.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
WORK=${1:-$(mktemp -d)}; mkdir -p "$WORK"
REPO=${REPO:-git@github.com:ZhuowenZou/neural-sheaf-diffusion.git}
REV=${REV:-071e1db}                       # revision that produced results/histgeom_2026_09_24
PROCS=${PROCS:-32}; CONDA=${CONDA:-conda}
echo "[1/6] fresh environment in $WORK/env"; t0=$(date +%s)
$CONDA create -y -q -p "$WORK/env" python=3.9 >/dev/null
PY="$WORK/env/bin/python"
"$PY" -m pip install -q --no-cache-dir -r "$HERE/requirements-synthetic.txt"
"$PY" -m pip freeze > "$WORK/pip_freeze.txt"
echo "[2/6] clean clone at $REV"
git clone -q "$REPO" "$WORK/src"; git -C "$WORK/src" checkout -q "$REV"
cd "$WORK/src"; export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONNOUSERSITE=1
echo "[3/6] unit tests";                       "$PY" -m pytest -q -p no:cacheprovider exp/histgeom/test_core.py
echo "[4/6] sweep + analysis";                  "$PY" -m exp.histgeom.sweep --out "$WORK/out" --procs "$PROCS"
                                                "$PY" -m exp.histgeom.analyze --out "$WORK/out"
echo "[5/6] exactness, cost, figures";          "$PY" -m exp.histgeom.exactness --out "$WORK/out" > /dev/null
                                                "$PY" -m exp.histgeom.cost --out "$WORK/out" > /dev/null
                                                "$PY" -m exp.histgeom.figure --out "$WORK/out"; "$PY" -m exp.histgeom.figure --out "$WORK/out" --suffix ""
echo "[6/6] compare with archived records";     "$PY" "$HERE/compare_synthetic.py" "$WORK/out" "$REV" | tee "$WORK/COMPARISON.md"
echo "done in $(( $(date +%s) - t0 )) s; workdir $WORK"
