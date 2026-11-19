#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
lab_python="${PYTHON:-python}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export MPLBACKEND=Agg
for stage in environment regression classification capymoa clustering automl; do
    "$lab_python" benchmark.py "$stage"
done
"$lab_python" plot_results.py
"$lab_python" build_report.py
"$lab_python" build_notebook.py
