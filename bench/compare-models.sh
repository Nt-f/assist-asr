#!/bin/sh
# Benchmarks this project ("ours") and the bare model ("stock") for each model, one after the other:
# vLLM is restarted with each, then put back to what .env says.
#   bench/compare-models.sh [--noise 5] data/sets/a data/sets/b ...
# BENCH_ARGS: more for bench.py run (e.g. --realtime). Results: bench/results/<model>.json; the tables: python bench/bench.py report bench/results/*.json
set -e
cd "$(dirname "$0")/.."
PY=${PY:-python3}
NOISE=""
[ "$1" = "--noise" ] && { NOISE="--noise $2"; shift 2; }
[ $# -gt 0 ] || { echo "usage: $0 [--noise N] DATASET..."; exit 1; }
SETS=""; for d in "$@"; do SETS="$SETS --dataset $d"; done
WPORT=$(grep -s '^WYOMING_PORT=' .env | cut -d= -f2); VPORT=$(grep -s '^VLLM_PORT=' .env | cut -d= -f2)
for m in ${MODELS:-Qwen/Qwen3-ASR-1.7B Qwen/Qwen3-ASR-0.6B}; do
  echo "== $m"
  ASR_MODEL=$m docker compose up -d --build --wait
  $PY bench/bench.py run --tag "$(basename "$m")" $NOISE $BENCH_ARGS $SETS \
    --system ours=wyoming://127.0.0.1:${WPORT:-10300} --system stock=vllm://127.0.0.1:${VPORT:-8100}
done
echo "== back to .env's model"
docker compose up -d --wait
$PY bench/bench.py report bench/results/*.json
