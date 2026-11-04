#!/bin/sh
# Speed on the CPU: vLLM's CPU build (docker-compose.cpu.yml) as a second stack beside the GPU one
# (project assist-asr-cpu, ports 10310/8110), each model on each set of cores, one after the other:
# two at once would share memory bandwidth and slow each other.
#   MODELS="Qwen/Qwen3-ASR-1.7B Qwen/Qwen3-ASR-0.6B" CORES="all 0-1" bench/speed.sh data/sets/speed
# CORES: "all", or CPU lists as taskset takes them (0-1 = two cores; on most CPUs 0-N are one thread
# per physical core, the second threads come after). BENCH_ARGS: more for bench.py run. Results: bench/results/cpu-<cores>-<model>.json
set -e
cd "$(dirname "$0")/.."
PY=${PY:-python3}
SETS=""; for d in "$@"; do SETS="$SETS --dataset $d"; done
[ -n "$SETS" ] || { echo "usage: $0 DATASET..."; exit 1; }
DC="docker compose -p assist-asr-cpu -f docker-compose.yml -f docker-compose.cpu.yml"
export WYOMING_PORT=10310 VLLM_PORT=8110 RECORD_DIR=
for m in ${MODELS:-Qwen/Qwen3-ASR-1.7B Qwen/Qwen3-ASR-0.6B}; do
  for c in ${CORES:-all 0-1}; do
    echo "== $m on CPU cores: $c"
    if [ "$c" = all ]; then export CPU_CORES=; else export CPU_CORES=$c; fi
    ASR_MODEL=$m $DC up -d --build --wait --force-recreate
    # once to warm up (first requests compile), then measured
    $PY bench/bench.py run --tag warm --out /tmp/assist-asr-warm.json --limit 5 $SETS \
      --system ours=wyoming://127.0.0.1:10310 --system stock=vllm://127.0.0.1:8110 >/dev/null
    $PY bench/bench.py run --tag "cpu-$c-$(basename "$m")" $BENCH_ARGS $SETS \
      --system ours=wyoming://127.0.0.1:10310 --system stock=vllm://127.0.0.1:8110
  done
done
$DC down
