#!/usr/bin/env bash
# Which llama.cpp GPU (Vulkan) settings give right results in an assist-asr CT. Some iGPU drivers compute
# garbage (NaN) in some of llama.cpp's fast paths; this runs the model once per setting, each with one path
# turned off, and shows which ones come out right. In the Proxmox host's shell, with the CT's id:
#
#   bash -c "$(curl -fsSL https://raw.githubusercontent.com/Nt-f/assist-asr/main/proxmox/gpu-check.sh)" -- 113
#
# Whether a setting breaks can change from one start of llama.cpp to the next; "soak" after the id starts the
# candidates many times each and counts:  ... -- 113 soak
# "builds" does that for older llama.cpp releases (default settings), to see whether a newer one broke it;
# "mesa" for the Vulkan driver's (RADV's) debug switches, to see what kind of fault it is.
#
# The CT needs the GPU (pct set <id> --dev0 /dev/dri/renderD128,gid=993). Its services stop meanwhile
# (a few minutes) and start again after.
set -euo pipefail
ID=${1:?usage: gpu-check.sh <CT id> [soak|builds|mesa]}
MODE=${2:-}
command -v pct >/dev/null || { echo "run this on the Proxmox VE host"; exit 1; }
echo "== host GPU"
lspci -nn | grep -iE 'vga|display|3d controller' || true
grep -m1 'model name' /proc/cpuinfo || true
echo "kernel $(uname -r); $(dpkg -l 'pve-firmware' 2>/dev/null | awk '/^ii/ {print $2, $3}')"
pct config "$ID" | grep -E '^(dev[0-9]|cores|memory):' || true
pct exec "$ID" -- test -e /dev/dri/renderD128 || { echo "CT $ID has no GPU: pct set $ID --dev0 /dev/dri/renderD128,gid=993 && pct reboot $ID"; exit 1; }

{ echo "MODE=$MODE"; cat; } <<'IN_CT' | pct exec "$ID" -- bash -s
set -u
M=$(sed -n 's/^LIGHT_MODEL=//p' /etc/assist-asr/assist-asr.env)
export HOME=/var/lib/assist-asr LD_LIBRARY_PATH=/app
systemctl stop assist-asr-wyoming assist-asr-llama
trap 'kill $pid 2>/dev/null; systemctl start assist-asr-llama assist-asr-wyoming' EXIT
pid=""
echo "== the CT's Vulkan driver: $(dpkg -l mesa-vulkan-drivers 2>/dev/null | awk '/^ii/ {print $2, $3}')"
echo "== llama.cpp's devices"
/app/llama-server --list-devices 2>&1 | grep -vE '^(warn|load_backend)' | head -5
echo "== $M, one request per setting"

# 3 s of a gliding tone and noise: content does not matter, broken settings give NaN on anything
python3 - <<'PY'
import math, random, struct, wave
random.seed(1)
with wave.open("/tmp/gpu-check.wav", "wb") as w:
    w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000)
    w.writeframes(b"".join(struct.pack("<h", int(6000 * math.sin(2 * math.pi * (200 + 60 * i / 16000) * i / 16000)
                                                 + random.gauss(0, 800))) for i in range(48000)))
PY

ask() {  # as the server asks: a context prompt of ~1500 characters of names (~450 tokens), 20 alternatives
         # per token, prompt cache, slot 0; twice, the second from the cache. Per request: tokens, how many
         # without a probability (NaN), the time, the text.
python3 - <<'PY'
import base64, json, re, time, urllib.request
u = "http://127.0.0.1:8181"
wav = open("/tmp/gpu-check.wav", "rb").read()
words = [l.strip() for f in ("/etc/assist-asr/vocab.txt", "/etc/assist-asr/commands.txt")
         for l in open(f) if l.strip() and not l.startswith("#")]
ctx = ""
while len(ctx) < 1500:
    ctx += ", ".join(re.sub(r"\s*>.*", "", w) for w in words) + ", "
mk = json.load(urllib.request.urlopen(u + "/props"))["media_marker"]
p = (f"<|im_start|>system\n{ctx[:1500]}<|im_end|>\n<|im_start|>user\n{mk}<|im_end|>\n"
     "<|im_start|>assistant\nlanguage English<asr_text>")
body = {"prompt": {"prompt_string": p, "multimodal_data": [base64.b64encode(wav).decode()]},
        "temperature": 0, "n_predict": 60, "n_probs": 20, "cache_prompt": True, "id_slot": 0}
out, bad = [], False
for _ in range(2):
    t = time.time()
    r = json.load(urllib.request.urlopen(urllib.request.Request(u + "/completion", json.dumps(body).encode(),
                                                                {"Content-Type": "application/json"}), timeout=300))
    cp = r.get("completion_probabilities") or []
    nan = sum(e.get("logprob") is None for e in cp)
    bad |= bool(nan)  # (a tone with names as context: the model may recite them; only NaN is a fault)
    out.append(f"{len(cp):2d} tok {nan:2d} NaN {time.time() - t:4.2f} s")
print(("BAD " if bad else "OK  ") + " | ".join(out) + f": {r.get('content', '')[:30]!r}")
PY
}

BIN=/app
try() {  # name, environment, llama-server options; prints OK/BAD with the numbers
  printf '%-38s' "$1"
  env LD_LIBRARY_PATH=$BIN $2 $BIN/llama-server -hf "$M" -c 8192 -np 2 --host 127.0.0.1 --port 8181 $3 >/tmp/gpu-check.log 2>&1 &
  pid=$!
  for _ in $(seq 120); do
    curl -sf http://127.0.0.1:8181/health >/dev/null 2>&1 && break
    python3 -c 'import urllib.request; urllib.request.urlopen("http://127.0.0.1:8181/health")' 2>/dev/null && break
    kill -0 $pid 2>/dev/null || break
    sleep 1
  done
  if kill -0 $pid 2>/dev/null; then ask 2>&1 | tail -1; else echo "did not start: $(grep -m1 -E ' E |error' /tmp/gpu-check.log)"; fi
  kill $pid 2>/dev/null; wait $pid 2>/dev/null; pid=""
}
ALL="GGML_VK_DISABLE_COOPMAT=1 GGML_VK_DISABLE_COOPMAT2=1 GGML_VK_DISABLE_INTEGER_DOT_PRODUCT=1 GGML_VK_DISABLE_F16=1 GGML_VK_DISABLE_BFLOAT16=1 GGML_VK_DISABLE_MMVQ=1 GGML_VK_DISABLE_DOT2=1 GGML_VK_DISABLE_FUSION=1 GGML_VK_DISABLE_GRAPH_OPTIMIZE=1"
if [ "$MODE" = mesa ]; then  # RADV's debug switches, 4 starts each, llama.cpp's default settings
  tally=()
  for e in "" RADV_DEBUG=zerovram RADV_DEBUG=syncshaders RADV_DEBUG=nocache \
           "RADV_DEBUG=zerovram,syncshaders" ACO_DEBUG=noopt GGML_VK_DISABLE_ASYNC=1; do
    name=${e:-"(none)"}; ok=0
    for i in 1 2 3 4; do try "$name ($i/4)" "$e" "-ngl 99" | tee /tmp/gpu-check.line; grep -q ' OK ' /tmp/gpu-check.line && ok=$((ok + 1)); done
    tally+=("$(printf '%-36s %d of 4 starts OK' "$name" "$ok")")
  done
  echo "== tally"; printf '%s\n' "${tally[@]}"
  rm -f /tmp/gpu-check.wav /tmp/gpu-check.log /tmp/gpu-check.line
  exit 0
fi
if [ "$MODE" = builds ]; then  # the CT's llama.cpp and older releases, 4 starts each, default settings
  W=/var/tmp/gpu-check-builds; mkdir -p "$W"
  tally=()
  for b in image b12098 b12000 b11500 b11000 b10000; do
    if [ "$b" = image ]; then BIN=/app; name="the CT's (image's nightly)"
    else
      BIN=$W/llama-$b; name=$b
      [ -x "$BIN/llama-server" ] || curl -fsSL "https://github.com/ggml-org/llama.cpp/releases/download/$b/llama-$b-bin-ubuntu-vulkan-x64.tar.gz" | tar xz -C "$W" \
        || { echo "$b: download failed"; continue; }
    fi
    ok=0
    for i in 1 2 3 4; do try "$name ($i/4)" "" "-ngl 99" | tee /tmp/gpu-check.line; grep -q ' OK ' /tmp/gpu-check.line && ok=$((ok + 1)); done
    tally+=("$(printf '%-36s %d of 4 starts OK' "$name" "$ok")")
  done
  echo "== tally"; printf '%s\n' "${tally[@]}"
  rm -rf "$W" /tmp/gpu-check.wav /tmp/gpu-check.log /tmp/gpu-check.line
  exit 0
fi
if [ "$MODE" = soak ]; then  # each candidate N starts; the tally at the end
  I=GGML_VK_DISABLE_INTEGER_DOT_PRODUCT=1
  soak() {
    local n=$1 name=$2 ok=0; shift 2
    for i in $(seq "$n"); do try "$name ($i/$n)" "$@" | tee /tmp/gpu-check.line; grep -q ' OK ' /tmp/gpu-check.line && ok=$((ok + 1)); done
    tally+=("$(printf '%-36s %2d of %2d starts OK' "$name" "$ok" "$n")")
  }
  tally=()
  soak 6  "gpu"                         ""          "-ngl 99"
  soak 12 "gpu, no integer dot product" "$I"        "-ngl 99"
  soak 4  "gpu, all fast paths off"     "$ALL"      "-ngl 99 -fa off"
  echo "== tally"; printf '%s\n' "${tally[@]}"
  rm -f /tmp/gpu-check.wav /tmp/gpu-check.log /tmp/gpu-check.line
  exit 0
fi
F=GGML_VK_DISABLE_FUSION=1
try "cpu (reference)"              ""                                        "-ngl 0 -dev none --no-mmproj-offload"
try "gpu"                          ""                                        "-ngl 99"
try "gpu, all fast paths off"      "$ALL"                                    "-ngl 99 -fa off"
try "gpu, no fusion"               "$F"                                      "-ngl 99"
try "gpu, flash attention off"     ""                                        "-ngl 99 -fa off"
try "gpu, no fusion, fa off"       "$F"                                      "-ngl 99 -fa off"
try "gpu, KV cache f32"            ""                                        "-ngl 99 -ctk f32 -ctv f32"
try "gpu, no integer dot product"  "GGML_VK_DISABLE_INTEGER_DOT_PRODUCT=1"   "-ngl 99"
try "gpu, no f16"                  "GGML_VK_DISABLE_F16=1"                   "-ngl 99"
try "gpu, no bfloat16"             "GGML_VK_DISABLE_BFLOAT16=1"              "-ngl 99"
try "gpu, no mmvq"                 "GGML_VK_DISABLE_MMVQ=1"                  "-ngl 99"
try "gpu, no dot2"                 "GGML_VK_DISABLE_DOT2=1"                  "-ngl 99"
try "gpu, small batches (-ub 32)"  ""                                        "-ngl 99 -ub 32"
try "gpu, host memory"             "GGML_VK_PREFER_HOST_MEMORY=1"            "-ngl 99"
try "gpu, serialized submissions"  "GGML_VK_SERIALIZE_SUBMISSIONS=1"         "-ngl 99"
try "gpu, encoder on the cpu"      ""                                        "-ngl 99 --no-mmproj-offload"
rm -f /tmp/gpu-check.wav /tmp/gpu-check.log
IN_CT
echo "== done; the services are running again"
