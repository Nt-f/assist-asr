#!/usr/bin/env bash
# Which llama.cpp GPU (Vulkan) settings give right results in an assist-asr CT. Some iGPU drivers compute
# garbage (NaN) in some of llama.cpp's fast paths; this runs the model once per setting, each with one path
# turned off, and shows which ones come out right. In the Proxmox host's shell, with the CT's id:
#
#   bash -c "$(curl -fsSL https://raw.githubusercontent.com/Nt-f/assist-asr/main/proxmox/gpu-check.sh)" -- 113
#
# The CT needs the GPU (pct set <id> --dev0 /dev/dri/renderD128,gid=993). Its services stop meanwhile
# (a few minutes) and start again after.
set -euo pipefail
ID=${1:?usage: gpu-check.sh <CT id>}
command -v pct >/dev/null || { echo "run this on the Proxmox VE host"; exit 1; }
echo "== host GPU"
lspci -nn | grep -iE 'vga|display|3d controller' || true
grep -m1 'model name' /proc/cpuinfo || true
pct config "$ID" | grep -E '^(dev[0-9]|cores|memory):' || true
pct exec "$ID" -- test -e /dev/dri/renderD128 || { echo "CT $ID has no GPU: pct set $ID --dev0 /dev/dri/renderD128,gid=993 && pct reboot $ID"; exit 1; }

pct exec "$ID" -- bash -s <<'IN_CT'
set -u
M=$(sed -n 's/^LIGHT_MODEL=//p' /etc/assist-asr/assist-asr.env)
export HOME=/var/lib/assist-asr LD_LIBRARY_PATH=/app
systemctl stop assist-asr-wyoming assist-asr-llama
trap 'kill $pid 2>/dev/null; systemctl start assist-asr-llama assist-asr-wyoming' EXIT
pid=""
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

ask() {  # one request: tokens, how many without a probability (NaN), the text, the time
python3 - <<'PY'
import base64, json, time, urllib.request
u = "http://127.0.0.1:8181"
wav = open("/tmp/gpu-check.wav", "rb").read()
mk = json.load(urllib.request.urlopen(u + "/props"))["media_marker"]
p = f"<|im_start|>user\n{mk}<|im_end|>\n<|im_start|>assistant\nlanguage English<asr_text>"
body = {"prompt": {"prompt_string": p, "multimodal_data": [base64.b64encode(wav).decode()]},
        "temperature": 0, "n_predict": 60, "n_probs": 5}
t = time.time()
r = json.load(urllib.request.urlopen(urllib.request.Request(u + "/completion", json.dumps(body).encode(),
                                                            {"Content-Type": "application/json"}), timeout=300))
cp = r.get("completion_probabilities") or []
nan = sum(e.get("logprob") is None for e in cp)
print(f"{'OK  ' if not nan and len(cp) < 60 else 'BAD '} {len(cp):3d} tokens, {nan:3d} NaN, {time.time() - t:5.2f} s: {r.get('content', '')[:50]!r}")
PY
}

try() {  # name, environment, llama-server options
  printf '%-34s' "$1"
  env $2 /app/llama-server -hf "$M" -c 4096 -np 1 --host 127.0.0.1 --port 8181 $3 >/tmp/gpu-check.log 2>&1 &
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
try "cpu (reference)"              ""                                        "-ngl 0 -dev none --no-mmproj-offload"
try "gpu"                          ""                                        "-ngl 99"
try "gpu, all fast paths off"      "$ALL"                                    "-ngl 99 -fa off"
try "gpu, flash attention off"     ""                                        "-ngl 99 -fa off"
try "gpu, KV cache f32"            ""                                        "-ngl 99 -ctk f32 -ctv f32"
try "gpu, no coopmat"              "GGML_VK_DISABLE_COOPMAT=1 GGML_VK_DISABLE_COOPMAT2=1" "-ngl 99"
try "gpu, no integer dot product"  "GGML_VK_DISABLE_INTEGER_DOT_PRODUCT=1"   "-ngl 99"
try "gpu, no f16"                  "GGML_VK_DISABLE_F16=1"                   "-ngl 99"
try "gpu, no bfloat16"             "GGML_VK_DISABLE_BFLOAT16=1"              "-ngl 99"
try "gpu, no mmvq"                 "GGML_VK_DISABLE_MMVQ=1"                  "-ngl 99"
try "gpu, no dot2"                 "GGML_VK_DISABLE_DOT2=1"                  "-ngl 99"
try "gpu, no fusion"               "GGML_VK_DISABLE_FUSION=1"                "-ngl 99"
try "gpu, no graph optimize"       "GGML_VK_DISABLE_GRAPH_OPTIMIZE=1"        "-ngl 99"
rm -f /tmp/gpu-check.wav /tmp/gpu-check.log
IN_CT
echo "== done; the services are running again"
