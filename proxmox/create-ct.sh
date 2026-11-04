#!/bin/bash
# Creates the speech-to-text CT on a Proxmox VE host from the template build-template.sh made.
#
#   ./create-ct.sh --template assist-asr-light-<date>-amd64.tar.zst [options]
#
#   --id N              CT id (default: the next free one)
#   --storage NAME      for the CT's disk (default: local-lvm)
#   --bridge NAME       network bridge (default: vmbr0); the CT takes an address by DHCP
#   --cores N           CPU cores (default: 4)
#   --memory MB         RAM (default: 4096; the 1.7B model wants ~4 GB, the 0.6B ~2)
#   --disk GB           disk (default: 8: the system ~1.5 GB, models 1-2.5 GB)
#   --model 0.6B|1.7B   (default: 0.6B; on a CPU the better pick, see README "Results")
#   --gpu auto|yes|no   share the host's GPU (/dev/dri/renderD128: AMD or Intel iGPU, through Vulkan) with
#                       the CT (default: auto, when the host has one). Shared, not passed through: the host
#                       and other CTs keep it, and no GPU reset trouble as with a VM.
#   --hass-url URL      Home Assistant, e.g. http://192.168.1.10:8123 (or later in /etc/assist-asr/assist-asr.env)
#   --hass-token TOKEN  a long-lived access token (HA: your profile -> Security)
set -euo pipefail

TEMPLATE="" ID="" STORAGE=local-lvm BRIDGE=vmbr0 CORES=4 MEMORY=4096 DISK=8 MODEL=0.6B GPU=auto
HASS_URL="" HASS_TOKEN=""
while [ $# -gt 0 ]; do
  case "$1" in
    --template) TEMPLATE=$2; shift 2 ;;
    --id) ID=$2; shift 2 ;;
    --storage) STORAGE=$2; shift 2 ;;
    --bridge) BRIDGE=$2; shift 2 ;;
    --cores) CORES=$2; shift 2 ;;
    --memory) MEMORY=$2; shift 2 ;;
    --disk) DISK=$2; shift 2 ;;
    --model) MODEL=$2; shift 2 ;;
    --gpu) GPU=$2; shift 2 ;;
    --hass-url) HASS_URL=$2; shift 2 ;;
    --hass-token) HASS_TOKEN=$2; shift 2 ;;
    -h|--help) sed -n '2,/^set -euo/p' "$0" | sed '$d; s/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown option $1 (--help)"; exit 1 ;;
  esac
done
command -v pct >/dev/null || { echo "run this on the Proxmox VE host (pct not found)"; exit 1; }
[ -f "$TEMPLATE" ] || { echo "--template: the .tar.zst build-template.sh made"; exit 1; }
case "$MODEL" in 0.6B|1.7B) ;; *) echo "--model 0.6B or 1.7B"; exit 1 ;; esac

# the template goes where Proxmox looks for CT templates (storage "local")
CACHE=/var/lib/vz/template/cache
cp -n "$TEMPLATE" "$CACHE/" || true
VOL="local:vztmpl/$(basename "$TEMPLATE")"
ID=${ID:-$(pvesh get /cluster/nextid)}

create() {  # Proxmox sets up hostname and network itself for the OS types it knows
  pct create "$ID" "$VOL" --hostname assist-asr --cores "$CORES" --memory "$MEMORY" --swap 512 \
    --rootfs "$STORAGE:$DISK" --net0 "name=eth0,bridge=$BRIDGE,ip=dhcp" \
    --unprivileged 1 --features nesting=1 --onboot 1 --ostype "$1"
}
# The template is Ubuntu 26.04 (llama.cpp's own base); a Proxmox that does not know that release yet
# refuses "ubuntu": then "unmanaged", and the CT configures its network itself (DHCP, systemd-networkd).
if ! create ubuntu 2>/tmp/assist-asr-create.err; then
  if grep -qi "unsupported" /tmp/assist-asr-create.err; then
    echo "Proxmox does not know this Ubuntu release yet: creating it as an unmanaged OS"
    create unmanaged
  else
    cat /tmp/assist-asr-create.err; exit 1
  fi
fi

if [ "$GPU" = yes ] || { [ "$GPU" = auto ] && [ -e /dev/dri/renderD128 ]; }; then
  [ -e /dev/dri/renderD128 ] || { echo "--gpu yes: this host has no /dev/dri/renderD128"; exit 1; }
  # gid 993 = the CT's render group (fixed in the template)
  pct set "$ID" --dev0 /dev/dri/renderD128,gid=993
  echo "GPU: /dev/dri/renderD128 shared with CT $ID"
fi

pct start "$ID"
sleep 5
set_env() { pct exec "$ID" -- sed -i "s|^$1=.*|$1=$2|" /etc/assist-asr/assist-asr.env; }
set_env LIGHT_MODEL "ggml-org/Qwen3-ASR-$MODEL-GGUF:Q8_0"
[ -n "$HASS_URL" ] && set_env HASS_URL "$HASS_URL"
[ -n "$HASS_TOKEN" ] && set_env HASS_TOKEN "$HASS_TOKEN"
pct exec "$ID" -- systemctl restart assist-asr-llama assist-asr-wyoming

IP=""
for _ in $(seq 30); do
  IP=$(pct exec "$ID" -- sh -c "ip -4 -o addr show eth0 | awk '{print \$4}' | cut -d/ -f1" 2>/dev/null || true)
  [ -n "$IP" ] && break
  sleep 2
done
cat <<MSG

CT $ID is up${IP:+ at $IP}. The model downloads on first start (1-2.5 GB): watch it with
  pct exec $ID -- journalctl -fu assist-asr-llama -u assist-asr-wyoming
It's ready once the log says "listening on 10300". Then in Home Assistant: Settings -> Devices & services ->
Add integration -> Wyoming Protocol -> host ${IP:-<the CT's address>}, port 10300.
Settings: pct exec $ID -- nano /etc/assist-asr/assist-asr.env (then restart both services).
MSG
