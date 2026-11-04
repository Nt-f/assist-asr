#!/usr/bin/env bash
# assist-asr on Proxmox VE: makes the speech-to-text CT (the light version, systemd services, no Docker)
# from the template GitHub Actions builds for each release. In the Proxmox host's shell:
#
#   bash -c "$(curl -fsSL https://raw.githubusercontent.com/Nt-f/assist-asr/main/proxmox/install.sh)"
#
# It asks for the settings, or takes them as options: --help.
set -euo pipefail

REPO=Nt-f/assist-asr
TEMPLATE="" VERSION="" ID="" STORAGE="" TSTORAGE="" BRIDGE=vmbr0 CORES=4 MEMORY=4096 DISK=8 MODEL=0.6B GPU=auto
HASS_URL="" HASS_TOKEN="" YES=0

usage() { cat <<'EOF_HELP'
It asks for the settings. Or pass them, with --yes to skip the questions (after `bash -s --` when piped):

  --hass-url URL      Home Assistant, e.g. http://192.168.1.10:8123 (or later in /etc/assist-asr/assist-asr.env)
  --hass-token TOKEN  a long-lived access token (HA: your profile -> Security)
  --model 0.6B|1.7B   (default: 0.6B; on a CPU the better pick, see README "Results")
  --gpu auto|yes|no   share the host's GPU (/dev/dri/renderD128: AMD or Intel iGPU, through Vulkan) with
                      the CT (default: auto, when the host has one). Shared, not passed through: the host
                      and other CTs keep it, and no GPU reset trouble as with a VM.
  --id N              CT id (default: the next free one)
  --storage NAME      for the CT's disk (default: local-lvm, else local-zfs, else the first that takes CTs)
  --tstorage NAME     where the template is kept (default: local, else the first that takes templates)
  --bridge NAME       network bridge (default: vmbr0); the CT takes an address by DHCP
  --cores N           CPU cores (default: 4)
  --memory MB         RAM (default: 4096; the 1.7B model wants ~4 GB, the 0.6B ~2)
  --disk GB           disk (default: 8: the system ~1.5 GB, models 1-2.5 GB)
  --version TAG       a release (default: the latest)
  --template FILE     a template built with proxmox/build-template.sh, instead of a release's
  --yes               no questions
EOF_HELP
}
say() { printf '\e[1;36m==>\e[0m %s\n' "$*"; }
die() { printf '\e[1;31merror:\e[0m %s\n' "$*" >&2; exit 1; }

while [ $# -gt 0 ]; do
  case "$1" in
    --template) TEMPLATE=$2; shift 2 ;;
    --version) VERSION=$2; shift 2 ;;
    --id) ID=$2; shift 2 ;;
    --storage) STORAGE=$2; shift 2 ;;
    --tstorage) TSTORAGE=$2; shift 2 ;;
    --bridge) BRIDGE=$2; shift 2 ;;
    --cores) CORES=$2; shift 2 ;;
    --memory) MEMORY=$2; shift 2 ;;
    --disk) DISK=$2; shift 2 ;;
    --model) MODEL=$2; shift 2 ;;
    --gpu) GPU=$2; shift 2 ;;
    --hass-url) HASS_URL=$2; shift 2 ;;
    --hass-token) HASS_TOKEN=$2; shift 2 ;;
    -y|--yes) YES=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown option $1 (--help)" ;;
  esac
done

command -v pct >/dev/null || die "run this on the Proxmox VE host (pct not found)"
[ "$(id -u)" = 0 ] || die "run this as root"
[ -n "$TEMPLATE" ] || [ "$(dpkg --print-architecture)" = amd64 ] \
  || die "the released template is amd64; on ARM build one with proxmox/build-template.sh and pass --template"

storages() { pvesm status -content "$1" -enabled 1 2>/dev/null | awk 'NR>1 {print $1}'; }
pick() {  # the first of the preferred names that exists, else the first one
  local all; all=$(storages "$1"); shift
  for s in "$@"; do grep -qx "$s" <<<"$all" && { echo "$s"; return; }; done
  head -1 <<<"$all"
}
STORAGE=${STORAGE:-$(pick rootdir local-lvm local-zfs)}
TSTORAGE=${TSTORAGE:-$(pick vztmpl local)}
[ -n "$STORAGE" ] || die "no storage takes CT disks (pvesm status -content rootdir)"
[ -n "$TSTORAGE" ] || die "no storage takes CT templates (pvesm status -content vztmpl)"
ID=${ID:-$(pvesh get /cluster/nextid)}
HAS_GPU=0; [ -e /dev/dri/renderD128 ] && HAS_GPU=1

# ---- questions (whiptail, as Proxmox has it) -------------------------------------------------------------
if [ "$YES" = 0 ]; then
  { : </dev/tty; } 2>/dev/null || die "no terminal to ask in: pass the settings and --yes (--help)"
  exec </dev/tty   # also when piped: curl ... | bash
  T="assist-asr"
  ask() { whiptail --title "$T" "$@" 3>&1 1>&2 2>&3; }

  MODE=$(ask --menu "Speech-to-text for Home Assistant (Qwen3-ASR on llama.cpp) in a new CT." 12 70 2 \
    default "CT $ID, $CORES cores, $((MEMORY / 1024)) GB RAM, ${DISK} GB on $STORAGE" \
    advanced "choose the CT's id, storage, network and size") || exit 1
  HASS_URL=$(ask --inputbox "Home Assistant's address (its names are what the model listens for).\nEmpty: set it later." 10 70 "${HASS_URL:-http://homeassistant.local:8123}") || exit 1
  HASS_TOKEN=$(ask --passwordbox "A long-lived access token: in Home Assistant, your profile -> Security -> Long-lived access tokens.\nEmpty: set it later." 11 70 "$HASS_TOKEN") || exit 1
  MODEL=$(ask --default-item "$MODEL" --menu "Model" 11 70 2 \
    0.6B "1 GB, faster; the pick on a CPU" \
    1.7B "2.5 GB, more accurate; wants ~4 GB RAM and a GPU") || exit 1
  if [ "$HAS_GPU" = 1 ] && [ "$GPU" = auto ]; then
    if ask --yesno "This host has a GPU (/dev/dri/renderD128). Share it with the CT (Vulkan)?\n\nA small iGPU can be slower than the CPU; README \"Results\"." 11 70; then GPU=yes; else GPU=no; fi
  fi
  if [ "$MODE" = advanced ]; then
    ID=$(ask --inputbox "CT id" 8 50 "$ID") || exit 1
    ITEMS=()
    for s in $(storages rootdir); do
      ITEMS+=("$s" "$(pvesm status -storage "$s" | awk 'NR==2 {printf "%s, %.0f GB free", $2, $6/1048576}')")
    done
    STORAGE=$(ask --default-item "$STORAGE" --menu "Storage for the CT's disk" 14 70 6 "${ITEMS[@]}") || exit 1
    BRIDGE=$(ask --inputbox "Network bridge (the CT takes an address by DHCP)" 8 60 "$BRIDGE") || exit 1
    CORES=$(ask --inputbox "CPU cores" 8 50 "$CORES") || exit 1
    MEMORY=$(ask --inputbox "RAM in MB (the 1.7B model wants ~4096, the 0.6B ~2048)" 8 60 "$MEMORY") || exit 1
    DISK=$(ask --inputbox "Disk in GB" 8 50 "$DISK") || exit 1
  fi
  ask --yesno "Create CT $ID?\n\n  model       $MODEL\n  GPU         $GPU\n  Home Asst.  ${HASS_URL:-(later)}\n  token       $([ -n "$HASS_TOKEN" ] && echo set || echo '(later)')\n  storage     $STORAGE, ${DISK} GB\n  cores/RAM   $CORES / $MEMORY MB\n  bridge      $BRIDGE" 17 60 || exit 1
fi

case "$MODEL" in 0.6B|1.7B) ;; *) die "--model 0.6B or 1.7B" ;; esac
case "$GPU" in auto|yes|no) ;; *) die "--gpu auto, yes or no" ;; esac
pct status "$ID" >/dev/null 2>&1 && die "CT $ID exists already"

# ---- the template -----------------------------------------------------------------------------------------
if [ -n "$TEMPLATE" ]; then
  [ -f "$TEMPLATE" ] || die "--template $TEMPLATE: no such file"
  FILE=$(basename "$TEMPLATE")
  DEST=$(pvesm path "$TSTORAGE:vztmpl/$FILE")
  [ -f "$DEST" ] || cp "$TEMPLATE" "$DEST"
else
  if [ -z "$VERSION" ]; then  # the latest release's tag, from where GitHub redirects
    VERSION=$(curl -fsSLI -o /dev/null -w '%{url_effective}' "https://github.com/$REPO/releases/latest")
    VERSION=${VERSION##*/}
    [ "$VERSION" != releases ] && [ "$VERSION" != latest ] || die "$REPO has no release yet"
  fi
  URL=https://github.com/$REPO/releases/download/$VERSION/assist-asr-light-amd64.tar.zst
  FILE=assist-asr-light-$VERSION-amd64.tar.zst
  DEST=$(pvesm path "$TSTORAGE:vztmpl/$FILE")
  SUM=$(curl -fsSL "$URL.sha256" | awk '{print $1}') || die "release $VERSION has no template ($URL.sha256)"
  if [ -f "$DEST" ] && [ "$(sha256sum "$DEST" | awk '{print $1}')" = "$SUM" ]; then
    say "template $VERSION: already on $TSTORAGE"
  else
    say "downloading the template, $VERSION (~250 MB)"
    curl -fL --progress-bar -o "$DEST.part" "$URL"
    [ "$(sha256sum "$DEST.part" | awk '{print $1}')" = "$SUM" ] || { rm -f "$DEST.part"; die "checksum mismatch"; }
    mv "$DEST.part" "$DEST"
  fi
fi
VOL="$TSTORAGE:vztmpl/$FILE"

# ---- the CT -------------------------------------------------------------------------------------------------
create() {  # Proxmox sets up hostname and network itself for the OS types it knows
  pct create "$ID" "$VOL" --hostname assist-asr --cores "$CORES" --memory "$MEMORY" --swap 512 \
    --rootfs "$STORAGE:$DISK" --net0 "name=eth0,bridge=$BRIDGE,ip=dhcp" \
    --unprivileged 1 --features nesting=1 --onboot 1 --cmode console --ostype "$1"
}
say "creating CT $ID"
ERR=$(mktemp)
# The template is Ubuntu 26.04 (llama.cpp's own base); a Proxmox that does not know that release yet
# refuses "ubuntu": then "unmanaged", and the CT configures its network itself (DHCP, systemd-networkd).
if ! create ubuntu 2>"$ERR"; then
  if grep -qi "unsupported" "$ERR"; then
    say "Proxmox does not know this Ubuntu release yet: creating it as an unmanaged OS"
    create unmanaged
  else
    cat "$ERR" >&2; rm -f "$ERR"; exit 1
  fi
fi
rm -f "$ERR"

if [ "$GPU" = yes ] || { [ "$GPU" = auto ] && [ "$HAS_GPU" = 1 ]; }; then
  [ "$HAS_GPU" = 1 ] || die "--gpu yes: this host has no /dev/dri/renderD128"
  # gid 993 = the CT's render group (fixed in the template)
  pct set "$ID" --dev0 /dev/dri/renderD128,gid=993
  say "GPU: /dev/dri/renderD128 shared with CT $ID"
fi

# The settings go in before the first start, so the right model is the one downloaded. Rewritten in place
# (not sed -i), which keeps the file's owner and mode as the unprivileged CT sees them.
pct mount "$ID" >/dev/null
ENVF=/var/lib/lxc/$ID/rootfs/etc/assist-asr/assist-asr.env
NEW=$(LM="ggml-org/Qwen3-ASR-$MODEL-GGUF:Q8_0" HU="$HASS_URL" HT="$HASS_TOKEN" awk '
  /^LIGHT_MODEL=/ { $0 = "LIGHT_MODEL=" ENVIRON["LM"] }
  /^HASS_URL=/   && ENVIRON["HU"] != "" { $0 = "HASS_URL=" ENVIRON["HU"] }
  /^HASS_TOKEN=/ && ENVIRON["HT"] != "" { $0 = "HASS_TOKEN=" ENVIRON["HT"] }
  { print }' "$ENVF")
printf '%s\n' "$NEW" > "$ENVF"
pct unmount "$ID"

pct start "$ID"
IP=""
for _ in $(seq 30); do
  IP=$(pct exec "$ID" -- sh -c "ip -4 -o addr show eth0 | awk '{print \$4}' | cut -d/ -f1" 2>/dev/null || true)
  [ -n "$IP" ] && break
  sleep 2
done
HOST=${IP:-"<the CT's address>"}
cat <<MSG

CT $ID is up${IP:+ at $IP}. The model downloads on first start (1-2.5 GB): watch it in the CT's Console
(Proxmox's web UI), or with  pct exec $ID -- journalctl -fu assist-asr-llama -u assist-asr-wyoming
It's ready once the log says "listening on 10300". Then in Home Assistant: Settings -> Devices & services ->
Add integration -> Wyoming Protocol -> host $HOST, port 10300.
Settings: pct exec $ID -- nano /etc/assist-asr/assist-asr.env (then restart both services). A shell: pct enter $ID
MSG
