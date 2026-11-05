#!/usr/bin/env bash
# assist-asr on a Mac with Apple silicon, on the GPU through Metal: the light version without Docker (Docker on a Mac
# has no GPU, so there it runs on the CPU only). Homebrew's llama.cpp (built with Metal) and the Wyoming server in a
# Python venv, both kept running by launchd. In Terminal:
#
#   bash -c "$(curl -fsSL https://raw.githubusercontent.com/Nt-f/assist-asr/main/macos/install.sh)"
#
# or from a clone: macos/install.sh. It asks for the settings, or takes them as options: --help.
# --update brings the server and llama.cpp up to date, --uninstall takes it all away again.
set -euo pipefail

REPO=Nt-f/assist-asr
REF=main
HASS_URL="" HASS_TOKEN="" MODEL=1.7B PORT=10300 YES=0 ACTION=install SYSTEM=0
APP="$HOME/Library/Application Support/assist-asr"
LOGS="$HOME/Library/Logs/assist-asr"
AGENTS="$HOME/Library/LaunchAgents"
LLAMA_PORT=8120

usage() { cat <<'EOF_HELP'
It asks for the settings. Or pass them, with --yes to skip the questions:

  --hass-url URL      Home Assistant, e.g. http://192.168.1.10:8123 (or later in the settings file)
  --hass-token TOKEN  a long-lived access token (HA: your profile -> Security)
  --model 0.6B|1.7B   Qwen3-ASR's size (default: 1.7B: on Metal the more accurate one is fast too)
  --port N            the port Home Assistant connects to (default: 10300)
  --ref REF           a branch or tag of the repository to install from (default: main; ignored in a clone)
  --yes               no questions

  --update            the server from the repository again, llama.cpp from Homebrew; settings and models stay
  --uninstall         stop and remove the services and the app folder (the downloaded model stays in
                      ~/Library/Caches/llama.cpp; delete it yourself if you want the space back)

Settings afterwards: ~/Library/Application Support/assist-asr/assist-asr.env (then: --update, or
launchctl kickstart -k gui/$(id -u)/com.assist-asr.wyoming). Logs: ~/Library/Logs/assist-asr.
The services run while you are logged in (launchd agents): on a Mac that is always on, turn on automatic login.
EOF_HELP
}
say() { printf '\033[1;36m==>\033[0m %s\n' "$*"; }
die() { printf '\033[1;31merror:\033[0m %s\n' "$*" >&2; exit 1; }

while [ $# -gt 0 ]; do
  case "$1" in
    --hass-url) HASS_URL=$2; shift 2 ;;
    --hass-token) HASS_TOKEN=$2; shift 2 ;;
    --model) MODEL=$2; shift 2 ;;
    --port) PORT=$2; shift 2 ;;
    --ref) REF=$2; shift 2 ;;
    -y|--yes) YES=1; shift ;;
    --update) ACTION=update; YES=1; shift ;;
    --uninstall) ACTION=uninstall; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown option $1 (--help)" ;;
  esac
done

[ "$(uname -s)" = Darwin ] || die "this is for macOS; on Linux use Docker (README) or proxmox/install.sh"
[ "$(uname -m)" = arm64 ] || say "not Apple silicon: no Metal here, llama.cpp will run on the CPU"

UIDN=$(id -u)
unload() { launchctl bootout "gui/$UIDN/$1" 2>/dev/null || true; }

if [ "$ACTION" = uninstall ]; then
  for s in com.assist-asr.wyoming com.assist-asr.llama; do unload "$s"; rm -f "$AGENTS/$s.plist"; done
  rm -rf "$APP"
  say "removed (the model is still in ~/Library/Caches/llama.cpp, the logs in $LOGS)"
  exit 0
fi

command -v brew >/dev/null || die "needs Homebrew: https://brew.sh (then run this again)"

# --- the settings -----------------------------------------------------------------------------------------------
ENV="$APP/assist-asr.env"
if [ "$ACTION" = install ] && [ ! -f "$ENV" ] && [ "$YES" = 0 ]; then
  [ -n "$HASS_URL" ] || read -r -p "Home Assistant's URL (e.g. http://192.168.1.10:8123; empty: none): " HASS_URL </dev/tty
  if [ -n "$HASS_URL" ] && [ -z "$HASS_TOKEN" ]; then
    read -r -s -p "A long-lived access token (HA: your profile -> Security): " HASS_TOKEN </dev/tty; echo
  fi
  read -r -p "Model, 0.6B or 1.7B [$MODEL]: " m </dev/tty; MODEL=${m:-$MODEL}
fi
case "$MODEL" in 0.6B|1.7B) ;; *) die "--model is 0.6B or 1.7B" ;; esac

# --- llama.cpp and Python ---------------------------------------------------------------------------------------
if [ "$ACTION" = update ]; then
  say "llama.cpp from Homebrew (upgrade)"; brew upgrade llama.cpp 2>/dev/null || brew install llama.cpp
else
  say "llama.cpp from Homebrew (Metal on Apple silicon)"; brew list llama.cpp >/dev/null 2>&1 || brew install llama.cpp
fi
LLAMA_SERVER=$(command -v llama-server || echo "$(brew --prefix)/bin/llama-server")
[ -x "$LLAMA_SERVER" ] || die "no llama-server after installing llama.cpp"
PY=$(command -v python3.12 || command -v python3.13 || true)
if [ -z "$PY" ]; then say "Python 3.12 from Homebrew"; brew install python@3.12; PY="$(brew --prefix)/bin/python3.12"; fi

# --- the server's files: from this clone, or the repository -----------------------------------------------------
mkdir -p "$APP/app" "$APP/config" "$APP/data" "$LOGS" "$AGENTS"
HERE=$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" 2>/dev/null && pwd || true)
if [ -n "$HERE" ] && [ -f "$HERE/../wyoming/server.py" ]; then
  SRC="$HERE/.."; say "the server from this clone ($SRC)"
else
  SRC=$(mktemp -d); trap 'rm -rf "$SRC"' EXIT
  say "the server from github.com/$REPO ($REF)"
  curl -fsSL "https://github.com/$REPO/archive/$REF.tar.gz" | tar -xz -C "$SRC" --strip-components 1
fi
cp "$SRC/wyoming/server.py" "$SRC/plugin/asr_bias.py" "$APP/app/"
for f in "$SRC"/config/*; do   # the config files are the user's after the first install: new ones only
  [ -e "$APP/config/$(basename "$f")" ] || cp "$f" "$APP/config/"
done

if [ ! -x "$APP/venv/bin/python" ]; then say "Python environment"; "$PY" -m venv "$APP/venv"; fi
"$APP/venv/bin/pip" install -q --upgrade pip
"$APP/venv/bin/pip" install -q wyoming==1.10.2 pysilero-vad==3.4.0 zeroconf

if [ ! -f "$ENV" ]; then
  umask 077
  cat > "$ENV" <<EOF_ENV
# assist-asr's settings (README "Settings"). After a change: macos/install.sh --update, or
#   launchctl kickstart -k gui/\$(id -u)/com.assist-asr.wyoming   (and com.assist-asr.llama for LIGHT_MODEL)
HASS_URL=$HASS_URL
HASS_TOKEN=$HASS_TOKEN
LIGHT_MODEL=ggml-org/Qwen3-ASR-$MODEL-GGUF:Q8_0
LLAMA_ARGS=
PORT=$PORT
ZEROCONF=assist-asr
LANGUAGE=en
BIAS=1
CONTEXT=1
WAKE_FORCE=auto
ALTERNATIVES=
RECORD_DIR=
EOF_ENV
  umask 022
fi

# --- what launchd runs -------------------------------------------------------------------------------------------
cat > "$APP/run-llama.sh" <<EOF_RUN
#!/bin/sh
# llama.cpp's server on Metal: everything on the GPU (-ngl 99), the audio encoder included.
set -a; . "$ENV"; set +a
exec "$LLAMA_SERVER" -hf "\$LIGHT_MODEL" -c 8192 -np 2 -ngl 99 --host 127.0.0.1 --port $LLAMA_PORT \$LLAMA_ARGS
EOF_RUN
cat > "$APP/run-wyoming.sh" <<EOF_RUN
#!/bin/sh
# the Wyoming server Home Assistant talks to, in front of llama-server
set -a; . "$ENV"; set +a
export BACKEND=llamacpp LLAMA_URL=http://127.0.0.1:$LLAMA_PORT
export BIAS_FILE="$APP/data/bias.json" CACHE_FILE="$APP/data/names-cache.json"
export COMMANDS_FILE="$APP/config/commands.txt" VOCAB_FILE="$APP/config/vocab.txt"
cd "$APP/app"
exec "$APP/venv/bin/python" -u server.py
EOF_RUN
chmod 755 "$APP/run-llama.sh" "$APP/run-wyoming.sh"

plist() {  # label, script, log name
  cat > "$AGENTS/$1.plist" <<EOF_PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>$1</string>
  <key>ProgramArguments</key><array><string>/bin/sh</string><string>$2</string></array>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
  <key>ThrottleInterval</key><integer>10</integer>
  <key>ProcessType</key><string>Interactive</string>
  <key>StandardOutPath</key><string>$LOGS/$3.log</string>
  <key>StandardErrorPath</key><string>$LOGS/$3.log</string>
</dict></plist>
EOF_PLIST
}
plist com.assist-asr.llama "$APP/run-llama.sh" llama
plist com.assist-asr.wyoming "$APP/run-wyoming.sh" wyoming
for s in com.assist-asr.llama com.assist-asr.wyoming; do
  unload "$s"
  launchctl bootstrap "gui/$UIDN" "$AGENTS/$s.plist"
done

# --- up? ---------------------------------------------------------------------------------------------------------
say "starting (the first start downloads the model, 1-2.5 GB: a few minutes)"
for i in $(seq 1 180); do
  curl -sf "http://127.0.0.1:$LLAMA_PORT/health" >/dev/null 2>&1 && break
  sleep 5
done
curl -sf "http://127.0.0.1:$LLAMA_PORT/health" >/dev/null 2>&1 || die "llama-server isn't up: $LOGS/llama.log"
# what llama.cpp finds (Metal shows as MTL0, e.g. "MTL0: Apple M2 Max"): asked, not guessed from the log
DEVICES=$("$LLAMA_SERVER" --list-devices 2>&1 | grep -E "^\s*[A-Za-z]+[0-9]+:" || true)
if printf '%s' "$DEVICES" | grep -q "MTL"; then
  say "llama.cpp runs on Metal: $(printf '%s' "$DEVICES" | grep MTL | head -1 | sed 's/^ *//')"
else
  say "llama.cpp found no Metal device here: it runs on the CPU (${DEVICES:-no devices listed})"
fi
for i in $(seq 1 60); do
  "$APP/venv/bin/python" -c "import socket; socket.create_connection(('127.0.0.1', $PORT), 2)" 2>/dev/null && break
  sleep 2
done
"$APP/venv/bin/python" -c "import socket; socket.create_connection(('127.0.0.1', $PORT), 2)" 2>/dev/null ||
  die "the Wyoming server isn't up: $LOGS/wyoming.log"
IP=$(ipconfig getifaddr en0 2>/dev/null || ipconfig getifaddr en1 2>/dev/null || echo "<this Mac's address>")
say "ready. In Home Assistant: Settings -> Devices & services -> Add integration -> Wyoming Protocol,"
say "host $IP, port $PORT (or accept the discovered \"assist-asr\")"
