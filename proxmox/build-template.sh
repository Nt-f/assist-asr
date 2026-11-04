#!/bin/sh
# Builds the Proxmox CT template: dist/assist-asr-light-<date>-amd64.tar.zst (a root filesystem, ~0.5 GB).
# Runs anywhere with Docker; GitHub Actions builds it for each release (.github/workflows/proxmox-template.yml),
# and proxmox/install.sh downloads that; install.sh --template takes one built here instead.
# ARCH=arm64 for ARM (needs Docker's QEMU emulation on an x86 machine). NAME overrides the file name.
set -e
cd "$(dirname "$0")/.."
ARCH=${ARCH:-amd64}
NAME=${NAME:-assist-asr-light-$(date +%Y%m%d)-$ARCH}
mkdir -p dist
docker build --platform "linux/$ARCH" -f proxmox/Dockerfile -t "assist-asr-ct:$ARCH" .
cid=$(docker create --platform "linux/$ARCH" "assist-asr-ct:$ARCH")
trap 'docker rm -f "$cid" >/dev/null 2>&1; rm -f "dist/$NAME.tar"' EXIT
docker export "$cid" > "dist/$NAME.tar"
# Docker's marker file: systemd would think it runs in Docker
tar --delete -f "dist/$NAME.tar" .dockerenv 2>/dev/null || true
zstd -q -19 -T0 -f "dist/$NAME.tar" -o "dist/$NAME.tar.zst"
ls -lh "dist/$NAME.tar.zst"
echo "Next, on the Proxmox host: install.sh --template $NAME.tar.zst"
