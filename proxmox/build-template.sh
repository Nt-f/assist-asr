#!/bin/sh
# Builds the Proxmox CT template: dist/assist-asr-light-<date>-amd64.tar.zst (a root filesystem, ~0.5 GB).
# Runs anywhere with Docker; GitHub Actions builds it for each release (.github/workflows/proxmox-template.yml),
# and proxmox/install.sh downloads that; install.sh --template takes one built here instead.
# ARCH=arm64 for ARM (needs Docker's QEMU emulation on an x86 machine). NAME overrides the file name, VERSION
# what the CT reports as its release (install.sh --update).
set -e
cd "$(dirname "$0")/.."
ARCH=${ARCH:-amd64}
NAME=${NAME:-assist-asr-light-$(date +%Y%m%d)-$ARCH}
mkdir -p dist
trap 'rm -f "dist/$NAME.tar"' EXIT
# The root filesystem straight to a tar (BuildKit): no container, so no /.dockerenv (systemd would think it
# runs in Docker) and no editing the archive (tar --delete corrupted it, on GitHub's runner).
docker build --platform "linux/$ARCH" -f proxmox/Dockerfile --build-arg "VERSION=${VERSION:-$NAME}" --output "type=tar,dest=dist/$NAME.tar" .
tar tf "dist/$NAME.tar" >/dev/null || { echo "the archive does not read cleanly"; exit 1; }
if tar tf "dist/$NAME.tar" | grep -qx '\(\./\)\?\.dockerenv'; then echo "the archive has /.dockerenv"; exit 1; fi
zstd -q -19 -T0 -f "dist/$NAME.tar" -o "dist/$NAME.tar.zst"
zstd -dc "dist/$NAME.tar.zst" | tar tf - >/dev/null
ls -lh "dist/$NAME.tar.zst"
echo "Next, on the Proxmox host: install.sh --template $NAME.tar.zst"
