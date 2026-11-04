# The root filesystem of the Proxmox CT template (proxmox/build-template.sh exports it; nothing here runs
# as a Docker container). Built on llama.cpp's Vulkan image: its llama-server has the CPU and Vulkan
# backends and needs that image's glibc (2.43: newer than Debian 13's), so the CT is that Ubuntu.
FROM ghcr.io/ggml-org/llama.cpp:server-vulkan

RUN apt-get update \
 && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      systemd systemd-sysv systemd-resolved python3 python3-venv ca-certificates iproute2 openssh-server \
 && rm -rf /var/lib/apt/lists/* \
 # llvmpipe would pass for a GPU when none is passed in (Vulkan on the CPU, far slower than the CPU backend)
 && rm -f /usr/share/vulkan/icd.d/lvp_icd*.json \
 # a fixed id for the render group: install.sh hands /dev/dri/renderD128 to gid 993
 && (getent group render >/dev/null && groupmod -g 993 render || groupadd -g 993 render) \
 && useradd --system --home-dir /var/lib/assist-asr --create-home --groups render,video assist-asr \
 && python3 -m venv /opt/assist-asr/venv \
 && /opt/assist-asr/venv/bin/pip install --no-cache-dir wyoming==1.10.2 zeroconf==0.151.5 pysilero-vad==3.4.0

COPY wyoming/server.py plugin/asr_bias.py /opt/assist-asr/
COPY config/commands.txt config/vocab.txt /etc/assist-asr/
COPY proxmox/rootfs/ /

# the settings file holds the HA token: root's only (systemd reads it before dropping to assist-asr)
RUN chmod 600 /etc/assist-asr/assist-asr.env \
 && systemctl enable assist-asr-llama assist-asr-wyoming systemd-networkd systemd-resolved \
 && systemctl mask systemd-networkd-wait-online.service \
 # the console (Proxmox: Console, cmode console) shows the services' output, not a login prompt
 && systemctl mask console-getty.service \
 && echo assist-asr > /etc/hostname \
 # the image finds llama.cpp's libraries through Docker's ENV, which a root filesystem does not keep
 && echo /app > /etc/ld.so.conf.d/llama-cpp.conf && ldconfig \
 # the image's own entrypoint is llama-server; a CT boots systemd
 && ln -sf /lib/systemd/systemd /sbin/init

ENTRYPOINT ["/sbin/init"]
