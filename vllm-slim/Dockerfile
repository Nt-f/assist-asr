# A small vLLM for CUDA: pip's vLLM on a slim Python, ~8 GB unpacked instead of the official image's 22.
# Kept: what a Qwen3-ASR server loads (torch, the CUDA runtime libraries it maps, triton + a C compiler for
# its launchers). Left out: FlashInfer's prebuilt kernels and JIT cache (~9 GB: attention here is vLLM's own
# FLASH_ATTN, and FlashInfer's sampler is switched off below), the CUDA toolkit (nothing of it is loaded),
# and, deleted after the install, what the server never loads (listed from /proc/*/maps of a running one):
# CUDA libraries for other jobs, profiling static libraries (gcc keeps its own: triton links launchers),
# CUDA headers, video decoding, DeepSeek's MLA kernels.
FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends gcc libc6-dev libgomp1 \
 && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir "vllm[audio]==0.31.0" \
 && cd /usr/local/lib/python3.12/site-packages \
 && rm -rf PyNvVideoCodec* tokenspeed_triton* tokenspeed_mla* \
      nvidia/cu13/lib/libnvrtc.alt.so.* nvidia/cu13/lib/libcusolverMg.so.* nvidia/cu13/lib/libnvvm.so.* \
      nvidia/cu13/lib/libnvperf_*.so nvidia/*/include nvidia/cu13/bin \
 && rm -f triton/backends/nvidia/lib/cupti/*_static.a nvidia/cu13/lib/*_static.a \
 && find / -xdev -type d -name tests -path "*site-packages/*" -prune -exec rm -rf {} + \
 && rm -rf /root/.cache
ENV VLLM_USE_FLASHINFER_SAMPLER=0
ENTRYPOINT ["vllm", "serve"]
