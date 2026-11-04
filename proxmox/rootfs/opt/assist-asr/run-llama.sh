#!/bin/sh
# llama.cpp's server for assist-asr-llama.service: the GPU through Vulkan when one is passed in.
dev=$DEVICE
if [ "$dev" = auto ]; then
  if [ -e /dev/dri/renderD128 ]; then dev=gpu; else dev=cpu; fi
fi
# cpu: nothing on the GPU, not even the audio encoder or big batches (-ngl 0 alone leaves those on it)
if [ "$dev" = gpu ]; then gpu="-ngl 99"; else gpu="-ngl 0 -dev none --no-mmproj-offload"; fi
# Fusion off on the GPU: with it, a Radeon 680M (RADV) computes NaN (gpu-check.sh); it costs next to nothing
# elsewhere. Any value in the settings file keeps it off too; to have it on, delete it here.
[ "$dev" = gpu ] && [ -z "${GGML_VK_DISABLE_FUSION+x}" ] && export GGML_VK_DISABLE_FUSION=1
echo "assist-asr: $LIGHT_MODEL on the $dev"
exec /app/llama-server -hf "$LIGHT_MODEL" -c 8192 -np 2 $gpu --host 127.0.0.1 --port 8080 $LLAMA_ARGS
