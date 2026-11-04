#!/bin/sh
# llama.cpp's server for assist-asr-llama.service: the GPU through Vulkan when one is passed in.
dev=$DEVICE
if [ "$dev" = auto ]; then
  if [ -e /dev/dri/renderD128 ]; then dev=gpu; else dev=cpu; fi
fi
if [ "$dev" = gpu ]; then ngl=99; else ngl=0; fi
echo "assist-asr: $LIGHT_MODEL on the $dev"
exec /app/llama-server -hf "$LIGHT_MODEL" -c 4096 -np 1 -ngl "$ngl" --host 127.0.0.1 --port 8080 $LLAMA_ARGS
