#!/bin/sh
# asr-engine's start: the model and its audio encoder downloaded once from Hugging Face into $MODELS
# (MODEL is repo:quant as for llama-server -hf, e.g. ggml-org/Qwen3-ASR-0.6B-GGUF:Q8_0), then the engine.
set -e
MODEL=${MODEL:-ggml-org/Qwen3-ASR-0.6B-GGUF:Q8_0}
MODELS=${MODELS:-/models}
repo=${MODEL%%:*} quant=${MODEL##*:}
name=$(basename "$repo" | sed 's/-GGUF$//')
mkdir -p "$MODELS"
for f in "$name-$quant.gguf" "mmproj-$name-$quant.gguf"; do
  if [ ! -s "$MODELS/$f" ]; then
    echo "asr-engine: downloading $repo/$f"
    curl -fL --retry 3 -o "$MODELS/$f.part" "https://huggingface.co/$repo/resolve/main/$f"
    mv "$MODELS/$f.part" "$MODELS/$f"
  fi
done
exec /app/asr-engine -m "$MODELS/$name-$quant.gguf" --mmproj "$MODELS/mmproj-$name-$quant.gguf" "$@"
