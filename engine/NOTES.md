# asr-engine: what was measured and why it looks like this

Measured 2026-11-04 on a Ryzen 9 7900X (12 cores), Qwen3-ASR-0.6B Q8, llama.cpp build b12098.

## Why the engine exists

The name check scores 4–6 candidate transcripts on the same audio. llama-server can't do that cheaply:

| approach on llama-server (CPU) | 6 candidates | why |
|---|---:|---|
| one after another, one slot, grammar-forced | 650–950 ms | each candidate is ~5 generated tokens at ~25 ms |
| parallel slots | ~1000 ms | each extra slot encodes the audio again (95 prompt tokens each) |
| parallel slots after `/slots/N?action=restore` of slot 0 | ~1000 ms | the restore takes 8 ms, but the server doesn't match the restored audio and encodes it again |
| parallel slots already holding this audio | 330 ms | batching works, but real requests always bring new audio |
| branch and bound on first-token bounds | no gain | first tokens are shared by many names; the bounds are far too loose |

llama-server also has no prompt-token logprobs (`echo` on `/completion` returns none; `/v1/completions` doesn't
take audio). So `engine.cpp` does it on the library: the audio is encoded once into sequence 0, each candidate
gets a copy (`llama_memory_seq_cp`, free in a unified KV cache), and all tails go through one `llama_decode`
with logits on every token. Result: **67–79 ms for 6 candidates**, ~100 ms for the whole name check
(llama-server: ~550 ms), and slightly better scores (no grammar-chosen tokenisations: "Sasha" -7.2 instead
of -12.6).

## The open gap: plain decodes are ~60–70 ms slower than llama-server

| per request (new audio, context cached) | llama-server | asr-engine |
|---|---:|---:|
| prompt: audio encode + ~74 positions | 140–160 ms | ~195 ms |
| of which the audio embeddings' decode | 63–70 ms | 95–99 ms |
| per generated token | 16.5–17.5 ms | ~17.8 ms |
| whole decode, 12 different clips | 248 ms | 312 ms |

Ruled out (no change): thread count (6/12/24), an explicit thread pool (`llama_attach_threadpool`), unified vs
non-unified KV, number of sequences (2/8/16), context size (2048–8192), ubatch (128–1024), the top-k code
(heap instead of a full sort), weight repacking (neither repacks: both map the GGUF), flash attention
(both use it).

Fixed along the way: the system turn (HA's names + vocabulary, ~450 tokens) was re-read on every request;
now the leading text chunk is kept when it's unchanged (`g_pre`), as llama-server's prompt cache does.

Likely cause of the rest: llama-server adds the encoded audio to the same batch as the following text
(`add_mtmd_chunk` in `tools/server/server-context.cpp`) instead of decoding it on its own, as the public
`mtmd_helper_eval_chunk_single` does, and its embedding batches go through `common_batch`. Matching that
means code against server internals, not the public API.

## Net effect (light version, CPU, 0.6B, names set)

llama-server → engine:
- median 319 → 392 ms
- mean 510 → 454 ms
- 90th percentile 1006 → 665 ms
- names exactly right 74.6 → 78.5%
- HA understood names as said 83 → 92%

Better accuracy and a better worst case, with a slower typical request. That's why `compose.light.yml`
(llama-server) stays the default and `compose.engine.yml` is the option.
