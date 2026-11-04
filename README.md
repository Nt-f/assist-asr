# assist-asr

Speech to text for Home Assistant voice assistants that hears **commands and your own names**
(your music library, your devices) far better than a speech model on its own. It runs Qwen3-ASR on
your GPU with [vLLM](https://github.com/vllm-project/vllm) and serves it to Home Assistant over the
Wyoming protocol, like `wyoming-whisper` does.

On top of the plain model it:

- **expects a command first.** While decoding, the first words are pushed towards what people say to an
  assistant (`play`, `shuffle`, `pause`, `turn off`, `set a timer`, …).
- **expects your names after a command.** After `play`/`shuffle` it listens for your Music Assistant
  library, and after `turn on`/`turn off` for your Home Assistant devices and areas. Everywhere else
  nothing is pushed, so questions and free speech come out as said.
- **checks names on the whole word.** If the name it heard isn't one of yours ("shuffle Dola"), the
  model scores your closest names against the same audio. The best one replaces it only if the audio
  supports it nearly as well and clearly better than any other name ("shuffle Zola").
- **gives an empty transcript on silence.** It doesn't invent "Okay." or "I'm not sure.", and it drops
  a leading wake word ("Alexa, pause" → "Pause"), for satellites that send the wake word along.

## Results

Measured 2026-11-04 on an AMD Ryzen 9 7900X (12 cores) with an RTX 3090. The models,
run three ways:

- **ours**: this project.
- **stock**: the same model in the same vLLM with nothing around it.
- **old**: `rhasspy/wyoming-whisper --stt-library qwen3-asr`, Qwen3-ASR 0.6B as ONNX int4 on the CPU.

The datasets:

- **names** (130): synthetic voices saying "play / shuffle / put on <name>" for 40 names from one Music
  Assistant library, and "turn on/off <device>" for 15 Home Assistant devices, each said 2 ways. Plus
  20 well-known names that aren't in the library, which must come out unchanged. Many of the library
  names are hard for a speech model: titles in other languages and unusual spellings.
- **rec** (22): real requests to a voice satellite, one speaker, wake word included. Small, but real.
- **corpus** (307): synthetic voices (8 accents, ages from child to elderly, whispers) saying everyday
  commands after 8 different wake words.

With **noise**, room noise is mixed in at 5 dB below the speech.

### Accuracy: requests exactly right (%)

| dataset | 1.7B ours | 1.7B stock | 0.6B ours | 0.6B stock | old (0.6B int4) |
|---|---:|---:|---:|---:|---:|
| names | **75.4** | 64.6 | 72.3 | 59.2 | 56.9 |
| names, noise | **76.9** | 63.8 | 70.8 | 56.9 | 56.2 |
| rec | **100.0** | 72.7 | 86.4 | 77.3 | 45.5 |
| rec, noise | **81.8** | 77.3 | 77.3 | 72.7 | 40.9 |
| corpus | **96.1** | 93.8 | 94.5 | 85.3 | 86.0 |
| corpus, noise | **96.1** | 93.2 | 92.8 | 83.1 | 85.7 |

Names right, on the names set: 1.7B 76% ours against 65% stock, 0.6B 74% against 62%, old 58%. The
20 names that aren't in the library were never swapped for one that is. The push did once mangle one
that stock got right, out of 40: "play some Mozart" came out as "mochot", pulled towards the library's
many "Mo…" names. That's the price of a push, and what `W_SLOT` trades.

### Speed: median time from the end of speech to the transcript

On 62 requests (rec, plus 40 from names). Ours is fed at speaking pace, as a satellite sends it, and
includes the name check when one runs. Light = llama.cpp with Q8 models; full = vLLM with the original
bf16 weights.

| where | 1.7B ours | 1.7B stock | 0.6B ours | 0.6B stock |
|---|---:|---:|---:|---:|
| GPU (RTX 3090), light | **66 ms** | 61 ms | **50 ms** | 42 ms |
| GPU (RTX 3090), full | 112 ms | 68 ms | 73 ms | 40 ms |
| CPU, 12 cores, light | 563 ms | 554 ms | **271 ms** | 230 ms |
| CPU, 12 cores, full | 1083 ms | 779 ms | 472 ms | 313 ms |
| CPU, 2 cores, light | 1271 ms | 1163 ms | **654 ms** | 516 ms |
| CPU, 2 cores, full | 1706 ms | 1138 ms | 706 ms | 427 ms |
| Radeon 610M iGPU (2 CUs), light | 1543 ms | 1635 ms | 633 ms | 612 ms |

The old setup (`wyoming-whisper`, Qwen3-ASR 0.6B ONNX int4, CPU only) took 303 ms on 12 cores and
1229 ms on 2.

Since these were measured, the name check got faster and more accurate (scored in one pass, about 40 ms
on vLLM) and a mangled wake word is forced: on vLLM, names now come out 83–85% exactly right (from 77–78%),
everyday commands 98–99% (from 96%), your noisy recordings 91% (from 82–86%).

Exactly right on the same 62 requests: 1.7B 81–84% ours against 66% stock, 0.6B 73–77% against 58–65%,
on every runtime. The Q8 models of the light version are as accurate as vLLM's bf16 here.

How to read these numbers:

- **Ours costs little on top of the bare model:** 5–10 ms on a GPU with the light version, more on vLLM
  (its plugin reads tokens back at each step). The name check adds about 150 ms (light on a CPU: up to
  about 0.8 s, one candidate after another) on the roughly one in three named requests whose name isn't in the list as heard.
- **On a CPU, the light 0.6B is the pick.** It's more accurate than either 1.7B without this project, and
  about a quarter of a second on a desktop CPU.
- **A small iGPU doesn't help.** The 610M was slower than 2 CPU cores. It does take the load off the CPU.
- **The old CPU setup** is fast on many cores but the least accurate, and slow on few.

Small datasets: one real speaker, and synthetic voices that sometimes mispronounce the names. Measure
on your own requests (below) before you trust any of this for your home.


## Pick a version

| Version | Runs on | Disk | 1.7B per request | 0.6B per request |
|---|---|---:|---:|---:|
| **Light, CUDA** (`compose.light.yml` + `compose.light.cuda.yml`) | NVIDIA GPU | 4.9 GB + model | 66 ms | 50 ms |
| **Full** (`docker-compose.yml`, vLLM) | NVIDIA GPU, ~6 GB free | 7.8 GB + model | 112 ms | 73 ms |
| **Light, CPU** (`compose.light.yml`) | any x86 or ARM CPU (a Pi too) | 1.2 GB + model | 563 ms (12 cores) | 271 ms (12 cores), 654 ms (2 cores) |
| **Light, Vulkan** (`+ compose.light.vulkan.yml`) | AMD or Intel GPU / iGPU | 1.2 GB + model | depends on the GPU | see below |
| **Proxmox CT** (`proxmox/`) | a Proxmox host: CPU, or its iGPU shared in | 0.2 GB template + model | as Light, CPU or Vulkan | |

Models: 1.0 GB (0.6B) or 2.5 GB (1.7B) for the light versions; 1.8 GB or 4.4 GB for the full one. The
speeds are medians of real requests on an RTX 3090 and a Ryzen 9 7900X (the Results above).

- **On an NVIDIA GPU, the light CUDA version is the pick**: faster than vLLM here, the same accuracy, and
  a smaller download. The full version is the one the accuracy tables above were measured with first;
  it remains for vLLM's own features.
- **Without one, the light CPU version with the 0.6B model**: about a quarter of a second on a desktop CPU.
- **An iGPU only helps when it's a capable one.** The Radeon 610M here (2 compute units) was slower than
  2 CPU cores: 0.6 s for the 0.6B. A Radeon 680M/780M (12 units) or a recent Intel iGPU should do far
  better, but that's unmeasured. Try both with the benchmark below.

Optional for all of them, but this is where the accuracy comes from: Home Assistant (your device and area
names) and Music Assistant (your music library).

## Set up

```sh
git clone https://github.com/Nt-f/assist-asr.git && cd assist-asr
cp .env.example .env
```

Edit `.env` and set:

- `HASS_URL`: Home Assistant's address, e.g. `http://192.168.1.10:8123`.
- `HASS_TOKEN`: a long-lived access token. In Home Assistant, open your profile, then **Security** →
  **Long-lived access tokens** → **Create token**.

Then start one version:

```sh
docker compose -f compose.light.yml -f compose.light.cuda.yml up -d   # light, NVIDIA GPU
docker compose -f compose.light.yml up -d                             # light, CPU
docker compose -f compose.light.yml -f compose.light.vulkan.yml up -d # light, AMD/Intel GPU or iGPU
docker compose up -d                                                  # full (vLLM, NVIDIA GPU)
```

For the light versions, `LIGHT_MODEL` in `.env` picks the model: `ggml-org/Qwen3-ASR-0.6B-GGUF:Q8_0` (the
default) or `ggml-org/Qwen3-ASR-1.7B-GGUF:Q8_0`. For the full version it's `ASR_MODEL`. The full version
builds its vLLM image (`vllm-slim/`) on first start, which takes a few minutes.

The first start downloads the model and takes a few minutes. Watch it with `docker compose logs -f`
(add the same `-f` files as above). It's ready once the log says `listening on 10300`.

In Home Assistant:

1. Go to **Settings → Devices & services → Add integration → Wyoming Protocol**.
2. Enter this machine's address and port `10300`.
3. Go to **Settings → Voice assistants**, open your assistant and pick it as **Speech-to-text**.

### The light version on asr-engine (experimental)

`compose.engine.yml` runs the light version on `engine/`: a small server on llama.cpp's own library
instead of llama-server. It encodes the audio once and scores all of a name check's candidates in one
batched pass. Measured on a 12-core CPU with the 0.6B:
- **Name check:** about 100 ms instead of about 550 ms with llama-server.
- **Accuracy:** names 78.5% exactly right instead of 74.6%.
- **Speed:** slowest 10% of requests 665 ms instead of 1006 ms, but the median plain request is about
  70 ms slower (392 against 319 ms).

```sh
docker compose -f compose.engine.yml up -d            # CPU; ENGINE_VARIANT=server-vulkan or server-cuda for a GPU
```

It's built against llama.cpp build b12098 (`engine/Dockerfile`); its image and source move together.

### On Proxmox

`proxmox/` builds a CT template and makes a container from it: the light version without Docker, with
systemd services. If the host has an AMD or Intel iGPU, it's shared with the container (not passed
through, so the host and other containers keep it, and there's none of the reset trouble of handing an
AMD iGPU to a VM).

In the Proxmox host's shell (nothing to install there, no Docker, no git):

```sh
bash -c "$(curl -fsSL https://raw.githubusercontent.com/Nt-f/assist-asr/main/proxmox/install.sh)"
```

It asks for Home Assistant's address and token, the model, and whether to share the GPU (or "advanced" for
the id, storage, bridge, cores, memory and disk). It then downloads the template from the latest release
(~250 MB, checked against its SHA-256), creates the container, and prints its address. Home Assistant also discovers it by itself (mDNS): it
shows up under Settings → Devices & services, ready to add. The container's Console in Proxmox shows the
services' output. Settings live in the container's `/etc/assist-asr/assist-asr.env`.

Without questions, e.g. for a script:

```sh
curl -fsSL https://raw.githubusercontent.com/Nt-f/assist-asr/main/proxmox/install.sh | bash -s -- --yes \
    --hass-url http://192.168.1.10:8123 --hass-token <token> [--model 1.7B] [--gpu no]
```

`--help` lists every option. The template is built by GitHub Actions
(`.github/workflows/proxmox-template.yml`) for each `v*` tag. To build one yourself, on any machine with
Docker: `proxmox/build-template.sh`, then `install.sh --template assist-asr-light-<date>-amd64.tar.zst` on
the host.

The template is llama.cpp's own base, Ubuntu 26.04. A Proxmox that doesn't know that release yet makes it
an "unmanaged" container, which configures its own network by DHCP. It was tested by booting it under
systemd on a PC with an iGPU passed in, but not yet on a Proxmox host.

## Settings

All of these go in `.env`. Restart with `docker compose up -d` after a change.

| Setting | Default | What it does |
|---|---|---|
| `ASR_MODEL` | `Qwen/Qwen3-ASR-1.7B` | Or `Qwen/Qwen3-ASR-0.6B`: smaller and less accurate (see the results). |
| `GPU_MEMORY` | `0.26` | Share of GPU memory vLLM takes. If vLLM says free memory is too low, lower it; around 6 GB is plenty. |
| `WYOMING_PORT` | `10300` | Port Home Assistant connects to. |
| `BIAS` | `1` | `0` turns off the command/name push and the name check. |
| `CONTEXT` | `1` | Also gives the model your names and `config/vocab.txt` as a prompt. |
| `W_CMD`, `W_SLOT` | `3.0`, `2.5` | Push strength towards commands and names. Higher fixes more, but risks hearing a name that wasn't said. |
| `SLOT_PRIOR`, `SLOT_MARGIN` | `8`, `3` | Name check: how far below what was heard a name may score and still win, and how far it must beat the next name. |
| `WAKE_FORCE` | `auto` | When a transcript starts with what sounds like a mangled wake word ("Lex allowed her"), decode again with the wake phrase forced and keep that if the audio supports it. `off` turns it off. |
| `ALTERNATIVES` | (off) | Experimental: an HA sensor (e.g. `sensor.asr_alternatives`) that gets what the request might also have been, for an LLM agent. See below. |
| `WAKE_WORDS` | `alexa\|…\|nabu` | Wake words dropped from the start of a transcript ("hey"/"ok" in front of one too: Hey Jarvis, Okay Nabu). |
| `ZEROCONF` | (off; `assist-asr` on Proxmox) | Announces the server by mDNS under this name, so Home Assistant discovers it. Only where the server is on the LAN itself (the Proxmox CT); not from Docker's default network. |
| `RECORD_DIR` | `/data/rec` | Keeps every request's audio in `data/rec`, for building your own test set. Leave empty to turn it off. |

The files in `config/` are reread every 10 minutes:

- `commands.txt`: how requests start, one per line. `play > music` means a name from your music follows
  "play"; `turn off > home` means a device or area follows "turn off".
- `vocab.txt`: words for the prompt (`CONTEXT=1`).
- `music.txt`, `home.txt` (create them if you need them): more names, one per line, added to what Home
  Assistant and Music Assistant provide. Without Home Assistant these are the only names.

`docker compose logs wyoming` shows every request with where its time went, e.g.
`1.9 s audio, 1536 ms speech, 99 ms (vad 0, decode 99, name check 0): 'Alexa, play X.' -> 'Play X.'`.
A `name check` line shows the scores whenever a name was weighed against your list.

## Alternatives for your LLM (experimental)

Most requests aren't matched by Home Assistant's own sentences and go to the conversation agent, often an
LLM. With `ALTERNATIVES=sensor.asr_alternatives`, each request also writes what it might have been to that
sensor, before the transcript goes to HA. The alternatives are the names that came close in the name check,
and, where a word was a close call, the transcript decoded again from that word with the other choice.
Requests with something unsure take about 100 ms longer.

The sensor's `alternatives` attribute has up to three, and `transcript` has what was sent. In your LLM agent's
prompt (its instructions are a template):

```jinja
{% set alts = state_attr('sensor.asr_alternatives', 'alternatives') %}
{% if alts %}Speech recognition was unsure. The request may also have been: {{ alts | join(' / ') }}.
Prefer the reading that names a device, area or music that exists here.{% endif %}
```

There is one sensor for all satellites, so two requests at the same moment can overwrite each other. It's
off by default until this has been tried with a real LLM.

## vLLM on a CPU

The full version also has a CPU build of vLLM (`docker compose -f docker-compose.yml -f docker-compose.cpu.yml up -d`,
`CPU_CORES` to limit it). The light CPU version is faster (271 ms against 472 ms for the 0.6B on 12 cores)
and 10 GB smaller, so this is only for comparisons.

## Benchmark

`bench/` compares speech-to-text systems on sets of short requests. It works for any Wyoming server, not
only this one.

```sh
python3 -m venv .venv && .venv/bin/pip install -r bench/requirements.txt
```

### A dataset

A dataset is a folder of 16 kHz mono WAV files with a `manifest.jsonl`, one line per clip:

```json
{"file": "0001.wav", "text": "Alexa, shuffle Zola.", "name": "Zola"}
```

`name` is optional. When it's there, the report also counts whether that name came out right. Instead of
a manifest, you can put a `<clip>.txt` with the words said next to each `<clip>.wav`. To convert other
audio: `ffmpeg -i in.m4a -ar 16000 -ac 1 -c:a pcm_s16le out.wav`.

### Your own recordings (the best test)

1. With `RECORD_DIR` set (the default), use your voice assistant as usual. Every request lands in
   `data/rec`.
2. Label them:

   ```sh
   .venv/bin/python bench/label.py data/rec --out data/sets/mine
   ```

   It shows what was heard. Press Enter if that's right, or type what you actually said. Type `p` to play
   the clip. Add ` | <name>` after the text when the request contained one of your names, e.g.
   `Alexa, shuffle Zola | Zola`.

### Synthetic names

You can generate a set of "play <your music>" and "turn off <your device>" requests in eight synthetic
voices with [OmniVoice](https://github.com/k2-fsa/OmniVoice). It needs a GPU with about 4 GB free, so
stop vLLM first if memory is tight.

```sh
uv venv -p 3.11 omnivoice && VIRTUAL_ENV=$PWD/omnivoice uv pip install omnivoice==0.2.1 torch==2.9.1 torchaudio==2.9.1 transformers==5.16.1
omnivoice/bin/python bench/synth.py data/sets/names --names data/bias.json
```

`data/bias.json` holds the names the server collected. The set also includes 20 well-known names that
aren't in your lists ("play Taylor Swift"); those must come out unchanged. Any list of sentences works
too: `bench/synth.py OUT --texts sentences.txt`.

### Run, report, compare

```sh
# this server against the bare model in the same vLLM, clean and with room noise 5 dB under the speech
.venv/bin/python bench/bench.py run --tag mine --noise 5 --dataset data/sets/mine \
    --system ours=wyoming://127.0.0.1:10300 --system stock=vllm://127.0.0.1:8100
.venv/bin/python bench/bench.py report bench/results/*.json      # tables in Markdown
.venv/bin/python bench/bench.py diff bench/results/mine.json ours stock   # the clips they disagree on

# both model sizes, one after the other (restarts vLLM with each, then goes back to .env's)
PY=.venv/bin/python bench/compare-models.sh --noise 5 data/sets/mine data/sets/names

# on the CPU, each model on all cores and on two (a second stack on ports 10310/8110, beside the GPU one)
PY=.venv/bin/python CORES="all 0-1" bench/speed.sh data/sets/mine
```

Add `--realtime` to `bench.py run`, or `BENCH_ARGS=--realtime` to the scripts, to stream audio at speaking
pace the way a satellite does. It's slower, but it's the fair way to time a server that works while the
audio is still arriving, like this one does.

Any other Wyoming speech-to-text server can be added as another `--system name=wyoming://host:port`,
and a bare llama.cpp server with Qwen3-ASR as `name=llamacpp://host:port`.
Scores:

- **exact**: the share of requests whose words all came out right. Case, punctuation, a leading wake word
  and numbers versus words ("72" or "seventy two") don't count against it.
- **WER**: word error rate.
- **names**: the share of named requests where the name came out right.
- **ms**: median time from the end of the audio to the transcript.

Timings are only comparable when nothing else heavy runs at the same time. On a CPU, two runs side by
side also share memory bandwidth, so run them one after another, as the scripts do.

## How it works

```
Home Assistant ──Wyoming──▶ wyoming/server.py ──HTTP──▶ vLLM: Qwen3-ASR + plugin/asr_bias.py
                              │ VAD, wake word, name check     │ pushes tokens by position
                              └─ data/bias.json ◀── HA + Music Assistant names, config/*.txt
```

- `wyoming/server.py`:
  - Finds the speech while the audio streams in (Silero VAD), and sends only that.
  - Asks vLLM for the transcript and the probability of each word.
  - Runs the name check and drops the wake word.
  - Every 10 minutes, writes your names and the command phrases to `data/bias.json`.
- `plugin/asr_bias.py`: a vLLM logits processor. While the model writes the transcript, it reads what's
  written so far to know where it is: wake word, command, or the name after a command. It then raises
  the score of the next tokens that continue a phrase of the right list. It matches on letters, not
  tokens, so it doesn't matter how the model splits a name into tokens. It only looks among the 40
  tokens the model already rates highest, so it can't put in a name the audio doesn't support.

  The name check scores candidate names on the whole word: names only differ after their first letter,
  and one pass can't go back once it has written it. On vLLM each candidate is scored in a single forward
  pass with the candidate text as part of the prompt (`prompt_logprobs`, through a chat template the
  server sends: vLLM runs with `--trust-request-chat-template`), all in parallel, with only the compared
  names as context: about 40 ms a check. On llama.cpp each candidate is generated under a grammar, one
  after another: fast on a GPU, up to about 0.8 s on a CPU.
- HA's names and Music Assistant's library are refreshed in the background every 10 minutes and kept in
  `data/names-cache.json`, so no request waits for HA, and a restart has your names even with HA down.
- When a request doesn't ask for it, the plugin leaves it alone. That's why `vllm://` in the benchmark
  is the bare model.

## Files

```
docker-compose.yml   the full version: vLLM (built from vllm-slim/) + the Wyoming server
compose.official.yml   ... with vLLM's official image instead (22 GB)
docker-compose.cpu.yml ... with vLLM's CPU build
compose.light.yml    the light version: llama.cpp + the Wyoming server, CPU
compose.light.cuda.yml, compose.light.vulkan.yml   ... on an NVIDIA GPU, on an AMD/Intel GPU or iGPU
vllm-slim/           the small vLLM image (what a Qwen3-ASR server loads, nothing else)
proxmox/             install.sh (run on the host), build-template.sh, the CT's services (rootfs/)
engine/              asr-engine: llama.cpp's library, one batched pass per name check (compose.engine.yml)
.env.example         settings (copy to .env)
config/              commands.txt, vocab.txt, optional music.txt and home.txt
plugin/asr_bias.py   the vLLM plugin
wyoming/             the Wyoming server and its Dockerfile
bench/               bench.py (run, report, diff), label.py, synth.py, compare-models.sh, speed.sh
data/                bias.json, rec/ (recordings), sets/ (datasets); not in git
```

MIT (LICENSE) for this project's own code. The models (Qwen3-ASR, OmniVoice), vLLM and llama.cpp have their own licences.
