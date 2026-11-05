#!/usr/bin/env python3
"""Wyoming speech-to-text in front of Qwen3-ASR on vLLM, biased towards a home assistant's words.

Per request (audio-stop):
1. Silero VAD: under MIN_SPEECH_MS of speech -> empty transcript (Qwen3-ASR never answers nothing:
   on silence it says "Okay." or reads its context back). Else only the speech + PAD_MS is sent.
2. Decode (chat completions, for logprobs) with asr_bias.py in vLLM pushing tokens by position:
   wake word, then command phrases (config/commands.txt), then after "play"/"shuffle" Music
   Assistant's library and after "turn on/off" HA's entity and area names.
3. Name check: a name after such a command that is not in its list is weighed against the
   closest names of the list on the same audio (the decoder forced to write each); the best one
   replaces it when the audio supports it nearly as well and clearly better than the rest.
4. A leading wake word is stripped ("Alexa, pause" -> "Pause"), for "Wake word in the request".

Environment (defaults):
  VLLM_URL (http://vllm:8000), MODEL (qwen3-asr), LANGUAGE (en), PORT (10300)
  HASS_URL, HASS_TOKEN: HA's names and Music Assistant's library (both optional)
  HASS_DOMAINS (entity domains whose names count), CONTEXT_REFRESH (600 s)
  CONTEXT (1: HA names + config/vocab.txt as Qwen3-ASR's context prompt, CONTEXT_MAX_CHARS 1500)
  BIAS (1), BIAS_FILE (/data/bias.json, where vLLM's plugin reads it), COMMANDS_FILE, VOCAB_FILE
  W_CMD (3.0), W_SLOT (2.5): logit added on a command's / a name's path
  SLOT_PRIOR (8), SLOT_MARGIN (3): the name check's thresholds, in nats; NAME_CANDIDATES (6, llama.cpp 4)
  BACKEND (vllm): or llamacpp, with LLAMA_URL (http://llama:8080): compose.light.yml
  WAKE_FORCE (auto: a mangled wake word decoded again with it forced; off), WAKE_PRIOR (8), WAKE_PHRASES
  ALTERNATIVES (unset; e.g. sensor.asr_alternatives: close alternatives for an LLM agent), ALT_RANGE (6)
  WAKE_WORDS (alexa|alexis|alex|amazon|computer|echo|ziggy|jarvis|mycroft|nabu; "hey"/"ok" before one is dropped too)
  VAD_THRESHOLD (0.3), MIN_SPEECH_MS (150), PAD_MS (300), VAD_BOOST (30: dB a quiet request is raised by
  for the VAD alone; 0 off); the short path under SURE_MS (300): SHORT_MS (96), SHORT_THRESHOLD (0.1),
  SHORT_LP (-2); LLAMA_WAIT (20 s for a llama-server restarting)
  ZEROCONF (unset; a name, e.g. assist-asr): announce by mDNS for HA's discovery (needs the zeroconf package)
  RECORD_DIR (unset): keep each request's audio + transcript there (bench/label.py turns them
  into a dataset)
"""

import array
import asyncio
import collections
import base64
import http.client
import threading
import urllib.error
import urllib.parse
import difflib
import io
import json
import logging
import math
import os
import re
import time
import urllib.request
import uuid
import wave
from concurrent.futures import ThreadPoolExecutor

import asr_bias

from pysilero_vad import SileroVoiceActivityDetector
from wyoming.asr import Transcribe, Transcript
from wyoming.audio import AudioChunk, AudioChunkConverter, AudioStart, AudioStop
from wyoming.event import Event
from wyoming.info import AsrModel, AsrProgram, Attribution, Describe, Info
from wyoming.server import AsyncEventHandler, AsyncServer

BACKEND = os.environ.get("BACKEND", "vllm")  # vllm; engine (engine/: llama.cpp's library, CPU/Vulkan/CUDA); llamacpp (llama-server)
LLAMA = BACKEND in ("llamacpp", "engine")  # the llama.cpp path (prompts, restarts) is shared
ENGINE = BACKEND == "engine"
VLLM_URL = os.environ.get("VLLM_URL", "http://vllm:8000").rstrip("/")
LLAMA_URL = os.environ.get("LLAMA_URL", "http://llama:8080").rstrip("/")
MODEL = os.environ.get("MODEL", "qwen3-asr")
LANGUAGE = os.environ.get("LANGUAGE", "en")
WAKE = os.environ.get("WAKE_WORDS", "alexa|alexis|alex|amazon|computer|echo|ziggy|jarvis|mycroft|nabu")
HASS_URL = os.environ.get("HASS_URL", "").rstrip("/")
HASS_TOKEN = os.environ.get("HASS_TOKEN", "")
HASS_DOMAINS = set(os.environ.get(
    "HASS_DOMAINS",
    "light,switch,fan,cover,climate,media_player,scene,script,lock,vacuum,input_boolean,timer",
).split(","))
CONTEXT_MAX = int(os.environ.get("CONTEXT_MAX_CHARS", "1500"))
CONTEXT_REFRESH = float(os.environ.get("CONTEXT_REFRESH", "600"))
PORT = int(os.environ.get("PORT", "10300"))
ZEROCONF = os.environ.get("ZEROCONF", "")  # a name: announce this server by mDNS for HA's discovery
VOCAB_FILE = os.environ.get("VOCAB_FILE", "/config/vocab.txt")
VAD_THRESHOLD = float(os.environ.get("VAD_THRESHOLD", "0.3"))
MIN_SPEECH_MS = int(os.environ.get("MIN_SPEECH_MS", "150"))
PAD_MS = int(os.environ.get("PAD_MS", "300"))
# Under the gate: a whole command only (config/commands.txt: "Stop.", "Next."), the model sure of it (SHORT_LP
# nats), with at least SHORT_MS of windows at SHORT_THRESHOLD; 0 off.
SHORT_MS = int(os.environ.get("SHORT_MS", "96"))
SHORT_THRESHOLD = float(os.environ.get("SHORT_THRESHOLD", "0.1"))
SURE_MS = int(os.environ.get("SURE_MS", "300"))  # less speech than this goes the short path's way too
SHORT_LP = float(os.environ.get("SHORT_LP", "-2"))
VAD_BOOST = float(os.environ.get("VAD_BOOST", "30"))  # dB at most a quiet request is raised by for the VAD; 0: off
RECORD_DIR = os.environ.get("RECORD_DIR", "")
USE_CONTEXT = os.environ.get("CONTEXT", "1") != "0"
BIAS_FILE = (os.environ.get("BIAS_FILE", "/data/bias.json")
             if os.environ.get("BIAS", "1") != "0" else "")
COMMANDS_FILE = os.environ.get("COMMANDS_FILE", "/config/commands.txt")
W_CMD = float(os.environ.get("W_CMD", "3.0"))
W_SLOT = float(os.environ.get("W_SLOT", "2.5"))
SLOT_PRIOR = float(os.environ.get("SLOT_PRIOR", "8"))
SLOT_MARGIN = float(os.environ.get("SLOT_MARGIN", "3"))
# Names weighed in a check: on vLLM they are scored in parallel, on llama.cpp one after another (~25 ms a
# token on a CPU), hence fewer there.
# A mangled wake word ("Lex allowed her" for "Alexa, louder"): decoded again with the wake phrase forced
# first, kept when the audio supports it within WAKE_PRIOR nats (auto) or never (off).
WAKE_FORCE = os.environ.get("WAKE_FORCE", "auto")
WAKE_PRIOR = float(os.environ.get("WAKE_PRIOR", "8"))
# How each wake word is said in full ("hey"/"okay" included), for forcing it.
WAKE_PHRASES = [p for p in os.environ.get(
    "WAKE_PHRASES", "Alexa|Computer|Echo|Amazon|Ziggy|Hey Jarvis|Hey Mycroft|Okay Nabu").split("|") if p]
# Alternatives for an LLM conversation agent (off unless set): the transcripts it might also have been, in
# an HA sensor its prompt can read (README "Alternatives for your LLM"). ALT_RANGE: how close (nats).
ALTERNATIVES = os.environ.get("ALTERNATIVES", "")  # e.g. sensor.asr_alternatives
ALT_RANGE = float(os.environ.get("ALT_RANGE", "6"))
NAME_EARLY_STOP = os.environ.get("NAME_EARLY_STOP", "1") != "0"  # llama.cpp: stop once no name left can win
NAME_CANDIDATES = int(os.environ.get("NAME_CANDIDATES", "4" if BACKEND == "llamacpp" else "6"))  # the engine scores them at once
# Words between a command and its name that are not part of the name ("play some Zola",
# "turn off the lamp"); a name may still start with them ("The Beatles").
FILLERS = {"*": ["the", "my", "some", "a"],
           "music": ["music by", "songs by", "something by", "the artist", "the album", "the playlist",
                     "the song", "artist", "album", "playlist", "song", "the band"],
           "home": ["all the", "all"]}

# "Alexa, stop." -> "stop."; "Alexa." alone -> "" (HA then ends the run as "nothing said").
# The comma/space after the word is required, so "Alexander" stays.
WAKE_RE = re.compile(rf"^\s*(?:(?:hey|ok(?:ay)?)[\s,.!]+)?(?:{WAKE})(?:[\s,.!?;:]+|$)", re.I)

log = logging.getLogger("qwen-asr-wyoming")
_context = {"text": "", "at": 0.0}


def strip_wake(text: str) -> str:
    out = WAKE_RE.sub("", text, count=1).strip()
    if out != text.strip() and out:
        out = out[0].upper() + out[1:]
    return out


class StreamVad:
    """Silero VAD fed while the audio streams in, so the speech is found by the time it ends
    (all at once at the end: ~150 sequential 32 ms windows for 5 s, all on the waiting time)."""

    def __init__(self):
        self.vad = SileroVoiceActivityDetector()
        self.n = self.vad.chunk_bytes()  # 512 samples, 32 ms
        self.pos = 0
        self.hits = []
        self.weak = []  # windows at SHORT_THRESHOLD or more
        self.quiet = False  # span(): its speech found only boosted
        self.short = False  # span(): under the gate, but enough for a short command (SHORT_MS)

    def feed(self, pcm) -> None:
        while self.pos + self.n <= len(pcm):
            p = self.vad(bytes(pcm[self.pos:self.pos + self.n]))
            if p >= VAD_THRESHOLD:
                self.hits.append(self.pos)
            if p >= SHORT_THRESHOLD:
                self.weak.append(self.pos)
            self.pos += self.n

    def span(self, total: int, pcm=None):
        """(start, end) in bytes of 16 kHz s16 mono or None below MIN_SPEECH_MS, and speech ms.
        With pcm, a quiet request is looked at again at a normal level (boosted)."""
        ms = len(self.hits) * self.n // 32
        if ms < SURE_MS and pcm is not None and VAD_BOOST > 0:
            # only where the gate would say "nothing": a request it passes keeps its span (boosted, the
            # span grew and 3 of 18 one-word clips at -30 dB came out worse)
            self.boosted(pcm)
            ms = len(self.hits) * self.n // 32
        pad = PAD_MS * 32
        if ms < SURE_MS and ms >= MIN_SPEECH_MS:
            # Just over the gate: as much noise as speech. White noise at -45..-50 dBFS got 160-192 ms, and
            # with the context in the prompt came out "Alexa, set a timer." (read back); a quiet "Stop" at
            # 256 ms cut to its span came out "Um.". The short path's rules for these too.
            self.short = True
            return (0, total), ms
        if ms < MIN_SPEECH_MS:
            # A command cut short (the replay after the wake word starting in it: "Stop" as "op") or said
            # very quietly: Silero gives such a fragment 0-96 ms (issue #1), where plain Qwen3-ASR still hears
            # "Stop". Decoded anyway, and the handler keeps only a whole command the model is sure of.
            if SHORT_MS and len(self.weak) * 32 >= SHORT_MS:
                self.short = True
                return (0, total), ms
            return None, ms
        if self.quiet:
            # Found only boosted, the span is too tight: quiet one-word clips cut to it came out "Um." for
            # "Stop", "We." for "Resume", which the whole request (as plain Qwen3-ASR gets it) did not.
            return (0, total), ms
        return (max(0, self.hits[0] - pad), min(total, self.hits[-1] + self.n + pad)), ms

    def boosted(self, pcm) -> None:
        """Silero's probabilities fall with the level: an Echo's own recordings turned down 30 dB (where
        an Echo Dot 2's requests arrive, issue #1) kept 32-512 ms of 1.6-2.2 s of speech at 0.3, so "Stop",
        "Next", "Pause" came back empty. Brought up to -6 dBFS (at most VAD_BOOST dB) the same clips kept
        all but ~5 % of it, even at -40 dB, while noise alone (white/brown -70..-40 dBFS, the rooms' own
        tails, hum) stayed at 0 ms. Only for requests the gate fails whose peak is under -12 dBFS: ~30 ms
        for 5 s."""
        a = array.array("h", bytes(pcm[:len(pcm) // 2 * 2]))
        n = self.n // 2
        peaks = sorted(max(max(a[i:i + n]), -min(a[i:i + n])) for i in range(0, len(a) - n + 1, n))
        if len(peaks) < 2:
            return
        peak = max(peaks[-2], 1)  # the second loudest window: one click doesn't set the gain
        gain_db = min(VAD_BOOST, -6 - 20 * math.log10(peak / 32768))
        if gain_db < 6:
            return
        g = 10 ** (gain_db / 20)
        b = array.array("h", (max(-32768, min(32767, int(x * g))) for x in a)).tobytes()
        vad, hits, weak = SileroVoiceActivityDetector(), [], []
        for pos in range(0, len(b) - self.n + 1, self.n):
            p = vad(b[pos:pos + self.n])
            if p >= VAD_THRESHOLD:
                hits.append(pos)
            if p >= SHORT_THRESHOLD:
                weak.append(pos)
        if len(weak) > len(self.weak):
            self.weak = weak
        if len(hits) > len(self.hits):
            log.info("quiet request (+%.0f dB): %d ms of speech found, %d at its own level",
                     gain_db, len(hits) * 32, len(self.hits) * 32)
            self.hits = hits
            self.quiet = True


def echoes_context(text: str, ctx: str) -> bool:
    """The model read (part of) its context back: 3+ of its entries in the transcript."""
    if not ctx:
        return False
    t = " " + re.sub(r"[^a-z0-9]+", " ", text.lower()) + " "
    order = [e for e in (re.sub(r"[^a-z0-9]+", " ", e.lower()).strip() for e in re.split(r"[,.]", ctx)) if e]
    entries = set(order)
    found = sum(1 for e in entries if f" {e} " in t)
    if found >= 3 and len(t.split()) > 8:
        return True
    # Or short: three entries in a row as the context has them ("Home Assistant, Drop In, Music Assistant."),
    # which no request says.
    return any(f" {order[i]} {order[i + 1]} {order[i + 2]} " in t for i in range(len(order) - 2))


def _hass(method: str, path: str, body=None):
    req = urllib.request.Request(
        HASS_URL + path, method=method,
        data=json.dumps(body).encode() if body is not None else None,
        headers={"Authorization": f"Bearer {HASS_TOKEN}", "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=5) as r:
        return r.read().decode()


def ha_names() -> list:
    names, seen = [], set()
    try:
        areas = _hass("POST", "/api/template",
                      {"template": "{{ areas() | map('area_name') | join('\n') }}"})
        names += [a for a in areas.splitlines() if a.strip()]
    except Exception as e:  # areas are a bonus; entity names matter more
        log.warning("HA areas: %s", e)
    for s in json.loads(_hass("GET", "/api/states")):
        if s["entity_id"].split(".")[0] in HASS_DOMAINS:
            n = (s.get("attributes") or {}).get("friendly_name")
            if n:
                names.append(n)
    out = []
    for n in names:
        k = n.strip().lower()
        if k and k not in seen:
            seen.add(k)
            out.append(n.strip())
    return out


def context_of(names: list) -> str:
    text = ", ".join(names)
    return text[:CONTEXT_MAX].rsplit(",", 1)[0] if len(text) > CONTEXT_MAX else text


def music_library() -> list:
    """Names in Music Assistant's library (artists, albums, playlists, tracks, radio) through HA."""
    entry = _hass("POST", "/api/template", {"template":
        "{{ config_entry_id(integration_entities('music_assistant') | first | default('')) or '' }}"}).strip()
    if not entry:
        return []
    out = []
    for kind in ("artist", "album", "playlist", "track", "radio"):
        offset = 0
        while True:
            r = json.loads(_hass("POST", "/api/services/music_assistant/get_library?return_response",
                                 {"config_entry_id": entry, "media_type": kind, "limit": 500,
                                  "offset": offset}))
            items = r["service_response"]["items"]
            out += [i["name"] for i in items if i.get("name")]
            if len(items) < 500:
                break
            offset += 500
    return out


def read_commands():
    cmds, openers = [], {}
    try:
        with open(COMMANDS_FILE) as f:
            for l in f:
                l = l.split("#")[0].strip()
                if ">" in l:
                    phrase, kind = (x.strip() for x in l.split(">", 1))
                    openers[phrase] = kind
                elif l:
                    cmds.append(l)
    except OSError:
        pass
    return cmds, openers


def read_list(path: str) -> list:
    """config/music.txt, config/home.txt: names of one's own on top of what HA gives (or without HA)."""
    try:
        with open(path) as f:
            return [l.split("#")[0].strip() for l in f if l.split("#")[0].strip()]
    except OSError:
        return []


def write_bias(names: list, music: list) -> None:
    cmds, openers = read_commands()
    conf = os.path.dirname(COMMANDS_FILE)  # /config in Docker, /etc/assist-asr in the Proxmox CT
    names = names + read_list(os.path.join(conf, "home.txt"))
    music = music + read_list(os.path.join(conf, "music.txt"))
    words = [w for w in re.split(r"\|", WAKE) if w]
    home = names + ["lights", "light", "lamp", "lamps", "fan", "tv", "heating", "everything", "blinds",
                    "curtains", "shades", "garage door", "door", "front door", "window", "windows"]
    d = {"wake": words, "commands": cmds, "openers": openers, "fillers": FILLERS,
         "lists": {"music": music + ["music", "some music", "something", "anything", "radio", "the radio",
                                     "my music", "my playlist", "playlist", "songs", "a song"],
                   "home": home}}
    _context["lists"] = {k: {asr_bias.norm(n).strip(): n for n in v if asr_bias.norm(n).strip()}
                         for k, v in d["lists"].items()}
    tmp = BIAS_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(d, f)
    os.replace(tmp, BIAS_FILE)


def read_vocab() -> str:
    try:
        with open(VOCAB_FILE) as f:
            words = [l.split("#")[0].strip() for l in f]
    except OSError:
        return ""
    return ", ".join(w for w in words if w)


# HA's names and Music Assistant's library: fetched in the background every CONTEXT_REFRESH (a request
# never waits for HA), kept on disk (CACHE_FILE) so a restart has them at once, also with HA down.
CACHE_FILE = os.environ.get("CACHE_FILE", os.path.join(os.path.dirname(BIAS_FILE) or "/data", "names-cache.json"))


async def context() -> str:
    """The context prompt, as of the last refresh. Same text every request until the next one: it comes
    before the audio in the prompt, so vLLM's prefix cache keeps its KV and it costs little per request."""
    return _context.get("text", "")


async def apply_names(names: list, music: list) -> None:
    ha, vocab = context_of(names), read_vocab()
    _context["text"] = ". ".join(p for p in (ha, vocab) if p) if USE_CONTEXT else ""
    log.info("context: %d chars of HA names, %d of vocabulary%s", len(ha), len(vocab),
             "" if USE_CONTEXT else " (not sent)")
    if BIAS_FILE:
        try:
            await asyncio.to_thread(write_bias, names, music)
            log.info("bias: %d HA names, %d music names, w_cmd %.1f w_slot %.1f",
                     len(names), len(music), W_CMD, W_SLOT)
        except OSError as e:
            log.warning("bias file: %s", e)


def load_cache() -> bool:
    try:
        with open(CACHE_FILE) as f:
            c = json.load(f)
        _context["names"], _context["music"] = c.get("names", []), c.get("music", [])
        return True
    except (OSError, ValueError):
        return False


def save_cache() -> None:
    tmp = CACHE_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump({"names": _context.get("names", []), "music": _context.get("music", []), "at": time.time()}, f)
    os.replace(tmp, CACHE_FILE)


async def refresh() -> bool:
    """Names from HA and Music Assistant; True when HA answered. The last ones stay when it does not."""
    ok = False
    if HASS_URL and HASS_TOKEN:
        try:
            _context["names"] = await asyncio.to_thread(ha_names)
            ok = True
        except Exception as e:
            log.warning("HA names: %s", e)
        if BIAS_FILE and ok:
            try:
                _context["music"] = await asyncio.to_thread(music_library)
            except Exception as e:
                log.warning("Music Assistant library: %s", e)
        if ok:
            try:
                await asyncio.to_thread(save_cache)
            except OSError as e:
                log.warning("names cache: %s", e)
    await apply_names(_context.get("names", []), _context.get("music", []))
    return ok


async def refresher() -> None:
    while True:  # again in 30 s while HA is unreachable (restarting), else every CONTEXT_REFRESH
        await asyncio.sleep(CONTEXT_REFRESH if await refresh() else 30)


def _chat(wav: bytes, prompt: str, xargs=None, top: int = 0) -> dict:
    msgs = [{"role": "user", "content": [{"type": "input_audio", "input_audio": {
        "data": base64.b64encode(wav).decode(), "format": "wav"}}]}]
    if prompt:
        msgs.insert(0, {"role": "system", "content": prompt})
    body = {"model": MODEL, "messages": msgs, "temperature": 0, "max_tokens": 200, "logprobs": True}
    if top:
        body["top_logprobs"] = top
    if xargs:
        body["vllm_xargs"] = xargs
    req = urllib.request.Request(VLLM_URL + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())["choices"][0]


def transcribe(wav: bytes, language: str, prompt: str, bias=None, wake: str = "", slot: int = 0):
    """wake: a wake phrase forced as the transcript's start ("Alexa,"); it is then part of the returned
    head, not of the text or tokens, and transcribe.logprob covers it too."""
    if LLAMA:
        return _lc_transcribe(wav, language, prompt, bias, wake, slot)
    return _vllm_transcribe(wav, language, prompt, bias, wake)


def force_score(wav: bytes, prompt: str, text: str) -> float:
    if LLAMA:
        return _lc_force(wav, prompt, text)
    return _vllm_force(wav, prompt, text)


def _vllm_transcribe(wav: bytes, language: str, prompt: str, bias=None, wake: str = ""):
    """(transcript, tokens after <asr_text> as (text, logprob, {alternative: logprob}), head).
    Through chat completions (not /audio/transcriptions) for the logprobs; Qwen3-ASR then names the
    language itself ("language English<asr_text>..."), head is that part. bias: (w_cmd, w_slot)."""
    xargs = None
    if bias and BIAS_FILE:
        xargs = {"asr_bias": BIAS_FILE, "w_cmd": bias[0], "w_slot": bias[1],
                 "head": f"language {LANGUAGES.get(language, language)}<asr_text>{wake}"}
    c = _chat(wav, prompt, xargs, top=20)
    content = c["message"]["content"] or ""
    head, sep, text = content.partition("<asr_text>")
    if not sep:
        head, text = "", content if not content.startswith("language") else ""
    toks = []
    seen = not sep
    transcribe.logprob = 0.0  # of all that follows <asr_text>: what the name check compares with
    for t in (c.get("logprobs") or {}).get("content") or []:
        if not seen:
            seen = t["token"] == "<asr_text>"
            continue
        transcribe.logprob += t["logprob"]
        if t["token"].startswith("<|"):
            continue
        toks.append((t["token"], t["logprob"], {a["token"]: a["logprob"] for a in t["top_logprobs"]}))
    if wake:  # the forced wake phrase belongs to the head, as on llama.cpp (where it is in the prompt)
        acc, n = "", 0
        while n < len(toks) and len(acc) < len(wake):
            acc += toks[n][0]
            n += 1
        if acc == wake and text.startswith(wake):
            toks, text = toks[n:], text[len(wake):]
            return text.strip(), toks, head + sep + wake
    return text.strip(), toks, head + sep


# Qwen3-ASR's own chat template, plus the assistant's turn written out: with it a candidate transcript is
# part of the prompt, and one forward pass gives every one of its tokens' logprobs (prompt_logprobs), where
# generating it under force took a step per token (~150 ms a name check, against ~30). Needs vLLM started
# with --trust-request-chat-template (docker-compose.yml has it); without, the forced generation below.
SCORE_TEMPLATE = (
    "{%- set ns = namespace(system_text='') -%}"
    "{%- for m in messages -%}{%- if m.role == 'system' -%}"
    "{%- if m.content is string -%}{%- set ns.system_text = ns.system_text + m.content -%}"
    "{%- else -%}{%- for c in m.content -%}{%- if c.type == 'text' and (c.text is defined) -%}"
    "{%- set ns.system_text = ns.system_text + c.text -%}{%- endif -%}{%- endfor -%}{%- endif -%}"
    "{%- endif -%}{%- endfor -%}"
    "{%- set ns2 = namespace(audio_tokens='') -%}"
    "{%- for m in messages -%}{%- if m.content is not string -%}{%- for c in m.content -%}"
    "{%- if c.type == 'audio' or ('audio' in c) or ('audio_url' in c) -%}"
    "{%- set ns2.audio_tokens = ns2.audio_tokens + '<|audio_start|><|audio_pad|><|audio_end|>' -%}"
    "{%- endif -%}{%- endfor -%}{%- endif -%}{%- endfor -%}"
    "{{- '<|im_start|>system\\n' + ns.system_text + '<|im_end|>\\n' -}}"
    "{{- '<|im_start|>user\\n' + ns2.audio_tokens + '<|im_end|>\\n' -}}"
    "{%- for m in messages -%}{%- if m.role == 'assistant' -%}"
    "{%- if m.content is string -%}{%- set t = m.content -%}"
    "{%- else -%}{%- set t = m.content | selectattr('type', 'equalto', 'text') | map(attribute='text') | join('') -%}"
    "{%- endif -%}"
    "{{- '<|im_start|>assistant\\n' + t + '<|im_end|>\\n' -}}{%- endif -%}{%- endfor -%}")
_score = {"prefill": True}


def _vllm_score(wav: bytes, prompt: str, text: str) -> float:
    msgs = [{"role": "user", "content": [{"type": "input_audio", "input_audio": {
        "data": base64.b64encode(wav).decode(), "format": "wav"}}]}, {"role": "assistant", "content": text}]
    if prompt:
        msgs.insert(0, {"role": "system", "content": prompt})
    body = {"model": MODEL, "messages": msgs, "max_tokens": 1, "temperature": 0, "prompt_logprobs": 0,
            "add_generation_prompt": False, "chat_template": SCORE_TEMPLATE}
    req = urllib.request.Request(VLLM_URL + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        pl = json.loads(r.read()).get("prompt_logprobs") or []
    lp, seen = 0.0, False
    for pos in pl:  # each: {token id: {logprob, decoded_token}} for the prompt's token there
        if not pos:
            continue
        e = next(iter(pos.values()))
        tok = e.get("decoded_token", "")
        if seen:
            lp += e["logprob"]
            if tok == "<|im_end|>":
                break
        seen = seen or tok == "<asr_text>"
    if not seen:
        raise ValueError("no <asr_text> in the scored prompt")
    return lp


def _vllm_force(wav: bytes, prompt: str, text: str) -> float:
    """How well the audio supports text: summed raw logprob of its tokens after <asr_text>, the end included.
    Scored in one pass (above); generated under force when the server refuses request templates."""
    if _score["prefill"]:
        try:
            return _vllm_score(wav, prompt, text)
        except urllib.error.HTTPError as e:
            if e.code != 400:
                raise
            _score["prefill"] = False
            log.warning("vLLM refuses request chat templates (start it with --trust-request-chat-template): "
                        "the name check generates its candidates instead, slower")
    c = _chat(wav, prompt, {"force": text})
    lp, seen = 0.0, False
    for t in (c.get("logprobs") or {}).get("content") or []:
        if seen:
            lp += t["logprob"]
        seen = seen or t["token"] == "<asr_text>"
    return lp


# llama.cpp (BACKEND=llamacpp). Its server has no hook inside decoding, so the push is done from outside:
# one decode with the 20 likeliest tokens at each step; where a pushed one beats what was chosen, the
# decode goes on from there (that token appended to the prompt). The audio's encoding and everything
# before stays in the slot's cache, so a restart costs only the new tokens: 0-2 restarts a request.
# All requests go to slot 0, one at a time: each slot caches separately, and a second one would encode
# the audio again (on a CPU the costliest part).
_lc = {"marker": None}
# Where the audio goes in a prompt. llama-server makes its media marker up anew at every start
# ("<__media_Z9xf6p...__>", GET /props), so prompts carry this stand-in and _lc_post puts the server's
# current marker in just before sending. Cached once, a llama-server restarted under us (a crash, a driver
# update, a new image) cost every request after it a 400 ("number of media markers in text (0) does not
# match number of bitmaps (1)") until this server was restarted too (issue #1).
MEDIA = "@@assist-asr-media@@"
LLAMA_WAIT = float(os.environ.get("LLAMA_WAIT", "20"))  # s a request waits for a llama-server (re)starting


def _lc_marker(fresh: bool = False) -> str:
    if fresh or _lc["marker"] is None:
        with urllib.request.urlopen(LLAMA_URL + "/props", timeout=10) as r:
            _lc["marker"] = json.loads(r.read())["media_marker"]
    return _lc["marker"]


def _lc_post(path: str, body: dict) -> dict:
    raw = json.dumps(body)
    deadline = time.monotonic() + LLAMA_WAIT
    fresh, retried = False, False
    while True:
        try:
            data = raw.replace(MEDIA, json.dumps(_lc_marker(fresh))[1:-1]).encode()
            fresh = False
            req = urllib.request.Request(LLAMA_URL + path, data=data, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=120) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            msg = e.read().decode(errors="replace")
            if e.code == 400 and not retried:
                # A llama-server started since? Its error says "number of media markers ..." or only
                # "Failed to tokenize prompt", by version: the marker asked again, retried once if new.
                retried, old = True, _lc["marker"]
                if _lc_marker(True) != old:
                    log.warning("llama.cpp's media marker changed (restarted): retried with the new one")
                    continue
            if e.code == 503 and time.monotonic() < deadline:  # still loading the model
                time.sleep(1)
                continue
            raise RuntimeError(f"llama.cpp {path}: HTTP {e.code}: {msg[:200]}") from None
        except (urllib.error.URLError, ConnectionError) as e:
            # down or starting (its marker will be new): wait a little, else fail this request
            if time.monotonic() >= deadline:
                raise RuntimeError(f"llama.cpp unreachable at {LLAMA_URL}: {e}") from None
            fresh = True
            time.sleep(1)


def _lc_prompt(prompt: str, language: str) -> str:
    system = f"<|im_start|>system\n{prompt}<|im_end|>\n" if prompt else ""
    return (f"{system}<|im_start|>user\n{MEDIA}<|im_end|>\n<|im_start|>assistant\n"
            f"language {LANGUAGES.get(language, language)}<asr_text>")


def _engine_split(prompt_string: str):
    """The engine keeps everything up to "<asr_text>" encoded; what follows is sent as text."""
    i = prompt_string.index("<asr_text>") + len("<asr_text>")
    return prompt_string[:i], prompt_string[i:]


def _engine_score(wav: bytes, prompt: str, given: str, tails: list, end: bool = True, language: str = "") -> list:
    base = _lc_prompt(prompt, language or LANGUAGE)
    r = _lc_post("/score", {"wav": base64.b64encode(wav).decode(), "prompt": base, "given": given,
                            "tails": tails, "end": end})
    return r["scores"]


def _lc_complete(wav: bytes, prompt_string: str, n_probs: int = 0, grammar: str = "", slot: int = 0) -> list:
    if ENGINE:  # no grammar there: forcing goes through _engine_score
        base, text = _engine_split(prompt_string)
        return _lc_post("/generate", {"wav": base64.b64encode(wav).decode(), "prompt": base, "text": text,
                                      "n_probs": n_probs, "max_tokens": 200}).get("completion_probabilities") or []
    body = {"prompt": {"prompt_string": prompt_string, "multimodal_data": [base64.b64encode(wav).decode()]},
            "temperature": 0, "n_predict": 200, "n_probs": n_probs, "cache_prompt": True, "id_slot": slot}
    if grammar:
        body["grammar"] = grammar
    steps = _lc_post("/completion", body).get("completion_probabilities") or []
    if any(e.get("logprob") is None for e in steps):  # NaN in JSON: the backend computed garbage
        raise RuntimeError("llama.cpp returned no probabilities (NaN): its GPU backend fails on this "
                           "GPU; proxmox/gpu-check.sh finds settings that work, DEVICE=cpu runs it on the CPU")
    return steps


def _lc_transcribe(wav: bytes, language: str, prompt: str, bias=None, wake: str = "", slot: int = 0):
    head = f"language {LANGUAGES.get(language, language)}<asr_text>{wake}"
    p = _lc_prompt(prompt, language) + wake
    b = asr_bias.load(BIAS_FILE) if bias and BIAS_FILE else None
    text, toks, lp = "", [], 0.0

    def score(at: str, tok: str, tok_lp: float) -> float:
        # the end of the transcript ends the last word, as in the vLLM plugin
        return tok_lp + b.weight(asr_bias.norm(at + (tok or " ")), bias[0], bias[1], bias[0])

    for _ in range(6):
        steps = _lc_complete(wav, p + text, n_probs=20, slot=slot)
        restart = False
        for e in steps:
            tok = e["token"]
            alts = {a["token"]: a["logprob"] for a in e.get("top_logprobs") or []}
            alts.setdefault(tok, e["logprob"])
            if b is not None:
                best = max(alts, key=lambda a: score(text, a, alts[a]))
                if best != tok and score(text, best, alts[best]) > score(text, tok, e["logprob"]) + 1e-6:
                    lp += alts[best]
                    if best == "":  # the push says: end here
                        break
                    toks.append((best, alts[best], alts))
                    text += best
                    restart = True
                    break
            lp += e["logprob"]
            if tok == "":  # end of the transcript
                break
            toks.append((tok, e["logprob"], alts))
            text += tok
        if not restart:
            break
    if wake and slot == 0:  # the forced phrase was in the prompt: its own score, so totals compare with a
        lp += _lc_prefix_lp(wav, prompt, language, wake)  # free decode (alternatives, slot 1, need none)
    transcribe.logprob = lp
    return text.strip(), toks, head


def _lc_prefix_lp(wav: bytes, prompt: str, language: str, wake: str) -> float:
    """Raw logprob of the transcript starting with wake: its tokens forced, the rest left open."""
    if ENGINE:
        return _engine_score(wav, prompt, "", [wake], end=False, language=language)[0]
    lit = wake.replace("\\", "\\\\").replace('"', '\\"')
    steps = _lc_complete(wav, _lc_prompt(prompt, language), n_probs=1,
                         grammar=f'root ::= "{lit}" [^\\x00]*')
    lp, acc = 0.0, ""
    for e in steps:
        if len(acc) >= len(wake):
            break
        acc += e["token"]
        lp += e["logprob"]
    return lp


def _lc_force(wav: bytes, prompt: str, text: str, tail: str = None) -> float:
    """Summed raw logprob of text with llama.cpp forced to write it (a grammar of just that string; the
    probabilities it reports are the model's own, before the grammar). With tail: text is given (in the
    prompt) and only tail is forced and scored."""
    head, _, body = text.partition("<asr_text>")
    language = next((k for k, v in LANGUAGES.items() if head == f"language {v}"), LANGUAGE)
    given, forced = (body, tail) if tail is not None else ("", body)
    if ENGINE:
        return _engine_score(wav, prompt, given, [forced], language=language)[0]
    lit = forced.replace("\\", "\\\\").replace('"', '\\"')
    steps = _lc_complete(wav, _lc_prompt(prompt, language) + given, n_probs=1, grammar=f'root ::= "{lit}"')
    return sum(e["logprob"] for e in steps)


def _sounds(s: str) -> str:
    """Crude sound-alike spelling for picking candidates ("Dola", "Zola", "to la")."""
    s = re.sub(r"[^a-z]", "", s.lower())
    for a, b in (("ph", "f"), ("ck", "k"), ("c", "k"), ("q", "k"), ("z", "s"), ("x", "ks"),
                 ("w", "v"), ("y", "i"), ("ee", "i"), ("oo", "u"), ("h", "")):
        s = s.replace(a, b)
    return re.sub(r"(.)\1+", r"\1", s)


_pool = ThreadPoolExecutor(8)
LANGUAGES = {"en": "English", "de": "German", "fr": "French", "es": "Spanish", "it": "Italian",
             "nl": "Dutch", "pt": "Portuguese", "ru": "Russian", "zh": "Chinese", "ja": "Japanese",
             "ko": "Korean", "ar": "Arabic", "he": "Hebrew"}


def word_alternatives(raw: str, toks, head: str, wav: bytes, language: str, prompt: str, n: int = 2) -> list:
    """Transcripts that go another way at an unsure word: where a word's first token had a close second
    (within ALT_RANGE / 3 nats), decoded again with the text up to there and that second forced, the rest
    left to the model (as the wake word forcing does). At most n, at the closest calls; on vLLM in parallel."""
    if not toks:
        return []
    spots, pos = [], 0
    for i, (t, lp, alt) in enumerate(toks):
        word_start = i == 0 or t[:1] == " "
        if word_start and asr_bias.norm(t).strip():
            for a, l in alt.items():
                if (a[:1] == " " or i == 0) and asr_bias.norm(a).strip() and \
                        asr_bias.norm(a).strip() != asr_bias.norm(t).strip() and l >= lp - ALT_RANGE / 3:
                    spots.append((lp - l, pos, a))
        pos += len(t)
    text = "".join(t for t, _, _ in toks)
    _, _, given = head.partition("<asr_text>")  # a forced wake phrase stays in front

    def decode(spot):
        _, p, a = spot
        forced = given + text[:p] + a
        try:
            # llama.cpp: slot 1, beside the name check in slot 0 (each slot caches the audio on its own)
            _, rest_toks, _ = transcribe(wav, language, prompt, (W_CMD, W_SLOT), forced, slot=1)
        except Exception:
            return ""
        rest = "".join(t for t, _, _ in rest_toks)  # the tokens keep their spaces; the text is trimmed
        whole = rest if rest.startswith(forced) else forced + rest
        return whole.strip()

    picked = sorted(spots)[:n]
    if LLAMA:  # one slot / one engine: one after another
        return [w for w in map(decode, picked) if w]
    return [w for w in _pool.map(decode, picked) if w]


def publish_alternatives(text: str, alts: list) -> None:
    """The sensor's state: the alternatives joined (HA cuts states at 255); the full list in attributes."""
    seen, uniq = {asr_bias.norm(text).strip()}, []
    for a in alts:
        k = asr_bias.norm(strip_wake(a)).strip()
        if k and k not in seen:
            seen.add(k)
            uniq.append(strip_wake(a))
    body = {"state": " | ".join(uniq)[:250] or "none",
            "attributes": {"transcript": text, "alternatives": uniq[:3], "friendly_name": "Speech-to-text alternatives",
                           "icon": "mdi:account-voice"}}
    _hass_keep("POST", f"/api/states/{ALTERNATIVES}", body)


_keep = {"conn": None, "lock": threading.Lock()}


def _hass_keep(method: str, path: str, body) -> None:
    """One kept-open connection to HA for what goes out on every request (a new one costs a handshake),
    1 s at most: a slow HA never holds up the transcript long."""
    u = urllib.parse.urlsplit(HASS_URL)
    data = json.dumps(body).encode()
    headers = {"Authorization": f"Bearer {HASS_TOKEN}", "Content-Type": "application/json"}
    with _keep["lock"]:
        for attempt in (1, 2):  # a connection HA closed meanwhile: once more on a new one
            if _keep["conn"] is None:
                cls = http.client.HTTPSConnection if u.scheme == "https" else http.client.HTTPConnection
                _keep["conn"] = cls(u.hostname, u.port, timeout=1)
            try:
                _keep["conn"].request(method, (u.path.rstrip("/") + path), body=data, headers=headers)
                r = _keep["conn"].getresponse()
                r.read()
                return
            except (OSError, http.client.HTTPException):
                _keep["conn"].close()
                _keep["conn"] = None
                if attempt == 2:
                    raise


# Whether the last requests began with a wake word: does the satellite send it ("Wake word in the request")?
_wake_seen = collections.deque(maxlen=20)


# What Qwen3-ASR says to next to nothing: never a request on its own
FILLER_WORDS = {"a", "ah", "uh", "um", "hmm", "mm", "oh", "okay", "ok", "yeah", "yes", "no", "you", "thank you",
                "thanks", "bye", "so", "the", "i", "and", "huh", "hm"}


def is_command(text: str) -> bool:
    """A whole command of config/commands.txt ("Stop.", "next track"), wake word aside."""
    t = asr_bias.norm(strip_wake(text)).strip()
    if not t:
        return False
    cmds, openers = read_commands()
    return t in {asr_bias.norm(c).strip() for c in cmds + list(openers)}


def wake_guess(raw: str, toks) -> str:
    """The wake phrase a transcript without one probably starts with, mangled ("Lex allowed her",
    "Hey, Minecraft"), or "": by sound against its first words, or a wake word among the first token's
    alternatives."""
    if WAKE_FORCE == "off" or not raw or WAKE_RE.match(raw):
        return ""
    if WAKE_FORCE == "auto" and len(_wake_seen) >= 5 and sum(_wake_seen) < len(_wake_seen) / 4:
        # The satellite doesn't send the wake word (most don't): there is none to mend, and forcing one only
        # made things up (issue #1's 99 requests without it were where assist-asr fell behind plain Qwen).
        return ""
    bias = asr_bias.load(BIAS_FILE) if BIAS_FILE else None
    if bias is not None and asr_bias.norm(raw).strip() + " " in bias.cmds:
        return ""  # a whole command as it is ("Louder.", "Next."): nothing of it went into a wake word
    words = raw.split()
    alts = toks[0][2] if toks else {}
    best, best_s = "", 0.0
    for ph in WAKE_PHRASES:
        n = len(ph.split())
        s = max(difflib.SequenceMatcher(None, _sounds(" ".join(words[:k])), _sounds(ph)).ratio()
                for k in (n, n + 1))
        if s < 0.4:
            # On short, quiet requests the first token's alternatives are flat: some wake word's start was
            # always among them above -10, and that alone made "Next." a mangled "Okay Nabu" (issue #1).
            continue
        p = asr_bias.norm(ph).strip()
        if any(len(asr_bias.norm(a).strip()) >= 3 and p.startswith(asr_bias.norm(a).strip()) and lp > -10
               for a, lp in alts.items()):
            s += 0.5
        if s > best_s:
            best, best_s = ph, s
    return best if best_s >= 0.6 else ""


_sounds_cache = {}


def _sounds_of(kind: str, names: dict) -> dict:
    """Each name's sound-alike spelling, worked out once per list (the lists change every 10 minutes at most)."""
    held = _sounds_cache.get(kind)
    if held is None or held[0] is not names:  # a refresh made a new list
        held = _sounds_cache[kind] = (names, {n: _sounds(n) for n in names})
    return held[1]


def name_check(raw: str, toks, head: str, wav: bytes, prompt: str, heard_lp: float, close=None) -> str:
    """close: a list the names that came close are added to (as whole transcripts), for ALTERNATIVES."""
    bias = asr_bias.load(BIAS_FILE) if BIAS_FILE else None
    lists = _context.get("lists") or {}
    if bias is None or not toks:
        return raw
    prefix, slot_i, kind = "", None, None
    for i, (t, _, _) in enumerate(toks):
        k, p = bias.region(asr_bias.norm(prefix + t))
        if k not in (None, "cmd") and p.strip():
            slot_i, kind = i, k
            break
        prefix += t
    if slot_i is None or kind not in lists or not lists[kind]:
        return raw
    rest = "".join(t for t, _, _ in toks[slot_i:])
    m = re.match(r"^(.*?)([\s.!?,]*)$", rest, re.S)
    rest, ending = m.group(1), m.group(2)
    names = lists[kind]
    firsts = {asr_bias.norm(a).strip(): lp for a, lp in toks[slot_i][2].items()}
    # Words between command and name ("play some X", "turn off the X") stay as heard.
    # A name may start with those words itself ("The Fate of Ophelia" heard as "the Fate of a Feeling"):
    # such a name takes their place instead of following them ("play the The Fate of Ophelia" scored so badly
    # that the right name never won; issue #1). So every candidate is scored from sp, the words before the
    # filler, with a tail of its own.
    nr = asr_bias.norm(rest)
    sp, filler, filler_raw = prefix, "", ""
    for f in sorted(bias.kind_fillers.get(kind, []) + bias.fillers, key=len, reverse=True):
        if nr.startswith(f) and nr[len(f):].strip() and nr.strip() not in names:
            n = len(f.split())
            words = rest.split(" ")
            lead = [w for w in words if w]
            cut = rest.find(lead[n - 1]) + len(lead[n - 1])
            filler, filler_raw = f, rest[:cut]
            prefix, rest = prefix + rest[:cut], rest[cut:].lstrip()
            firsts = {}
            break
    heard = asr_bias.norm(rest).strip()
    if not heard:
        return raw
    if heard in names:
        # One of the names, as said: written as HA / Music Assistant spell it, punctuation and all
        # ("Hillel's MacBook Pro AirPlay" -> "Hillel’s MacBook Pro (AirPlay)"), since HA's own sentences
        # match a name only exactly.
        exact = (prefix + " " + names[heard] + ending).strip()
        if exact != raw and asr_bias.norm(exact) == asr_bias.norm(raw):
            log.info("name spelled as listed (%s): %r -> %r", kind, raw, exact)
            return exact
        return raw
    if any(f" {n} " in f" {heard} " for n in names):
        return raw  # a known name with more around it ("living room lights"): HA's own sentences
    whole = "".join(t for t, _, _ in toks)
    hs, hs_whole = _sounds(heard), _sounds(filler + heard)
    t_pick = time.monotonic()
    sounds = _sounds_of(kind, names)
    # names that start with the filler: by how the filler and what followed sound together
    like = {n: difflib.SequenceMatcher(None, hs_whole if filler and n.startswith(filler) else hs, sounds[n]).ratio()
            for n in names}

    def tail(c):
        """What candidate c makes of the transcript after sp."""
        lead = "" if filler and c.startswith(filler) else filler_raw
        return lead + " " + names[c] + ending

    def pick(n):
        s1 = max([lp for a, lp in firsts.items() if a and n.startswith(a)], default=-15.0)
        return max(s1, -15.0) + 6 * like[n]

    # The first letter alone pulls towards the wrong names (" D" -0.07 for "Zola" heard as "Dola"),
    # so half the shortlist is by sound only.
    by_sound = sorted(names, key=like.get, reverse=True)
    half = NAME_CANDIDATES // 2
    cands = list(dict.fromkeys(by_sound[:NAME_CANDIDATES - half] + sorted(names, key=pick, reverse=True)[:half]))
    pruned = []
    if firsts:
        # A name can do no better than its first token did in the decode itself: skip those that lose
        # by SLOT_PRIOR already there (on a CPU every scoring pass costs ~150 ms of the cores).
        alts = toks[slot_i][2]
        floor = min(alts.values()) if alts else -30.0
        heard_slot = sum(lp for _, lp, _ in toks[slot_i:])

        def bound(n):
            # tokens of the decode that may start n, plus a few unseen ones at most as likely as the least
            first = [lp for a, lp in alts.items() if asr_bias.norm(a).strip() and n.startswith(asr_bias.norm(a).strip())]
            return math.log(sum(math.exp(x) for x in first + [floor] * 5))

        b = {c: bound(c) for c in cands}
        pruned = [c for c in cands if b[c] < heard_slot - SLOT_PRIOR]
        if pruned:  # the best a pruned one could score, for the margin below
            pruned_best = max(b[c] for c in pruned) + (heard_lp - heard_slot)
        cands = [c for c in cands if c not in pruned]
        if not cands:
            return raw
    texts = [whole] + [sp + tail(c) for c in cands]
    # what was heard is scored by its own decode (raw logprobs, before the plugin's push)
    t_score = time.monotonic()
    if ENGINE:
        # All at once: the engine copies the encoded audio to a sequence per candidate and scores them
        # in one batched pass (~70 ms on a desktop CPU for 6, where llama-server took ~0.7 s).
        heard_tail = whole[len(sp):] if whole.startswith(sp) else filler_raw + " " + rest + ending
        _, _, given = (head + sp).partition("<asr_text>")
        scores = _engine_score(wav, prompt, given, [heard_tail] + [tail(c) for c in cands])
        if pruned:
            pruned_best = max(b[c] for c in pruned)
    elif LLAMA:
        # One slot, one at a time, and only from the name on: the words before it are the same for all,
        # so they go into the (cached) prompt and each candidate costs just its own few tokens (forcing
        # the whole sentence each time took ~1 s a check on a CPU). What was heard is scored the same way.
        heard_tail = whole[len(sp):] if whole.startswith(sp) else filler_raw + " " + rest + ending
        hyp_s = _lc_force(wav, prompt, head + sp, heard_tail)
        # Branch and bound: a name scores at most what its first token did in the decode (b), so the most
        # promising go first, and the check stops once no name left could win or come within SLOT_MARGIN
        # of the best: usually 1-2 of the candidates are scored. (In parallel slots llama.cpp's server
        # would encode the audio again in each: on a CPU that costs more than it saves, measured.)
        got, skipped = {}, []
        order = sorted(cands, key=lambda c: b[c], reverse=True) if firsts else list(cands)
        for c in order:
            if firsts and got and NAME_EARLY_STOP:
                best_s = max(got.values())
                if b[c] < best_s - SLOT_MARGIN or (b[c] < hyp_s - SLOT_PRIOR and best_s < hyp_s - SLOT_PRIOR):
                    skipped = [x for x in order if x not in got]
                    break
            got[c] = _lc_force(wav, prompt, head + sp, tail(c))
        scores = [hyp_s] + [got.get(c, float("-inf")) for c in cands]
        bounds_left = [b[c] for c in pruned + skipped] if firsts else []
        if bounds_left:  # the bounds are of the name's first token alone: right in this space as they are
            pruned, pruned_best = pruned + skipped, max(bounds_left)
        log.debug("name check: scored %d of %d (bounds %s, best %.1f, heard %.1f)", len(got), len(cands),
                 [round(b[c], 1) for c in order] if firsts else "-", max(got.values()), hyp_s)
    elif _score["prefill"]:
        # One pass each, in parallel, what was heard too, and without the context prompt: a request with
        # prompt_logprobs cannot use vLLM's prefix cache, so the ~450 tokens of context were prefilled again
        # for every candidate (6: ~220 ms, without: ~65). The check compares candidates on the same audio,
        # which the context does not change; the decode itself keeps it.
        # A small context of its own instead: just the names compared (~30 tokens). The context helps the
        # scores (measured: 82 % right with the full one, 80 % with none); this keeps that where it counts.
        short = ", ".join(names[c] for c in cands)
        scores = list(_pool.map(lambda t: force_score(wav, short, head + t), texts))
        if pruned:
            pruned_best = max(b[c] for c in pruned) + (scores[0] - heard_slot)
    else:
        scores = [heard_lp] + list(_pool.map(lambda t: force_score(wav, prompt, head + t), texts[1:]))
    hyp, ranked = scores[0], sorted(zip(scores[1:], cands), reverse=True)
    (best, name), second = ranked[0], (ranked[1][0] if len(ranked) > 1 else -1e9)
    if firsts and pruned:
        second = max(second, pruned_best)
    ok = best >= hyp - SLOT_PRIOR and best - second >= SLOT_MARGIN
    log.info("name check (%s, pick %d ms, score %d ms): %r %.1f vs %s -> %s", kind,
             (t_score - t_pick) * 1000, (time.monotonic() - t_score) * 1000, rest, hyp,
             ", ".join(f"{names[c]!r} {s:.1f}" for s, c in ranked[:3]), names[name] if ok else "kept")
    if close is not None:  # the runners-up the audio supports nearly as well, and what was heard if replaced
        close += [texts[1 + cands.index(c)].strip() for s, c in ranked
                  if s >= hyp - ALT_RANGE and not (ok and c == name)]
        if ok:
            close.append(raw)
    return texts[1 + cands.index(name)].strip() if ok else raw


INFO = Info(asr=[AsrProgram(
    name="qwen3-asr-vllm",
    attribution=Attribution(name="Qwen", url="https://github.com/QwenLM/Qwen3-ASR"),
    installed=True, description="Qwen3-ASR-1.7B on vLLM", version="1.0",
    models=[AsrModel(name="Qwen/Qwen3-ASR-1.7B",
                     attribution=Attribution(name="Qwen", url="https://huggingface.co/Qwen/Qwen3-ASR-1.7B"),
                     installed=True, description="Qwen3-ASR-1.7B", version="1.7B",
                     languages=["en"] if LANGUAGE == "en" else [LANGUAGE])])])


class Handler(AsyncEventHandler):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.conv = AudioChunkConverter(rate=16000, width=2, channels=1)
        self.pcm = bytearray()
        self.language = LANGUAGE
        self.record = bool(RECORD_DIR)
        self.vad = None

    async def handle_event(self, event: Event) -> bool:
        if Describe.is_type(event.type):
            await self.write_event(INFO.event())
        elif Transcribe.is_type(event.type):
            t = Transcribe.from_event(event)
            if t.language:
                self.language = t.language.split("-")[0].split("_")[0]
            if (t.context or {}).get("no_record"):  # bench/bench.py: its clips are datasets already
                self.record = False
        elif AudioStart.is_type(event.type):
            self.pcm.clear()
            self.vad = StreamVad()
        elif AudioChunk.is_type(event.type):
            self.pcm += self.conv.convert(AudioChunk.from_event(event)).audio
            if self.vad is None:  # a client that sends no audio-start
                self.vad = StreamVad()
            self.vad.feed(self.pcm)
        elif AudioStop.is_type(event.type):
            t0 = time.monotonic()
            pcm = bytes(self.pcm)
            vad = self.vad or StreamVad()
            self.vad = None
            await asyncio.to_thread(vad.feed, pcm)  # what is left (all of it if streamed faster than real time)
            span, speech_ms = await asyncio.to_thread(vad.span, len(pcm), pcm)
            t_vad = time.monotonic()
            t_dec = t_chk = t_vad
            raw = text = ""
            if span is None:
                log.info("%.1f s audio, %d ms speech: no speech, empty transcript",
                         len(pcm) / 32000, speech_ms)
            elif vad.short:
                buf = io.BytesIO()
                with wave.open(buf, "wb") as w:
                    w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000)
                    w.writeframes(pcm[span[0]:span[1]])
                try:  # no context: on next to no audio that is what the model would read back
                    heard, _, _ = await asyncio.to_thread(transcribe, buf.getvalue(), self.language, "", None)  # nor a push
                    lp = transcribe.logprob
                except Exception as e:
                    log.error("transcription failed: %s", e)
                    heard, lp = "", -1e9
                words = asr_bias.norm(strip_wake(heard)).split()
                # under the gate a whole command only; just over it also a few words the model is sure of
                # ("On.", "Off."), but not what it says to near silence
                ok = lp >= SHORT_LP and (is_command(heard) or (
                    speech_ms >= MIN_SPEECH_MS and 0 < len(words) <= 3 and " ".join(words) not in FILLER_WORDS))
                raw = text = strip_wake(heard) if ok else ""
                log.info("%.1f s audio, %d ms speech, %d ms weak: %r %.1f: %s", len(pcm) / 32000, speech_ms,
                         len(vad.weak) * 32, heard, lp, "a command, kept" if ok else "not a command, empty transcript")
            else:
                buf = io.BytesIO()
                with wave.open(buf, "wb") as w:
                    w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000)
                    w.writeframes(pcm[span[0]:span[1]])
                ctx = await context()
                wav = buf.getvalue()
                toks, head, heard_lp = [], "", 0.0
                try:
                    raw, toks, head = await asyncio.to_thread(transcribe, wav, self.language, ctx,
                                                              (W_CMD, W_SLOT))
                    heard_lp = transcribe.logprob
                except Exception as e:
                    log.error("transcription failed: %s", e)
                if echoes_context(raw, ctx):
                    # Seen on real audio (a room ~15 dB under the talker after the Echo's gain): "Alexa,"
                    # then the whole context list. Without context the same audio is right; ~60 ms more.
                    log.info("context read back (%r...), asked again without it", raw[:60])
                    ctx = ""
                    try:
                        raw, toks, head = await asyncio.to_thread(transcribe, wav, self.language, "",
                                                                  (W_CMD, W_SLOT))
                        heard_lp = transcribe.logprob
                    except Exception as e:
                        log.error("transcription failed: %s", e)
                        raw, toks = "", []
                if raw:
                    _wake_seen.append(bool(WAKE_RE.match(raw)))
                guess = wake_guess(raw, toks)
                if guess:
                    try:
                        w = await asyncio.to_thread(transcribe, wav, self.language, ctx, (W_CMD, W_SLOT),
                                                    guess + ",")
                        w_lp = transcribe.logprob
                        keep = w_lp >= heard_lp - WAKE_PRIOR
                        # A forced "Alexa," is how the context starts its vocabulary ("Alexa, Home
                        # Assistant, Drop In, ..."), and on quiet audio the model went on reading it from
                        # there (issue #1: "Next track" came out as config/vocab.txt).
                        echo = echoes_context(w[0], ctx)
                        # and it must still sound like what was heard: "Lex allowed her" is "Alexa, louder";
                        # "Louder." is not "Alexa, set a timer." (a context entry the forced "Alexa," led to)
                        like = difflib.SequenceMatcher(None, _sounds(raw), _sounds(guess + " " + w[0])).ratio()
                        keep = keep and not echo and like >= 0.5
                        log.info("wake word forced (%s): %r %.1f -> %r %.1f, like %.2f: %s", guess, raw, heard_lp,
                                 w[0], w_lp, like, "kept" if keep else "context read back" if echo
                                 else "not what was heard" if like < 0.5 else "not supported by the audio")
                        if keep:
                            raw, toks, head = w
                            raw = raw[:1].upper() + raw[1:]
                            heard_lp = w_lp
                    except Exception as e:
                        log.error("wake word forcing failed: %s", e)
                t_dec = t_chk = time.monotonic()
                alts = []
                alt_task = None
                if ALTERNATIVES and HASS_URL and HASS_TOKEN and raw:
                    # alongside the name check: both need only the decode (vLLM batches them on the GPU)
                    alt_task = asyncio.ensure_future(asyncio.to_thread(
                        word_alternatives, raw, toks, head, wav, self.language, ctx))
                decoded = raw
                if BIAS_FILE and raw:
                    try:
                        raw = await asyncio.to_thread(name_check, raw, toks, head, wav, ctx, heard_lp, alts)
                        # (alts: this request's list of close transcripts, for ALTERNATIVES)
                    except Exception as e:
                        log.error("name check failed: %s", e)
                    t_chk = time.monotonic()
                text = strip_wake(raw)
                if alt_task is not None:
                    try:
                        words = await alt_task
                        if raw == decoded:  # the name check rewrote it: its own alternatives stand
                            alts += words
                    except Exception as e:
                        log.warning("alternatives: %s", e)
                    try:  # before the transcript goes out: the LLM reads the sensor right after
                        await asyncio.to_thread(publish_alternatives, text, alts)
                    except Exception as e:
                        log.warning("alternatives sensor: %s", e)
                log.info("%.1f s audio, %d ms speech, %d ms (vad %d, decode %d, name check %d): %r -> %r",
                         len(pcm) / 32000, speech_ms, (time.monotonic() - t0) * 1000, (t_vad - t0) * 1000,
                         (t_dec - t_vad) * 1000, (t_chk - t_dec) * 1000, raw, text)
            if RECORD_DIR and self.record:
                name = os.path.join(RECORD_DIR, time.strftime("%Y%m%d-%H%M%S"))
                with wave.open(name + ".wav", "wb") as w:
                    w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000)
                    w.writeframes(pcm)
                with open(name + ".json", "w") as f:
                    json.dump({"span_ms": [span[0] // 32, span[1] // 32] if span else None,
                               "speech_ms": speech_ms, "raw": raw, "text": text}, f)
            await self.write_event(Transcript(text=text).event())
            self.pcm.clear()
            return False
        return True


async def _later(fn, delay: float):
    await asyncio.sleep(delay)
    await fn()


async def warm_up():
    """One request at start: the plugin reads the model's vocabulary on first use (~5-8 s), better
    now than on the first thing someone says."""
    noise = (b"\x10\x00\xf0\xff" * 8000)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000); w.writeframes(noise)
    for _ in range(60):
        try:
            await asyncio.to_thread(transcribe, buf.getvalue(), LANGUAGE, "", (W_CMD, W_SLOT))
            if BIAS_FILE:
                await asyncio.to_thread(force_score, buf.getvalue(), "", "language English<asr_text>Pause.")
            log.info("warmed up")
            return
        except Exception:
            await asyncio.sleep(5)  # vLLM still starting


async def announce():
    """mDNS (_wyoming._tcp), so Home Assistant finds this server by itself (ZEROCONF: the name it shows).
    Only on the LAN's own network (Proxmox CT, Docker's host network); retried until the CT has an address."""
    try:
        from wyoming.zeroconf import HomeAssistantZeroconf
    except ImportError:
        log.warning("ZEROCONF is set but the zeroconf package is missing (pip install zeroconf)")
        return
    while True:
        try:
            zc = HomeAssistantZeroconf(port=PORT, name=ZEROCONF)
            await zc.register_server()
            log.info("announced as %s (%s:%d) by mDNS", ZEROCONF, zc.host, PORT)
            return
        except Exception as e:
            log.warning("mDNS announcement failed (%s); retrying", e)
            await asyncio.sleep(10)


async def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if load_cache():  # last time's names at once; HA asked in the background
        await apply_names(_context.get("names", []), _context.get("music", []))
        log.info("names from %s; refreshing from HA in the background", CACHE_FILE)
        asyncio.get_running_loop().create_task(refresher())
    else:  # nothing on disk yet: wait for HA once, then in the background
        ok = await refresh()
        asyncio.get_running_loop().create_task(_later(refresher, CONTEXT_REFRESH if ok else 30))
    await warm_up()
    log.info("listening on %d, %s", PORT, f"{BACKEND} at {LLAMA_URL}" if LLAMA else f"vLLM at {VLLM_URL}")
    if ZEROCONF:
        asyncio.get_running_loop().create_task(announce())
    await AsyncServer.from_uri(f"tcp://0.0.0.0:{PORT}").run(Handler)


def health() -> int:
    """For Docker's healthcheck: this server's port open and its model server answering. The port alone
    said "healthy" for hours while llama-server was gone (an NVIDIA driver update) and every request came
    back empty."""
    import socket
    try:
        socket.create_connection(("127.0.0.1", PORT), 2).close()
        with urllib.request.urlopen((LLAMA_URL if LLAMA else VLLM_URL) + "/health", timeout=5):
            pass
        return 0
    except Exception as e:
        print(f"unhealthy: {e}")
        return 1


if __name__ == "__main__":
    import sys
    if sys.argv[1:] == ["--health"]:
        sys.exit(health())
    asyncio.run(main())
