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
  WAKE_WORDS (alexa|alexis|alex|amazon|computer|echo|ziggy|jarvis|mycroft|nabu; "hey"/"ok" before one is dropped too)
  VAD_THRESHOLD (0.3), MIN_SPEECH_MS (150), PAD_MS (300)
  RECORD_DIR (unset): keep each request's audio + transcript there (bench/label.py turns them
  into a dataset)
"""

import asyncio
import base64
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

BACKEND = os.environ.get("BACKEND", "vllm")  # vllm, or llamacpp (llama.cpp's llama-server: CPU, Vulkan, CUDA)
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
VOCAB_FILE = os.environ.get("VOCAB_FILE", "/config/vocab.txt")
VAD_THRESHOLD = float(os.environ.get("VAD_THRESHOLD", "0.3"))
MIN_SPEECH_MS = int(os.environ.get("MIN_SPEECH_MS", "150"))
PAD_MS = int(os.environ.get("PAD_MS", "300"))
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
NAME_CANDIDATES = int(os.environ.get("NAME_CANDIDATES", "4" if BACKEND == "llamacpp" else "6"))
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

    def feed(self, pcm) -> None:
        while self.pos + self.n <= len(pcm):
            if self.vad(bytes(pcm[self.pos:self.pos + self.n])) >= VAD_THRESHOLD:
                self.hits.append(self.pos)
            self.pos += self.n

    def span(self, total: int):
        """(start, end) in bytes of 16 kHz s16 mono or None below MIN_SPEECH_MS, and speech ms."""
        ms = len(self.hits) * self.n // 32
        if ms < MIN_SPEECH_MS:
            return None, ms
        pad = PAD_MS * 32
        return (max(0, self.hits[0] - pad), min(total, self.hits[-1] + self.n + pad)), ms


def echoes_context(text: str, ctx: str) -> bool:
    """The model read (part of) its context back: 3+ of its entries in the transcript."""
    if not ctx:
        return False
    t = " " + re.sub(r"[^a-z0-9]+", " ", text.lower()) + " "
    entries = {re.sub(r"[^a-z0-9]+", " ", e.lower()).strip() for e in re.split(r"[,.]", ctx)}
    found = sum(1 for e in entries if e and f" {e} " in t)
    return found >= 3 and len(t.split()) > 8


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


async def context() -> str:
    # Same text on every request until the next refresh: it comes before the audio in the
    # prompt, so vLLM's prefix cache keeps its KV and the context costs little per request.
    if time.monotonic() - _context["at"] > CONTEXT_REFRESH:
        _context["at"] = time.monotonic()  # one try per period even if HA is down
        names = _context.get("names", [])
        if HASS_URL and HASS_TOKEN:
            try:
                names = _context["names"] = await asyncio.to_thread(ha_names)
            except Exception as e:  # keep the last names HA gave; HA restarting: again in 30 s
                log.warning("HA context: %s", e)
                _context["at"] = time.monotonic() - CONTEXT_REFRESH + 30
            if BIAS_FILE:
                try:
                    _context["music"] = await asyncio.to_thread(music_library)
                except Exception as e:
                    log.warning("Music Assistant library: %s", e)
        ha, vocab = context_of(names), read_vocab()
        _context["text"] = ". ".join(p for p in (ha, vocab) if p) if USE_CONTEXT else ""
        log.info("context: %d chars of HA names, %d of vocabulary%s", len(ha), len(vocab),
                 "" if USE_CONTEXT else " (not sent)")
        if BIAS_FILE:
            music = _context.get("music", [])
            try:
                await asyncio.to_thread(write_bias, names, music)
                log.info("bias: %d HA names, %d music names, w_cmd %.1f w_slot %.1f",
                         len(names), len(music), W_CMD, W_SLOT)
            except OSError as e:
                log.warning("bias file: %s", e)
    return _context["text"]


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


def transcribe(wav: bytes, language: str, prompt: str, bias=None):
    if BACKEND == "llamacpp":
        return _lc_transcribe(wav, language, prompt, bias)
    return _vllm_transcribe(wav, language, prompt, bias)


def force_score(wav: bytes, prompt: str, text: str) -> float:
    if BACKEND == "llamacpp":
        return _lc_force(wav, prompt, text)
    return _vllm_force(wav, prompt, text)


def _vllm_transcribe(wav: bytes, language: str, prompt: str, bias=None):
    """(transcript, tokens after <asr_text> as (text, logprob, {alternative: logprob}), head).
    Through chat completions (not /audio/transcriptions) for the logprobs; Qwen3-ASR then names the
    language itself ("language English<asr_text>..."), head is that part. bias: (w_cmd, w_slot)."""
    xargs = None
    if bias and BIAS_FILE:
        xargs = {"asr_bias": BIAS_FILE, "w_cmd": bias[0], "w_slot": bias[1],
                 "head": f"language {LANGUAGES.get(language, language)}<asr_text>"}
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
    return text.strip(), toks, head + sep


def _vllm_force(wav: bytes, prompt: str, text: str) -> float:
    """How well the audio supports text: summed raw logprob with the decoder forced to write it."""
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


def _lc_post(path: str, body: dict) -> dict:
    req = urllib.request.Request(LLAMA_URL + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())


def _lc_prompt(prompt: str, language: str) -> str:
    if _lc["marker"] is None:
        with urllib.request.urlopen(LLAMA_URL + "/props", timeout=10) as r:
            _lc["marker"] = json.loads(r.read())["media_marker"]
    system = f"<|im_start|>system\n{prompt}<|im_end|>\n" if prompt else ""
    return (f"{system}<|im_start|>user\n{_lc['marker']}<|im_end|>\n<|im_start|>assistant\n"
            f"language {LANGUAGES.get(language, language)}<asr_text>")


def _lc_complete(wav: bytes, prompt_string: str, n_probs: int = 0, grammar: str = "") -> list:
    body = {"prompt": {"prompt_string": prompt_string, "multimodal_data": [base64.b64encode(wav).decode()]},
            "temperature": 0, "n_predict": 200, "n_probs": n_probs, "cache_prompt": True, "id_slot": 0}
    if grammar:
        body["grammar"] = grammar
    return _lc_post("/completion", body).get("completion_probabilities") or []


def _lc_transcribe(wav: bytes, language: str, prompt: str, bias=None):
    head = f"language {LANGUAGES.get(language, language)}<asr_text>"
    p = _lc_prompt(prompt, language)
    b = asr_bias.load(BIAS_FILE) if bias and BIAS_FILE else None
    text, toks, lp = "", [], 0.0

    def score(at: str, tok: str, tok_lp: float) -> float:
        # the end of the transcript ends the last word, as in the vLLM plugin
        return tok_lp + b.weight(asr_bias.norm(at + (tok or " ")), bias[0], bias[1], bias[0])

    for _ in range(6):
        steps = _lc_complete(wav, p + text, n_probs=20)
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
    transcribe.logprob = lp
    return text.strip(), toks, head


def _lc_force(wav: bytes, prompt: str, text: str, tail: str = None) -> float:
    """Summed raw logprob of text with llama.cpp forced to write it (a grammar of just that string; the
    probabilities it reports are the model's own, before the grammar). With tail: text is given (in the
    prompt) and only tail is forced and scored."""
    head, _, body = text.partition("<asr_text>")
    language = next((k for k, v in LANGUAGES.items() if head == f"language {v}"), LANGUAGE)
    given, forced = (body, tail) if tail is not None else ("", body)
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


def name_check(raw: str, toks, head: str, wav: bytes, prompt: str, heard_lp: float) -> str:
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
    nr = asr_bias.norm(rest)
    for f in sorted(bias.kind_fillers.get(kind, []) + bias.fillers, key=len, reverse=True):
        if nr.startswith(f) and nr[len(f):].strip() and nr.strip() not in names:
            n = len(f.split())
            words = rest.split(" ")
            lead = [w for w in words if w]
            cut = rest.find(lead[n - 1]) + len(lead[n - 1])
            prefix, rest = prefix + rest[:cut], rest[cut:].lstrip()
            firsts = {}
            break
    heard = asr_bias.norm(rest).strip()
    if not heard or heard in names:
        return raw
    if any(f" {n} " in f" {heard} " for n in names):
        return raw  # a known name with more around it ("living room lights"): HA's own sentences
    whole = "".join(t for t, _, _ in toks)
    hs = _sounds(heard)

    def pick(n):
        s1 = max([lp for a, lp in firsts.items() if a and n.startswith(a)], default=-15.0)
        return max(s1, -15.0) + 6 * difflib.SequenceMatcher(None, hs, _sounds(n)).ratio()

    # The first letter alone pulls towards the wrong names (" D" -0.07 for "Zola" heard as "Dola"),
    # so half the shortlist is by sound only.
    by_sound = sorted(names, key=lambda n: difflib.SequenceMatcher(None, hs, _sounds(n)).ratio(),
                      reverse=True)
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
    texts = [whole] + [prefix + " " + names[c] + ending for c in cands]
    # what was heard is scored by its own decode (raw logprobs, before the plugin's push)
    if BACKEND == "llamacpp":
        # One slot, one at a time, and only from the name on: the words before it are the same for all,
        # so they go into the (cached) prompt and each candidate costs just its own few tokens (forcing
        # the whole sentence each time took ~1 s a check on a CPU). What was heard is scored the same way.
        heard_tail = whole[len(prefix):] if whole.startswith(prefix) else " " + rest + ending
        tails = [heard_tail] + [" " + names[c] + ending for c in cands]
        scores = [_lc_force(wav, prompt, head + prefix, t) for t in tails]
        if pruned:  # the bounds are of the name's first token alone: right in this space as they are
            pruned_best = max(b[c] for c in pruned)
    else:
        scores = [heard_lp] + list(_pool.map(lambda t: force_score(wav, prompt, head + t), texts[1:]))
    hyp, ranked = scores[0], sorted(zip(scores[1:], cands), reverse=True)
    (best, name), second = ranked[0], (ranked[1][0] if len(ranked) > 1 else -1e9)
    if firsts and pruned:
        second = max(second, pruned_best)
    ok = best >= hyp - SLOT_PRIOR and best - second >= SLOT_MARGIN
    log.info("name check (%s): %r %.1f vs %s -> %s", kind, rest, hyp,
             ", ".join(f"{names[c]!r} {s:.1f}" for s, c in ranked[:3]), names[name] if ok else "kept")
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
            span, speech_ms = vad.span(len(pcm))
            t_vad = time.monotonic()
            t_dec = t_chk = t_vad
            raw = text = ""
            if span is None:
                log.info("%.1f s audio, %d ms speech: no speech, empty transcript",
                         len(pcm) / 32000, speech_ms)
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
                t_dec = t_chk = time.monotonic()
                if BIAS_FILE and raw:
                    try:
                        raw = await asyncio.to_thread(name_check, raw, toks, head, wav, ctx, heard_lp)
                    except Exception as e:
                        log.error("name check failed: %s", e)
                    t_chk = time.monotonic()
                text = strip_wake(raw)
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


async def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    await context()
    await warm_up()
    log.info("listening on %d, %s", PORT, f"llama.cpp at {LLAMA_URL}" if BACKEND == "llamacpp" else f"vLLM at {VLLM_URL}")
    await AsyncServer.from_uri(f"tcp://0.0.0.0:{PORT}").run(Handler)


if __name__ == "__main__":
    asyncio.run(main())
