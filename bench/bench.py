#!/usr/bin/env python3
"""Benchmark speech-to-text systems on datasets of short voice-assistant requests.

  python bench/bench.py run --tag 1.7b \\
      --system ours=wyoming://127.0.0.1:10300 --system stock=vllm://127.0.0.1:8100 \\
      --dataset data/sets/names --dataset data/sets/rec [--noise 5]
  python bench/bench.py report bench/results/*.json            # markdown tables
  python bench/bench.py diff bench/results/1.7b.json ours stock  # clips where two systems differ

A dataset is a folder of 16 kHz mono WAVs with manifest.jsonl, one line per clip:
  {"file": "0001.wav", "text": "Alexa, play Zola.", "name": "Zola"}
("name" optional: then "names" counts whether that name came out right). Without a manifest every
<clip>.wav needs a <clip>.txt beside it with the words said.

Systems:
  wyoming://host:port   any Wyoming speech-to-text server (this project's, wyoming-whisper, ...)
  vllm://host:port      vLLM's /v1/audio/transcriptions as is: the model with nothing around it
                        ("stock"); add /<served model name> if it is not qwen3-asr
  llamacpp://host:port  llama.cpp's llama-server with a Qwen3-ASR GGUF and its mmproj (CPU, Vulkan
                        on an iGPU, CUDA...), English forced as vLLM does

Scoring, after a leading wake word is dropped from both sides and numbers are spelled out:
  exact  the transcript's words are the words said (punctuation and case aside)
  WER    word error rate over all clips
  names  of the clips with a "name": that name is in the transcript, spelled as listed
  intent with --hass: of the clips whose true sentence Home Assistant's own sentences understand, the
         share where the transcript gets the same intent with the same names (HA's Assist debug API:
         nothing is carried out). HASS_URL/HASS_TOKEN from the environment or ./.env
  ms     median time from the end of the audio to the transcript (--realtime: audio streamed at
         speaking pace, so work a server does while it arrives is not counted, as in real use)
--noise N adds a second pass with room noise mixed in at N dB below the speech (seeded: the
same noise every run).
"""
import argparse
import asyncio
import glob
import io
import json
import os
import re
import statistics
import sys
import time
import urllib.request
import uuid
import wave
import zlib

import numpy as np

WAKE = r"(?:(?:hey|ok(?:ay)?)[\s,.!]+)?(?:alexa|alexis|alex|amazon|computer|echo|ziggy|jarvis|mycroft|nabu)"

ONES = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen " \
       "sixteen seventeen eighteen nineteen".split()
TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()


def spell(n: int) -> str:
    if n < 20:
        return ONES[n]
    if n < 100:
        return TENS[n // 10] + ("" if n % 10 == 0 else " " + ONES[n % 10])
    if n < 1000:
        return ONES[n // 100] + " hundred" + ("" if n % 100 == 0 else " " + spell(n % 100))
    return str(n)


def words(s: str) -> list:
    s = re.sub(rf"^\s*{WAKE}[\s,.!?;:]+", "", s, flags=re.I)
    s = re.sub(rf"^\s*{WAKE}\s*[.!?]?\s*$", "", s, flags=re.I)
    s = s.lower().replace("%", " percent").replace("&", " and ").replace("’", "'").replace("'", "")
    s = re.sub(r"\d+", lambda m: " " + spell(int(m.group())) + " ", s)
    return re.sub(r"[^a-z0-9À-ɏ]+", " ", s).split()


def edits(a: list, b: list) -> int:
    d = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        prev, d[0] = d[0], i
        for j, y in enumerate(b, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (x != y))
    return d[-1]


def load_dataset(path: str) -> list:
    man = os.path.join(path, "manifest.jsonl")
    rows = []
    if os.path.exists(man):
        for l in open(man, encoding="utf-8"):
            if l.strip():
                r = json.loads(l)
                rows.append(dict(r, path=os.path.join(path, r["file"])))
    else:
        for w in sorted(glob.glob(os.path.join(path, "*.wav"))):
            t = w[:-4] + ".txt"
            if os.path.exists(t):
                rows.append({"file": os.path.basename(w), "text": open(t).read().strip(), "path": w})
    if not rows:
        sys.exit(f"{path}: no clips (manifest.jsonl, or <clip>.wav + <clip>.txt)")
    return rows


def read_pcm(path: str) -> bytes:
    with wave.open(path) as w:
        if w.getframerate() != 16000 or w.getnchannels() != 1 or w.getsampwidth() != 2:
            sys.exit(f"{path}: needs 16 kHz mono 16 bit (ffmpeg -i in -ar 16000 -ac 1 -c:a pcm_s16le out.wav)")
        return w.readframes(w.getnframes())


def add_noise(pcm: bytes, snr_db: float, seed: int) -> bytes:
    """Room-like noise (brownish, the lowest rumble taken out) at snr_db under the speech."""
    rng = np.random.default_rng(seed)
    x = np.frombuffer(pcm, np.int16).astype(float)
    n = np.cumsum(rng.standard_normal(len(x)))
    n -= np.convolve(n, np.ones(800) / 800, "same")
    n *= np.sqrt(np.mean(x ** 2) + 1e-9) / np.sqrt(np.mean(n ** 2) + 1e-9) * 10 ** (-snr_db / 20)
    return np.clip(x + n, -32768, 32767).astype(np.int16).tobytes()


def wav_bytes(pcm: bytes) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000); w.writeframes(pcm)
    return buf.getvalue()


async def run_wyoming(host: str, port: int, pcm: bytes, language: str, realtime: bool = False):
    from wyoming.asr import Transcribe, Transcript
    from wyoming.audio import AudioChunk, AudioStart, AudioStop
    from wyoming.client import AsyncTcpClient

    c = AsyncTcpClient(host, port)
    await c.connect()
    try:
        # no_record: this project's server keeps no recording of benchmark clips (others ignore it)
        await c.write_event(Transcribe(language=language, context={"no_record": True}).event())
        await c.write_event(AudioStart(16000, 2, 1).event())
        start = time.monotonic()
        for i in range(0, len(pcm), 2048):
            if realtime:  # as a satellite streams: 64 ms of audio every 64 ms
                await asyncio.sleep(max(0.0, start + i / 32000 - time.monotonic()))
            await c.write_event(AudioChunk(16000, 2, 1, pcm[i:i + 2048]).event())
        t = time.monotonic()
        await c.write_event(AudioStop().event())
        while True:
            e = await asyncio.wait_for(c.read_event(), 60)
            if e is None:
                raise ConnectionError("closed")
            if Transcript.is_type(e.type):
                return Transcript.from_event(e).text, time.monotonic() - t
    finally:
        await c.disconnect()


def run_vllm(url: str, model: str, pcm: bytes, language: str):
    b = uuid.uuid4().hex
    fields = {"model": model, "language": language, "temperature": "0", "response_format": "json"}
    body = b"".join(f'--{b}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode()
                    for k, v in fields.items())
    body += (f'--{b}\r\nContent-Disposition: form-data; name="file"; filename="a.wav"\r\n'
             f"Content-Type: audio/wav\r\n\r\n").encode() + wav_bytes(pcm) + f"\r\n--{b}--\r\n".encode()
    req = urllib.request.Request(url + "/v1/audio/transcriptions", data=body,
                                 headers={"Content-Type": f"multipart/form-data; boundary={b}"})
    t = time.monotonic()
    with urllib.request.urlopen(req, timeout=60) as r:
        text = json.loads(r.read())["text"]
    return text, time.monotonic() - t


LANG_NAMES = {"en": "English", "de": "German", "fr": "French", "es": "Spanish", "it": "Italian", "nl": "Dutch"}
_markers = {}


def run_llamacpp(url: str, pcm: bytes, language: str):
    """llama.cpp's llama-server with a Qwen3-ASR GGUF + mmproj: the raw prompt, so the language is forced
    as vLLM's transcription endpoint forces it ("language English<asr_text>" ends the prompt)."""
    import base64
    if url not in _markers:
        _markers[url] = json.load(urllib.request.urlopen(url + "/props", timeout=10))["media_marker"]
    p = (f"<|im_start|>user\n{_markers[url]}<|im_end|>\n<|im_start|>assistant\n"
         f"language {LANG_NAMES.get(language, language)}<asr_text>")
    body = {"prompt": {"prompt_string": p, "multimodal_data": [base64.b64encode(wav_bytes(pcm)).decode()]},
            "temperature": 0, "n_predict": 200, "cache_prompt": False}
    req = urllib.request.Request(url + "/completion", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t = time.monotonic()
    with urllib.request.urlopen(req, timeout=120) as r:
        text = json.loads(r.read())["content"]
    return text, time.monotonic() - t


def parse_system(spec: str):
    name, _, url = spec.partition("=")
    m = re.match(r"^(wyoming|vllm|llamacpp)://([^:/]+):(\d+)(?:/(.+))?$", url)
    if not name or not m:
        sys.exit(f"--system {spec}: expected name=wyoming://host:port, vllm://host:port[/model] or llamacpp://host:port")
    return name, m.group(1), m.group(2), int(m.group(3)), m.group(4) or "qwen3-asr"


def score(row: dict, hyp: str) -> dict:
    ref, got = words(row["text"]), words(hyp)
    out = {"exact": ref == got, "errors": edits(ref, got), "ref_words": len(ref)}
    if row.get("name"):
        n = " ".join(words(row["name"]))
        out["name_ok"] = f" {n} " in f" {' '.join(got)} "
    return out


async def cmd_run(a):
    systems = [parse_system(s) for s in a.system]
    conds = [("clean", None)] + ([(f"noise {a.noise:g} dB", a.noise)] if a.noise is not None else [])
    results = {"tag": a.tag, "runs": []}
    for ds in a.dataset:
        rows = load_dataset(ds)[:a.limit or None]
        dname = os.path.basename(os.path.normpath(ds))
        for cname, snr in conds:
            for sname, kind, host, port, model in systems:
                clips, t0 = [], time.monotonic()
                for i, r in enumerate(rows):
                    pcm = read_pcm(r["path"])
                    if snr is not None:
                        pcm = add_noise(pcm, snr, zlib.crc32(r["file"].encode()))
                    try:
                        if kind == "wyoming":
                            hyp, dt = await run_wyoming(host, port, pcm, a.language, a.realtime)
                        elif kind == "llamacpp":
                            hyp, dt = await asyncio.to_thread(run_llamacpp, f"http://{host}:{port}", pcm, a.language)
                        else:
                            hyp, dt = await asyncio.to_thread(run_vllm, f"http://{host}:{port}", model, pcm,
                                                              a.language)
                    except Exception as e:  # a failed request is a wrong transcript, and said so
                        print(f"  {r['file']}: {sname} failed: {e}", file=sys.stderr)
                        hyp, dt = "", float("nan")
                    clips.append(dict(file=r["file"], text=r["text"], name=r.get("name"),
                                      in_list=r.get("in_list"), hyp=hyp, ms=round(dt * 1000, 1),
                                      **score(r, hyp)))
                if a.hass:
                    await add_intents(clips, a.language)
                run = dict(dataset=dname, condition=cname, system=sname, clips=clips)
                results["runs"].append(run)
                s = summary(run)
                print(f"{a.tag:8} {sname:10} {dname:12} {cname:12} exact {s['exact']:5.1f}%  "
                      f"WER {s['wer']:5.1f}%  names {s['names']}  "
                      f"{'intent ' + s['intent'] + '  ' if a.hass else ''}{s['ms']:.0f} ms  "
                      f"({len(clips)} clips, {time.monotonic() - t0:.0f} s)", flush=True)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)
    print(f"-> {a.out}")


def hass_env():
    env = {k: os.environ.get(k, "") for k in ("HASS_URL", "HASS_TOKEN")}
    if not all(env.values()) and os.path.exists(".env"):
        for l in open(".env"):
            k, _, v = l.strip().partition("=")
            if k in env and not env[k]:
                env[k] = v
    if not all(env.values()):
        sys.exit("--hass: set HASS_URL and HASS_TOKEN (environment or ./.env)")
    return env["HASS_URL"].rstrip("/"), env["HASS_TOKEN"]


async def hass_understands(sentences: list, language: str) -> list:
    """What HA's built-in agent makes of each sentence, nothing carried out: (intent, slots, all matched)."""
    import aiohttp
    url, token = hass_env()
    out = []
    async with aiohttp.ClientSession() as s, s.ws_connect(url.replace("http", "ws", 1) + "/api/websocket") as ws:
        await ws.receive_json()
        await ws.send_json({"type": "auth", "access_token": token})
        if (await ws.receive_json()).get("type") != "auth_ok":
            sys.exit("--hass: HA refused the token")
        for i in range(0, len(sentences), 50):
            await ws.send_json({"id": i + 1, "type": "conversation/agent/homeassistant/debug",
                                "sentences": sentences[i:i + 50] or [""], "language": language})
            r = await ws.receive_json()
            for res in (r.get("result") or {}).get("results") or [None] * len(sentences[i:i + 50]):
                if not res:
                    out.append(None)
                    continue
                # a slot is {"value": ..., "text": ...}, or for some intents just its value
                slots = tuple(sorted((k, json.dumps(v.get("value") if isinstance(v, dict) else v,
                                                    sort_keys=True).lower())
                                     for k, v in (res.get("slots") or {}).items()))
                out.append((res["intent"]["name"], slots, bool(res.get("match"))))
    return out


async def add_intents(clips: list, language: str) -> None:
    unwake = lambda s: re.sub(rf"^\s*{WAKE}[\s,.!?;:]+", "", s, flags=re.I).strip()
    sents = [unwake(c["text"]) for c in clips] + [unwake(c["hyp"]) for c in clips]
    res = await hass_understands(sents, language)
    for c, ref, hyp in zip(clips, res[:len(clips)], res[len(clips):]):
        # only where HA understands what was really said; a hit = the same intent and names, fully matched
        c["intent_ok"] = (hyp == ref) if ref and ref[2] else None


def summary(run: dict) -> dict:
    c = run["clips"]
    for x in c:  # scored again from the transcripts: results of older runs stay comparable
        x.update(score({"text": x["text"], "name": x.get("name")}, x["hyp"]))
    named = [x for x in c if "name_ok" in x]
    ms = [x["ms"] for x in c if x["ms"] == x["ms"]]
    return {"n": len(c), "exact": 100 * sum(x["exact"] for x in c) / len(c),
            "wer": 100 * sum(x["errors"] for x in c) / max(1, sum(x["ref_words"] for x in c)),
            "names": f"{100 * sum(x['name_ok'] for x in named) / len(named):.0f}%" if named else "-",
            "intent": (lambda k: f"{100 * sum(x['intent_ok'] for x in k) / len(k):.0f}% of {len(k)}" if k else "-")(
                [x for x in c if x.get("intent_ok") is not None]),
            "ms": statistics.median(ms) if ms else float("nan")}


def cmd_report(a):
    runs = []
    for p in a.results:
        r = json.load(open(p, encoding="utf-8"))
        runs += [dict(x, tag=r["tag"]) for x in r["runs"]]
    cols = list(dict.fromkeys(f"{x['tag']} {x['system']}" for x in runs))
    sets = list(dict.fromkeys((x["dataset"], x["condition"]) for x in runs))
    by = {(f"{x['tag']} {x['system']}", x["dataset"], x["condition"]): summary(x) for x in runs}
    for metric, title, fmt in (("exact", "Exactly right (%)", "{:.1f}"), ("wer", "Word error rate (%)", "{:.1f}"),
                               ("names", "Names right", "{}"),
                               ("intent", "Home Assistant understood it as said (of the clips it understands)", "{}"),
                               ("ms", "Median latency (ms)", "{:.0f}")):
        if metric in ("names", "intent") and all(by[k][metric] == "-" for k in by):
            continue
        print(f"\n### {title}\n")
        print("| dataset | " + " | ".join(cols) + " |")
        print("|---|" + "---:|" * len(cols))
        for ds, cond in sets:
            n = next((by[k]["n"] for k in by if k[1:] == (ds, cond)), 0)
            cells = [fmt.format(by[(c, ds, cond)][metric]) if (c, ds, cond) in by else "" for c in cols]
            print(f"| {ds}, {cond} ({n}) | " + " | ".join(cells) + " |")


def cmd_diff(a):
    r = json.load(open(a.result, encoding="utf-8"))
    runs = {(x["system"], x["dataset"], x["condition"]): x for x in r["runs"]}
    for (s, ds, cond), x in runs.items():
        if s != a.a or (a.b, ds, cond) not in runs:
            continue
        y = runs[(a.b, ds, cond)]
        summary(x), summary(y)
        for p, q in zip(x["clips"], y["clips"]):
            # better = exactly right, else at least the name right
            rank_p, rank_q = (p["exact"], bool(p.get("name_ok"))), (q["exact"], bool(q.get("name_ok")))
            if rank_p != rank_q:
                print(f"{ds}, {cond}: {a.a if rank_p > rank_q else a.b} better: {p['text']!r}\n"
                      f"    {a.a}: {p['hyp']!r}\n    {a.b}: {q['hyp']!r}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--system", action="append", required=True)
    r.add_argument("--dataset", action="append", required=True)
    r.add_argument("--tag", default="run")
    r.add_argument("--noise", type=float)
    r.add_argument("--language", default="en")
    r.add_argument("--limit", type=int, help="only the first N clips of each dataset")
    r.add_argument("--hass", action="store_true",
                   help="also score whether Home Assistant understands each transcript as the truth (intent)")
    r.add_argument("--realtime", action="store_true",
                   help="stream audio to Wyoming servers at speaking pace, as a satellite does (slower)")
    r.add_argument("--out")
    p = sub.add_parser("report")
    p.add_argument("results", nargs="+")
    d = sub.add_parser("diff")
    d.add_argument("result")
    d.add_argument("a")
    d.add_argument("b")
    a = ap.parse_args()
    if a.cmd == "run":
        a.out = a.out or os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", f"{a.tag}.json")
        asyncio.run(cmd_run(a))
    elif a.cmd == "report":
        cmd_report(a)
    else:
        cmd_diff(a)


if __name__ == "__main__":
    main()
