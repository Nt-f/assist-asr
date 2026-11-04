#!/usr/bin/env python3
"""Make a synthetic test set: sentences said by OmniVoice's designed voices (nobody's voice cloned).

  python bench/synth.py OUT --texts sentences.txt            # one sentence a line ("text<TAB>name" optional)
  python bench/synth.py OUT --names data/bias.json [--per-name 2] [--outside 20]

--names takes the lists the Wyoming server wrote (your Music Assistant library and HA names) and
makes "Alexa, play <name>." / "shuffle" / "turn off the <name>." sentences from a sample of them,
plus names that are not in your lists ("play Taylor Swift"), which must come out unchanged.

Writes OUT/<id>.wav (16 kHz mono) and OUT/manifest.jsonl, the benchmark's dataset format.
Needs OmniVoice on a GPU (README "Benchmark"): uv pip install omnivoice==0.2.1 torch==2.9.1
torchaudio==2.9.1 transformers==5.16.1. ~4 GB of GPU memory. Seeded: the same input makes the same set.
"""
import argparse
import json
import os
import random
import wave
import zlib

import numpy as np

VOICES = {
    "us-f-young": "female, young adult, american accent",
    "us-m-middle": "male, middle-aged, low pitch, american accent",
    "us-m-elderly": "male, elderly, low pitch, american accent",
    "gb-f-elderly": "female, elderly, british accent",
    "gb-m-young": "male, young adult, british accent",
    "au-f-middle": "female, middle-aged, australian accent",
    "in-m-young": "male, young adult, indian accent",
    "ca-f-teen": "female, teenager, canadian accent",
}
REFERENCE = "A few friends are coming over tonight, we will cook together and play a round of cards afterwards."
MUSIC = ["Alexa, play {}.", "Alexa, shuffle {}.", "Alexa, play some {}.", "Alexa, put on {}."]
HOME = ["Alexa, turn on the {}.", "Alexa, turn off the {}.", "Alexa, turn off {}."]
OUTSIDE = ["Taylor Swift", "the Beatles", "jazz", "classical music", "Bohemian Rhapsody", "lo-fi beats",
           "Fleetwood Mac", "Hamilton", "Mozart", "Ed Sheeran", "relaxing piano", "the radio", "Coldplay",
           "Billie Eilish", "Abbey Road", "workout music", "Beyonce", "Frank Sinatra", "rain sounds", "Adele"]


def sentences(bias_json, per_name, outside, seed=7):
    d = json.load(open(bias_json))
    rnd = random.Random(seed)
    music = sorted({n for n in d["lists"].get("music", []) if 2 < len(n) < 40})
    generic = {"music", "some music", "something", "anything", "radio", "the radio", "my music", "my playlist",
               "playlist", "songs", "a song", "lights", "light", "lamp", "lamps", "fan", "tv", "heating",
               "everything", "blinds", "curtains", "shades", "garage door", "door", "front door", "window", "windows"}
    music = [n for n in music if n.lower() not in generic]
    home = sorted({n for n in d["lists"].get("home", []) if n.lower() not in generic and 2 < len(n) < 40})
    out = []
    for n in rnd.sample(music, min(40, len(music))):
        for t in rnd.sample(MUSIC, per_name):
            out.append((t.format(n), n, "music", True))
    for n in rnd.sample(home, min(15, len(home))):
        for t in rnd.sample(HOME, min(per_name, len(HOME))):
            out.append((t.format(n), n, "home", True))
    for n in OUTSIDE[:outside]:
        out.append((rnd.choice(MUSIC).format(n), n, "music", False))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--texts")
    ap.add_argument("--names")
    ap.add_argument("--per-name", type=int, default=2)
    ap.add_argument("--outside", type=int, default=20)
    a = ap.parse_args()
    if a.texts:
        items = []
        for l in open(a.texts, encoding="utf-8"):
            l = l.rstrip("\n")
            if l.strip() and not l.startswith("#"):
                t, _, n = l.partition("\t")
                items.append((t.strip(), n.strip() or None, None, None))
    elif a.names:
        items = sentences(a.names, a.per_name, a.outside)
    else:
        ap.error("--texts or --names")

    import torch
    import torchaudio
    from omnivoice.models.omnivoice import OmniVoice

    m = OmniVoice.from_pretrained("k2-fsa/OmniVoice", device_map="cuda", dtype=torch.float16)
    os.makedirs(os.path.join(a.out, "voices"), exist_ok=True)

    def wav16(au):
        x = torchaudio.functional.resample(torch.as_tensor(au, dtype=torch.float32), m.sampling_rate, 16000).numpy()
        return np.clip(np.round(x * 32767), -32768, 32767).astype("<i2")

    def write(path, pcm):
        with wave.open(path, "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000); w.writeframes(pcm.tobytes())

    refs = {}
    for v, ins in VOICES.items():
        p = os.path.join(a.out, "voices", v + ".wav")
        if not os.path.exists(p):
            torch.manual_seed(zlib.crc32(v.encode()))
            write(p, wav16(m.generate(text=REFERENCE, language="en", instruct=ins)[0]))
        refs[v] = m.create_voice_clone_prompt(ref_audio=p, ref_text=REFERENCE)

    voices = list(VOICES)
    rows = []
    for i, (text, name, kind, listed) in enumerate(items):
        rows.append(dict(id=f"{i:04d}", file=f"{i:04d}.wav", text=text, voice=voices[i % len(voices)],
                         **({"name": name} if name else {}), **({"kind": kind} if kind else {}),
                         **({"in_list": listed} if listed is not None else {})))
    for v in voices:
        mine = [r for r in rows if r["voice"] == v]
        for b in range(0, len(mine), 16):
            batch = mine[b:b + 16]
            torch.manual_seed(zlib.crc32(f"{v} {b}".encode()))
            aus = m.generate(text=[r["text"] for r in batch], language=["en"] * len(batch),
                             voice_clone_prompt=[refs[v]] * len(batch))
            for r, au in zip(batch, aus):
                write(os.path.join(a.out, r["file"]), wav16(au))
            print(f"{v}: {min(b + 16, len(mine))} of {len(mine)}", flush=True)
    with open(os.path.join(a.out, "manifest.jsonl"), "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{len(rows)} clips in {a.out}")


if __name__ == "__main__":
    main()
