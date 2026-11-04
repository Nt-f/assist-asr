#!/usr/bin/env python3
"""Turn the requests the Wyoming server recorded (RECORD_DIR, data/rec) into a benchmark dataset.

  python bench/label.py [data/rec] [--out data/sets/rec]

For each recording not labelled yet it prints what was heard and asks what was said:
  Enter       it heard right: keep that as the truth
  any text    what you really said (the wake word may be left out)
  p           play it (aplay, paplay or ffplay, whichever is there), then ask again
  s           skip for now        d   drop it (deleted from the recordings)        q   stop here
A name of your music or devices said in it? End the text with " | name", e.g.
  Alexa, shuffle Zola | Zola
Then the report also counts whether that name came out right.
Labels go to OUT/manifest.jsonl (the WAVs are copied there), so the recordings folder can be emptied.
"""
import argparse
import glob
import json
import os
import shutil
import subprocess


def play(path):
    for cmd in (["aplay", "-q"], ["paplay"], ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet"]):
        if shutil.which(cmd[0]):
            subprocess.run(cmd + [path])
            return
    print("  (no aplay, paplay or ffplay here)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("rec", nargs="?", default="data/rec")
    ap.add_argument("--out", default="data/sets/rec")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    man = os.path.join(a.out, "manifest.jsonl")
    done = set()
    if os.path.exists(man):
        done = {json.loads(l)["file"] for l in open(man, encoding="utf-8") if l.strip()}
    todo = [w for w in sorted(glob.glob(os.path.join(a.rec, "*.wav"))) if os.path.basename(w) not in done]
    print(f"{len(todo)} recordings to label, {len(done)} labelled in {man}")
    with open(man, "a", encoding="utf-8") as f:
        for w in todo:
            j = w[:-4] + ".json"
            heard = json.load(open(j)).get("raw", "") if os.path.exists(j) else ""
            while True:
                ans = input(f"\n{os.path.basename(w)}  heard: {heard!r}\n  said? ").strip()
                if ans == "p":
                    play(w)
                    continue
                break
            if ans == "q":
                break
            if ans == "s":
                continue
            if ans == "d":
                os.remove(w)
                if os.path.exists(j):
                    os.remove(j)
                continue
            text, _, name = (ans or heard).partition("|")
            row = {"file": os.path.basename(w), "text": text.strip()}
            if name.strip():
                row["name"] = name.strip()
            shutil.copy2(w, os.path.join(a.out, row["file"]))
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            print(f"  -> {row}")


if __name__ == "__main__":
    main()
