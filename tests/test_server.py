#!/usr/bin/env python3
"""Unit tests of wyoming/server.py's own logic, without a model (stand-ins for llama-server).

  python tests/test_server.py        # or pytest tests/test_server.py
  docker run --rm -v "$PWD:/src" -w /src --entrypoint python assist-asr-light-wyoming tests/test_server.py

Needs what the Wyoming image has (wyoming, pysilero-vad).
"""
import json
import math
import os
import random
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
TMP = tempfile.mkdtemp()
os.environ.update(BACKEND="llamacpp", BIAS_FILE=os.path.join(TMP, "bias.json"), CACHE_FILE=os.path.join(TMP, "c.json"),
                  COMMANDS_FILE=os.path.join(HERE, "..", "config", "commands.txt"), LLAMA_WAIT="5")
sys.path[:0] = [os.path.join(HERE, "..", "wyoming"), os.path.join(HERE, "..", "plugin")]
import server  # noqa: E402

VOCAB = ", ".join(l.split("#")[0].strip() for l in open(os.path.join(HERE, "..", "config", "vocab.txt"))
                  if l.split("#")[0].strip())
CTX = "Kitchen, Living Room, Standing Lamp, Boiler. " + VOCAB


def test_echo_guard():
    e = server.echoes_context
    # issue #1: what came back for "Next track", and the same read back short
    assert e("Home Assistant, Drop In, Music Assistant, set a timer, timer, alarm, snooze.", CTX)
    assert e("Alexa, Home Assistant, Drop In, Music Assistant.", CTX)
    assert e("Kitchen, Living Room, Standing Lamp.", CTX)
    for ok in ["Next track.", "Set a timer for five minutes.", "Turn off the standing lamp in the kitchen.",
               "Alexa, snooze.",
               "Set a timer, cancel the timer.", "Volume up, volume down."]:
        assert not e(ok, CTX), ok
    assert not e("Home Assistant, Drop In, Music Assistant.", "")


class FakeLlama(BaseHTTPRequestHandler):
    """llama-server as far as _lc_post sees it: a marker of its own per 'start'; a prompt with another
    gets the 400 issue #1 saw."""
    marker = "<__media_one__>"
    seen = []

    def log_message(self, *a):
        pass

    def _send(self, code, body):
        b = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        self._send(200, {"media_marker": FakeLlama.marker})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        p = body["prompt"]["prompt_string"]
        FakeLlama.seen.append(p)
        if p.count(FakeLlama.marker) != 1:
            # what llama.cpp b12098 says (older ones: "number of media markers in text (0) does not match ...")
            return self._send(400, {"error": {"code": 400, "message": "Failed to tokenize prompt"}})
        self._send(200, {"completion_probabilities": [{"token": "Next", "logprob": -0.1, "top_logprobs": []}]})


def test_marker_after_restart():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), FakeLlama)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    server.LLAMA_URL = f"http://127.0.0.1:{srv.server_port}"
    server._lc["marker"] = None
    try:
        p = server._lc_prompt("", "en")
        assert server._lc_complete(b"", p)[0]["token"] == "Next"
        FakeLlama.marker = "<__media_two__>"  # llama-server restarted: a new marker
        assert server._lc_complete(b"", p)[0]["token"] == "Next"
        assert "<__media_two__>" in FakeLlama.seen[-1]
        assert server.MEDIA not in FakeLlama.seen[-1]
    finally:
        srv.shutdown()


def test_name_with_filler():
    """issue #1: "Play the Fate of Ophelia" heard as "the Fate of a Feeling": "the" is a filler, so the
    candidates were "Play the The Fate of Ophelia" and the right name never won. Scored by a stand-in
    that likes exactly the right sentence."""
    import asyncio
    music = ["The Fate of Ophelia", "Fate of the World", "Feeling Good", "The Feeling", "Zola"]
    asyncio.run(server.apply_names(["Kitchen", "Standing Lamp"], music))
    calls = []

    def force(wav, prompt, given, tail=None):
        calls.append((given, tail))
        _, _, said = given.partition("<asr_text>")
        whole = said + tail
        return -1.0 if whole == "Play The Fate of Ophelia." else -6.0 if whole == "Play the Fate of a Feeling." else -20.0

    server._lc_force, real = force, server._lc_force
    try:
        toks = [(t, -0.2, {t: -0.2}) for t in ["Play", " the", " Fate", " of", " a", " Feeling", "."]]
        out = server.name_check("Play the Fate of a Feeling.", toks, "language English<asr_text>", b"", "", -3.0)
    finally:
        server._lc_force = real
    assert out == "Play The Fate of Ophelia.", out
    assert not any("the The" in (g + (t or "")) for g, t in calls), calls
    # a filler before a name that doesn't start with it stays: "play some Zola" heard as "some Dola"
    server._lc_force = lambda wav, prompt, given, tail=None: (
        -1.0 if given.partition("<asr_text>")[2] + tail == "Play some Zola." else -20.0)
    try:
        toks = [(t, -0.2, {t: -0.2}) for t in ["Play", " some", " D", "ola", "."]]
        out = server.name_check("Play some Dola.", toks, "language English<asr_text>", b"", "", -3.0)
    finally:
        server._lc_force = real
    assert out == "Play some Zola.", out


def _tone_speech(level_db: float, secs: float = 1.0) -> bytes:
    """Not speech: for the VAD's gain path only (that it runs, and leaves noise alone)."""
    import array
    r = random.Random(1)
    amp = 32768 * 10 ** (level_db / 20)
    return array.array("h", (int(max(-32768, min(32767, r.gauss(0, amp)))) for _ in range(int(16000 * secs)))).tobytes()


def test_vad_boost_leaves_noise():
    for lvl in (-70, -50, -40):
        v = server.StreamVad()
        pcm = _tone_speech(lvl, 2.0)
        v.feed(pcm)
        span, ms = v.span(len(pcm), pcm)
        assert span is None and ms == 0, (lvl, ms)


def test_strip_wake():
    assert server.strip_wake("Alexa, stop.") == "Stop."
    assert server.strip_wake("Alexa.") == ""
    assert server.strip_wake("Alexander the Great.") == "Alexander the Great."


if __name__ == "__main__":
    n = 0
    for k, f in list(globals().items()):
        if k.startswith("test_") and callable(f):
            f()
            n += 1
            print("ok", k)
    print(f"{n} passed")
