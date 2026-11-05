"""Biasing Qwen3-ASR towards home-assistant words while it decodes, by position in the request.

A vLLM logits processor (shallow fusion with prefix tries). At each step the transcript so far
says where we are:

  [wake word] <command> [<slot>]
  "Alexa,"     "shuffle"  "Zola"

- In the wake word: nothing (the wrapper strips it; it is never boosted, so never made up).
- At the command: tokens that continue a command phrase ("shuffle", "turn on", "pause",
  "what time is it") get W_CMD added to their logit.
- After a command that opens a slot ("play" -> music, "turn off" -> home): tokens that continue
  a name of that list (Music Assistant's library, HA's entity and area names) get W_SLOT.
- Anywhere else (free speech, a question past its first words): nothing.

Matching is on normalised characters (lowercase, punctuation as spaces), not on tokens: the
model may spell "Zola" " Z"+"ol"+"a" or " Zo"+"la" and both are on the path. Only the top
TOP_K tokens by the model's own logit are looked at, so a name the audio does not support is
never pulled in: the boost only decides between what the model already finds plausible
(measured on a real artist's name: its first letter at -0.14, then its next piece 0.6 behind a wrong one).

Second use, force = a transcript: the decoder may only write that text (any tokenisation of it,
the model's likeliest), so the logprobs vLLM returns (raw, before this processor) score how well the
audio supports it. The wrapper uses it to weigh library names against what was heard ("Dola" ->
"Zola"): the first letter alone cannot tell names apart, the whole word can.

Per request (vllm_xargs): asr_bias = path of the bias JSON (written by the Wyoming wrapper,
reread when it changes), w_cmd, w_slot, top_k. Without asr_bias the request is left alone.
"""

import bisect
import json
import os
import re
import threading

_NON = re.compile(r"[^0-9a-zÀ-ɏ]+")


def norm(s: str) -> str:
    """Lowercase words with single spaces; punctuation is a word break. Leading space dropped,
    a trailing one kept (it says the last word is complete)."""
    s = s.lower().replace("&", " and ").replace("’", "").replace("'", "")
    return _NON.sub(" ", s).lstrip()


def on_path(lst, p: str) -> bool:
    """p is a prefix of a phrase of the sorted list (phrases end with a space)."""
    i = bisect.bisect_left(lst, p)
    return i < len(lst) and lst[i].startswith(p)


class Bias:
    def __init__(self, d: dict):
        def phrases(xs):
            return sorted({n + " " for n in (norm(x).strip() for x in xs) if n})

        self.wake = phrases(d.get("wake", []))
        self.lead = [norm(x).strip() + " " for x in d.get("lead", ["hey", "ok", "okay"])]
        self.openers = {norm(k).strip(): v for k, v in d.get("openers", {}).items()}
        self.cmds = phrases(list(d.get("commands", [])) + list(self.openers))
        self.lists = {k: phrases(v) for k, v in d.get("lists", {}).items()}
        self.fillers = [norm(f).strip() + " " for f in d.get("fillers", {}).get("*", [])]
        self.kind_fillers = {k: [norm(f).strip() + " " for f in v]
                             for k, v in d.get("fillers", {}).items() if k != "*"}

    def region(self, t: str):
        """(kind, partial): kind "cmd", a slot list's name, or None (no bias here)."""
        for pre in self.lead:
            if t.startswith(pre) and any(t[len(pre):].startswith(w[:2]) for w in self.wake):
                t = t[len(pre):]
                break
        for w in self.wake:
            if t.startswith(w):
                t = t[len(w):]
                break
        else:
            if t and any(w.startswith(t) for w in self.wake):
                return None, t  # inside the wake word
        best = None
        for o, kind in self.openers.items():
            if t.startswith(o + " ") and (best is None or len(o) > len(best[0])):
                best = (o, kind)
        if best:
            return best[1], t[len(best[0]) + 1:]
        return "cmd", t

    def weight(self, t: str, w_cmd: float, w_slot: float, w_wake: float = 0.0) -> float:
        kind, p = self.region(t)
        if kind is None:
            # Inside the wake word: with "Wake word in the request" every request starts with it, and
            # the wrapper strips it, so one made up costs nothing; a mangled one ("Lexa allowed her")
            # takes the command with it.
            return w_wake if w_wake and p.strip() and on_path(self.wake, p) else 0.0
        if kind is None or not p.strip():
            return 0.0  # nothing of a phrase yet (e.g. the space after "shuffle")
        if kind == "cmd":
            if not on_path(self.cmds, p):
                return 0.0
            if p.endswith(" ") and p.strip() in self.openers:
                return 0.0  # "shuffle." is a command that needs a slot: no push to end it
            return w_cmd
        lst = self.lists.get(kind)
        if not lst:
            return 0.0
        if on_path(lst, p):
            return w_slot
        for f in self.kind_fillers.get(kind, []) + self.fillers:
            if p.startswith(f) and p[len(f):].strip() and on_path(lst, p[len(f):]):
                return w_slot
            if f.startswith(p):  # "the" of "turn off the lamp": dropped when only the name had a push
                return w_slot
        return 0.0


_cache = {}
_lock = threading.Lock()


def load(path: str):
    try:
        st = os.stat(path)
    except OSError:
        return None
    key = (st.st_mtime_ns, st.st_size)
    with _lock:
        c = _cache.get(path)
        if c and c[0] == key:
            return c[1]
        try:
            with open(path, encoding="utf-8") as f:
                b = Bias(json.load(f))
        except (OSError, ValueError):
            return c[1] if c else None
        _cache[path] = (key, b)
        return b


try:  # the vLLM side (model runner V2's interface, vLLM 0.31); imports without vLLM for tests
    from vllm.v1.worker.gpu.sample.logits_processor import LogitsProcessor as _Base
except ImportError:  # pragma: no cover
    _Base = object


class AsrBias(_Base):
    """Requests without vllm_xargs asr_bias are declined in add_request and never touched."""

    def __init__(self, vllm_config, req_states):
        self.rs = req_states
        self.cfg = vllm_config
        self.req = {}  # slot -> (path, w_cmd, w_slot, top_k, w_wake)
        self.force = {}  # slot -> text the decoder must write
        self.done = set()  # slots past every command and name: nothing left to push, no GPU reads
        self.vocab = None

    @classmethod
    def validate_params(cls, sp):
        xa = sp.extra_args or {}
        if "asr_bias" in xa:
            for k in ("w_cmd", "w_slot", "top_k"):
                if k in xa:
                    float(xa[k])

    def _vocab(self):
        # Lazily: the frontend process loads this class too (to validate), and needs no vocab.
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(self.cfg.model_config.tokenizer, trust_remote_code=True)
        vocab = [None] * max(len(tok), self.rs.vocab_size)
        special = set(tok.all_special_ids) | set(tok.get_added_vocab().values())
        ends = {tok.convert_tokens_to_ids(t) for t in ("<|im_end|>", "<|endoftext|>")}
        for i in range(len(tok)):
            if i in ends:
                vocab[i] = " "  # the end of the transcript ends the last word
            elif i not in special:
                s = tok.decode([i])
                if "\ufffd" not in s:
                    vocab[i] = s
        self.marker = tok.convert_tokens_to_ids("<asr_text>")
        self.vocab = vocab
        # Forcing: every token's text, special ones too, indexed by its first characters.
        self.ftext = [tok.decode([i]) for i in range(len(tok))]
        self.by1, self.by2 = {}, {}
        for i, t in enumerate(self.ftext):
            if not t or "\ufffd" in t:
                continue
            if len(t) == 1:
                self.by1.setdefault(t, []).append(i)
            else:
                self.by2.setdefault(t[:2], []).append(i)
        self.end_id = tok.convert_tokens_to_ids("<|im_end|>")

    def add_request(self, req_idx, sp):
        xa = sp.extra_args or {}
        self.req.pop(req_idx, None)
        self.force.pop(req_idx, None)
        self.done.discard(req_idx)
        if xa.get("force"):
            self.force[req_idx] = str(xa["force"])
            return True
        if not xa.get("asr_bias"):
            return False
        w_cmd = float(xa.get("w_cmd", 3.0))
        self.req[req_idx] = (xa["asr_bias"], w_cmd, float(xa.get("w_slot", 2.5)),
                             int(xa.get("top_k", 40)), float(xa.get("w_wake", w_cmd)),
                             str(xa.get("head", "")), not xa.get("no_early_stop"))
        return True

    def _force(self, logits, row, slot, end, text, ids=None, open_end=False):
        """Only tokens that continue text; at its end <|im_end|> (open_end: anything)."""
        if ids is None:
            start = int(self.rs.prompt_len.np[slot])
            ids = self.rs.all_token_ids.gpu[slot, start:end].tolist() if end > start else []
        done = "".join(self.ftext[i] for i in ids)
        rem = text[len(done):] if text.startswith(done) else ""
        if not rem and open_end:
            return False
        if rem:
            allowed = [i for i in self.by1.get(rem[0], [])]
            allowed += [i for i in self.by2.get(rem[:2], []) if rem.startswith(self.ftext[i])]
        else:
            allowed = []
        if not allowed:
            allowed = [self.end_id]
        keep = logits[row, allowed].clone()
        logits[row].fill_(float("-inf"))
        logits[row, allowed] = keep
        return True

    def apply(self, logits, ctx):
        import torch

        slots = ctx.idx_mapping_np
        if not any(int(s) in self.req or int(s) in self.force for s in slots):
            return logits
        if self.vocab is None:
            self._vocab()
        vocab = self.vocab
        if logits.shape[0] != len(slots):
            return logits  # speculative decoding: rows are not requests; not used here
        for row, slot in enumerate(slots):
            if int(slot) in self.force:
                self._force(logits, row, slot, int(ctx.seq_lens_upper_bound_np[row]),
                            self.force[int(slot)])
                continue
            r = self.req.get(int(slot))
            if r is None or int(slot) in self.done:
                continue
            path, w_cmd, w_slot, k, w_wake, head, early_stop = r
            start = int(self.rs.prompt_len.np[slot])
            end = int(ctx.seq_lens_upper_bound_np[row])
            # the tokens so far and the k likeliest next ones in one copy from the GPU (each is a sync)
            top_t = torch.topk(logits[row], k).indices
            if end > start:
                both = torch.cat([self.rs.all_token_ids.gpu[slot, start:end].to(top_t.dtype), top_t]).tolist()
                ids, top = both[:end - start], both[end - start:]
            else:
                ids, top = [], top_t.tolist()
            # head ("language English<asr_text>", maybe a wake phrase after it): forced until all of it is
            # written (_force says False once it is, or once the text went another way)
            if head and self._force(logits, row, slot, end, head, ids, True):
                continue
            bias = load(path)
            if bias is None:
                continue
            if self.marker in ids:  # language not forced: "language English<asr_text>..."
                ids = ids[ids.index(self.marker) + 1:]
            elif ids and (vocab[ids[0]] or "").strip().lower() == "language":
                continue
            text = "".join(vocab[i] or "" for i in ids)
            t_now = norm(text)
            kind, p = bias.region(t_now)
            lead = len(t_now.split()) <= 1 and any(l.startswith(t_now) or t_now.startswith(l) for l in bias.lead)
            if early_stop and kind is not None and p.strip() and not lead and not bias.weight(t_now, 1.0, 1.0, 1.0):
                # Off every path ("what is the cap..."): it cannot come back on one, so the rest of this
                # request is decoded as if the plugin were not there (each GPU read here costs ~1.4 ms).
                self.done.add(int(slot))
                continue
            idx, add = [], []
            for i in top:
                s = vocab[i] if i < len(vocab) else None
                if s is None:
                    continue
                w = bias.weight(norm(text + s), w_cmd, w_slot, w_wake)
                if w:
                    idx.append(i)
                    add.append(w)
            if idx:
                logits[row, idx] += torch.tensor(add, dtype=logits.dtype, device=logits.device)
        return logits
