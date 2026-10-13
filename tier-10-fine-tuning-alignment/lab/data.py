"""Tiny char-level data world for the Tier 10 lab.

Base corpus: random 'sentences' of 2-5 words from a 12-word vocabulary (the "pretraining" text).
Tasks (the "instructions"):  up  -> upper-case the words        "up cat sun"  -> "CAT SUN"
                             rv  -> reverse the word order      "rv cat sun"  -> "sun cat"
Chat template: <|user|>{text}<|asst|>{answer}<|end|>   (specials are single token ids)
"""
import random
import re
import torch

WORDS = ["cat", "dog", "sun", "moon", "tree", "fish", "bird", "rain", "wind", "star", "leaf", "rock"]
SPECIALS = ["<|pad|>", "<|user|>", "<|asst|>", "<|end|>"]
CHARS = list("abcdefghijklmnopqrstuvwxyz ABCDEFGHIJKLMNOPQRSTUVWXYZ\n")
ITOS = SPECIALS + CHARS
STOI = {s: i for i, s in enumerate(ITOS)}
PAD, USER, ASST, END = (STOI[s] for s in SPECIALS)
VOCAB = len(ITOS)
_SPLIT = re.compile("(" + "|".join(re.escape(s) for s in SPECIALS) + ")")


def encode(text: str) -> list[int]:
    """Specials in the text become ONE id each (this is also the prompt-injection surface, see README)."""
    out = []
    for piece in _SPLIT.split(text):
        if piece in STOI and piece in SPECIALS:
            out.append(STOI[piece])
        else:
            out += [STOI[c] for c in piece]
    return out


def decode(ids) -> str:
    return "".join(ITOS[i] for i in ids)


def render_chat(messages: list[dict], add_generation_prompt: bool = False) -> str:
    """Chat template renderer (10.5). messages = [{'role':'user'|'assistant','content':...}]"""
    tag = {"user": "<|user|>", "assistant": "<|asst|>"}
    s = ""
    for m in messages:
        s += tag[m["role"]] + m["content"] + ("<|end|>" if m["role"] == "assistant" else "")
    if add_generation_prompt:
        s += "<|asst|>"
    return s


def answer(task: str, words: list[str]) -> str:
    if task == "up":
        return " ".join(w.upper() for w in words)
    if task == "rv":
        return " ".join(reversed(words))
    raise ValueError(task)


def sloppy(task: str, words: list[str]) -> str:
    """A plausible-but-wrong answer (used as the 'rejected' response)."""
    return " ".join(words) if task == "up" else " ".join(words)


def split_combos(seed=0):
    """Disjoint train/test prompts: ordered word pairs and triples."""
    rng = random.Random(seed)
    combos = [[a, b] for a in WORDS for b in WORDS if a != b]
    rng.shuffle(combos)
    n_test = len(combos) // 4
    return combos[n_test:], combos[:n_test]


def base_batch(rng: random.Random, batch=32, length=32):
    rows = []
    for _ in range(batch):
        s = ""
        while len(s) < length + 1:
            s += " ".join(rng.choice(WORDS) for _ in range(rng.randint(2, 5))) + "\n"
        rows.append(encode(s[: length + 1]))
    t = torch.tensor(rows)
    return t[:, :-1], t[:, 1:].clone()


def sft_example(task, words, ans=None):
    """Returns (ids, loss_mask). Loss only on the assistant answer (+<|end|>), not the prompt."""
    ans = answer(task, words) if ans is None else ans
    p = encode(render_chat([{"role": "user", "content": f"{task} " + " ".join(words)}], True))
    a = encode(ans) + [END]
    return p + a, [0] * len(p) + [1] * len(a)


def collate(examples, length=40):
    ids = torch.full((len(examples), length), PAD)
    mask = torch.zeros(len(examples), length)
    for i, (x, m) in enumerate(examples):
        x, m = x[:length], m[:length]
        ids[i, : len(x)] = torch.tensor(x)
        mask[i, : len(m)] = torch.tensor(m, dtype=torch.float)
    # inputs are ids[:, :-1]; targets ids[:, 1:]; mask aligned to targets
    return ids[:, :-1], ids[:, 1:].clone(), mask[:, 1:]


def sft_batch(rng, combos, tasks=("up",), batch=32, noise=0.0):
    ex = []
    for _ in range(batch):
        t = rng.choice(tasks)
        w = rng.choice(combos)
        a = sloppy(t, w) if rng.random() < noise else None
        ex.append(sft_example(t, w, a))
    return collate(ex)
