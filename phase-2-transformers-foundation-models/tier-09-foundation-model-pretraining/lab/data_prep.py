"""Tier 9 lab, part 1: pretraining data preparation.

A miniature of the real pipeline (Tier 9 sections 9.2-9.13):

    acquire -> clean -> quality filter -> license filter -> exact dedup
            -> near-dup dedup (MinHash + LSH) -> decontaminate -> mix -> tokenize

Everything is deterministic given a seed and uses only the standard library
and numpy.  The "web crawl" is SYNTHETIC (generated below) so the lab needs no
network and no downloads; it deliberately contains the junk a real crawl has:
HTML boilerplate, too-short pages, exact copies, near-copies, restricted
licenses and a leaked benchmark document.
"""
from __future__ import annotations

import hashlib
import random
import re
import unicodedata
from collections import Counter, defaultdict

import numpy as np

# --------------------------------------------------------------------------
# 1. Acquire (synthetic crawl with realistic junk)
# --------------------------------------------------------------------------
_SUBJ = ["the cat", "a dog", "the farmer", "the pilot", "a student", "the engineer", "my sister", "the old sailor"]
_VERB = ["sees", "builds", "paints", "repairs", "carries", "studies", "finds", "describes"]
_OBJ = ["a red boat", "the wooden bridge", "an old map", "the small garden", "a heavy box", "the quiet river",
        "a bright lamp", "the broken clock"]
_WHEN = ["in the morning", "after the storm", "before dinner", "during winter", "at the harbor", "near the hill"]
_CODE_NAMES = ["total", "count", "items", "value", "result", "data", "index", "name"]
_CODE_FUNCS = ["add", "scale", "clip", "merge", "count", "shift"]
_CHAT = ["could you help me with", "i am trying to understand", "thanks, that explains", "what is the best way to learn",
         "please show an example of", "i still do not get"]
_TOPICS = ["gradient descent", "binary search", "tokenizers", "unit tests", "the weather", "cooking rice", "sorting lists"]

LICENSES = {"wiki": "cc-by-sa", "code": "mit", "chat": "cc0", "web": "cc-by"}
ALLOWED_LICENSES = {"cc-by-sa", "mit", "cc0", "cc-by", "public-domain"}  # REVIEW_REQUIRED: legal policy is org-specific


def _sentence(r: random.Random) -> str:
    return f"{r.choice(_SUBJ)} {r.choice(_VERB)} {r.choice(_OBJ)} {r.choice(_WHEN)}."


def _doc_text(source: str, r: random.Random) -> str:
    if source == "wiki":
        return " ".join(_sentence(r).capitalize() for _ in range(r.randint(5, 9)))
    if source == "code":
        f, a, b = r.choice(_CODE_FUNCS), r.choice(_CODE_NAMES), r.choice(_CODE_NAMES)
        n = r.randint(2, 9)
        return (f"def {f}_{a}({a}, {b}):\n    result = {a} + {b} * {n}\n    return result\n\n"
                f"print({f}_{a}({r.randint(1, 9)}, {r.randint(1, 9)}))\n") * r.randint(1, 2)
    if source == "chat":
        return " ".join(f"{r.choice(_CHAT)} {r.choice(_TOPICS)} ?" for _ in range(r.randint(4, 7)))
    # "web": prose plus boilerplate
    return " ".join(_sentence(r) for _ in range(r.randint(4, 8)))


def make_synthetic_crawl(seed: int = 0, n_docs: int = 500) -> list[dict]:
    """Return raw documents: dicts with id, text, source, license, url."""
    r = random.Random(seed)
    docs: list[dict] = []
    sources = ["wiki"] * 4 + ["code"] * 2 + ["chat"] * 2 + ["web"] * 2
    for i in range(n_docs):
        s = r.choice(sources)
        docs.append({"id": f"d{i}", "text": _doc_text(s, r), "source": s, "license": LICENSES[s], "url": f"http://ex.org/{s}/{i}"})
    base = list(docs)
    # --- injected junk -----------------------------------------------------
    for k in range(max(5, n_docs // 20)):  # exact duplicates (mirrors)
        d = dict(r.choice(base)); d["id"] = f"exact{k}"; docs.append(d)
    for k in range(max(8, n_docs // 12)):  # near duplicates: tweak one word
        d = dict(r.choice([b for b in base if b["source"] in ("wiki", "web")]))
        words = d["text"].split(); words[r.randrange(len(words))] = "today"
        d["text"] = " ".join(words); d["id"] = f"near{k}"; docs.append(d)
    for k in range(max(5, n_docs // 25)):  # html boilerplate pages
        docs.append({"id": f"html{k}", "source": "web", "license": "unknown", "url": f"http://ex.org/h/{k}",
                     "text": f"<html><body><div class='nav'><a href='/'>Home</a> | <a href='/x'>Menu</a></div><p>{_sentence(r)}</p>"
                             f"<script>var x=1;</script><div>Cookie banner: accept all cookies!!!</div></body></html>"})
    for k in range(max(5, n_docs // 25)):  # too short / symbol soup
        docs.append({"id": f"junk{k}", "source": "web", "license": "unknown", "url": "http://ex.org/j",
                     "text": r.choice(["ok", "click here >>>", "### $$$ %%% ^^^ &&& *** ((( )))", "404 not found"])})
    for k in range(max(5, n_docs // 25)):  # restricted license
        d = dict(r.choice(base)); d["id"] = f"lic{k}"; d["license"] = "all-rights-reserved"
        d["text"] = _sentence(r) + " " + d["text"]; docs.append(d)
    r.shuffle(docs)
    return docs


# --------------------------------------------------------------------------
# 2. Clean (9.3)
# --------------------------------------------------------------------------
_TAG = re.compile(r"<script.*?</script>|<style.*?</style>|<[^>]+>", re.S | re.I)
_CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f​‍﻿]")
_BOILER = re.compile(r"cookie banner:.*?(?:\.|!+|$)", re.I)


def clean_text(text: str) -> str:
    """NFC-normalise, strip HTML/script, drop control & zero-width chars, collapse blanks.

    Newlines/indentation are kept for code documents (lossy cleaning of code is a classic bug)."""
    t = unicodedata.normalize("NFC", text)
    t = _TAG.sub(" ", t)
    t = _BOILER.sub(" ", t)
    t = _CTRL.sub("", t)
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


# --------------------------------------------------------------------------
# 3. Quality filter (9.4) -- heuristic rules in the spirit of C4/Gopher/RefinedWeb
#    (thresholds here are illustrative, tune on a labelled sample in real use)
# --------------------------------------------------------------------------
def quality_filter(text: str, min_words: int = 8, max_symbol_ratio: float = 0.25,
                   max_repeated_line_frac: float = 0.5) -> tuple[bool, str]:
    words = text.split()
    if len(words) < min_words:
        return False, "too_short"
    n = len(text)
    symbols = sum(1 for c in text if not (c.isalnum() or c.isspace() or c in ".,;:'\"()_-=+*/%<>"))
    if symbols / max(1, n) > max_symbol_ratio:
        return False, "symbol_ratio"
    lines = [l for l in text.splitlines() if l.strip()]
    if len(lines) > 3:
        rep = 1 - len(set(lines)) / len(lines)
        # code legitimately repeats lines, so only flag extreme repetition
        if rep > max_repeated_line_frac and rep > 0.8:
            return False, "repeated_lines"
    return True, "ok"


# --------------------------------------------------------------------------
# 4. Dedup (9.5): exact hash + MinHash/LSH near-duplicates
# --------------------------------------------------------------------------
def _norm_for_hash(t: str) -> str:
    return re.sub(r"\s+", " ", t.lower()).strip()


def exact_dedup(docs: list[dict]) -> tuple[list[dict], int]:
    seen, out = set(), []
    for d in docs:
        h = hashlib.sha256(_norm_for_hash(d["text"]).encode()).hexdigest()
        if h not in seen:
            seen.add(h); out.append(d)
    return out, len(docs) - len(out)


_PRIME = (1 << 61) - 1


def shingles(text: str, k: int = 3) -> set[int]:
    """Word k-shingles hashed to 64-bit ints (blake2b -> stable across runs, unlike hash())."""
    w = _norm_for_hash(text).split()
    if len(w) < k:
        w = w + [""] * (k - len(w))
    return {int.from_bytes(hashlib.blake2b(" ".join(w[i:i + k]).encode(), digest_size=8).digest(), "big")
            for i in range(len(w) - k + 1)}


def minhash_signature(sh: set[int], num_perm: int = 64, seed: int = 1) -> np.ndarray:
    """MinHash: for each of num_perm hash functions h(x)=(a*x+b) mod p keep the min over the set.
    P[sig_i(A)==sig_i(B)] = Jaccard(A,B)."""
    rng = np.random.default_rng(seed)
    a = [int(x) | 1 for x in rng.integers(1, _PRIME, size=num_perm, dtype=np.uint64)]
    b = [int(x) for x in rng.integers(0, _PRIME, size=num_perm, dtype=np.uint64)]
    return np.array([min((ai * x + bi) % _PRIME for x in sh) for ai, bi in zip(a, b)], dtype=np.uint64)


def jaccard(a: set, b: set) -> float:
    return len(a & b) / max(1, len(a | b))


def estimated_jaccard(s1: np.ndarray, s2: np.ndarray) -> float:
    return float(np.mean(s1 == s2))


def near_dedup(docs: list[dict], threshold: float = 0.7, num_perm: int = 64, bands: int = 16
               ) -> tuple[list[dict], int]:
    """Keep the first doc of every near-duplicate cluster.

    LSH: split each signature into `bands` bands of r=num_perm/bands rows; docs sharing any
    whole band land in the same bucket and become candidates (probability of a candidate
    pair at Jaccard s is 1-(1-s^r)^bands).  Candidates are verified with the signature estimate."""
    r = num_perm // bands
    sh = [shingles(d["text"]) for d in docs]
    sig = [minhash_signature(s, num_perm) for s in sh]
    buckets: dict[tuple, list[int]] = defaultdict(list)
    keep, dropped = [], set()
    for i, s in enumerate(sig):
        cand: set[int] = set()
        keys = [(b, s[b * r:(b + 1) * r].tobytes()) for b in range(bands)]
        for k in keys:
            cand.update(buckets[k])
        if any(j not in dropped and estimated_jaccard(s, sig[j]) >= threshold for j in cand):
            dropped.add(i); continue
        for k in keys:
            buckets[k].append(i)
        keep.append(i)
    return [docs[i] for i in keep], len(dropped)


# --------------------------------------------------------------------------
# 5. Contamination (9.11, 9.12): n-gram overlap against evaluation sets
# --------------------------------------------------------------------------
def ngrams(text: str, n: int = 8) -> set[tuple]:
    w = _norm_for_hash(text).split()
    return {tuple(w[i:i + n]) for i in range(len(w) - n + 1)}


def contamination_rate(train_docs: list[dict], bench_texts: list[str], n: int = 8) -> float:
    """Fraction of benchmark items that share at least one n-gram with the training set."""
    train_grams: set[tuple] = set()
    for d in train_docs:
        train_grams |= ngrams(d["text"], n)
    hit = sum(1 for b in bench_texts if ngrams(b, n) & train_grams)
    return hit / max(1, len(bench_texts))


def decontaminate(docs: list[dict], bench_texts: list[str], n: int = 8) -> tuple[list[dict], int]:
    bench_grams: set[tuple] = set()
    for b in bench_texts:
        bench_grams |= ngrams(b, n)
    out = [d for d in docs if not (ngrams(d["text"], n) & bench_grams)]
    return out, len(docs) - len(out)


# --------------------------------------------------------------------------
# 6. Mixture (9.6-9.8): sample documents so token share follows target weights
# --------------------------------------------------------------------------
def mix_sources(docs: list[dict], weights: dict[str, float], total_docs: int, seed: int = 0) -> list[dict]:
    """Sample (with replacement only when a source is too small = up-sampling/epochs) so that the
    DOC share per source follows `weights` (normalised)."""
    z = sum(weights.values())
    by_src: dict[str, list[dict]] = defaultdict(list)
    for d in docs:
        by_src[d["source"]].append(d)
    rng = random.Random(seed)
    out: list[dict] = []
    for s, w in weights.items():
        pool = by_src.get(s, [])
        k = round(total_docs * w / z)
        if not pool or k == 0:
            continue
        out += rng.sample(pool, k) if k <= len(pool) else [rng.choice(pool) for _ in range(k)]
    rng.shuffle(out)
    return out


# --------------------------------------------------------------------------
# 7. The whole funnel with a report
# --------------------------------------------------------------------------
def run_pipeline(raw: list[dict], bench_texts: list[str] | None = None, near_threshold: float = 0.7
                 ) -> tuple[list[dict], dict]:
    """acquire -> clean -> quality -> license -> exact -> near -> decontaminate. Returns (docs, report)."""
    rep: dict = {"stages": [("acquired", len(raw))], "dropped": {}}

    def stage(name: str, docs: list[dict], dropped: int | dict) -> list[dict]:
        rep["stages"].append((name, len(docs))); rep["dropped"][name] = dropped
        return docs

    docs = [{**d, "text": clean_text(d["text"])} for d in raw]
    stage("cleaned", docs, 0)
    reasons: Counter = Counter(); kept = []
    for d in docs:
        ok, why = quality_filter(d["text"])
        if ok: kept.append(d)
        else: reasons[why] += 1
    docs = stage("quality_filter", kept, dict(reasons))
    lic = [d for d in docs if d["license"] in ALLOWED_LICENSES]
    docs = stage("license_filter", lic, len(docs) - len(lic))
    docs, n = exact_dedup(docs); stage("exact_dedup", docs, n)
    docs, n = near_dedup(docs, near_threshold); stage("near_dedup", docs, n)
    if bench_texts:
        docs, n = decontaminate(docs, bench_texts); stage("decontaminated", docs, n)
    rep["source_counts"] = dict(Counter(d["source"] for d in docs))
    return docs, rep


# --------------------------------------------------------------------------
# 8. Tokenize into a flat stream, deterministic train/val split by doc hash
# --------------------------------------------------------------------------
def split_train_val(docs: list[dict], val_frac: float = 0.1) -> tuple[list[dict], list[dict]]:
    """Hash-based split: a document always lands on the same side (stable across runs/reorderings)."""
    tr, va = [], []
    for d in docs:
        h = int(hashlib.sha256(d["text"].encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
        (va if h < val_frac else tr).append(d)
    return tr, va


def docs_to_stream(tok, docs: list[dict], eos_id: int = 2) -> np.ndarray:
    """Concatenate docs, each followed by <EOS>; returns uint16 array (vocab < 65536)."""
    ids: list[int] = []
    for d in docs:
        ids.extend(tok.encode(d["text"])); ids.append(eos_id)
    return np.array(ids, dtype=np.uint16)


def stream_fingerprint(arr: np.ndarray) -> str:
    return hashlib.sha256(arr.tobytes()).hexdigest()[:16]
