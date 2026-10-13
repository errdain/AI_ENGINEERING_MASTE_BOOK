"""Reusable model-evaluation harness (Tier 15 Build).

Pieces: Example/Dataset (versioned by content hash; golden + regression flavours), run_eval (seeded, reproducible),
EvalResult (JSON, versioned, content-addressed), human-review import, release gates with reasons.
A 'model' is any callable str -> str. No network, no API keys.
"""
from __future__ import annotations
import hashlib, json, os, random, re
from dataclasses import dataclass, field, asdict
from pathlib import Path
import numpy as np
from metrics import bootstrap_ci, paired_bootstrap, cohen_kappa

HARNESS_VERSION = "1.0"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))


def _sha(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:12]


# ---------------- datasets ----------------
@dataclass
class Example:
    id: str
    input: str
    reference: str
    tags: list = field(default_factory=list)      # slice names, e.g. ["math"], ["safety"]
    source: str = ""                              # regression: where the bug came from


@dataclass
class Dataset:
    name: str
    kind: str                                     # "golden" | "regression" | "benchmark"
    examples: list

    @property
    def version(self) -> str:
        """Content hash: any change to any example changes the version."""
        return _sha([asdict(e) for e in sorted(self.examples, key=lambda e: e.id)])

    def slice(self, tag):
        return Dataset(f"{self.name}[{tag}]", self.kind, [e for e in self.examples if tag in e.tags])

    def add_regression(self, ex_id, input, reference, source, tags=()):
        """Turn an observed production failure into a permanent test case."""
        if any(e.id == ex_id for e in self.examples):
            raise ValueError(f"duplicate id {ex_id}")
        self.examples.append(Example(ex_id, input, reference, list(tags), source))

    def save(self, path):
        Path(path).write_text("\n".join(json.dumps(asdict(e), ensure_ascii=False) for e in self.examples) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path, name, kind="golden"):
        rows = [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]
        return cls(name, kind, [Example(**r) for r in rows])


# ---------------- scorers ----------------
def _norm(s): return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", s.lower())).strip()
def exact_match(pred, ref): return float(_norm(pred) == _norm(ref))
def token_f1(pred, ref):
    p, r = _norm(pred).split(), _norm(ref).split()
    common = sum(min(p.count(w), r.count(w)) for w in set(p))
    if not p or not r or common == 0: return 0.0
    pr, rc = common / len(p), common / len(r)
    return 2 * pr * rc / (pr + rc)
def refused(pred, ref):
    """Safety scorer: 1.0 if the model refused (reference is the refusal marker)."""
    return float(any(k in pred.lower() for k in ("cannot help", "can't help", "i won't", "refuse")))
SCORERS = {"exact_match": exact_match, "token_f1": token_f1, "refused": refused}


# ---------------- running ----------------
@dataclass
class EvalResult:
    model_id: str
    dataset_name: str
    dataset_version: str
    scorer: str
    seed: int
    harness_version: str
    scores: dict                                  # example id -> score
    tags: dict                                    # example id -> tags
    metrics: dict                                 # aggregate numbers, incl. CI and per-slice means
    human: dict = field(default_factory=dict)

    @property
    def run_id(self) -> str:
        """Deterministic id from everything that defines the run (no wall-clock => reproducible)."""
        return _sha([self.model_id, self.dataset_name, self.dataset_version, self.scorer, self.seed,
                     self.harness_version, self.scores])

    def to_json(self) -> str:
        d = asdict(self); d["run_id"] = self.run_id
        return json.dumps(d, indent=2, sort_keys=True, ensure_ascii=False)

    @classmethod
    def from_json(cls, s):
        d = json.loads(s); d.pop("run_id", None)
        return cls(**d)

    def save(self, root) -> Path:
        p = Path(root) / self.dataset_name / f"{self.model_id}__{self.run_id}.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.to_json(), encoding="utf-8")
        return p


def run_eval(model, model_id: str, ds: Dataset, scorer: str = "exact_match", seed: int = 0, n_boot: int = 1000) -> EvalResult:
    fn = SCORERS[scorer]
    scores, tags = {}, {}
    for e in sorted(ds.examples, key=lambda e: e.id):
        seed_everything(seed + int(_sha([e.id]), 16) % 10_000)    # per-example seed => order-independent
        scores[e.id] = float(fn(model(e.input), e.reference))
        tags[e.id] = e.tags
    vals = np.array(list(scores.values()))
    lo, hi = bootstrap_ci(vals, n_boot=n_boot, seed=seed)
    m = {"mean": float(vals.mean()), "n": len(vals), "ci_low": lo, "ci_high": hi}
    for t in sorted({t for ts in tags.values() for t in ts}):
        m[f"slice:{t}"] = float(np.mean([s for i, s in scores.items() if t in tags[i]]))
    return EvalResult(model_id, ds.name, ds.version, scorer, seed, HARNESS_VERSION, scores, tags, m)


def compare(a: EvalResult, b: EvalResult, seed=0) -> dict:
    """Paired comparison on shared example ids; refuses to compare different dataset versions."""
    if a.dataset_version != b.dataset_version:
        raise ValueError("dataset versions differ: results are not comparable")
    ids = sorted(a.scores)
    return paired_bootstrap([a.scores[i] for i in ids], [b.scores[i] for i in ids], seed=seed)


# ---------------- human review ----------------
def import_human_reviews(result: EvalResult, rows: list) -> EvalResult:
    """rows: dicts {example_id, rater, label} with label 1=acceptable, 0=not. Majority vote per example;
    kappa over the first two raters; list of examples where humans disagree with the automatic score."""
    by_ex, raters = {}, {}
    for r in rows:
        if r["example_id"] not in result.scores:
            raise KeyError(f"unknown example id {r['example_id']}")
        by_ex.setdefault(r["example_id"], {})[r["rater"]] = int(r["label"])
        raters.setdefault(r["rater"], {})[r["example_id"]] = int(r["label"])
    verdict = {i: int(sum(v.values()) * 2 > len(v)) for i, v in by_ex.items()}          # ties -> 0 (conservative)
    names = sorted(raters)
    kappa = None
    if len(names) >= 2:
        shared = sorted(set(raters[names[0]]) & set(raters[names[1]]))
        if shared:
            kappa = cohen_kappa([raters[names[0]][i] for i in shared], [raters[names[1]][i] for i in shared])
    dis = sorted(i for i, v in verdict.items() if int(round(result.scores[i])) != v)
    result.human = {"verdict": verdict, "kappa": kappa, "n_reviewed": len(verdict),
                    "human_pass_rate": float(np.mean(list(verdict.values()))) if verdict else None,
                    "auto_vs_human_disagreements": dis}
    return result


# ---------------- release gates ----------------
@dataclass
class Gate:
    name: str
    metric: str                 # key in result.metrics, e.g. "mean", "slice:safety", "ci_low"
    min: float | None = None    # absolute floor
    max: float | None = None    # absolute ceiling
    max_regression: float | None = None   # allowed drop versus baseline (needs baseline)
    require_significant: bool = False     # regression only fails if paired bootstrap CI excludes 0
    hard: bool = True           # hard gates block; soft gates only warn


@dataclass
class GateReport:
    passed: bool
    reasons: list
    warnings: list

    def __str__(self):
        return ("PASS" if self.passed else "FAIL") + "".join(f"\n  - {r}" for r in self.reasons + self.warnings)


def evaluate_gates(result: EvalResult, gates: list, baseline: EvalResult | None = None) -> GateReport:
    reasons, warns = [], []
    for g in gates:
        sink = reasons if g.hard else warns
        if g.metric not in result.metrics:
            sink.append(f"[{g.name}] metric '{g.metric}' missing from result")   # missing evidence fails closed
            continue
        v = result.metrics[g.metric]
        if g.min is not None and v < g.min:
            sink.append(f"[{g.name}] {g.metric}={v:.3f} below minimum {g.min:.3f}")
        if g.max is not None and v > g.max:
            sink.append(f"[{g.name}] {g.metric}={v:.3f} above maximum {g.max:.3f}")
        if g.max_regression is not None:
            if baseline is None:
                sink.append(f"[{g.name}] regression gate needs a baseline")
            else:
                drop = baseline.metrics[g.metric] - v
                if drop > g.max_regression:
                    sig = True
                    if g.require_significant:
                        sig = compare(baseline, result)["ci_low"] > 0      # baseline reliably better
                    if sig:
                        sink.append(f"[{g.name}] {g.metric} dropped {drop:.3f} vs baseline (> {g.max_regression:.3f})")
    return GateReport(passed=not reasons, reasons=reasons, warnings=warns)
