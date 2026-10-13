"""Generate Tier 15 figures + GIFs.  Run:  python scripts/make_figures.py   (numpy + matplotlib only, a few seconds)"""
import sys, pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "lab"))
from metrics import *
from toy import *

OUT = HERE.parent / "figures"; OUT.mkdir(exist_ok=True)
C = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2", "#B279A2"]
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 100, "savefig.bbox": "tight", "axes.titleweight": "bold"})


def save(fig, name):
    fig.savefig(OUT / name, dpi=100); plt.close(fig)


def fig_perplexity():
    train = "the cat sat on the mat . the dog sat on the log . the cat saw the dog . the dog saw the cat ."
    held = "the cat sat on the log . the dog saw the mat ."
    scr = "log the . mat saw on dog the sat cat . the"
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    alphas = [0.01, 0.03, 0.1, 0.3, 1, 3]
    for name, txt, c in [("train text", train, C[0]), ("held-out (same grammar)", held, C[2]), ("scrambled words", scr, C[3])]:
        ax[0].plot(alphas, [perplexity(BigramLM(train, a).token_nlls(txt)) for a in alphas], "o-", c=c, label=name)
    ax[0].set_xscale("log"); ax[0].set_yscale("log"); ax[0].set_xlabel("smoothing alpha"); ax[0].set_ylabel("perplexity (lower = better)")
    ax[0].set_title("Perplexity of a toy bigram LM"); ax[0].legend(); ax[0].grid(alpha=.3)
    V = len(BigramLM(train).vocab)
    ax[1].bar(["uniform\nguess (=V)", "train", "held-out", "scrambled"],
              [V] + [perplexity(BigramLM(train, .1).token_nlls(t)) for t in (train, held, scr)], color=[C[5], C[0], C[2], C[3]])
    ax[1].set_ylabel("perplexity at alpha = 0.1"); ax[1].set_title("Perplexity = effective branching factor")
    save(fig, "01_perplexity.png")


def fig_prf1():
    rng = np.random.default_rng(0)
    y = (rng.random(3000) < 0.2).astype(int); s = np.clip(rng.normal(0.35 + 0.3 * y, 0.17), 0, 1)
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    pr = pr_curve(y, s); ax[0].plot([p[2] for p in pr], [p[1] for p in pr], c=C[0])
    for t, c in [(0.35, C[1]), (0.5, C[2]), (0.65, C[3])]:
        m = prf1(y, (s >= t).astype(int)); ax[0].scatter(m["recall"], m["precision"], c=c, s=60, zorder=3, label=f"t={t}: P={m['precision']:.2f} R={m['recall']:.2f}")
    ax[0].axhline(y.mean(), ls="--", c="gray", label="random = base rate"); ax[0].set_xlabel("recall"); ax[0].set_ylabel("precision")
    ax[0].set_title("Precision-recall curve (20% positives)"); ax[0].legend(fontsize=8)
    ts = np.linspace(0.05, 0.95, 60); M = [prf1(y, (s >= t).astype(int)) for t in ts]
    for k, c in [("precision", C[0]), ("recall", C[1]), ("f1", C[2])]:
        ax[1].plot(ts, [m[k] for m in M], c=c, label=k)
    ax[1].set_xlabel("decision threshold"); ax[1].set_title("Moving the threshold trades P for R"); ax[1].legend()
    m = prf1(y, (s >= .5).astype(int)); cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
    ax[2].imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2): ax[2].text(j, i, cm[i, j], ha="center", va="center", fontsize=16, color="black" if cm[i, j] < cm.max() / 2 else "white")
    ax[2].set_xticks([0, 1], ["pred 0", "pred 1"]); ax[2].set_yticks([0, 1], ["true 0", "true 1"]); ax[2].set_title("Confusion matrix at t = 0.5")
    ax[2].spines[:].set_visible(False)
    save(fig, "02_precision_recall_threshold.png")


def fig_calibration_gif():
    lg, y = make_classifier_logits(5000, signal=1.5, sharpen=3.0, seed=1)
    Ts = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 3.0, 2.0, 1.0]
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.2))
    fig.subplots_adjust(bottom=0.2)

    def draw(i):
        for a in ax: a.clear()
        T = Ts[i]; p = softmax(lg, T); conf = p.max(1); ok = p.argmax(1) == y
        rows = reliability_bins(conf, ok, 10); e = ece(conf, ok)
        ax[0].plot([0, 1], [0, 1], "k--", label="perfect calibration")
        xs = [(r[0] + r[1]) / 2 for r in rows if r[2]]; ys = [r[4] for r in rows if r[2]]
        ax[0].bar(xs, ys, width=0.09, color=C[0], alpha=.8, label="accuracy in bin")
        ax[0].set_xlim(0, 1); ax[0].set_ylim(0, 1); ax[0].set_xlabel("confidence"); ax[0].set_ylabel("accuracy")
        ax[0].set_title(f"Reliability diagram, T={T:.1f}   ECE={e:.3f}"); ax[0].legend(loc="upper left")
        ax[1].hist(conf, bins=20, range=(0, 1), color=C[1]); ax[1].set_xlabel("confidence"); ax[1].set_ylabel("count")
        ax[1].set_title(f"Confidence histogram (accuracy {ok.mean():.2f} is constant)")
    FuncAnimation(fig, draw, frames=len(Ts)).save(OUT / "03_calibration_temperature.gif", writer=PillowWriter(fps=1.5), dpi=70)
    plt.close(fig)


def fig_bootstrap():
    rng = np.random.default_rng(3); fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    ns = [50, 100, 200, 500, 1000, 2000, 5000]
    for acc, c in [(0.7, C[0]), (0.5, C[1])]:
        w = [wilson_interval(int(round(acc * n)), n) for n in ns]
        ax[0].fill_between(ns, [a for a, _ in w], [b for _, b in w], color=c, alpha=.3, label=f"true acc {acc}"); ax[0].plot(ns, [acc] * len(ns), c=c)
    ax[0].set_xscale("log"); ax[0].set_xlabel("benchmark size (examples)"); ax[0].set_ylabel("accuracy, 95% CI"); ax[0].set_title("Small benchmarks cannot resolve small gaps"); ax[0].legend()
    n = 300; a = (rng.random(n) < .72).astype(float)
    b = np.where((a == 1) & (rng.random(n) < .06), 0.0, a)      # B = A except it loses ~6% of A's wins (strongly correlated errors)
    ra, rb = bootstrap_ci(a), bootstrap_ci(b); pb = paired_bootstrap(a, b)
    ax[1].errorbar([0, 1], [a.mean(), b.mean()], yerr=[[a.mean() - ra[0], b.mean() - rb[0]], [ra[1] - a.mean(), rb[1] - b.mean()]], fmt="o", c=C[0], capsize=6, label="each model's own CI")
    ax[1].set_xticks([0, 1], ["model A", "model B"]); ax[1].set_xlim(-.5, 1.5)
    ax[1].set_title(f"n={n}: CIs overlap, paired diff A-B {pb['diff']:+.3f} [{pb['ci_low']:+.3f}, {pb['ci_high']:+.3f}]", fontsize=10); ax[1].set_ylim(0.55, 0.8); ax[1].set_ylabel("accuracy, 95% bootstrap CI"); ax[1].legend(loc="lower right")
    save(fig, "04_benchmark_confidence.png")


def fig_contamination():
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2)); rng = np.random.default_rng(0)
    fr = np.linspace(0, 1, 11); true_skill = .55
    for lab, mem, c in [("memorises 100% of leaked items", 1.0, C[3]), ("memorises 50%", .5, C[1])]:
        ax[0].plot(fr * 100, 100 * (true_skill * (1 - fr * mem) + fr * mem), "o-", c=c, label=lab)
    ax[0].axhline(55, ls="--", c="gray", label="true ability (55%)"); ax[0].set_xlabel("% of test set leaked into training"); ax[0].set_ylabel("measured accuracy %")
    ax[0].set_title("Contamination inflates scores"); ax[0].legend(fontsize=9)
    train = ["the quick brown fox jumps over the lazy dog near the river bank", "pack my box with five dozen liquor jugs today"]
    tests = ["the quick brown fox jumps over the lazy dog", "quick brown fox jumps over the lazy cat", "a completely different question about physics and math"]
    ov = [ngram_overlap(t, train, 4) for t in tests]
    ax[1].barh(["verbatim leak", "light paraphrase", "clean"], ov, color=[C[3], C[1], C[2]]); ax[1].set_xlim(0, 1)
    ax[1].set_xlabel("fraction of test 4-grams found in training"); ax[1].set_title("n-gram overlap detector (toy)"); ax[1].invert_yaxis()
    save(fig, "05_contamination.png")


def fig_robust_shift():
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2)); nz = np.linspace(0, 1, 11)
    for acc, c in [(.95, C[0]), (.8, C[1])]:
        ax[0].plot(nz * 100, [100 * noisy_accuracy(acc, x, 1) for x in nz], "o-", c=c, label=f"clean acc {acc:.0%}")
    ax[0].axhline(25, ls=":", c="gray", label="chance (4 options)"); ax[0].set_xlabel("% of inputs perturbed"); ax[0].set_ylabel("accuracy %")
    ax[0].set_title("Robustness: accuracy under perturbation"); ax[0].legend()
    rng = np.random.default_rng(0); n = 6000
    def acc_at(shift):
        x = rng.normal(0, 1, (n, 2)) + np.array([shift, 0]); yy = (x[:, 0] - shift + x[:, 1] > 0).astype(int)
        pred = (x[:, 0] + x[:, 1] > 0).astype(int)                     # decision rule learned at shift 0
        return (pred == yy).mean()
    sh = np.linspace(0, 3, 13); ax[1].plot(sh, [100 * acc_at(s) for s in sh], "o-", c=C[3])
    ax[1].set_xlabel("covariate shift (std devs)"); ax[1].set_ylabel("accuracy %"); ax[1].set_title("Distribution shift: fixed model, moving data")
    save(fig, "06_robustness_and_shift.png")


def fig_fairness():
    rng = np.random.default_rng(1); n = 4000
    g = rng.integers(0, 2, n); base = np.where(g == 0, .5, .2); y = (rng.random(n) < base).astype(int)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    cases = {"perfect classifier\n(predict truth)": y, "parity-forced\n(equal selection)": None}
    p_perf = y.copy(); p_par = np.where(g == 0, (rng.random(n) < .35), (rng.random(n) < .35)).astype(int)
    labels = ["selection rate", "TPR", "FPR"]
    for k, (name, p) in enumerate([("perfect classifier", p_perf), ("forced equal selection", p_par)]):
        r = group_rates(y, p, g); x = np.arange(3) + k * 3.6
        ax[k].bar(np.arange(3) - .2, [r[0]["selection"], r[0]["tpr"], r[0]["fpr"]], .4, color=C[0], label="group 0 (base rate 50%)")
        ax[k].bar(np.arange(3) + .2, [r[1]["selection"], r[1]["tpr"], r[1]["fpr"]], .4, color=C[1], label="group 1 (base rate 20%)")
        ax[k].set_xticks(range(3), labels); ax[k].set_ylim(0, 1.15)
        ax[k].set_title(f"{name}\nDP gap {demographic_parity_diff(p, g):.2f} | EO gap {equalized_odds_diff(y, p, g):.2f}", fontsize=10)
        ax[k].legend(fontsize=8, loc="upper right")
    save(fig, "07_fairness_dp_vs_eo.png")


def fig_judge():
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2)); pbs = np.linspace(0, 3, 7)
    cons, firsts = [], []
    for pb in pbs:
        c, f = swap_consistency(MockJudge(pb, 0, seed=1), .65, .55, 100, 100, n=600); cons.append(c); firsts.append(f)
    ax[0].plot(pbs, cons, "o-", c=C[0], label="swap consistency"); ax[0].plot(pbs, firsts, "s-", c=C[3], label="win-rate of slot 1")
    ax[0].axhline(.5, ls=":", c="gray"); ax[0].set_xlabel("injected position bias (logit)"); ax[0].set_title("Position bias in a mock judge"); ax[0].legend()
    L = np.linspace(100, 400, 7); ax[1].plot(L, [MockJudge(0, 1.0).p_first(.5, .6, l, 100) for l in L], "o-", c=C[1], label="verbosity bias 1.0")
    ax[1].plot(L, [MockJudge(0, 0).p_first(.5, .6, l, 100) for l in L], "o-", c=C[2], label="no bias")
    ax[1].set_xlabel("length of the (worse) answer A vs B=100 words"); ax[1].set_ylabel("P(judge picks A)"); ax[1].set_title("Verbose answers win even when worse"); ax[1].legend()
    save(fig, "08_llm_judge_bias.png")


def fig_funnel_gate():
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
    stages = [("attack prompts", 1000), ("not refused", 220), ("harmful content", 60), ("verified by human", 41)]
    for i, (n, v) in enumerate(stages):
        ax[0].barh(i, v, color=C[i], height=.7); ax[0].text(v + 10, i, f"{v}", va="center")
    ax[0].set_yticks(range(4), [s[0] for s in stages]); ax[0].invert_yaxis(); ax[0].set_xlabel("count (illustrative numbers)")
    ax[0].set_title("Red-team attack-success funnel: ASR = 41/1000 = 4.1%", fontsize=10)
    gates = [("accuracy >= 0.80", .84, .80, True), ("CI low >= 0.78", .79, .78, True), ("safety refusal >= 0.99", .97, .99, False),
             ("ECE <= 0.05", .04, .05, True), ("no regression > 0.02", .03, .02, False)]
    for i, (nm, v, th, ok) in enumerate(gates):
        ax[1].barh(i, 1, color=C[2] if ok else C[3], height=.7)
        ax[1].text(.02, i, f"{'PASS' if ok else 'FAIL'}  {nm}   (value {v})", va="center", color="white", weight="bold")
    ax[1].set_yticks([]); ax[1].set_xticks([]); ax[1].invert_yaxis(); ax[1].spines[:].set_visible(False)
    ax[1].set_title("Release gate: ONE hard failure blocks the release", fontsize=10)
    save(fig, "09_redteam_funnel_and_gate.png")


def fig_frontier_gif():
    rng = np.random.default_rng(0); n = 4000
    harmful = rng.random(n) < .3; risk = np.clip(rng.normal(np.where(harmful, .7, .35), .18), 0, 1)
    ths = np.linspace(0.05, 0.95, 40); refuse_h = [(risk[harmful] >= t).mean() for t in ths]; help_ = [(risk[~harmful] < t).mean() for t in ths]
    # t: refuse when risk >= t
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2)); idx = list(range(0, 40, 3)) + [39]
    fig.subplots_adjust(bottom=0.2)

    def draw(k):
        for a in ax: a.clear()
        i = idx[k]; t = ths[i]
        ax[0].hist(risk[~harmful], bins=30, range=(0, 1), alpha=.7, color=C[0], label="benign requests"); ax[0].hist(risk[harmful], bins=30, range=(0, 1), alpha=.7, color=C[3], label="harmful requests")
        ax[0].axvline(t, c="k", ls="--"); ax[0].set_xlabel("risk score; refuse if >= threshold"); ax[0].set_title(f"Refusal threshold = {t:.2f}"); ax[0].legend()
        ax[1].plot(help_, refuse_h, c="gray"); ax[1].scatter(help_[i], refuse_h[i], c=C[1], s=90, zorder=3)
        ax[1].set_xlabel("helpfulness: benign requests answered"); ax[1].set_ylabel("safety: harmful requests refused")
        ax[1].set_xlim(0, 1.02); ax[1].set_ylim(0, 1.02); ax[1].set_title("Safety vs capability frontier (toy)")
    FuncAnimation(fig, draw, frames=len(idx)).save(OUT / "10_safety_capability_frontier.gif", writer=PillowWriter(fps=2), dpi=70)
    plt.close(fig)


if __name__ == "__main__":
    for f in [fig_perplexity, fig_prf1, fig_calibration_gif, fig_bootstrap, fig_contamination, fig_robust_shift, fig_fairness, fig_judge, fig_funnel_gate, fig_frontier_gif]:
        f(); print("ok", f.__name__)
