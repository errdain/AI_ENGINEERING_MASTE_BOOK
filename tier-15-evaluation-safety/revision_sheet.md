# Tier 15 — Foundation-Model Evaluation & Safety — Revision Sheet

> Status: DRAFT. Generated from the concept files `15.01-why-foundation-model-evaluation-is-difficult.md` … `15.35-safety-vs-capability-trade-offs.md` (sections 18 *Rapid-fire answer* and 23 *One-line memory hook*); the concept file is the authoritative source.
> Code evidence: see `lab/` (run `python -m pytest -q lab`) and `reviews/author_open_items.md`.

| Concept | Rapid-fire answer | Memory hook |
|---|---|---|
| 15.1 Why foundation-model evaluation is difficult | Open-ended outputs, unbounded inputs, contamination, proxy metrics, stochasticity, tail risk: so measure with uncertainty, slices and gates, not one number. | One number from one run on one prompt is an anecdote, not an evaluation. |
| 15.2 Evaluation objectives | Decision first, metric second, threshold third. | No decision, no metric. |
| 15.3 Offline vs online evaluation | Offline = safe and fast; online = real; use both and loop failures back. | Offline earns the right to try; online decides if it worked. |
| 15.4 Intrinsic vs extrinsic evaluation | Intrinsic = how well it models text; extrinsic = how well it does the job. | Intrinsic asks how well it models; extrinsic asks how well it works. |
| 15.5 Perplexity | PPL = exp(mean NLL); lower is better; tokenizer-specific. | Perplexity is how many tokens the model is torn between, on average. |
| 15.6 Benchmark design | Valid, clean, sliced, sized, scored, versioned, protected. | A benchmark is a measuring instrument: calibrate it, size it, protect it. |
| 15.7 Benchmark datasets | Public for comparison, private for decisions. | Leaderboards compare models; your own set chooses one. |
| 15.8 Task-specific evaluation | Real task, right scorer, validated, sliced. | Test the job, not the résumé. |
| 15.9 Accuracy | Correct/total; beware imbalance. | Accuracy is an average that can hide the one case you care about. |
| 15.10 Precision/recall/F1 in model evaluation | P = TP/(TP+FP), R = TP/(TP+FN), F1 harmonic; threshold slides along the PR curve. | Precision asks 'of what I flagged, how much is right?'; recall asks 'of what's there, how much did I find?'. |
| 15.11 Calibration | Confidence should equal accuracy; ECE measures the gap; temperature scaling is the cheap fix. | Say 80% only if you are right 80% of the time. |
| 15.12 Factuality | Claims, not sentences; supported vs contradicted vs unverifiable. | Fluent is not true; check claim by claim. |
| 15.13 Hallucination | Unsupported-but-confident; measure by claims, with CIs, vs abstention. | Confidently unsupported: count it, slice it, trade it against refusals. |
| 15.14 Groundedness | Supported by the given context, claim by claim, plus citations and abstention. | Grounded = the sources say so, not merely 'it sounds right'. |
| 15.15 Robustness | Same meaning, same answer: measure drop and consistency. | If rewording changes the answer, the answer was fragile. |
| 15.16 Adversarial testing | Search for failures; ASR with budget; feed back into regressions. | Test where it breaks, not where it works. |
| 15.17 Distribution-shift evaluation | Split by the axis that will change; compare accuracy and calibration. | Test where the data is going, not where it came from. |
| 15.18 Bias evaluation | Per-group rates, gaps with CIs, counterfactual pairs. | Same input except the group word: did the answer change? |
| 15.19 Fairness considerations | Choose the fairness criterion deliberately; they conflict when base rates differ. | Fairness is a choice among competing definitions, not a number you find. |
| 15.20 Privacy evaluation | Probe, canary, detect, bound; zero observed is not zero risk. | Zero leaks seen is a bound, not a guarantee. |
| 15.21 Memorization/data-leakage evaluation | Compare flagged vs clean; trust the clean score. | A score is only skill if the answers were not in the training data. |
| 15.22 Model safety | Per-category harm and over-refusal, adversarially tested, human-verified, gated. | Safety is judged at the worst category, not the average. |
| 15.23 Harmful-output evaluation | Per-category harmful rate, severity-weighted, over-refusal, with intervals. | Count harms by category with error bars, and count over-refusals too. |
| 15.24 Red teaming | Diverse testers, verified ASR with budget, fix, retest, regress. | Break it yourselves first, then freeze every break into a test. |
| 15.25 Regression datasets | Every fixed bug becomes a test; zero tolerance; versioned. | A fixed bug that is not a test is a bug that will come back. |
| 15.26 Golden datasets | Stratified, verified, frozen, versioned, protected. | One trusted ruler, never edited without a new version number. |
| 15.27 Human evaluation | Rubric, blind, overlap, kappa, adjudicate. | Agree beyond chance, or the labels mean nothing. |
| 15.28 LLM-as-judge concepts and limitations | Swap, control length, different family, calibrate to humans. | A judge is an instrument with a bias you must measure. |
| 15.29 Evaluation reproducibility | Seed, pin, record, re-run. | If you cannot re-run it, you cannot trust it. |
| 15.30 Evaluation versioning | Hash the data, record the config, compare like with like. | A score without a version is a rumour. |
| 15.31 Evaluation pipelines | Load, run, score, aggregate, store, compare, gate. | Evaluation as CI: every change gets the same inspection. |
| 15.32 Evaluation gates | Explicit thresholds, lower bounds, slices, fail closed, reasons. | One hard failure blocks, and it must say why. |
| 15.33 Model comparison | Paired bootstrap on the same items; effect size, interval, margin. | Same questions, paired differences, honest interval. |
| 15.34 Release/readiness criteria | Gates plus operations plus sign-offs; criteria fixed in advance. | Green benchmarks are necessary, not sufficient. |
| 15.35 Safety vs capability trade-offs | Plot the frontier; compare at equal points; gate both axes. | Every refusal threshold buys safety with helpfulness; show the whole curve. |
