# Tier 10 — Fine-Tuning & Alignment — Revision Sheet

> Status: DRAFT. Generated from the concept files `10.01-base-models-vs-instruction-tuned-models.md` … `10.35-fine-tuning-cost-and-deployment-trade-offs.md` (sections 18 *Rapid-fire answer* and 23 *One-line memory hook*); the concept file is the authoritative source.
> Code evidence: see `lab/` (run `python -m pytest -q lab`) and `reviews/author_open_items.md`.

| Concept | Rapid-fire answer | Memory hook |
|---|---|---|
| 10.1 Base models vs instruction-tuned models | Base = knows; instruct = behaves. | Base = knows; instruct = behaves. |
| 10.2 Post-training overview | Post-training = SFT, then preferences, then safety/distill/merge. | Pretrain for knowledge, post-train for behaviour. |
| 10.3 Supervised fine-tuning (SFT) | SFT = next-token loss on answers only. | Show, don't tell, and only grade the answer. |
| 10.4 Instruction datasets | Instruction data = prompt/answer demonstrations; quality over quantity. | Teach with a clean workbook, not a junk drawer. |
| 10.5 Chat templates | The template is part of the model. | The template is part of the model. |
| 10.6 SFT data curation | Curate: clean, balanced, held-out, versioned. | Garbage demos in, garbage habits out. |
| 10.7 Synthetic training data | Synthetic data scales, but only verified data is safe. | Teacher-made data is only as good as its filter. |
| 10.8 Reasoning-data construction | Sample many, keep what a checker accepts. | Keep the solutions a checker believes. |
| 10.9 Full fine-tuning | Full FT = everything moves. | Full FT: maximum power, maximum bill. |
| 10.10 Parameter-efficient fine-tuning (PEFT) | PEFT = freeze the giant, train the sliver. | Freeze the giant, train the sliver. |
| 10.11 LoRA | delta W = (alpha/r) B A, B = 0 at init, mergeable. | Two thin matrices, one zero start. |
| 10.12 LoRA rank | Rank = adapter capacity knob. | Rank: how thick the correction is. |
| 10.13 LoRA alpha | Alpha/r = the adapter's volume. | Alpha/r is the volume knob. |
| 10.14 LoRA target modules | Target the layers that must change. | Put the notes where the changes must go. |
| 10.15 QLoRA | 4-bit frozen base, 16-bit trainable adapter. | Freeze it small, adapt it big. |
| 10.16 Adapters | Adapters add layers; LoRA adds a mergeable delta. | Adapters add a module; LoRA adds a delta. |
| 10.17 Prompt tuning | Soft prompts: tuned vectors, frozen model. | Tune the whisper, freeze the speaker. |
| 10.18 Quantization-aware fine-tuning concepts | PTQ after, QAT during, QLoRA beside. | PTQ after, QAT during, QLoRA beside. |
| 10.19 Preference data | Preference data = prompt + better + worse. | Teach by comparison, not just example. |
| 10.20 Preference optimization | Optimise 'better', not just 'correct text'. | Optimise better, not just correct. |
| 10.21 Reward models | RM: a learned referee, hackable. | RM = learned referee. |
| 10.22 RLHF | RLHF: reward model plus RL plus KL leash. | RLHF: reward, but on a leash. |
| 10.23 PPO intuition for alignment | PPO: clip, advantage, KL leash. | Step small, stay near home. |
| 10.24 On-policy vs off-policy alignment intuition | On-policy: my samples. Off-policy: someone else's. | On-policy: learn from your own samples. |
| 10.25 Direct Preference Optimization (DPO) | The policy is its own reward model. | The policy is its own reward model. |
| 10.26 Distillation | Distill: student copies teacher's distribution. | Small model, big teacher's habits. |
| 10.27 Continued fine-tuning | Keep the old skills while adding new ones. | New skill in, keep old skills. |
| 10.28 Catastrophic forgetting | New task in, old skill out, unless you replay. | New task in, old skill out, unless you replay. |
| 10.29 Catastrophic-forgetting evaluation | Measure deltas before you ship. | Test what you might have broken. |
| 10.30 Adapter merging | Folding is exact; combining is not. | Fold one exactly; stack many carefully. |
| 10.31 Model merging | Merge weights, then verify behaviour. | Same base, add the deltas, then test. |
| 10.32 Checkpoint merging | Average the last few steps of the walk. | Average the last few steps. |
| 10.33 Fine-tuning evaluation | Held-out, output-level, regressions, seeds. | If you didn't hold it out, you didn't test it. |
| 10.34 Fine-tuning data/security considerations | You ship what you train on. | You ship what you train on. |
| 10.35 Fine-tuning cost and deployment trade-offs | Prompt first, LoRA next, full FT last. | Prompt first, LoRA next, full FT last. |
