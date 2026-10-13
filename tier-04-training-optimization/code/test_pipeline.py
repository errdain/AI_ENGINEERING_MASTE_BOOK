import math

import pytest
import torch

import pipeline as P


def data(n=400, seed=1):
    return P.make_dataset(n, seed)


def weights(trainer):
    return torch.cat([p.detach().flatten() for p in trainer.model.parameters()])


def test_schedule_warms_up_then_decays_to_the_floor():
    f = lambda s: P.warmup_cosine(s, total=100, warmup=10, peak=1.0, floor_frac=0.1)
    assert f(0) == pytest.approx(0.1) and f(9) == pytest.approx(1.0)
    assert f(10) == pytest.approx(1.0)                                     # peak right after warm-up
    assert f(55) < f(10) and f(100) == pytest.approx(0.1)                  # decays to the floor
    assert all(f(s) >= f(s + 1) - 1e-12 for s in range(10, 100))          # monotone decay


def test_no_weight_decay_for_biases_and_norm_gains():
    model = P.build_model(P.Config())
    groups = P.param_groups(model, 0.1)
    assert groups[0]["weight_decay"] == 0.1 and groups[1]["weight_decay"] == 0.0
    assert all(p.dim() >= 2 for p in groups[0]["params"]) and all(p.dim() == 1 for p in groups[1]["params"])


def test_accumulation_matches_a_bigger_batch():
    cfg_big = P.Config(micro_batch=64, accum=1, steps=10, clip_norm=None, patience=0, warmup=2)
    cfg_acc = P.Config(micro_batch=16, accum=4, steps=10, clip_norm=None, patience=0, warmup=2)
    # same global stream: micro-batch size 16 x 4 must see the same examples as one batch of 64 -> use a fixed permutation
    big, acc = P.Trainer(cfg_big, data(), data(200, 2)), P.Trainer(cfg_acc, data(), data(200, 2))
    class Fixed(P.Stream):
        def indices(self, j):
            return torch.arange(j * self.mb, (j + 1) * self.mb) % self.n
    big.stream, acc.stream = Fixed(400, 64, 0), Fixed(400, 16, 0)
    big.fit(); acc.fit()
    assert torch.allclose(weights(big), weights(acc), atol=1e-5)


def test_resume_equals_an_uninterrupted_run(tmp_path):
    cfg = P.Config(steps=60, eval_every=10, patience=0, dropout=0.1)
    full = P.Trainer(cfg, data(), data(200, 2)); full.fit()
    path = str(tmp_path / "ckpt.pt")
    first = P.Trainer(cfg, data(), data(200, 2)); first.fit(until=30); first.save(path)
    second = P.Trainer(cfg, data(), data(200, 2)); second.resume(path); second.fit()
    assert torch.allclose(weights(full), weights(second), atol=1e-6)        # includes dropout: the RNG state was saved
    assert second.history[-1]["train_loss"] == pytest.approx(full.history[-1]["train_loss"], abs=1e-6)


def test_resuming_without_optimizer_state_differs(tmp_path):
    cfg = P.Config(steps=60, eval_every=10, patience=0)
    full = P.Trainer(cfg, data(), data(200, 2)); full.fit()
    first = P.Trainer(cfg, data(), data(200, 2)); first.fit(until=30)
    cold = P.Trainer(cfg, data(), data(200, 2)); cold.model.load_state_dict(first.model.state_dict()); cold.step = 30; cold.fit()
    assert not torch.allclose(weights(full), weights(cold), atol=1e-6)


def test_clipping_bounds_the_global_norm():
    tr = P.Trainer(P.Config(clip_norm=0.05), data(), data(100, 2))
    row = tr.train_step()
    after = torch.sqrt(sum((p.grad ** 2).sum() for p in tr.model.parameters() if p.grad is not None)) if any(p.grad is not None for p in tr.model.parameters()) else torch.tensor(0.0)
    assert row["grad_norm"] > 0.05                                         # the raw gradient was larger than the limit
    assert after <= 0.05 + 1e-6                                            # and was clipped to it


def test_same_seed_same_result_different_seed_different_result():
    cfg = P.Config(steps=40, patience=0)
    a = P.Trainer(cfg, data(), data(100, 2)); a.fit()
    b = P.Trainer(cfg, data(), data(100, 2)); b.fit()
    c = P.Trainer(P.Config(steps=40, patience=0, seed=7), data(), data(100, 2)); c.fit()
    assert torch.equal(weights(a), weights(b)) and not torch.allclose(weights(a), weights(c))


def test_non_finite_gradients_skip_the_update():
    tr = P.Trainer(P.Config(), data(), data(100, 2))
    before = weights(tr).clone()
    tr.Xtr = tr.Xtr.clone(); tr.Xtr[:] = float("inf")                       # corrupt the data
    row = tr.train_step()
    assert not math.isfinite(row["grad_norm"]) and tr.skipped == 1
    assert torch.equal(weights(tr), before)                                 # weights untouched


def test_the_pipeline_learns_and_early_stops():
    tr = P.Trainer(P.Config(steps=600, eval_every=20, patience=6, lr=1e-2, hidden=64), P.make_dataset(2000, 1), P.make_dataset(1000, 2))
    hist = tr.fit()
    final = [r for r in hist if "val_acc" in r][-1]
    assert final["val_acc"] > 0.78 and tr.best["state"] is not None


def test_bf16_autocast_runs_and_stays_close_to_fp32():
    a = P.Trainer(P.Config(steps=30, patience=0, bf16=False), data(), data(100, 2)); a.fit()
    b = P.Trainer(P.Config(steps=30, patience=0, bf16=True), data(), data(100, 2)); b.fit()
    assert abs(a.history[-1]["train_loss"] - b.history[-1]["train_loss"]) < 0.15


def test_profiler_records_every_phase_and_flop_count_scales():
    tr = P.Trainer(P.Config(steps=10, patience=0), data(), data(100, 2)); tr.fit()
    shares = tr.timer.shares()
    assert set(shares) == {"data", "forward", "backward", "optimizer"} and sum(shares.values()) == pytest.approx(1.0)
    assert P.step_flops(P.Config(accum=4)) == 4 * P.step_flops(P.Config(accum=1))
