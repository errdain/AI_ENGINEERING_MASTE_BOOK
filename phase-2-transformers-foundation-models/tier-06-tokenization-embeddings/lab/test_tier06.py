import numpy as np
import pytest

from mini_bpe import ByteBPE, compression_ratio, merge_pair
from positional import (alibi_bias, alibi_slopes, apply_rope, embedding_lookup,
                        padding_mask, sinusoidal)

CORPUS = "low lower lowest low low newer newest wider widest " * 20


def test_merge_pair_non_overlapping():
    assert merge_pair([1, 1, 1], (1, 1), 9) == [9, 1]


def test_roundtrip_ascii_and_unicode():
    tok = ByteBPE.train(CORPUS, 30)
    for s in ["low lower", "naïve café 日本語 🙂", "", "  spaces  "]:
        assert tok.decode(tok.encode(s)) == s


def test_more_merges_means_fewer_tokens():
    short = len(ByteBPE.train(CORPUS, 5).encode(CORPUS))
    long_ = len(ByteBPE.train(CORPUS, 40).encode(CORPUS))
    assert long_ < short


def test_vocab_size_and_determinism():
    a, b = ByteBPE.train(CORPUS, 10), ByteBPE.train(CORPUS, 10)
    assert a.merges == b.merges
    assert a.vocab_size == 4 + 256 + 10


def test_special_tokens_skipped_in_decode():
    tok = ByteBPE.train(CORPUS, 5)
    ids = tok.encode("low", add_special=True)
    assert ids[0] == 1 and ids[-1] == 2
    assert tok.decode(ids) == "low"


def test_compression_improves():
    assert compression_ratio(ByteBPE.train(CORPUS, 40), CORPUS) > 1.5


def test_sinusoidal_shape_and_range():
    pe = sinusoidal(50, 16)
    assert pe.shape == (50, 16) and np.abs(pe).max() <= 1.0
    assert np.allclose(pe[0, 0::2], 0) and np.allclose(pe[0, 1::2], 1)


def test_rope_preserves_norm_and_is_relative():
    rng = np.random.default_rng(0)
    q, k = rng.normal(size=(1, 8)), rng.normal(size=(1, 8))
    # same q,k content placed at positions (m,n) -> score depends only on m-n
    def score(m, n):
        Q = apply_rope(np.repeat(q, 20, 0))[m]
        K = apply_rope(np.repeat(k, 20, 0))[n]
        return Q @ K
    assert np.isclose(score(5, 2), score(12, 9))
    assert not np.isclose(score(5, 2), score(5, 3))
    x = rng.normal(size=(6, 8))
    assert np.allclose(np.linalg.norm(apply_rope(x), axis=1), np.linalg.norm(x, axis=1))


def test_alibi():
    s = alibi_slopes(8)
    assert np.isclose(s[0], 0.5) and np.all(np.diff(s) < 0)
    b = alibi_bias(4, 0.5)
    assert b[3, 3] == 0 and b[3, 0] == -1.5 and b[0, 3] == -np.inf


def test_embedding_and_mask():
    t = np.arange(12.0).reshape(4, 3)
    assert np.array_equal(embedding_lookup(t, [2, 0]), t[[2, 0]])
    assert padding_mask([2, 4]).tolist() == [[1, 1, 0, 0], [1, 1, 1, 1]]
