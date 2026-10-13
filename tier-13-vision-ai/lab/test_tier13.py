import numpy as np, torch, torch.nn.functional as F, pytest
from vision import *
from models import *


@pytest.mark.parametrize("n,k,s,p", [(16, 3, 1, 0), (16, 3, 1, 1), (17, 3, 2, 1), (32, 7, 2, 3), (5, 5, 1, 0), (10, 4, 3, 0)])
def test_conv_output_size_formula_matches_torch(n, k, s, p):
    out = F.conv2d(torch.zeros(1, 1, n, n), torch.zeros(1, 1, k, k), stride=s, padding=p)
    assert out.shape[-1] == conv_out_size(n, k, s, p)
    assert conv_out_size(n, 3, 1, 1) == n  # "same" padding for k=3


def test_dilated_formula():
    out = F.conv2d(torch.zeros(1, 1, 20, 20), torch.zeros(1, 1, 3, 3), dilation=2)
    assert out.shape[-1] == conv_out_size(20, 3, 1, 0, dilation=2) == 16


def test_conv_equals_manual_loops_and_torch():
    rng = np.random.default_rng(0)
    x, w, b = rng.normal(size=(3, 9, 9)), rng.normal(size=(4, 3, 3, 3)), rng.normal(size=4)
    for s, p in [(1, 0), (2, 1), (1, 2)]:
        mine = conv2d_naive(x, w, b, stride=s, pad=p)
        ref = F.conv2d(torch.tensor(x)[None], torch.tensor(w), torch.tensor(b), stride=s, padding=p)[0].numpy()
        np.testing.assert_allclose(mine, ref, atol=1e-9)


def test_edge_kernel_responds_at_edge_only():
    img = np.zeros((8, 8)); img[:, 4:] = 1
    g = conv2d_single(img, KERNELS["sobel-x"], pad=0)
    assert np.allclose(g[:, 0], 0) and np.all(g[:, 2:4] > 0) and np.allclose(g[:, -1], 0)


def test_max_pool_and_receptive_field():
    x = np.arange(16.).reshape(4, 4)
    np.testing.assert_array_equal(max_pool2d(x), [[5, 7], [13, 15]])
    # three 3x3 stride-1 convs -> RF 7 (same as one 7x7); conv3 + pool2 + conv3 -> 3,4,8
    assert receptive_field([(3, 1)] * 3)[-1][0] == 7
    assert [r for r, _ in receptive_field([(3, 1), (2, 2), (3, 1)])] == [3, 4, 8]


def test_patchify_roundtrip_and_count():
    img = np.random.default_rng(1).normal(size=(3, 16, 16)).astype(np.float32)
    for p in (2, 4, 8, 16):
        pt = patchify(img, p)
        assert pt.shape == ((16 // p) ** 2, p * p * 3)
        np.testing.assert_array_equal(unpatchify(pt, p, 3, 16, 16), img)
    np.testing.assert_array_equal(patchify(img, 4)[1], img[:, 0:4, 4:8].transpose(1, 2, 0).ravel())  # raster order
    with pytest.raises(AssertionError):
        patchify(img, 5)


def test_torch_patches_match_numpy_and_patch_embed_is_conv():
    vit = TinyViT().eval()
    x = torch.randn(2, 3, 16, 16)
    np.testing.assert_allclose(vit.patches(x)[1].numpy(), patchify(x[1].numpy(), 4), atol=1e-6)
    conv = nn.Conv2d(3, 32, 4, stride=4)  # linear on flattened patch == strided conv
    lin_w = conv.weight.permute(0, 2, 3, 1).reshape(32, -1)
    a = vit.patches(x) @ lin_w.T + conv.bias
    b = conv(x).flatten(2).transpose(1, 2)
    torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-4)


def test_iou_known_values():
    assert iou((0, 0, 2, 2), (0, 0, 2, 2)) == 1
    assert iou((0, 0, 2, 2), (3, 3, 4, 4)) == 0
    assert iou((0, 0, 2, 2), (1, 0, 3, 2)) == pytest.approx(2 / 6)
    assert iou((0, 0, 4, 4), (1, 1, 3, 3)) == pytest.approx(4 / 16)
    assert iou((0, 0, 1, 1), (1, 0, 2, 1)) == 0  # touching edges


def test_nms():
    boxes = [(0, 0, 10, 10), (1, 1, 11, 11), (50, 50, 60, 60), (0, 0, 10, 9)]
    scores = [0.8, 0.9, 0.7, 0.6]
    assert nms(boxes, scores, 0.5) == [1, 2]
    assert nms(boxes, scores, 0.99) == [1, 0, 2, 3]
    assert nms(boxes, scores, 1.0) == [1, 0, 2, 3]  # nothing suppressed at thr=1


def test_dataset_deterministic_and_balanced():
    a, ya, _ = make_dataset(64, seed=3)
    b, yb, _ = make_dataset(64, seed=3)
    np.testing.assert_array_equal(a, b); np.testing.assert_array_equal(ya, yb)
    assert a.min() >= 0 and a.max() <= 1 and len(set(ya.tolist())) == 4


def test_normalize():
    x = np.ones((3, 2, 2), np.float32) * 0.5
    np.testing.assert_allclose(normalize(x, [0.5] * 3, [0.25] * 3), 0)


def test_contrastive_loss_properties():
    torch.manual_seed(0)
    B, D = 16, 32
    z = torch.randn(B, D)
    aligned = clip_loss(z, z.clone(), 0.05)
    rand = clip_loss(z, torch.randn(B, D), 0.05)
    assert aligned < 0.01 < rand                                     # matched pairs -> ~0 loss
    assert clip_loss(z, torch.randn(B, D), 1e6).item() == pytest.approx(np.log(B), abs=1e-3)  # T->inf = uniform
    assert clip_loss(z, z * 7.3, 0.1).item() == pytest.approx(clip_loss(z, z, 0.1).item(), abs=1e-5)  # scale invariant
    p = torch.randperm(B)
    assert clip_loss(z, z[p], 0.1).item() == pytest.approx(clip_loss(z[p], z, 0.1).item(), abs=1e-5)  # symmetric
    zt = z.clone().requires_grad_(); clip_loss(zt, z, 0.1).backward(); assert zt.grad.abs().sum() > 0


def test_residual_block_identity_when_f_zero():
    blk = ResBlock(4)
    for p in blk.parameters(): torch.nn.init.zeros_(p)
    x = torch.rand(1, 4, 5, 5)
    torch.testing.assert_close(blk(x), x)  # relu(x + 0) = x for x >= 0


def test_models_forward_and_embed_shapes():
    x = torch.randn(5, 3, 16, 16)
    for m in (TinyCNN(), TinyResNet(), TinyViT()):
        assert m(x).shape == (5, 4) and m.embed(x).shape == (5, 32)


def test_classifiers_beat_chance():
    Xtr, ytr, _ = make_dataset(600, seed=0); Xte, yte, _ = make_dataset(200, seed=1)
    assert fit_classifier(TinyCNN(), Xtr, ytr, Xte, yte, epochs=15) > 0.6   # chance 0.25


def test_retrieval_recall_above_chance_after_clip_training():
    Xtr, ys, yc = make_dataset(900, seed=0); Xte, ys2, yc2 = make_dataset(300, seed=1)
    img, txt = TinyCNN(), TextTower()
    fit_clip(img, txt, Xtr, yc, ys, epochs=12)
    r, chance = retrieval_recall(img, txt, Xte, yc2, ys2, k=1)
    assert r > 3 * chance and r > 0.5
