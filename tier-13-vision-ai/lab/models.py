"""Tier 13 lab (torch part): tiny CNN, tiny ResNet, tiny ViT, text tower, CLIP-style loss, retrieval."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(2)


class TinyCNN(nn.Module):
    def __init__(self, n_classes=4, width=16, emb=32):
        super().__init__()
        self.f = nn.Sequential(nn.Conv2d(3, width, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                               nn.Conv2d(width, 2 * width, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                               nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(2 * width, emb))
        self.head = nn.Linear(emb, n_classes)

    def embed(self, x):
        return self.f(x)

    def forward(self, x):
        return self.head(F.relu(self.embed(x)))


class ResBlock(nn.Module):
    """y = relu(x + F(x)) -- the identity path lets gradients flow unchanged."""
    def __init__(self, c):
        super().__init__()
        self.c1, self.c2 = nn.Conv2d(c, c, 3, padding=1), nn.Conv2d(c, c, 3, padding=1)

    def forward(self, x):
        return F.relu(x + self.c2(F.relu(self.c1(x))))


class TinyResNet(nn.Module):
    def __init__(self, n_classes=4, width=16, emb=32, blocks=2):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, width, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2))
        self.blocks = nn.Sequential(*[ResBlock(width) for _ in range(blocks)])
        self.pool = nn.Sequential(nn.MaxPool2d(2), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(width, emb))
        self.head = nn.Linear(emb, n_classes)

    def embed(self, x):
        return self.pool(self.blocks(self.stem(x)))

    def forward(self, x):
        return self.head(F.relu(self.embed(x)))


class TinyViT(nn.Module):
    """patchify -> linear patch embedding -> [CLS] + learned positions -> Transformer encoder -> CLS."""
    def __init__(self, size=16, patch=4, d=32, depth=2, heads=4, n_classes=4, emb=32):
        super().__init__()
        self.patch, self.n = patch, (size // patch) ** 2
        self.proj = nn.Linear(patch * patch * 3, d)       # == a conv with kernel=stride=patch
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        self.pos = nn.Parameter(torch.randn(1, self.n + 1, d) * 0.02)
        layer = nn.TransformerEncoderLayer(d, heads, 2 * d, dropout=0.0, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, depth, enable_nested_tensor=False)
        self.norm, self.out = nn.LayerNorm(d), nn.Linear(d, emb)
        self.head = nn.Linear(emb, n_classes)

    def patches(self, x):  # (B,3,H,W) -> (B,N,p*p*3)
        p = self.patch
        B, C, H, W = x.shape
        t = x.reshape(B, C, H // p, p, W // p, p).permute(0, 2, 4, 3, 5, 1)
        return t.reshape(B, (H // p) * (W // p), p * p * C)

    def embed(self, x):
        t = self.proj(self.patches(x))
        t = torch.cat([self.cls.expand(len(t), -1, -1), t], 1) + self.pos
        return self.out(self.norm(self.enc(t)[:, 0]))

    def forward(self, x):
        return self.head(F.relu(self.embed(x)))


class TextTower(nn.Module):
    """Captions here are 2 tokens [colour_id, shape_id]; tower = embed, concat, MLP."""
    def __init__(self, n_colors=3, n_shapes=4, d=16, emb=32):
        super().__init__()
        self.ce, self.se = nn.Embedding(n_colors, d), nn.Embedding(n_shapes, d)
        self.mlp = nn.Sequential(nn.Linear(2 * d, 64), nn.ReLU(), nn.Linear(64, emb))

    def forward(self, color, shape):
        return self.mlp(torch.cat([self.ce(color), self.se(shape)], -1))


def clip_loss(img_emb, txt_emb, temperature=0.07):
    """Symmetric InfoNCE. Logits = cos-sim / T; row i's positive is column i."""
    zi, zt = F.normalize(img_emb, dim=-1), F.normalize(txt_emb, dim=-1)
    logits = zi @ zt.T / temperature
    y = torch.arange(len(zi), device=logits.device)
    return 0.5 * (F.cross_entropy(logits, y) + F.cross_entropy(logits.T, y))


def count_params(m):
    return sum(p.numel() for p in m.parameters())


def fit_classifier(model, X, y, Xv, yv, epochs=30, lr=3e-3, bs=64, seed=0):
    torch.manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    X, y, Xv, yv = map(torch.as_tensor, (X, y, Xv, yv))
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(len(X))
        for i in range(0, len(X), bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            F.cross_entropy(model(X[idx]), y[idx]).backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        return (model(Xv).argmax(1) == yv).float().mean().item()


def fit_clip(img_enc, txt_enc, X, yc, ys, epochs=15, bs=48, lr=3e-3, temperature=0.1, seed=0):
    """Train both towers with the contrastive loss on (image, caption) pairs.
    NOTE: duplicate captions inside a batch are 'false negatives' -- unavoidable with 12 distinct captions;
    the lab keeps bs small relative to #captions-collisions are discussed in the README."""
    torch.manual_seed(seed)
    params = list(img_enc.parameters()) + list(txt_enc.parameters())
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-2)
    X, yc, ys = map(torch.as_tensor, (X, yc, ys))
    for _ in range(epochs):
        perm = torch.randperm(len(X))
        for i in range(0, len(X) - bs + 1, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            clip_loss(img_enc.embed(X[idx]), txt_enc(yc[idx], ys[idx]), temperature).backward()
            opt.step()


@torch.no_grad()
def retrieval_recall(img_enc, txt_enc, X, yc, ys, k=1, n_colors=3, n_shapes=4):
    """Text->image retrieval over the gallery X: query = caption, hit if top-k images
    have the same (colour, shape).  Returns (recall@k, chance)."""
    img_enc.eval(); txt_enc.eval()
    zi = F.normalize(img_enc.embed(torch.as_tensor(X)), dim=-1)
    c = torch.arange(n_colors).repeat_interleave(n_shapes)
    s = torch.arange(n_shapes).repeat(n_colors)
    zt = F.normalize(txt_enc(c, s), dim=-1)
    top = (zt @ zi.T).topk(k, dim=1).indices
    yc, ys = torch.as_tensor(yc), torch.as_tensor(ys)
    hit = ((yc[top] == c[:, None]) & (ys[top] == s[:, None])).any(1).float().mean().item()
    frac = np.mean([((yc == ci) & (ys == si)).float().mean().item() for ci, si in zip(c.tolist(), s.tolist())])
    chance = 1 - (1 - frac) ** k  # approx: k random draws
    return hit, float(chance)
