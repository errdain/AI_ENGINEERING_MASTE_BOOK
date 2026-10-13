"""End-to-end Tier 13 lab: CNN vs ResNet vs ViT on synthetic shapes, then toy CLIP + retrieval.
Run: python run_lab.py   (about 30-50 s on 2 CPU threads)"""
import time
from vision import make_dataset, SHAPES
from models import *


def main():
    Xtr, ys_tr, yc_tr = make_dataset(1200, seed=0)
    Xte, ys_te, yc_te = make_dataset(400, seed=1)
    print(f"data: train {Xtr.shape}, test {Xte.shape}, classes={SHAPES}, chance=25%")
    print(f"{'model':10s} {'params':>8s} {'test acc':>9s} {'secs':>6s}")
    for name, cls in [("CNN", TinyCNN), ("ResNet", TinyResNet), ("ViT", TinyViT)]:
        torch.manual_seed(0)  # seed the weight init too
        m = cls()
        t = time.time()
        acc = fit_classifier(m, Xtr, ys_tr, Xte, ys_te)
        print(f"{name:10s} {count_params(m):8d} {acc:9.3f} {time.time() - t:6.1f}")
    torch.manual_seed(0)
    img, txt = TinyCNN(), TextTower()
    r0 = retrieval_recall(img, txt, Xte, yc_te, ys_te, k=1)
    fit_clip(img, txt, Xtr, yc_tr, ys_tr)
    r1 = retrieval_recall(img, txt, Xte, yc_te, ys_te, k=1)
    print(f"text->image recall@1 (12 captions): before {r0[0]:.2f} | after {r1[0]:.2f} | chance ~{r1[1]:.2f}")


if __name__ == "__main__":
    main()
