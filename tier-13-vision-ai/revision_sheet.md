# Tier 13 — Vision AI — Revision Sheet

> Status: DRAFT. Generated from the concept files `13.01-digital-image-representation.md` … `13.32-vision-deployment-performance-considerations.md` (sections 18 *Rapid-fire answer* and 23 *One-line memory hook*); the concept file is the authoritative source.
> Code evidence: see `lab/` (run `python -m pytest -q lab`) and `reviews/author_open_items.md`.

| Concept | Rapid-fire answer | Memory hook |
|---|---|---|
| 13.1 Digital image representation | A grid of quantized colour/intensity samples (H x W x C). | An image is just a big table of numbers. |
| 13.2 Pixels, channels and image tensors | (Batch, Channels, Height, Width) array of numbers, e.g. (B,3,224,224). | Batch, Channel, Height, Width: B-C-H-W, in that order for PyTorch. |
| 13.3 Image preprocessing | Resize, scale, normalize exactly as at training; augment only in training. | Feed the model the pixels it was raised on. |
| 13.4 Convolutional Neural Networks (CNNs) | Stacked shared-weight local filters + pooling: locality and translation-equivariance priors. | Small filters, shared everywhere, stacked deep. |
| 13.5 Convolution | Slide a shared kernel, multiply-and-sum: out = floor((n+2p-k)/s)+1. | Slide, multiply, sum - same weights everywhere. |
| 13.6 Kernels/filters | The small learnable weight array a convolution slides; defines the detected pattern. | A kernel is a tiny pattern the image is asked to match everywhere. |
| 13.7 Stride | Step between kernel positions; stride s shrinks each dimension by ~s. | Stride is how big a hop the kernel takes. |
| 13.8 Padding | Border added before convolution to control output size and edge treatment. | Padding is a frame that keeps edges from being short-changed. |
| 13.9 Pooling | Parameter-free window summary (max/mean) that shrinks maps and adds local invariance. | Pooling keeps the loudest voice in each neighbourhood. |
| 13.10 Receptive fields | Input area that can influence a unit; grows with depth, kernel, stride. | A neuron can only see what its receptive field covers. |
| 13.11 Feature maps | Spatial map of one filter's responses; a layer outputs C of them. | A feature map is a heat map for one pattern. |
| 13.12 Residual vision networks | y = x + F(x): identity shortcut keeps gradients flowing in deep nets. | Learn the change, keep the original. |
| 13.13 ResNet | Four-stage CNN of residual blocks (stem 7x7/2, GAP head); ResNet-50 uses bottlenecks. | ResNet = residual blocks, four stages, one pooled vector. |
| 13.14 Image embeddings | Fixed-size vector from an image encoder; nearby = similar. | An embedding is an image's address in meaning-space. |
| 13.15 Vision Transformer (ViT) | Patch tokens + Transformer encoder (+CLS, +position embeddings). | An image is worth 16x16 words. |
| 13.16 Image patches | Cut image into p x p blocks, flatten each -> N=(H/p)(W/p) tokens. | Chop the picture into tiles; each tile is a token. |
| 13.17 Patch embeddings | Linear (=strided conv) projection of flattened patches to d dims, plus positions. | A patch embedding is a word-embedding for pixels. |
| 13.18 Vision encoders | Pretrained image-to-features network (CNN/ViT) used as a front end. | The encoder is the eyes; everything else is the brain. |
| 13.19 Contrastive learning | InfoNCE: softmax over in-batch similarities where the diagonal is the right answer. | Pull pairs together, push strangers apart. |
| 13.20 CLIP | Contrastively trained image+text encoders sharing one embedding space; zero-shot via text prompts. | One space, two languages: pixels and words. |
| 13.21 Shared vision-language embedding spaces | One vector space where matching images and text are neighbours. | Words and pictures, same map. |
| 13.22 Vision-language models (VLMs) | Vision encoder + projector + LLM: images become tokens the LLM can read. | A VLM is an LLM that learned to read image tokens. |
| 13.23 Cross-modal attention concepts | Text queries attend to image keys/values (or share one sequence). | Words look at pictures through attention. |
| 13.24 Visual instruction tuning | SFT of a VLM on (image, instruction, answer) data. | Teach the VLM to answer, not just describe. |
| 13.25 OCR fundamentals | Detect text regions, recognise characters, assemble reading order. | OCR is detect, read, order - and errors stack. |
| 13.26 Object detection | Predict class + box + score per object; prune duplicates with IoU-based NMS. | Boxes, scores, and NMS to keep only the best. |
| 13.27 Image segmentation | Per-pixel classification (semantic/instance/panoptic). | Every pixel gets a vote. |
| 13.28 Document-image understanding | Extract structured meaning (fields, tables, layout) from page images. | Read the page, not just the words. |
| 13.29 Chart/table visual understanding | Recover cell structure from table images and data values from plots. | Numbers in pictures need a ruler and a reconciliation. |
| 13.30 Diffusion-model introduction | Learn to reverse gradual noising; generate by iterative denoising from noise. | Destroy with noise slowly, learn to repair, generate by repairing noise. |
| 13.31 Vision evaluation | Task-matched metrics, chance baselines, slices and robustness - not one average. | If you didn't slice it, it probably fails somewhere. |
| 13.32 Vision deployment/performance considerations | Profile, then cut pixels/tokens, batch, quantize, cascade - and monitor drift. | Measure first, then shrink pixels, tokens and precision. |
