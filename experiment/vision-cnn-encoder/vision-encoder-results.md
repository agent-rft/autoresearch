# Vision Encoder Experiment Results

**Date**: 2026-03-19
**Dataset**: AIOmarRehan/space-multimodal-dataset (250 samples)
**Task**: Vision-language modeling on satellite images (Earth/Mars)
**Architecture**: Custom VLM (Vision Encoder + Multimodal Transformer Decoder)
**Training**: 20 epochs, batch_size=8, lr=1e-4, AdamW

## Results

| Rank | Encoder | Final Loss | Params | Checkpoints |
|------|---------|-----------|--------|------------|
| 1 | **resnet** | **0.7784** | 9.9M | best, epoch00, 04, 08, 12, 16, 19 |
| 2 | cnn | 0.8764 | 9.3M | best, epoch00, 04, 08, 12, 16, 19 |
| 3 | hybrid | 0.8784 | 9.5M | best, epoch00, 04, 08, 12, 16, 19 |
| 4 | vit | 0.9339 | 10.9M | best, epoch00, 04, 08, 12, 16, 19 |

## Model Architecture

- **Vision Encoder**: Custom (no external deps)
  - CNN: 4-layer custom convnet with residual connections, AdaptiveAvgPool2d → 4x4
  - ViT: 4-layer Transformer encoder, 16x16 patches
  - ResNet: 4-stage ResNet-style with skip connections, AdaptiveAvgPool2d → 4x4
  - Hybrid: 3-stage CNN + 2-layer Transformer encoder

- **Multimodal Transformer**: 4 layers, 4 heads, 256 dim, RMSNorm
- **Image tokens**: num_patches + 1 CLS token
- **Fusion**: concat [img_tokens, text_tokens] → positional embedding → transformer → lm_head
- **x0 residual**: x = x + 0.1 * x0 for skip connection

## Key Observations

1. **ResNet wins** — The deeper residual architecture (4 stages) outperforms shallow CNN
2. **CNN and Hybrid are close** — Both around 0.87-0.88 final loss
3. **ViT is worst** — Despite having most parameters (10.9M), performs worst
4. **All encoders learn well** — Loss drops from ~8.3 to <1.0 over 20 epochs

## Checkpoints

All saved to `~/.cache/autoresearch/vision_checkpoints/`:
- `best_<encoder>.pt` — Best checkpoint by val loss
- `epoch<nn>_<encoder>.pt` — Checkpoints at epochs 0, 4, 8, 12, 16, 19

## Bug Fixes

1. CNN encoder: Fixed `nn.Linear(512*16, embed_dim*2)` → `nn.Linear(512, embed_dim)` (was applying to wrong tensor shape)
2. Image normalization: Replaced buggy `ByteStorage.from_buffer` with `np.array()` + PIL resize
3. Typo: `model.state()` → `model.state_dict()`
