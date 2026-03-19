# Freeze vs Finetune Vision Encoder Experiment

**Date**: 2026-03-19
**Dataset**: AIOmarRehan/space-multimodal-dataset (250 samples)
**Training**: 20 epochs, batch_size=8, AdamW. Fine-tune lr=1e-4, vision_lr=1e-5.

## Results

| Encoder | Frozen | Finetune | Delta | Winner | Improvement |
|---------|--------|----------|-------|--------|-------------|
| **resnet** | 0.8873 | **0.8522** | +0.0351 | finetune | +4.0% |
| **cnn** | 0.9178 | **0.8922** | +0.0256 | finetune | +2.8% |
| **hybrid** | 0.9739 | **0.9612** | +0.0127 | finetune | +1.3% |
| **vit** | 1.1389 | **1.0352** | +0.1037 | finetune | +9.1% |

## Key Findings

1. **Finetune wins across all encoders** — Every vision encoder benefits from fine-tuning
2. **ViT benefits most from fine-tuning** — Largest absolute improvement (+0.104) and largest relative (9.1%)
3. **ResNet remains best overall** — Both frozen and fine-tuned ResNet achieve lowest loss
4. **Hybrid has smallest gap** — The hybrid CNN+Transformer architecture adapts more easily

## Why Finetune Wins

- Custom vision encoders are trained from scratch on random init
- No pretrained weights to rely on — must adapt to satellite imagery domain
- The `img_proj` layer alone isn't enough to bridge pretrained vision → text space
- Fine-tuning allows the vision encoder to learn satellite-specific features (ocean textures, cloud patterns, Mars terrain)

## Checkpoints

All saved to `~/.cache/autoresearch/vision_checkpoints_ft/`:
- `best_frozen_<encoder>.pt` / `best_finetune_<encoder>.pt`
- `epoch<nn>_frozen_<encoder>.pt` / `epoch<nn>_finetune_<encoder>.pt`

## Configuration

| Parameter | Value |
|-----------|-------|
| Main model lr | 1e-4 |
| Vision encoder lr (finetune) | 1e-5 |
| Weight decay | 0.1 |
| betas | (0.9, 0.95) |
| Gradient clip | 1.0 |
| Epochs | 20 |
| Batch size | 8 |
| Image size | 224x224 |
| Context length | 128 |
| Transformer layers | 4 |
| Embedding dim | 256 |
