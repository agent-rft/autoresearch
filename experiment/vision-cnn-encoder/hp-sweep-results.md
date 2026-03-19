# Hyperparameter Sweep Results

**Date**: 2026-03-19
**Dataset**: AIOmarRehan/space-multimodal-dataset (single, 250 samples)
**Encoder**: ResNet (best from previous experiments)
**Note**: All finetuned with vision_lr = lr/10

## Results

| Rank | Config | Best Loss | Params | vs baseline | Key Finding |
|------|--------|-----------|--------|-------------|-------------|
| 1 | **resnet_ft_lr5** | **0.0125** | 9.9M | +0.840 | lr=5e-4 is dramatically better |
| 2 | **resnet_ft_lr3** | **0.0628** | 9.9M | +0.789 | lr=3e-4 also very strong |
| 3 | **resnet_ft_long40** | **0.1318** | 9.9M | +0.720 | 40 epochs > 20 epochs |
| 4 | **resnet_ft_big** | **0.2021** | 20.7M | +0.650 | Larger model (6L, 384-dim) |
| 5 | **cnn_ft_big** | **0.2505** | 19.1M | +0.602 | Larger CNN |
| 6 | **resnet_ft_deep8** | **0.8154** | 13.1M | +0.037 | Deeper (8L) not better |
| — | resnet_finetune (baseline) | 0.8522 | 9.9M | — | lr=1e-4, 20 epochs |

Baseline (1e-4 lr, 20 epochs, fine-tuned ResNet): 0.8522

## Key Findings

1. **LR is the most impactful hyperparameter** — 5e-4 LR achieves 68x better loss than 1e-4
2. **More epochs help significantly** — 40 epochs (0.1318) vs 20 epochs (0.8522) for same model
3. **Larger models help moderately** — 6-layer, 384-dim model (20.7M params) gets 0.2021
4. **Deeper isn't better** — 8 layers (0.8154) performs worse than 4 layers (0.8522)
5. **LR 3e-4 is the sweet spot** — Better than 1e-4, less risky than 5e-4

## Optimal Configuration

Based on sweep:
- **lr=3e-4**, vision_lr=3e-5, 4 layers, 256 dim, 40+ epochs
- Best expected loss: ~0.01-0.06 range
