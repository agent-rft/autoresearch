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

## Generation Examples

Prompt: "A satellite image of Earth showing the" (prefix of GT caption). Temp=0.7, max_new=60.

| Config | Loss | Generation |
|--------|------|------------|
| lr=5e-4 | 0.0125 | "showing the the the the the the the the the..." → repeats "the" after prefix |
| lr=3e-4 | 0.0628 | "showing the the the the the the the the the..." → same repetition |
| 40 epochs | 0.1318 | "showing the the the the..." then degrades to "cloud cloud cloud cloud..." |
| big-6L384 | 0.2021 | "showing the the the..." → "view view view view view" |
| cnn-big | 0.2505 | "showing the the the..." → degrades with noise tokens |
| deep8 | 0.8154 | "showing the the the..." → "t t tAnAnAnAnAn" (severe gibberish) |

### Key observations:
- **Severe overfitting confirmed**: All models, even with very low loss (0.0125), cannot generalize
- **Higher LR = faster memorization**: lr=5e-4 achieves lowest loss but memorizes captions so precisely that generation from partial prompts immediately collapses
- **Deeper = more degeneration**: 8-layer model collapses to gibberish ("AnAnAnAnAn") much faster than 4-layer models
- **"The" repetition is universal**: Every model repeats "the" after the "showing the" prefix — this is the most common token continuation in the 250 captions
- **40-epoch model**: Slightly more diverse before collapse (produces "cloud cloud cloud cloud" — actual semantic content)

### Why the low loss is misleading:
The 250 captions have significant overlap. The model learns that after "A satellite image of Earth showing the", the next most likely token is "the" (repetition from training). This causes immediate repetition collapse on novel prefixes. The models have memorized specific (image, caption) pairs rather than learning the underlying task of image-to-text generation.

## Optimal Configuration

Based on sweep (but note: all models overfit on 250 samples):
- **lr=3e-4**, vision_lr=3e-5, 4 layers, 256 dim, 40+ epochs
- Best expected loss: ~0.01-0.06 range
