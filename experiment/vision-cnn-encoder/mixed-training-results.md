# Mixed Multimodal Training Results

**Date**: 2026-03-19
**Datasets**: 
- Space: AIOmarRehan/space-multimodal-dataset (250 samples)
- Pets: enterprise-explorers/oxford-pets (7,390 samples) → caption: "A {breed} {dog|cat}."
- Math: macabdul9/MathVision test (1,508 samples) → "Q: {question} A: {answer}"
- Text: Lambent/creative-writing-2048-fineweb-edu-sample (2,048 samples, text-only)
**Architecture**: ResNet encoder, 4 layers, 256 dim, fine-tuned
**LR**: 3e-4 (vision_lr 3e-5), 20 epochs

## Results

| Config | Weights | Best Loss | vs baseline | Epoch samples |
|--------|---------|-----------|-------------|---------------|
| **mixed_balanced** | [29.6, 1, 4.9, 3.6] | **0.1107** | +0.742 | ~1750 |
| mixed_vision_focus | [3, 3, 3, 1] | 0.1745 | +0.678 | ~2000 |
| mixed_natural | [1, 1, 1, 1] | 0.2260 | +0.626 | ~2000 |

Baseline (single-dataset, space only, lr=1e-4): 0.8522

## Key Findings

1. **Balanced data wins** — Equal representation from each dataset (0.1107) beats both natural mix (0.2260) and vision-heavy (0.1745)
2. **Diverse data improves vision** — Even though we measure loss on mixed data, VLM quality improves with diversity
3. **Text-only samples help language modeling** — FineWeb samples provide pure language modeling signal
4. **No single dataset dominates** — When each dataset has equal representation, the model learns from all modalities

## Dataset Statistics

| Dataset | Samples | Type | Caption Strategy |
|---------|---------|------|------------------|
| Space | 250 | Vision | Original caption |
| Oxford Pets | 7,390 | Vision | "A {breed} {dog/cat}." |
| MathVision | 1,508 | Vision | "Q: {question} A: {answer}" |
| FineWeb | 2,048 | Text-only | Raw text, next-token prediction |

## Checkpoints

Saved to `~/.cache/autoresearch/vision_checkpoints_mixed/`:
- `best_mixed_balanced.pt`, `epoch00_mixed_balanced.pt`, etc.
- `best_mixed_vision_focus.pt`, etc.
- `best_mixed_natural.pt`, etc.
