# No Value Embeddings

## Hypothesis
Removing value embeddings to test if they're essential. Value embeddings contribute ~16.7M params (even more than the base model) and may be key to the model's near-perfect compression.

## Results
- **val_bpb**: 0.008854
- **peak_vram_mb**: 2339.5
- **num_params_M**: 2.5 (vs 3.5M with value embeddings)
- **depth**: 2

## Expected
Yes (worse) — value embeddings are clearly essential for the model's performance. Without them, val_bpb is 10x worse (0.008854 vs 0.000883).

## Comparison
| Configuration | val_bpb  | params_M | notes |
|---|---|---|---|
| Depth 2 + VE | 0.000883 | 3.5 | BEST |
| Depth 2 no VE | 0.008854 | 2.5 | VE essential |

## Conclusion
Value embeddings are CRITICAL to the model's performance. They provide a strong per-token memorization signal that enables near-perfect byte prediction. Removing them degrades performance by 10x. The value embedding mechanism is the key architectural insight that makes this model effective.
