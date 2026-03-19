# Single Layer (DEPTH=1)

## Hypothesis
Pushing depth to minimum — single transformer layer. The value embeddings + x0 lambdas should enable strong performance even with a single layer.

## Results
- **val_bpb**: 0.000963
- **peak_vram_mb**: 2234.1
- **num_params_M**: 3.3
- **depth**: 1

## Expected
Yes (worse than DEPTH=2) — single layer is slightly worse than 2 layers, confirming DEPTH=2 is the sweet spot.

## Scaling Summary
| Depth | val_bpb  | params_M |
|-------|----------|----------|
| 1     | 0.000963 | 3.3      |
| 2     | 0.000883 | 3.5      |
| 4     | 0.000885 | 11.5     |
| 8     | 0.000939 | 50.3     |

## Conclusion
DEPTH=2 is optimal. Single layer loses some performance, and additional layers (4, 8) add parameters without benefit.
