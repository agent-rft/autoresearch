# Minimal Depth (DEPTH=2)

## Hypothesis
Further reducing depth from 4 to 2 layers to find the minimum depth needed for this task. The value embeddings + x0 lambdas mechanism should enable strong performance even with minimal depth.

## Results
- **val_bpb**: 0.000883
- **peak_vram_mb**: 2352.6
- **num_params_M**: 3.5 (vs 11.5 for DEPTH=4, vs 50.3 for DEPTH=8)
- **depth**: 2

## Expected
Yes — the 2-layer model achieves the BEST val_bpb yet (0.000883 vs 0.000885 for 4-layer and 0.000939 for 8-layer). The value embeddings + x0 lambdas mechanism enables strong performance even with minimal depth.

## Scaling Summary
| Depth | val_bpb  | params_M |
|-------|----------|----------|
| 2     | 0.000883 | 3.5      |
| 4     | 0.000885 | 11.5     |
| 8     | 0.000939 | 50.3     |

The shallower the better! This suggests the model is highly efficient and extra layers add complexity without benefit.

## Conclusion
DEPTH=2 is the sweet spot — best val_bpb with the fewest parameters. The value embedding + x0 lambda mechanism is very effective even with just 2 transformer layers.
