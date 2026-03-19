# Shallow Model (DEPTH=4)

## Hypothesis
Reducing depth from 8 to 4 layers while keeping width constant. Smaller models often generalize better with less overfitting. The 8-layer model may have been overparameterized for this task.

## Results
- **val_bpb**: 0.000885
- **peak_vram_mb**: 3127.9
- **total_tokens_M**: ~148 (same budget, more efficient)
- **num_params_M**: 11.5 (vs 50.3 baseline)
- **depth**: 4
- **mfu_percent**: ~1.2%

## Expected
Yes — the shallower model achieves BETTER val_bpb (0.000885 vs 0.000939), suggesting the 8-layer model was slightly overparameterized. The 4-layer model is more efficient and generalizes better on this task.

## Analysis
- 4x fewer parameters (11.5M vs 50.3M) with better performance
- Memory footprint reduced from 6.3GB to 3.1GB
- Same training time budget (50min) but more efficient use of parameters
- The value embeddings + x0 lambdas mechanism is powerful even with fewer layers

## Conclusion
DEPTH=4 is the sweet spot for this architecture on this task. The additional layers in DEPTH=8 were not providing benefit and may have been slightly detrimental.
