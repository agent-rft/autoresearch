# MoE with Shared Expert (4x2)

## Hypothesis
Replace MLP with a Mixture of Experts layer featuring 4 routed experts (top-2 selection) plus a shared expert that always participates. This should increase model capacity in the FFN layers while keeping compute manageable.

## Results
- **val_bpb**: 1.562749
- **peak_vram_mb**: 22071.3
- **total_tokens_M**: ~4.0M (estimated, only ~25 steps completed)
- **num_params_M**: 117.5
- **depth**: 8
- **mfu_percent**: ~0.2%

## Expected
Yes (worse than baseline) — but NOT due to MoE itself. The platform lacks FA3 flash attention and Triton (torch.compile), causing massive throughput degradation (~3K tokens/sec vs ~500K+ on H100 with FA3+compile). Only ~25 steps completed in 50min budget vs ~950 expected. The high val_bpb reflects insufficient training tokens, not architectural failure.

## Analysis
Platform constraints make this experiment non-comparable to baseline:
- No flash attention: SDPA fallback is ~100x slower and more memory-hungry
- No Triton: torch.compile disabled, losing major throughput gains
- Batch size reduced to 16 (from 128) due to MoE memory overhead + SDPA cost

The MoE itself should be fine architecturally — all forward/backward passes worked correctly. The throughput was simply too low to complete meaningful training in 50 minutes.

## Next Steps
1. First establish a baseline on this platform (no MoE, verify setup)
2. Try smaller MoE variants (fewer experts, smaller models)
3. Consider chunked attention for sliding windows without FA3
