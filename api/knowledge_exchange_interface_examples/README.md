# Granularity Examples

These examples use the same `api.unified_online_rl.run_training` runner as the
reference interfaces.

- `layer_grained.py` ranks complete transformer layers by their mean cached FBS score.
- `block_grained.py` ranks consecutive layer blocks. Use
  `BlockGrainedSmallModelScalingInterface(block_size=N)` to select the block size.
- `attention_head_grained.py` ranks complete attention heads (Q/K/V channels are
  selected together) and uses layer groups for projections and MLP layers.  Select
  it with `--knowledge-exchange-granularity attention_head` (the `head` alias is
  also accepted by the VLA adapter example).

The generated high-score groups retain all neurons. Other groups retain the top 2%
of neurons by FBS score. Set `MWE=1` when invoking either `*_verify.sh` to use the
runner's short smoke-test path.
