# R006: all-head grouped decode through 4096 nominal tokens

## Scope and status

`grouped_triton` is a new, v16-inspired CUDA backend shared by `santapp` (K-means
parents) and `hierarchical` (contiguous parents). It targets host launch gaps and
per-query-head scheduling, not a promise of beating dense FlashAttention or
exceeding a particular cluster utilization threshold. See `../VALIDATION.md`.
The original numerical prefill kernels, parent policies, frozen actual-key team
leaders, and reference attention estimators are retained.

## Budget contract

With P=16 and R=4, nominal team size is four. The sample budget is NOT a hard
cap on actual selected rows: real teams can have unequal sizes.

| Nominal tokens per query head | Requested teams K | GPU top-K output |
|---:|---:|---:|
| 128 | 32 | 33 |
| 256 | 64 | 65 |
| 512 | 128 | 129 |
| 1024 | 256 | 257 |
| 2048 | 512 | 513 |
| 4096 | 1024 | 1025 |

The grouped runtime supports integer K from 1 through 1024. The smoke covers the
six values above. There is no silent K31/124-token substitution. Configurations
must have integral P/R and a positive nominal budget divisible by P/R. If a KV
head has at most K active teams, all its teams are selected with probability one.
The original Torch decoder remains available for larger budgets.

## Execution

For each transformer layer and decode token:

1. One grouped native-precision route launch operates on all query heads. A CTA
   processes one KV head and 64 leaders, reusing that leader tile across its GQA
   queries. Padded query dimension is 16; supported GQA ratios are 1 through 16.
   It computes FP32 logits and counter-Philox Gumbel priorities.
2. A single batched `torch.topk` CUDA operation across the head dimension returns
   K+1 priorities/IDs per head. This deliberately replaces v16's K31-only local
   top32/merge primitive. PyTorch can launch multiple CUDA kernels internally;
   this implementation does not promise exactly four kernel launches.
3. One compact-plan Triton launch constructs selected-team IDs, prefix sums and
   log inclusion probabilities using the global (K+1)th priority. Epoch stamps
   record GQA union membership without per-token clearing or host row lists.
4. One split-KV selected-attention launch processes all query heads, with 16
   splits by default. Each split traverses selected rows plus the exact generated
   suffix. Binary search over compact prefixes replaces the serial O(K) team-span
   traversal, keeping K=1024 viable without 1024 host launches or giant expanded
   selected K/V tensors.
5. One stable softmax-reduction launch merges splits. A separate bounded
   diagnostics launch records six scalars per KV head, not a selected-row log.

Only route physically groups GQA queries in each CTA. Selected attention is still
per-query/per-split, as in v16; launching all heads together is not the same as
loading their selected K/V union exactly once. The reported GQA union is a
logical traffic statistic, not a measured DRAM transaction count.

## Layout and cache lifecycle

Setup packs existing summaries per layer into contiguous team-major K/V arrays
`[Hkv,prompt_tokens,D]`, native leaders `[Hkv,padded_teams,D]`, starts/lengths, and
an original-row-to-packed-row inverse permutation. Original live K/V is retained
for the exact suffix and the reference path. Packing does not recluster keys.

The final prompt token is re-fed as the first sparse query, exactly as before.
The cache append hook refreshes the rewritten row in both packed K and V. Frozen
routing leaders remain the prefill leaders; they are not recomputed or refreshed.
All prompt rows remain sampled, including that final prompt row. The first sparse
query has zero exact suffix rows; subsequent queries append generated-only rows.

The additional packed prompt K/V storage is `2 * layers * Hkv * N * D * bytes`.
For 28 layers, four KV heads, D=128, and FP16, this is 0.4375 GiB at N=8192 and
1.75 GiB at N=32768, on top of the original cache. Leaders, metadata, workspaces,
model weights and prefill temporaries add more. Peak memory is logged, not
assumed to fit every 24 GB device. No quantization or CPU offload was introduced.

## Numerical and stochastic semantics

Leader scores remain dot(q,leader)/sqrt(D)+log(team_size). Whole teams are selected
without replacement. Conditional inclusion probabilities use the global next
priority, and sampled row logits receive -log(pi); exact suffix rows receive no
correction. Empty softmax splits merge safely.

Native FP16/BF16 leaders are only accepted when casting the stored actual-key
FP32 leaders back to the original cache dtype is lossless. Route uses native MMA
with FP32 accumulation, so score roundoff can differ from the Torch reference.
Counter RNG identities include seed, layer, decode epoch and query head/team.
The old sequential PyTorch RNG stream is not reproduced, and tied `torch.topk`
indices are not promised. Identical generated text for an old seed is therefore
not a correctness requirement. Task-level behavior still needs real-model runs.

## Choosing the backend

Bundled `default.yaml`, `smoke.yaml` and `triton.yaml` explicitly select grouped
decode for both team methods. `torch.yaml` selects Torch prefill and decode.
A legacy config that omits `decode_backend` still resolves to Torch; prefill and
decode are independently selectable.

```bash
santapp-ruler run --config configs/default.yaml \
  --backends santapp hierarchical --samples-per-head 4096 \
  --decode-backend grouped_triton --run-dir runs/new-grouped

# Keep accelerated prefill while restoring only the old decoder:
santapp-ruler run --config configs/default.yaml \
  --backends santapp hierarchical --samples-per-head 4096 \
  --decode-backend torch --run-dir runs/new-torch-reference
```

`decode_splits: 16` is configurable per team backend in YAML (1 through 64).
The supplied deployment tests D=128/GQA=7. Runtime supports D=32/64/128/256;
untested shapes should be validated before research use. CUDA FP16/BF16 only;
there is no hidden CPU or reference fallback after a grouped failure.

## Timing, synchronization, and telemetry

Packing, setup validation and JIT warmup are reported as `decode_prepare_seconds`,
separate from measured decode. The hot decoder has no per-head Python loop or
CUDA-to-CPU `.item()`, `.cpu()` or NumPy conversion. Allocation/layout checks use
host tensor metadata only. Small diagnostics transfer once after timed decode.

With normal EOS stopping enabled, generation retains a per-token EOS scalar
readback. The utilization smoke explicitly disables EOS stopping and performs
128 real autoregressive steps, then transfers token IDs once. This setting is
for measurement, not official accuracy scoring. Reference smoke uses the same
128-step rule. Remaining Transformer/Python launches and prefill gaps can still
reduce utilization; no duty-cycle target is guaranteed.

`SANTAPP_PHASE_LOGS=1` emits timestamped prepare/decode boundaries.
`SANTAPP_GPU_TELEMETRY=1` in the indexed runner samples `nvidia-smi` approximately
once per second, producing `gpu-utilization.csv` and decode-window summaries in
`decode-utilization.json`. This is device busy-time telemetry, not SM occupancy
and not a reproduction of the cluster's aggregation window. Missing samples
produce null summaries rather than invented measurements. No dummy GPU work is
run. See `NAUTILUS_GROUPED_DECODE.md` for deployment.

## External implementation references

- Triton compiler API: https://github.com/triton-lang/triton/blob/main/python/triton/compiler/compiler.py
- PyTorch topk (including tied-index caveat): https://docs.pytorch.org/docs/stable/generated/torch.topk.html
- NRP utilization/resource policies: https://nrp.ai/documentation/userdocs/start/policies/

These references describe external interfaces; they are not evidence that this
new implementation has executed successfully on a GPU.
