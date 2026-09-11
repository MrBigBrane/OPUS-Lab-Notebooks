# Model portability

Configuration is geometry-aware: query heads, KV heads, head dimension, layer count, and GQA grouping are read from the loaded model. The sparse cache patch is currently guarded to Hugging Face `model_type: qwen2`, covering the intended Qwen2/Qwen2.5 path.

Both hierarchical variants assume that keys entering the custom cache have already received the model's RoPE transformation. The Qwen patch applies RoPE before `_append_cache`:

- `santapp` clusters those cached coordinates directly with global MiniBatchKMeans.
- `hierarchical` forms positional parents but still constructs leaders and teams from those same post-RoPE cached keys.

A port to Llama 8B or another GQA family must validate:

1. attention projection and forward-call signatures;
2. the precise RoPE implementation and whether stored keys are pre- or post-RoPE;
3. cache append/trim behavior and position IDs;
4. query-head-to-KV-head mapping;
5. full-attention versus sliding-window layer policy;
6. tokenizer/chat-template behavior;
7. memory-safe context, generation, and K-means preprocessing settings.

Do not remove the model-type guard merely to make another checkpoint load. First establish stock-SDPA parity with `santapp-ruler fidelity`, then add a regression test proving that the K-means feature tensor equals that model family's post-RoPE cached keys, and finally run task-level smoke tests for both parent policies.


## R005 native-preparation boundary

Native prefill is the default; `--prefill-backend torch` retains the original
reference. The demonstrated GPU workflow is Qwen2.5-7B with head dimension 128.
Additional synthetic dimensions are regression coverage, not a guarantee for
arbitrary models. A10/Triton 3.6.0 D=7 k-means assignment has a known limitation;
see [KNOWN_ISSUES.md](KNOWN_ISSUES.md). Numerical kernels are unchanged from R004.

The A10 two-token smoke passed 8/8 with expandable segments. That is not proof
of memory headroom for every larger S, prompt, or generation length. Keep the
allocator setting, preserve original generation budgets for accuracy testing,
and stress the maximum intended workload before broad deployment.
