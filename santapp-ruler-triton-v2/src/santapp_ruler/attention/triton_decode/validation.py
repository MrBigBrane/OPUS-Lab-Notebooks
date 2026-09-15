"""Correctness fixtures/oracles. Never imported or executed inside model decode."""
from __future__ import annotations

from types import SimpleNamespace
import math

import torch

from ..sampling import log_gumbel_tail_probability


def fixture(tokens=513, *, kv_heads=4, dim=128, suffix=0, device="cpu",
            dtype=torch.float16, irregular=True):
    """Actual-key leaders, ragged counts, arbitrary permutations, nontrivial strides."""
    generator = torch.Generator(device=device).manual_seed(7823)
    key_buffer = torch.randn((kv_heads, tokens+suffix+7, dim), device=device,
                             dtype=dtype, generator=generator) * 0.35
    value_buffer = torch.randn(key_buffer.shape, device=device, dtype=dtype, generator=generator)
    key = key_buffer[:, :tokens+suffix]
    value = value_buffer[:, :tokens+suffix]
    summaries = []
    for head in range(kv_heads):
        pattern = (1, 2, 5, 8, 31, 3) if irregular else (4,)
        lens, remaining, i = [], tokens, head
        while remaining:
            n = min(remaining, pattern[i % len(pattern)])
            lens.append(n)
            remaining -= n
            i += 1
        lengths = torch.tensor(lens, dtype=torch.long, device=device)
        starts = lengths.cumsum(0)-lengths
        members = torch.randperm(tokens, generator=generator, device=device)
        leader_ids = members[starts]
        summaries.append(SimpleNamespace(
            members=members, starts=starts, lengths_long=lengths,
            leader_keys=key[head, leader_ids].float(), num_teams=len(lens),
        ))
    return summaries, key, value


def attention_oracle(query, full_key, full_value, summaries, selected, logp, prompt_tokens):
    """Use ORIGINAL live cache rows: catches stale packed-cache and mapping bugs."""
    group = query.shape[0] // full_key.shape[0]
    outputs = []
    for h in range(query.shape[0]):
        kv = h // group
        summary = summaries[kv]
        ids = selected[h][selected[h] >= 0].long()
        rows, correction = [], []
        for slot, team in enumerate(ids.tolist()):
            start = int(summary.starts[team])
            length = int(summary.lengths_long[team])
            rows.append(summary.members[start:start+length])
            correction.append(logp[h, slot].expand(length))
        indices = torch.cat(rows)
        scores = full_key[kv, indices].float() @ query[h].float() / math.sqrt(query.shape[-1])
        scores = scores-torch.cat(correction)
        values = full_value[kv, indices].float()
        if full_key.shape[1] > prompt_tokens:
            exact = full_key[kv, prompt_tokens:].float() @ query[h].float() / math.sqrt(query.shape[-1])
            scores = torch.cat((scores, exact))
            values = torch.cat((values, full_value[kv, prompt_tokens:].float()))
        outputs.append(torch.softmax(scores, 0) @ values)
    return torch.stack(outputs)


def check_selection(decoder):
    """Discrete correctness against the actual FP32 priorities, including ties."""
    p = decoder.packed
    selected = decoder.selected.cpu()
    priority = decoder.priorities.cpu()
    logits = decoder.logits.cpu()
    logp = decoder.logp.cpu()
    prefix = decoder.prefix.cpu()
    rows = decoder.rows.cpu()
    lengths = p.lengths.cpu()
    for h in range(decoder.heads):
        kv = h // decoder.group
        active = p.counts_host[kv]
        count = min(active, decoder.budget)
        ids = selected[h, :count].long()
        assert len(set(ids.tolist())) == count
        assert bool(((ids >= 0) & (ids < active)).all())
        assert bool((selected[h, count:] == -1).all())
        expected_lengths = lengths[kv, ids].long()
        torch.testing.assert_close(prefix[h, 1:count+1].long(), expected_lengths.cumsum(0), rtol=0, atol=0)
        assert int(prefix[h, 0]) == 0
        assert int(rows[h]) == int(expected_lengths.sum())
        assert bool((prefix[h, count:] == rows[h]).all())
        if count == active:
            assert set(ids.tolist()) == set(range(active))
            torch.testing.assert_close(logp[h, :count], torch.zeros(count), rtol=0, atol=0)
        else:
            top = torch.topk(priority[h, :active], count+1, sorted=True)
            actual_values = priority[h, ids]
            # Tied team IDs are not promised by torch.topk; priority multiset is exact.
            torch.testing.assert_close(actual_values.sort(descending=True).values,
                                       top.values[:count], rtol=0, atol=0)
            expected_lp = log_gumbel_tail_probability(logits[h, ids]-top.values[count])
            torch.testing.assert_close(logp[h, :count], expected_lp, rtol=2e-5, atol=2e-5)


def check_statistics(decoder, stats, *, epoch):
    """Exact counts, including union across query heads and variable team sizes."""
    del epoch
    selected = decoder.selected.cpu()
    lengths = decoder.packed.lengths.cpu()
    lp = decoder.logp.cpu()
    stats = stats.cpu()
    for kv in range(decoder.kv_heads):
        block = selected[kv*decoder.group:(kv+1)*decoder.group]
        ids = block[block >= 0].long()
        probs = lp[kv*decoder.group:(kv+1)*decoder.group][block >= 0].exp()
        expected = torch.tensor([
            int(lengths[kv, ids].sum()), int(lengths[kv, ids.unique()].sum()),
            ids.numel(), float(probs.sum()), float(probs.min()), float(probs.max()),
        ], dtype=stats.dtype)
        torch.testing.assert_close(stats[kv, :3], expected[:3], rtol=0, atol=0)
        torch.testing.assert_close(stats[kv, 3:], expected[3:], rtol=2e-5, atol=2e-5)


@torch.inference_mode()
def validate_case(tokens=513, budget=32, *, dtype=torch.float16, suffixes=(0, 1, 17),
                  dim=128, kv_heads=4, group=7, irregular=True, refresh=True,
                  splits=16, summaries_override=None):
    from .runtime import LayerDecoder
    summaries, key, value = fixture(tokens, kv_heads=kv_heads, dim=dim,
                                    suffix=max(suffixes), device="cuda", dtype=dtype,
                                    irregular=irregular)
    if summaries_override is not None:
        summaries = summaries_override(key[:, :tokens])
    decoder = LayerDecoder(summaries, key[:, :tokens], value[:, :tokens],
                           query_heads=kv_heads*group, budget=budget, splits=splits)
    q = torch.randn((kv_heads*group, dim), device="cuda", dtype=dtype)*0.5
    if refresh:
        # Frozen leaders must not change even if the re-fed row was a leader.
        frozen = decoder.packed.leaders.clone()
        key[:, tokens-1].add_(0.125)
        value[:, tokens-1].mul_(0.7)
        decoder.refresh(key, value, tokens-1, tokens)
        torch.testing.assert_close(decoder.packed.leaders, frozen, rtol=0, atol=0)
        for head in range(kv_heads):
            row = int(decoder.packed.inverse[head, tokens-1])
            torch.testing.assert_close(decoder.packed.key[head, row], key[head, tokens-1], rtol=0, atol=0)
            torch.testing.assert_close(decoder.packed.value[head, row], value[head, tokens-1], rtol=0, atol=0)
    reports = []
    for suffix in suffixes:
        stats = torch.empty((kv_heads, 6), device="cuda", dtype=torch.float64)
        output = decoder.run(q, key[:, :tokens+suffix], value[:, :tokens+suffix],
                             seed=0x123456789, layer=2, epoch=suffix, statistics=stats).clone()
        check_selection(decoder)
        check_statistics(decoder, stats, epoch=suffix)
        # Score error is assessed separately from selected-attention error.
        previous_tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        try:
            expected = torch.bmm(q.float().reshape(kv_heads, group, dim),
                                 decoder.packed.leaders.float().transpose(1, 2))/math.sqrt(dim)
        finally:
            torch.backends.cuda.matmul.allow_tf32 = previous_tf32
        for h, active in enumerate(decoder.packed.counts_host):
            ref = expected[h, :, :active]+decoder.packed.lengths[h, :active].float().log()[None, :]
            actual = decoder.logits[h*group:(h+1)*group, :active]
            torch.testing.assert_close(actual, ref, rtol=2e-4, atol=2e-4)
        oracle = attention_oracle(q, key[:, :tokens+suffix], value[:, :tokens+suffix],
                                   summaries, decoder.selected, decoder.logp, tokens)
        tolerance = 2e-2 if dtype == torch.bfloat16 else 3e-3
        torch.testing.assert_close(output.float(), oracle.float(), rtol=tolerance, atol=tolerance)
        assert bool(torch.isfinite(output).all())
        reports.append({"suffix": suffix, "max_output_abs_error": float((output.float()-oracle).abs().max()),
                        "status": "passed"})
    # RNG is deterministic at fixed logical identity and changes with epoch/layer.
    decoder.route(q, seed=99, layer=1, epoch=1)
    a = decoder.priorities.clone()
    decoder.route(q, seed=99, layer=1, epoch=1)
    torch.testing.assert_close(a, decoder.priorities, rtol=0, atol=0)
    decoder.route(q, seed=99, layer=1, epoch=2)
    assert not torch.equal(a, decoder.priorities)
    decoder.route(q, seed=99, layer=2, epoch=1)
    assert not torch.equal(a, decoder.priorities)
    return {"tokens": tokens, "requested_teams": budget, "nominal_budget_P16_R4": budget*4,
            "dtype": str(dtype), "kv_heads": kv_heads, "query_heads": kv_heads*group,
            "team_counts": list(decoder.packed.counts_host), "cases": reports, "status": "passed"}
