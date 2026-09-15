# RULER prompting policy

## Default: Qwen instruction template

The shipped configurations use:

```yaml
model:
  name: Qwen/Qwen2.5-7B-Instruct
  revision: a09a35458c702b33eeacc393d103063234e8bc28
  prompt_format: chat_template
```

The complete prepared RULER example is passed as one `user` message to the
checkpoint tokenizer:

```python
tokenizer.apply_chat_template(
    [{"role": "user", "content": prompt}],
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt",
)
```

The harness does not add a custom system prompt, rewrite the benchmark
instruction, or append task-specific hints. Qwen's tokenizer-owned template may
supply its own default system text when no explicit system message is provided;
that behavior is part of the pinned tokenizer revision.

This is the primary protocol because the default checkpoint is instruction
tuned and this is its intended model interface. All four attention backends
consume the same rendered token IDs.

## Token-budget enforcement

Chat framing adds template tokens. Therefore every selected example is rendered
with the actual pinned tokenizer before execution. The harness requires:

```text
rendered prompt tokens + task generation budget <= requested context length
```

An over-budget example raises an error rather than being silently truncated.
Use `santapp-ruler validate-data` to check selected examples without loading the
model onto the GPU.

## Explicit raw alternative

A raw-completion comparison remains available:

```yaml
model:
  prompt_format: raw
```

In this mode the tokenizer is called with `add_special_tokens=False`. Treat raw
and chat-template runs as different prompting protocols and label them
separately in result tables.

## Version pin

`a09a35458c702b33eeacc393d103063234e8bc28` is used for both `AutoTokenizer.from_pretrained` and
`AutoModelForCausalLM.from_pretrained`, so model weights, tokenizer files, and
the chat template resolve from one immutable Qwen repository snapshot. The
convenience RULER dataset mirrors still use their separately configured dataset
revision.
