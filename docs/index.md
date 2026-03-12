---
layout: home
---

An open-source reimplementation of the [ROME](https://github.com/kmeng01/rome) (Rank-One Model Editing) method, extended to support every major state-of-the-art open-source language model.

> A joint project by [Security@FIT](https://www.fit.vut.cz/research/group/security@fit/.en) and [Red Hat](https://www.redhat.com) as part of the [LLM Forensics research initiative](https://research.redhat.com/blog/research_project/llm-forensics/).

---

## What it does

ROME enables precise, targeted knowledge injection into transformer LLMs by performing a rank-one update to a single MLP layer. This framework reimplements the method in a clean, modular way and extends support to modern OSS models beyond the original GPT-J scope.

---

## Supported models

| Model | Causal Trace | Weight Editing |
|---|---|---|
| gpt2-medium | ✅ | ✅ |
| gpt2-large | ✅ | ✅ |
| gpt2-xl | ✅ | ✅ |
| gpt-j-6b | ✅ | ✅ |
| qwen3-0.6b | ✅ | ✅ |
| qwen3-1.7b | ✅ | ✅ |
| qwen3-4b | ✅ | ✅ |
| qwen3-8b | ✅ | ✅ |
| granite4-micro | ✅ | — |

---

## Quick start

```bash
# Single ROME edit
python -m src.cli +command=rome model=gpt2-medium

# Compute second-moment statistics (required once per model)
python -m src.cli +command=second-moment model=gpt2-medium
```

See the [README](https://github.com/Security-FIT/Latium#readme) for full usage and configuration options.


---

## Contributors

- [@JakubResh](https://github.com/JakubResh)
- [@olexamatej](https://github.com/olexamatej)

---

[View on GitHub](https://github.com/Security-FIT/Latium) &nbsp;·&nbsp; [Open an issue](https://github.com/Security-FIT/Latium/issues)
