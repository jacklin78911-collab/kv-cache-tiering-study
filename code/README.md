# Experiment Code

Measurement harness and analysis tooling for the KV-cache tiering study.
This is the first batch (core harness + analysis); policy-installation
scripts that inline vLLM source patches will be added separately.

## Files

| File | Purpose |
|---|---|
| `tbt_run8.py` | Streaming time-between-tokens / TTFT harness. Multi-turn sessions with optional CPU-tier eviction policy and real ShareGPT trace replay (`--trace-file`). |
| `sharegpt_probe.py` | Profiles real multi-turn conversations (prefix-length and turn-count distributions); selects replay candidates. |
| `make_figures.py` | Generates report Figures 1–3 (matplotlib). |
| `make_tables.py` | Generates report Tables 1–3 from inline data and engine logs. |

## Environment

vLLM v0.21.0, Qwen2.5-3B-Instruct-AWQ. Tested on an RTX 4060 laptop (WSL2)
and a dual-socket server with an RTX 4090D.

Set the HF mirror and offline flags before running anything that builds an
engine:

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

## Note on comments

Inline comments and docstrings are currently in Chinese; English translation
is in progress.

## Related

- Technical report: [`../kv-tiering-technical-report.pdf`](../kv-tiering-technical-report.pdf)
- Upstream RFC: [vllm-project/vllm#45405](https://github.com/vllm-project/vllm/issues/45405)
