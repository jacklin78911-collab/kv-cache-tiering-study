# KV-Cache Tiering Policies for Multi-Turn LLM Serving

A measurement-and-mechanism study of how the policy layer above a
GPU-to-CPU/CXL-like second memory tier can convert tier capacity into
useful multi-turn prefix recovery, in the context of vLLM's native KV
offloading path.

## Technical Report

**[When LRU Delivers Nothing: An Anatomy of KV-Cache Tiering Policies for Multi-Turn LLM Serving](https://github.com/jacklin78911-collab/kv-cache-tiering-study/blob/main/kv-tiering-technical-report.pdf)** (PDF)

### Summary

- In a controlled multi-turn workload, a second memory tier reduces second-turn TTFT by 4.2× (661 ms → 158 ms) via prefix recovery; a gentle NUMA proxy for CXL-like memory adds no measurable penalty in the tested unsaturated regime.
- Under constrained tier capacity, stock LRU and ARC behave identically and deliver zero complete prefix recoveries on the tested multi-turn workload.
- A simple chain-aware prototype recovers complete prefixes and reduces store traffic on a symmetric revisit workload, but adversarial experiments expose three structured failure modes: starvation, survivorship bias, and a stub equilibrium.
- A real-trace probe using ShareGPT characterizes when the second tier is actually exercised, situating the synthetic workloads and clarifying the regime where tiering policy matters.

### Related Upstream Proposal

This work motivates an upstream RFC to forward request context to vLLM's cache-policy layer: [vllm-project/vllm#45405](https://github.com/vllm-project/vllm/issues/45405).

## Setup

Experiments were run on vLLM v0.21.0 with Qwen2.5-3B-Instruct-AWQ, using an RTX 4060 laptop GPU and a dual-socket server with an RTX 4090D. The experiment harness and policy prototypes are being cleaned up and will be added separately.

## Author

Lin Liqian, Beijing University of Posts and Telecommunications
