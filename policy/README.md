# Chain-Aware Eviction Policy (prototypes v0 - v3)

Prototype `CachePolicy` implementations for vLLM's native KV offloading path,
exploring chain-aware (prefix-completeness-aware) eviction. These are the policy
prototypes behind the failure taxonomy in the technical report.

**For vLLM v0.21.0.** Each installer writes a new `chain.py` into
`vllm/v1/kv_offload/cpu/policies/` and (v0, v2) patches `manager.py` to register
the policy / forward request context. New files carry the Apache-2.0 SPDX header
since they live in the vLLM tree; the installer scripts are this project's work.

## Install order

```bash
python setup_chain_policy.py   # v0: installs chain.py + registers "chain" in manager
python chain_v1_install.py     # v1: whole-chain MRU + guard oldest chains
python setup_chain_v2.py       # v2: forwards req_context; revisit-aware guarding
python chain_v3_install.py     # v3: block-budget guard + ghost-chain credit
```
Each has uniqueness checks, `.bak` backups, and AST verification. `setup_chain_policy.py undo`
and `setup_chain_v2.py undo` restore the pristine manager.

## Version-to-finding map (see technical report §4-§5)

| Version | Mechanism | Outcome in the report |
|---|---|---|
| v0 | head-run-length sort | degenerates to LRU under symmetric load |
| v1 | whole-chain MRU + guard oldest + write-rejection | the chain-v1 result (complete recoveries); honestly "early-arrival pinning" |
| v2 | reuse-credit + sticky-hint guarding | starvation under over-protection |
| v3 | block-budget guard + ghost-chain credit | mitigates two failure modes, still falls into the stub equilibrium |

The point of these prototypes is the *mechanisms of failure they expose*, not a
production-ready policy. See the report for the full analysis.
