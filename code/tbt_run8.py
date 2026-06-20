#!/usr/bin/env python3
"""tbt_run8: supports real ShareGPT trace replay (--trace-file).
With an empty trace-file, behavior is identical to v7 (synthetic). In trace mode:
  - a session = a real multi-turn conversation from the trace (human/gpt alternating),
    prefix = the real turn1
  - each human turn is replayed at its real token count; one-shot vs revisit is
    controlled by --revisit-set (one-shot sessions send only turn1; revisited
    sessions send all of their real turns)
  - prefix lengths are heterogeneous; no longer homogenized
v7 docstring:
  K-turn sessions + optional oracle hint. Otherwise same as v6: --revisit-frac
  controls the fraction of sessions that have a second turn (the revisit set is
  randomly sampled by seed, not fixed to the first few). Otherwise same as v5:
  two turns per session:
    turn1 = long prompt -> generates turn_tokens
    (idle for idle_s, during which other sessions' prefixes crowd the pool ->
     this session's prefix is evicted)
    turn2 = same prompt + fixed follow-up -> hit / recover / recompute the prefix
  Decision metric: turn2 TTFT = prefix recovery cost (A=recompute, seconds /
  S=local load / R,P=remote load).
  Note: turn2 does not concatenate turn1's generated text (a tokenizer round-trip
  would break per-token prefix consistency); it reuses only the prompt prefix.
  Fidelity loss = turn_tokens tokens, declared in the record.
Usage:
  python tbt_run4.py --tag M0 --outdir ~/ablogs4 --n-sessions 6 \
      --prompt-tokens 9216 --turn-tokens 128 --idle-s 30 --stagger-s 10 \
      --num-blocks 2048 --max-model-len 12288
"""
import argparse
import asyncio
import json
import os
import random
import socket
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")


def check_environment() -> None:
    try:
        with open("/proc/version") as f:
            if "linux" not in f.read().lower():
                sys.exit("[AB-FATAL] not a Linux environment, exiting.")
    except FileNotFoundError:
        sys.exit("[AB-FATAL] no /proc/version, exiting.")
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(0.3)
    try:
        if s.connect_ex(("127.0.0.1", 8000)) == 0:
            sys.exit("[AB-FATAL] localhost:8000 is in use, kill it first.")
    finally:
        s.close()


def make_long_prompt(i: int, n_tokens: int, seed: int = 42) -> str:
    rng = random.Random(seed + i)
    vocab = ["system", "memory", "cache", "block", "tensor", "kernel", "stream",
             "latency", "bandwidth", "schedule", "queue", "batch", "token",
             "prefill", "decode", "page", "swap", "tier", "policy", "evict"]
    words = [f"session{i}"] + [vocab[rng.randrange(len(vocab))] for _ in range(n_tokens - 1)]
    return "Technical notes:\n" + " ".join(words)


def load_trace_sessions(path: str, n_sessions: int, seed: int = 7,
                        min_prefix: int = 256, max_prefix: int = 4096):
    """Read sessions from the ShareGPT json, stratify-sample n_sessions by turn1
    prefix length, preserving real length heterogeneity. Returns list[dict]:
    {id, turns:[human_text, ...]}.
    Takes only human-turn text (gpt turns are implicitly included in the next
    human turn's concatenated prefix).
    """
    import json as _json
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")
    data = _json.loads(Path(path).expanduser().read_text())
    cand = []
    for conv in data:
        turns = conv.get("conversations", [])
        if not turns or turns[0].get("from") != "human":
            continue
        ok = all(t.get("from") == ("human" if i % 2 == 0 else "gpt")
                 for i, t in enumerate(turns))
        if not ok or len(turns) < 3:
            continue
        p0 = len(tok(turns[0].get("value", ""), add_special_tokens=False)["input_ids"])
        if not (min_prefix <= p0 <= max_prefix):
            continue
        cand.append({"id": conv.get("id", "?"), "p0": p0,
                     "turns": [t["value"] for t in turns]})  # all-turn text (human+gpt alternating)
    cand.sort(key=lambda c: c["p0"])
    if len(cand) < n_sessions:
        raise SystemExit(f"[FATAL] only {len(cand)} candidates < n_sessions {n_sessions}")
    # Stratified sampling: take n_sessions at equal intervals, covering the full prefix-length spectrum
    step = len(cand) / n_sessions
    picked = [cand[int(k * step)] for k in range(n_sessions)]
    rng = random.Random(seed)
    rng.shuffle(picked)  # shuffle arrival order, decoupled from revisit-set
    print(f"[AB-TRACE] {len(cand)} candidates; picked {n_sessions}; "
          f"prefix tokens = {[p['p0'] for p in picked]}", flush=True)
    return picked


async def timed_request(engine, prompt: str, sp, rid: str) -> dict:
    t_submit = time.perf_counter()
    events: list[list[float]] = []
    async for out in engine.generate(prompt, sp, request_id=rid):
        events.append([time.perf_counter(), len(out.outputs[0].token_ids)])
    ttft = (events[0][0] - t_submit) if events else float("nan")
    return {"req": rid, "t_submit": t_submit, "events": events, "ttft_s": ttft}


async def amain(args) -> None:
    from vllm import SamplingParams
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.v1.engine.async_llm import AsyncLLM

    ea_kwargs = dict(
        model="Qwen/Qwen2.5-3B-Instruct-AWQ",
        num_gpu_blocks_override=args.num_blocks,
        gpu_memory_utilization=args.gpu_mem_util,
        max_model_len=args.max_model_len,
        enforce_eager=True,
    )
    if args.offload_gib > 0:
        ea_kwargs["kv_offloading_size"] = args.offload_gib
        if args.eviction_policy:
            from vllm.config import KVTransferConfig
            ea_kwargs["kv_transfer_config"] = KVTransferConfig(
                kv_connector_extra_config={"eviction_policy": args.eviction_policy})
    if args.offload_gib == 0:
        assert "kv_offloading_size" not in ea_kwargs, "baseline contaminated"
    print(f"[AB-KWARGS] tag={args.tag} {ea_kwargs}", flush=True)

    engine = AsyncLLM.from_engine_args(AsyncEngineArgs(**ea_kwargs))
    sp = SamplingParams(temperature=0.0, max_tokens=args.turn_tokens, ignore_eos=True)
    print(f"[AB-EPOCH] epoch={time.time():.3f} perf={time.perf_counter():.3f}",
          flush=True)

    results: list[dict] = []
    trace = None
    if args.trace_file:
        trace = load_trace_sessions(args.trace_file, args.n_sessions)
    if args.revisit_set:
        revisit_set = set(int(x) for x in args.revisit_set.split(","))
    else:
        n_rev = max(1, round(args.revisit_frac * args.n_sessions))
        revisit_set = set(random.Random(123).sample(range(args.n_sessions), n_rev))
    print(f"[AB-REVISIT] frac={args.revisit_frac} set={sorted(revisit_set)}", flush=True)

    def sp_for(i: int):
        if args.hints == "oracle" and i in revisit_set:
            return SamplingParams(temperature=0.0, max_tokens=args.turn_tokens,
                                  ignore_eos=True,
                                  extra_args={"kv_transfer_params":
                                              {"chain_hint": "sticky"}})
        return sp

    async def session(i: int):
        await asyncio.sleep(i * args.stagger_s)
        spi = sp_for(i)
        if trace is not None:
            # Real trace: use the conversation's human turns; prefix accumulates real context per turn
            conv = trace[i]
            human_turns = conv["turns"][0::2]  # human turns are at even positions
            n_turns = len(human_turns) if i in revisit_set else 1
            ctx = ""
            for t in range(1, n_turns + 1):
                if t > 1:
                    await asyncio.sleep(args.idle_s)
                # Accumulate: all prior human+gpt turns + current human turn
                ctx = "".join(conv["turns"][:2 * (t - 1)])  # up to the previous gpt turn
                prompt = ctx + human_turns[t - 1]
                r = await timed_request(engine, prompt, spi, f"{args.tag}-s{i}t{t}")
                r["turn"] = t
                results.append(r)
                print(f"[TURN] {r['req']} ttft={r['ttft_s']:.3f}s", flush=True)
            return
        # Synthetic path (original v7)
        prompt = make_long_prompt(i, args.prompt_tokens)
        n_turns = args.n_turns if i in revisit_set else 1
        for t in range(1, n_turns + 1):
            if t > 1:
                await asyncio.sleep(args.idle_s)
                prompt = prompt + f"\n\nFollow-up {t}: summarize the notes above."
            r = await timed_request(engine, prompt, spi, f"{args.tag}-s{i}t{t}")
            r["turn"] = t
            results.append(r)
            print(f"[TURN] {r['req']} ttft={r['ttft_s']:.3f}s", flush=True)

    t0 = time.perf_counter()
    await asyncio.gather(*[session(i) for i in range(args.n_sessions)])
    wall = time.perf_counter() - t0

    outdir = Path(args.outdir).expanduser()
    outdir.mkdir(parents=True, exist_ok=True)
    fp = outdir / f"{args.tag}.events.jsonl"
    with fp.open("w") as f:
        for r in results:
            r["tag"] = args.tag
            f.write(json.dumps(r) + "\n")

    med = lambda xs: xs[len(xs) // 2] if xs else float("nan")
    parts = []
    max_turn = max((r["turn"] for r in results), default=1)
    for t in range(1, max_turn + 1):
        ts = sorted(r["ttft_s"] for r in results if r["turn"] == t)
        if not ts:
            continue
        parts.append(f"t{t}_med={med(ts):.3f} t{t}_all={[round(x,3) for x in ts]}")
    print(f"[AB-RESULT] tag={args.tag} offload_gib={args.offload_gib} "
          f"wall_s={wall:.2f} hints={args.hints} " + "  ".join(parts), flush=True)
    print(f"[AB-EVENTS] written -> {fp}", flush=True)
    engine.shutdown()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tag", required=True)
    p.add_argument("--outdir", default="ablogs")
    p.add_argument("--offload-gib", type=float, default=0.0)
    p.add_argument("--num-blocks", type=int, default=2048)
    p.add_argument("--gpu-mem-util", type=float, default=0.85)
    p.add_argument("--n-sessions", type=int, default=6)
    p.add_argument("--prompt-tokens", type=int, default=9216)
    p.add_argument("--turn-tokens", type=int, default=128)
    p.add_argument("--idle-s", type=float, default=30.0)
    p.add_argument("--stagger-s", type=float, default=10.0)
    p.add_argument("--max-model-len", type=int, default=12288)
    p.add_argument("--n-turns", type=int, default=3)
    p.add_argument("--trace-file", default="",
                   help="path to ShareGPT json; empty = synthetic workload (same as v7)")
    p.add_argument("--hints", default="none", choices=["none", "oracle"])
    p.add_argument("--revisit-set", default="",
                   help="explicit comma-separated list of revisited session ids, overrides revisit-frac")
    p.add_argument("--revisit-frac", type=float, default=1.0,
                   help="fraction of sessions with a second turn (0~1], sampled by fixed seed")
    p.add_argument("--eviction-policy", default="",
                   help="CPU-tier eviction policy: lru|arc|(custom registered name); empty = engine default (lru)")
    args = p.parse_args()
    check_environment()
    asyncio.run(amain(args))


if __name__ == "__main__":
    main()
