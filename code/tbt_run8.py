#!/usr/bin/env python3
"""tbt_run8: 支持真实 ShareGPT trace 重放(--trace-file)。
空 trace-file 时行为完全同 v7(合成)。trace 模式下:
  - 会话=trace 里的真实多轮对话(human/gpt 交替), prefix=真实 turn1
  - 每个 human 轮按真实 token 数重放; 一次性 vs 回访由 --revisit-set 控制
    (一次性会话只发 turn1; 回访会话发其全部真实轮)
  - 前缀长度异质, 不再 homogenize
v7 docstring:
  K 轮会话 + 可选 oracle hint。其余同 v6:--revisit-frac 控制有第二轮的会话比例
(回访集合由种子随机抽取, 不固定在前几个)。其余同 v5:每 session 两轮:
  turn1 = 长 prompt -> 生成 turn_tokens
  (空闲 idle_s, 期间其他 session 的前缀挤占池子 -> 本 session 前缀被驱逐)
  turn2 = 同一 prompt + 固定追问 -> 命中/恢复/重算 9k 前缀
裁决指标: turn2 TTFT = 前缀恢复成本 (A=重算秒级 / S=本地load / R,P=远端load)。
注: turn2 不拼接 turn1 生成文本(tokenizer 往返会破坏前缀逐 token 一致性),
只复用 prompt 前缀, 保真度损失=turn_tokens 个 token, 记录声明。
用法:
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
                sys.exit("[AB-FATAL] 非 Linux 环境，退出。")
    except FileNotFoundError:
        sys.exit("[AB-FATAL] 无 /proc/version，退出。")
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(0.3)
    try:
        if s.connect_ex(("127.0.0.1", 8000)) == 0:
            sys.exit("[AB-FATAL] localhost:8000 被占，先 kill。")
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
    """从 ShareGPT json 读会话, 按 turn1 前缀长度分层抽 n_sessions 个,
    保留真实长度异质性。返回 list[dict]: {id, turns:[human_text,...]}。
    只取 human 轮文本(gpt 轮作为前缀累积的一部分隐式包含在下一个 human 的拼接里)。
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
                     "turns": [t["value"] for t in turns]})  # 全轮文本(human+gpt 交替)
    cand.sort(key=lambda c: c["p0"])
    if len(cand) < n_sessions:
        raise SystemExit(f"[FATAL] 候选仅 {len(cand)} < n_sessions {n_sessions}")
    # 分层抽样: 等距取 n_sessions 个, 覆盖前缀长度全谱
    step = len(cand) / n_sessions
    picked = [cand[int(k * step)] for k in range(n_sessions)]
    rng = random.Random(seed)
    rng.shuffle(picked)  # 打乱到达顺序, 与 revisit-set 解耦
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
        assert "kv_offloading_size" not in ea_kwargs, "baseline 被污染"
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
            # 真实 trace: 用对话的 human 轮; prefix 随轮累积真实上下文
            conv = trace[i]
            human_turns = conv["turns"][0::2]  # human 在偶数位
            n_turns = len(human_turns) if i in revisit_set else 1
            ctx = ""
            for t in range(1, n_turns + 1):
                if t > 1:
                    await asyncio.sleep(args.idle_s)
                # 累积: 之前所有 human+gpt 轮 + 当前 human 轮
                ctx = "".join(conv["turns"][:2 * (t - 1)])  # 到上一轮 gpt 为止
                prompt = ctx + human_turns[t - 1]
                r = await timed_request(engine, prompt, spi, f"{args.tag}-s{i}t{t}")
                r["turn"] = t
                results.append(r)
                print(f"[TURN] {r['req']} ttft={r['ttft_s']:.3f}s", flush=True)
            return
        # 合成路径(原 v7)
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
                   help="ShareGPT json 路径; 空=合成负载(同 v7)")
    p.add_argument("--hints", default="none", choices=["none", "oracle"])
    p.add_argument("--revisit-set", default="",
                   help="显式回访会话id逗号表, 覆盖 revisit-frac")
    p.add_argument("--revisit-frac", type=float, default=1.0,
                   help="有第二轮的会话比例(0~1], 种子固定抽取")
    p.add_argument("--eviction-policy", default="",
                   help="CPU 层驱逐策略: lru|arc|(自定义注册名); 空=引擎默认(lru)")
    args = p.parse_args()
    check_environment()
    asyncio.run(amain(args))


if __name__ == "__main__":
    main()
