#!/usr/bin/env python3
"""ShareGPT trace 探查 + 重放候选筛选。
目的: 在改 harness 之前, 看清真实多轮对话的前缀长度/轮数分布,
据此决定 (a) 选哪些会话做重放 (b) 受限池容量怎么定 (c) 对抗集是否还需人工构造。
不跑引擎, 纯 tokenizer 统计。

前置 (在笔记本上一次性):
  export HF_ENDPOINT=https://hf-mirror.com
  huggingface-cli download anon8231489123/ShareGPT_Vicuna_unfiltered \
    ShareGPT_V3_unfiltered_cleaned_split.json --repo-type dataset \
    --local-dir ~/sharegpt
用法:
  python sharegpt_probe.py ~/sharegpt/ShareGPT_V3_unfiltered_cleaned_split.json
"""
import json
import sys
import statistics as st
from pathlib import Path


def main():
    path = Path(sys.argv[1]).expanduser()
    from transformers import AutoTokenizer
    import os
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")

    data = json.loads(path.read_text())
    print(f"[LOAD] {len(data)} raw conversations")

    # 规整: 仅保留 human 开头、human/gpt 交替、≥2 轮(≥1 个 follow-up)的会话
    sessions = []  # 每条: list[(role, n_tokens)]
    for conv in data:
        turns = conv.get("conversations", [])
        if not turns or turns[0].get("from") != "human":
            continue
        seq = []
        ok = True
        for i, t in enumerate(turns):
            want = "human" if i % 2 == 0 else "gpt"
            if t.get("from") != want:
                ok = False
                break
            n = len(tok(t.get("value", ""), add_special_tokens=False)["input_ids"])
            seq.append((t["from"], n))
        if not ok or len(seq) < 3:  # 至少 human-gpt-human = 一个 t2
            continue
        sessions.append(seq)

    print(f"[CLEAN] {len(sessions)} usable multi-turn sessions "
          f"(human-first, alternating, >=1 follow-up)")

    # 关键分布
    def pctl(xs, ps=(10, 50, 90, 99)):
        xs = sorted(xs)
        return {p: xs[min(len(xs) - 1, int(len(xs) * p / 100))] for p in ps}

    n_human_turns = [sum(1 for r, _ in s if r == "human") for s in sessions]
    # turn1 前缀 = 第一个 human 的 token 数(我们重放的 prefix)
    t1_prefix = [s[0][1] for s in sessions]
    # 累积到第二个 human 之前的上下文 = t2 到来时需恢复的前缀
    t2_context = []
    for s in sessions:
        acc = 0
        for i, (r, n) in enumerate(s):
            acc += n
            if r == "human" and i >= 2:  # 第二个 human
                t2_context.append(acc - n)  # 它之前的累积
                break

    print("\n=== 分布画像 (用于定 harness 参数) ===")
    print(f"human 轮数:       {pctl(n_human_turns)}")
    print(f"turn1 前缀 token: {pctl(t1_prefix)}")
    print(f"t2 恢复上下文:    {pctl(t2_context)}  <- 这是'完整链'的真实长度分布")
    print(f"turn1 前缀均值:   {st.mean(t1_prefix):.0f}  中位: {st.median(t1_prefix):.0f}")

    # 重放候选: 前缀够长(进得了缓存且重算可观)、轮数≥2
    cand = [s for s in sessions if 256 <= s[0][1] <= 4096]
    print(f"\n[CANDIDATES] {len(cand)} sessions with 256<=t1_prefix<=4096 "
          f"(适合 4060/4096-ctx 重放)")
    print("建议: 从候选里按前缀长度分层抽 10-12 个做第一批重放,"
          "保留真实长度异质性(别再homogenize)")


if __name__ == "__main__":
    main()
