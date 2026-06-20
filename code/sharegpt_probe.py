#!/usr/bin/env python3
"""ShareGPT trace profiling + replay-candidate selection.
Purpose: before modifying the harness, characterize the prefix-length / turn-count
distributions of real multi-turn conversations, in order to decide
(a) which sessions to replay, (b) how to size the constrained pool,
(c) whether an adversarial set still needs to be constructed by hand.
Does not run the engine; pure tokenizer statistics.

Prerequisite (once, on the laptop):
  export HF_ENDPOINT=https://hf-mirror.com
  hf download anon8231489123/ShareGPT_Vicuna_unfiltered \
    ShareGPT_V3_unfiltered_cleaned_split.json --repo-type dataset \
    --local-dir ~/sharegpt
Usage:
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

    # Normalize: keep only human-first, strictly alternating human/gpt sessions
    # with >=2 turns (>=1 follow-up).
    sessions = []  # each: list[(role, n_tokens)]
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
        if not ok or len(seq) < 3:  # at least human-gpt-human = one t2
            continue
        sessions.append(seq)

    print(f"[CLEAN] {len(sessions)} usable multi-turn sessions "
          f"(human-first, alternating, >=1 follow-up)")

    # Key distributions
    def pctl(xs, ps=(10, 50, 90, 99)):
        xs = sorted(xs)
        return {p: xs[min(len(xs) - 1, int(len(xs) * p / 100))] for p in ps}

    n_human_turns = [sum(1 for r, _ in s if r == "human") for s in sessions]
    # turn1 prefix = token count of the first human turn (the prefix we replay)
    t1_prefix = [s[0][1] for s in sessions]
    # context accumulated before the second human turn = prefix to recover at t2
    t2_context = []
    for s in sessions:
        acc = 0
        for i, (r, n) in enumerate(s):
            acc += n
            if r == "human" and i >= 2:  # second human turn
                t2_context.append(acc - n)  # accumulation before it
                break

    print("\n=== Distribution profile (for setting harness parameters) ===")
    print(f"human turns:        {pctl(n_human_turns)}")
    print(f"turn1 prefix tokens:{pctl(t1_prefix)}")
    print(f"t2 recovery context:{pctl(t2_context)}  <- real length distribution of a 'complete chain'")
    print(f"turn1 prefix mean:  {st.mean(t1_prefix):.0f}  median: {st.median(t1_prefix):.0f}")

    # Replay candidates: prefix long enough (cacheable, recompute non-trivial), >=2 turns
    cand = [s for s in sessions if 256 <= s[0][1] <= 4096]
    print(f"\n[CANDIDATES] {len(cand)} sessions with 256<=t1_prefix<=4096 "
          f"(suitable for 4060 / 4096-ctx replay)")
    print("Suggestion: stratify-sample 10-12 by prefix length for the first replay batch, "
          "preserving real length heterogeneity (do not homogenize again).")


if __name__ == "__main__":
    main()
