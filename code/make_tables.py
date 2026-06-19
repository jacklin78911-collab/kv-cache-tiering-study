#!/usr/bin/env python3
"""Tables 1/2 generated from inline data (LaTeX+Markdown); Table 3 extracted live from ~/ablogs7 logs.
Usage: python make_tables.py [logdir]   # logdir defaults to ~/ablogs7
"""
import re
import statistics as st
import sys
from pathlib import Path

OUT = Path("tables_out"); OUT.mkdir(exist_ok=True)
med = lambda xs: sorted(xs)[len(xs)//2]  # same convention as the harness

# ---------- Table 1: four arms (Phase 3, 4090D) ----------
T1 = {
 "A (recompute)": [[.574,.594,.656,.659,.661,.666],[.579,.594,.658,.661,.662,.681],[.586,.595,.660,.667,.669,.671]],
 "S (local CPU)": [[.151,.153,.154,.157,.161,.164],[.139,.149,.157,.158,.159,.184],[.153,.159,.166,.168,.169,.172]],
 "R (remote, CPU remote)": [[.141,.145,.148,.154,.156,.156],[.153,.154,.154,.154,.157,.183],[.151,.161,.164,.168,.171,.178]],
 "P (remote pool, CPU local)": [[.161,.163,.166,.168,.169,.199],[.156,.160,.164,.165,.169,.201],[.156,.157,.160,.161,.162,.170]],
}
rows1 = []
for arm, runs in T1.items():
    meds = [med(r)*1000 for r in runs]
    allv = sorted(v*1000 for r in runs for v in r)
    rows1.append((arm, f"{med(meds):.0f}", f"{meds[0]:.0f}/{meds[1]:.0f}/{meds[2]:.0f}",
                  f"[{allv[0]:.0f}, {allv[-1]:.0f}]"))
with open(OUT/"table1.md", "w") as f:
    f.write("| Arm | t2 TTFT med (ms) | per-run med | range |\n|---|---|---|---|\n")
    for r in rows1: f.write("| " + " | ".join(r) + " |\n")
with open(OUT/"table1.tex", "w") as f:
    f.write("\\begin{tabular}{lccc}\n\\toprule\nArm & Median (ms) & Per-run & Range \\\\\n\\midrule\n")
    for r in rows1: f.write(" & ".join(r) + " \\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n")

# ---------- Table 2: three policies (symmetric workload, 4060) ----------
T2 = [
 # policy, t2 med x3(s), load ops x3, store ops x3, complete recoveries x3
 ("LRU",      [.864,.871,.792], [0,0,0],  [70,70,70], [0,0,0]),
 ("ARC",      [.892,.837,.838], [0,0,0],  [70,70,70], [0,0,0]),
 ("chain-v1", [.602,.556,.566], [3,3,4],  [52,52,52], [3,3,3]),  # all three runs verified by hit-tokens (2256/2048/2016, identical)
]
with open(OUT/"table2.md", "w") as f:
    f.write("| Policy | t2 TTFT med (s) ×3 | load ops ×3 | store ops ×3 | complete recoveries ×3 |\n|---|---|---|---|---|\n")
    for p, t, l, s_, c in T2:
        f.write(f"| {p} | {'/'.join(f'{x:.3f}' for x in t)} | {'/'.join(map(str,l))} | "
                f"{'/'.join(map(str,s_))} | {'/'.join(map(str,c))} |\n")

# ---------- Table 3: W-series hit-tokens distribution (extracted from logs) ----------
logdir = Path(sys.argv[1]).expanduser() if len(sys.argv) > 1 else Path.home()/"ablogs7"
tags = ["W0","W1","W2","W3h","W3o","W5h","W5o","W4u"]
pat = re.compile(r"Request (\w+)-s(\d+)t(\d+)-\w+ hit (\d+) offloaded tokens after (\d+) GPU hit tokens")
lines = ["| Config | turn | hit-tokens per revisited session (sorted) | complete (≥2000) |",
         "|---|---|---|---|"]
for tag in tags:
    fp = logdir/f"{tag}.log"
    if not fp.exists():
        lines.append(f"| {tag} | - | (log missing) | - |"); continue
    per_turn: dict[int, list[int]] = {}
    for m in pat.finditer(fp.read_text(errors="ignore")):
        if m.group(1) != tag: continue
        per_turn.setdefault(int(m.group(3)), []).append(int(m.group(4)))
    if not per_turn:
        lines.append(f"| {tag} | - | (no hits logged) | 0 |"); continue
    for t in sorted(per_turn):
        v = sorted(per_turn[t])
        lines.append(f"| {tag} | t{t} | {v} | {sum(x>=2000 for x in v)} |")
(OUT/"table3.md").write_text("\n".join(lines) + "\n")
print("tables written to", OUT)
for p in sorted(OUT.iterdir()): print(" -", p.name)
