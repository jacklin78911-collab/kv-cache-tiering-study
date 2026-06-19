#!/usr/bin/env python3
"""Generator for report Figures 1-3. Single matplotlib toolchain, outputs PDF+PNG."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({"font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False})

# ---------- Figure 2: 2nd-turn TTFT band scatter ----------
sym = {  # symmetric cyclic workload (N series, 3 runs each)
 "LRU":  [[.541,.548,.568,.871,.875,.932],[.552,.554,.567,.864,.881,.934],[.559,.560,.562,.792,.879,.914]],
 "ARC":  [[.549,.551,.552,.892,.902,.918],[.548,.559,.561,.837,.869,.890],[.548,.554,.564,.838,.877,.944]],
 "chain-v1":[[.076,.115,.568,.602,.859,.881],[.070,.143,.530,.556,.881,.908],[.081,.105,.428,.566,.891,.943]],
}
adv = {  # adversarial revisit set (VA series, 1 run each)
 "LRU":  [[.548,.562,.563,.814,.881,.913]],
 "chain-v1":[[.514,.516,.520,.821,.837,.873]],
}
t1_med_sym, t1_med_adv = 0.84, 0.77  # recompute baseline band

fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6), sharey=True)
for ax, data, t1, title in [(axes[0], sym, t1_med_sym, "Symmetric cyclic revisit"),
                            (axes[1], adv, t1_med_adv, "Adversarial revisit set")]:
    names = list(data)
    ax.axhspan(t1*0.90, t1*1.14, color="0.92", zorder=0)
    ax.text(-0.38, t1*1.02, "recompute", color="0.45", fontsize=7,
            va="center", ha="left")
    rng = np.random.default_rng(7)
    for i, n in enumerate(names):
        pts = np.concatenate(data[n])
        x = i + rng.uniform(-0.12, 0.12, len(pts))
        ax.scatter(x, pts, s=14, alpha=0.8, zorder=3)
    ax.set_xticks(range(len(names)), names)
    ax.set_title(title, fontsize=9)
    ax.set_ylim(0, 1.0)
axes[0].set_ylabel("2nd-turn TTFT (s)")
fig.tight_layout()
fig.savefig("fig2_policy_tiers.pdf"); fig.savefig("fig2_policy_tiers.png", dpi=200)

# ---------- Figure 1: architecture schematic ----------
fig, ax = plt.subplots(figsize=(7.0, 2.4)); ax.axis("off")
def box(x, y, w, h, label, fc="#eef2f7"):
    ax.add_patch(plt.Rectangle((x, y), w, h, fc=fc, ec="0.3", lw=0.8))
    ax.text(x+w/2, y+h/2, label, ha="center", va="center", fontsize=8)
def arrow(x0, y0, x1, y1, label="", dy=0.04):
    ax.annotate("", (x1, y1), (x0, y0),
                arrowprops=dict(arrowstyle="->", lw=0.9, color="0.25"))
    if label:
        ax.text((x0+x1)/2, (y0+y1)/2+dy, label, fontsize=7,
                ha="center", color="0.25")
box(0.02, 0.55, 0.20, 0.32, "Scheduler\n(prefix-cache evictions,\nlookups)")
box(0.02, 0.08, 0.20, 0.32, "GPU worker\n(KV blocks)")
box(0.30, 0.30, 0.22, 0.42, "Offloading\nConnector")
box(0.60, 0.30, 0.18, 0.42, "Offloading\nManager\n(2nd-tier pool)")
box(0.84, 0.30, 0.14, 0.42, "Cache\nPolicy\n(LRU/ARC/...)", fc="#fdebd0")
arrow(0.22, 0.71, 0.30, 0.60); ax.text(0.255, 0.70, "events", fontsize=7, color="0.25")
arrow(0.22, 0.24, 0.30, 0.40); ax.text(0.205, 0.13, "DMA via CUDA streams", fontsize=7, color="0.25")
arrow(0.52, 0.51, 0.60, 0.51, "store/load/\nlookup/touch")
arrow(0.78, 0.51, 0.84, 0.51, "get/insert/\nevict")
ax.text(0.91, 0.18, "opaque keys only:\nno request identity", fontsize=7,
        ha="center", color="#b03a2e")
fig.savefig("fig1_architecture.pdf", bbox_inches="tight")
fig.savefig("fig1_architecture.png", dpi=200, bbox_inches="tight")

# ---------- Figure 3: stub-equilibrium basin, three forces ----------
fig, ax = plt.subplots(figsize=(4.6, 3.0)); ax.axis("off")
ax.add_patch(plt.Circle((0.5, 0.42), 0.16, fc="#f9e7e7", ec="#b03a2e", lw=1.2))
ax.text(0.5, 0.42, "stub\nequilibrium", ha="center", va="center", fontsize=9,
        color="#b03a2e")
forces = [
 (0.10, 0.85, "3:1 capacity\noversubscription", "≤2 complete chains\ncan coexist"),
 (0.50, 0.95, "MRU whole-chain\neviction", "reconstructions are\nnewest → die first"),
 (0.90, 0.85, "credit farming\n(survivorship bias)", "stubs outrank\nreconstructions"),
]
for x, y, name, sub in forces:
    ax.text(x, y, name, ha="center", fontsize=8.5, weight="bold")
    ax.text(x, y-0.10, sub, ha="center", fontsize=7, color="0.35")
    ax.annotate("", (0.5+(x-0.5)*0.32, 0.42+(y-0.55)*0.45), (x, y-0.16),
                arrowprops=dict(arrowstyle="->", lw=1.0, color="0.3"))
ax.text(0.5, 0.06, "each force alone preserves the basin;\nescape requires breaking all three",
        ha="center", fontsize=7.5, style="italic", color="0.3")
fig.savefig("fig3_basin.pdf", bbox_inches="tight")
fig.savefig("fig3_basin.png", dpi=200, bbox_inches="tight")
print("figures ok")
