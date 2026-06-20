#!/usr/bin/env python3
"""Regenerate Figure 2 from parsed N-series logs.

This script reads only parsed CSVs produced by parse_logs.py:
  - session_turns.csv
  - run_summary.csv

It filters to the N-series policy runs used in the report:
  - N0s/N0sb/N0sc: LRU
  - N1s/N1sb/N1sc: ARC
  - N3s/N3sb/N3sc: chain-v1

The output plot answers: how does second-turn TTFT differ across LRU, ARC,
and chain-v1 on the symmetric cyclic revisit workload?
"""

from __future__ import annotations

import argparse
import math
import statistics
from collections import OrderedDict
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont


RUNS = OrderedDict(
    [
        ("LRU", ["N0s", "N0sb", "N0sc"]),
        ("ARC", ["N1s", "N1sb", "N1sc"]),
        ("chain-v1", ["N3s", "N3sb", "N3sc"]),
    ]
)
EXPECTED_MEDIAN_BANDS = {
    "LRU": (0.68, 0.72),
    "ARC": (0.68, 0.72),
    "chain-v1": (0.50, 0.58),
}
RECOVERED_LOW_BAND = (0.07, 0.14)
COMPLETE_THRESHOLD = 2000


def load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        r"C:\Windows\Fonts\arialbd.ttf" if bold else r"C:\Windows\Fonts\arial.ttf",
        r"C:\Windows\Fonts\segoeuib.ttf" if bold else r"C:\Windows\Fonts\segoeui.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default()


def median(values: list[float]) -> float:
    return float(statistics.median(values))


def clean_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() == "true"
    return bool(value)


def validate_inputs(session_df: pd.DataFrame, summary_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    run_tags = [tag for tags in RUNS.values() for tag in tags]
    missing_session = sorted(set(run_tags) - set(session_df["run_tag"]))
    missing_summary = sorted(set(run_tags) - set(summary_df["run_tag"]))
    if missing_session or missing_summary:
        raise SystemExit(
            "Missing required N-series runs: "
            f"session_turns missing={missing_session}, run_summary missing={missing_summary}"
        )

    ttft = session_df[
        (session_df["run_tag"].isin(run_tags))
        & (session_df["turn"] == 2)
        & (session_df["source"] == "events")
    ].copy()
    ttft = ttft[pd.notna(ttft["ttft_s"])].copy()
    if len(ttft) != 54:
        raise SystemExit(f"Expected 54 N-series turn=2 event rows, found {len(ttft)}")

    evidence = session_df[
        (session_df["run_tag"].isin(run_tags))
        & (session_df["turn"] == 2)
        & (pd.to_numeric(session_df["hit_tokens"], errors="coerce") >= COMPLETE_THRESHOLD)
    ].copy()
    complete_keys = {
        (str(row.run_tag), int(row.session_id), int(row.turn)): float(row.hit_tokens)
        for row in evidence.itertuples()
    }

    ttft["complete_recovery_from_hit_tokens"] = [
        (str(row.run_tag), int(row.session_id), int(row.turn)) in complete_keys
        for row in ttft.itertuples()
    ]
    ttft["complete_hit_tokens"] = [
        complete_keys.get((str(row.run_tag), int(row.session_id), int(row.turn)), 0.0)
        for row in ttft.itertuples()
    ]
    ttft["policy"] = [
        next(policy for policy, tags in RUNS.items() if tag in tags)
        for tag in ttft["run_tag"]
    ]
    return ttft, evidence


def verify(ttft: pd.DataFrame, summary_df: pd.DataFrame) -> str:
    lines: list[str] = []
    failures: list[str] = []

    lines.append("Per-run second-turn TTFT medians:")
    for policy, tags in RUNS.items():
        low, high = EXPECTED_MEDIAN_BANDS[policy]
        for tag in tags:
            values = ttft.loc[ttft["run_tag"] == tag, "ttft_s"].astype(float).tolist()
            med = median(values)
            summary_med = float(summary_df.loc[summary_df["run_tag"] == tag, "turn2_ttft_s_p50"].iloc[0])
            if abs(med - summary_med) > 1e-9:
                failures.append(f"{tag}: recomputed median {med:.6f} != run_summary {summary_med:.6f}")
            rounded = round(med, 2)
            if not (low <= rounded <= high):
                failures.append(f"{tag}: rounded median {rounded:.2f}s outside expected {low:.2f}-{high:.2f}s")
            lines.append(f"  {tag:5s} {policy:8s} median={med:.6f}s rounded={rounded:.2f}s")

    recovered = ttft[
        (ttft["policy"] == "chain-v1")
        & (ttft["complete_recovery_from_hit_tokens"])
    ].sort_values(["run_tag", "session_id"])
    if len(recovered) != 9:
        failures.append(f"chain-v1 complete recovery rows: expected 9, found {len(recovered)}")

    lines.append("")
    lines.append("chain-v1 complete recoveries from hit-tokens:")
    low_band_count = 0
    for row in recovered.itertuples():
        rounded = round(float(row.ttft_s), 2)
        in_low_band = RECOVERED_LOW_BAND[0] <= rounded <= RECOVERED_LOW_BAND[1]
        low_band_count += int(in_low_band)
        marker = "low-band" if in_low_band else "outside-low-band"
        lines.append(
            f"  {row.run_tag:5s} session={int(row.session_id)} "
            f"hit_tokens={int(row.complete_hit_tokens)} ttft={float(row.ttft_s):.6f}s "
            f"rounded={rounded:.2f}s {marker}"
        )

    if low_band_count < 6:
        failures.append(
            "chain-v1 recovered low-band rows: expected at least 6 complete recoveries "
            f"rounding to {RECOVERED_LOW_BAND[0]:.2f}-{RECOVERED_LOW_BAND[1]:.2f}s, found {low_band_count}"
        )

    if failures:
        raise SystemExit("Verification failed:\n" + "\n".join(f"  - {f}" for f in failures))

    lines.append("")
    lines.append("Verification passed.")
    return "\n".join(lines)


def draw_centered(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, font, fill) -> None:
    bbox = draw.textbbox((0, 0), text, font=font)
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    draw.text((xy[0] - w / 2, xy[1] - h / 2), text, font=font, fill=fill)


def render_plot(ttft: pd.DataFrame, output: Path) -> None:
    width, height = 1200, 800
    left, right, top, bottom = 110, 70, 104, 172
    plot_w, plot_h = width - left - right, height - top - bottom
    y_min, y_max = 0.0, 1.0
    colors = {"LRU": "#4C78A8", "ARC": "#F58518", "chain-v1": "#54A24B"}

    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    title_font = load_font(26, bold=True)
    label_font = load_font(22)
    small_font = load_font(17)
    tiny_font = load_font(15)

    def x_of(policy_idx: int, offset: float = 0.0) -> float:
        span = plot_w / 3
        return left + span * (policy_idx + 0.5 + offset)

    def y_of(value: float) -> float:
        return top + (y_max - value) / (y_max - y_min) * plot_h

    # Grid and axes.
    for tick in [i / 10 for i in range(0, 11, 1)]:
        y = y_of(tick)
        fill = "#E6E8EB" if tick not in (0.0, 1.0) else "#AEB4BA"
        draw.line((left, y, width - right, y), fill=fill, width=1)
        draw.text((left - 48, y - 9), f"{tick:.1f}", font=tiny_font, fill="#4A4F55")

    draw.line((left, top, left, height - bottom), fill="#30343A", width=2)
    draw.line((left, height - bottom, width - right, height - bottom), fill="#30343A", width=2)

    draw_centered(draw, (width // 2, 30), "Second-turn TTFT by policy (symmetric cyclic revisit).", title_font, "#1F2933")
    draw_centered(
        draw,
        (width // 2, 62),
        "Open circles mark complete prefix recoveries (>=2000 hit-tokens).",
        label_font,
        "#1F2933",
    )
    # Rotated y label.
    y_label = Image.new("RGBA", (360, 50), (255, 255, 255, 0))
    yd = ImageDraw.Draw(y_label)
    yd.text((0, 5), "2nd-turn TTFT (s)", font=label_font, fill="#1F2933")
    y_label = y_label.rotate(90, expand=True)
    img.paste(y_label, (22, top + plot_h // 2 - y_label.height // 2), y_label)

    for policy_idx, (policy, tags) in enumerate(RUNS.items()):
        draw_centered(draw, (int(x_of(policy_idx)), height - bottom + 38), policy, label_font, "#1F2933")
        for run_idx, tag in enumerate(tags):
            subset = ttft[ttft["run_tag"] == tag].sort_values("session_id")
            run_offset = [-0.12, 0.0, 0.12][run_idx]
            for row in subset.itertuples():
                sid_offset = (int(row.session_id) - 2.5) * 0.012
                x = x_of(policy_idx, run_offset + sid_offset)
                y = y_of(float(row.ttft_s))
                complete = bool(row.complete_recovery_from_hit_tokens)
                if complete:
                    r = 9
                    draw.ellipse((x - r, y - r, x + r, y + r), fill="white", outline=colors[policy], width=3)
                else:
                    r = 6
                    draw.ellipse((x - r, y - r, x + r, y + r), fill=colors[policy], outline=colors[policy])

            med = median(subset["ttft_s"].astype(float).tolist())
            mx = x_of(policy_idx, run_offset)
            my = y_of(med)
            draw.line((mx - 35, my, mx + 35, my), fill="#111827", width=3)
            draw.text((mx - 22, height - bottom + 72), tag, font=tiny_font, fill="#4A4F55")

    # Legend.
    legend_x, legend_y = width - right - 350, top + 8
    draw.rectangle((legend_x - 16, legend_y - 12, legend_x + 338, legend_y + 132), fill="white", outline="#D1D5DB")
    for idx, policy in enumerate(RUNS):
        y = legend_y + idx * 28
        draw.ellipse((legend_x, y, legend_x + 14, y + 14), fill=colors[policy], outline=colors[policy])
        draw.text((legend_x + 24, y - 2), policy, font=small_font, fill="#1F2933")
    y = legend_y + 92
    draw.ellipse((legend_x, y, legend_x + 18, y + 18), fill="white", outline="#54A24B", width=3)
    draw.text((legend_x + 24, y - 2), "complete recovery (hit-tokens >= 2000)", font=small_font, fill="#1F2933")

    note = "LRU and ARC have identical store/load segment and hit-token counts; visual scatter reflects TTFT measurement jitter."
    caption = "Data: analysis/all_ablogs_parsed/session_turns.csv and run_summary.csv; N-series only."
    draw_centered(draw, (width // 2, height - 46), note, tiny_font, "#4A4F55")
    draw_centered(draw, (width // 2, height - 22), caption, tiny_font, "#4A4F55")

    output.parent.mkdir(parents=True, exist_ok=True)
    img.save(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--parsed-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "all_ablogs_parsed",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent / "fig2_from_logs.png",
    )
    args = parser.parse_args()

    session_path = args.parsed_dir / "session_turns.csv"
    summary_path = args.parsed_dir / "run_summary.csv"
    if not session_path.exists() or not summary_path.exists():
        raise SystemExit(f"Missing parsed CSVs under {args.parsed_dir}")

    session_df = pd.read_csv(session_path)
    summary_df = pd.read_csv(summary_path)
    ttft, _evidence = validate_inputs(session_df, summary_df)
    print(verify(ttft, summary_df))
    render_plot(ttft, args.out)
    print(f"Wrote {args.out.resolve()}")


if __name__ == "__main__":
    main()
