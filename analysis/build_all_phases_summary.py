#!/usr/bin/env python3
"""Build a one-row-per-run summary across all archived/live experiment phases.

This wrapper deliberately parses one source directory at a time. Run tags are
not globally unique across phases, so the stable key is:

    (phase, source_dir, run_tag)
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from parse_logs import make_tables


ANALYSIS_DIR = Path(__file__).resolve().parent
EXTRACTED_DIR = ANALYSIS_DIR / "extracted"
OUT_PATH = ANALYSIS_DIR / "all_phases_summary.csv"

LIVE_WSL_DISTRO = "Ubuntu-24.04"
LIVE_ROOTS = {
    "ablogs6": {
        "source_dir": "/home/jacklin/ablogs6",
        "parse_path": Path(rf"\\wsl$\{LIVE_WSL_DISTRO}\home\jacklin\ablogs6"),
    },
    "ablogs7": {
        "source_dir": "/home/jacklin/ablogs7",
        "parse_path": Path(rf"\\wsl$\{LIVE_WSL_DISTRO}\home\jacklin\ablogs7"),
    },
    "ablogs8": {
        "source_dir": "/home/jacklin/ablogs8",
        "parse_path": Path(rf"\\wsl$\{LIVE_WSL_DISTRO}\home\jacklin\ablogs8"),
    },
}

COLUMNS = [
    "phase",
    "run_tag",
    "policy",
    "offload_gib",
    "complete_recovery_count",
    "hit_tokens_p50",
    "hit_tokens_max",
    "second_turn_ttft_p50",
    "store_segments",
    "load_segments",
    "debug_offload_present",
    "source_dir",
]


def phase_for_dir(name: str) -> str:
    if name in {"ablogs", "ablogs2"}:
        return "phase2"
    if name in {"ablogs3", "ablogs4", "ablogs5"}:
        return "phase3"
    if name == "ablogs6":
        return "phase4a"
    if name == "ablogs7":
        return "phase4a-W-series"
    if name == "ablogs8":
        return "phase4b-realtrace"
    return "unknown"


def configured_sources() -> list[dict[str, Any]]:
    sources: list[dict[str, Any]] = []
    for name in ["ablogs", "ablogs2", "ablogs3", "ablogs4", "ablogs5"]:
        path = EXTRACTED_DIR / name
        sources.append({
            "name": name,
            "phase": phase_for_dir(name),
            "source_dir": str(path.resolve()),
            "parse_path": path,
        })

    for name, spec in LIVE_ROOTS.items():
        sources.append({
            "name": name,
            "phase": phase_for_dir(name),
            "source_dir": spec["source_dir"],
            "parse_path": spec["parse_path"],
        })
    return sources


def csv_value(value: Any) -> Any:
    if value is None:
        return "NA"
    return value


def master_row(source: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
    debug_present = bool(row.get("debug_offload_present"))
    recovery_count = row.get("complete_recovery_count") if debug_present else None
    return {
        "phase": source["phase"],
        "run_tag": csv_value(row.get("run_tag")),
        "policy": csv_value(row.get("policy")),
        "offload_gib": csv_value(row.get("offload_gib")),
        "complete_recovery_count": csv_value(recovery_count),
        "hit_tokens_p50": csv_value(row.get("hit_tokens_p50")),
        "hit_tokens_max": csv_value(row.get("hit_tokens_max")),
        "second_turn_ttft_p50": csv_value(row.get("turn2_ttft_s_p50")),
        "store_segments": csv_value(row.get("store_segments")),
        "load_segments": csv_value(row.get("load_segments")),
        "debug_offload_present": debug_present,
        "source_dir": source["source_dir"],
    }


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    all_rows: list[dict[str, Any]] = []
    warnings: list[str] = []

    for source in configured_sources():
        parse_path = Path(source["parse_path"])
        if not parse_path.exists():
            warnings.append(f"missing source directory: {source['source_dir']}")
            continue

        _, run_rows, _, _, source_warnings = make_tables(
            roots=[parse_path],
            complete_threshold=2000,
            include_log_only_hits=True,
        )
        warnings.extend(f"{source['source_dir']}: {warning}" for warning in source_warnings)
        all_rows.extend(master_row(source, row) for row in run_rows)

    seen_keys: set[tuple[str, str, str]] = set()
    duplicate_keys: list[tuple[str, str, str]] = []
    for row in all_rows:
        key = (str(row["phase"]), str(row["source_dir"]), str(row["run_tag"]))
        if key in seen_keys:
            duplicate_keys.append(key)
        seen_keys.add(key)
    if duplicate_keys:
        raise SystemExit(f"duplicate (phase, source_dir, run_tag) keys: {duplicate_keys}")

    all_rows.sort(key=lambda r: (str(r["phase"]), str(r["source_dir"]), str(r["run_tag"])))
    write_csv(all_rows, OUT_PATH)

    print(f"wrote {OUT_PATH}")
    print(f"rows: {len(all_rows)}")
    print("rows by source:")
    for source in configured_sources():
        count = sum(1 for row in all_rows if row["source_dir"] == source["source_dir"])
        status = "present" if Path(source["parse_path"]).exists() else "missing"
        print(f"  {source['phase']}\t{source['source_dir']}\t{status}\t{count} runs")

    if warnings:
        print()
        print("warnings:")
        for warning in sorted(set(warnings)):
            print(f"  - {warning}")


if __name__ == "__main__":
    main()
