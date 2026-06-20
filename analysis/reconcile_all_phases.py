#!/usr/bin/env python3
"""Reconcile headline report numbers against parsed experiment logs."""

from __future__ import annotations

import csv
import statistics
import sys
from pathlib import Path
from typing import Any

from parse_logs import make_tables


ANALYSIS_DIR = Path(__file__).resolve().parent
SUMMARY_PATH = ANALYSIS_DIR / "all_phases_summary.csv"
LIVE_WSL_DISTRO = "Ubuntu-24.04"
ABLOGS6_PARSE = Path(rf"\\wsl$\{LIVE_WSL_DISTRO}\home\jacklin\ablogs6")
ABLOGS7_PARSE = Path(rf"\\wsl$\{LIVE_WSL_DISTRO}\home\jacklin\ablogs7")

PHASE3_EXPECTED_MS = {"A": 661.0, "S": 158.0, "R": 154.0, "P": 165.0}
PHASE3_TOLERANCE_MS = 5.0
CHAIN_EXPECTED_HITS_BY_SESSION = [2256, 2048, 2016]


def read_summary() -> list[dict[str, str]]:
    with SUMMARY_PATH.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def number(value: str) -> float | None:
    if value in {"", "NA", None}:
        return None
    return float(value)


def int_number(value: str) -> int | None:
    parsed = number(value)
    if parsed is None:
        return None
    return int(parsed)


def rows_for_source(rows: list[dict[str, str]], source_suffix: str) -> list[dict[str, str]]:
    return [row for row in rows if row["source_dir"].replace("\\", "/").endswith(source_suffix)]


def row_for(rows: list[dict[str, str]], source_suffix: str, run_tag: str) -> dict[str, str]:
    matches = [row for row in rows_for_source(rows, source_suffix) if row["run_tag"] == run_tag]
    if len(matches) != 1:
        raise SystemExit(f"expected one row for {source_suffix} {run_tag}, found {len(matches)}")
    return matches[0]


def status(ok: bool) -> str:
    return "MATCH" if ok else "MISMATCH vs expected"


def print_check(label: str, ok: bool, actual: str, expected: str, failures: list[str]) -> None:
    print(f"{status(ok)} - {label}")
    print(f"  actual:   {actual}")
    print(f"  expected: {expected}")
    if not ok:
        failures.append(label)


def parse_live_hits(root: Path) -> list[dict[str, Any]]:
    session_rows, _, _, _, _ = make_tables(
        roots=[root],
        complete_threshold=2000,
        include_log_only_hits=True,
    )
    return session_rows


def hit_values(
    session_rows: list[dict[str, Any]],
    run_tag: str,
    *,
    complete_only: bool = False,
    nonzero_only: bool = False,
) -> list[dict[str, Any]]:
    rows = [row for row in session_rows if row["run_tag"] == run_tag]
    if complete_only:
        rows = [row for row in rows if row.get("complete_recovery") is True]
    if nonzero_only:
        rows = [row for row in rows if row.get("hit_tokens") not in (None, 0)]
    return rows


def hit_tokens_by_session(session_rows: list[dict[str, Any]], run_tag: str) -> list[int]:
    rows = hit_values(session_rows, run_tag, complete_only=True)
    rows.sort(key=lambda row: (row.get("session_id") is None, row.get("session_id")))
    return [int(row["hit_tokens"]) for row in rows]


def sorted_nonzero_hits(session_rows: list[dict[str, Any]], run_tag: str) -> list[int]:
    return sorted(int(row["hit_tokens"]) for row in hit_values(session_rows, run_tag, nonzero_only=True))


def main() -> int:
    summary_rows = read_summary()
    failures: list[str] = []

    print("Reconciliation report")
    print(f"source summary: {SUMMARY_PATH}")
    print(f"phase3 TTFT approximate tolerance: +/- {PHASE3_TOLERANCE_MS:.0f} ms")
    print()

    phase3_rows = rows_for_source(summary_rows, "analysis/extracted/ablogs5")
    phase3_parts: list[str] = []
    phase3_ok = True
    for arm, expected_ms in PHASE3_EXPECTED_MS.items():
        vals = [
            number(row["second_turn_ttft_p50"]) * 1000.0
            for row in phase3_rows
            if row["run_tag"].startswith(arm) and number(row["second_turn_ttft_p50"]) is not None
        ]
        median_ms = statistics.median(vals) if vals else None
        if median_ms is None or abs(median_ms - expected_ms) > PHASE3_TOLERANCE_MS:
            phase3_ok = False
        display_vals = ", ".join(f"{v:.1f}" for v in vals)
        phase3_parts.append(f"{arm}: median={median_ms:.1f} ms, runs=[{display_vals}]")
    expected_phase3 = ", ".join(f"{arm}~{expected:.0f} ms" for arm, expected in PHASE3_EXPECTED_MS.items())
    print_check(
        "Phase 3 four-arm 2nd-turn TTFT medians (ablogs5 A/S/R/P)",
        phase3_ok,
        "; ".join(phase3_parts),
        expected_phase3,
        failures,
    )
    print()

    phase4a_policy_ok = True
    phase4a_policy_parts: list[str] = []
    for family, tags in {"LRU": ["N0s", "N0sb", "N0sc"], "ARC": ["N1s", "N1sb", "N1sc"]}.items():
        for tag in tags:
            row = row_for(summary_rows, "ablogs6", tag)
            complete = int_number(row["complete_recovery_count"])
            store = int_number(row["store_segments"])
            load = int_number(row["load_segments"])
            ok = (complete, store, load) == (0, 70, 0)
            phase4a_policy_ok = phase4a_policy_ok and ok
            phase4a_policy_parts.append(f"{tag}({family}): complete={complete}, store={store}, load={load}")
    print_check(
        "Phase 4a LRU/ARC segment and recovery counts (ablogs6)",
        phase4a_policy_ok,
        "; ".join(phase4a_policy_parts),
        "all LRU/ARC N-series symmetric runs complete=0, store=70, load=0",
        failures,
    )
    print()

    ablogs6_sessions = parse_live_hits(ABLOGS6_PARSE)
    chain_ok = True
    chain_parts: list[str] = []
    for tag in ["N3s", "N3sb", "N3sc"]:
        row = row_for(summary_rows, "ablogs6", tag)
        complete = int_number(row["complete_recovery_count"])
        hits = hit_tokens_by_session(ablogs6_sessions, tag)
        ok = complete == 3 and hits == CHAIN_EXPECTED_HITS_BY_SESSION
        chain_ok = chain_ok and ok
        chain_parts.append(f"{tag}: complete={complete}, hits_by_session={hits}")
    print_check(
        "Phase 4a chain-v1 complete recoveries and hit-tokens (ablogs6)",
        chain_ok,
        "; ".join(chain_parts),
        "each of N3s/N3sb/N3sc complete=3, hits_by_session=[2256, 2048, 2016]",
        failures,
    )
    print()

    ablogs7_sessions = parse_live_hits(ABLOGS7_PARSE)
    stub_ok = True
    stub_parts: list[str] = []
    for tag in ["W3h", "W3o", "W5h", "W5o"]:
        row = row_for(summary_rows, "ablogs7", tag)
        complete = int_number(row["complete_recovery_count"])
        hits = sorted_nonzero_hits(ablogs7_sessions, tag)
        ok = complete == 0 and bool(hits) and min(hits) >= 192 and max(hits) <= 224
        stub_ok = stub_ok and ok
        stub_parts.append(f"{tag}: complete={complete}, nonzero_hits={hits}")
    print_check(
        "Phase 4a W-series stub plateau (ablogs7)",
        stub_ok,
        "; ".join(stub_parts),
        "W3h/W3o/W5h/W5o complete=0 and nonzero hit-tokens in [192, 224]",
        failures,
    )
    print()

    w4u_row = row_for(summary_rows, "ablogs7", "W4u")
    w4u_complete = int_number(w4u_row["complete_recovery_count"])
    w4u_hits = sorted_nonzero_hits(ablogs7_sessions, "W4u")
    w4u_ok = w4u_complete == 12 and len(w4u_hits) == 12 and all(v >= 2240 for v in w4u_hits)
    print_check(
        "Phase 4a W4u ceiling (ablogs7)",
        w4u_ok,
        f"W4u: complete={w4u_complete}, nonzero_hits={w4u_hits}",
        "complete=12-equivalent and all nonzero hit-tokens >=2240",
        failures,
    )
    print()

    phase4b_ok = True
    phase4b_parts: list[str] = []
    for tag, expected_store in {"R0": 199, "R2": 548}.items():
        row = row_for(summary_rows, "ablogs8", tag)
        store = int_number(row["store_segments"])
        load = int_number(row["load_segments"])
        ok = (store, load) == (expected_store, 0)
        phase4b_ok = phase4b_ok and ok
        phase4b_parts.append(f"{tag}: store={store}, load={load}")
    r1 = row_for(summary_rows, "ablogs8", "R1")
    phase4b_parts.append(f"R1 observed: store={int_number(r1['store_segments'])}, load={int_number(r1['load_segments'])}")
    print_check(
        "Phase 4b real-trace store/load segment counts (ablogs8)",
        phase4b_ok,
        "; ".join(phase4b_parts),
        "R0 store=199 load=0; R2 store=548 load=0",
        failures,
    )

    print()
    if failures:
        print("MISMATCHES PRESENT - stopping without attempting fixes.")
        for failure in failures:
            print(f"  - {failure}")
        return 1

    print("All requested headline checks MATCH the expected report values.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
