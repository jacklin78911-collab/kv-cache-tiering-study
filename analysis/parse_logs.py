#!/usr/bin/env python3
"""Parse KV-cache tiering experiment logs into tidy analysis tables.

Inputs:
  - JSONL event traces written by tbt_run*.py, one request per line.
  - vLLM/offload DEBUG sidecar logs containing:
      * "Submitted ('GPU', 'CPU') transfer ..." store segments
      * "Submitted ('CPU', 'GPU') transfer ..." load segments
      * "Request ... hit N offloaded tokens after M GPU hit tokens"

Outputs:
  - session_turns.csv: one row per session turn/request.
  - run_summary.csv: one row per run tag.
  - hit_tokens.csv: per-request maximum recovered hit-tokens.
  - transfer_ops.csv: one row per observed transfer segment.

Only metrics already present in the logs are extracted. When a sidecar log is
missing, or when DEBUG offload lines are absent, the affected counts are left as
NA instead of being inferred.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import re
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

try:
    import pandas as pd
except ModuleNotFoundError:  # pragma: no cover - exercised only without pandas.
    pd = None


REQUEST_PATTERNS = (
    re.compile(r"^(?P<tag>.+?)-s(?P<session>\d+)t(?P<turn>\d+)(?:[-_].*)?$"),
    re.compile(r"^(?P<tag>.+?)-req(?P<session>\d+)(?:[-_].*)?$"),
    re.compile(r"^(?P<session>\d+)(?:[-_].*)?$"),
)
HIT_RE = re.compile(
    r"Request\s+(?P<req>\S+)\s+hit\s+(?P<hit>\d+)\s+offloaded tokens"
    r"\s+after\s+(?P<gpu>\d+)\s+GPU hit tokens"
)
TRANSFER_RE = re.compile(
    r"Submitted\s+\('(?P<src>GPU|CPU)',\s+'(?P<dst>GPU|CPU)'\)\s+"
    r"transfer\s+(?P<transfer_id>\d+)"
)
AB_KWARGS_RE = re.compile(r"\[AB-KWARGS\]\s+tag=(?P<tag>\S+)\s+(?P<kwargs>\{.*\})")
AB_RESULT_RE = re.compile(r"\[AB-RESULT\]\s+tag=(?P<tag>\S+)\s+(?P<body>.*)")
AB_REVISIT_RE = re.compile(r"\[AB-REVISIT\].*set=\[(?P<set>[^\]]*)\]")
TAG_POLICY_OVERRIDES = {
    # Report Figure/Table 2 symmetric policy runs.
    "N0s": "LRU",
    "N0sb": "LRU",
    "N0sc": "LRU",
    "N1s": "ARC",
    "N1sb": "ARC",
    "N1sc": "ARC",
    "N3s": "chain-v1",
    "N3sb": "chain-v1",
    "N3sc": "chain-v1",
    # Report Figure 2 adversarial revisit-set runs.
    "VA0": "LRU",
    "VA1": "chain-v1",
}


@dataclass
class LogParse:
    tag: str
    log_path: Path
    policy: str | None = None
    offload_gib: float | None = None
    hints: str | None = None
    revisit_set: set[int] | None = None
    store_segments: int | None = None
    load_segments: int | None = None
    debug_offload_present: bool = False
    hit_by_request: dict[str, dict[str, Any]] = field(default_factory=dict)
    transfer_rows: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def parse_number(value: str) -> int | float | str:
    try:
        if any(ch in value for ch in ".eE"):
            return float(value)
        return int(value)
    except ValueError:
        return value


def percentile(values: Iterable[float], p: float) -> float | None:
    xs = sorted(v for v in values if v is not None and not math.isnan(v))
    if not xs:
        return None
    k = min(len(xs) - 1, max(0, round((p / 100.0) * (len(xs) - 1))))
    return xs[k]


def median(values: Iterable[float]) -> float | None:
    xs = [v for v in values if v is not None and not math.isnan(v)]
    if not xs:
        return None
    return float(statistics.median(xs))


def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        if math.isnan(value):
            return "NA"
        return f"{value:.{digits}f}"
    return str(value)


def request_parts(req_id: str, default_tag: str) -> dict[str, Any]:
    for pattern in REQUEST_PATTERNS:
        m = pattern.match(req_id)
        if not m:
            continue
        groups = m.groupdict()
        return {
            "run_tag": groups.get("tag") or default_tag,
            "session_id": int(groups["session"]),
            "turn": int(groups.get("turn") or 1),
        }
    return {"run_tag": default_tag, "session_id": None, "turn": None}


def discover_roots(paths: list[str]) -> list[Path]:
    if paths:
        roots = [Path(p).expanduser().resolve() for p in paths]
        return [p for p in roots if p.exists()]

    home = Path.home()
    candidates: list[Path] = []
    candidates.extend(home.glob("ablogs*"))
    desktop = home / "Desktop"
    if desktop.exists():
        candidates.extend(desktop.glob("*/ablogs*"))
        candidates.extend(desktop.glob("ablogs*"))
    cwd = Path.cwd()
    candidates.extend(cwd.glob("ablogs*"))
    candidates.extend(cwd.glob("*/ablogs*"))

    seen: set[Path] = set()
    roots: list[Path] = []
    for path in candidates:
        try:
            resolved = path.resolve()
        except OSError:
            continue
        if resolved in seen or not resolved.is_dir():
            continue
        seen.add(resolved)
        roots.append(resolved)
    return sorted(roots)


def parse_body_kv(body: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for token in body.split():
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        out[key] = parse_number(value.rstrip(","))
    return out


def infer_policy(tag: str, kwargs: dict[str, Any] | None, offload_gib: float | None) -> str:
    if tag in TAG_POLICY_OVERRIDES:
        return TAG_POLICY_OVERRIDES[tag]
    if kwargs:
        cfg = kwargs.get("kv_transfer_config")
        cfg_text = repr(cfg)
        m = re.search(r"eviction_policy['\"]?\s*[:=]\s*['\"]([^'\"]+)", cfg_text)
        if m:
            return m.group(1)
        extra = kwargs.get("kv_connector_extra_config")
        if isinstance(extra, dict) and extra.get("eviction_policy"):
            return str(extra["eviction_policy"])
        if kwargs.get("kv_offloading_size") is None:
            return "baseline"

    if offload_gib == 0:
        return "baseline"
    if offload_gib and offload_gib > 0:
        return "lru"
    if tag.startswith(("TA", "A")):
        return "baseline"
    if tag.startswith(("TS", "S")):
        return "lru"
    return "unknown"


def parse_log_file(path: Path, default_tag: str) -> LogParse:
    parsed = LogParse(tag=default_tag, log_path=path)
    kwargs: dict[str, Any] | None = None
    store_segments = 0
    load_segments = 0

    if not path.exists():
        parsed.warnings.append(f"missing sidecar log: {path}")
        return parsed

    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line_no, line in enumerate(f, start=1):
            if m := AB_KWARGS_RE.search(line):
                parsed.tag = m.group("tag")
                try:
                    kwargs = ast.literal_eval(m.group("kwargs"))
                except (SyntaxError, ValueError):
                    kwargs = None
                if kwargs and "kv_offloading_size" in kwargs:
                    parsed.offload_gib = float(kwargs["kv_offloading_size"])
            if m := AB_RESULT_RE.search(line):
                parsed.tag = m.group("tag")
                kv = parse_body_kv(m.group("body"))
                if "offload_gib" in kv:
                    parsed.offload_gib = float(kv["offload_gib"])
                if "hints" in kv:
                    parsed.hints = str(kv["hints"])
            if m := AB_REVISIT_RE.search(line):
                items = [x.strip() for x in m.group("set").split(",") if x.strip()]
                parsed.revisit_set = {int(x) for x in items}

            if m := HIT_RE.search(line):
                parsed.debug_offload_present = True
                req_id = m.group("req")
                hit_tokens = int(m.group("hit"))
                gpu_hit_tokens = int(m.group("gpu"))
                old = parsed.hit_by_request.get(req_id)
                if old is None or hit_tokens >= old["hit_tokens"]:
                    parsed.hit_by_request[req_id] = {
                        "request_id": req_id,
                        "hit_tokens": hit_tokens,
                        "gpu_hit_tokens": gpu_hit_tokens,
                        "hit_line": line_no,
                    }

            if m := TRANSFER_RE.search(line):
                parsed.debug_offload_present = True
                src = m.group("src")
                dst = m.group("dst")
                if src == "GPU" and dst == "CPU":
                    op = "store"
                    store_segments += 1
                elif src == "CPU" and dst == "GPU":
                    op = "load"
                    load_segments += 1
                else:
                    op = "other"
                parsed.transfer_rows.append({
                    "run_tag": parsed.tag,
                    "log_path": str(path),
                    "line_no": line_no,
                    "transfer_id": int(m.group("transfer_id")),
                    "src": src,
                    "dst": dst,
                    "op": op,
                })

    if parsed.debug_offload_present:
        parsed.store_segments = store_segments
        parsed.load_segments = load_segments
    else:
        parsed.store_segments = None
        parsed.load_segments = None
        parsed.warnings.append(f"no DEBUG offload transfer/hit lines in: {path}")
    parsed.policy = infer_policy(parsed.tag, kwargs, parsed.offload_gib)
    return parsed


def parse_event_file(path: Path, default_tag: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as exc:
                rows.append({
                    "run_tag": default_tag,
                    "request_id": None,
                    "session_id": None,
                    "turn": None,
                    "parse_error": f"{path}:{line_no}: {exc}",
                })
                continue
            req_id = str(rec.get("req", ""))
            parts = request_parts(req_id, default_tag)
            turn = int(rec.get("turn") or parts["turn"] or 1)
            events = rec.get("events") or []
            if rec.get("ttft_s") is not None:
                ttft_s = float(rec["ttft_s"])
            elif events:
                ttft_s = float(events[0][0]) - float(rec["t_submit"])
            else:
                ttft_s = None
            rows.append({
                "run_tag": str(rec.get("tag") or parts["run_tag"]),
                "request_id": req_id,
                "session_id": parts["session_id"],
                "turn": turn,
                "t_submit": rec.get("t_submit"),
                "ttft_s": ttft_s,
                "prompt_tokens": rec.get("prompt_tokens"),
                "generated_tokens": events[-1][1] if events else 0,
                "events_path": str(path),
                "source": "events",
                "parse_error": None,
            })
    return rows


def assign_arrival_order(rows: list[dict[str, Any]]) -> None:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["run_tag"])].append(row)
    for run_rows in grouped.values():
        first_turns = [
            r for r in run_rows
            if r.get("turn") == 1 and r.get("session_id") is not None
        ]
        first_turns.sort(key=lambda r: (
            float(r["t_submit"]) if r.get("t_submit") is not None else math.inf,
            int(r["session_id"]),
        ))
        order_by_session = {
            int(r["session_id"]): idx for idx, r in enumerate(first_turns)
        }
        for row in run_rows:
            sid = row.get("session_id")
            if sid is None:
                row["arrival_order"] = None
                row["stagger_index"] = None
            else:
                row["arrival_order"] = order_by_session.get(int(sid), int(sid))
                row["stagger_index"] = int(sid)


def make_tables(
    roots: list[Path],
    complete_threshold: int,
    include_log_only_hits: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    event_rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    logs_by_tag: dict[str, LogParse] = {}
    event_files_by_tag: dict[str, Path] = {}

    for root in roots:
        for event_path in sorted(root.glob("*.events.jsonl")):
            tag = event_path.name.replace(".events.jsonl", "")
            event_files_by_tag[tag] = event_path
            event_rows.extend(parse_event_file(event_path, tag))
        for log_path in sorted(root.glob("*.log")):
            tag = log_path.stem
            parsed = parse_log_file(log_path, tag)
            key = parsed.tag
            if key in logs_by_tag:
                key = tag
                parsed.tag = tag
            logs_by_tag[key] = parsed

    for tag, event_path in event_files_by_tag.items():
        if tag not in logs_by_tag:
            warnings.append(f"missing sidecar log for event file: {event_path}")
            logs_by_tag[tag] = LogParse(tag=tag, log_path=event_path.with_suffix(".log"))
            logs_by_tag[tag].warnings.append(f"missing sidecar log: {event_path.with_suffix('.log')}")

    for parsed in logs_by_tag.values():
        warnings.extend(parsed.warnings)

    assign_arrival_order(event_rows)

    hit_rows: list[dict[str, Any]] = []
    for parsed in logs_by_tag.values():
        for req_id, hit in parsed.hit_by_request.items():
            parts = request_parts(req_id, parsed.tag)
            hit_rows.append({
                "run_tag": parts["run_tag"],
                "request_id": req_id,
                "session_id": parts["session_id"],
                "turn": parts["turn"],
                "hit_tokens": hit["hit_tokens"],
                "gpu_hit_tokens": hit["gpu_hit_tokens"],
                "complete_recovery": hit["hit_tokens"] >= complete_threshold,
                "hit_line": hit["hit_line"],
                "log_path": str(parsed.log_path),
            })

    hit_by_req = {row["request_id"]: row for row in hit_rows}
    out_rows: list[dict[str, Any]] = []
    seen_requests: set[str] = set()
    for row in event_rows:
        tag = str(row["run_tag"])
        parsed = logs_by_tag.get(tag)
        hit = hit_by_req.get(row["request_id"])
        debug_present = parsed.debug_offload_present if parsed else False
        hit_tokens = hit["hit_tokens"] if hit else (0 if debug_present else None)
        gpu_hit_tokens = hit["gpu_hit_tokens"] if hit else (0 if debug_present else None)
        is_revisited = bool(
            (row.get("turn") and row["turn"] > 1)
            or (
                parsed
                and parsed.revisit_set is not None
                and row.get("session_id") in parsed.revisit_set
            )
        )
        complete = None if hit_tokens is None else hit_tokens >= complete_threshold
        out_rows.append({
            **row,
            "policy": parsed.policy if parsed else None,
            "offload_gib": parsed.offload_gib if parsed else None,
            "hints": parsed.hints if parsed else None,
            "is_revisited": is_revisited,
            "hit_tokens": hit_tokens,
            "gpu_hit_tokens": gpu_hit_tokens,
            "complete_recovery": complete,
            "log_path": str(parsed.log_path) if parsed else None,
            "run_store_segments": parsed.store_segments if parsed else None,
            "run_load_segments": parsed.load_segments if parsed else None,
            "run_debug_offload_present": debug_present,
        })
        seen_requests.add(row["request_id"])

    if include_log_only_hits:
        for hit in hit_rows:
            if hit["request_id"] in seen_requests:
                continue
            tag = str(hit["run_tag"])
            parsed = logs_by_tag.get(tag)
            is_revisited = bool(
                (hit.get("turn") and hit["turn"] > 1)
                or (
                    parsed
                    and parsed.revisit_set is not None
                    and hit.get("session_id") in parsed.revisit_set
                )
            )
            out_rows.append({
                "run_tag": tag,
                "request_id": hit["request_id"],
                "session_id": hit["session_id"],
                "turn": hit["turn"],
                "t_submit": None,
                "ttft_s": None,
                "prompt_tokens": None,
                "generated_tokens": None,
                "events_path": None,
                "source": "log_hit_only",
                "parse_error": None,
                "arrival_order": hit["session_id"],
                "stagger_index": hit["session_id"],
                "policy": parsed.policy if parsed else None,
                "offload_gib": parsed.offload_gib if parsed else None,
                "hints": parsed.hints if parsed else None,
                "is_revisited": is_revisited,
                "hit_tokens": hit["hit_tokens"],
                "gpu_hit_tokens": hit["gpu_hit_tokens"],
                "complete_recovery": hit["complete_recovery"],
                "log_path": hit["log_path"],
                "run_store_segments": parsed.store_segments if parsed else None,
                "run_load_segments": parsed.load_segments if parsed else None,
                "run_debug_offload_present": parsed.debug_offload_present if parsed else True,
            })

    transfer_rows: list[dict[str, Any]] = []
    for parsed in logs_by_tag.values():
        transfer_rows.extend(parsed.transfer_rows)

    run_rows: list[dict[str, Any]] = []
    for tag in sorted(logs_by_tag):
        parsed = logs_by_tag[tag]
        rows = [r for r in out_rows if r["run_tag"] == tag]
        hit_values = [r["hit_tokens"] for r in rows if r.get("hit_tokens") is not None]
        turn2_ttft = [
            r["ttft_s"] for r in rows
            if r.get("turn") == 2 and r.get("ttft_s") is not None
        ]
        all_ttft = [r["ttft_s"] for r in rows if r.get("ttft_s") is not None]
        complete_count = sum(1 for r in rows if r.get("complete_recovery") is True)
        run_rows.append({
            "run_tag": tag,
            "policy": parsed.policy,
            "offload_gib": parsed.offload_gib,
            "session_turn_rows": len(rows),
            "event_rows": sum(1 for r in rows if r.get("source") == "events"),
            "hit_request_rows": len(parsed.hit_by_request),
            "complete_recovery_count": complete_count,
            "hit_tokens_min": min(hit_values) if hit_values else None,
            "hit_tokens_p50": median(hit_values),
            "hit_tokens_max": max(hit_values) if hit_values else None,
            "ttft_s_p50": median(all_ttft),
            "turn2_ttft_s_p50": median(turn2_ttft),
            "store_segments": parsed.store_segments,
            "load_segments": parsed.load_segments,
            "debug_offload_present": parsed.debug_offload_present,
            "events_path": str(event_files_by_tag[tag]) if tag in event_files_by_tag else None,
            "log_path": str(parsed.log_path),
        })

    for tag, event_path in event_files_by_tag.items():
        if tag not in logs_by_tag:
            run_rows.append({
                "run_tag": tag,
                "policy": None,
                "offload_gib": None,
                "session_turn_rows": sum(1 for r in out_rows if r["run_tag"] == tag),
                "event_rows": sum(1 for r in out_rows if r["run_tag"] == tag),
                "hit_request_rows": 0,
                "complete_recovery_count": None,
                "hit_tokens_min": None,
                "hit_tokens_p50": None,
                "hit_tokens_max": None,
                "ttft_s_p50": median(r["ttft_s"] for r in out_rows if r["run_tag"] == tag),
                "turn2_ttft_s_p50": median(
                    r["ttft_s"] for r in out_rows
                    if r["run_tag"] == tag and r.get("turn") == 2
                ),
                "store_segments": None,
                "load_segments": None,
                "debug_offload_present": False,
                "events_path": str(event_path),
                "log_path": None,
            })

    return out_rows, run_rows, hit_rows, transfer_rows, warnings


def write_table(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def maybe_dataframe(rows: list[dict[str, Any]]) -> Any:
    if pd is None:
        return rows
    return pd.DataFrame(rows)


def print_summary(
    roots: list[Path],
    session_rows: list[dict[str, Any]],
    run_rows: list[dict[str, Any]],
    hit_rows: list[dict[str, Any]],
    transfer_rows: list[dict[str, Any]],
    warnings: list[str],
    outdir: Path,
) -> None:
    print("parsed log roots:")
    for root in roots:
        print(f"  - {root}")
    print()
    print(f"session_turn rows: {len(session_rows)}")
    print(f"run rows: {len(run_rows)}")
    print(f"hit-token request rows: {len(hit_rows)}")
    print(f"transfer segment rows: {len(transfer_rows)}")
    print()

    if pd is not None and run_rows:
        df = pd.DataFrame(run_rows)
        cols = [
            "run_tag", "policy", "session_turn_rows", "event_rows",
            "complete_recovery_count", "hit_tokens_p50", "hit_tokens_max",
            "turn2_ttft_s_p50", "store_segments", "load_segments",
            "debug_offload_present",
        ]
        print("per-run summary:")
        print(df[cols].to_string(index=False))
    elif run_rows:
        print("per-run summary:")
        header = (
            "run_tag", "policy", "rows", "events", "complete", "hit_p50",
            "hit_max", "t2_ttft_p50", "store_seg", "load_seg", "debug"
        )
        print(" ".join(f"{h:>14}" for h in header))
        for row in run_rows:
            values = (
                row["run_tag"], row["policy"], row["session_turn_rows"],
                row["event_rows"], row["complete_recovery_count"],
                fmt(row["hit_tokens_p50"]), fmt(row["hit_tokens_max"]),
                fmt(row["turn2_ttft_s_p50"]), fmt(row["store_segments"]),
                fmt(row["load_segments"]), row["debug_offload_present"],
            )
            print(" ".join(f"{str(v):>14}" for v in values))
    print()

    turn2 = [r["ttft_s"] for r in session_rows if r.get("turn") == 2 and r.get("ttft_s") is not None]
    if turn2:
        print(
            "second-turn TTFT summary: "
            f"n={len(turn2)} p50={fmt(median(turn2))}s "
            f"p90={fmt(percentile(turn2, 90))}s"
        )
    else:
        print("second-turn TTFT summary: no turn=2 rows found in visible event logs")

    complete_total = sum(1 for r in session_rows if r.get("complete_recovery") is True)
    print(f"complete recoveries (hit_tokens >= threshold): {complete_total}")

    missing_debug = [
        r["run_tag"] for r in run_rows
        if r.get("log_path") and not r.get("debug_offload_present")
    ]
    if missing_debug:
        print()
        print("DEBUG offload lines absent for these run logs; segment/hit counts are NA:")
        print("  " + ", ".join(missing_debug))

    if warnings:
        print()
        print("warnings:")
        for warning in sorted(set(warnings)):
            print(f"  - {warning}")

    print()
    print("wrote:")
    for name in ("session_turns.csv", "run_summary.csv", "hit_tokens.csv", "transfer_ops.csv"):
        print(f"  - {outdir / name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "log_roots",
        nargs="*",
        help="Directories containing *.events.jsonl and *.log files. Defaults to discovered ablogs* dirs.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory for parsed CSV outputs.",
    )
    parser.add_argument(
        "--complete-threshold",
        type=int,
        default=2000,
        help="Complete recovery threshold in offloaded hit-tokens.",
    )
    parser.add_argument(
        "--no-log-only-hits",
        action="store_true",
        help="Do not add rows for hit-token requests that lack JSONL event rows.",
    )
    args = parser.parse_args()

    roots = discover_roots(args.log_roots)
    if not roots:
        raise SystemExit("No ablogs* directories found. Pass one or more log directories explicitly.")

    session_rows, run_rows, hit_rows, transfer_rows, warnings = make_tables(
        roots=roots,
        complete_threshold=args.complete_threshold,
        include_log_only_hits=not args.no_log_only_hits,
    )

    outdir = args.outdir.resolve()
    write_table(session_rows, outdir / "session_turns.csv")
    write_table(run_rows, outdir / "run_summary.csv")
    write_table(hit_rows, outdir / "hit_tokens.csv")
    write_table(transfer_rows, outdir / "transfer_ops.csv")

    # Materialize the requested tidy dataframe when pandas is available.
    _session_turn_df = maybe_dataframe(session_rows)
    _ = _session_turn_df

    print_summary(
        roots=roots,
        session_rows=session_rows,
        run_rows=run_rows,
        hit_rows=hit_rows,
        transfer_rows=transfer_rows,
        warnings=warnings,
        outdir=outdir,
    )


if __name__ == "__main__":
    main()
