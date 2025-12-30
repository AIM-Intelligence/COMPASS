#!/usr/bin/env python3
"""Simulate Policy Compliance Filter outcomes on existing judge results.

This script reads policy compliance filter outputs (JSONL) and previously
produced judge results (JSONL). It re-computes the final `is_correct` values
according to the filter decision rules described in the project README,
without re-running response generation or judging.

Usage (from repository root)::

    python scripts/filtered_judge_results.py \
        --filter-dir results/filter_results \
        --judge-dir results/judge_results \
        --output-dir results/filtered_judge_results

You can also target specific files with `--filter-paths` and `--judge-paths`.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


@dataclass(frozen=True)
class FilterRecord:
    """Captures the subset of filter metadata we need for aggregation."""

    decision: str
    category: Optional[str]
    source_file: Path
    company: Optional[str]
    query_type: Optional[str]
    model: Optional[str]


def collect_paths(default_dir: Path, explicit_paths: Optional[List[str]], pattern: str) -> List[Path]:
    """Resolve a list of file paths either from explicit inputs or a glob."""

    if explicit_paths:
        resolved: List[Path] = []
        for raw in explicit_paths:
            path = Path(raw)
            if not path.exists():
                raise FileNotFoundError(f"Specified path does not exist: {raw}")
            if path.is_dir():
                resolved.extend(sorted(path.glob(pattern)))
            else:
                resolved.append(path)
        return resolved

    if not default_dir.exists():
        raise FileNotFoundError(f"Directory does not exist: {default_dir}")
    return sorted(p for p in default_dir.glob(pattern) if p.is_file())


def resolve_duplicate_record(
    existing: FilterRecord, candidate: FilterRecord, strategy: str
) -> FilterRecord:
    """Resolve duplicate filter results for the same id according to strategy."""

    if strategy == "error":
        raise RuntimeError(
            f"Duplicate filter id detected in {candidate.source_file}; also seen in {existing.source_file}"
        )

    if strategy == "first":
        return existing

    if strategy == "last":
        return candidate

    if strategy == "deny_over_allow":
        if existing.decision == "DENY" and candidate.decision != "DENY":
            return existing
        if candidate.decision == "DENY" and existing.decision != "DENY":
            return candidate
        return existing

    if strategy == "allow_over_deny":
        if existing.decision == "ALLOW" and candidate.decision != "ALLOW":
            return existing
        if candidate.decision == "ALLOW" and existing.decision != "ALLOW":
            return candidate
        return existing

    raise ValueError(f"Unknown duplicate strategy: {strategy}")


def load_filter_records(
    paths: Iterable[Path], duplicate_strategy: str
) -> Tuple[Dict[str, Dict[str, FilterRecord]], Counter]:
    """Load filter JSONL files grouped by request id and filter model."""

    details: Dict[str, Dict[str, FilterRecord]] = defaultdict(dict)
    duplicate_stats: Counter = Counter()

    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError as exc:  # pragma: no cover - defensive
                    raise RuntimeError(
                        f"Failed to parse JSON in {path} at line {line_number}: {exc}"
                    ) from exc

                record_id = payload.get("id")
                if not record_id:
                    raise RuntimeError(f"Missing 'id' field in {path} at line {line_number}")

                decision = payload.get("decision")
                if decision is None:
                    raise RuntimeError(
                        f"Missing 'decision' for id '{record_id}' in {path} line {line_number}"
                    )

                category = payload.get("category")
                normalized_decision = str(decision).upper()
                normalized_category = str(category).lower() if category is not None else None

                candidate = FilterRecord(
                    decision=normalized_decision,
                    category=normalized_category,
                    source_file=path,
                    company=payload.get("company"),
                    query_type=payload.get("query_type"),
                    model=payload.get("model"),
                )

                model_key = candidate.model or f"source:{path}"
                per_id_records = details[record_id]
                existing = per_id_records.get(model_key)
                if existing is None:
                    per_id_records[model_key] = candidate
                    continue

                resolved = resolve_duplicate_record(existing, candidate, duplicate_strategy)
                if resolved is existing:
                    duplicate_stats["ignored"] += 1
                else:
                    duplicate_stats["replaced"] += 1
                per_id_records[model_key] = resolved

    if duplicate_stats:
        print(
            "Resolved duplicate filter ids:",
            {k: duplicate_stats[k] for k in sorted(duplicate_stats)},
            file=sys.stderr,
        )

    return details, duplicate_stats


def apply_filter_logic(
    filter_record: FilterRecord, judge_is_correct: Optional[bool]
) -> Tuple[Optional[bool], str]:
    """Return the simulated final `is_correct` and a short action label."""

    decision = filter_record.decision
    category = filter_record.category

    if decision == "ALLOW":
        return judge_is_correct, "allow_passthrough"

    if decision == "DENY":
        if category == "allowlist":
            return False, "deny_allowlist_refusal"
        if category == "denylist":
            return True, "deny_denylist_block"
        raise RuntimeError(
            f"Unsupported filter category '{category}' for DENY decision (source={filter_record.source_file})"
        )

    raise RuntimeError(
        f"Unsupported filter decision '{decision}' (source={filter_record.source_file})"
    )


def derive_output_path(output_dir: Path, judge_file: Path) -> Path:
    """Create a stable output filename for a given judge file."""

    name = judge_file.name
    if name.startswith("judged_"):
        name = "filtered_" + name[len("judged_"):]
    else:
        name = f"filtered_{name}"
    return output_dir / name


def process_judge_file(
    judge_file: Path,
    filter_details: Dict[str, Dict[str, FilterRecord]],
    output_dir: Path,
    strict_missing: bool,
) -> Counter:
    """Apply filter logic to one judge file and write the simulated results."""

    stats: Counter = Counter()
    output_path = derive_output_path(output_dir, judge_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with judge_file.open("r", encoding="utf-8") as src, output_path.open(
        "w", encoding="utf-8"
    ) as dst:
        for line_number, line in enumerate(src, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:  # pragma: no cover - defensive
                raise RuntimeError(
                    f"Failed to parse JSON in {judge_file} at line {line_number}: {exc}"
                ) from exc

            record_id = payload.get("id")
            if record_id is None:
                raise RuntimeError(
                    f"Missing 'id' in judge file {judge_file} at line {line_number}"
                )

            original_is_correct = payload.get("is_correct")
            new_payload = dict(payload)
            new_payload.pop("is_correct", None)
            new_payload["judge_is_correct"] = original_is_correct

            per_model_records = filter_details.get(record_id)

            if not per_model_records:
                stats["missing_filter"] += 1
                if strict_missing:
                    raise KeyError(
                        f"No filter record found for id '{record_id}' required by {judge_file}"
                    )
                new_payload["filter_results"] = []
                new_payload["filter_models"] = []
                stats["records"] += 1
                dst.write(json.dumps(new_payload) + "\n")
                continue

            per_filter_results = []
            filter_models: List[Optional[str]] = []
            for rec in per_model_records.values():
                per_is_correct, per_action = apply_filter_logic(rec, original_is_correct)
                per_filter_results.append(
                    {
                        "decision": rec.decision,
                        "category": rec.category,
                        "source": str(rec.source_file),
                        "model": rec.model,
                        "policy_action": per_action,
                        "result_is_correct": per_is_correct,
                    }
                )
                if rec.model is not None:
                    filter_models.append(rec.model)
                stats["filter_records"] += 1
                stats[per_action] += 1

            new_payload["filter_results"] = per_filter_results
            new_payload["filter_models"] = filter_models

            stats["records"] += 1
            dst.write(json.dumps(new_payload) + "\n")

    return stats


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Combine policy filter outputs with judge results to simulate the"
            " end-to-end scenario without re-running generation."
        )
    )
    parser.add_argument(
        "--filter-dir",
        type=Path,
        default=Path("results/filter_results"),
        help="Directory containing filter JSONL files (default: results/filter_results).",
    )
    parser.add_argument(
        "--filter-paths",
        nargs="*",
        default=None,
        help="Optional explicit filter JSONL files or directories to process.",
    )
    parser.add_argument(
        "--filter-pattern",
        default="*.jsonl",
        help="Glob pattern when scanning filter directories (default: *.jsonl).",
    )
    parser.add_argument(
        "--duplicate-strategy",
        choices=["error", "first", "last", "deny_over_allow", "allow_over_deny"],
        default="deny_over_allow",
        help=(
            "How to resolve duplicate filter ids across files. "
            "Defaults to 'deny_over_allow', which prefers DENY decisions when available."
        ),
    )
    parser.add_argument(
        "--judge-dir",
        type=Path,
        default=Path("results/judge_results"),
        help="Directory containing judge JSONL files (default: results/judge_results).",
    )
    parser.add_argument(
        "--judge-paths",
        nargs="*",
        default=None,
        help="Optional explicit judge JSONL files or directories to process.",
    )
    parser.add_argument(
        "--judge-pattern",
        default="*.jsonl",
        help="Glob pattern when scanning judge directories (default: *.jsonl).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/filtered_judge_results"),
        help="Directory to write the simulated judge results (default: results/filtered_judge_results).",
    )
    parser.add_argument(
        "--strict-missing-filter",
        action="store_true",
        help="Fail if a judge record lacks a corresponding filter result.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    filter_paths = collect_paths(args.filter_dir, args.filter_paths, args.filter_pattern)
    if not filter_paths:
        raise RuntimeError(
            "No filter JSONL files found. Adjust --filter-dir, --filter-pattern, or --filter-paths."
        )

    judge_paths = collect_paths(args.judge_dir, args.judge_paths, args.judge_pattern)
    judge_paths = [p for p in judge_paths if p.suffix == ".jsonl"]
    if not judge_paths:
        raise RuntimeError(
            "No judge JSONL files found. Adjust --judge-dir, --judge-pattern, or --judge-paths."
        )

    filter_details, _ = load_filter_records(filter_paths, args.duplicate_strategy)

    overall_stats: Counter = Counter()
    for judge_file in judge_paths:
        stats = process_judge_file(
            judge_file=judge_file,
            filter_details=filter_details,
            output_dir=args.output_dir,
            strict_missing=args.strict_missing_filter,
        )
        overall_stats.update(stats)
        print(
            f"Processed {judge_file} → {derive_output_path(args.output_dir, judge_file)}"
            f" | records={stats.get('records', 0)}"
            f" filters={stats.get('filter_records', 0)}"
            f" allow={stats.get('allow_passthrough', 0)}"
            f" deny_allowlist={stats.get('deny_allowlist_refusal', 0)}"
            f" deny_denylist={stats.get('deny_denylist_block', 0)}"
            f" missing_filter={stats.get('missing_filter', 0)}"
        )

    print(
        "Done. Total records=", overall_stats.get("records", 0),
        "filter_results=", overall_stats.get("filter_records", 0),
        "allow_passthrough=", overall_stats.get("allow_passthrough", 0),
        "deny_allowlist_refusal=", overall_stats.get("deny_allowlist_refusal", 0),
        "deny_denylist_block=", overall_stats.get("deny_denylist_block", 0),
        "missing_filter=", overall_stats.get("missing_filter", 0),
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    sys.exit(main())
