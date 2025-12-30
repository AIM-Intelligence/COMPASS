#!/usr/bin/env python3
"""Aggregate judged JSONL results into model/company accuracy summary."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

# Maintains the table column order expected by downstream consumers
COMPANY_ORDER = [
    "AutoViaMotors",
    "CityGov",
    "FinSecure",
    "MediCarePlus",
    "PlanMyTrip",
    "TelePath",
    "TutoraVerse",
    "VirtuRecruit",
]

QUERY_CATEGORY_ORDER = [
    "Allowed Base Query",
    "Allowed Edge Query",
    "Denied Base Query",
    "Denied Edge Query",
]


TARGET_DIRECTORY_GLOBS = {
    "judge_results": "results/judge_results/judged_*.jsonl",
    "payloads_judge_results": "results/payloads_judge_results/judged_*.jsonl",
    "robust_judge_results": "results/robust_judge_results/judged_*.jsonl",
    "rag_judge_results": "results/rag_judge_results/judged_*.jsonl"
}

TARGET_DIRECTORY_OUTPUTS = {
    "judge_results": "results/judge_results/judged_accuracy_summary.csv",
    "payloads_judge_results": "results/payloads_judge_results/judged_accuracy_summary.csv",
    "robust_judge_results": "results/robust_judge_results/judged_accuracy_summary.csv",
    "rag_judge_results": "results/rag_judge_results/judged_accuracy_summary.csv"
}


def extract_model_company_query_type(path: Path) -> tuple[str, str, str]:
    """Derive metadata from the judged file name."""
    stem = path.stem  # remove .jsonl
    if not stem.startswith("judged_"):
        raise ValueError(f"Unexpected file name format: {path}")
    remainder = stem[len("judged_"):]
    try:
        prefix, _timestamp = remainder.rsplit("_", 1)
    except ValueError as exc:
        raise ValueError(f"Cannot split timestamp from {path}") from exc
    parts = prefix.split("_")
    
    # Try to align with known companies to handle query types with underscores
    company_index = -1
    for i, part in enumerate(parts):
        if part in COMPANY_ORDER:
            company_index = i
            break
            
    if company_index != -1:
        query_type = "_".join(parts[:company_index])
        company = parts[company_index]
        model = "_".join(parts[company_index + 1 :])
    else:
        # Fallback to simple split if company not found (legacy behavior)
        query_type, company, model = prefix.split("_", 2)

    return model, company, query_type


def categorize_query(row: dict) -> str:
    """Map a judged row to the scoreboard query category label."""
    query_type = row.get("query_type")
    row_id = row.get("id", "")
    if query_type == "base":
        if "-allowlist-" in row_id:
            return "Allowed Base Query"
        if "-denylist-" in row_id:
            return "Denied Base Query"
    elif query_type == "allowed_edge":
        return "Allowed Edge Query"
    elif query_type == "denied_edge":
        return "Denied Edge Query"
    return "Unknown"


def prettify_model_name(model: str) -> str:
    """Generate a human-friendly model label."""
    return model.replace("_", " ").replace("-", " ").title()


def load_stats(paths: list[Path]) -> dict:
    """Accumulate correctness tallies keyed by model/query category/company."""
    stats = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: [0, 0])))
    for path in paths:
        model, _company_from_name, _query_type_from_name = extract_model_company_query_type(path)
        with path.open() as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                category = categorize_query(row)
                if category == "Unknown":
                    continue
                company = row["company"]
                if not isinstance(row.get("is_correct"), bool):
                    print(f"WARN: {row['id']} is_correct is not a bool")
                    print(f" {row.get('is_correct')}")
                    print(f"FILE: {path}")
                    break
                    
                correct_flag = bool(row.get("is_correct"))
                bucket = stats[model][category][company]
                bucket[1] += 1
                if correct_flag:
                    bucket[0] += 1
    return stats


def format_percent(correct: int, total: int) -> str:
    if total == 0:
        return "N/A"
    return f"{(correct / total) * 100:.2f}%"


def compute_average(values: list[str]) -> str:
    numeric = []
    for value in values:
        if value.endswith("%"):
            numeric.append(float(value[:-1]))
    if not numeric:
        return "N/A"
    return f"{mean(numeric):.2f}%"


def build_rows(stats: dict, pretty_names: bool) -> list[dict[str, str]]:
    rows = []
    models = sorted(stats.keys(), key=lambda name: prettify_model_name(name) if pretty_names else name)
    for model in models:
        print(f"Model: {model}")
        model_label = prettify_model_name(model) if pretty_names else model
        for category in QUERY_CATEGORY_ORDER:
            row = {"Model": model_label, "Query Type": category}
            percentages = []
            for company in COMPANY_ORDER:
                correct, total = stats[model][category][company]
                value = format_percent(correct, total)
                row[company] = value
                percentages.append(value)
            row["Average"] = compute_average(percentages)
            rows.append(row)
    return rows


def write_csv(rows: list[dict[str, str]], output_path: Path) -> None:
    fieldnames = ["Model", "Query Type", *COMPANY_ORDER, "Average"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize judged JSONL accuracy scores.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--input-glob", default=None, help="Glob pattern for judged result files")
    group.add_argument(
        "--target-directory",
        choices=tuple(TARGET_DIRECTORY_GLOBS.keys()),
        help="Name of the results subdirectory to analyze (e.g. judge_results)",
    )
    parser.add_argument("--output", default=None, help="Path to write the summary CSV")
    parser.add_argument("--pretty-model-names", action="store_true", help="Beautify model identifiers in the output")
    args = parser.parse_args()

    default_directory = "judge_results"

    if args.target_directory:
        glob_patterns = [TARGET_DIRECTORY_GLOBS[args.target_directory]]
        output_location = args.output or TARGET_DIRECTORY_OUTPUTS[args.target_directory]
    else:
        glob_patterns = [args.input_glob or TARGET_DIRECTORY_GLOBS[default_directory]]
        output_location = args.output or TARGET_DIRECTORY_OUTPUTS[default_directory]

    matched_paths = set()
    for pattern in glob_patterns:
        matched_paths.update(Path().glob(pattern))

    paths = sorted(matched_paths)
    if not paths:
        patterns = ", ".join(glob_patterns)
        raise SystemExit(f"No files match pattern(s): {patterns}")

    stats = load_stats(paths)
    rows = build_rows(stats, pretty_names=args.pretty_model_names)
    write_csv(rows, Path(output_location))
    print(f"Saved results to {output_location}")


if __name__ == "__main__":
    main()
