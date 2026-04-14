#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Compare real validation metrics across PAL-DRL, NPE-DRL, D3QN, BC, DAgger, and APF.
"""

import csv
from pathlib import Path


ALGORITHM_SPECS = [
    {
        "algorithm": "PAL-DRL",
        "metrics_path": Path("./Validate/PAL-DRL/metrics.csv"),
        "validate_cmd": "python validate_paldrl.py",
    },
    {
        "algorithm": "NPE-DRL",
        "metrics_path": Path("./Validate/NPE-DRL/metrics.csv"),
        "validate_cmd": "python validate_npedrl.py",
    },
    {
        "algorithm": "D3QN",
        "metrics_path": Path("./Validate/D3QN/metrics.csv"),
        "validate_cmd": "python validate_d3qn.py",
    },
    {
        "algorithm": "BC",
        "metrics_path": Path("./Validate/BC/metrics.csv"),
        "validate_cmd": "python validate_bc.py",
    },
    {
        "algorithm": "DAgger",
        "metrics_path": Path("./Validate/DAgger/metrics.csv"),
        "validate_cmd": "python validate_dagger.py",
    },
    {
        "algorithm": "APF",
        "metrics_path": Path("./Validate/APF/metrics.csv"),
        "validate_cmd": "python validate_apf.py",
    },
]

RESULT_COLUMNS = [
    "algorithm",
    "success_rate_pct",
    "collision_rate_pct",
    "timeout_rate_pct",
    "average_step",
    "average_trajectory_length_m",
    "average_energy_consumption_j",
    "average_posture_stability_rad",
    "average_execution_time_s",
    "max_episode",
    "max_step_per_episode",
]

ROUND_COLUMNS = {
    "success_rate_pct": 2,
    "collision_rate_pct": 2,
    "timeout_rate_pct": 2,
    "average_step": 2,
    "average_trajectory_length_m": 4,
    "average_energy_consumption_j": 4,
    "average_posture_stability_rad": 4,
    "average_execution_time_s": 4,
}

INT_COLUMNS = {"max_episode", "max_step_per_episode"}


def load_metrics(spec):
    metrics_path = spec["metrics_path"]
    if not metrics_path.exists():
        return None, f"missing file: {metrics_path}"

    with metrics_path.open("r", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        fieldnames = reader.fieldnames or []
        missing_columns = [column for column in RESULT_COLUMNS if column not in fieldnames]
        rows = list(reader)

    if not rows:
        return None, f"empty file: {metrics_path}"
    if missing_columns:
        return None, f"missing columns {missing_columns} in {metrics_path}"

    raw_row = rows[0]
    row = {"algorithm": spec["algorithm"]}
    for column in RESULT_COLUMNS[1:]:
        try:
            if column in INT_COLUMNS:
                row[column] = int(float(raw_row[column]))
            else:
                row[column] = float(raw_row[column])
        except (TypeError, ValueError):
            return None, f"invalid numeric value for `{column}` in {metrics_path}"

    return row, None


def format_results(rows):
    formatted = []
    for row in rows:
        formatted_row = row.copy()
        for column, decimals in ROUND_COLUMNS.items():
            formatted_row[column] = round(formatted_row[column], decimals)
        formatted.append(formatted_row)
    return formatted


def print_table(rows):
    string_rows = []
    for row in rows:
        string_row = {}
        for column in RESULT_COLUMNS:
            value = row[column]
            if column in ROUND_COLUMNS:
                string_row[column] = f"{value:.{ROUND_COLUMNS[column]}f}"
            else:
                string_row[column] = str(value)
        string_rows.append(string_row)

    column_widths = {
        column: max(len(column), max(len(row[column]) for row in string_rows)) for column in RESULT_COLUMNS
    }

    header = "  ".join(column.ljust(column_widths[column]) for column in RESULT_COLUMNS)
    separator = "  ".join("-" * column_widths[column] for column in RESULT_COLUMNS)
    print(header)
    print(separator)
    for row in string_rows:
        print("  ".join(row[column].ljust(column_widths[column]) for column in RESULT_COLUMNS))


def write_comparison_csv(rows, output_path):
    with output_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=RESULT_COLUMNS)
        writer.writeheader()
        for row in rows:
            csv_row = {}
            for column in RESULT_COLUMNS:
                value = row[column]
                if column in ROUND_COLUMNS:
                    csv_row[column] = f"{value:.{ROUND_COLUMNS[column]}f}"
                else:
                    csv_row[column] = value
            writer.writerow(csv_row)


def is_algorithm_available(spec, rows):
    return any(spec["algorithm"] == row["algorithm"] for row in rows)


def sort_results(rows):
    order = {spec["algorithm"]: index for index, spec in enumerate(ALGORITHM_SPECS)}
    return sorted(rows, key=lambda row: order[row["algorithm"]])


def main():
    print("=" * 100)
    print("Algorithm Comparison Based on Validation Metrics")
    print("=" * 100)
    print("Metrics:")
    print("  success_rate_pct, collision_rate_pct, timeout_rate_pct")
    print("  average_step, average_trajectory_length_m, average_energy_consumption_j")
    print("  average_posture_stability_rad, average_execution_time_s")

    available_rows = []
    missing_specs = []
    invalid_specs = []

    for spec in ALGORITHM_SPECS:
        row, error = load_metrics(spec)
        if row is not None:
            available_rows.append(row)
        else:
            if error.startswith("missing file:"):
                missing_specs.append(spec)
            else:
                invalid_specs.append((spec, error))

    print("\nResult overview:")
    for spec in ALGORITHM_SPECS:
        if is_algorithm_available(spec, available_rows):
            print(f"  [FOUND]   {spec['algorithm']}: {spec['metrics_path']}")
        else:
            print(f"  [MISSING] {spec['algorithm']}: run `{spec['validate_cmd']}`")

    if invalid_specs:
        print("\nInvalid metrics files:")
        for spec, error in invalid_specs:
            print(f"  {spec['algorithm']}: {error}")

    if not available_rows:
        print("\nNo metrics.csv files were found. Run the validation scripts first.")
        return

    result_rows = sort_results(available_rows)
    result_rows = format_results(result_rows)

    output_path = Path("./Validate/algorithm_comparison.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_comparison_csv(result_rows, output_path)

    print("\nComparison table:")
    print_table(result_rows)
    print(f"\nComparison CSV saved to: {output_path}")

    if missing_specs:
        print("\nMissing results can be generated with:")
        for spec in missing_specs:
            print(f"  {spec['algorithm']}: {spec['validate_cmd']}")


if __name__ == "__main__":
    main()
