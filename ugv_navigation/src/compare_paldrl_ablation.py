#!/usr/bin/python3
# -*- coding: utf-8 -*-
import csv
from pathlib import Path

import numpy as np

import paldrl_ablation


VARIANT_ORDER = ["full", "wo_dropout", "wo_local_teacher", "wo_posture_reward", "wo_exploration"]
SEED_ORDER = [4, 8, 12]
METRIC_COLUMNS = [
    "success_rate_pct",
    "collision_rate_pct",
    "timeout_rate_pct",
    "average_step",
    "average_trajectory_length_m",
    "average_energy_consumption_j",
    "average_posture_stability_rad",
    "average_execution_time_s",
]
SUMMARY_COLUMNS = [
    "method",
    "sr_mean",
    "sr_std",
    "cr_mean",
    "cr_std",
    "tr_mean",
    "tr_std",
    "as_mean",
    "as_std",
    "atl_mean",
    "atl_std",
    "aec_mean",
    "aec_std",
    "aps_mean",
    "aps_std",
    "aet_mean",
    "aet_std",
]


def load_metrics(path):
    with path.open("r", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        rows = list(reader)
    if not rows:
        raise ValueError(f"Empty metrics file: {path}")
    return rows[0]


def summarize_variant(variant):
    metrics_rows = []
    missing = []
    for seed in SEED_ORDER:
        metrics_path = Path("./Validate/PAL-DRL-Ablation") / variant / f"seed_{seed}" / "metrics.csv"
        if not metrics_path.exists():
            missing.append((variant, seed, metrics_path))
            continue
        metrics_rows.append(load_metrics(metrics_path))
    return metrics_rows, missing


def format_mean_std(mean, std):
    return f"{mean:.2f} ± {std:.2f}"


def write_summary_csv(rows, output_path):
    with output_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def write_summary_markdown(rows, output_path):
    header = "| Method | SR (%) | CR (%) | TR (%) | AS | ATL (m) | AEC (J) | APS (rad) | AET (s) |\n"
    separator = "| --- | --- | --- | --- | --- | --- | --- | --- | --- |\n"
    lines = [header, separator]
    for row in rows:
        lines.append(
            "| {method} | {sr} | {cr} | {tr} | {as_} | {atl} | {aec} | {aps} | {aet} |\n".format(
                method=row["method"],
                sr=format_mean_std(row["sr_mean"], row["sr_std"]),
                cr=format_mean_std(row["cr_mean"], row["cr_std"]),
                tr=format_mean_std(row["tr_mean"], row["tr_std"]),
                as_=format_mean_std(row["as_mean"], row["as_std"]),
                atl=format_mean_std(row["atl_mean"], row["atl_std"]),
                aec=format_mean_std(row["aec_mean"], row["aec_std"]),
                aps=format_mean_std(row["aps_mean"], row["aps_std"]),
                aet=format_mean_std(row["aet_mean"], row["aet_std"]),
            )
        )
    with output_path.open("w") as md_file:
        md_file.writelines(lines)


def print_table(rows):
    print("Method                          SR (%)         CR (%)         TR (%)         AS             ATL (m)        AEC (J)        APS (rad)      AET (s)")
    print("------------------------------  -------------  -------------  -------------  -------------  -------------  -------------  -------------  -------------")
    for row in rows:
        print(
            f"{row['method']:<30}  "
            f"{format_mean_std(row['sr_mean'], row['sr_std']):<13}  "
            f"{format_mean_std(row['cr_mean'], row['cr_std']):<13}  "
            f"{format_mean_std(row['tr_mean'], row['tr_std']):<13}  "
            f"{format_mean_std(row['as_mean'], row['as_std']):<13}  "
            f"{format_mean_std(row['atl_mean'], row['atl_std']):<13}  "
            f"{format_mean_std(row['aec_mean'], row['aec_std']):<13}  "
            f"{format_mean_std(row['aps_mean'], row['aps_std']):<13}  "
            f"{format_mean_std(row['aet_mean'], row['aet_std']):<13}"
        )


def main():
    print("=" * 100)
    print("PAL-DRL Ablation Comparison")
    print("=" * 100)

    missing_items = []
    summary_rows = []

    for variant in VARIANT_ORDER:
        variant_rows, missing = summarize_variant(variant)
        missing_items.extend(missing)
        if not variant_rows:
            continue

        display_name = paldrl_ablation.ABLATION_VARIANTS[variant]["display_name"]
        metric_values = {}
        for column in METRIC_COLUMNS:
            metric_values[column] = np.array([float(row[column]) for row in variant_rows], dtype=np.float32)

        summary_rows.append(
            {
                "method": display_name,
                "sr_mean": float(np.mean(metric_values["success_rate_pct"])),
                "sr_std": float(np.std(metric_values["success_rate_pct"])),
                "cr_mean": float(np.mean(metric_values["collision_rate_pct"])),
                "cr_std": float(np.std(metric_values["collision_rate_pct"])),
                "tr_mean": float(np.mean(metric_values["timeout_rate_pct"])),
                "tr_std": float(np.std(metric_values["timeout_rate_pct"])),
                "as_mean": float(np.mean(metric_values["average_step"])),
                "as_std": float(np.std(metric_values["average_step"])),
                "atl_mean": float(np.mean(metric_values["average_trajectory_length_m"])),
                "atl_std": float(np.std(metric_values["average_trajectory_length_m"])),
                "aec_mean": float(np.mean(metric_values["average_energy_consumption_j"])),
                "aec_std": float(np.std(metric_values["average_energy_consumption_j"])),
                "aps_mean": float(np.mean(metric_values["average_posture_stability_rad"])),
                "aps_std": float(np.std(metric_values["average_posture_stability_rad"])),
                "aet_mean": float(np.mean(metric_values["average_execution_time_s"])),
                "aet_std": float(np.std(metric_values["average_execution_time_s"])),
            }
        )

    if missing_items:
        print("\nMissing results:")
        for variant, seed, metrics_path in missing_items:
            print(f"  {variant} seed {seed}: missing {metrics_path}")
            print(f"    python train_paldrl_ablation.py --variant {variant} --seed {seed}")
            print(f"    python validate_paldrl_ablation.py --variant {variant} --seed {seed}")

    if not summary_rows:
        print("\nNo ablation metrics.csv files were found. Run the ablation training/validation scripts first.")
        return

    output_dir = Path("./Validate/PAL-DRL-Ablation")
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "ablation_comparison.csv"
    md_path = output_dir / "ablation_comparison.md"

    write_summary_csv(summary_rows, csv_path)
    write_summary_markdown(summary_rows, md_path)

    print("\nAblation comparison table:")
    print_table(summary_rows)
    print(f"\nCSV saved to: {csv_path}")
    print(f"Markdown saved to: {md_path}")


if __name__ == "__main__":
    main()
