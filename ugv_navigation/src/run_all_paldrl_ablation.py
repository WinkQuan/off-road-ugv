#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""Run the full PAL-DRL ablation experiment matrix."""

from __future__ import print_function

import argparse
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_VARIANTS = ["full", "wo_dropout", "wo_local_teacher", "wo_posture_reward", "wo_exploration"]
DEFAULT_SEEDS = [4, 8, 12]
REQUIRED_GAZEBO_TOPICS = ["/gazebo/model_states", "/mycar/camera_right/image_raw_right"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run PAL-DRL ablation training, validation, and comparison in one command."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned commands and skip/run decisions without running anything.",
    )
    parser.add_argument(
        "--force-train",
        action="store_true",
        help="Retrain even when the final ONNX model already exists. Existing model directories are backed up first.",
    )
    parser.add_argument(
        "--force-validate",
        action="store_true",
        help="Revalidate even when metrics.csv already exists. Existing validation directories are backed up first.",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=DEFAULT_VARIANTS,
        default=DEFAULT_VARIANTS,
        help="Ablation variants to run. Defaults to the full ablation matrix.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=DEFAULT_SEEDS,
        help="Random seeds to run. Defaults to: 4 8 12.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue with later jobs after a train/validate command fails.",
    )
    return parser.parse_args()


def model_dir_for(variant, seed):
    return SCRIPT_DIR / "Model" / "PAL-DRL-Ablation" / variant / f"seed_{seed}"


def validate_dir_for(variant, seed):
    return SCRIPT_DIR / "Validate" / "PAL-DRL-Ablation" / variant / f"seed_{seed}"


def model_file_for(variant, seed):
    return model_dir_for(variant, seed) / "model_pal_drl_ablation.onnx"


def metrics_file_for(variant, seed):
    return validate_dir_for(variant, seed) / "metrics.csv"


def display_command(command):
    if command and Path(command[0]).name.startswith("python"):
        return " ".join(["python"] + command[1:])
    return " ".join(command)


def print_header(title):
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


def check_ros_gazebo_ready():
    try:
        completed = subprocess.run(
            ["rostopic", "list"],
            cwd=str(SCRIPT_DIR),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=10,
        )
    except FileNotFoundError as err:
        raise RuntimeError(
            "Cannot find `rostopic`. Please source ROS and the catkin workspace before running this script."
        ) from err
    except subprocess.TimeoutExpired as err:
        raise RuntimeError(
            "Timed out while querying ROS topics. Please confirm ROS/Gazebo is running."
        ) from err

    if completed.returncode != 0:
        stderr = completed.stderr.strip()
        detail = f"\nrostopic error: {stderr}" if stderr else ""
        raise RuntimeError(
            "ROS master is not reachable. Please start Gazebo first:\n"
            "  roslaunch mymodel_gazebo myworld.launch"
            f"{detail}"
        )

    available_topics = set(completed.stdout.splitlines())
    missing_topics = [topic for topic in REQUIRED_GAZEBO_TOPICS if topic not in available_topics]
    if missing_topics:
        raise RuntimeError(
            "Gazebo is reachable, but required topics are missing:\n"
            f"  {', '.join(missing_topics)}\n"
            "Please start the main world first:\n"
            "  roslaunch mymodel_gazebo myworld.launch"
        )

    print("ROS/Gazebo preflight passed.")


def backup_directory(path, log_dir, relative_root):
    if not path.exists():
        return None

    relative_path = path.relative_to(SCRIPT_DIR)
    backup_path = log_dir / "backups" / relative_root / relative_path
    backup_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(path), str(backup_path))
    return backup_path


def stream_command(command, log_path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"$ {display_command(command)}")
    print(f"Log: {log_path}")

    with log_path.open("w") as log_file:
        log_file.write(f"$ {display_command(command)}\n")
        log_file.flush()
        process = subprocess.Popen(
            command,
            cwd=str(SCRIPT_DIR),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log_file.write(line)
        return process.wait()


def run_stage(stage, command, log_path, continue_on_error):
    return_code = stream_command(command, log_path)
    if return_code == 0:
        print(f"{stage} finished successfully.")
        return True

    print(f"{stage} failed with exit code {return_code}.")
    if not continue_on_error:
        raise RuntimeError(f"{stage} failed. See log: {log_path}")
    return False


def maybe_backup_for_force_train(variant, seed, log_dir, dry_run):
    path = model_dir_for(variant, seed)
    if dry_run:
        print(f"[DRY-RUN] Would back up model directory before retraining: {path}")
        return

    backup_path = backup_directory(path, log_dir, "force-train")
    if backup_path:
        print(f"Backed up existing model directory to: {backup_path}")


def maybe_backup_for_force_validate(variant, seed, log_dir, dry_run):
    path = validate_dir_for(variant, seed)
    if dry_run:
        print(f"[DRY-RUN] Would back up validation directory before revalidating: {path}")
        return

    backup_path = backup_directory(path, log_dir, "force-validate")
    if backup_path:
        print(f"Backed up existing validation directory to: {backup_path}")


def command_for(script_name, variant, seed):
    return [sys.executable, script_name, "--variant", variant, "--seed", str(seed)]


def compare_command():
    return [sys.executable, "compare_paldrl_ablation.py"]


def planned_jobs(variants, seeds):
    for variant in variants:
        for seed in seeds:
            yield variant, seed


def all_default_metrics_exist():
    return all(metrics_file_for(variant, seed).exists() for variant in DEFAULT_VARIANTS for seed in DEFAULT_SEEDS)


def print_plan(args, log_dir):
    print_header("PAL-DRL Ablation Run Plan")
    print(f"Working directory: {SCRIPT_DIR}")
    print(f"Log directory: {log_dir}")
    print(f"Variants: {' '.join(args.variants)}")
    print(f"Seeds: {' '.join(str(seed) for seed in args.seeds)}")
    print(f"force_train={args.force_train}, force_validate={args.force_validate}")
    if args.dry_run:
        print("Dry run: ROS/Gazebo preflight and experiment commands will not be executed.")


def run_matrix(args, log_dir):
    failures = []

    for variant, seed in planned_jobs(args.variants, args.seeds):
        print_header(f"Variant={variant} Seed={seed}")

        train_log = log_dir / f"train_{variant}_seed_{seed}.log"
        validate_log = log_dir / f"validate_{variant}_seed_{seed}.log"
        model_file = model_file_for(variant, seed)
        metrics_file = metrics_file_for(variant, seed)

        should_train = args.force_train or not model_file.exists()
        should_validate = args.force_validate or should_train or not metrics_file.exists()
        train_failed = False

        if should_train:
            if args.force_train and model_dir_for(variant, seed).exists():
                maybe_backup_for_force_train(variant, seed, log_dir, args.dry_run)
            train_cmd = command_for("train_paldrl_ablation.py", variant, seed)
            if args.dry_run:
                print(f"[DRY-RUN] Would run training: {display_command(train_cmd)}")
            else:
                try:
                    ok = run_stage(f"Training {variant} seed {seed}", train_cmd, train_log, args.continue_on_error)
                    if not ok:
                        failures.append((variant, seed, "train"))
                        train_failed = True
                except RuntimeError:
                    failures.append((variant, seed, "train"))
                    raise
        else:
            print(f"Skip training: final ONNX model already exists: {model_file}")

        if train_failed:
            print(f"Skip validation: training failed for variant={variant}, seed={seed}.")
            continue

        if should_validate:
            stale_validation_dir = should_train and metrics_file.exists()
            if (args.force_validate or stale_validation_dir) and validate_dir_for(variant, seed).exists():
                maybe_backup_for_force_validate(variant, seed, log_dir, args.dry_run)
            validate_cmd = command_for("validate_paldrl_ablation.py", variant, seed)
            if args.dry_run:
                print(f"[DRY-RUN] Would run validation: {display_command(validate_cmd)}")
            else:
                try:
                    ok = run_stage(
                        f"Validation {variant} seed {seed}",
                        validate_cmd,
                        validate_log,
                        args.continue_on_error,
                    )
                    if not ok:
                        failures.append((variant, seed, "validate"))
                except RuntimeError:
                    failures.append((variant, seed, "validate"))
                    raise
        else:
            print(f"Skip validation: metrics already exists: {metrics_file}")

    return failures


def run_comparison(args, log_dir):
    print_header("Ablation Comparison")

    if args.dry_run:
        if all_default_metrics_exist():
            print(f"[DRY-RUN] Would run comparison: {display_command(compare_command())}")
        else:
            print("[DRY-RUN] Would skip comparison until all default 5 x 3 metrics.csv files are complete.")
        return

    if not all_default_metrics_exist():
        print("Skip comparison: not all default 5 x 3 metrics.csv files exist yet.")
        print("Missing metrics:")
        for variant, seed in planned_jobs(DEFAULT_VARIANTS, DEFAULT_SEEDS):
            metrics_file = metrics_file_for(variant, seed)
            if not metrics_file.exists():
                print(f"  {metrics_file}")
        return

    compare_log = log_dir / "compare_paldrl_ablation.log"
    run_stage("Ablation comparison", compare_command(), compare_log, args.continue_on_error)


def main():
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = SCRIPT_DIR / "Logs" / "PAL-DRL-Ablation" / timestamp

    print_plan(args, log_dir)

    if not args.dry_run:
        check_ros_gazebo_ready()
        log_dir.mkdir(parents=True, exist_ok=True)

    failures = run_matrix(args, log_dir)
    run_comparison(args, log_dir)

    if failures:
        print_header("Failures")
        for variant, seed, stage in failures:
            print(f"{stage}: variant={variant}, seed={seed}")
        return 1

    print_header("Done")
    print("PAL-DRL ablation runner finished.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
