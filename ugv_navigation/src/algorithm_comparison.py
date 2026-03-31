#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Comparison script for three algorithms: DQN vs D3QN vs BC
Displays performance metrics side-by-side for easy comparison
"""

import pandas as pd
import numpy as np
from pathlib import Path


def read_results(algorithm_name):
    """读取验证结果"""
    if algorithm_name == "DQN":
        print(f"\n{algorithm_name} Results:")
        print("Please run validate.py to generate results")
        return None
    elif algorithm_name == "D3QN":
        print(f"\n{algorithm_name} Results:")
        print("Please run validate_d3qn.py to generate results")
        return None
    elif algorithm_name == "BC":
        print(f"\n{algorithm_name} Results:")
        print("Please run validate_bc.py to generate results")
        return None


def create_comparison_table():
    """创建对比表格"""
    print("\n" + "=" * 80)
    print("Algorithm Comparison: DQN vs D3QN vs BC")
    print("=" * 80)

    algorithms = ["DQN (Original)", "D3QN (Triple Network)", "BC (Behavior Cloning)"]

    # 7个评价指标的说明
    metrics = {
        "Success Rate (%)": "Higher is better",
        "Collision Rate (%)": "Lower is better",
        "Timeout Rate (%)": "Lower is better",
        "Avg Steps": "Lower is better (faster)",
        "Avg Trajectory (m)": "Lower is better (shorter path)",
        "Avg Energy (J)": "Lower is better (energy efficient)",
        "Avg Posture Stability (rad)": "Lower is better (more stable)",
        "Avg Execution Time (s)": "Lower is better (real-time)",
    }

    print("\nMetrics to compare:")
    for i, (metric, desc) in enumerate(metrics.items(), 1):
        print(f"  {i}. {metric}: {desc}")

    print("\n" + "-" * 80)
    print("Algorithm Characteristics:")
    print("-" * 80)

    chars = {
        "DQN (Original)": [
            "Base Q-learning with replay buffer",
            "Dueling architecture for value/advantage separation",
            "Good balance between exploration and exploitation",
            "Stable learning with imitation loss mixing",
        ],
        "D3QN (Triple Network)": [
            "Two target networks for more stable Q-estimates",
            "Reduced overestimation bias",
            "Better convergence in complex environments",
            "Slightly higher computational cost",
        ],
        "BC (Behavior Cloning)": [
            "Pure imitation learning from expert (APF) demonstrations",
            "No exploration (deterministic policy)",
            "Fast convergence but limited to expert capability",
            "Best for structured navigation tasks",
        ],
    }

    for algo, points in chars.items():
        print(f"\n{algo}:")
        for point in points:
            print(f"  • {point}")

    print("\n" + "=" * 80)
    print("How to run the comparison:")
    print("=" * 80)
    print(
        """
1. Train DQN (original):
   python main.py            # Already trained
   python validate.py        # Test original model

2. Train D3QN (new algorithm):
   python main_d3qn.py       # Train D3QN
   python validate_d3qn.py   # Test D3QN model

3. Train BC (new algorithm):
   python main_bc.py         # Train BC
   python validate_bc.py     # Test BC model

4. Compare results:
   Run this script after generating all results
    """
    )

    print("\n" + "=" * 80)
    print("Expected Performance (Qualitative):")
    print("=" * 80)

    comparison_data = {
        "Metric": [
            "Success Rate",
            "Navigation Efficiency",
            "Energy Efficiency",
            "Posture Stability",
            "Training Stability",
            "Convergence Speed",
            "Real-time Performance",
        ],
        "DQN": [
            "Good (multi-loss)",
            "Good (learned policy)",
            "Medium",
            "Medium",
            "Good (stable training)",
            "Medium",
            "Good",
        ],
        "D3QN": [
            "Very Good (improved Q-estimates)",
            "Very Good (reduced overestimation)",
            "Very Good",
            "Very Good",
            "Very Good (dual targets)",
            "Slow (more updates)",
            "Good (same inference)",
        ],
        "BC": [
            "Good (expert dependent)",
            "Good (follows expert)",
            "Good (copies expert)",
            "Good (copies expert)",
            "Very Fast (classification)",
            "Very Fast (simple loss)",
            "Very Good (direct imitation)",
        ],
    }

    df = pd.DataFrame(comparison_data)
    print(df.to_string(index=False))

    print("\n" + "=" * 80)
    print("Recommended Use Cases:")
    print("=" * 80)
    print(
        """
DQN (Original):
  ✓ Good baseline, combines learning and imitation
  ✓ Suitable for environments with clear reward signals
  
D3QN (Triple Network):
  ✓ Complex environments with noisy rewards
  ✓ When stability is critical
  ✓ Best performance in challenging scenarios
  
BC (Behavior Cloning):
  ✓ When expert demonstrations are reliable
  ✓ Real-time constraints (fast inference)
  ✓ Structured navigation tasks like off-road
  ✓ Fast training when expert policy is known
    """
    )

    print("\n" + "=" * 80)


if __name__ == "__main__":
    create_comparison_table()

    print("\nTo add your results, manually edit this script or use:")
    print("  python validate.py > results_dqn.txt")
    print("  python validate_d3qn.py > results_d3qn.txt")
    print("  python validate_bc.py > results_bc.txt")
