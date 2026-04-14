# Off-Road UGV

## Overview

This repository is a ROS/Gazebo simulation workspace for off-road unmanned ground vehicle (UGV) navigation and path-planning experiments.

The main research algorithm in this project is `PAL-DRL`. The workspace also includes several comparison baselines:

- `NPE-DRL`
- `D3QN`
- `BC`
- `DAgger`
- `APF`

The repository combines Gazebo worlds and models, learning-based navigation pipelines, validation scripts, and automatic metric comparison utilities in a single catkin workspace.

## Workspace Structure

- `mymodel_gazebo`: Gazebo worlds and simulation models used by the project.
- `ugv_navigation`: Main UGV navigation package, including training, validation, environment logic, and comparison scripts.
- `teleop_twist_keyboard`: Keyboard teleoperation utility for manual control via `/cmd_vel`.
- `ugv_navigation_without_obstacle`: Simplified or legacy no-obstacle variant kept for auxiliary use.

The main experiment entrypoints are located in `ugv_navigation/src`.

## Tested Environment

- `Ubuntu 20.04`
- `ROS Noetic`

This project is organized as a catkin workspace and uses Gazebo through the ROS/Gazebo integration.

## Dependencies

At a practical level, the project depends on:

- ROS Noetic / catkin
- Gazebo / `gazebo_ros`
- Python 3
- Python packages used by the codebase:
  - `numpy`
  - `torch`
  - `onnxruntime`
  - `opencv-python`
  - `wandb`
  - `Pillow`
  - `torchvision`

Additional ROS Python dependencies are declared by the packages in this workspace.

## Build the Workspace

From the repository root:

```bash
catkin_make
source devel/setup.bash
```

## Launch Gazebo Worlds

The main Gazebo world launch files are:

```bash
roslaunch mymodel_gazebo myworld.launch
roslaunch mymodel_gazebo myworld_without_obstacle.launch
roslaunch mymodel_gazebo myworld_flat_land.launch
```

World descriptions:

- `myworld.launch`: obstacle-rich off-road world
- `myworld_without_obstacle.launch`: obstacle-free world
- `myworld_flat_land.launch`: flat-land world

## Algorithms

| Algorithm | Role | Description |
| --- | --- | --- |
| `PAL-DRL` | Main method | APF-guided deep reinforcement learning method used as the primary research algorithm. |
| `NPE-DRL` | Baseline | Precursor baseline of `PAL-DRL`, used as a direct comparison target. |
| `D3QN` | Baseline | Pure dueling double deep Q-network baseline without APF supervision. |
| `BC` | Baseline | Behavior cloning from APF expert demonstrations. |
| `DAgger` | Baseline | Dataset aggregation with policy rollouts relabeled by the APF expert. |
| `APF` | Baseline | Pure artificial potential field controller. |

Notes:

- `PAL-DRL` is the main algorithm of the project.
- `NPE-DRL` is the precursor baseline of `PAL-DRL`.
- `BC` and `DAgger` use APF as expert supervision.
- The comparison baselines that use APF are configured with full-obstacle APF using `gazebo_ugv.cylinder_pos`.

## Training

Run the experiment scripts from `ugv_navigation/src`:

```bash
cd ugv_navigation/src
```

Training entrypoints:

```bash
python train_paldrl.py
python train_npedrl.py
python train_d3qn.py
python train_bc.py
python train_dagger.py
```

Training artifacts are written under `Model/<Algorithm>/`.

## Validation and Comparison

Validation entrypoints:

```bash
cd ugv_navigation/src

python validate_paldrl.py
python validate_npedrl.py
python validate_d3qn.py
python validate_bc.py
python validate_dagger.py
python validate_apf.py
python algorithm_comparison.py
```

Each validation script writes `metrics.csv` to `Validate/<Algorithm>/`.

`algorithm_comparison.py` aggregates validation metrics from:

- `PAL-DRL`
- `NPE-DRL`
- `D3QN`
- `BC`
- `DAgger`
- `APF`

## Outputs

Generated experiment outputs are stored in:

- `ugv_navigation/src/Model/`: training artifacts
- `ugv_navigation/src/Validate/`: validation artifacts

Typical output files include:

- `.pth`
- `.onnx`
- `Reward`
- `Step`
- `Success_Rate`
- `*_velocity.csv`
- `*_trajectory.txt`
- `metrics.csv`
- `algorithm_comparison.csv`

## Notes

- This README reflects the current code layout of the repository.
- Some older internal documentation may still reference deprecated names such as `main.py`, `main_d3qn.py`, or `validate.py`.
- `tello_flight.py` is an independent experimental script and is not part of the main UGV training/validation pipeline.
- The training and validation scripts depend on an active ROS/Gazebo runtime and are not standalone Python-only programs.
- `ugv_navigation_without_obstacle` is kept as an auxiliary variant and is not the primary workflow described here.
