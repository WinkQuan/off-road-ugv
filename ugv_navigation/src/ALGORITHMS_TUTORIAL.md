# 算法对比实验快速指南

## 概述

本项目实现了三种深度强化学习算法用于UGV离线路径规划：

1. **DQN**（原始）- 现有实现
2. **D3QN**（新添加）- 改进的双目标网络
3. **BC**（新添加）- 行为克隆纯模仿学习

## 快速开始

### 原始DQN模型（已存在）
```bash
# 模型已预训练，直接验证
python validate.py
```

### 新增D3QN算法
```bash
# 1. 训练D3QN模型
python main_d3qn.py

# 2. 验证D3QN性能 (8个指标)
python validate_d3qn.py
```

### 新增BC算法
```bash
# 1. 训练BC模型
python main_bc.py

# 2. 验证BC性能 (8个指标)
python validate_bc.py
```

## 评价指标（8个）

所有脚本都会输出以下指标：

| 指标 | 说明 | 越大越好 |
|-----|------|---------|
| Success Rate | 成功到达的比例 | ✓ |
| Collision Rate | 碰撞的比例 | ✗ |
| Timeout Rate | 超时的比例 | ✗ |
| Avg Time Step | 平均步数 | ✗ |
| Avg Trajectory | 平均路程(m) | ✗ |
| Avg Energy | 平均能耗(J) | ✗ |
| Avg Posture Stability | 姿态稳定性(rad) | ✗ |
| Avg Execution Time | 执行时间(s) | ✗ |

## 算法特性对比

### D3QN (Dueling Double Deep Q-Network)
**优点:**
- 使用两个目标网络，更稳定的Q值估计
- 减少高估偏差（Overestimation Bias）
- 在复杂环境中收敛更好

**缺点:**
- 计算量略大（两个目标网络更新）
- 训练时间比DQN长

**适用场景:**
- 奖励信号复杂或嘈杂的环境
- 需要高稳定性的任务
- 对收敛性能要求高

### BC (Behavior Cloning)
**优点:**
- 完全来自专家演示的学习
- 训练速度快（无探索阶段）
- 推理快，适合实时应用

**缺点:**
- 学习能力受限于专家策略
- 无法超越专家表现
- 可能出现分布偏移（Distribution Shift）

**适用场景:**
- 有可靠专家策略（如APF）的任务
- 实时导航应用
- 快速原型验证

## 代码结构

```
ugv_navigation/src/
├── ddqn.py           # 原始DQN实现
├── d3qn.py           # D3QN新实现
├── bc.py             # BC新实现
├── main.py           # DQN训练脚本
├── main_d3qn.py      # D3QN训练脚本（新）
├── main_bc.py        # BC训练脚本（新）
├── validate.py       # DQN验证脚本
├── validate_d3qn.py  # D3QN验证脚本（新）
├── validate_bc.py    # BC验证脚本（新）
└── algorithm_comparison.py  # 对比分析脚本（新）
```

## 修改建议（不改现有代码）

三种新算法实现完全独立，只需要：

1. **改一个import**：从 `ddqn` 改为 `d3qn` 或 `bc`
2. **改一个类名**：从 `DQN()` 改为 `D3QN()` 或 `BC()`

示例：
```python
# 原始DQN
import ddqn
agent = ddqn.DQN(...)

# 改为D3QN
import d3qn
agent = d3qn.D3QN(...)

# 改为BC
import bc
agent = bc.BC(...)
```

## 性能对比流程

1. **单独训练评估**
   ```bash
   # 终端1: 训练D3QN
   python main_d3qn.py
   
   # 终端2: 训练BC（并行）
   python main_bc.py
   ```

2. **验证性能**
   ```bash
   python validate.py      # DQN
   python validate_d3qn.py # D3QN
   python validate_bc.py   # BC
   ```

3. **查看对比结果**
   ```bash
   python algorithm_comparison.py
   ```

## 参数调整建议

如需优化性能，可在训练脚本中调整：

### D3QN
- `target_update`: 更新频率（降低=更稳定但更新慢）
- `learning_rate`: 学习率（DQN默认1e-4，可尝试1e-5）
- `gamma`: 折扣因子（默认0.99）

### BC
- `learning_rate`: 通常比DQN高（默认1e-3）（可尝试1e-4至1e-2）
- `batch_size`: 批量大小（通常64-128）

## 常见问题

**Q: 为什么BC训练很快？**  
A: BC只使用CrossEntropyLoss进行分类，不需要Q值计算和目标网络更新。

**Q: D3QN一定比DQN好吗？**  
A: 不一定。在简单任务上两者相当，但在复杂/嘈杂环境D3QN更稳定。

**Q: BC能超越APF吗？**  
A: BC只能学到APF的策略，无法超越。若要更好性能，应选择DQN或D3QN。

**Q: 如何选择算法？**  
A: 
- **实时应用** → BC
- **需要高稳定性** → D3QN
- **通用性强** → DQN
- **最优性能** → D3QN

## 输出文件

训练后会生成：
```
Model/              → DQN模型
Model_D3QN/         → D3QN模型
Model_BC/           → BC模型
```

验证后会生成：
```
trajectory.txt              → DQN轨迹
velocity.csv                → DQN速度
D3QN_trajectory.txt         → D3QN轨迹
D3QN_velocity.csv           → D3QN速度
BC_trajectory.txt           → BC轨迹
BC_velocity.csv             → BC速度
```

## 技术细节

### D3QN核心改进
```python
# 使用两个目标网络
q_target_1 = target_net_1(next_states)[actions]
q_target_2 = target_net_2(next_states)[actions]
# 取最小值，减少高估
q_target = min(q_target_1, q_target_2)
```

### BC核心思想
```python
# 纯分类学习，无Q值
loss_bc = CrossEntropyLoss(
    policy_output,
    expert_action  # APF给出的动作
)
```

## 论文参考

- DQN: Mnih et al. "Playing Atari with Deep Reinforcement Learning" (2013)
- D3QN: 结合Dueling DQN + Double DQN + Triple network思想
- BC: Pomerleau "ALVINN: An autonomous land vehicle in a neural network" (1989)

---

**祝你实验顺利！如有问题，检查日志输出即可。**
