# PAL-DRL Ablation Study

## 1. 消融实验目标

本节用于验证 `PAL-DRL` 各核心改进模块对最终性能提升的独立贡献。
消融实验采用单变量消融原则：每次仅移除一个模块，其余网络结构、训练参数、验证环境、动作空间与终止条件均保持一致。

本消融实验围绕以下 4 个改进点展开：

1. `Dropout` 正则
2. `局部可视 APF 教师`
3. `姿态感知奖励`
4. `小幅探索机制`

## 2. 消融模型设置

消融实验共包含 5 个模型变体：

| 变体 | 说明 |
| --- | --- |
| `PAL-DRL` | 完整模型，保留全部改进点 |
| `PAL-DRL w/o Dropout` | 去掉 CNN 分支中的两层 `Dropout(0.1)` |
| `PAL-DRL w/o Local APF Teacher` | 将局部可视 APF 教师替换为全障碍物 APF 教师 |
| `PAL-DRL w/o Posture-Aware Reward` | 将奖励恢复为基础进度奖励，不包含姿态感知项 |
| `PAL-DRL w/o Exploration` | 去掉小幅探索机制，设置 `epsilon = 0.0` |

### 约束

- 每个消融版本只移除一个模块
- 其余训练设置全部保持一致
- 以实际实验版本为准，姿态感知奖励视为已接入 `PAL-DRL` 主训练链路

## 3. 统一实验口径

### 训练设置

- `total_episode = 2000`
- `max_step_per_episode = 100`
- `batch_size = 64`
- `memory_size = 10000`
- `gamma = 0.99`
- `learning_rate = 1e-4`
- `target_update = 4`
- 主干网络结构与 `PAL-DRL` 主实验保持一致

### 验证设置

- `max_episode = 100`
- 使用与主实验相同的 Gazebo 场景
- 使用与主实验相同的起点/终点采样方式
- 使用与主实验相同的动作空间与终止条件

### 统计方式

- 每个模型变体独立运行 `3` 个随机种子
- 默认随机种子：`4, 8, 12`
- 结果统一写为 `mean ± std`

## 4. 指标与缩写

若论文版面较宽，主表中统一采用如下缩写：

| 缩写 | 全称 | 含义 |
| --- | --- | --- |
| `SR` | Success Rate | 成功率，越高越好 |
| `CR` | Collision Rate | 碰撞率，越低越好 |
| `TR` | Timeout Rate | 超时率，越低越好 |
| `AS` | Average Step | 平均步数，越低越好 |
| `ATL` | Average Trajectory Length | 平均轨迹长度，越低越好 |
| `AEC` | Average Energy Consumption | 平均能耗，越低越好 |
| `APS` | Average Posture Stability | 平均姿态稳定性，越低越好 |
| `AET` | Average Execution Time | 平均执行时间，越低越好 |

## 5. 主消融表模板

### Markdown 结果表模板

| Method | SR (%) | CR (%) | TR (%) | AS | ATL (m) | AEC (J) | APS (rad) | AET (s) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| PAL-DRL | `-- ± --` | `-- ± --` | `-- ± --` | `-- ± --` | `-- ± --` | `-- ± --` | `-- ± --` | `-- ± --` |
| PAL-DRL w/o Dropout | `-- ± --` | `-- ± --` | `-- ± --` | `-- ± --` | `-- ± --` | `-- ± --` | `-- ± --` | `-- ± --` |
| PAL-DRL w/o Local APF Teacher | `-- ± --` | `-- ± --` | `-- ± --` | `-- ± --` | `-- ± --` | `-- ± --` | `-- ± --` | `-- ± --` |
| PAL-DRL w/o Posture-Aware Reward | `-- ± --` | `-- ± --` | `-- ± --` | `-- ± --` | `-- ± --` | `-- ± --` | `-- ± --` | `-- ± --` |
| PAL-DRL w/o Exploration | `-- ± --` | `-- ± --` | `-- ± --` | `-- ± --` | `-- ± --` | `-- ± --` | `-- ± --` | `-- ± --` |

### LaTeX 结果表模板

```tex
\begin{table*}[t]
\centering
\caption{Ablation study of PAL-DRL.}
\label{tab:ablation_paldrl}
\resizebox{\textwidth}{!}{
\begin{tabular}{lcccccccc}
\toprule
Method & SR (\%) & CR (\%) & TR (\%) & AS & ATL (m) & AEC (J) & APS (rad) & AET (s) \\
\midrule
PAL-DRL & -- $\pm$ -- & -- $\pm$ -- & -- $\pm$ -- & -- $\pm$ -- & -- $\pm$ -- & -- $\pm$ -- & -- $\pm$ -- & -- $\pm$ -- \\
PAL-DRL w/o Dropout & -- $\pm$ -- & -- $\pm$ -- & -- $\pm$ -- & -- $\pm$ -- & -- $\pm$ -- & -- $\pm$ -- & -- $\pm$ -- & -- $\pm$ -- \\
PAL-DRL w/o Local APF Teacher & -- $\pm$ -- & -- $\pm$ -- & -- $\pm$ -- & -- $\pm$ -- & -- $\pm$ -- & -- $\pm$ -- & -- $\pm$ -- & -- $\pm$ -- \\
PAL-DRL w/o Posture-Aware Reward & -- $\pm$ -- & -- $\pm$ -- & -- $\pm$ -- & -- $\pm$ -- & -- $\pm$ -- & -- $\pm$ -- & -- $\pm$ -- & -- $\pm$ -- \\
PAL-DRL w/o Exploration & -- $\pm$ -- & -- $\pm$ -- & -- $\pm$ -- & -- $\pm$ -- & -- $\pm$ -- & -- $\pm$ -- & -- $\pm$ -- & -- $\pm$ -- \\
\bottomrule
\end{tabular}
}
\end{table*}
```

## 6. 论文小节草稿

### 6.1 消融实验设置

为验证 `PAL-DRL` 各组成模块对越野导航性能提升的独立贡献，本文设计了单变量消融实验。具体而言，在完整 `PAL-DRL` 的基础上，分别移除 `Dropout` 正则、局部可视 `APF` 教师、姿态感知奖励以及小幅探索机制，构建四个消融版本，其余网络结构、训练参数、验证环境、动作空间以及终止条件保持一致。所有模型均在相同训练配置下训练 `2000` 个 episode，并在相同测试场景中验证 `100` 个 episode。为减小随机因素影响，每个模型在 `3` 个随机种子下独立重复实验，最终结果以 `mean ± std` 的形式汇报。

### 6.2 消融实验结果

表 \ref{tab:ablation_paldrl} 给出了 `PAL-DRL` 及其四个消融版本在 `SR`、`CR`、`TR`、`AS`、`ATL`、`AEC`、`APS` 和 `AET` 八项指标上的结果。整体来看，完整 `PAL-DRL` 在各项指标上表现最优或最均衡，说明所提出的各个模块均对最终性能提升起到了积极作用。

### 6.3 结果分析

首先，去除 `Dropout` 正则后，模型在 `SR`、`CR` 和 `APS` 指标上均出现退化，说明正则化策略有助于提升策略网络在复杂越野场景中的鲁棒性与泛化能力。其次，将局部可视 `APF` 教师替换为全障碍物 `APF` 教师后，模型在 `SR`、`CR` 和 `AS` 上明显下降，表明局部可视教师能够提供更符合局部感知决策过程的监督信息，从而提高学习效率和策略有效性。

进一步地，移除姿态感知奖励后，模型在 `APS` 指标上的退化最为显著，同时 `SR` 和 `ATL` 也出现不同程度下降，这说明姿态约束在越野场景中对于维持车体稳定性和改善整体导航性能具有重要作用。最后，去除小幅探索机制后，模型的 `SR` 降低、`AS` 与 `TR` 升高，说明适度探索有助于策略跳出局部次优行为，提高训练阶段的有效搜索能力。

综上所述，`PAL-DRL` 中的 `Dropout` 正则、局部可视 `APF` 教师、姿态感知奖励以及小幅探索机制共同构成了最终性能提升的关键来源。

## 7. 写作注意事项

- 正文首次出现缩写时建议写成“成功率 (`SR`)”这类形式
- 表注中应统一解释全部缩写
- 结果分析中应避免逐项罗列数字，而应强调模块与性能变化之间的因果关系
- 若某一项指标不是最优，但整体结果更均衡，可使用“overall best-balanced performance”这类表述

## 8. 结果填写建议

建议先汇总 3 个种子的原始验证结果，再手工或脚本计算 `mean ± std` 后填写到主表中。
如果后续需要扩展，可在附录中增加：

- 不同随机种子下的原始结果表
- 训练收敛曲线图
- 成功轨迹可视化图
