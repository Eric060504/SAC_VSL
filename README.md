# 基于 SAC 的高速公路施工区可变限速控制

一个使用 **Soft Actor-Critic (SAC)** 强化学习算法，在 **SUMO** 交通仿真环境中动态控制高速公路施工区 **可变限速（VSL）** 的研究项目。

## 概述

本项目利用深度强化学习实现高速公路施工区的智能交通控制。SAC 智能体学习对施工区三个路段上的网联自动驾驶车辆（CAV）设定最优限速值，以同时提升**交通安全**、**通行效率**、**驾乘舒适性**和**道路吞吐量**。

### 主要特性

- **SAC 算法**：最先进的离策略（off-policy）强化学习算法，具备自动熵调优以实现稳定探索
- **多目标奖励**：综合平衡安全性（基于碰撞时间 TTC）、效率（行程时间）、舒适性（加加速度最小化）和吞吐量
- **CAV 渗透率场景**：支持 25%、50%、75%、100% 四种 CAV 渗透率
- **SUMO 集成**：通过 TraCI 接口实现逼真的交通仿真
- **44 维状态空间**：包含路段级和车道级交通特征的全面观测
- **连续动作空间**：三个施工区段 E1/E2/E3 的可变限速值

## 项目结构

```
大创场景/
├── run.py                  # 命令行入口（训练与评估）
├── config.py               # 中心配置（路径、超参数、奖励权重）
├── sumo_env.py             # SUMO/TraCI 环境（Gym 风格接口）
├── sac_agent.py            # SAC 算法实现（Actor、Critic、ReplayBuffer）
├── sac_vsl.py              # 训练编排器（回合循环、日志记录、绘图）
├── detectors.add.xml       # SUMO 检测器定义
├── net.net.xml             # SUMO 道路网络定义
├── *.sumocfg               # 各 CAV 场景的 SUMO 配置文件
├── *.rou.xml               # 各场景的车辆流路径文件
├── e2_E*.out.xml / e3_*.xml  # SUMO 输出 / 检测器配置文件
├── checkpoints/            # 模型检查点目录（生成目录）
└── results/                # 训练图表和对比图（生成目录）
```

## 路网布局

仿真的道路网络为包含施工区的高速公路走廊：

```
E0 ──→ E1 ──→ E2 ──→ E3 ──→ E4 ──→ E5
(500m)  (500m) (500m) (500m) (200m) (300m)
        ↑ VSL   ↑ VSL   ↑ VSL
       控制段   控制段   控制段
```

- **E0**：上游接近段（500m，限速 120 km/h）
- **E1、E2、E3**：施工区段（各 500m，限速 80 km/h）—— **VSL 控制应用于此**
- **E4、E5**：恢复 / 下游段（200m + 300m）
- 所有路段均为 3 车道

VSL 值仅应用于 E1/E2/E3 路段上的 **CAV 车辆**。非 CAV（人工驾驶）车辆遵循默认跟驰行为（IDM 模型）。

## 强化学习公式

### 状态空间（44 维）

| 特征组 | 维度 | 描述 |
|---|---|---|
| 路段级特征（E0–E5） | 6 路段 × 5 = 30 | 速度比、占有率、密度、停驻比例、车道速度标准差 |
| 车道级速度（E1–E3） | 3 路段 × 3 车道 = 9 | 每车道平均速度归一化值 |
| 上游 E0 特征 | 3 | 流入比、速度比、占有率 |
| 时间编码 | 2 | 归一化仿真时间的 sin/cos |

### 动作空间（3 维）

连续值 `[-1, 1]` 映射到 VSL 值 `[8.33, 22.22] m/s`（30–80 km/h），对应路段 E1、E2、E3。动作通过指数移动平均（EMA，α=0.3）进行平滑。

### 奖励函数

复合奖励函数加权四个目标：

| 分量 | 权重 | 描述 |
|---|---|---|
| **安全** (r_safety) | 0.35 | 惩罚 TTC < 3s 的违规和车道间速度差 |
| **效率** (r_efficiency) | 0.35 | 基于 E1–E5 段平均行程时间与自由流基线（约 90s）的对比 |
| **舒适** (r_comfort) | 0.15 | 平均加加速度（加速度变化率）的指数衰减 |
| **吞吐量** (r_throughput) | 0.15 | 已完成车辆数与预期需求的比值 |

### 控制周期

- **决策间隔**：每 120 秒仿真时间
- **预热阶段**：前 600s（5 个间隔）不施加控制，用于路网车辆填充
- **单回合时长**：3600s（1 小时）→ 预热后共 30 个控制步

## 场景配置

| 场景 | CAV 比例 | SUMO 配置文件 | 随机种子 |
|---|---|---|---|
| cav25 | 25% | speedcontrol9.sumocfg | 0 |
| cav50 | 50% | speedcontrol10.sumocfg | 1 |
| cav75 | 75% | speedcontrol1.sumocfg | 2 |
| cav100 | 100% | speedcontrol2.sumocfg | 3 |

交通需求：总计 3,400 辆/小时，卡车占比 20%。

## 使用方法

### 环境要求

- **SUMO 1.26.0** 安装在 `D:/sumo-win64-1.26.0/sumo-1.26.0`（可在 `config.py` 中配置）
- Python 环境需安装 PyTorch、NumPy、Matplotlib

### 训练

```bash
# 训练单个场景（默认：cav100，300 回合）
python run.py --scenario cav100 --mode train --episodes 300

# 依次训练所有场景
python run.py --scenario all --mode train --episodes 300

# 使用 GUI 训练（用于调试）
python run.py --scenario cav50 --mode train --gui
```

### 评估

```bash
# 评估已训练的模型
python run.py --scenario cav100 --mode eval --checkpoint checkpoints/sac_vsl_cav100_best.pt

# 评估所有场景
python run.py --scenario all --mode eval --checkpoint checkpoints/

# 指定评估回合数
python run.py --scenario cav50 --mode eval --checkpoint checkpoints/sac_vsl_cav50_best.pt --episodes 20
```

### 其他参数

```bash
python run.py --help
# --scenario:  cav25 | cav50 | cav75 | cav100 | all（默认：cav100）
# --mode:      train | eval（默认：train）
# --episodes:  回合数（训练默认 300，评估默认 10）
# --gui:       启动 SUMO GUI 可视化
# --checkpoint: 检查点文件或目录路径
# --device:    auto | cpu | cuda（默认：auto）
```

## SAC 算法细节

本项目实现 **Soft Actor-Critic**，包含：

- **重参数化高斯策略**：随机动作用于探索，确定性均值用于评估
- **双 Q 网络**：裁剪双 Q 学习以减少价值过估计
- **软目标更新**：Polyak 平均（τ=0.005）确保稳定训练
- **自动熵调优**：可学习的 α 平衡探索与利用
- **层归一化（LayerNorm）**：应用于 Actor 和 Critic 网络以提升训练稳定性
- **梯度裁剪**：范数裁剪为 1.0

### 网络架构

Actor 和 Critic 共享相同的隐藏层结构：`[256, 256, 128]`，激活函数为 ReLU，并使用 LayerNorm。

### 关键超参数

| 参数 | 值 |
|---|---|
| Actor / Critic / Alpha 学习率 | 3e-4 |
| 折扣因子 (γ) | 0.99 |
| 批大小 | 256 |
| 经验回放缓冲区大小 | 1,000,000 |
| 每环境步更新次数 | 5 |
| 梯度裁剪范数 | 1.0 |
| 初始 log α | ln(0.1) ≈ -2.3026 |

## 输出结果

### 训练

- **检查点**：每 50 回合保存一次 + 最优模型（基于 10 回合滑动平均）+ 最终模型，存储于 `checkpoints/`
- **图表**：奖励曲线、损失/α 曲线、VSL 策略演化图，保存至 `results/`

### 评估

- 每回合指标打印至控制台
- 跨场景对比表格（使用 `--scenario all` 时）
- 对比柱状图保存至 `results/scenario_comparison.png`

## 参考文献

- Haarnoja, T., et al. "Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor." ICML 2018.
- Haarnoja, T., et al. "Soft Actor-Critic Algorithms and Applications." arXiv 2018.
- SUMO: "Microscopic Traffic Simulation using SUMO." IEEE ITSC 2018.

## 运行环境

- **Python**：Conda 环境 `LLM_Classification`，Python 可执行文件位于 `D:/anaconda/envs/LLM_Classification/python.exe`
- **SUMO**：版本 1.26.0，安装于 `D:/sumo-win64-1.26.0/sumo-1.26.0`
- **操作系统**：Windows 11
- **GPU**：支持 CUDA（自动检测）
