# SAC_VSL

基于 Soft Actor-Critic (SAC) 的高速公路施工区可变限速控制项目。项目使用 SUMO/TraCI 构建交通仿真环境，训练智能体为 E1-E3 区域内的 CAV 车辆输出统一的连续限速值，并与固定限速基线进行对比评估。

## 项目概述

本项目面向高速公路施工区的 CAV 可变限速控制。当前版本采用一个统一动作控制 E1、E2、E3 三个路段上的 CAV/CAV_truck 车辆，不再对三个路段分别输出限速。

主要特性：

- SAC 算法：双 Critic、Target Critic、自动熵调节，Actor/Critic 使用全连接网络。
- 连续 VSL 动作：限速范围为 60-120 km/h。
- 动作变化约束：相邻 5 min 控制周期限速变化不超过 20 km/h。
- 状态空间：E1-E6 每条路段每个车道的 10 s 窗口交通统计量，共 72 维。
- 奖励函数：E1-E4 平均速度收益与 TTC 风险惩罚组合。
- 评估对比：同时输出 RL 控制结果和固定 80 km/h 无控制基线。
- 多场景运行：支持 cav25、cav50、cav75、cav100，`--scenario all` 会并行运行四个场景。

## 目录结构

```text
大创场景/
├── README.md
└── 大创场景/
    ├── run.py                  # 命令行入口
    ├── config.py               # 路网、仿真、SAC、奖励和评估配置
    ├── sumo_env.py             # SUMO/TraCI 环境
    ├── sac_agent.py            # SAC 智能体
    ├── sac_vsl.py              # 训练、评估和结果绘图
    ├── net.net.xml             # SUMO 路网文件
    ├── detectors.add.xml       # 检测器配置
    ├── *.sumocfg               # 不同 CAV 渗透率场景配置
    ├── *.rou.xml               # 车辆流与路径文件
    ├── checkpoints/            # 训练生成的模型文件
    └── results/                # 训练与评估输出结果
```

`checkpoints/`、`results/`、`*.out.xml` 和 `__pycache__/` 属于运行生成文件，通常不需要提交到仓库。

## 路网与场景

仿真路网覆盖 E0-E7，其中控制、观测、奖励和评估范围如下：

| 用途 | 路段 |
| --- | --- |
| VSL 控制 | E1, E2, E3 |
| 状态观测 | E1, E2, E3, E4, E5, E6 |
| 奖励统计 | E1, E2, E3, E4 |
| 评估统计 | E1, E2, E3, E4, E5, E6 |
| 固定限速基线 | E1, E2, E3, E4, E5, E6 |

E5 路段长度为 350 m，配置已在 `net.net.xml`、`detectors.add.xml` 和 `config.py` 中同步。

四个 CAV 渗透率场景：

| 场景 | CAV 渗透率 | SUMO 配置 |
| --- | ---: | --- |
| cav25 | 25% | `speedcontrol9.sumocfg` |
| cav50 | 50% | `speedcontrol10.sumocfg` |
| cav75 | 75% | `speedcontrol1.sumocfg` |
| cav100 | 100% | `speedcontrol2.sumocfg` |

## 强化学习设置

### 状态空间

状态维度为 72：

```text
6 条观测路段 × 3 条车道 × 4 个特征 = 72
```

每个车道的 4 个特征为：

- 10 s 窗口平均速度
- 10 s 窗口车辆数量 / 路段长度
- 10 s 窗口速度标准差
- 10 s 窗口车辆数量 / 路段长度标准差

环境每秒采集一次 lane 级瞬时速度和车辆数量，并维护 10 s 滚动窗口。决策点返回窗口统计值，而不是当前瞬时交通状态。

### 动作空间

动作维度为 1。SAC 输出的连续动作会映射为 E1-E3 全区域 CAV 统一限速：

```text
VSL ∈ [60, 120] km/h
```

控制周期为 300 s。相邻控制周期的限速变化幅度被限制在 20 km/h 以内。

### 奖励函数

奖励统计范围为 E1-E4，默认每 3 s 采样一次：

```text
reward = speed_norm - ttc_risk
speed_norm = mean_speed_kmh / 120
ttc_risk = mean(TTC values where TTC < 3 s) / 3
```

如果当前统计窗口内没有 TTC < 3 s 的样本，则 `ttc_risk = 0`。

## 仿真与评估

仿真时长为 3 h：

```text
SIM_END = 10800 s
CONTROL_INTERVAL = 300 s
CONTROL_STEPS_PER_EPISODE = 36
WARMUP_SECONDS = 600 s
```

评估统计频率为 1 s。每次评估同时运行：

- RL 控制策略
- 固定 80 km/h 无控制基线

输出指标包括：

- E1-E6 车辆总通行时间
- CO2 排放
- TTC < 3 s 总数量
- E4 的 TTC < 3 s 数量
- E4 平均速度
- E1-E3 的 TTC < 3 s 数量
- E1-E3 平均速度

## 环境要求

- Windows 或 Linux
- Python 3.9+
- SUMO 1.26.0 或兼容版本
- PyTorch
- NumPy
- Matplotlib
- TraCI

默认 SUMO 路径在 `大创场景/config.py` 和 `大创场景/run.py` 中配置为：

```text
D:/sumo-win64-1.26.0/sumo-1.26.0
```

如果本机安装路径不同，需要先修改对应配置。

## 使用方法

进入代码目录：

```powershell
cd 大创场景
```

训练单个场景：

```powershell
python run.py --scenario cav100 --mode train --episodes 300
```

并行训练四个场景：

```powershell
python run.py --scenario all --mode train --episodes 300
```

评估单个场景：

```powershell
python run.py --scenario cav100 --mode eval --checkpoint checkpoints/sac_vsl_cav100_best.pt --episodes 1
```

并行评估四个场景：

```powershell
python run.py --scenario all --mode eval --checkpoint checkpoints/ --episodes 1
```

指定并行 worker 数：

```powershell
python run.py --scenario all --mode eval --checkpoint checkpoints/ --parallel-workers 4
```

查看命令行参数：

```powershell
python run.py --help
```

## 注意事项

- 当前版本的状态维度为 72、动作维度为 1。旧版本 checkpoint 如果基于 44 维状态或三段式动作训练，不能直接复用。
- `--scenario all` 在非 GUI 模式下会使用多进程并行运行四个场景，每个进程独立创建 SUMO/TraCI 连接。
- GUI 模式下建议单场景运行，避免多个 SUMO GUI 同时启动。
- 如果评估时没有提供 checkpoint，程序会提示并使用未训练智能体运行，结果仅用于流程检查。

## 参考

- Haarnoja, T. et al. Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor. ICML, 2018.
- Haarnoja, T. et al. Soft Actor-Critic Algorithms and Applications. arXiv, 2018.
- Lopez, P. A. et al. Microscopic Traffic Simulation using SUMO. IEEE ITSC, 2018.
