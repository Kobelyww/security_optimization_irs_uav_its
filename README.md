# Security Optimization of Communication for IRS-UAV Enabled Intelligent Transportation System

仿真代码目录 **`security_optimization_irs_uav_its`**。

本目录提供 **IRS 辅助 UAV** 场景下的 **通感一体化（ISAC）** 仿真环境与 **混合离散/连续动作** 的 DRL 实现，用于安全与感知联合资源分配实验。

## 建模要点

- **网络拓扑**：BS-MEC（多天线）服务地面合法用户；**D-UAV** 负责数据采集与卸载，**I-UAV** 携带 IRS，同时辅助 **D-UAV→BS-MEC 卸载链路** 与 **下行 ISAC**；存在窃听者（EAV）与待感知目标。
- **波形与发射**：下行采用 **OTFS-OFDMA**，在延迟–多普勒资源网格上做多用户 ISAC；发射信号包含 **多用户有源波束** 与 **人工噪声（AN）**；仿真中 BS 侧采用 **ZF 预编码 + 感知感知的零空间 AN**，在保障用户 SINR 的同时抑制窃听端并约束发射协方差。
- **IRS 与信道**：IRS 相移可优化级联信道；链路建模区分 **IRS 辅助链路的 LoS 主导** 与 **直射地面的 NLoS 主导**；可选 **城市/道路热点交通场**，对直射衰减、IRS 阴影与感知杂波等施加与无人机位置相关的调制。
- **MEC 与能耗**：**部分卸载比例** 与飞行/悬停能耗、时隙内可行性耦合，进入奖励中的惩罚项。
- **感知与 ISAC 指标**：用目标方向 **感知效用（如对 S_T 的对数刻画）** 与 **发射波束方向图相对期望图的 MSE** 等将雷达侧性能写入目标，使保密与感知 **联合优化** 而非仅通信速率。
- **混合动作**：**离散**——I-UAV / D-UAV 在候选部署点集合上选点；**连续**——功率分配、IRS 相位、AN 占功率比例、卸载比等；状态含通道/IRS 对齐、几何与（可选）交通等 **异构特征**。

## 技术特点

- **问题层面**：在 IRS–UAV ITS 设定下，将 **UAV 部署、OTFS-OFDMA 侧等效控制、IRS 相位、AN 与功率分配、MEC 卸载** 放在同一 **通感 + 物理层安全** 目标下，显式包含 **保密指标与波束/感知质量**。
- **方法层面**：**Attention PPO（APPO-SAC）** 带 **特征注意力**；**混合动作** 由 **离散部署头** 与 **连续 SAC 分支**（波束份额、IRS、AN、卸载）配合；可选用 **GAE** 稳定 PPO 更新。
- **波形与场景**：**OTFS-OFDMA ISAC**；支持 **双机 UAV + IRS**、可选 **交通拥塞场** 调制信道与感知杂波。
- **实验与范围**：训练收敛行为与多种 DRL 基线（Hybrid-SAC、PPO-SAC、DDPG-DQN 等）可对拍 sweep。**本仓库不包含** AO、Equal-Power 等非学习基线实现。

## MDP：状态空间、动作空间与奖励

本节对应 **`environment.py`** 中的观测、动作与奖励；状态对高维信道做了 **统计摘要**。

### 状态空间 $\mathcal{S}$

**符号化状态（高层）**（时隙 $i$），汇集信道、感知、交通与历史信息：

$$
s_i = \left\{\{\mathbf{h}_{D,I}[i]\},\{\mathbf{G}_{B,I}[i]\},\{\mathbf{H}_{u_m}[i]\},\mathbf{H}_{E}[i],\mathbf{H}_{\mathrm{T}}[i],\varrho(Q_I[i],i),\varrho(Q_D[i],i),\{Q_{u_m}[i]\},\{\varrho(Q_{u_m}[i],i)\},L(\zeta,\mathbf{R}_{\mathbf{x}_B}[i]),a_{i-1},r_i\right\},
$$

其中 $\varrho(\cdot,i)$ 为交通拥塞场；$L(\zeta,\mathbf{R}_{\mathbf{x}_B}[i])$ 表示与期望波束图相关的失配（实现中通过波束图 MSE 进入 **奖励** 而非直接逐元素展开在状态中）；$a_{i-1}, r_i$ 为上一动作与当前即时奖励记忆。

**本仓库观测向量** `env._get_state()` 由下列 **实值特征** 拼接（复信道先做幅度/相位 **压缩统计**，并含 IRS **对齐相位提示**）：

1. 每个用户 $m$：合法复合信道 $\mathbf{H}_{u_m}$（取 $\mathbf{\Theta}=\mathbf{I}$）的 **平均幅度、平均相位**（各 1 维，共 $2M$ 维）；  
2. 窃听信道 $\mathbf{H}_E$：**平均幅度、平均相位**（2 维）；  
3. 感知响应 $\mathbf{H}_{\mathrm{T}}$：**平均幅度**（1 维）；  
4. 卸载有效通道 $| \mathbf{h}_{I,B}^H \mathbf{\Theta}\,\mathbf{h}_{D,I} |$（零相位时，1 维）；  
5. 各用户 IRS **逐单元对齐角** $\angle(\mathbf{s}_{\mathrm{IRS},m}) - \angle(\mathbf{s}_{\mathrm{IRS,BI}})$ 归一化到 $[-1,1]$（共 $M N_{\mathrm{IRS}}$ 维）；窃听对齐（$N_{\mathrm{IRS}}$ 维）；  
6. I-UAV / D-UAV **水平位置** $q_I, q_D$ 除以 `AREA_RANGE`（4 维）；  
7. 若启用 **复杂交通**（`COMPLEX_TRAFFIC=1`）：两架 UAV 处 $\varrho$、各用户位置（归一化）及其 $\varrho$（共 $2 + 3M$ 维）；  
8. 时隙进度 $i / T_{\mathrm{ep}}$（1 维）、**上一时刻奖励** $r_{i-1}$（1 维）。

记 $M=$ `NUM_USERS`，$N_I=$ `NUM_DP_I`，$N_D=$ `NUM_DP_D`，$N=$ `n_irs`，则：

$$
d_s = \underbrace{2M + 4 + (M+1)N + 4 + 2}_{\text{与交通无关部分}} + \underbrace{(2 + 3M)\cdot \mathbb{1}_{\text{COMPLEX}}}_{\text{可选}} = 2M + (M+1)N + 10 + (2+3M)\cdot \mathbb{1}_{\text{COMPLEX}}.
$$

默认 $M=2, N=16$ 且无复杂交通时，$d_s=62$（等于 `env.state_dim`）。属性：`env.state_dim`、`env.cont_dim`、`env.disc_n_I`、`env.disc_n_D`。

### 动作空间 $\mathcal{A}$

**完整射频表征下的联合动作**（时隙 $i$：逐 RE 波束矩阵 + 协方差等）：

$$
a_i = \left\{\mathbf{q}_I[i],\mathbf{q}_D[i],\{\mathbf{W}[g,i]\}_{g=1}^{N_\tau N_\nu},\mathbf{\Theta}_n[i],\alpha[i],\mathbf{R}_z[i]\right\}.
$$

**本仓库接口**（PPO-SAC 等智能体一致）：  

- **离散部分**：$a^d_i = (d_I,d_D)$，其中 $d_I\in\{0,\ldots,N_I-1\}$、$d_D\in\{0,\ldots,N_D-1\}$，对应 I-UAV / D-UAV 在候选点集 `dp_I`、`dp_D` 中的索引。  
- **连续部分**：$\mathbf{u}_i\in[-1,1]^{d_c}$，$d_c = M + N + 2$（`env.cont_dim`）。环境 `decode_action` 将其映射为物理量：

  - **用户功率份额**：$\mathbf{p}_f = \mathrm{softmax}(\mathrm{clip}(\mathbf{u}_{1:M}))$（与总功率/AN 分割一起在 `_build_beamforming` 中使用）；  
  - **IRS 相位**：$\phi_n = \pi(\mathbf{u}_{M:M+N}+1) \in [0,2\pi]$；  
  - **AN 功率占比**：$f_{\mathrm{AN}} = \mathrm{clip}\left((u_{M+N}+1)/2,\,[0.05,0.5]\right)$；  
  - **卸载比例**：$\alpha = \mathrm{clip}\left((u_{d_c}+1)/2,\,[0.05,0.95]\right)$。

下行 **ZF 波束** 与 **零空间 AN 协方差** 在环境中由 $(\mathbf{p}_f,\phi,f_{\mathrm{AN}})$ 与当前信道 **解析生成**，网络不直接输出 $\mathbf{W}[g]$、$\mathbf{R}_z$ 的全部复元素，从而压缩连续动作维数。

### 奖励函数

**即时奖励（加权结构）**：

$$
r_i=\omega_1\sum_{m=1}^{M}R^{\mathrm{Sec}}_{u_m}[i]+\omega_2 S_{\mathrm{T}}[i]-\omega_3 L(\zeta,\mathbf{R}_{\mathbf{x}_B}[i])-\omega_4 \Phi_i,
$$

其中 $R^{\mathrm{Sec}}_{u_m}$ 为合法速率与窃听速率差（取非负后求和），$S_{\mathrm{T}}$ 为感知效用，$L$ 为波束图失配项，$\Phi_i$ 为能耗/时隙/飞行可行性及（可选）交通风险的惩罚。

**实现中**（`config.py`：`OMEGA_*`、`R_SEC_NORM`、`S_T_NORM`、`BP_MSE_NORM`、`PENALTY_NORM`、`REWARD_SCALE`）使用 **归一化标度** 的稳定形式：

$$
r_i = \eta \left(
\omega_1 \frac{R_{\mathrm{sec}}}{R_{\mathrm{sec}}^{\mathrm{norm}}}
+ \omega_2 \frac{S_{\mathrm{T}}}{S_{\mathrm{T}}^{\mathrm{norm}}}
- \omega_3 \frac{\mathrm{MSE}_{\mathrm{bp}}}{\mathrm{MSE}^{\mathrm{norm}}}
- \omega_4 \frac{\Phi_i}{\Phi^{\mathrm{norm}}}
\right),
$$

其中 $R_{\mathrm{sec}}$ 为系统保密和速率（各用户保密速率非负部分之和），$S_{\mathrm{T}}=\log_2(1+\Gamma_{\mathrm{T}}^{\mathrm{ITS}})$（交通场可调制感知 SNR），$\mathrm{MSE}_{\mathrm{bp}}$ 为角度格上相对期望波束的误差；$\Phi_i$ 含 $E_D,E_I$ 超限、飞行时隙违约及 `TRAFFIC_UAV_RISK_PENALTY` 等；$\eta=$ `REWARD_SCALE`。最后对 $r_i$ 做 `nan_to_num` 与幅值截断。

折扣因子 $\gamma$ 在 `config.GAMMA`（默认 0.99），用于 DRL 更新。

## 功能概览

- **`environment.py`**： IRS-UAV ISAC 环境（ZF 预编码、人工噪声、感知波束指标、流量场与可选 imperfect CSI 等）。
- **`agents.py`**：学习型智能体  
  **PPO-SAC**、**PPO-SAC-GAE**、**Attention PPO-SAC（APPO-SAC）**、**Hybrid-Action SAC**、**DDPG-DQN**。
- **`networks.py`**：与上述算法配套的神经网络模块（含注意力编码器版本）。
- **`config.py`**：信道几何、OTFS-OFDMA、奖励权重、DRL 超参数等（默认值见文件内说明，部分项可通过环境变量微调）。
- **`train.py`**：多阶段训练脚本（收敛曲线、IRS 规模扫描、功率扫描），结果写入本目录下的 `results.pkl`。

## 环境依赖

- Python 3.8+（建议）
- 见 `requirements.txt`：`numpy`、`PyTorch`

安装示例：

```bash
pip install -r requirements.txt
```

若有 GPU，请按 [PyTorch 官网](https://pytorch.org/) 选择匹配的 CUDA 版本安装。

## 目录结构（简要）

```
security_optimization_irs_uav_its/
  README.md
  requirements.txt
  __init__.py
  config.py
  environment.py
  agents.py
  networks.py
  train.py
  figures/          # 随代码发布的示意图与示例结果图
    MANIFEST.txt
```

## 运行训练

**请在本目录的上一级**（即包含 `security_optimization_irs_uav_its` 文件夹的仓库根目录）执行：

```bash
# 完整实验（耗时较长）
python -m security_optimization_irs_uav_its.train

# 快速冒烟测试（更少 episode / 更小扫描集合）
python -m security_optimization_irs_uav_its.train --quick
```

等价写法：

```bash
python security_optimization_irs_uav_its/train.py
python security_optimization_irs_uav_its/train.py --quick
```

训练结束后会在 `security_optimization_irs_uav_its/results.pkl` 中保存汇总结果。

## 在代码中使用环境与智能体

```python
from security_optimization_irs_uav_its import ISACEnv, PPOSACAgent, AttentionPPOSACAgent

env = ISACEnv(seed=42)
agent = AttentionPPOSACAgent(
    env.state_dim, env.cont_dim, env.disc_n_I, env.disc_n_D
)
s = env.reset()
aI, aD, cont, _ = agent.select_action(s)
s2, r, done, info = env.step(aI, aD, cont)
```

## 配置说明（节选）

`config.py` 中部分行为可通过环境变量覆盖，例如：

| 环境变量 | 含义（简述） |
|----------|----------------|
| `ISAC_SHOWCASE_IRS` | IRS 与直射链路对比强度（默认偏“展示”模式） |
| `ISAC_COMPLEX_TRAFFIC` | 是否启用道路/热点等交通场模型 |
| `ISAC_BASELINE_IMPERFECT_ZF_CSI` | 与 `ISACEnv(..., imperfect_zf_csi=True)` 配合的 ZF CSI 误差开关与强度 |

详细注释见 `config.py` 与 `environment.py` 文件内说明。

## 随附图片（`figures/`）

见 `figures/MANIFEST.txt`，当前包含：

- 系统/场景示意图：`system_model.jpg`
- 算法框架图：`algorithm_framework.png`
- 训练收敛示意：`training_convergence.png`
- ISAC 归一化波束图：`isac_beampattern_detection.png`

## 开源与引用

使用本仓库请注明来源与仓库链接。需要许可证时请在仓库根目录自行添加 `LICENSE`。

## 说明

- **不包含** AO / Equal-Power 等非学习基线智能体。
- 其他作图与后处理若存在于上层工程目录（如 `simulation/`），可自行对接。
