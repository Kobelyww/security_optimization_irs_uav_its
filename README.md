Simplified Chinese: [`readme_ZH.md`](readme_ZH.md).

# Security Optimization of Communication for IRS-UAV Enabled Intelligent Transportation System

Simulation code package **`security_optimization_irs_uav_its`**.

This directory provides an **integrated sensing and communication (ISAC)** simulation environment in an **IRS-assisted multi-UAV** setting, together with **hybrid discrete/continuous-action** deep reinforcement learning (DRL) agents for experiments on **joint secure communication and radar sensing** resource allocation.

## Modeling highlights

- **Network topology**: A multi-antenna BS-MEC serves ground legitimate users; a **D-UAV** collects and offloads data; an **I-UAV** carries an IRS and assists both the **D-UAV→BS-MEC offloading link** and **downlink ISAC**; an eavesdropper (EAV) and a sensing target are present.
- **Waveform and transmission**: Downlink uses **OTFS-OFDMA** with multi-user ISAC on a delay–Doppler resource grid. The transmit signal includes **multi-user active beamforming** and **artificial noise (AN)**. At the BS, the simulator uses **ZF precoding** and **sensing-aware null-space AN** to target user SINRs while limiting leakage to the eavesdropper and shaping the transmit covariance.
- **IRS and channels**: IRS phase shifts reshape cascaded channels. Links distinguish **LoS-dominated IRS-assisted paths** from **NLoS-dominated direct ground paths**. An optional **urban/road congestion traffic field** modulates direct loss, IRS shadowing, and sensing clutter as functions of UAV positions.
- **MEC and energy**: **Partial offloading ratios** are coupled with flight/hover energy and per-slot feasibility, and appear as penalty terms in the reward.
- **Sensing and ISAC metrics**: Target-direction **sensing utility** (e.g., log-type in $`S_{\mathrm{T}}`$) and **MSE of the transmit beampattern versus a desired pattern** encode radar-side performance so secrecy and sensing are **jointly optimized**, not communication rate alone.
- **Hybrid actions**: **Discrete**—I-UAV / D-UAV **deployment** over candidate sites; **continuous**—power split, IRS phase, AN power share, offloading ratio, etc.; the state stacks **heterogeneous** summaries of channels, IRS alignment, geometry, and optional traffic.

## Technical features

- **Problem**: Under an IRS–UAV ITS setup, **UAV placement, OTFS-OFDMA-side equivalent control, IRS phase, AN and power allocation, and MEC offloading** share one **ISAC + physical-layer security** objective with explicit **secrecy and beam/sensing quality** terms.
- **Methods**: **Attention PPO (APPO-SAC)** with **feature attention**; **hybrid actions** from a **discrete deployment head** and a **continuous SAC branch** (beam shares, IRS, AN, offloading); optional **GAE** for PPO stability.
- **Waveform and scenario**: **OTFS-OFDMA ISAC**; **two UAVs + IRS**; optional **congestion field** modulating channels and sensing clutter.
- **Experiments**: Training curves and sweeps can be compared with several DRL baselines (Hybrid-SAC, PPO-SAC, DDPG-DQN, etc.). **This repository does not** include non-learning baselines such as AO or Equal-Power.

## MDP: state, action, and reward

This section matches observations, actions, and rewards in **`environment.py`**. The state uses **statistical summaries** of high-dimensional channels.

### State space $`\mathcal{S}`$

**Symbolic state (high level)** at slot $`i`$, aggregating channel, sensing, traffic, and history:

```math
s_i = \lbrace \{\mathbf{h}_{D,I}[i]\},\{\mathbf{G}_{B,I}[i]\},\{\mathbf{H}_{u_m}[i]\},\mathbf{H}_{E}[i],\mathbf{H}_{\mathrm{T}}[i],\varrho(Q_I[i],i),\varrho(Q_D[i],i),\{Q_{u_m}[i]\},\{\varrho(Q_{u_m}[i],i)\},L(\zeta,\mathbf{R}_{\mathbf{x}_B}[i]),a_{i-1},r_i \rbrace,
```

Here $`\varrho(\cdot,i)`$ is the traffic congestion field; $`L(\zeta,\mathbf{R}_{\mathbf{x}_B}[i])`$ is mismatch vs. a desired beampattern (in code, beampattern MSE enters the **reward** rather than a full per-entry expansion in the state); $`a_{i-1}, r_i`$ memorize the previous action and instantaneous reward.

**Observation vector** from `env._get_state()` **concatenates** the following **real features** (complex channels are compressed by magnitude/phase **statistics**, plus IRS **alignment hints**):

1. Per user $`m`$: **mean magnitude and mean phase** of the legitimate composite channel $`\mathbf{H}_{u_m}`$ with $`\mathbf{\Theta}=\mathbf{I}`$ (1 + 1 dims, $`2M`$ total);  
2. Eavesdropper channel $`\mathbf{H}_E`$: **mean magnitude and mean phase** (2 dims);  
3. Sensing response $`\mathbf{H}_{\mathrm{T}}`$: **mean magnitude** (1 dim);  
4. Offloading effective channel $`| \mathbf{h}_{I,B}^H \mathbf{\Theta}\,\mathbf{h}_{D,I} |`$ at zero phase (1 dim);  
5. Per-user IRS **per-element alignment** $`\angle(\mathbf{s}_{\mathrm{IRS},m}) - \angle(\mathbf{s}_{\mathrm{IRS,BI}})`$ scaled to $`[-1,1]`$ ($`M N_{\mathrm{IRS}}`$ dims); eavesdropper alignment ($`N_{\mathrm{IRS}}`$ dims);  
6. I-UAV / D-UAV **horizontal positions** $`q_I, q_D`$ normalized by `AREA_RANGE` (4 dims);  
7. If **complex traffic** is on (`COMPLEX_TRAFFIC=1`): $`\varrho`$ at both UAVs, each user’s (normalized) position and its $`\varrho`$ ($`2 + 3M`$ dims);  
8. Slot progress $`i / T_{\mathrm{ep}}`$ (1 dim) and **previous reward** $`r_{i-1}`$ (1 dim).

Let $`M=`$ `NUM_USERS`, $`N_I=`$ `NUM_DP_I`, $`N_D=`$ `NUM_DP_D`, $`N=`$ `n_irs`. Then:

```math
d_s = \underbrace{2M + 4 + (M+1)N + 4 + 2}_{\text{traffic-independent part}} + \underbrace{(2 + 3M)\cdot \mathbb{1}_{\text{COMPLEX}}}_{\text{optional}} = 2M + (M+1)N + 10 + (2+3M)\cdot \mathbb{1}_{\text{COMPLEX}}.
```

With default $`M=2, N=16`$ and no complex traffic, $`d_s=62`$ (matches `env.state_dim`). See also `env.cont_dim`, `env.disc_n_I`, `env.disc_n_D`.

### Action space $`\mathcal{A}`$

**Full RF joint action** at slot $`i`** (per-RE beam matrices, covariance, etc.):

```math
a_i = \lbrace \mathbf{q}_I[i],\mathbf{q}_D[i],\{\mathbf{W}[g,i]\}_{g=1}^{N_\tau N_\nu},\mathbf{\Theta}_n[i],\alpha[i],\mathbf{R}_z[i] \rbrace.
```

**Environment API** (consistent with PPO-SAC-style agents):

- **Discrete**: $`a^d_i = (d_I,d_D)`$ with $`d_I\in\{0,\ldots,N_I-1\}`$, $`d_D\in\{0,\ldots,N_D-1\}`$, indices into `dp_I`, `dp_D`.  
- **Continuous**: $`\mathbf{u}_i\in[-1,1]^{d_c}`$, $`d_c = M + N + 2`$ (`env.cont_dim`). `decode_action` maps to:

  - **User power shares**: $`\mathbf{p}_f = \mathrm{softmax}(\mathrm{clip}(\mathbf{u}_{1:M}))`$ (used with total power / AN split in `_build_beamforming`);  
  - **IRS phase**: $`\phi_n = \pi(\mathbf{u}_{M:M+N}+1) \in [0,2\pi]`$;  
  - **AN power fraction**: $`f_{\mathrm{AN}} = \mathrm{clip}((u_{M+N}+1)/2,\,[0.05,0.5])`$;  
  - **Offloading ratio**: $`\alpha = \mathrm{clip}((u_{d_c}+1)/2,\,[0.05,0.95])`$.

Downlink **ZF beams** and **null-space AN covariance** are **generated analytically** in the environment from $`(\mathbf{p}_f,\phi,f_{\mathrm{AN}})`$ and current channels; the network does not output every entry of $`\mathbf{W}[g]`$, $`\mathbf{R}_z`$, which keeps the continuous dimension small.

### Reward

**Instantaneous reward (weighted)**:

```math
r_i=\omega_1\sum_{m=1}^{M}R^{\mathrm{Sec}}_{u_m}[i]+\omega_2 S_{\mathrm{T}}[i]-\omega_3 L(\zeta,\mathbf{R}_{\mathbf{x}_B}[i])-\omega_4 \Phi_i,
```

where $`R^{\mathrm{Sec}}_{u_m}`$ is the non-negative excess of legitimate rate over eavesdropping rate, $`S_{\mathrm{T}}`$ is sensing utility, $`L`$ is beampattern mismatch, and $`\Phi_i`$ penalizes energy, timing/flight feasibility, and optional traffic risk.

**In code** (`config.py`: `OMEGA_*`, `R_SEC_NORM`, `S_T_NORM`, `BP_MSE_NORM`, `PENALTY_NORM`, `REWARD_SCALE`) a **normalized** form is used:

```math
r_i = \eta (
\omega_1 \frac{R_{\mathrm{sec}}}{R_{\mathrm{sec}}^{\mathrm{norm}}}
+ \omega_2 \frac{S_{\mathrm{T}}}{S_{\mathrm{T}}^{\mathrm{norm}}}
- \omega_3 \frac{\mathrm{MSE}_{\mathrm{bp}}}{\mathrm{MSE}^{\mathrm{norm}}}
- \omega_4 \frac{\Phi_i}{\Phi^{\mathrm{norm}}}
),
```

with $`R_{\mathrm{sec}}`$ the sum-secrecy rate (non-negative per-user secrecy rates), $`S_{\mathrm{T}}=\log_2(1+\Gamma_{\mathrm{T}}^{\mathrm{ITS}})`$ (sensing SNR can be modulated by the traffic field), $`\mathrm{MSE}_{\mathrm{bp}}`$ angular-grid error vs. a desired beam; $`\Phi_i`$ includes $`E_D,E_I`$ violations, flight/slot violations, and `TRAFFIC_UAV_RISK_PENALTY`; $`\eta=`$ `REWARD_SCALE`. $`r_i`$ is passed through `nan_to_num` and clipped.

Discount $`\gamma`$ is `config.GAMMA` (default 0.99) for DRL updates.

## Package overview

- **`environment.py`**: IRS-UAV ISAC environment (ZF, AN, sensing/beam metrics, traffic field, optional imperfect CSI).
- **`agents.py`**: **PPO-SAC**, **PPO-SAC-GAE**, **Attention PPO-SAC (APPO-SAC)**, **Hybrid-Action SAC**, **DDPG-DQN**.
- **`networks.py`**: Neural modules for the above (including attention encoders).
- **`config.py`**: Geometry, OTFS-OFDMA, reward weights, DRL hyperparameters (defaults and env-var overrides documented in-file).
- **`train.py`**: Multi-stage training (convergence, IRS-size sweep, power sweep); results saved to `results.pkl` in this folder.

## Requirements

- Python 3.8+ recommended
- See `requirements.txt`: `numpy`, `PyTorch`

Install:

```bash
pip install -r requirements.txt
```

For GPU, pick a matching CUDA build from the [PyTorch website](https://pytorch.org/).

## Layout

```
security_optimization_irs_uav_its/
  README.md
  readme_ZH.md
  requirements.txt
  __init__.py
  config.py
  environment.py
  agents.py
  networks.py
  train.py
  figures/          # diagrams and example result plots
    MANIFEST.txt
```

## Training

Run from the **parent directory of** `security_optimization_irs_uav_its` (repository root that contains this package):

```bash
# Full experiment (longer run)
python -m security_optimization_irs_uav_its.train

# Quick smoke test (fewer episodes / smaller sweeps)
python -m security_optimization_irs_uav_its.train --quick
```

Equivalent:

```bash
python security_optimization_irs_uav_its/train.py
python security_optimization_irs_uav_its/train.py --quick
```

Summaries are written to `security_optimization_irs_uav_its/results.pkl`.

## Using the environment and agents in code

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

## Configuration (excerpt)

Some `config.py` behavior can be overridden via environment variables:

| Variable | Role |
|----------|------|
| `ISAC_SHOWCASE_IRS` | Strength of IRS vs. direct link contrast (defaults to a “showcase” bias). |
| `ISAC_COMPLEX_TRAFFIC` | Enable road/hotspot-style traffic field model. |
| `ISAC_BASELINE_IMPERFECT_ZF_CSI` | ZF CSI error toggle/strength with `ISACEnv(..., imperfect_zf_csi=True)`. |

See comments in `config.py` and `environment.py`.

## Figures (`figures/`)

See `figures/MANIFEST.txt`. Currently includes:

- System/scenario sketch: `system_model.jpg`
- Algorithm block diagram: `algorithm_framework.png`
- Training convergence example: `training_convergence.png`
- ISAC normalized beampattern: `isac_beampattern_detection.png`

## License and citation

- **No `LICENSE` file is bundled**; add one at the repo root if you need explicit terms.
- If you use this code, please cite or credit the repository you obtained it from.

## Notes

- **Not included**: AO / Equal-Power and other **non-learning** baseline agents.
- Additional plotting or post-processing may live in a parent project (e.g. `simulation/`); wire them up as needed.
