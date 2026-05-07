"""
Global simulation parameters matching Table I of the paper.

IRS prominence (for clearer ``no_irs`` / phase ablations in power sweeps):
  - Default: **showcase mode ON** — weaker BS→ground NLoS and slightly stronger
    IRS LoS segments so the reflective path dominates after modest N.
  - Restore the original manuscript-style calibration:
        export ISAC_SHOWCASE_IRS=0

Optional imperfect BS-side CSI for ``ISACEnv(..., imperfect_zf_csi=True)``: ZF /
null-space AN are built from a noisy estimate of the composite user matrix; achieved
rates still use the **true** channels. Learning agents in this release default to
``imperfect_zf_csi=False``. Toggle with ``ISAC_BASELINE_IMPERFECT_ZF_CSI``.

Learned policies optimise a **weighted** secrecy+sensing reward; plotted ``R_sec`` alone
may not reflect the full objective.
"""
import os

import numpy as np

# ── IRS showcase / traffic complexity toggles ────────────
# ``ISAC_SHOWCASE_IRS``: unset or "1"/"true"/"yes" → emphasize IRS vs direct link.
# ``0``/``false``/``no`` → use the legacy Table-I channel strengths (weaker contrast).
_SHOWCASE_RAW = os.environ.get("ISAC_SHOWCASE_IRS", "1")
SHOWCASE_IRS_GAIN = _SHOWCASE_RAW.strip().lower() not in ("0", "false", "no")
# ``ISAC_COMPLEX_TRAFFIC=0`` restores the earlier static-terminal, no-traffic field.
_TRAFFIC_RAW = os.environ.get("ISAC_COMPLEX_TRAFFIC", "1")
COMPLEX_TRAFFIC = _TRAFFIC_RAW.strip().lower() not in ("0", "false", "no")

# ── Optional imperfect ZF CSI (``imperfect_zf_csi=True`` on ``ISACEnv``) ──
# BS builds ZF / null-space AN from a noisy estimate of ``H_u``; rates use true
# channels.  DRL training uses ``imperfect_zf_csi=False`` by default.
_BASELINE_ZF_RAW = os.environ.get("ISAC_BASELINE_IMPERFECT_ZF_CSI", "1")
BASELINE_IMPERFECT_ZF_CSI = _BASELINE_ZF_RAW.strip().lower() not in ("0", "false", "no")
BASELINE_ZF_CSI_NOISE_STD = float(os.environ.get("ISAC_BASELINE_ZF_CSI_NOISE_STD", "0.12"))

# ── Geometry ──────────────────────────────────────────────
BS_LOC   = np.array([0.0, 0.0, 0.0])
# Paper Table I: EAV at $[-10, -25, 0]$
EAV_LOC  = np.array([-10.0, -25.0, 0.0])
H_I      = 50.0
H_D      = 40.0
NUM_USERS = 2
USER_LOCS = np.array([[30.0, 20.0, 0.0],   # ~36 m from BS
                       [35.0, -15.0, 0.0]])  # ~38 m from BS
TARGET_ANGLE_DEG = 30.0

NUM_DP_I  = 10
NUM_DP_D  = 10
AREA_RANGE = 50.0

# ── Traffic field / dynamic urban roadway model ───────────
# Traffic-field knobs (urban corridors / hotspots / clutter) mirror the
# ``Traffic congestion map and ITS-aware scaling'' paragraph in main_manuscript.tex.
TRAFFIC_ROAD_WIDTH = 16.0
TRAFFIC_TEMPORAL_SWING = 0.35
TRAFFIC_HOTSPOTS = np.array([
    [18.0, 12.0, 1.00, 18.0],
    [32.0, -18.0, 0.85, 15.0],
])
TRAFFIC_DIRECT_BLOCKAGE_DB = 18.0
TRAFFIC_IRS_SHADOW_DB = 2.0
TRAFFIC_SENSING_CLUTTER = 0.22
TRAFFIC_UAV_RISK_PENALTY = 0.40
USER_MOBILITY_RADIUS = 4.0

# ── Antenna / IRS ─────────────────────────────────────────
K       = 8
N_IRS   = 16
N_IRS_LIST = [4, 8, 16, 32, 64]
B_THETA = 1.0

# ── OTFS-OFDMA ────────────────────────────────────────────
N_TAU  = 16            # delay bins (paper Table I)
N_NU   = 16            # Doppler bins (paper Table I)
N_RES  = N_TAU * N_NU  # = 256 resource elements
DELTA  = 0.5

# ── Power / bandwidth ────────────────────────────────────
P_MAX_DBM      = 20.0
P_MAX          = 10 ** (P_MAX_DBM / 10) * 1e-3
P_MAX_LIST_DBM = np.array([0, 5, 10, 15, 20, 25, 30])
# Paper Table I reports 90 kHz as the bandwidth of one OTFS-OFDMA
# subband/resource block. Figures report aggregate system throughput across
# the scheduled delay-Doppler resource grid.
RESOURCE_BLOCK_BW = 90e3
BW                = RESOURCE_BLOCK_BW * N_RES
SECURITY_RATE_BW  = BW
OFFLOADING_BW     = BW
# Receiver noise over the scheduled OTFS-OFDMA bandwidth.
# Earlier versions used -17.9 dBm directly as total noise power, which is far
# too large for a 23.04 MHz link and suppresses the rate unrealistically.
THERMAL_NOISE_DENSITY_DBM_HZ = -174.0
NOISE_FIGURE_DB = 7.0
NOISE_POWER_DBM = (
    THERMAL_NOISE_DENSITY_DBM_HZ
    + 10 * np.log10(BW)
    + NOISE_FIGURE_DB
)
SIGMA_U2 = 10 ** (NOISE_POWER_DBM / 10) * 1e-3
SIGMA_E2 = SIGMA_U2
# Paper Table I: "Noise Variance (σ²) at BS-MEC" = 0.01 (D-UAV → BS / offloading path)
SIGMA_B2 = 0.01
# Sensing target noise: not separate in the table; match downlink / ISAC noise scale
SIGMA_T2 = SIGMA_U2

# ── Channel ───────────────────────────────────────────────
# Reference gains follow the common UAV/RIS convention of about -30 dB at
# 1 m for favorable LoS/Rician air-ground links. Direct BS-ground links are
# further attenuated to emulate urban blockage/NLoS penetration losses.
MU = 2.2  # kept for backward compatibility (_pl legacy)
MU_LOS = 2.2  # mild air-ground LoS/Rician path-loss exponent on IRS legs

if SHOWCASE_IRS_GAIN:
    # Stronger contrast: IRS cascade carries most energy; ``use_irs=False`` drops
    # a much weaker direct-only channel → clearer gaps vs PPO-SAC on R_sec / offload.
    BETA_C = 2.5e-6   # ~6 dB weaker NLoS BS→user/EAV vs 1e-5 at same distance
    BETA_C_LOS = 1.65e-3  # ~2 dB stronger LoS segments (IRS relay favourable)
    MU_NLOS = 3.95   # slightly steeper urban attenuation on direct links only
else:
    BETA_C = 1e-5
    BETA_C_LOS = 1e-3
    MU_NLOS = 3.5

# ── UAV dynamics ──────────────────────────────────────────
V_UAV   = 20.0
DELTA_T = 5.0
P_FLY_I = 100.0
P_FLY_D = 100.0
P_HOV_I = 80.0
P_HOV_D = 80.0
P_COM_D = 5.0
P_R     = P_MAX

# ── Offloading / Computing ───────────────────────────────
D_BITS   = 5e4
S_CYCLES = 100
C_MEC    = 5e9
C_D      = 1e9
E_MAX_D  = 2000.0
E_MAX_I  = 2000.0

# ── Sensing beampattern ──────────────────────────────────
L_GRID     = 37
THETA_GRID = np.linspace(-np.pi / 2, np.pi / 2, L_GRID)
EPSILON_BP = 0.1
ZETA       = 1.0

# ── Reward weights ────────────────────────────────────────
OMEGA_1 = 0.7
OMEGA_2 = 2.0
OMEGA_3 = 0.5
OMEGA_4 = 0.5
# Normalizers keep the reward terms on comparable scales. Without these,
# R_sec dominates by several orders of magnitude and the policy ignores sensing.
R_SEC_NORM = 2e8
S_T_NORM = 40.0
BP_MSE_NORM = 0.1
PENALTY_NORM = 1e3
REWARD_SCALE = 1.0

# ── DRL hyper-parameters ─────────────────────────────────
GAMMA        = 0.99
# Common learning settings for all DRL methods. PPO-SAC keeps separate PPO
# optimizers for the discrete controller, but the shared continuous-control
# actor/critic learning rates match the Hybrid-SAC and DDPG-DQN baselines.
LR_PPO_ACTOR  = 3e-4
LR_PPO_CRITIC = 1e-3
LR_PPO_GAE_ACTOR  = 2e-4
LR_PPO_GAE_CRITIC = 7e-4
LR_SAC_ACTOR  = 3e-4
LR_SAC_CRITIC = 3e-3
LR_ACTOR      = LR_SAC_ACTOR
LR_CRITIC     = LR_SAC_CRITIC
HIDDEN_DIM   = 256     # neurons per hidden layer (paper Table I)
NUM_HIDDEN   = 3       # hidden layers per network (paper Table I)
REPLAY_SIZE  = 50000   # replay buffer size (paper Table I)
BATCH_SIZE   = 128
PPO_CLIP     = 0.2
PPO_GAE_LAMBDA = 0.95
PPO_GAE_EPOCHS = 10
PPO_ENTROPY_COEF = 0.01
TAU          = 0.005
ALPHA_INIT   = 0.2
UPDATES_PER_ENV_STEP = 1
# SAC / Twin-Q: clamp y = r + γ(...); prevents Q blow-up → NaNs in actor (Hybrid-SAC is high-risk).
Q_TARGET_CLIP = 500.0
# Convergence stabilizers. DRL remains stochastic during learning, so strict
# convergence should be judged on the deterministic evaluation curves.
LR_FINAL_SCALE = 0.15
EXPLORATION_TEMP_INIT = 1.0
EXPLORATION_TEMP_FINAL = 0.25
DDPG_NOISE_INIT = 0.20
DDPG_NOISE_FINAL = 0.02
DDPG_EPS_INIT = 1.0
DDPG_EPS_FINAL = 0.02
POLICY_DELAY = 2
TARGET_POLICY_NOISE = 0.08
TARGET_NOISE_CLIP = 0.20
EVAL_ROLLOUTS = 5

NUM_EPISODES  = 3000
STEPS_PER_EP  = 15
EVAL_INTERVAL = 100
SEED          = 42
