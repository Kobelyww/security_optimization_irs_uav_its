"""
IRS-UAV ISAC environment with ZF beamforming built-in.
Agent controls: power allocation, IRS phases, AN fraction, offloading ratio.
Continuous action dim = M + n_irs + 2  (typically ~20).

Set ``use_irs=False`` to simulate **no reflective IRS paths**: user/eavesdropper
channels retain only the direct BS→terminal NLOS links; IRS-assisted offloading
rate is zero (baseline vs IRS-aided power sweeps).

When ``imperfect_zf_csi=True``, the BS builds ZF / null-space AN from a noisy
estimate of the composite user matrix ``H_u`` (see ``BASELINE_ZF_CSI_*`` in
``config``); achieved rates still use the **true** channels (precoder mismatch).

The default environment also includes a lightweight urban-traffic field:
road/intersection density creates direct-link blockage, sensing clutter, UAV risk
penalty, and small road-side user mobility. Disable with
``ISAC_COMPLEX_TRAFFIC=0`` for the earlier static setting.
"""
import numpy as np

from .config import *


def steering_vector(theta, k=K, delta=DELTA):
    n = np.arange(k)
    return np.exp(1j * 2 * np.pi * delta * np.sin(theta) * n).reshape(-1, 1)


def desired_beampattern(target_theta, grid=THETA_GRID, bw_rad=np.deg2rad(20)):
    return np.where(np.abs(grid - target_theta) <= bw_rad / 2, 1.0, 0.0)


class RunningMeanStd:
    def __init__(self, shape):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var  = np.ones(shape, dtype=np.float64)
        self.count = 1e-4

    def update(self, x):
        batch_mean = np.mean(x, axis=0) if x.ndim > 1 else x
        batch_var  = np.var(x, axis=0) if x.ndim > 1 else np.zeros_like(x)
        batch_count = x.shape[0] if x.ndim > 1 else 1
        delta = batch_mean - self.mean
        total = self.count + batch_count
        self.mean += delta * batch_count / total
        self.var = (self.var * self.count + batch_var * batch_count +
                    delta ** 2 * self.count * batch_count / total) / total
        self.count = total

    def normalize(self, x):
        return (x - self.mean) / (np.sqrt(self.var) + 1e-8)


class ISACEnv:
    def __init__(
        self,
        n_irs=N_IRS,
        p_max=P_MAX,
        seed=SEED,
        *,
        use_irs: bool = True,
        imperfect_zf_csi: bool = False,
    ):
        self.rng = np.random.RandomState(seed)
        self.n_irs = n_irs
        self.p_max = p_max
        self.use_irs = bool(use_irs)
        # When True, ZF / null-space AN are built from noisy estimates of H_u; rates
        # are still evaluated with the true composite channels (precoder mismatch).
        self.imperfect_zf_csi = bool(imperfect_zf_csi)
        self.M  = NUM_USERS
        self.K  = K

        self.dp_I = self._gen_dp(NUM_DP_I, H_I)
        self.dp_D = self._gen_dp(NUM_DP_D, H_D)
        self.target_theta = np.deg2rad(TARGET_ANGLE_DEG)
        self.P_d = desired_beampattern(self.target_theta)
        self.a_target = steering_vector(self.target_theta, self.K)
        self.user_locs = USER_LOCS.astype(float).copy()
        self._traffic_phase = 0.0

        # action: [power_frac_user_1..M, IRS_phases_1..n_irs, an_frac, alpha]
        self.cont_dim = self.M + self.n_irs + 2
        self.disc_n_I = NUM_DP_I
        self.disc_n_D = NUM_DP_D

        self._sample_fading()   # initialise before first reset/build
        self.reset()
        self.state_dim = len(self._get_state())
        self.obs_rms = RunningMeanStd(self.state_dim)

    def _gen_dp(self, n, h):
        xy = self.rng.uniform(-AREA_RANGE, AREA_RANGE, (n, 2))
        z  = np.full((n, 1), h)
        return np.hstack([xy, z])

    # ── channels ──────────────────────────────────────────
    def _dist(self, a, b):
        return max(np.linalg.norm(a - b), 1.0)

    def _pl(self, d):
        """Generic path loss (kept for backward compatibility)."""
        return np.sqrt(BETA_C * d ** (-MU))

    def _pl_los(self, d):
        """LoS path loss for IRS links (BS↔IRS, IRS↔node).
        Uses BETA_C_LOS (includes IRS element gain / favorable LoS conditions).
        """
        return np.sqrt(BETA_C_LOS * d ** (-MU_LOS))

    def _pl_nlos(self, d):
        """NLOS path loss for direct links (BS↔user, BS↔EAV).
        Urban NLOS with exponent MU_NLOS (see ``config``). Used only for direct
        BS→user / BS→eavesdropper legs; IRS-assisted legs use ``_pl_los``.
        """
        return np.sqrt(BETA_C * d ** (-MU_NLOS))

    def _traffic_density(self, xy, progress=None):
        """Dimensionless traffic density at an xy point (roads + intersections)."""
        if not COMPLEX_TRAFFIC:
            return 0.0
        xy = np.asarray(xy, dtype=float)[:2]
        if progress is None:
            progress = float(self.i / max(STEPS_PER_EP, 1))
        # Horizontal and diagonal arterial-road corridors.
        road_y = np.exp(-0.5 * (xy[1] / TRAFFIC_ROAD_WIDTH) ** 2)
        road_diag = np.exp(-0.5 * ((xy[0] + xy[1] - 18.0) / TRAFFIC_ROAD_WIDTH) ** 2)
        density = 0.35 + 0.35 * road_y + 0.30 * road_diag
        for hx, hy, amp, width in TRAFFIC_HOTSPOTS:
            d2 = (xy[0] - hx) ** 2 + (xy[1] - hy) ** 2
            density += amp * np.exp(-0.5 * d2 / (width ** 2))
        wave = 1.0 + TRAFFIC_TEMPORAL_SWING * np.sin(2 * np.pi * progress + self._traffic_phase)
        return float(np.clip(density * wave, 0.0, 1.6))

    def _link_traffic_density(self, a, b):
        mid = 0.5 * (np.asarray(a, dtype=float)[:2] + np.asarray(b, dtype=float)[:2])
        return self._traffic_density(mid)

    @staticmethod
    def _db_to_amp_loss(loss_db):
        return 10.0 ** (-float(loss_db) / 20.0)

    def _update_mobile_users(self):
        """Move ground users slightly along road directions in the traffic model."""
        if not COMPLEX_TRAFFIC:
            self.user_locs = USER_LOCS.astype(float).copy()
            return
        progress = float(self.i / max(STEPS_PER_EP, 1))
        locs = USER_LOCS.astype(float).copy()
        for m in range(self.M):
            phase = self._traffic_phase + 0.7 * m
            locs[m, 0] += USER_MOBILITY_RADIUS * np.sin(2 * np.pi * progress + phase)
            locs[m, 1] += 0.5 * USER_MOBILITY_RADIUS * np.cos(2 * np.pi * progress + phase)
        self.user_locs = locs

    def _sample_fading(self):
        """Sample quasi-static LoS fading phases once per episode.

        Each IRS link is modelled as a rank-1 LoS channel:
            G_BI = pl * a_IRS(theta) * a_BS(theta)^H
        represented by two random unit steering vectors.
        With optimal IRS phases the coherent gain scales as N (not sqrt(N)),
        matching the standard LoS IRS channel assumption in the literature.
        """
        n, k, m = self.n_irs, self.K, self.M
        u = self.rng.uniform
        # IRS ↔ BS link  (N steering on IRS side, K steering on BS side)
        self._sv_IRS_BI = np.exp(1j * u(0, 2*np.pi, n))   # IRS steering (BS–IRS)
        self._sv_BS_BI  = np.exp(1j * u(0, 2*np.pi, k))   # BS  steering (BS–IRS)
        # IRS ↔ user m
        self._sv_IRS_Iu = [np.exp(1j * u(0, 2*np.pi, n)) for _ in range(m)]
        self._sv_BS_um  = [np.exp(1j * u(0, 2*np.pi, k)) for _ in range(m)]
        # IRS ↔ EAV
        self._sv_IRS_IE = np.exp(1j * u(0, 2*np.pi, n))
        self._sv_BS_E   = np.exp(1j * u(0, 2*np.pi, k))
        # D-UAV → IRS offload path
        self._sv_DI     = np.exp(1j * u(0, 2*np.pi, n))
        self._sv_IB     = np.exp(1j * u(0, 2*np.pi, n))

    def _build_channels(self):
        """Rebuild channels with separate LoS (IRS) and NLOS (direct) models.

        IRS links  (BS↔IRS, IRS↔user, IRS↔EAV): rank-1 LoS, _pl_los().
          G_BI = pl_BI_los * sv_IRS_BI * sv_BS_BI^H  (N×K)
          With optimal IRS phases → coherent N-fold gain in amplitude.

        Direct links (BS→user, BS→EAV): NLOS, _pl_nlos() with MU_NLOS from config.
          Urban blockage weakens these relative to the cascaded IRS path (especially
          when ``SHOWCASE_IRS_GAIN`` is enabled in ``config``).

        At current geometry (users at ~36 m, I-UAV at ~50 m),
        N_crit ≈ 5 → IRS dominates for N≥8 in the sweep range [4,24].
        """
        d_BI = self._dist(BS_LOC, self.q_I)
        d_DI = self._dist(self.q_D, self.q_I)
        pl_BI_los = self._pl_los(d_BI)    # LoS: BS↔IRS

        # Rank-1 LoS: G_BI[i,k] = pl_BI_los * sv_IRS[i] * sv_BS[k]^*
        self.G_BI = pl_BI_los * np.outer(self._sv_IRS_BI,
                                          self._sv_BS_BI.conj())  # (n_irs, K)

        self.h_Ium, self.h_Bum = [], []
        for m in range(self.M):
            user_loc = self.user_locs[m]
            d_Ium = self._dist(self.q_I, user_loc)
            d_Bum = self._dist(BS_LOC, user_loc)
            den_direct = self._link_traffic_density(BS_LOC, user_loc)
            den_irs = self._link_traffic_density(self.q_I, user_loc)
            self.h_Ium.append(
                self._pl_los(d_Ium)
                * self._db_to_amp_loss(TRAFFIC_IRS_SHADOW_DB * den_irs)
                * self._sv_IRS_Iu[m].reshape(-1, 1)
            )
            self.h_Bum.append(
                self._pl_nlos(d_Bum)
                * self._db_to_amp_loss(TRAFFIC_DIRECT_BLOCKAGE_DB * den_direct)
                * self._sv_BS_um[m].reshape(-1, 1)
            )

        self.h_DI = self._pl_los(d_DI) * self._sv_DI.reshape(-1, 1)
        self.h_IB = pl_BI_los           * self._sv_IB.reshape(-1, 1)

        d_IE = self._dist(self.q_I, EAV_LOC)
        d_BE = self._dist(BS_LOC,   EAV_LOC)
        den_ie = self._link_traffic_density(self.q_I, EAV_LOC)
        den_be = self._link_traffic_density(BS_LOC, EAV_LOC)
        self.h_IE = (
            self._pl_los(d_IE)
            * self._db_to_amp_loss(TRAFFIC_IRS_SHADOW_DB * den_ie)
            * self._sv_IRS_IE.reshape(-1, 1)
        )
        self.h_BE = (
            self._pl_nlos(d_BE)
            * self._db_to_amp_loss(TRAFFIC_DIRECT_BLOCKAGE_DB * den_be)
            * self._sv_BS_E.reshape(-1, 1)
        )

        self.H_T = steering_vector(self.target_theta, self.K).conj().T

    def _Theta(self, phases):
        return np.diag(B_THETA * np.exp(1j * phases))

    def _H_user(self, m, Th):
        if not self.use_irs:
            return self.h_Bum[m].conj().T
        return (self.h_Ium[m].conj().T @ Th @ self.G_BI + self.h_Bum[m].conj().T)

    def _H_eav(self, Th):
        if not self.use_irs:
            return self.h_BE.conj().T
        return (self.h_IE.conj().T @ Th @ self.G_BI + self.h_BE.conj().T)

    def _perturb_zf_channel_observation(self, H_u: np.ndarray) -> np.ndarray:
        """Imperfect BS-side CSI: Frobenius-normalised complex Gaussian mismatch."""
        if not self.imperfect_zf_csi:
            return H_u
        scale = BASELINE_ZF_CSI_NOISE_STD
        fro = np.linalg.norm(H_u, "fro") + 1e-12
        E = (
            self.rng.standard_normal(H_u.shape)
            + 1j * self.rng.standard_normal(H_u.shape)
        ) / np.sqrt(2.0)
        en = np.linalg.norm(E, "fro") + 1e-12
        return H_u + E * (scale * fro / en)

    # ── ZF beamforming + sensing-aware null-space AN ─────
    def _build_beamforming(self, Th, power_fracs, an_frac):
        """Zero-Forcing beamforming with sensing-aware null-space AN.

        ZF eliminates inter-user interference (unlike MRT whose SINR saturates
        at K/M with more users):
            H_u[j] @ W_ZF[:,m] = 0  for j ≠ m

        Combined with target-directed null-space AN (H_u @ Rz @ H_u^H ≈ 0):
          - User SINR ∝ p_sig / σ²  → grows unboundedly with p_max
          - EAV SINR  is bounded by AN power  → R_eav saturates
          - Sensing energy is concentrated toward the target steering vector
          → R_sec = R_user − R_eav  increases with p_max  ✓

        With LoS IRS + NLOS direct (N_crit ≈ 5), R_sec also grows with N ✓

        When ``self.imperfect_zf_csi`` is True, ZF uses a noisy estimate of H_u;
        SINR / secrecy are still computed with the **true** composite channels.
        """
        p_sig = self.p_max * (1.0 - an_frac)
        p_an  = self.p_max * an_frac

        # Stack user channels  H_u ∈ ℂ^{M × K}
        H_u = np.vstack([self._H_user(m, Th) for m in range(self.M)])
        H_zf = self._perturb_zf_channel_observation(H_u)

        # ZF pseudo-inverse:  W_ZF = H_u^H (H_u H_u^H)^{-1}  (K × M)
        HHH = H_zf @ H_zf.conj().T + 1e-12 * np.eye(self.M)
        try:
            W_ZF = H_zf.conj().T @ np.linalg.inv(HHH)   # K × M
        except np.linalg.LinAlgError:
            W_ZF = H_zf.conj().T                          # fallback to MRT

        W = np.zeros((self.K, self.M), dtype=complex)
        for m in range(self.M):
            w    = W_ZF[:, m:m+1]
            norm = np.linalg.norm(w) + 1e-12
            W[:, m:m+1] = w / norm * np.sqrt(p_sig * power_fracs[m])

        # Null-space projection:  P_null = I − H^H (H H^H)^{-1} H.
        # Shape AN along the target steering vector after projection, so the
        # same covariance supports sensing while remaining nearly invisible to users.
        P_null = np.eye(self.K) - H_zf.conj().T @ np.linalg.solve(HHH, H_zf)
        P_null = 0.5 * (P_null + P_null.conj().T)
        target_null = P_null @ self.a_target
        target_norm = np.linalg.norm(target_null)
        if target_norm > 1e-8:
            v_t = target_null / target_norm
            Rz = p_an * (v_t @ v_t.conj().T)  # trace(Rz) = p_an
        else:
            tr_P = max(float(np.real(np.trace(P_null))), 1e-10)
            Rz = P_null * (p_an / tr_P)
        return W, Rz

    # ── metrics ───────────────────────────────────────────
    def _sinr(self, H, W, Rz, m):
        sig = float(np.abs(H @ W[:, m:m+1]) ** 2)
        itf = sum(float(np.abs(H @ W[:, j:j+1]) ** 2)
                  for j in range(self.M) if j != m)
        noise = float(np.real(H @ Rz @ H.conj().T)) + SIGMA_U2
        return sig / (itf + noise + 1e-15)

    def _sinr_eav(self, H_E, W, Rz, m):
        sig = float(np.abs(H_E @ W[:, m:m+1]) ** 2)
        itf = sum(float(np.abs(H_E @ W[:, j:j+1]) ** 2)
                  for j in range(self.M) if j != m)
        noise = float(np.real(H_E @ Rz @ H_E.conj().T)) + SIGMA_E2
        return sig / (itf + noise + 1e-15)

    def _secrecy_rates_detail(self, Th, W, Rz):
        """Legitimate sum-rate, eavesdrop sum-rate, secrecy rate, and spectral form.

        Uses the same OTFS-grid bandwidth multiplier SECURITY_RATE_BW as in the
        paper setup. Per-user rates use the MISO BC model: stream m decoded at
        user m and at the eavesdropper with respective SINRs.
        """
        H_E = self._H_eav(Th)
        bw = SECURITY_RATE_BW
        r_user_list = []
        r_eav_list = []
        r_sec = 0.0
        for m in range(self.M):
            H_um = self._H_user(m, Th)
            r_u = bw * np.log2(1 + self._sinr(H_um, W, Rz, m))
            r_e = bw * np.log2(1 + self._sinr_eav(H_E, W, Rz, m))
            r_user_list.append(float(r_u))
            r_eav_list.append(float(r_e))
            r_sec += max(r_u - r_e, 0.0)
        r_sec = float(r_sec)
        r_user_sum = float(sum(r_user_list))
        r_eav_sum = float(sum(r_eav_list))
        return {
            "R_sec": r_sec,
            "R_user_sum": r_user_sum,
            "R_eav_sum": r_eav_sum,
            "R_sec_bps_hz": float(r_sec / bw),
            "R_user_per_user": r_user_list,
            "R_eav_per_user": r_eav_list,
        }

    def _secrecy_rate_total(self, Th, W, Rz):
        return self._secrecy_rates_detail(Th, W, Rz)["R_sec"]

    def _sensing_utility(self, Rx):
        gamma = float(np.real(self.H_T @ Rx @ self.H_T.conj().T)) / SIGMA_T2
        if COMPLEX_TRAFFIC:
            density = 0.5 * (
                self._traffic_density(self.q_I[:2])
                + self._traffic_density(self.q_D[:2])
            )
            gamma /= 1.0 + TRAFFIC_SENSING_CLUTTER * density
        return float(np.log2(1 + gamma))

    def _beampattern_mse(self, Rx):
        mse = 0.0
        for l, th in enumerate(THETA_GRID):
            a = steering_vector(th, self.K)
            p_bp = float(np.real(a.conj().T @ Rx @ a))
            mse += (ZETA * self.P_d[l] - p_bp) ** 2
        return mse / L_GRID

    def _offloading_rate(self, Th):
        if not self.use_irs:
            return 0.0
        eff = self.h_IB.conj().T @ Th @ self.h_DI
        snr = P_R * float(np.abs(eff) ** 2) / SIGMA_B2
        return float(OFFLOADING_BW * np.log2(1 + snr))

    def _energy(self, alpha, Th):
        R_DB = self._offloading_rate(Th)
        T_off = alpha * D_BITS / max(R_DB, 1.0)
        T_com_mec = alpha * D_BITS * S_CYCLES / C_MEC
        T_com_d   = (1 - alpha) * D_BITS * S_CYCLES / C_D
        T_fly_I = self._dist(self.prev_q_I, self.q_I) / V_UAV
        T_fly_D = self._dist(self.prev_q_D, self.q_D) / V_UAV
        T_wait  = abs(T_fly_I - T_fly_D)
        flag = 1.0 if T_fly_I <= T_fly_D else 0.0
        E_D = T_fly_D * P_FLY_D + (1-flag)*T_wait*P_HOV_D + (P_HOV_D+P_R)*T_off + (P_COM_D+P_HOV_D)*T_com_d
        E_I = T_fly_I * P_FLY_I + flag*T_wait*P_HOV_I + P_HOV_I*T_off + P_HOV_I*T_com_mec
        fly_ok = (T_fly_I < DELTA_T) and (T_fly_D < DELTA_T)
        return E_D, E_I, fly_ok

    # ── action decoding ───────────────────────────────────
    def decode_action(self, disc_I, disc_D, cont):
        """cont in [-1,1]^cont_dim.  NaN-safe."""
        cont = np.nan_to_num(np.clip(np.asarray(cont, dtype=np.float64), -1, 1))

        pf_raw = cont[:self.M]
        pf_raw = np.clip(pf_raw, -5, 5)
        pf = np.exp(pf_raw) / (np.sum(np.exp(pf_raw)) + 1e-8)

        phases = (cont[self.M:self.M + self.n_irs] + 1) * np.pi
        phases = np.clip(phases, 0, 2 * np.pi)

        # `pf` is only the relative allocation among users. The signal/AN split
        # is applied once in `_build_beamforming`; applying it here as well
        # double-counts AN and biases the policy toward the minimum AN fraction.
        an_frac = float(np.clip((cont[self.M + self.n_irs] + 1) / 2, 0.05, 0.5))

        alpha = float(np.clip((cont[-1] + 1) / 2, 0.05, 0.95))

        q_I = self.dp_I[disc_I]
        q_D = self.dp_D[disc_D]
        return pf, phases, an_frac, alpha, q_I, q_D

    # ── state ─────────────────────────────────────────────
    def _get_state(self):
        Th0 = np.eye(self.n_irs, dtype=complex)
        feats = []

        # Global channel features (path-loss + mean phase)
        for m in range(self.M):
            H_um = self._H_user(m, Th0)
            feats.extend([float(np.abs(H_um).mean()), float(np.angle(H_um).mean())])
        H_E = self._H_eav(Th0)
        feats.extend([float(np.abs(H_E).mean()), float(np.angle(H_E).mean())])
        feats.append(float(np.abs(self.H_T).mean()))
        eff_off = self.h_IB.conj().T @ Th0 @ self.h_DI
        feats.append(float(np.abs(eff_off)))

        # Per-element IRS alignment phases (key for learning optimal IRS policy):
        #   align_m[i] = angle(sv_IRS_Iu[m][i]) − angle(sv_IRS_BI[i])
        # This is the exact optimal IRS phase for element i for user m.
        # Providing it in the state lets the agent directly learn to mimic it.
        for m in range(self.M):
            align = np.angle(self._sv_IRS_Iu[m] * np.conj(self._sv_IRS_BI))
            feats.extend(list(align / np.pi))   # normalised to [-1, 1]
        # EAV alignment (agent learns to misalign IRS w.r.t. eavesdropper)
        align_e = np.angle(self._sv_IRS_IE * np.conj(self._sv_IRS_BI))
        feats.extend(list(align_e / np.pi))

        # UAV positions and episode progress
        feats.extend([float(x) for x in self.q_I[:2] / AREA_RANGE])
        feats.extend([float(x) for x in self.q_D[:2] / AREA_RANGE])
        if COMPLEX_TRAFFIC:
            feats.append(float(self._traffic_density(self.q_I[:2])))
            feats.append(float(self._traffic_density(self.q_D[:2])))
            for m in range(self.M):
                feats.extend([float(x) for x in self.user_locs[m, :2] / AREA_RANGE])
                feats.append(float(self._traffic_density(self.user_locs[m, :2])))
        feats.append(float(self.i / STEPS_PER_EP))
        feats.append(float(self.prev_reward))

        arr = np.array(feats, dtype=np.float32)
        return np.nan_to_num(arr, nan=0.0, posinf=1e6, neginf=-1e6)

    # ── reset / step ──────────────────────────────────────
    def reset(self):
        self.q_I = self.dp_I[self.rng.randint(NUM_DP_I)].copy()
        self.q_D = self.dp_D[self.rng.randint(NUM_DP_D)].copy()
        self.prev_q_I = self.q_I.copy()
        self.prev_q_D = self.q_D.copy()
        self.i = 0
        self.prev_reward = 0.0
        self._traffic_phase = self.rng.uniform(0.0, 2.0 * np.pi)
        self._update_mobile_users()
        self._sample_fading()   # draw quasi-static fading once per episode
        self._build_channels()
        return self._get_state()

    def step(self, disc_I, disc_D, cont):
        pf, phases, an_frac, alpha, q_I_new, q_D_new = self.decode_action(
            disc_I, disc_D, cont)

        self.prev_q_I = self.q_I.copy()
        self.prev_q_D = self.q_D.copy()
        self.q_I = q_I_new
        self.q_D = q_D_new
        self._update_mobile_users()
        self._build_channels()

        Th = self._Theta(phases)
        W, Rz = self._build_beamforming(Th, pf, an_frac)

        Rx = np.zeros((self.K, self.K), dtype=complex)
        for m in range(self.M):
            Rx += W[:, m:m+1] @ W[:, m:m+1].conj().T
        Rx += Rz

        rates = self._secrecy_rates_detail(Th, W, Rz)
        R_sec = rates["R_sec"]
        S_T    = self._sensing_utility(Rx)
        bp_mse = self._beampattern_mse(Rx)
        E_D, E_I, fly_ok = self._energy(alpha, Th)

        penalty = 0.0
        if E_D > E_MAX_D:
            penalty += (E_D - E_MAX_D) / E_MAX_D
        if E_I > E_MAX_I:
            penalty += (E_I - E_MAX_I) / E_MAX_I
        if not fly_ok:
            penalty += 1.0
        if COMPLEX_TRAFFIC:
            traffic_risk = 0.5 * (
                self._traffic_density(self.q_I[:2])
                + self._traffic_density(self.q_D[:2])
            )
            penalty += TRAFFIC_UAV_RISK_PENALTY * traffic_risk

        r_sec_term = R_sec / R_SEC_NORM
        sensing_term = S_T / S_T_NORM
        bp_term = bp_mse / BP_MSE_NORM
        penalty_term = penalty / PENALTY_NORM

        reward = (OMEGA_1 * r_sec_term + OMEGA_2 * sensing_term
                  - OMEGA_3 * bp_term - OMEGA_4 * penalty_term) * REWARD_SCALE
        reward = float(np.nan_to_num(reward, nan=0.0, posinf=1e4, neginf=-1e4))
        reward = np.clip(reward, -1e4, 1e4)

        self.prev_reward = reward
        self.i += 1
        if COMPLEX_TRAFFIC:
            # The metrics above use the current-slot geometry. Refresh channels
            # for the next observation after advancing the traffic clock.
            self._update_mobile_users()
            self._build_channels()
        done = self.i >= STEPS_PER_EP
        info = {
            "R_sec": R_sec,
            "R_sec_bps_hz": rates["R_sec_bps_hz"],
            "R_user_sum": rates["R_user_sum"],
            "R_eav_sum": rates["R_eav_sum"],
            "R_user_per_user": rates["R_user_per_user"],
            "R_eav_per_user": rates["R_eav_per_user"],
            "S_T": S_T,
            "bp_mse": bp_mse,
            "E_D": E_D,
            "E_I": E_I,
            "penalty": penalty,
        }
        return self._get_state(), float(reward), done, info
