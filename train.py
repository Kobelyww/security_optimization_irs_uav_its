"""
Training loop for all agents + parameter sweeps.

From repository root (parent of ``security_optimization_irs_uav_its``):

    python -m security_optimization_irs_uav_its.train
    python -m security_optimization_irs_uav_its.train --quick

Or:

    python security_optimization_irs_uav_its/train.py
"""
import os
import pickle
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np
import torch

from security_optimization_irs_uav_its.agents import (
    AttentionPPOSACAgent,
    DDPGDQNAgent,
    HybridSACAgent,
    PPOSACAgent,
    PPOSACGAEAgent,
)
from security_optimization_irs_uav_its.config import *
from security_optimization_irs_uav_its.environment import ISACEnv

OUT = _THIS_DIR


def set_training_progress(agent, ep, n_ep):
    if hasattr(agent, "set_training_progress"):
        agent.set_training_progress(ep / max(n_ep - 1, 1))


def make_eval_env(env):
    eval_env = ISACEnv(n_irs=env.n_irs, p_max=env.p_max, seed=SEED + 12345)
    eval_env.dp_I = env.dp_I.copy()
    eval_env.dp_D = env.dp_D.copy()
    return eval_env


def eval_agent_metrics(agent, env, n_ep=EVAL_ROLLOUTS):
    """Deterministic exploitation metrics on a fixed evaluation distribution."""
    vals = {
        "reward": [],
        "r_sec": [],
        "s_t": [],
        "bp_mse": [],
        "r_user_sum": [],
        "r_eav_sum": [],
        "r_sec_bps_hz": [],
    }
    eval_env = make_eval_env(env)
    for _ in range(n_ep):
        s = eval_env.reset()
        acc = {k: 0.0 for k in vals}
        for _ in range(STEPS_PER_EP):
            aI, aD, c, _ = agent.select_action(s, explore=False)
            s, r, d, info = eval_env.step(aI, aD, c)
            acc["reward"] += r
            acc["r_sec"] += info["R_sec"]
            acc["s_t"] += info["S_T"]
            acc["bp_mse"] += info["bp_mse"]
            acc["r_user_sum"] += info["R_user_sum"]
            acc["r_eav_sum"] += info["R_eav_sum"]
            acc["r_sec_bps_hz"] += info["R_sec_bps_hz"]
            if d:
                break
        for k in vals:
            vals[k].append(acc[k] / STEPS_PER_EP)
    return {k: float(np.mean(v)) for k, v in vals.items()}


def append_eval(eval_hist, ep, metrics):
    eval_hist["episode"].append(ep + 1)
    for k, v in metrics.items():
        eval_hist[k].append(v)


def new_eval_history():
    return {
        "episode": [],
        "reward": [],
        "r_sec": [],
        "s_t": [],
        "bp_mse": [],
        "r_user_sum": [],
        "r_eav_sum": [],
        "r_sec_bps_hz": [],
    }


def train_ppo_sac(env, agent, n_ep=NUM_EPISODES, tag="PPO-SAC", verbose=True):
    """Returns per-episode averages:
    reward, R_sec, S_T, bp_mse, R_user_sum, R_eav_sum, R_sec_bps_hz.
    """
    rh, rsh, sth, bph = [], [], [], []
    ruh, reh, rzh = [], [], []
    eval_hist = new_eval_history()
    for ep in range(n_ep):
        set_training_progress(agent, ep, n_ep)
        s = env.reset()
        er = ers = est = ebp = 0.0
        eru = ere = erhz = 0.0
        for _ in range(STEPS_PER_EP):
            aI, aD, c, lp = agent.select_action(s)
            s2, r, d, info = env.step(aI, aD, c)
            agent.store_ppo(s, aI, aD, lp, r, s2, d)
            agent.store_sac(s, c, r, s2, d)
            agent.update_sac()
            er += r
            ers += info["R_sec"]
            est += info["S_T"]
            ebp += info["bp_mse"]
            eru += info["R_user_sum"]
            ere += info["R_eav_sum"]
            erhz += info["R_sec_bps_hz"]
            s = s2
            if d:
                break
        agent.update_ppo()
        rh.append(er / STEPS_PER_EP)
        rsh.append(ers / STEPS_PER_EP)
        sth.append(est / STEPS_PER_EP)
        bph.append(ebp / STEPS_PER_EP)
        ruh.append(eru / STEPS_PER_EP)
        reh.append(ere / STEPS_PER_EP)
        rzh.append(erhz / STEPS_PER_EP)
        if verbose and (ep+1) % 500 == 0:
            print(f"  [{tag}] ep {ep+1}/{n_ep}  r={np.mean(rh[-100:]):.3f}  Rsec={np.mean(rsh[-100:]):.1f}")
        if (ep + 1) % EVAL_INTERVAL == 0 or ep == n_ep - 1:
            append_eval(eval_hist, ep, eval_agent_metrics(agent, env))
    return rh, rsh, sth, bph, ruh, reh, rzh, eval_hist


def train_hybrid(env, agent, n_ep=NUM_EPISODES, tag="Hybrid-SAC", verbose=True):
    """Returns per-episode averages (same tuple as train_ppo_sac)."""
    rh, rsh, sth, bph = [], [], [], []
    ruh, reh, rzh = [], [], []
    eval_hist = new_eval_history()
    for ep in range(n_ep):
        set_training_progress(agent, ep, n_ep)
        s = env.reset()
        er = ers = est = ebp = 0.0
        eru = ere = erhz = 0.0
        for _ in range(STEPS_PER_EP):
            aI, aD, c, _ = agent.select_action(s)
            s2, r, d, info = env.step(aI, aD, c)
            full = np.concatenate([c, np.eye(agent.n_I)[aI], np.eye(agent.n_D)[aD]])
            agent.store(s, full, r, s2, d)
            agent.update()
            er += r
            ers += info["R_sec"]
            est += info["S_T"]
            ebp += info["bp_mse"]
            eru += info["R_user_sum"]
            ere += info["R_eav_sum"]
            erhz += info["R_sec_bps_hz"]
            s = s2
            if d:
                break
        rh.append(er / STEPS_PER_EP)
        rsh.append(ers / STEPS_PER_EP)
        sth.append(est / STEPS_PER_EP)
        bph.append(ebp / STEPS_PER_EP)
        ruh.append(eru / STEPS_PER_EP)
        reh.append(ere / STEPS_PER_EP)
        rzh.append(erhz / STEPS_PER_EP)
        if verbose and (ep+1) % 500 == 0:
            print(f"  [{tag}] ep {ep+1}/{n_ep}  r={np.mean(rh[-100:]):.3f}")
        if (ep + 1) % EVAL_INTERVAL == 0 or ep == n_ep - 1:
            append_eval(eval_hist, ep, eval_agent_metrics(agent, env))
    return rh, rsh, sth, bph, ruh, reh, rzh, eval_hist


def train_ddpg(env, agent, n_ep=NUM_EPISODES, tag="DDPG-DQN", verbose=True):
    """Returns per-episode averages (same tuple as train_ppo_sac)."""
    rh, rsh, sth, bph = [], [], [], []
    ruh, reh, rzh = [], [], []
    eval_hist = new_eval_history()
    for ep in range(n_ep):
        set_training_progress(agent, ep, n_ep)
        s = env.reset()
        er = ers = est = ebp = 0.0
        eru = ere = erhz = 0.0
        for _ in range(STEPS_PER_EP):
            aI, aD, c, _ = agent.select_action(s)
            s2, r, d, info = env.step(aI, aD, c)
            agent.store(s, c, aI, aD, r, s2, d)
            agent.update()
            er += r
            ers += info["R_sec"]
            est += info["S_T"]
            ebp += info["bp_mse"]
            eru += info["R_user_sum"]
            ere += info["R_eav_sum"]
            erhz += info["R_sec_bps_hz"]
            s = s2
            if d:
                break
        rh.append(er / STEPS_PER_EP)
        rsh.append(ers / STEPS_PER_EP)
        sth.append(est / STEPS_PER_EP)
        bph.append(ebp / STEPS_PER_EP)
        ruh.append(eru / STEPS_PER_EP)
        reh.append(ere / STEPS_PER_EP)
        rzh.append(erhz / STEPS_PER_EP)
        if verbose and (ep+1) % 500 == 0:
            print(f"  [{tag}] ep {ep+1}/{n_ep}  r={np.mean(rh[-100:]):.3f}")
        if (ep + 1) % EVAL_INTERVAL == 0 or ep == n_ep - 1:
            append_eval(eval_hist, ep, eval_agent_metrics(agent, env))
    return rh, rsh, sth, bph, ruh, reh, rzh, eval_hist


def make_agents(env):
    return {
        "ppo_sac": (train_ppo_sac,
                    PPOSACAgent(env.state_dim, env.cont_dim, env.disc_n_I, env.disc_n_D)),
        "ppo_sac_gae": (train_ppo_sac,
                        PPOSACGAEAgent(env.state_dim, env.cont_dim, env.disc_n_I, env.disc_n_D)),
        "attn_ppo_sac": (train_ppo_sac,
                         AttentionPPOSACAgent(env.state_dim, env.cont_dim, env.disc_n_I, env.disc_n_D)),
        "hybrid":  (train_hybrid,
                    HybridSACAgent(env.state_dim, env.cont_dim, env.disc_n_I, env.disc_n_D)),
        "ddpg":    (train_ddpg,
                    DDPGDQNAgent(env.state_dim, env.cont_dim, env.disc_n_I, env.disc_n_D)),
    }


# ═══════════════════════════════════════════════════════════
if __name__ == "__main__":
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    quick = "--quick" in sys.argv
    CONV_EP  = 800  if quick else NUM_EPISODES
    SWEEP_EP = 500  if quick else 1500
    n_list   = [4, 16, 24]  if quick else N_IRS_LIST
    p_list   = [10, 20, 30] if quick else list(P_MAX_LIST_DBM)

    results = {}

    # ── 1  Convergence ────────────────────────────────────
    print("\n=== Phase 1: Convergence ===")
    env = ISACEnv(seed=SEED)
    ag  = PPOSACAgent(env.state_dim, env.cont_dim, env.disc_n_I, env.disc_n_D)
    rh, rsh, *_ = train_ppo_sac(env, ag, CONV_EP)
    results["conv_reward"] = rh

    # deployment snapshot
    s = env.reset(); dI_list, dD_list = [], []
    for _ in range(STEPS_PER_EP):
        aI, aD, c, _ = ag.select_action(s, explore=False)
        dI_list.append(env.dp_I[aI].tolist())
        dD_list.append(env.dp_D[aD].tolist())
        s, _, d, _ = env.step(aI, aD, c)
        if d:
            break
    results["dep_I"]     = dI_list
    results["dep_D"]     = dD_list
    results["dp_I_cand"] = env.dp_I.tolist()
    results["dp_D_cand"] = env.dp_D.tolist()

    # ── 2  N_IRS sweep ────────────────────────────────────
    print("\n=== Phase 2: R_SEC vs N_IRS ===")
    results["n_irs_list"] = n_list
    for name in ["ppo_sac", "hybrid", "ddpg"]:
        vals = []
        for n in n_list:
            e = ISACEnv(n_irs=n, seed=SEED)
            entries = make_agents(e)
            trainer, agent = entries[name]
            result = trainer(e, agent, SWEEP_EP, tag=f"{name}-N{n}")
            rsh = result[1]
            vals.append(float(np.mean(rsh[-100:])))
        results[f"n_irs_{name}"] = vals

    # ── 3  Power sweep ────────────────────────────────────
    print("\n=== Phase 3: R_SEC vs Power ===")
    results["p_list"] = p_list
    for name in ["ppo_sac", "hybrid", "ddpg"]:
        vals = []
        for pdbm in p_list:
            pw = 10**(pdbm/10)*1e-3
            e = ISACEnv(p_max=pw, seed=SEED)
            entries = make_agents(e)
            trainer, agent = entries[name]
            result = trainer(e, agent, SWEEP_EP, tag=f"{name}-P{pdbm}")
            rsh = result[1]
            vals.append(float(np.mean(rsh[-100:])))
        results[f"p_{name}"] = vals

    with open(os.path.join(OUT, "results.pkl"), "wb") as f:
        pickle.dump(results, f)
    print(f"\nSaved results.pkl")
