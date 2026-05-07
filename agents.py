"""
Learning agents for mixed discrete/continuous control (ISAC / IRS-UAV setting).

Included:
  - PPO-SAC: PPO for discrete UAV deployment, SAC for continuous controls.
  - PPO-SAC-GAE: PPO branch uses GAE advantages.
  - Attention PPO-SAC (APPO-SAC): feature-attention encoders in PPO and SAC.
  - Hybrid-SAC: one Gaussian actor over concatenated continuous + discrete logits.
  - DDPG-DQN: deterministic continuous actor, double DQN heads for deployment.

All algorithms share replay/batch sizes, target update rate, and per-step
update budget (see ``config.py``).
"""
import copy
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F

from .config import (
    ALPHA_INIT,
    BATCH_SIZE,
    DDPG_EPS_FINAL,
    DDPG_EPS_INIT,
    DDPG_NOISE_FINAL,
    DDPG_NOISE_INIT,
    EXPLORATION_TEMP_FINAL,
    EXPLORATION_TEMP_INIT,
    GAMMA,
    LR_ACTOR,
    LR_CRITIC,
    LR_FINAL_SCALE,
    LR_PPO_ACTOR,
    LR_PPO_CRITIC,
    LR_PPO_GAE_ACTOR,
    LR_PPO_GAE_CRITIC,
    LR_SAC_ACTOR,
    LR_SAC_CRITIC,
    POLICY_DELAY,
    PPO_CLIP,
    PPO_ENTROPY_COEF,
    PPO_GAE_EPOCHS,
    PPO_GAE_LAMBDA,
    Q_TARGET_CLIP,
    REPLAY_SIZE,
    TARGET_NOISE_CLIP,
    TARGET_POLICY_NOISE,
    TAU,
    UPDATES_PER_ENV_STEP,
)
from .networks import (
    BASELINE_HIDDEN,
    BASELINE_LAYERS,
    AttentionPPOActor,
    AttentionPPOCritic,
    AttentionSACActor,
    AttentionTwinQ,
    DDPGActor,
    DQN,
    PPOActor,
    PPOCritic,
    SACActor,
    TwinQ,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_GRAD = 1.0


def clip_and_step(opt, params, max_grad=MAX_GRAD):
    torch.nn.utils.clip_grad_norm_(params, max_grad)
    opt.step()


def anneal(init, final, progress):
    """Cosine-free monotone annealing used by all learning baselines."""
    p = float(np.clip(progress, 0.0, 1.0))
    return final + (init - final) * (1.0 - p)


class StabilizedAgentMixin:
    def _init_training_schedule(self, optimizers):
        self._optimizers = list(optimizers)
        self._base_lrs = [[g["lr"] for g in opt.param_groups] for opt in self._optimizers]
        self.explore_temp = EXPLORATION_TEMP_INIT

    def set_training_progress(self, progress):
        lr_scale = anneal(1.0, LR_FINAL_SCALE, progress)
        self.explore_temp = anneal(EXPLORATION_TEMP_INIT, EXPLORATION_TEMP_FINAL, progress)
        for opt, base_lrs in zip(self._optimizers, self._base_lrs):
            for group, base_lr in zip(opt.param_groups, base_lrs):
                group["lr"] = base_lr * lr_scale


class ReplayBuffer:
    def __init__(self, cap=REPLAY_SIZE):
        self.buf = deque(maxlen=cap)

    def push(self, *args):
        self.buf.append(args)

    def sample(self, n=BATCH_SIZE):
        idx = np.random.choice(len(self.buf), n, replace=False)
        batch = [self.buf[i] for i in idx]
        return [np.array(x, dtype=np.float32) for x in zip(*batch)]

    def __len__(self):
        return len(self.buf)


# ═══════════════════════════════════════════════════════════
#  PPO-SAC — full capacity, multi-step SAC
# ═══════════════════════════════════════════════════════════
class PPOSACAgent(StabilizedAgentMixin):
    SAC_UPDATES_PER_STEP = UPDATES_PER_ENV_STEP

    def __init__(self, s_dim, c_dim, n_I, n_D):
        self.ppo_a = PPOActor(s_dim, n_I, n_D).to(device)
        self.ppo_c = PPOCritic(s_dim).to(device)
        self.sac_a = SACActor(s_dim, c_dim).to(device)
        self.sac_q = TwinQ(s_dim, c_dim).to(device)
        self.sac_qt = copy.deepcopy(self.sac_q)

        self.opt_pa = torch.optim.Adam(self.ppo_a.parameters(), lr=LR_PPO_ACTOR)
        self.opt_pc = torch.optim.Adam(self.ppo_c.parameters(), lr=LR_PPO_CRITIC)
        self.opt_sa = torch.optim.Adam(self.sac_a.parameters(), lr=LR_SAC_ACTOR)
        self.opt_sq = torch.optim.Adam(self.sac_q.parameters(), lr=LR_SAC_CRITIC)

        self.log_alpha = torch.tensor(np.log(ALPHA_INIT), requires_grad=True, device=device)
        self.opt_al = torch.optim.Adam([self.log_alpha], lr=LR_SAC_ACTOR)
        self.target_entropy = -c_dim * 0.5

        self.buf = ReplayBuffer()
        self.ppo_mem = []
        self._init_training_schedule([
            self.opt_pa, self.opt_pc, self.opt_sa, self.opt_sq, self.opt_al
        ])

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def select_action(self, s, explore=True):
        st = torch.FloatTensor(s).unsqueeze(0).to(device)
        with torch.no_grad():
            aI, aD, lp = self.ppo_a.act(st, deterministic=not explore)
            if explore:
                cont, _ = self.sac_a.sample(st, temperature=self.explore_temp)
            else:
                cont = self.sac_a.deterministic(st)
        return int(aI), int(aD), cont.cpu().numpy().flatten(), float(lp)

    def store_ppo(self, s, aI, aD, lp, r, s2, d):
        self.ppo_mem.append((s, aI, aD, lp, r, s2, float(d)))

    def store_sac(self, s, c, r, s2, d):
        self.buf.push(s, c, r, s2, float(d))

    def update_ppo(self, epochs=6):
        if not self.ppo_mem:
            return
        s, aI, aD, olp, rew, s2, dn = zip(*self.ppo_mem)
        st   = torch.FloatTensor(np.array(s)).to(device)
        aIt  = torch.LongTensor(aI).to(device)
        aDt  = torch.LongTensor(aD).to(device)
        olpt = torch.FloatTensor(olp).to(device)
        rt   = torch.FloatTensor(rew).to(device)
        s2t  = torch.FloatTensor(np.array(s2)).to(device)
        dt   = torch.FloatTensor(dn).to(device)

        with torch.no_grad():
            tgt = rt + GAMMA * self.ppo_c(s2t) * (1 - dt)
        val = self.ppo_c(st)
        adv = (tgt - val).detach()
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        for _ in range(epochs):
            nlp, ent = self.ppo_a.evaluate(st, aIt, aDt)
            ratio = (nlp - olpt).exp()
            clip  = ratio.clamp(1 - PPO_CLIP, 1 + PPO_CLIP)
            la = -torch.min(ratio * adv, clip * adv).mean() - 0.01 * ent.mean()
            self.opt_pa.zero_grad(); la.backward()
            clip_and_step(self.opt_pa, self.ppo_a.parameters())

            lc = F.smooth_l1_loss(self.ppo_c(st), tgt)
            self.opt_pc.zero_grad(); lc.backward()
            clip_and_step(self.opt_pc, self.ppo_c.parameters())
        self.ppo_mem.clear()

    def update_sac(self):
        if len(self.buf) < BATCH_SIZE * 2:
            return
        for _ in range(self.SAC_UPDATES_PER_STEP):
            self._sac_step()

    def _sac_step(self):
        s, a, r, s2, d = self.buf.sample()
        st  = torch.FloatTensor(s).to(device)
        at  = torch.FloatTensor(a).to(device)
        rt  = torch.FloatTensor(r).unsqueeze(-1).to(device)
        s2t = torch.FloatTensor(s2).to(device)
        dt  = torch.FloatTensor(d).unsqueeze(-1).to(device)

        with torch.no_grad():
            a2, lp2 = self.sac_a.sample(s2t)
            q1t, q2t = self.sac_qt(s2t, a2)
            target = rt + GAMMA * (1 - dt) * (torch.min(q1t, q2t) - self.alpha * lp2)
            target = torch.clamp(target, -Q_TARGET_CLIP, Q_TARGET_CLIP)

        q1, q2 = self.sac_q(st, at)
        lq = F.smooth_l1_loss(q1, target) + F.smooth_l1_loss(q2, target)
        if not torch.isfinite(lq):
            return
        self.opt_sq.zero_grad(); lq.backward()
        clip_and_step(self.opt_sq, self.sac_q.parameters())

        an, lpn = self.sac_a.sample(st)
        q1n, q2n = self.sac_q(st, an)
        la = (self.alpha.detach() * lpn - torch.min(q1n, q2n)).mean()
        if not torch.isfinite(la):
            return
        self.opt_sa.zero_grad(); la.backward()
        clip_and_step(self.opt_sa, self.sac_a.parameters())

        al = -(self.log_alpha * (lpn.detach() + self.target_entropy)).mean()
        if not torch.isfinite(al):
            return
        self.opt_al.zero_grad(); al.backward(); self.opt_al.step()

        for p, tp in zip(self.sac_q.parameters(), self.sac_qt.parameters()):
            tp.data.copy_(TAU * p.data + (1 - TAU) * tp.data)


# ═══════════════════════════════════════════════════════════
#  PPO-SAC-GAE — enhanced PPO discrete controller, original PPO-SAC preserved
# ═══════════════════════════════════════════════════════════
class PPOSACGAEAgent(PPOSACAgent):
    """PPO-SAC variant with GAE advantages for the discrete deployment policy.

    The original PPOSACAgent is intentionally left unchanged so both versions
    can be trained and plotted side by side.
    """

    def __init__(self, s_dim, c_dim, n_I, n_D):
        super().__init__(s_dim, c_dim, n_I, n_D)
        for group in self.opt_pa.param_groups:
            group["lr"] = LR_PPO_GAE_ACTOR
        for group in self.opt_pc.param_groups:
            group["lr"] = LR_PPO_GAE_CRITIC
        self._init_training_schedule([
            self.opt_pa, self.opt_pc, self.opt_sa, self.opt_sq, self.opt_al
        ])

    def update_ppo(self, epochs=PPO_GAE_EPOCHS):
        if not self.ppo_mem:
            return
        s, aI, aD, olp, rew, s2, dn = zip(*self.ppo_mem)
        st   = torch.FloatTensor(np.array(s)).to(device)
        aIt  = torch.LongTensor(aI).to(device)
        aDt  = torch.LongTensor(aD).to(device)
        olpt = torch.FloatTensor(olp).to(device)
        rt   = torch.FloatTensor(rew).to(device)
        s2t  = torch.FloatTensor(np.array(s2)).to(device)
        dt   = torch.FloatTensor(dn).to(device)

        with torch.no_grad():
            val = self.ppo_c(st)
            next_val = self.ppo_c(s2t)
            deltas = rt + GAMMA * next_val * (1 - dt) - val
            adv = torch.zeros_like(rt)
            gae = torch.tensor(0.0, device=device)
            for t in reversed(range(len(rt))):
                gae = deltas[t] + GAMMA * PPO_GAE_LAMBDA * (1 - dt[t]) * gae
                adv[t] = gae
            ret = adv + val
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        for _ in range(epochs):
            nlp, ent = self.ppo_a.evaluate(st, aIt, aDt)
            ratio = (nlp - olpt).exp()
            clip = ratio.clamp(1 - PPO_CLIP, 1 + PPO_CLIP)
            la = -torch.min(ratio * adv, clip * adv).mean() - PPO_ENTROPY_COEF * ent.mean()
            if not torch.isfinite(la):
                return
            self.opt_pa.zero_grad(); la.backward()
            clip_and_step(self.opt_pa, self.ppo_a.parameters())

            lc = F.smooth_l1_loss(self.ppo_c(st), ret)
            if not torch.isfinite(lc):
                return
            self.opt_pc.zero_grad(); lc.backward()
            clip_and_step(self.opt_pc, self.ppo_c.parameters())
        self.ppo_mem.clear()


# ═══════════════════════════════════════════════════════════
#  APPO-SAC (proposed) — attention-guided PPO-SAC, feature-attention state encoder
# ═══════════════════════════════════════════════════════════
class AttentionPPOSACAgent(PPOSACAgent):
    """PPO-SAC with feature-attention encoders in both PPO and SAC branches."""

    def __init__(self, s_dim, c_dim, n_I, n_D):
        self.ppo_a = AttentionPPOActor(s_dim, n_I, n_D).to(device)
        self.ppo_c = AttentionPPOCritic(s_dim).to(device)
        self.sac_a = AttentionSACActor(s_dim, c_dim).to(device)
        self.sac_q = AttentionTwinQ(s_dim, c_dim).to(device)
        self.sac_qt = copy.deepcopy(self.sac_q)

        self.opt_pa = torch.optim.Adam(self.ppo_a.parameters(), lr=LR_PPO_ACTOR)
        self.opt_pc = torch.optim.Adam(self.ppo_c.parameters(), lr=LR_PPO_CRITIC)
        self.opt_sa = torch.optim.Adam(self.sac_a.parameters(), lr=LR_SAC_ACTOR)
        self.opt_sq = torch.optim.Adam(self.sac_q.parameters(), lr=LR_SAC_CRITIC)

        self.log_alpha = torch.tensor(np.log(ALPHA_INIT), requires_grad=True, device=device)
        self.opt_al = torch.optim.Adam([self.log_alpha], lr=LR_SAC_ACTOR)
        self.target_entropy = -c_dim * 0.5

        self.buf = ReplayBuffer()
        self.ppo_mem = []
        self._init_training_schedule([
            self.opt_pa, self.opt_pc, self.opt_sa, self.opt_sq, self.opt_al
        ])


# ═══════════════════════════════════════════════════════════
#  Hybrid-Action SAC — single SAC with mixed action space
# ═══════════════════════════════════════════════════════════
class HybridSACAgent(StabilizedAgentMixin):
    DISCRETE_TEMP = 0.5

    def __init__(self, s_dim, c_dim, n_I, n_D):
        self.n_I, self.n_D, self.c_dim = n_I, n_D, c_dim
        total = c_dim + n_I + n_D
        self.actor    = SACActor(s_dim, total).to(device)
        self.critic   = TwinQ(s_dim, total, hid=BASELINE_HIDDEN, n_layers=BASELINE_LAYERS).to(device)
        self.critic_t = copy.deepcopy(self.critic)
        self.opt_a  = torch.optim.Adam(self.actor.parameters(), lr=LR_ACTOR)
        self.opt_c  = torch.optim.Adam(self.critic.parameters(), lr=LR_CRITIC)
        self.log_alpha = torch.tensor(np.log(ALPHA_INIT), requires_grad=True, device=device)
        self.opt_al = torch.optim.Adam([self.log_alpha], lr=LR_ACTOR)
        # The discrete deployment code is projected to one-hot actions before
        # entering Q. Use the continuous dimensions for SAC entropy tuning.
        self.target_entropy = -c_dim * 0.5
        self.buf = ReplayBuffer()
        self._init_training_schedule([self.opt_a, self.opt_c, self.opt_al])

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def select_action(self, s, explore=True):
        st = torch.FloatTensor(s).unsqueeze(0).to(device)
        with torch.no_grad():
            if explore:
                a, _ = self.actor.sample(st, temperature=self.explore_temp)
            else:
                a = self.actor.deterministic(st)
        a = a.cpu().numpy().flatten()
        cont = a[:self.c_dim]
        dI = int(np.argmax(a[self.c_dim:self.c_dim + self.n_I]))
        dD = int(np.argmax(a[self.c_dim + self.n_I:]))
        return dI, dD, cont, 0.0

    def store(self, s, full_a, r, s2, d):
        self.buf.push(s, full_a, r, s2, float(d))

    def _project_action(self, a):
        """Match actor actions to the replay format: cont + one-hot I + one-hot D.

        The critic is trained with one-hot discrete deployment actions. Feeding
        raw tanh outputs during actor/target updates creates an off-distribution
        Q query and can make Hybrid-SAC drift downward late in training. The
        straight-through one-hot projection keeps the forward action valid while
        preserving a softmax gradient for the actor.
        """
        cont = a[:, :self.c_dim]
        i0 = self.c_dim
        d0 = self.c_dim + self.n_I
        logits_i = a[:, i0:d0] / self.DISCRETE_TEMP
        logits_d = a[:, d0:d0 + self.n_D] / self.DISCRETE_TEMP

        prob_i = F.softmax(logits_i, dim=-1)
        prob_d = F.softmax(logits_d, dim=-1)
        hard_i = F.one_hot(prob_i.argmax(dim=-1), self.n_I).to(prob_i.dtype)
        hard_d = F.one_hot(prob_d.argmax(dim=-1), self.n_D).to(prob_d.dtype)
        disc_i = prob_i + (hard_i - prob_i).detach()
        disc_d = prob_d + (hard_d - prob_d).detach()
        return torch.cat([cont, disc_i, disc_d], dim=-1)

    def update(self):
        if len(self.buf) < BATCH_SIZE * 2:
            return
        s, a, r, s2, d = self.buf.sample()
        st  = torch.FloatTensor(s).to(device)
        at  = torch.FloatTensor(a).to(device)
        rt  = torch.FloatTensor(r).unsqueeze(-1).to(device)
        s2t = torch.FloatTensor(s2).to(device)
        dt  = torch.FloatTensor(d).unsqueeze(-1).to(device)

        with torch.no_grad():
            a2, lp2 = self.actor.sample(s2t)
            a2_q = self._project_action(a2)
            q1t, q2t = self.critic_t(s2t, a2_q)
            target = rt + GAMMA * (1 - dt) * (torch.min(q1t, q2t) - self.alpha * lp2)
            target = torch.clamp(target, -Q_TARGET_CLIP, Q_TARGET_CLIP)
        q1, q2 = self.critic(st, at)
        lc = F.smooth_l1_loss(q1, target) + F.smooth_l1_loss(q2, target)
        if not torch.isfinite(lc):
            return
        self.opt_c.zero_grad(); lc.backward()
        clip_and_step(self.opt_c, self.critic.parameters())

        an, lpn = self.actor.sample(st)
        an_q = self._project_action(an)
        q1n, q2n = self.critic(st, an_q)
        la = (self.alpha.detach() * lpn - torch.min(q1n, q2n)).mean()
        if not torch.isfinite(la):
            return
        self.opt_a.zero_grad(); la.backward()
        clip_and_step(self.opt_a, self.actor.parameters())

        al = -(self.log_alpha * (lpn.detach() + self.target_entropy)).mean()
        if not torch.isfinite(al):
            return
        self.opt_al.zero_grad(); al.backward(); self.opt_al.step()

        for p, tp in zip(self.critic.parameters(), self.critic_t.parameters()):
            tp.data.copy_(TAU * p.data + (1 - TAU) * tp.data)


# ═══════════════════════════════════════════════════════════
#  DDPG-DQN — deterministic policy + ε-greedy (no entropy)
# ═══════════════════════════════════════════════════════════
class DDPGDQNAgent(StabilizedAgentMixin):
    def __init__(self, s_dim, c_dim, n_I, n_D):
        self.n_I, self.n_D = n_I, n_D
        self.actor    = DDPGActor(s_dim, c_dim).to(device)
        self.critic   = TwinQ(s_dim, c_dim, hid=BASELINE_HIDDEN, n_layers=BASELINE_LAYERS).to(device)
        self.actor_t  = copy.deepcopy(self.actor)
        self.critic_t = copy.deepcopy(self.critic)
        self.dqn      = DQN(s_dim, n_I, n_D).to(device)
        self.dqn_t    = copy.deepcopy(self.dqn)
        self.opt_a = torch.optim.Adam(self.actor.parameters(), lr=LR_ACTOR)
        self.opt_c = torch.optim.Adam(self.critic.parameters(), lr=LR_CRITIC)
        self.opt_d = torch.optim.Adam(self.dqn.parameters(), lr=LR_CRITIC)
        self.buf = ReplayBuffer()
        self.eps = DDPG_EPS_INIT
        self.noise_std = DDPG_NOISE_INIT
        self.update_step = 0
        self._init_training_schedule([self.opt_a, self.opt_c, self.opt_d])

    def set_training_progress(self, progress):
        super().set_training_progress(progress)
        self.eps = anneal(DDPG_EPS_INIT, DDPG_EPS_FINAL, progress)
        self.noise_std = anneal(DDPG_NOISE_INIT, DDPG_NOISE_FINAL, progress)

    def select_action(self, s, explore=True):
        st = torch.FloatTensor(s).unsqueeze(0).to(device)
        with torch.no_grad():
            cont = self.actor(st).cpu().numpy().flatten()
            qI, qD = self.dqn(st)
        if explore and np.random.rand() < self.eps:
            dI, dD = np.random.randint(self.n_I), np.random.randint(self.n_D)
        else:
            dI, dD = int(qI.argmax(-1)), int(qD.argmax(-1))
        if explore:
            cont = cont + np.random.normal(0, self.noise_std, cont.shape)
            cont = np.clip(cont, -1, 1)
        return dI, dD, cont, 0.0

    def store(self, s, c, aI, aD, r, s2, d):
        self.buf.push(s, c, aI, aD, r, s2, float(d))

    def update(self):
        if len(self.buf) < BATCH_SIZE * 2:
            return
        self.update_step += 1
        s, a, aI, aD, r, s2, d = self.buf.sample()
        st  = torch.FloatTensor(s).to(device)
        at  = torch.FloatTensor(a).to(device)
        aIt = torch.LongTensor(aI.astype(np.int64)).to(device)
        aDt = torch.LongTensor(aD.astype(np.int64)).to(device)
        rt  = torch.FloatTensor(r).unsqueeze(-1).to(device)
        s2t = torch.FloatTensor(s2).to(device)
        dt  = torch.FloatTensor(d).unsqueeze(-1).to(device)

        with torch.no_grad():
            a2 = self.actor_t(s2t)
            noise = torch.randn_like(a2).clamp(-TARGET_NOISE_CLIP, TARGET_NOISE_CLIP)
            a2 = (a2 + TARGET_POLICY_NOISE * noise).clamp(-1, 1)
            q1t, q2t = self.critic_t(s2t, a2)
            target = rt + GAMMA * (1 - dt) * torch.min(q1t, q2t)
            target = torch.clamp(target, -Q_TARGET_CLIP, Q_TARGET_CLIP)
        q1, q2 = self.critic(st, at)
        lc = F.smooth_l1_loss(q1, target) + F.smooth_l1_loss(q2, target)
        if not torch.isfinite(lc):
            return
        self.opt_c.zero_grad(); lc.backward()
        clip_and_step(self.opt_c, self.critic.parameters())

        if self.update_step % POLICY_DELAY == 0:
            an = self.actor(st)
            q1n, _ = self.critic(st, an)
            la = -q1n.mean()
            if not torch.isfinite(la):
                return
            self.opt_a.zero_grad(); la.backward()
            clip_and_step(self.opt_a, self.actor.parameters())

        with torch.no_grad():
            qI_next, qD_next = self.dqn(s2t)
            next_i = qI_next.argmax(-1, keepdim=True)
            next_d = qD_next.argmax(-1, keepdim=True)
            qI_t, qD_t = self.dqn_t(s2t)
            next_val = (
                qI_t.gather(1, next_i).squeeze(-1)
                + qD_t.gather(1, next_d).squeeze(-1)
            ) / 2
            dqn_tgt = rt.squeeze() + GAMMA * (1 - dt.squeeze()) * \
                      next_val
        qI, qD = self.dqn(st)
        q_disc = (
            qI.gather(1, aIt.view(-1, 1)).squeeze(-1)
            + qD.gather(1, aDt.view(-1, 1)).squeeze(-1)
        ) / 2
        ld = F.smooth_l1_loss(q_disc, dqn_tgt)
        self.opt_d.zero_grad(); ld.backward()
        clip_and_step(self.opt_d, self.dqn.parameters())

        if self.update_step % POLICY_DELAY == 0:
            for p, tp in zip(self.actor.parameters(), self.actor_t.parameters()):
                tp.data.copy_(TAU * p.data + (1 - TAU) * tp.data)
        for p, tp in zip(self.critic.parameters(), self.critic_t.parameters()):
            tp.data.copy_(TAU * p.data + (1 - TAU) * tp.data)
        if self.update_step % POLICY_DELAY == 0:
            for p, tp in zip(self.dqn.parameters(), self.dqn_t.parameters()):
                tp.data.copy_(TAU * p.data + (1 - TAU) * tp.data)
