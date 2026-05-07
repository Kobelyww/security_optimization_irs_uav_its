"""
Neural network modules.

All learning algorithms use the same hidden width/depth so comparisons reflect
algorithmic differences rather than unequal model capacity.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal

from .config import HIDDEN_DIM, NUM_HIDDEN

LOG_STD_MIN, LOG_STD_MAX = -5, 2

BASELINE_HIDDEN = HIDDEN_DIM
BASELINE_LAYERS = NUM_HIDDEN


def mlp(in_d, out_d, hid=HIDDEN_DIM, n=NUM_HIDDEN):
    layers = []
    prev = in_d
    for _ in range(n):
        layers += [nn.Linear(prev, hid), nn.ReLU()]
        prev = hid
    layers.append(nn.Linear(prev, out_d))
    return nn.Sequential(*layers)


# ═════════════ PPO-SAC (proposed, full capacity) ═════════════

class PPOActor(nn.Module):
    def __init__(self, s_dim, n_I, n_D):
        super().__init__()
        self.trunk = mlp(s_dim, HIDDEN_DIM)
        self.head_I = nn.Linear(HIDDEN_DIM, n_I)
        self.head_D = nn.Linear(HIDDEN_DIM, n_D)

    def forward(self, s):
        h = F.relu(self.trunk(s))
        return self.head_I(h), self.head_D(h)

    def act(self, s, deterministic=False):
        lI, lD = self.forward(s)
        dI, dD = Categorical(logits=lI), Categorical(logits=lD)
        if deterministic:
            aI, aD = lI.argmax(-1), lD.argmax(-1)
        else:
            aI, aD = dI.sample(), dD.sample()
        return aI, aD, dI.log_prob(aI) + dD.log_prob(aD)

    def evaluate(self, s, aI, aD):
        lI, lD = self.forward(s)
        dI, dD = Categorical(logits=lI), Categorical(logits=lD)
        return dI.log_prob(aI) + dD.log_prob(aD), dI.entropy() + dD.entropy()


class PPOCritic(nn.Module):
    def __init__(self, s_dim):
        super().__init__()
        self.net = mlp(s_dim, 1)
    def forward(self, s):
        return self.net(s).squeeze(-1)


class SACActor(nn.Module):
    def __init__(self, s_dim, a_dim, hid=HIDDEN_DIM, n_layers=NUM_HIDDEN):
        super().__init__()
        self.trunk = mlp(s_dim, hid, hid=hid, n=n_layers)
        self.mu   = nn.Linear(hid, a_dim)
        self.lstd = nn.Linear(hid, a_dim)

    def forward(self, s):
        h = F.relu(self.trunk(s))
        return self.mu(h), self.lstd(h).clamp(LOG_STD_MIN, LOG_STD_MAX)

    def sample(self, s, temperature=1.0):
        mu, lstd = self.forward(s)
        # Guard against exploding Q-training (mu NaN → Normal fails during long Hybrid-SAC runs).
        mu = torch.nan_to_num(mu, nan=0.0, posinf=20.0, neginf=-20.0)
        std = (lstd.exp() * temperature).clamp(min=1e-6, max=100.0)
        dist = Normal(mu, std, validate_args=False)
        x = dist.rsample()
        a = torch.tanh(x)
        lp = (dist.log_prob(x) - torch.log(1 - a.pow(2) + 1e-6)).sum(-1, keepdim=True)
        return a, lp

    def deterministic(self, s):
        mu, _ = self.forward(s)
        mu = torch.nan_to_num(mu, nan=0.0, posinf=20.0, neginf=-20.0)
        return torch.tanh(mu)


class TwinQ(nn.Module):
    def __init__(self, s_dim, a_dim, hid=HIDDEN_DIM, n_layers=NUM_HIDDEN):
        super().__init__()
        self.q1 = mlp(s_dim + a_dim, 1, hid=hid, n=n_layers)
        self.q2 = mlp(s_dim + a_dim, 1, hid=hid, n=n_layers)
    def forward(self, s, a):
        sa = torch.cat([s, a], -1)
        return self.q1(sa), self.q2(sa)


# ═════════════ Attention-guided PPO-SAC modules ═════════════

class FeatureAttentionEncoder(nn.Module):
    """Lightweight feature-attention encoder for vector observations.

    The environment state is already a structured vector.  A feature-wise gate
    learns which channel, IRS-alignment, traffic, and sensing entries should be
    emphasized before the shared hidden representation is formed.
    """
    def __init__(self, s_dim, hid=HIDDEN_DIM):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(s_dim, hid),
            nn.ReLU(),
            nn.Linear(hid, s_dim),
        )
        self.value = nn.Linear(s_dim, hid)
        self.norm = nn.LayerNorm(hid)

    def forward(self, s):
        w = F.softmax(self.gate(s), dim=-1)
        attended = s * w * s.shape[-1]
        return F.relu(self.norm(self.value(attended)))


class AttentionPPOActor(nn.Module):
    def __init__(self, s_dim, n_I, n_D):
        super().__init__()
        self.encoder = FeatureAttentionEncoder(s_dim, HIDDEN_DIM)
        self.trunk = mlp(HIDDEN_DIM, HIDDEN_DIM, hid=HIDDEN_DIM, n=max(NUM_HIDDEN - 1, 1))
        self.head_I = nn.Linear(HIDDEN_DIM, n_I)
        self.head_D = nn.Linear(HIDDEN_DIM, n_D)

    def forward(self, s):
        h = F.relu(self.trunk(self.encoder(s)))
        return self.head_I(h), self.head_D(h)

    def act(self, s, deterministic=False):
        lI, lD = self.forward(s)
        dI, dD = Categorical(logits=lI), Categorical(logits=lD)
        if deterministic:
            aI, aD = lI.argmax(-1), lD.argmax(-1)
        else:
            aI, aD = dI.sample(), dD.sample()
        return aI, aD, dI.log_prob(aI) + dD.log_prob(aD)

    def evaluate(self, s, aI, aD):
        lI, lD = self.forward(s)
        dI, dD = Categorical(logits=lI), Categorical(logits=lD)
        return dI.log_prob(aI) + dD.log_prob(aD), dI.entropy() + dD.entropy()


class AttentionPPOCritic(nn.Module):
    def __init__(self, s_dim):
        super().__init__()
        self.encoder = FeatureAttentionEncoder(s_dim, HIDDEN_DIM)
        self.value = mlp(HIDDEN_DIM, 1, hid=HIDDEN_DIM, n=max(NUM_HIDDEN - 1, 1))
    def forward(self, s):
        return self.value(self.encoder(s)).squeeze(-1)


class AttentionSACActor(nn.Module):
    def __init__(self, s_dim, a_dim, hid=HIDDEN_DIM, n_layers=NUM_HIDDEN):
        super().__init__()
        self.encoder = FeatureAttentionEncoder(s_dim, hid)
        self.trunk = mlp(hid, hid, hid=hid, n=max(n_layers - 1, 1))
        self.mu = nn.Linear(hid, a_dim)
        self.lstd = nn.Linear(hid, a_dim)

    def forward(self, s):
        h = F.relu(self.trunk(self.encoder(s)))
        return self.mu(h), self.lstd(h).clamp(LOG_STD_MIN, LOG_STD_MAX)

    def sample(self, s, temperature=1.0):
        mu, lstd = self.forward(s)
        mu = torch.nan_to_num(mu, nan=0.0, posinf=20.0, neginf=-20.0)
        std = (lstd.exp() * temperature).clamp(min=1e-6, max=100.0)
        dist = Normal(mu, std, validate_args=False)
        x = dist.rsample()
        a = torch.tanh(x)
        lp = (dist.log_prob(x) - torch.log(1 - a.pow(2) + 1e-6)).sum(-1, keepdim=True)
        return a, lp

    def deterministic(self, s):
        mu, _ = self.forward(s)
        mu = torch.nan_to_num(mu, nan=0.0, posinf=20.0, neginf=-20.0)
        return torch.tanh(mu)


class AttentionTwinQ(nn.Module):
    def __init__(self, s_dim, a_dim, hid=HIDDEN_DIM, n_layers=NUM_HIDDEN):
        super().__init__()
        in_dim = s_dim + a_dim
        self.q1_enc = FeatureAttentionEncoder(in_dim, hid)
        self.q2_enc = FeatureAttentionEncoder(in_dim, hid)
        self.q1 = mlp(hid, 1, hid=hid, n=max(n_layers - 1, 1))
        self.q2 = mlp(hid, 1, hid=hid, n=max(n_layers - 1, 1))
    def forward(self, s, a):
        sa = torch.cat([s, a], -1)
        return self.q1(self.q1_enc(sa)), self.q2(self.q2_enc(sa))


# ═════════════ Baselines (same capacity as proposed) ═════════════

class DDPGActor(nn.Module):
    def __init__(self, s_dim, a_dim):
        super().__init__()
        self.net = mlp(s_dim, a_dim, hid=BASELINE_HIDDEN, n=BASELINE_LAYERS)
    def forward(self, s):
        return torch.tanh(self.net(s))


class DQN(nn.Module):
    def __init__(self, s_dim, n_I, n_D):
        super().__init__()
        self.qI = mlp(s_dim, n_I, hid=BASELINE_HIDDEN, n=BASELINE_LAYERS)
        self.qD = mlp(s_dim, n_D, hid=BASELINE_HIDDEN, n=BASELINE_LAYERS)
    def forward(self, s):
        return self.qI(s), self.qD(s)
