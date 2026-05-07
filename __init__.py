"""
IRS-UAV ISAC simulation: environment, DRL agents, and training utilities.

``figures/`` holds the manuscript system model, algorithm block diagram, and
exported result plots.  Run training from the parent directory::

    python -m security_optimization_irs_uav_its.train
    python -m security_optimization_irs_uav_its.train --quick
"""

from .agents import (
    AttentionPPOSACAgent,
    DDPGDQNAgent,
    HybridSACAgent,
    PPOSACAgent,
    PPOSACGAEAgent,
    ReplayBuffer,
    StabilizedAgentMixin,
    anneal,
    clip_and_step,
    device,
)
from .config import HIDDEN_DIM, NUM_HIDDEN
from .environment import ISACEnv, desired_beampattern, steering_vector
from .networks import (
    AttentionPPOActor,
    AttentionPPOCritic,
    AttentionSACActor,
    AttentionTwinQ,
    DDPGActor,
    DQN,
    FeatureAttentionEncoder,
    PPOActor,
    PPOCritic,
    SACActor,
    TwinQ,
)

__all__ = [
    "AttentionPPOSACAgent",
    "AttentionPPOActor",
    "AttentionPPOCritic",
    "AttentionSACActor",
    "AttentionTwinQ",
    "DDPGDQNAgent",
    "DDPGActor",
    "DQN",
    "FeatureAttentionEncoder",
    "HybridSACAgent",
    "HIDDEN_DIM",
    "ISACEnv",
    "NUM_HIDDEN",
    "PPOSACAgent",
    "PPOSACGAEAgent",
    "PPOActor",
    "PPOCritic",
    "ReplayBuffer",
    "SACActor",
    "StabilizedAgentMixin",
    "TwinQ",
    "anneal",
    "clip_and_step",
    "desired_beampattern",
    "device",
    "steering_vector",
]
