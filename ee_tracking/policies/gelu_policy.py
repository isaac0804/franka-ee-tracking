"""Custom MLP policies: LayerNorm + GELU, skip connections, asymmetric critic.

Motivation
----------
After training, the actor's Tanh pre-activations have std ≈ 2-3, putting
32–49% of neuron-activations into the hard-saturation zone (|x| > 2, gradient
< 0.07).  The critic — identical architecture, same obs — has std ≈ 0.8 and
< 2% saturation.  The actor is gradient-starved despite the critic finding the
value landscape just fine.

Two changes fix this:
  • GELU       — no upper saturation; gradient stays near 1 for large positive
                  pre-activations, and near 0 (soft gate) for large negatives.
  • LayerNorm  — keeps pre-activations at std ≈ 1 regardless of weight
                  magnitudes, preventing drift-induced saturation.

Additional improvements in AsymGELUNormPolicy
---------------------------------------------
  • Skip connections — each hidden block adds x to its output when in_dim ==
                        out_dim (standard residual trick for gradient flow).
                        First block never skips (93 ≠ 256), subsequent same-dim
                        blocks always skip.
  • Asymmetric critic — critic gets a deeper arch ([256,256,256] vs [256,256]
                         for the actor) since value prediction is harder than
                         policy output and benefits from more capacity.
  • weight_decay       — pass via policy_kwargs.optimizer_kwargs to counter
                         weight growth at higher learning rates.

Usage in a config yaml
----------------------
    # Symmetric, no-skip (original):
    train:
      policy: "GELUNormPolicy"
      policy_kwargs:
        net_arch: [256, 256]

    # Asymmetric + skip connections:
    train:
      policy: "AsymGELUNormPolicy"
      policy_kwargs:
        net_arch:
          pi: [256, 256]
          vf: [256, 256, 256]
        optimizer_kwargs:
          weight_decay: 0.0001

Then train.py resolves the name via ee_tracking.policies.POLICY_REGISTRY.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from stable_baselines3.common.policies import ActorCriticPolicy


# ---------------------------------------------------------------------------
# Building block: Linear → LayerNorm → GELU  (+optional skip)
# ---------------------------------------------------------------------------

class SkipBlock(nn.Module):
    """One hidden layer: Linear → LayerNorm → GELU with optional residual skip.

    Skip is added when in_dim == out_dim (no projection needed).
    When dimensions differ (e.g. first block 93 → 256) there is no skip.
    """

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.norm   = nn.LayerNorm(out_dim)
        self.act    = nn.GELU()
        self.skip   = (in_dim == out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.norm(self.linear(x)))
        return h + x if self.skip else h


# ---------------------------------------------------------------------------
# MlpExtractors
# ---------------------------------------------------------------------------

class GELUNormMlpExtractor(nn.Module):
    """Symmetric actor/critic, no skip connections.

    Drop-in replacement for SB3's MlpExtractor.
    Each block: Linear → LayerNorm → GELU (no residual).
    Actor and critic share the same arch; fully separate weights.
    """

    def __init__(self, feature_dim: int, net_arch: list[int]) -> None:
        super().__init__()

        def build(in_dim: int, arch: list[int]) -> tuple[nn.Sequential, int]:
            layers: list[nn.Module] = []
            for out_dim in arch:
                layers += [nn.Linear(in_dim, out_dim), nn.LayerNorm(out_dim), nn.GELU()]
                in_dim = out_dim
            return nn.Sequential(*layers), in_dim

        self.policy_net, self.latent_dim_pi = build(feature_dim, net_arch)
        self.value_net,  self.latent_dim_vf = build(feature_dim, net_arch)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.policy_net(features), self.value_net(features)

    def forward_actor(self, features: torch.Tensor) -> torch.Tensor:
        return self.policy_net(features)

    def forward_critic(self, features: torch.Tensor) -> torch.Tensor:
        return self.value_net(features)


class AsymGELUNormMlpExtractor(nn.Module):
    """Asymmetric actor/critic with skip connections.

    Actor arch  (pi): e.g. [256, 256]         — 2 hidden layers
    Critic arch (vf): e.g. [256, 256, 256]    — 3 hidden layers (more capacity)

    Each block is a SkipBlock (Linear → LN → GELU + residual when dims match).
    First block never skips (feature_dim ≠ hidden_dim in general).
    """

    def __init__(
        self,
        feature_dim: int,
        actor_arch: list[int],
        critic_arch: list[int],
    ) -> None:
        super().__init__()

        def build(in_dim: int, arch: list[int]) -> tuple[nn.Sequential, int]:
            blocks: list[nn.Module] = []
            for out_dim in arch:
                blocks.append(SkipBlock(in_dim, out_dim))
                in_dim = out_dim
            return nn.Sequential(*blocks), in_dim

        self.policy_net, self.latent_dim_pi = build(feature_dim, actor_arch)
        self.value_net,  self.latent_dim_vf = build(feature_dim, critic_arch)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.policy_net(features), self.value_net(features)

    def forward_actor(self, features: torch.Tensor) -> torch.Tensor:
        return self.policy_net(features)

    def forward_critic(self, features: torch.Tensor) -> torch.Tensor:
        return self.value_net(features)


# ---------------------------------------------------------------------------
# Policies
# ---------------------------------------------------------------------------

class GELUNormPolicy(ActorCriticPolicy):
    """Symmetric GELU+LN policy (no skip, same arch for actor and critic).

    All other SB3 machinery (VecNormalize, PPO clip, log_std, etc.) unchanged.
    """

    def _build_mlp_extractor(self) -> None:
        arch = self.net_arch
        if isinstance(arch, dict):
            arch = arch.get("pi", [256, 256])
        self.mlp_extractor = GELUNormMlpExtractor(self.features_dim, arch or [256, 256])


class AsymGELUNormPolicy(ActorCriticPolicy):
    """Asymmetric GELU+LN policy with skip connections.

    Reads net_arch as a dict:
        net_arch: {pi: [256, 256], vf: [256, 256, 256]}
    Falls back to symmetric [256, 256] / [256, 256, 256] if not a dict.

    Skip connections are added automatically by SkipBlock wherever
    in_dim == out_dim (all same-size hidden layers after the first).
    """

    def _build_mlp_extractor(self) -> None:
        arch = self.net_arch
        if isinstance(arch, dict):
            actor_arch  = arch.get("pi", [256, 256])
            critic_arch = arch.get("vf", [256, 256, 256])
        else:
            actor_arch  = arch or [256, 256]
            critic_arch = arch or [256, 256, 256]

        self.mlp_extractor = AsymGELUNormMlpExtractor(
            self.features_dim, actor_arch, critic_arch
        )


# ---------------------------------------------------------------------------
# Registry — train.py resolves policy names through this dict
# ---------------------------------------------------------------------------

POLICY_REGISTRY: dict[str, type] = {
    "GELUNormPolicy":     GELUNormPolicy,
    "AsymGELUNormPolicy": AsymGELUNormPolicy,
}
