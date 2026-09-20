"""The supervised model, wrapped as an actor-critic for `DanceEnv`.

The environment's action is the normalised pose delta, which is exactly what
`MotionTransformer.delta` emits. So the checkpoint from `supervised-training` *is* a policy
for this environment, and fine-tuning starts from a model that already dances instead of from
noise. Three things are bolted on, all initialised so that the policy's initial mean action is
bit-for-bit the supervised prediction:

    log_std       per-channel exploration noise, small at the start
    lookahead     the music past the current frame, which the causal backbone never saw;
                  its contribution to the mean is zero-initialised
    value         a critic on the shared trunk

Starting *at* the pretrained policy is what makes the KL term in `train_rl.py` meaningful:
the reward here is a proxy (see the README), and a policy free to wander away from the
supervised prior will find ways to score well that do not look like dancing.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "supervised-training"))

from model import ModelConfig, MotionTransformer  # noqa: E402


class ActorCritic(nn.Module):
    def __init__(self, cfg: ModelConfig, log_std_init: float = -3.0):
        super().__init__()
        self.backbone = MotionTransformer(cfg)
        self.cfg = cfg

        # Exploration noise in normalised pose units. e^-3 ~ 0.05, about a twentieth of a
        # standard deviation per frame: enough to explore, small enough that the rollout still
        # looks like a dance rather than a seizure.
        self.log_std = nn.Parameter(torch.full((cfg.motion_dim,), log_std_init))

        # The music the causal backbone cannot see. Pooled, then projected into the mean
        # through a zero-initialised layer, so at step 0 it contributes exactly nothing.
        self.lookahead = nn.Sequential(
            nn.Linear(cfg.audio_dim + 2, cfg.d_model), nn.GELU(),
            nn.Linear(cfg.d_model, cfg.d_model), nn.GELU(),
        )
        self.lookahead_delta = nn.Linear(cfg.d_model, cfg.motion_dim)
        nn.init.zeros_(self.lookahead_delta.weight)
        nn.init.zeros_(self.lookahead_delta.bias)

        self.value = nn.Sequential(
            nn.Linear(2 * cfg.d_model, cfg.d_model), nn.GELU(),
            nn.Linear(cfg.d_model, 1),
        )

    # ------------------------------------------------------------------ construction
    @classmethod
    def from_supervised(cls, checkpoint: Path, device: str = "cpu", **kwargs) -> "ActorCritic":
        state = torch.load(checkpoint, map_location=device, weights_only=False)
        policy = cls(ModelConfig(**state["config"]), **kwargs)
        policy.backbone.load_state_dict(state["model"])
        return policy.to(device)

    # ------------------------------------------------------------------ forward
    def _trunk(self, prev_motion: torch.Tensor, audio: torch.Tensor) -> torch.Tensor:
        """`MotionTransformer.forward` stopped one layer early: the hidden state, not the pose.

        Kept in step with `model.py` by using its own modules; if that forward pass changes,
        this must change with it.
        """
        b, t, _ = audio.shape
        net = self.backbone
        x = net.input(torch.cat([prev_motion, audio], dim=-1)) + net.pos[:, :t]
        mask = nn.Transformer.generate_square_subsequent_mask(t, device=audio.device)
        return net.norm(net.blocks(x, mask=mask, is_causal=True))

    def forward(self, obs: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """obs tensors: audio (B, P + F, A), pose (B, P, M), beat (B, 2).

        Returns (mean action, std, value). The pose history holds frames t-P+1..t, so the
        audio that belongs to the frames being *predicted* (t-P+2..t+1) is the window shifted
        by one; everything past it is lookahead.
        """
        pose, audio, beat = obs["pose"], obs["audio"], obs["beat"]
        window = pose.shape[1]
        aligned = audio[:, 1: window + 1]
        future = audio[:, window + 1:]

        h = self._trunk(pose, aligned)[:, -1]  # the position that predicts frame t+1

        if future.shape[1] == 0:  # no lookahead configured: fall back to the current frame
            future = audio[:, window: window + 1]
        beat_broadcast = beat[:, None, :].expand(-1, future.shape[1], -1)
        look = self.lookahead(torch.cat([future, beat_broadcast], dim=-1)).mean(dim=1)

        mean = self.backbone.delta(h) + self.lookahead_delta(look)
        value = self.value(torch.cat([h, look], dim=-1)).squeeze(-1)
        return mean, self.log_std.exp().expand_as(mean), value

    # ------------------------------------------------------------------ acting
    def distribution(self, obs: dict) -> tuple[torch.distributions.Normal, torch.Tensor]:
        mean, std, value = self(obs)
        return torch.distributions.Normal(mean, std), value

    @torch.no_grad()
    def act(self, obs: dict, deterministic: bool = False):
        """One action per batch element, with its log-probability and the critic's value."""
        dist, value = self.distribution(obs)
        action = dist.mean if deterministic else dist.sample()
        return action, dist.log_prob(action).sum(-1), value

    def evaluate(self, obs: dict, action: torch.Tensor):
        """Log-probability, entropy and value of actions taken earlier, for the PPO update."""
        dist, value = self.distribution(obs)
        return dist.log_prob(action).sum(-1), dist.entropy().sum(-1), value, dist


def to_tensors(obs: dict, device: str) -> dict:
    return {k: torch.as_tensor(v, dtype=torch.float32, device=device) for k, v in obs.items()}
