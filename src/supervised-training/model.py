"""Audio-conditioned autoregressive transformer: music in, dance out.

At frame t the model sees the music up to t and the pose at t-1, and predicts the pose at t:

    pose[t] = f(audio[0..t], pose[0..t-1])

A causal mask makes every position a training example, so one forward pass supervises the
whole window (teacher forcing). Generation replays the same step autoregressively.

This is deliberately the simplest architecture that works. The obvious upgrades, in order of
expected payoff: a diffusion head over whole windows (EDGE-style, which removes the
autoregressive drift entirely), a VQ-VAE motion codebook with a GPT over the codes
(Bailando), and richer audio features (Jukebox embeddings instead of MFCCs).
"""

from dataclasses import asdict, dataclass

import torch
from torch import nn

from features import AUDIO_DIM, MOTION_DIM


@dataclass
class ModelConfig:
    motion_dim: int = MOTION_DIM
    audio_dim: int = AUDIO_DIM
    d_model: int = 384
    n_layers: int = 6
    n_heads: int = 6
    d_ff: int = 1536
    dropout: float = 0.1
    max_len: int = 1024  # longest window the learned positional embedding supports

    def to_dict(self) -> dict:
        return asdict(self)


class MotionTransformer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.input = nn.Linear(cfg.motion_dim + cfg.audio_dim, cfg.d_model)
        self.pos = nn.Parameter(torch.zeros(1, cfg.max_len, cfg.d_model))
        nn.init.normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model, nhead=cfg.n_heads, dim_feedforward=cfg.d_ff,
            dropout=cfg.dropout, batch_first=True, norm_first=True, activation="gelu",
        )
        self.blocks = nn.TransformerEncoder(layer, num_layers=cfg.n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(cfg.d_model)
        # Predict the change from the previous pose: near-zero deltas are an easy starting
        # point and keep early training from drifting away from a plausible body.
        self.delta = nn.Linear(cfg.d_model, cfg.motion_dim)
        nn.init.zeros_(self.delta.weight)
        nn.init.zeros_(self.delta.bias)

    def forward(self, prev_motion: torch.Tensor, audio: torch.Tensor) -> torch.Tensor:
        """prev_motion, audio: (B, T, dim). Returns predicted motion (B, T, motion_dim)."""
        b, t, _ = audio.shape
        if t > self.cfg.max_len:
            raise ValueError(f"window {t} exceeds max_len {self.cfg.max_len}")
        x = self.input(torch.cat([prev_motion, audio], dim=-1)) + self.pos[:, :t]
        mask = nn.Transformer.generate_square_subsequent_mask(t, device=audio.device)
        h = self.norm(self.blocks(x, mask=mask, is_causal=True))
        return prev_motion + self.delta(h)

    @torch.no_grad()
    def generate(self, audio: torch.Tensor, seed: torch.Tensor | None = None) -> torch.Tensor:
        """Roll out motion for a whole track. audio: (1, T, A); seed: (1, S, M) starting poses.

        Recomputes the full prefix each step, which is O(T^2) but simple; a KV cache is the
        fix if generation is ever slow enough to matter.
        """
        self.eval()
        t = audio.shape[1]
        motion = (seed.clone() if seed is not None
                  else torch.zeros(1, 1, self.cfg.motion_dim, device=audio.device))
        while motion.shape[1] < t:
            step = motion.shape[1]
            window = slice(max(0, step - self.cfg.max_len + 1), step)
            pred = self.forward(motion[:, window], audio[:, window.start:step])
            motion = torch.cat([motion, pred[:, -1:]], dim=1)
        return motion[:, :t]
