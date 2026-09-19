"""Windowed (audio, motion) clips for training, with normalisation statistics.

Each cached clip is cut into fixed-length overlapping windows. Both streams are normalised
per channel using statistics from the training split only, so the model doesn't see test data.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class Stats:
    """Per-channel mean/std for audio and motion, saved alongside a checkpoint."""
    audio_mean: np.ndarray
    audio_std: np.ndarray
    motion_mean: np.ndarray
    motion_std: np.ndarray

    @staticmethod
    def fit(clips: list[dict]) -> "Stats":
        audio = np.concatenate([c["audio"] for c in clips])
        motion = np.concatenate([c["motion"] for c in clips])
        eps = 1e-6
        return Stats(audio.mean(0), audio.std(0) + eps, motion.mean(0), motion.std(0) + eps)

    def normalise_audio(self, a):
        return (a - self.audio_mean) / self.audio_std

    def normalise_motion(self, m):
        return (m - self.motion_mean) / self.motion_std

    def denormalise_motion(self, m):
        mean, std = self.motion_mean, self.motion_std
        if torch.is_tensor(m):
            mean = torch.as_tensor(mean, dtype=m.dtype, device=m.device)
            std = torch.as_tensor(std, dtype=m.dtype, device=m.device)
        return m * std + mean

    def to_dict(self) -> dict:
        return {k: np.asarray(v) for k, v in self.__dict__.items()}

    @staticmethod
    def from_dict(d: dict) -> "Stats":
        return Stats(*(np.asarray(d[k]) for k in
                       ("audio_mean", "audio_std", "motion_mean", "motion_std")))


def load_clips(cache_dir: Path) -> list[dict]:
    clips = []
    for path in sorted(Path(cache_dir).glob("*.npz")):
        with np.load(path) as z:
            clips.append({"name": path.stem, "audio": z["audio"], "motion": z["motion"],
                          "fps": float(z["fps"])})
    if not clips:
        raise FileNotFoundError(f"No prepared clips in {cache_dir}; run prepare_data.py first.")
    return clips


def split_clips(clips: list[dict], holdout: float = 0.15, seed: int = 0) -> tuple[list[dict], list[dict]]:
    """Split by clip, not by window, so windows of one dance never straddle the split.

    A stricter split would hold out whole *songs* (the mXXX field of the AIST name): chunks
    of one song share a soundtrack, and the model could otherwise memorise it.
    """
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(clips))
    n_val = max(1, int(round(holdout * len(clips))))
    val = [clips[i] for i in order[:n_val]]
    train = [clips[i] for i in order[n_val:]]
    return train, val


class DanceWindows(Dataset):
    """Fixed-length windows: audio (T, 35) and the motion (T, 99) danced to it."""

    def __init__(self, clips: list[dict], stats: Stats, window: int, stride: int):
        self.clips, self.stats, self.window = clips, stats, window
        self.index = [(i, s) for i, c in enumerate(clips)
                      for s in range(0, len(c["motion"]) - window + 1, stride)]
        if not self.index:
            raise ValueError(f"No clip is at least {window} frames long.")

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int):
        clip_idx, start = self.index[i]
        clip = self.clips[clip_idx]
        sl = slice(start, start + self.window)
        audio = self.stats.normalise_audio(clip["audio"][sl])
        motion = self.stats.normalise_motion(clip["motion"][sl])
        return torch.from_numpy(audio.astype(np.float32)), torch.from_numpy(motion.astype(np.float32))
