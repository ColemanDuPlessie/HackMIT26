"""Audio and motion representations shared by data prep, training and generation.

Audio: the 35-dim per-frame feature used by the AIST++ baselines (FACT, EDGE):
    envelope (1) + MFCC (20) + chroma (12) + onset peak one-hot (1) + beat one-hot (1)
sampled at the motion frame rate, so audio[t] lines up with motion[t].

Motion: MediaPipe world landmarks, (T, 33, 3) metres, hip-centred, flattened to (T, 99).
That is what `reward-calculation` scores and what `pose-estimation/retarget.py` consumes,
so a generated clip can be scored and turned into a MuJoCo humanoid without conversion.

Note MediaPipe world landmarks carry no global translation: the dancer never leaves the
origin. Fine for AIST's mostly-in-place choreography; a travelling dance needs a root
trajectory channel added here.
"""

import numpy as np

NUM_LANDMARKS = 33
MOTION_DIM = NUM_LANDMARKS * 3
AUDIO_DIM = 35
SAMPLE_RATE = 22050
N_MFCC = 20


def audio_features(wav: np.ndarray, sr: int, fps: float, n_frames: int | None = None) -> np.ndarray:
    """(n_frames, 35) features at the motion frame rate. `wav` is mono float32."""
    import librosa  # imported lazily so training-free tools don't need it

    hop = int(round(sr / fps))
    envelope = librosa.onset.onset_strength(y=wav, sr=sr, hop_length=hop)
    mfcc = librosa.feature.mfcc(y=wav, sr=sr, n_mfcc=N_MFCC, hop_length=hop)
    chroma = librosa.feature.chroma_cens(y=wav, sr=sr, hop_length=hop)

    peaks = librosa.onset.onset_detect(onset_envelope=envelope, sr=sr, hop_length=hop)
    _, beats = librosa.beat.beat_track(onset_envelope=envelope, sr=sr, hop_length=hop, tightness=100)

    frames = len(envelope)
    peak_hot = np.zeros(frames, dtype=np.float32)
    beat_hot = np.zeros(frames, dtype=np.float32)
    peak_hot[np.clip(peaks, 0, frames - 1)] = 1.0
    beat_hot[np.clip(beats, 0, frames - 1)] = 1.0

    feats = np.concatenate([
        envelope[None, :], mfcc, chroma, peak_hot[None, :], beat_hot[None, :],
    ], axis=0).T.astype(np.float32)
    assert feats.shape[1] == AUDIO_DIM, feats.shape
    return fit_length(feats, n_frames) if n_frames else feats


def fit_length(x: np.ndarray, n: int) -> np.ndarray:
    """Pad (edge) or trim the first axis to exactly n; audio and video lengths differ by a frame or two."""
    if len(x) == n:
        return x
    if len(x) > n:
        return x[:n]
    return np.concatenate([x, np.repeat(x[-1:], n - len(x), axis=0)], axis=0)


def flatten_motion(world: np.ndarray) -> np.ndarray:
    """(T, 33, 3) -> (T, 99)."""
    return world.reshape(len(world), MOTION_DIM).astype(np.float32)


def unflatten_motion(motion: np.ndarray) -> np.ndarray:
    """(T, 99) -> (T, 33, 3)."""
    return motion.reshape(len(motion), NUM_LANDMARKS, 3).astype(np.float32)


def fill_gaps(world: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate frames MediaPipe missed (NaN). Returns (filled, detected mask)."""
    detected = np.isfinite(world).all(axis=(1, 2))
    if detected.all() or not detected.any():
        return world, detected
    idx = np.flatnonzero(detected)
    flat = world[detected].reshape(len(idx), -1)
    filled = np.stack([np.interp(np.arange(len(world)), idx, flat[:, c])
                       for c in range(flat.shape[1])], axis=1)
    return filled.reshape(world.shape).astype(np.float32), detected


def to_keypoints(world: np.ndarray, fps: float) -> dict:
    """Package generated motion like extract_keypoints' output, for retarget.py and the web UI.

    `image` is unknown for generated motion, so it is filled with NaN; retarget only uses
    `world`, `visibility` and `fps`.
    """
    return {
        "world": world.astype(np.float32),
        "image": np.full((len(world), NUM_LANDMARKS, 3), np.nan, dtype=np.float32),
        "visibility": np.ones((len(world), NUM_LANDMARKS), dtype=np.float32),
        "fps": np.float32(fps),
    }


def kinematic_beats(world: np.ndarray, fps: float) -> np.ndarray:
    """Frame indices where motion 'hits': local minima of overall joint speed.

    Used by the beat-alignment metric, which rewards dancing on the music's beat rather
    than copying one particular choreography.
    """
    speed = np.linalg.norm(np.diff(world, axis=0), axis=2).mean(axis=1)
    if len(speed) < 3:
        return np.zeros(0, dtype=int)
    lower_than_neighbours = (speed[1:-1] < speed[:-2]) & (speed[1:-1] < speed[2:])
    quiet = speed[1:-1] < np.median(speed)  # ignore dips during fast passages
    return np.flatnonzero(lower_than_neighbours & quiet) + 1
