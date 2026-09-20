"""PPO fine-tuning of a supervised dance model against `DanceEnv`'s reward.

This is stage two, never stage one. `supervised-training/README.md` argues --- correctly ---
that policy gradients are a bad way to learn choreography from labelled clips: every frame of
every clip is already a supervised target, and throwing that away for one scalar per rollout
costs orders of magnitude in samples. What RL buys is the part that is *not* differentiable
and *not* labelled: beat alignment, not drifting into a mean pose, not sliding the feet. So
the policy is initialised from `model.pt` and kept near it by a KL penalty, and the run is
measured in minutes, not hours.

Usage:
    uv run train_rl.py --smoke-test                                   # no dataset needed
    uv run train_rl.py --checkpoint ../supervised-training/checkpoints/model.pt --minutes 20
    uv run train_rl.py --w-imitate 0 --minutes 20                     # free dance, not imitation

Outputs in --out:
    policy.pt       actor-critic weights plus the config needed to rebuild it
    rollout.npz     the most recent greedy episode, ready for retarget.py / the web UI
"""

from __future__ import annotations

import argparse
import copy
import signal
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "supervised-training"))

import features  # noqa: E402
from dataset import Stats  # noqa: E402
from model import ModelConfig  # noqa: E402

from env import DanceEnv, EnvConfig, VecDanceEnv  # noqa: E402
from policy import ActorCritic, to_tensors  # noqa: E402
from reward import RewardConfig  # noqa: E402


def gae(rewards, values, dones, last_value, gamma: float, lam: float):
    """Generalised advantage estimation over a (steps, envs) rollout.

    gamma defaults to 0.995 rather than the usual 0.99: at 60 fps that is a ~2.3 s half-life,
    which is roughly a musical phrase. At 0.99 the policy cannot see past half a second and
    has no way to learn that a move should resolve on the next downbeat.
    """
    steps = len(rewards)
    advantages = np.zeros_like(rewards)
    running = np.zeros_like(last_value)
    for t in reversed(range(steps)):
        next_value = last_value if t == steps - 1 else values[t + 1]
        alive = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * alive - values[t]
        running = delta + gamma * lam * alive * running
        advantages[t] = running
    return advantages, advantages + values


def gaussian_kl(mean_p, std_p, mean_q, std_q):
    """KL(p || q) for diagonal Gaussians, summed over action dimensions."""
    var_p, var_q = std_p ** 2, std_q ** 2
    return (torch.log(std_q / std_p) + (var_p + (mean_p - mean_q) ** 2) / (2 * var_q) - 0.5).sum(-1)


def rollout(vec, policy, device, steps: int, gamma: float = 0.995):
    """Collect `steps` transitions from every env. Returns tensors ready for the update."""
    obs = vec.last_obs
    buffers = {"obs": [], "action": [], "logprob": [], "value": [], "reward": [],
               "done": [], "terms": []}
    episodes = []

    for _ in range(steps):
        obs_t = to_tensors(obs, device)
        action, logprob, value = policy.act(obs_t)
        next_obs, reward, terminated, truncated, infos = vec.step(
            action.cpu().numpy()[:, None, :])  # (N, frames_per_step=1, motion_dim)

        done = terminated | truncated
        # A truncated episode (the music ended) still has future value; a terminated one (the
        # body came apart) does not. Bootstrapping the former here lets both be marked done.
        if truncated.any():
            finals = [infos[i]["final_obs"] for i in np.flatnonzero(truncated)]
            with torch.no_grad():
                _, _, final_value = policy(to_tensors(
                    {k: np.stack([f[k] for f in finals]) for k in finals[0]}, device))
            reward = reward.copy()
            reward[truncated] += gamma * final_value.cpu().numpy()

        buffers["obs"].append(obs)
        buffers["action"].append(action.cpu().numpy())
        buffers["logprob"].append(logprob.cpu().numpy())
        buffers["value"].append(value.cpu().numpy())
        buffers["reward"].append(reward)
        buffers["done"].append(done.astype(np.float32))
        buffers["terms"].append([i["terms"] for i in infos])
        episodes += [i["episode"] for i in infos if "episode" in i]
        obs = next_obs

    vec.last_obs = obs
    with torch.no_grad():
        _, _, last_value = policy(to_tensors(obs, device))
    return buffers, last_value.cpu().numpy(), episodes


def update(policy, reference, optimiser, batch, args, device) -> dict:
    """PPO epochs over one rollout, with a KL anchor to the frozen supervised policy."""
    n = len(batch["advantage"])
    logs: dict[str, list[float]] = {}
    for _ in range(args.epochs):
        for start in range(0, n, args.minibatch):
            idx = batch["order"][start: start + args.minibatch]
            obs = {k: v[idx] for k, v in batch["obs"].items()}
            logprob, entropy, value, dist = policy.evaluate(obs, batch["action"][idx])

            ratio = (logprob - batch["logprob"][idx]).exp()
            advantage = batch["advantage"][idx]
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
            policy_loss = -torch.min(ratio * advantage,
                                     ratio.clamp(1 - args.clip, 1 + args.clip) * advantage).mean()
            value_loss = torch.nn.functional.mse_loss(value, batch["return"][idx])

            with torch.no_grad():
                ref_mean, ref_std, _ = reference(obs)
            kl = gaussian_kl(dist.mean, dist.stddev, ref_mean, ref_std).mean()

            loss = (policy_loss + args.vf_coef * value_loss
                    - args.entropy_coef * entropy.mean() + args.kl_coef * kl)
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), args.clip_grad)
            optimiser.step()

            for key, value_ in (("policy", policy_loss), ("value", value_loss),
                                ("entropy", entropy.mean()), ("kl", kl)):
                logs.setdefault(key, []).append(value_.item())
    return {k: float(np.mean(v)) for k, v in logs.items()}


def flatten(buffers, advantages, returns, device) -> dict:
    """(steps, envs, ...) -> (steps * envs, ...) tensors on `device`."""
    obs = {key: torch.as_tensor(np.concatenate([o[key] for o in buffers["obs"]]),
                                dtype=torch.float32, device=device)
           for key in buffers["obs"][0]}
    as_t = lambda x: torch.as_tensor(np.concatenate(x) if isinstance(x, list) else x.reshape(-1),
                                     dtype=torch.float32, device=device)
    return {"obs": obs, "action": as_t(buffers["action"]), "logprob": as_t(buffers["logprob"]),
            "advantage": as_t(advantages), "return": as_t(returns),
            "order": np.random.permutation(advantages.size)}


@torch.no_grad()
def greedy_episode(env: DanceEnv, policy, device, clip: str | None = None) -> tuple[float, dict]:
    """One noise-free episode, for reporting and for saving something watchable."""
    obs, _ = env.reset(options={"clip": clip} if clip else None)
    total, terms = 0.0, {}
    while True:
        action, _, _ = policy.act(to_tensors({k: v[None] for k, v in obs.items()}, device),
                                  deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action.cpu().numpy())
        total += reward
        for k, v in info["terms"].items():
            terms[k] = terms.get(k, 0.0) + v
        if terminated or truncated:
            return total, {"frac": info["episode"]["frac"], "terms": terms}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path,
                        default=ROOT.parent / "supervised-training" / "checkpoints" / "model.pt")
    parser.add_argument("--cache", type=Path,
                        default=ROOT.parent / "supervised-training" / "cache")
    parser.add_argument("--out", type=Path, default=ROOT / "runs")
    parser.add_argument("--split", default="val", choices=["train", "val", "all"],
                        help="which clips to fine-tune on; val keeps the supervised model honest")
    parser.add_argument("--minutes", type=float, default=20.0)
    parser.add_argument("--envs", type=int, default=16)
    parser.add_argument("--steps", type=int, default=128, help="frames collected per env per update")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--minibatch", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-5, help="an order below supervised: this is a nudge")
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--lam", type=float, default=0.95)
    parser.add_argument("--clip", type=float, default=0.2)
    parser.add_argument("--vf-coef", type=float, default=0.5, dest="vf_coef")
    parser.add_argument("--entropy-coef", type=float, default=1e-4, dest="entropy_coef")
    parser.add_argument("--kl-coef", type=float, default=0.1, dest="kl_coef",
                        help="pull toward the supervised policy; the reward is a proxy and will be gamed")
    parser.add_argument("--clip-grad", type=float, default=1.0, dest="clip_grad")
    parser.add_argument("--max-seconds", type=float, default=8.0, dest="max_seconds",
                        help="episode length; shorter episodes mean more of them per minute")
    parser.add_argument("--w-imitate", type=float, default=RewardConfig.w_imitate, dest="w_imitate")
    parser.add_argument("--w-musicality", type=float, default=RewardConfig.w_musicality,
                        dest="w_musicality")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else
                                     "mps" if torch.backends.mps.is_available() else "cpu")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    if args.smoke_test:
        smoke_test(args)
        return

    if not args.checkpoint.exists():
        sys.exit(f"No checkpoint at {args.checkpoint}; train the supervised model first.")
    args.out.mkdir(parents=True, exist_ok=True)
    device = args.device

    reward_cfg = RewardConfig(w_imitate=args.w_imitate, w_musicality=args.w_musicality)
    env_cfg = EnvConfig(max_seconds=args.max_seconds)
    template = DanceEnv.from_checkpoint(args.checkpoint, args.cache, split=args.split,
                                        cfg=env_cfg, reward_cfg=reward_cfg)
    vec = VecDanceEnv(template.clips, template.stats, n=args.envs, cfg=env_cfg,
                      reward_cfg=reward_cfg)
    print(f"{len(template.clips)} {args.split} clips, {args.envs} envs on {device}")

    policy = ActorCritic.from_supervised(args.checkpoint, device)
    reference = copy.deepcopy(policy).eval().requires_grad_(False)
    optimiser = torch.optim.AdamW(policy.parameters(), lr=args.lr, weight_decay=0.0)

    # One fixed clip for before/after, or the two numbers would be on different music.
    eval_clip = template.clips[0]["name"]
    base, base_info = greedy_episode(template, policy, device, eval_clip)
    print(f"supervised policy on {eval_clip}: return {base:.1f} over {base_info['frac']:.0%} of the clip")

    stop = {"now": False}
    signal.signal(signal.SIGINT, lambda *_: stop.update(now=True))
    deadline = time.time() + args.minutes * 60
    vec.last_obs = vec.reset()
    iteration = 0

    while not stop["now"] and time.time() < deadline:
        iteration += 1
        buffers, last_value, episodes = rollout(vec, policy, device, args.steps, args.gamma)
        values = np.array(buffers["value"])
        advantages, returns = gae(np.array(buffers["reward"]), values,
                                  np.array(buffers["done"]), last_value, args.gamma, args.lam)
        logs = update(policy, reference, optimiser,
                      flatten(buffers, advantages, returns, device), args, device)

        finished = np.mean([e["r"] for e in episodes]) if episodes else float("nan")
        left = max(0.0, deadline - time.time()) / 60
        print(f"iter {iteration:4d}  return/ep {finished:8.2f}  "
              f"reward/frame {np.mean(buffers['reward']):6.3f}  "
              f"kl {logs['kl']:.4f}  value {logs['value']:.3f}  "
              f"entropy {logs['entropy']:7.1f}  {len(episodes):3d} eps  {left:.0f} min left")

    total, info = greedy_episode(template, policy, device, eval_clip)
    print(f"\nfine-tuned policy on {eval_clip}: return {total:.1f} over {info['frac']:.0%} "
          f"of the clip (was {base:.1f} over {base_info['frac']:.0%})")
    print("  terms: " + "  ".join(f"{k} {v:+.1f}" for k, v in sorted(info["terms"].items())))

    torch.save({"policy": policy.state_dict(), "config": policy.cfg.to_dict(),
                "stats": template.stats.to_dict(), "reward": vars(reward_cfg),
                "iterations": iteration}, args.out / "policy.pt")
    template.save_episode(args.out / "rollout.npz")
    print(f"\nWrote {args.out / 'policy.pt'} and {args.out / 'rollout.npz'}; watch it with\n"
          f"  uv run ../pose-estimation/retarget.py {args.out / 'rollout.npz'} -o motion.npz --render out.mp4")


def synthetic_clips(n: int = 4, frames: int = 240, fps: float = 30.0) -> list[dict]:
    """Clips with the shape and the beat structure of real ones, for the smoke test.

    The "dance" is a slow sinusoid around a plausible skeleton, so bone lengths wobble but
    nothing explodes, and beats land every half second.
    """
    rng = np.random.default_rng(0)
    clips = []
    rest = rng.normal(0, 0.25, (features.NUM_LANDMARKS, 3)).astype(np.float32)
    for i in range(n):
        t = np.arange(frames)[:, None, None] / fps
        world = rest[None] + 0.08 * np.sin(2 * np.pi * (1.0 + 0.2 * i) * t + rest[None])
        audio = rng.normal(0, 1, (frames, features.AUDIO_DIM)).astype(np.float32)
        audio[:, -1] = 0.0
        audio[::15, -1] = 1.0  # a beat every half second
        clips.append({"name": f"gSY_sBM_c01_d01_mSY{i}_ch01_00", "fps": fps, "audio": audio,
                      "motion": features.flatten_motion(world.astype(np.float32))})
    return clips


def smoke_test(args):
    """End to end on synthetic data: env shapes, reward terms, a few PPO updates."""
    torch.manual_seed(0)
    np.random.seed(0)
    device = "cpu"  # tiny model; MPS/CUDA launch overhead dominates
    clips = synthetic_clips()
    stats = Stats.fit(clips)
    cfg = EnvConfig(pose_history=16, audio_future=8, seed_frames=16, max_seconds=3.0)
    reward_cfg = RewardConfig(imitate_window_s=1.0, imitate_every=10)

    env = DanceEnv(clips, stats, cfg, reward_cfg)
    obs, info = env.reset()
    assert obs["audio"].shape == (cfg.pose_history + cfg.audio_future, features.AUDIO_DIM), obs["audio"].shape
    assert obs["pose"].shape == (cfg.pose_history, features.MOTION_DIM), obs["pose"].shape
    assert obs["beat"].shape == (2,)
    print(f"env: {info['clip']} {info['frames']} frames, obs "
          f"{ {k: v.shape for k, v in obs.items()} }")

    # A zero action repeats the previous pose exactly: the degenerate policy the energy term exists
    # to punish. It should survive (bones intact) and score below the reference's own dance.
    frozen, terms = 0.0, {}
    while True:
        obs, r, terminated, truncated, step_info = env.step(np.zeros((1, features.MOTION_DIM)))
        frozen += r
        for k, v in step_info["terms"].items():
            terms[k] = terms.get(k, 0.0) + v
        if terminated or truncated:
            break
    print(f"frozen-pose policy: return {frozen:.1f}, terms "
          + "  ".join(f"{k} {v:+.1f}" for k, v in sorted(terms.items())))
    assert not terminated, "a frozen pose has intact bones; it should not terminate"
    assert terms.get("energy", 0.0) < 0, "the energy term should punish standing still"

    # Replaying the reference dance's own deltas must reproduce it, and score well.
    obs, _ = env.reset(options={"clip": clips[0]["name"]})
    real = stats.normalise_motion(clips[0]["motion"])
    replay, replay_terms, t = 0.0, {}, cfg.seed_frames
    while True:
        obs, r, terminated, truncated, step_info = env.step((real[t] - real[t - 1])[None])
        replay += r
        t += 1
        for k, v in step_info["terms"].items():
            replay_terms[k] = replay_terms.get(k, 0.0) + v
        if terminated or truncated:
            break
    error = np.abs(env.world[: env.t + 1] - features.unflatten_motion(clips[0]["motion"])[: env.t + 1]).max()
    print(f"reference replay: return {replay:.1f}, max reconstruction error {error:.2e}, terms "
          + "  ".join(f"{k} {v:+.1f}" for k, v in sorted(replay_terms.items())))
    assert error < 1e-4, "replaying the reference deltas should reproduce the reference"
    assert replay > frozen, "the real dance should outscore a frozen pose"

    # The calibration invariant from reward.py: the real dance pays ~nothing on the penalty
    # terms, because each threshold is a high percentile of its own frames. Break this and the
    # policy is quietly taught to move less than a human.
    penalties = -sum(min(0.0, replay_terms.get(k, 0.0)) for k in ("bone", "energy", "jerk", "foot"))
    assert penalties < 0.1 * replay, \
        f"reference replay should pay ~no penalties, paid {penalties:.1f} of {replay:.1f}"

    # PPO on a tiny model.
    model_cfg = ModelConfig(d_model=64, n_layers=2, n_heads=2, d_ff=128, max_len=128)
    policy = ActorCritic(model_cfg).to(device)
    reference = copy.deepcopy(policy).eval().requires_grad_(False)
    mean, std, value = policy(to_tensors({k: v[None] for k, v in obs.items()}, device))
    assert torch.allclose(mean, torch.zeros_like(mean)), \
        "a fresh policy's mean action must be the supervised one (zero delta here)"
    assert std.shape == mean.shape and value.shape == (1,)

    vec = VecDanceEnv(clips, stats, n=4, cfg=cfg, reward_cfg=reward_cfg)
    vec.last_obs = vec.reset()
    optimiser = torch.optim.AdamW(policy.parameters(), lr=1e-4)
    args.epochs, args.minibatch = 2, 32
    args.clip, args.vf_coef, args.entropy_coef, args.kl_coef, args.clip_grad = 0.2, 0.5, 1e-4, 0.1, 1.0
    for iteration in range(3):
        buffers, last_value, episodes = rollout(vec, policy, device, 16)
        advantages, returns = gae(np.array(buffers["reward"]), np.array(buffers["value"]),
                                  np.array(buffers["done"]), last_value, 0.995, 0.95)
        logs = update(policy, reference, optimiser,
                      flatten(buffers, advantages, returns, device), args, device)
        print(f"  ppo iter {iteration}: reward/frame {np.mean(buffers['reward']):+.3f}  "
              f"kl {logs['kl']:.5f}  value {logs['value']:.3f}")
        assert all(np.isfinite(v) for v in logs.values()), logs
    print("smoke test passed")


if __name__ == "__main__":
    main()
