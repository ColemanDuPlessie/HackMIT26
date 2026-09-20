"""Train a tracking policy with PPO (Stable-Baselines3).

    uv run train_ppo.py --steps 200000                  # one clip
    uv run train_ppo.py --motion a.npz b.npz --envs 8   # several clips in parallel
    uv run train_ppo.py --check                         # 2k steps, just to prove the loop runs

Each parallel env gets its own clip (cycling if there are fewer clips than envs), so a policy
trained on several clips sees them interleaved rather than one after another.

Expect this to need far more steps than it looks: DeepMimic-style tracking of a single clip is
usually millions of steps. `--check` exists so you can confirm the plumbing in seconds instead
of discovering a shape error twenty minutes in.
"""

import argparse
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

from dance_env import DanceTrackingEnv, _demo_motion

ROOT = Path(__file__).resolve().parent


def env_factory(motion: Path, seed: int):
    def build():
        env = Monitor(DanceTrackingEnv(motion))
        env.reset(seed=seed)
        return env
    return build


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--motion", type=Path, nargs="*", help="retargeted motion .npz files")
    parser.add_argument("--out", type=Path, default=ROOT / "runs")
    parser.add_argument("--steps", type=int, default=200_000)
    parser.add_argument("--envs", type=int, default=8)
    parser.add_argument("--check", action="store_true", help="2k steps as a plumbing test")
    parser.add_argument("--tensorboard", action="store_true",
                        help="log to runs/tb (needs `uv add tensorboard`)")
    args = parser.parse_args()

    motions = args.motion or [_demo_motion()]
    if args.check:
        args.steps, args.envs = 2_000, 2
    args.out.mkdir(parents=True, exist_ok=True)

    envs = SubprocVecEnv([env_factory(motions[i % len(motions)], i) for i in range(args.envs)])
    # Observations mix radians, metres and velocities; normalising saves the policy from
    # learning the scale differences itself. Rewards are already in [0, 1] per step.
    envs = VecNormalize(envs, norm_obs=True, norm_reward=False, clip_obs=10.0)

    model = PPO("MlpPolicy", envs, verbose=1, n_steps=512, batch_size=1024, gae_lambda=0.95,
                gamma=0.99, learning_rate=3e-4, ent_coef=0.0,
                policy_kwargs={"net_arch": [512, 256]},
                tensorboard_log=str(args.out / "tb") if args.tensorboard else None)
    callback = CheckpointCallback(save_freq=max(1, 50_000 // args.envs), save_path=str(args.out),
                                  name_prefix="ppo")
    model.learn(total_timesteps=args.steps, callback=callback, progress_bar=False)

    model.save(args.out / "ppo_final")
    envs.save(str(args.out / "vecnormalize.pkl"))
    print(f"\nSaved {args.out / 'ppo_final.zip'}")

    rewards = []
    obs = envs.reset()
    for _ in range(300):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _ = envs.step(action)
        rewards.append(np.mean(reward))
    print(f"mean reward per step after training: {np.mean(rewards):.3f} "
          f"(1.0 is perfect tracking; ~0.2 is falling over)")


if __name__ == "__main__":
    main()
