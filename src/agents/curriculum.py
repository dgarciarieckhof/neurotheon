"""
AdaptiveCurriculum (stable)
Swaps VecEnv at milestones, but only between rollouts, preventing
broken-pipe errors with SubprocVecEnv.
"""
from __future__ import annotations
import copy
from typing import Callable, List, Tuple, Dict
import gymnasium as gym

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize
from stable_baselines3.common.vec_env.base_vec_env import VecEnv


class AdaptiveCurriculum(BaseCallback):
    def __init__(
        self,
        env_ctor: Callable[..., gym.Env],
        milestones: List[Tuple[int, Dict]],
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.env_ctor = env_ctor
        self.milestones = sorted(milestones, key=lambda x: x[0])
        self.next_idx = 0
        self._pending_update: Dict | None = None  # kwargs to apply

    # ---------- helper ------------------------------------------------
    def _factory_vec(self, template_vec: VecEnv, kw: Dict) -> VecEnv:
        n_envs = template_vec.num_envs

        def _maker(i: int):
            return lambda: self.env_ctor(seed=i, **kw)

        if isinstance(template_vec, SubprocVecEnv):
            base = SubprocVecEnv([_maker(i) for i in range(n_envs)])
        else:  # DummyVecEnv or other
            base = DummyVecEnv([_maker(i) for i in range(n_envs)])

        if isinstance(template_vec, VecNormalize):
            new_vec = VecNormalize(
                base,
                norm_obs=template_vec.norm_obs,
                norm_reward=template_vec.norm_reward,
                clip_obs=template_vec.clip_obs,
                clip_reward=template_vec.clip_reward,
            )
            if getattr(template_vec, "obs_rms", None) is not None:
                new_vec.obs_rms = copy.deepcopy(template_vec.obs_rms)
                new_vec.ret_rms = copy.deepcopy(template_vec.ret_rms)
            new_vec.training = template_vec.training
            return new_vec
        return base

    # ---------- callbacks --------------------------------------------
    def _on_step(self) -> bool:
        if self.next_idx >= len(self.milestones):
            return True

        threshold, kw = self.milestones[self.next_idx]
        if self.num_timesteps >= threshold and self._pending_update is None:
            if self.verbose:
                print(f"\n🏆  Curriculum flag @ {self.num_timesteps:,} "
                      f"→ will apply {kw} after this rollout")
            self._pending_update = kw
            self.next_idx += 1
        return True

    def _on_rollout_end(self) -> None:
        if self._pending_update is None:
            return

        old_vec = self.model.get_env()
        new_vec = self._factory_vec(old_vec, self._pending_update)
        self.model.set_env(new_vec)

        # ─ tell PPO the new starting observation ──────────────────
        import numpy as np
        new_obs = new_vec.reset()
        self.model._last_obs = new_obs
        self.model._last_episode_starts = np.ones((new_vec.num_envs,), dtype=bool)

        old_vec.close()
        if self.verbose:
            print("✅  Curriculum applied:", self._pending_update)
        self._pending_update = None