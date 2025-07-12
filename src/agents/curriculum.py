from stable_baselines3.common.callbacks import BaseCallback


class CurriculumCallback(BaseCallback):
    """
    Swap in a harder copy of the environment at step milestones.

    milestones = [(t0, kwargs0), (t1, kwargs1), ...]
    """
    def __init__(self, env_ctor, milestones, verbose: int = 0):
        super().__init__(verbose)
        self.env_ctor = env_ctor
        self.milestones = sorted(milestones, key=lambda x: x[0])
        self.next_idx = 0

    # SB3 populates `model` and hence `self.training_env`
    @property
    def _training_env(self):
        return self.model.get_env()

    def _on_step(self) -> bool:
        if self.next_idx >= len(self.milestones):
            return True

        threshold, new_kwargs = self.milestones[self.next_idx]
        if self.num_timesteps >= threshold:
            if self.verbose:
                print(f"🏆 Curriculum level-up → {new_kwargs}")
            self._training_env.envs[0] = self.env_ctor(**new_kwargs)
            self.next_idx += 1
        return True