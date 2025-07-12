from stable_baselines3.common.callbacks import BaseCallback

class SaveVecNormCallback(BaseCallback):
    """Save VecNormalize stats on the same schedule as checkpoints."""
    def __init__(self, vecenv, save_path: str, save_freq: int):
        super().__init__()
        self.vecenv = vecenv
        self.save_path = save_path
        self.save_freq = save_freq

    def _on_step(self) -> bool:
        if self.num_timesteps % self.save_freq == 0:
            self.vecenv.save(self.save_path)
        return True