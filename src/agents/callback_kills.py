from stable_baselines3.common.callbacks import BaseCallback

class KillCountCallback(BaseCallback):
    """Logs enemies killed per episode to TensorBoard."""
    def __init__(self):
        super().__init__()
        self.kills_this_ep = 0

    def _on_step(self) -> bool:
        info = self.locals.get("infos", [{}])[0]
        if info.get("killed_enemy"):
            self.kills_this_ep += 1
        return True

    def _on_rollout_end(self) -> None:
        # rollout_end is called at episode end in SB3
        self.logger.record("rollout/enemies_killed", self.kills_this_ep)
        self.kills_this_ep = 0
