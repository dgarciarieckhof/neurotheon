"""
callback_stats.py
Logs per-episode analytics to TensorBoard:
  enemies_killed, allies_hit, neutral_hit,
  shots_fired, invalid_shots (cool-down), accuracy.
Works with any VecEnv (Subproc or Dummy).
"""
from stable_baselines3.common.callbacks import BaseCallback


class StatsCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self._reset_ep()

    # ---------------------------------------------------------------
    def _reset_ep(self):
        self.kills = self.allies = self.neutrals = 0
        self.shots = self.invalid = 0

    # ---------------------------------------------------------------
    def _on_step(self) -> bool:
        info   = self.locals["infos"][0]
        action = self.locals["actions"][0]

        if action != 0:
            self.shots += 1

        hit = info.get("hit")
        if hit == "enemy":
            self.kills += 1
        elif hit == "ally":
            self.allies += 1
        elif hit == "neutral":
            self.neutrals += 1

        # invalid shot: trigger pulled before cooldown expired
        if action != 0 and hit is None:
            vec       = self.locals["env"]          # VecEnv reference
            steps     = vec.get_attr("steps")[0]
            last_shot = vec.get_attr("last_shot_step")[0]
            cooldown  = vec.get_attr("cooldown_steps")[0]
            if (steps - last_shot) < cooldown:
                self.invalid += 1
        return True

    # ---------------------------------------------------------------
    def _on_rollout_end(self) -> None:
        acc = self.kills / max(self.shots, 1)
        self.logger.record("rollout/enemies_killed", self.kills)
        self.logger.record("rollout/allies_hit",     self.allies)
        self.logger.record("rollout/neutral_hit",    self.neutrals)
        self.logger.record("rollout/shots_fired",    self.shots)
        self.logger.record("rollout/invalid_shots",  self.invalid)
        self.logger.record("rollout/accuracy",       acc)
        self._reset_ep()
