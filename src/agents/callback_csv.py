import csv, os, datetime as dt
from stable_baselines3.common.callbacks import BaseCallback

class EpisodeCSVCallback(BaseCallback):
    """Write one row per finished episode to data/episodes.csv."""
    def __init__(self, csv_path="data/episodes.csv"):
        super().__init__()
        self.csv_path = csv_path
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        self.header_written = os.path.exists(csv_path)

    # ──────────────────────────────────────────────────────────
    def _on_step(self) -> bool:          # ← add this 3-line method
        return True                      # keep rollout running

    # ──────────────────────────────────────────────────────────
    def _on_rollout_end(self):
        logs = self.logger.name_to_value
        row = {
            "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
            "timesteps": self.num_timesteps,
            "enemies_killed": logs.get("rollout/enemies_killed", 0),
            "allies_hit":     logs.get("rollout/allies_hit", 0),
            "neutral_hit":    logs.get("rollout/neutral_hit", 0),
            "shots_fired":    logs.get("rollout/shots_fired", 0),
            "invalid_shots":  logs.get("rollout/invalid_shots", 0),
            "accuracy":       logs.get("rollout/accuracy", 0.0),
            "episode_reward": self.locals["rewards"].sum(),
            "episode_len":    len(self.locals["rewards"]),
        }
        with open(self.csv_path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=row.keys())
            if not self.header_written:
                w.writeheader()
                self.header_written = True
            w.writerow(row)