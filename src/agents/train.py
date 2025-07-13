"""
Train a PPO agent on TurretEnv with curriculum, VecNormalize and GPU auto-selection.
Checkpoints every 10k steps are written to runs/.
"""

from envs.turret_env import TurretEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback
from agents.curriculum import AdaptiveCurriculum
from agents.callback_vecnorm import SaveVecNormCallback
from agents.callback_kills import KillCountCallback
from agents.callback_stats import StatsCallback
from agents.callback_csv import EpisodeCSVCallback


TOTAL_STEPS = 500_000
CKPT_FREQ   = 50_000
WORKERS     = 4
save_freq_calls = CKPT_FREQ // WORKERS

def make_env(**kw):
    return TurretEnv(**kw)

def build_venv():
    # each worker gets its own seed
    def _factory(idx: int):
        return lambda: make_env(n_enemies=3, enemy_move_every=8, seed=idx)

    base = SubprocVecEnv([_factory(i) for i in range(WORKERS)])
    return VecNormalize(base, norm_obs=False, norm_reward=True, clip_reward=10.0)


def main():
    venv = build_venv()

    ckpt_cb   = CheckpointCallback(save_freq_calls, "runs", "ppo_turret_step")
    curric_cb = AdaptiveCurriculum(
        make_env,
        milestones=[
            # Phase 1 – Basic aim + low threat
            (0, dict(
                n_enemies=3,
                enemy_move_every=8,
                path_probs=(1.0, 0.0, 0.0),  # straight only
                cooldown_steps=0,
                sensor_min=5,
            )),

            # Phase 2 – More targets, some movement variety
            (100_000, dict(
                n_enemies=5,
                enemy_move_every=6,
                path_probs=(0.7, 0.2, 0.1),  # mostly straight, few zigzag
            )),

            # Phase 3 – Full mixed pathing, tighter timing
            (200_000, dict(
                n_enemies=7,
                enemy_move_every=5,
                path_probs=(0.4, 0.35, 0.25),  # introduce erratic patterns
                sensor_min=3,
            )),

            # Phase 4 – Fire cooldown and reduced visibility
            (300_000, dict(
                n_enemies=9,
                enemy_move_every=4,
                cooldown_steps=1,
                sensor_min=3,
            )),

            # Phase 5 – Final test: tight, reactive combat
            (400_000, dict(
                n_enemies=11,
                enemy_move_every=3,
                cooldown_steps=1,
                sensor_min=2,
                path_probs=(0.33, 0.33, 0.34),  # full pathing entropy
            )),
        ],
        verbose=1,
    )
    vec_cb = SaveVecNormCallback(venv, "runs/vecnorm.pkl", save_freq_calls)

    model = PPO(
        "MlpPolicy",
        venv,
        n_steps=1024,
        batch_size=4096,
        learning_rate=1e-4,
        tensorboard_log="logs",
        verbose=1,
        device="auto",
        policy_kwargs=dict(net_arch=[256, 256]),
    )

    model.learn(
        total_timesteps=TOTAL_STEPS,
        callback=[ckpt_cb, curric_cb,
                  KillCountCallback(),
                  StatsCallback(),
                  EpisodeCSVCallback(),
                  vec_cb]
    )

    model.save("runs/ppo_turret_final")
    venv.save("runs/vecnorm.pkl")
    print("✓ Training finished – weights + vecnorm saved in runs/")


# ————————————————————————————————————————————
if __name__ == "__main__":
    from multiprocessing import set_start_method, freeze_support
    freeze_support()                     # Windows-safe; no-op on Unix
    try:
        # On POSIX "fork" is fastest; fallback to default if already set
        set_start_method("fork")
    except RuntimeError:
        pass
    main()