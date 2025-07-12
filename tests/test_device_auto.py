import torch, pytest
from envs.turret_env import TurretEnv
from stable_baselines3 import PPO

def test_auto_device_cpu(monkeypatch):
    # Force CUDA unavailable so SB3 must pick CPU
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    model = PPO("MlpPolicy", TurretEnv(n_enemies=0), device="auto",
                policy_kwargs=dict(net_arch=[32]))
    assert model.device.type == "cpu"