"""
sensors.py
--------------------------------------------
Simulates a radar.
"""

import numpy as np

def detect(entity, agent_pos, sensor_range: int, rng) -> bool:
    """
    Probabilistically determine whether the turret detects `entity`.
    Visibility drops with distance and entity stealth.
    """
    # Euclidean distance on the grid
    dist = np.linalg.norm(entity["pos"] - agent_pos)
    base_prob = max(0.0, 1.0 - dist / sensor_range)
    return rng.random() < base_prob * (1.0 - entity["stealth"])