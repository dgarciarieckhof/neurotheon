"""
saliency.py – stable Grad-CAM-lite for SB3

Returns a 21×21 heat-map (0-1) for the policy-chosen action.
"""

from __future__ import annotations
import torch, numpy as np

def grad_cam(model, obs: np.ndarray, device: str = "cpu") -> np.ndarray:
    policy = model.policy.to(device)

    # Leaf tensor that SB3 won't clone internally
    inp = torch.tensor(obs, dtype=torch.float32,
                       device=device, requires_grad=True).unsqueeze(0)

    # Forward → action distribution (works in both SB3 v1 & v2)
    dist = policy.get_distribution(inp)
    logits = dist.distribution.logits          # shape (1, n_actions)
    act_id = torch.argmax(logits, dim=1)

    # Grad of selected logit w.r.t. *input* tensor
    grad, = torch.autograd.grad(
        logits[0, act_id], inp,
        retain_graph=False, create_graph=False
    )  

    cam = grad.abs().detach().cpu().numpy()[0].sum(axis=2)

    # Normalize 0-1 (avoid /0)
    cam -= cam.min()
    if cam.max() > 1e-8:
        cam /= cam.max()
    return cam
