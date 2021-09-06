import torch
import numpy as np
from fastdtw import fastdtw

def dtw(s, t):
    return fastdtw(s, t)[0]

def calc_optimal_grid(n):
    # Try to distribute n images as best as possible
    # For now: simple overestimation
    sides = int(np.ceil(np.sqrt(n)))
    return sides, sides

def gradcam(logits, features):
    grads = torch.autograd.grad(logits, features)[0].squeeze().detach()

    features = features.detach().squeeze()

    w = torch.mean(grads, axis=1)

    cam = torch.zeros_like(features[0])
    for k in range(features.shape[0]):
        cam += w[k] * features[k]

    return torch.nn.functional.relu(cam).squeeze().numpy()

def smape(a, b, axis=None):
    if isinstance(a, type(torch.zeros(0))) and isinstance(b, type(torch.zeros(0))):
        fn_abs = torch.abs
        fn_mean = torch.mean
    elif isinstance(a, type(np.zeros(0))) and isinstance(b, type(np.zeros(0))):
        fn_abs = np.abs
        fn_mean = np.mean
    else:
        raise NotImplementedError("Only supports both inputs to be torch tensors or numpy arrays")

    nom = fn_abs(a - b)
    denom = fn_abs(a) + fn_abs(b)
    if axis is not None:
        if len(a.shape) == 1:
            return nom / denom
        return fn_mean(nom / denom, axis=axis)
    return fn_mean(nom / denom)

# TODO: Make pytorch-independent
def mae(a, b):
    return torch.mean(torch.abs(a - b))