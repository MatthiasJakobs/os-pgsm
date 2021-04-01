import torch
import numpy as np

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