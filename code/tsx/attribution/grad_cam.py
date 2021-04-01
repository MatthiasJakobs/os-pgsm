import numpy as np
import torch
from torch.autograd import grad
from tsx.utils import prepare_for_pytorch

def Grad_CAM(x, class_id, model, normalize=True):
    # if not hasattr(model, 'get_features'):
    #     raise Exception("Model {} needs function `get_features` to calculate Grad_CAM".format(model.__class__.__name__))
    if not hasattr(model, 'reset_gradients'):
        raise Exception("Model {} needs function `reset_gradients` to calculate Grad_CAM".format(model.__class__.__name__))

    x = prepare_for_pytorch(x)
    cams = []

    # TODO: For now, do this per item. Batching needs a different way of computing the gradients
    for i in range(x.shape[0]):
        model.reset_gradients() # TODO: Is this necessary?
        out = model.forward(x[i].unsqueeze(0), return_intermediate=True)
        feats = out['feats']
        logits = out['logits']

        if model.forecaster:
            grads = grad(logits, feats)[0].squeeze()
        elif model.classifier:
            grads = grad(logits[..., class_id[i]], feats)[0]
        else:
            raise RuntimeError("Model {} is of unsupported type (needs to be classifier or forecaster".format(model))

        a = grads.detach()
        A = feats.detach()

        cams.append(torch.sum(torch.nn.functional.relu(a * A), axis=1).unsqueeze(0))

    cams = torch.cat(cams, axis=0).squeeze()
    if cams.shape[0] != 1:
        cams = cams.squeeze()

    if normalize:
        if len(cams.shape) == 1:
            cams = cams / torch.max(cams)
        else:
            cams = (cams.T / torch.max(cams, axis=1)[0]).T

    return cams

