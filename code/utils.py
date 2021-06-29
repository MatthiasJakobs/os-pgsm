import torch

def gradcam(logits, features):
    grads = torch.autograd.grad(logits, features)[0].squeeze().detach()

    features = features.detach().squeeze()

    w = torch.mean(grads, axis=1)

    cam = torch.zeros_like(features[0])
    for k in range(features.shape[0]):
        cam += w[k] * features[k]

    return torch.nn.functional.relu(cam).squeeze().numpy()