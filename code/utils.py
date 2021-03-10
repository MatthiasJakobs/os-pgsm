import torch

def stupid_gradcam(logits, features):
    grads = torch.autograd.grad(logits, features)[0].squeeze().detach()

    features = features.detach().squeeze()

    w = torch.mean(grads, axis=1)

    cam = torch.zeros_like(features[0])
    for k in range(features.shape[0]):
        cam += w[k] * features[k]

    return torch.nn.functional.relu(cam).squeeze().numpy()

def simple_gradcam(a, b):
    grads = torch.autograd.grad(a, b)[0].squeeze()
    w = grads.detach()
    #w = torch.mean(w, axis=1)
    A = b.detach()
    #torch.sum(torch.nn.functional.relu(torch.from_numpy(np.tile(test_w.unsqueeze(1).numpy(), (1,5))) * A), axis=1).squeeze().numpy()
    r = torch.sum(torch.nn.functional.relu(w * A), axis=1).squeeze().numpy()

    return r