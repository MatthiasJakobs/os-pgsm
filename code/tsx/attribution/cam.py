import numpy as np
from tsx.utils import prepare_for_pytorch

def ClassActivationMaps(x, class_id, model, normalize=True):
    if not hasattr(model, 'get_features'):
        raise Exception("Model {} needs function `get_features` to calculate CAM".format(model.__class__.__name__))
    if not hasattr(model, 'get_class_weights'):
        raise Exception("Model {} needs function `get_class_weights` to calculate CAM".format(model.__class__.__name__))

    x = prepare_for_pytorch(x)
    x = model.get_features(x, numpy=True) # S_k(x) ^= x[:, k, x]
    clz_weights = model.get_class_weights(numpy=True)[class_id] # w^{class_id}_k 
    if len(clz_weights.shape) == 1:
        clz_weights = np.expand_dims(clz_weights, 0)
    batch_size = x.shape[0]
    nr_features = x.shape[-1]

    batch_cam = np.zeros((batch_size, nr_features))

    # Can we do this in parallel for each element? e.g. by using batched matrix multiplication?
    for i in range(batch_size):
        sample_i = x[i]
        for m in range(sample_i.shape[0]):  # Zip loop?
            batch_cam[i, :] += clz_weights[i, m] * sample_i[m]

    if normalize:
        batch_cam = ((batch_cam.T - np.min(batch_cam, axis=1)) / (np.max(batch_cam, axis=1) - np.min(batch_cam, axis=1))).T
    return batch_cam
