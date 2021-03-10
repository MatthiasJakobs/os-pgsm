import torch
import numpy as np

def sliding_split(x, lag, use_torch=False):
    assert len(x.shape) == 1

    X = np.zeros((len(x)-lag, lag))
    y = np.zeros(len(X))

    for i in range(len(X)):
        X[i] = x[i:(i+lag)]
        y[i] = x[(i+lag)]

    if use_torch:
        return torch.from_numpy(X).float(), torch.from_numpy(y)

    return X, y

def equal_split(x, lag, use_torch=False):
    if len(x.shape) == 2:
        x = x.reshape(-1)

    X = torch.zeros((int(len(x)/lag), lag))
    y = torch.zeros(int((len(x)/lag)))

    for i, idx in enumerate(range(0, len(x)-lag, lag)):
        X[i] = x[idx:idx+lag]
        y[i] = x[idx+lag]

    if not use_torch:
        return X.numpy(), y.numpy()
    else:
        return X, y



def windowing(X, train_input_width=3, val_input_width=9, offset=0, label_width=1, use_torch=True, split_percentages=(0.4, 0.55, 0.05)):

    assert val_input_width % train_input_width == 0

    def _apply_window(x, input_width):
        x_w = []
        y_w = []

        for i in range(0, len(x), (input_width+offset+label_width)):
            x_w.append(np.expand_dims(x[i:(i+input_width)], 0))
            y_w.append(np.expand_dims(x[(i+input_width+offset):(i+input_width+offset+label_width)], 0))

        if x_w[-1].shape != x_w[0].shape:
            x_w = x_w[:-1]
            y_w = y_w[:-1]

        if y_w[-1].shape[-1] == 0:
            x_w = x_w[:-1]
            y_w = y_w[:-1]

        x_w = np.concatenate(x_w, axis=0)
        y_w = np.concatenate(y_w, axis=0)

        if use_torch:
            x_w = torch.from_numpy(x_w).float().unsqueeze(1)
            y_w = torch.from_numpy(y_w).float()

        return x_w, y_w

    X_train = X[0:int(split_percentages[0] * len(X))]
    X_val = X[len(X_train):len(X_train)+int(split_percentages[1] * len(X))]
    X_test = X[(len(X_train) + len(X_val)):(len(X_val) + len(X_train))+int(split_percentages[1] * len(X))]

    x_train_small, y_train_small = _apply_window(X_train, train_input_width)
    x_val_small, y_val_small = _apply_window(X_val, train_input_width)

    x_val = []

    for i in range(0, len(X_val), val_input_width):
        x_val.append(X_val[i:(i+val_input_width)].unsqueeze(0))

    x_val = torch.cat(x_val[:-1], 0)

    if use_torch and isinstance(X_test, np.ndarray):
        X_test = torch.from_numpy(X_test).unsqueeze(1).float()
    if use_torch and isinstance(x_val, np.ndarray):
        x_val = torch.from_numpy(x_val).unsqueeze(1).float()

    return [x_train_small, x_val_small], [y_train_small, y_val_small], x_val, X_test