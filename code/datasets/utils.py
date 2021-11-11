import torch
import numpy as np
import pandas as pd

def sliding_split(x, lag, z=1, use_torch=False):
    assert len(x.shape) == 1

    X = []
    y = []

    for i in range(0, len(x), z):
        if (i+lag) >= len(x):
            break
        #print(f"{i} From {i} to {i+lag}")
        X.append(x[i:(i+lag)].reshape(1, -1))
        y.append(x[(i+lag)])

    X = np.concatenate(X, axis=0)
    y = np.array(y)

    if use_torch:
        return torch.from_numpy(X).float(), torch.from_numpy(y)

    return X, y

def equal_split(x, lag, use_torch=False):
    return sliding_split(x, lag, z=lag, use_torch=use_torch)


def roc_matrix(rocs, z=1):
    lag = rocs.shape[-1]
    m = np.ones((len(rocs), lag + len(rocs) * z - z)) * np.nan

    offset = 0
    for i, roc in enumerate(rocs):
        m[i, offset:(offset+lag)] = roc
        offset += z

    return m

def roc_mean(roc_matrix):
    summation_matrix = roc_matrix.copy()
    summation_matrix[np.where(np.isnan(roc_matrix))] = 0
    sums = np.sum(summation_matrix, axis=0)
    nonzeros = np.sum(np.logical_not(np.isnan(roc_matrix)), axis=0)
    return sums / nonzeros


def train_test_split(x, split_percentages=(0.8, 0.2)):
    assert not isinstance(x, pd.core.frame.DataFrame)
    L = len(x)
    start_train = 0
    stop_train = int(np.floor(L * split_percentages[0]))
    start_test = stop_train
    stop_test = -1

    return x[start_train:stop_train], x[start_test:stop_test]

def _apply_window(x, input_width, offset=0, label_width=1, use_torch=True):
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

def _val_split(X_val, lag, big_lag, use_torch=True):
    x_val = []

    for i in range(0, len(X_val), big_lag):
        x_val.append(X_val[i:(i+big_lag)].unsqueeze(0))

    x_val = torch.cat(x_val[:-1], 0)
    if use_torch and isinstance(x_val, np.ndarray):
        x_val = torch.from_numpy(x_val).unsqueeze(1).float()

    return x_val

def windowing(X, train_input_width=3, val_input_width=9, offset=0, label_width=1, use_torch=True, split_percentages=(0.5, 0.25, 0.25)):

    assert val_input_width % train_input_width == 0

    p_train_val = split_percentages[0] + split_percentages[1]
    p_test = split_percentages[2]
    X_train_and_val, X_test = train_test_split(X, split_percentages=(p_train_val, p_test))
    X_train, X_val = train_test_split(X_train_and_val, split_percentages=(2.0/3.0, 1.0/3.0))

    x_train_small, y_train_small = _apply_window(X_train, train_input_width, offset=offset, label_width=label_width, use_torch=use_torch)
    x_val_small, y_val_small = _apply_window(X_val, train_input_width)

    x_val_big = _val_split(X_val, train_input_width, val_input_width, use_torch=use_torch)

    if use_torch and isinstance(X_test, np.ndarray):
        X_test = torch.from_numpy(X_test).unsqueeze(1).float()

    return [x_train_small, x_val_small], [y_train_small, y_val_small], x_val_big, X_val, X_test

def simple_windowing(X):
    lag = 5
    [x_train_small, x_val_small], [y_train_small, y_val_small], x_val_big, X_val, X_test = windowing(X, train_input_width=lag, val_input_width=25, use_torch=True)

    X_train_val = torch.cat([x_train_small, x_val_small], axis=0)
    y_train_val = torch.cat([y_train_small, y_val_small], axis=0)

    return [X_train_val, y_train_val], X_test

def get_all_M4(lag):
    if lag != 5:
        raise Exception("Currently only supports lag 5")

    with open("code/datasets/m4_lag5.txt") as f:
        datasets = [d.replace("\n", "") for d in f.readlines()]

    return datasets


    
