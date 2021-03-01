import numpy as np 
import torch
import torch.nn as nn
import torch.optim as optim
from tsx.attribution import Grad_CAM
from tsx.models.forecaster import Shallow_FCN, Shallow_CNN_RNN
from tsx.datasets.ucr import load_itapowdem
from tsx.visualizations import plot_cam
from tsx.metrics import smape, mae
from tsx.utils import to_numpy
import matplotlib.pyplot as plt 
from wheatprice import WheatPrice
from jena_climate import Jena_Climate
from ranker import Ranker
from tsx.distances import dtw
from tqdm import tqdm, trange

def split_up_dataset(x, lag=5):
    splits = []
    if len(x.shape) == 1:
        if isinstance(x, torch.Tensor):
            splits.append(x[:lag].unsqueeze(0))
        if isinstance(x, np.ndarray):
            splits.append(np.expand_dims(x[:lag], 0))
        for i in range(1, len(x)-lag+1):
            slic = x[i:(i+lag)]
            if len(slic.shape) == 1:
                if isinstance(slic, torch.Tensor):
                    slic = slic.unsqueeze(0)
                if isinstance(slic, np.ndarray):
                    slic = np.expand_dims(slic, 0)
            splits.append(slic)
    if len(x.shape) == 2:
        # Take shapes (n, m) to (k, 5)
        for x_i in x:
            s = split_up_dataset(x_i, lag=lag)
            splits.append(s)

    if isinstance(splits[0], np.ndarray):
        return np.concatenate(splits, axis=0)
    if isinstance(splits[0], torch.Tensor):
        return torch.cat(splits, dim=0)


def split_timeseries_percent(X, percentages):
    total_length = len(X)
    splits = []
    last_segment = 0
    for i, p in enumerate(percentages):
        segment_size = int(np.floor(total_length * p))
        splits.append(X[last_segment:(last_segment+segment_size), ])
        last_segment += segment_size

    return splits

def windowing(X, train_input_width=3, val_input_width=9, offset=0, label_width=1, use_torch=True, split_percentages=(0.4, 0.55, 0.05)):

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

    x_train, y_train = _apply_window(X_train, train_input_width)
    #x_val, y_val = _apply_window(X_val, val_input_width)

    x_val = []
    y_val = None # TODO

    for i in range(0, len(X_val), val_input_width):
        x_val.append(X_val[i:(i+val_input_width)].unsqueeze(0))

    x_val = torch.cat(x_val[:-1], 0)

    if use_torch and isinstance(X_test, np.ndarray):
        X_test = torch.from_numpy(X_test).unsqueeze(1).float()
    if use_torch and isinstance(x_val, np.ndarray):
        x_val = torch.from_numpy(x_val).unsqueeze(1).float()

    return [x_train, x_val], [y_train, y_val], X_test


def compare_logs(log_a, log_b, mse_baseline=None, mae_baseline=None, smape_baseline=None):

    _, axes = plt.subplots(2, 2, figsize=(10, 6))
    axes.flat[0].set_title("train_mse")
    axes.flat[0].plot(log_a.index, log_a['train_mse'], color="blue", label="CNN+LSTM")
    axes.flat[0].plot(log_b.index, log_b['train_mse'], color="green", label="CNN")
    axes.flat[0].legend()

    axes.flat[1].set_title("val_mae")
    axes.flat[1].plot(log_b.index, log_b['val_mae'], color="blue", label="CNN+LSTM")
    axes.flat[1].plot(log_a.index, log_a['val_mae'], color="green", label="CNN")
    axes.flat[1].hlines(mae_baseline, 0, log_a.index[-1], colors="red", label="Baseline")
    axes.flat[1].legend()

    axes.flat[2].set_title("val_smape")
    axes.flat[2].plot(log_b.index, log_b['val_smape'], color="blue", label="cnn+lstm")
    axes.flat[2].plot(log_a.index, log_a['val_smape'], color="green", label="cnn")
    axes.flat[2].hlines(smape_baseline, 0, log_a.index[-1], colors="red", label="baseline")
    axes.flat[2].legend()

    axes.flat[3].set_title("val_mse")
    axes.flat[3].plot(log_b.index, log_b['val_mse'], color="blue", label="cnn+lstm")
    axes.flat[3].plot(log_a.index, log_a['val_mse'], color="green", label="cnn")
    axes.flat[3].hlines(mse_baseline, 0, log_a.index[-1], colors="red", label="baseline")
    axes.flat[3].legend()

    plt.tight_layout()
    plt.savefig("../plots/poc_train_logs.pdf")

def rank_models(model_predictions, y, dist=smape):
    # TODO: Assumes two models for now, make this more general
    y = to_numpy(y)
    errors = np.zeros((len(model_predictions), len(y)))

    for i, pred in enumerate(model_predictions):
        errors[i] = dist(y, pred, axis=1)
    
    return np.argsort(-errors, axis=0)

def get_gradcam(models, preds, x, y):
    cams = []
    for i, m in enumerate(models):
        attr = Grad_CAM(x, preds[i], m)
        cams.append(attr)

    return cams

def split_array_at_zero(arr):
    indices = np.where(arr != 0)[0]
    splits = []
    i = 0
    while i+1 < len(indices):
        start = i
        stop = start
        j = i+1
        while j < len(indices):
            if indices[j] - indices[stop] == 1:
                stop = j
                j += 1
            else:
                break

        if start != stop:
            splits.append((indices[start], indices[stop]))
            i = stop
        else:
            i += 1
        # if t < len(indices) and indices[t] - indices[f] == 1:
        #     t += 1

        # if t != f:
        #     splits.append((f, t))

        # f = t
        # t += 1

    # for i in range(len(indices)):
    #     start = indices[i]
    #     current = start
    #     for j in range(i+1, len(indices)):
    #         if indices[j] - current == 1:
    #             current = indices[j]
    #         else:
    #             break
    #     if start != current:
    #         splits.append((start, current))

    #     if current == indices[-1]:
    #         break

    return splits

    # if len(indices) == 1 :
    #     return splits
    # for subarray in np.split(arr, indices):
    #     if len(subarray) < 2:
    #         continue
    #     if subarray[0] == 0:
    #         subarray = subarray[1:]
    #     if len(subarray) > 1:
    #         splits.append(subarray)

    # return splits

# Idea: Smooth out region of competence
#       - If a single point is surrounded by points below threshold, it will be deleted aswell (denoising)
#def get_rocs(rankings, predictions, models, validation_data, validation_label):
def get_rocs(methods, xs, threshold=.5):

    rocs = []
    for m, x_val in zip(methods, xs):
        rocs_i = []
        for region, x in zip(m, x_val):
            max_r = np.max(region) 
            normalized = region / max_r
            after_threshold = normalized * (normalized > threshold)

            if len(np.nonzero(after_threshold)[0]) > 0:
                indidces = split_array_at_zero(after_threshold)
                for (f, t) in indidces:
                    rocs_i.append(x[f:(t+1)])
                #rocs_i.extend(split_array_at_zero(after_threshold)) # TODO: Do not return the ROC as gradcam but use gradcam to finetune the ROC

        rocs.append(rocs_i)

    return rocs

def evaluate_test(model, x_test, lags=5):
    predictions = np.zeros_like(x_test)

    x = x_test[:lag]
    predictions[:lag] = x

    for x_i in range(lag, len(x_test)):
        x = torch.from_numpy(predictions[x_i-lag:x_i]).unsqueeze(0)
        predictions[x_i] = np.squeeze(model.predict(x.unsqueeze(0)))
        
    error = smape(x_test.numpy(), predictions)
    return predictions, error


def online_forecasting(rocs, models, X, lag=5, dist=dtw):
    predictions = np.zeros_like(X)

    x = X[:lag]
    predictions[:lag] = x

    for x_i in trange(lag, len(X)):
        x = torch.from_numpy(predictions[x_i-lag:x_i]).unsqueeze(0)

        best_model = -1
        best_indice = None
        smallest_distance = 1e8

        for i, m in enumerate(models):

            cam = Grad_CAM(x.unsqueeze(0), None, m).numpy()
            for r in rocs[i]:
                distance = dist(r, cam)
                if distance < smallest_distance:
                    best_model = i
                    smallest_distance = distance

        predictions[x_i] = models[best_model].predict(x.unsqueeze(0))

    error = smape(predictions, X.numpy())

    return np.array(predictions), error

def evaluate_model_on_val(models, x_val, lags=5):
    preds = []
    cs = []
    ls = []
    for m in models:
        predictions = np.squeeze(np.zeros_like(x_val))
        cams = np.squeeze(np.zeros_like(x_val))
        losses = np.zeros(len(x_val))
        for o, x in enumerate(x_val):
            prediction = np.squeeze(np.zeros_like(x))
            cam = np.squeeze(np.zeros_like(x))
            prediction[0:lags] = x[..., 0:lags]
            total_loss = 0
            for i in range(len(prediction)-lags):
                to = i + lags
                p_index = to
                sliced = x[..., i:to]
                res = m.forward(sliced.unsqueeze(0).unsqueeze(0), return_intermediate=True)
                logits = res['logits']
                feats = res['feats']
                pred_smape = smape(logits, x[..., to])
                total_loss += pred_smape.detach().squeeze().item()
                grads = torch.autograd.grad(pred_smape, feats)[0].squeeze()
                prediction[p_index] = logits.detach().squeeze().numpy()

                a = grads.detach()
                A = feats.detach()

                r = torch.sum(torch.nn.functional.relu(a * A), axis=1).squeeze().numpy()

                cam[i:to] = r

            predictions[o] = prediction
            cams[o] = cam
            losses[o] = total_loss

        preds.append(predictions)
        cs.append(cams)
        ls.append(losses)

    return preds, cs, ls

# TODO: Support preprocessing from sklearn as paramters
X = Jena_Climate().torch()

# TODO
lag = 5
[x_train, x_val], [y_train, y_val], x_test = windowing(X, train_input_width=lag, val_input_width=5*lag, use_torch=True)

epochs = 280
batch_size = 300
# TODO: Store model
model_a = Shallow_CNN_RNN(batch_size=batch_size, epochs=epochs, transformed_ts_length=x_train.shape[-1], hidden_states=100)
model_b = Shallow_FCN(batch_size=batch_size, ts_length=x_train.shape[-1], epochs=epochs)

logs_a = model_a.fit(x_train, y_train)
logs_b = model_b.fit(x_train, y_train)

[preds_a, preds_b], [cams_a, cams_b], [losses_a, losses_b] = evaluate_model_on_val([model_a, model_b], x_val)

rankings = (losses_a > losses_b).astype(np.int)

#x_val_split = split_up_dataset(x_val)

better_a_inds = np.where(np.logical_not(rankings))
better_b_inds = np.where(rankings)
better_a = cams_a[better_a_inds]
better_b = cams_b[better_b_inds]

#val_rocs = get_rocs([better_a, better_b], [x_val_split[better_a_inds], x_val_split[better_b_inds]])
val_rocs = get_rocs([better_a, better_b], [x_val[better_a_inds], x_val[better_b_inds]])

preds_a, smape_a = evaluate_test(model_a, x_test, lags=5)
preds_b, smape_b = evaluate_test(model_b, x_test, lags=5)
preds_o, smape_o = online_forecasting(val_rocs, [model_a, model_b], x_test)

print(smape_a)
print(smape_b)
print(smape_o)

#plot_cam(x_val[first_3_cnn], val_rocs[0][:3], title="CNN on validation", save_to="plots/cnn_rocs.pdf")
#plot_cam(x_val[first_3_rnn], val_rocs[1][:3], title="RNN on validation", save_to="plots/rnn_rocs.pdf")

###################
# Test models
##
# First baseline: Just choose the first model everytime
#print("Just using model 1: {}".format(smape(y_test.squeeze().numpy(), model_a.predict(x_test))))
# Second baseline: Just choose the second model everytime
#print("Just using model 2: {}".format(smape(y_test.squeeze().numpy(), model_b.predict(x_test))))
# Third baseline: Just predict the last value as the next
#print("x+1 = x: {}".format(smape(y_test.squeeze().numpy(), x_test[..., -1].squeeze().numpy())))
##
# Try switching classifiers based on ROCS
#print("Using online methods: {}".format(smape(y_test.squeeze().numpy(), online_forecasting(val_rocs, [model_a, model_b], x_test))))
###################

# Visualize train results
#compare_logs(logs_a, logs_b, smape_baseline=smape_baseline, mse_baseline=mse_baseline, mae_baseline=mae_baseline)
# Example Visualizations
#plot_cam(x_val[10:13], gradcams[0][10:13], title="Shallow_CNN_RNN", save_to="../plots/poc_gradcams_rnn.pdf")
#plot_cam(x_val[10:13], gradcams[1][10:13], title="Shallow_FCN", save_to="../plots/poc_gradcams_cnn.pdf")