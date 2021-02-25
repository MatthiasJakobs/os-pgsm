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
from tqdm import tqdm

def split_timeseries_percent(X, percentages):
    total_length = len(X)
    splits = []
    last_segment = 0
    for i, p in enumerate(percentages):
        segment_size = int(np.floor(total_length * p))
        splits.append(X[last_segment:(last_segment+segment_size), ])
        last_segment += segment_size

    return splits

def windowing(X, input_width=3, offset=0, label_width=1, split_percentages=(0.7, 0.2, 0.1)):
    nr_output_samples = int(np.floor(len(X) / (input_width + label_width + offset)))

    x_w = []
    y_w = []

    for i in range(0, len(X), (input_width+offset+label_width)):
        x_w.append(np.expand_dims(X[i:(i+input_width)], 0))
        y_w.append(np.expand_dims(X[(i+input_width+offset):(i+input_width+offset+label_width)], 0))

    if x_w[-1].shape != x_w[0].shape:
        x_w = x_w[:-1]
        y_w = y_w[:-1]

    if y_w[-1].shape[-1] == 0:
        x_w = x_w[:-1]
        y_w = y_w[:-1]

    x_w = np.concatenate(x_w, axis=0)
    y_w = np.concatenate(y_w, axis=0)

    return split_timeseries_percent(x_w, split_percentages), split_timeseries_percent(y_w, split_percentages)

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
        errors[i] = dist(y, pred.reshape(-1, 1), axis=1)
    
    return np.argsort(-errors, axis=0)

def get_gradcam(models, preds, x, y):
    cams = []
    for i, m in enumerate(models):
        attr = Grad_CAM(x, preds[i], m)
        cams.append(attr)

    return cams

# TODO: This is normalized, i.e., think about this threshold more?
# Idea: Smooth out region of competence
#       - If a single point is surrounded by points below threshold, it will be deleted aswell (denoising)
def get_rocs(rankings, predictions, models, validation_data, validation_label):

    rocs = []

    for i, m in enumerate(models):

        # region of competence shape: (nr_of_best, length_of_explanation)
        is_best = rankings[i]
        model_roc = np.zeros((np.sum(is_best), 3)) # TODO: This needs to be more generic

        for x_i in range(model_roc.shape[0]):
            m.reset_gradients() 
            out = m.forward(validation_data[np.where(is_best)][x_i].unsqueeze(0), return_intermediate=True)
            feats = out['feats']
            logits = out['logits']

            grads = torch.autograd.grad(smape(logits, validation_label[np.where(is_best)][x_i]), feats)[0].squeeze()

            a = grads.detach()
            A = feats.detach()

            r = torch.sum(torch.nn.functional.relu(a * A), axis=1).numpy()
            model_roc[x_i] = r / np.max(r)

        rocs.append(model_roc)

    return rocs

def online_forecasting(rocs, models, X, dist=dtw):
    predictions = []

    for x in tqdm(X):

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

        predictions.append(models[best_model].predict(x.unsqueeze(0)))

    return np.array(predictions)
    



# TODO: Support preprocessing from sklearn as paramters
X = Jena_Climate().torch()

[x_train, x_val, x_test], [y_train, y_val, y_test] = windowing(X, input_width=3)
x_train = torch.from_numpy(x_train).unsqueeze(1).float()
x_val = torch.from_numpy(x_val).unsqueeze(1).float()
x_test = torch.from_numpy(x_test).unsqueeze(1).float()
y_train = torch.from_numpy(y_train).float()
y_val = torch.from_numpy(y_val).float()
y_test = torch.from_numpy(y_test).float()

epochs = 70
batch_size = 300
# TODO: Store model
model_a = Shallow_CNN_RNN(batch_size=batch_size, epochs=epochs, transformed_ts_length=x_train.shape[-1], hidden_states=100)
model_b = Shallow_FCN(batch_size=batch_size, ts_length=x_train.shape[-1], epochs=epochs)

logs_a = model_a.fit(x_train, y_train, X_val=x_val, y_val=y_val)
logs_b = model_b.fit(x_train, y_train, X_val=x_val, y_val=y_val)

preds_a = model_a.predict(x_val)
preds_b = model_b.predict(x_val)

rankings = rank_models([preds_a, preds_b], y_val)

val_rocs = get_rocs(rankings, [preds_a, preds_b], [model_a, model_b], x_val, y_val)

###################
# Test models
##
# First baseline: Just choose the first model everytime
print("Just using model 1: {}".format(smape(y_test.squeeze().numpy(), model_a.predict(x_test))))
# Second baseline: Just choose the second model everytime
print("Just using model 2: {}".format(smape(y_test.squeeze().numpy(), model_b.predict(x_test))))
# Third baseline: Just predict the last value as the next
print("x+1 = x: {}".format(smape(y_test.squeeze().numpy(), x_test[..., -1].squeeze().numpy())))
##
# Try switching classifiers based on ROCS
print("Using online methods: {}".format(smape(y_test.squeeze().numpy(), online_forecasting(val_rocs, [model_a, model_b], x_test))))
###################

# Visualize train results
#compare_logs(logs_a, logs_b, smape_baseline=smape_baseline, mse_baseline=mse_baseline, mae_baseline=mae_baseline)
# Example Visualizations
#plot_cam(x_val[10:13], gradcams[0][10:13], title="Shallow_CNN_RNN", save_to="../plots/poc_gradcams_rnn.pdf")
#plot_cam(x_val[10:13], gradcams[1][10:13], title="Shallow_FCN", save_to="../plots/poc_gradcams_cnn.pdf")