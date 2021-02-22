import numpy as np 
import torch
import torch.nn as nn
import torch.optim as optim
from tsx.attribution import Grad_CAM
from tsx.models.forecaster import Shallow_FCN, Shallow_CNN_RNN
from tsx.datasets.ucr import load_itapowdem
from tsx.visualizations import plot_cam
from tsx.metrics import smape
import matplotlib.pyplot as plt 

# TODO: Preprocessing
# TODO: Support preprocessing from sklearn as paramters
ds = load_itapowdem()
A, _ = ds.torch(train=True)
B, _ = ds.torch(train=False)

X = torch.cat((A, B), dim=0)


def split_timeseries_percent(X, percentages):
    total_length = len(X)
    splits = []
    last_segment = 0
    for i, p in enumerate(percentages):
        segment_size = int(np.floor(total_length * p))
        splits.append(X[last_segment:(last_segment+segment_size), ])
        last_segment += segment_size

    return splits

def compare_logs(log_a, log_b, mse_baseline=None, smape_baseline=None):
    plt.figure()

    plt.subplot(1, 3, 1)
    plt.title("train_loss")
    plt.plot(log_a.index, log_a['train_loss'], color="blue", label="CNN+LSTM")
    plt.plot(log_b.index, log_b['train_loss'], color="green", label="CNN")
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.title("val_mse")
    plt.plot(log_b.index, log_b['val_loss'], color="blue", label="CNN+LSTM")
    plt.plot(log_a.index, log_a['val_loss'], color="green", label="CNN")
    plt.hlines(mse_bseline, 0, log_a.index[-1], colors="red", label="Baseline")
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.title("val_smape")
    plt.plot(log_b.index, log_b['val_smape'], color="blue", label="CNN+LSTM")
    plt.plot(log_a.index, log_a['val_smape'], color="green", label="CNN")
    plt.hlines(smape_baseline, 0, log_a.index[-1], colors="red", label="Baseline")
    plt.legend()

    plt.show()


intervals_percent = [.7, .2, .1]
x_train, x_val, x_test = split_timeseries_percent(X, intervals_percent)

y_train = x_train[..., -1]
y_val = x_val[..., -1]
y_test = x_test[..., -1]

x_train = x_train[..., :-1]
x_val = x_val[..., :-1]
x_test = x_test[..., :-1]

epochs = 50

model_a = Shallow_CNN_RNN(batch_size=20, epochs=epochs, hidden_states=30)
model_b = Shallow_FCN(batch_size=20, epochs=epochs)

logs_a = model_a.fit(x_train, y_train, X_val=x_val, y_val=y_val)
logs_b = model_b.fit(x_train, y_train, X_val=x_val, y_val=y_val)

baseline_prediction = x_val[..., -1]
smape_baseline = smape(baseline_prediction, y_val)
mse_bseline = torch.nn.MSELoss()(baseline_prediction, y_val)

print(logs_a)

compare_logs(logs_a, logs_b, smape_baseline=smape_baseline, mse_baseline=mse_bseline)

# loss_fn = nn.MSELoss()
# test_preds = torch.from_numpy(model.predict(x_test))
# print(loss_fn(test_preds, y_test.squeeze()))

# model = TimeSeries1DNet(n_classes=2, epochs=20)
# model.fit(x_train, y_train, X_test=x_test, y_test=y_test)

# example_x = x_test[10:13]
# example_prediction = model.predict(example_x)

# attr = Grad_CAM(example_x, example_prediction, model)

# plot_cam(example_x, attr)