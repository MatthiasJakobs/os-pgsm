import torch
import numpy as np
import matplotlib.pyplot as plt

from datasets import Bike_Total_Rents
from tsx.models.forecaster import AS_LSTM_01, AS_LSTM_02, Shallow_FCN, Shallow_CNN_RNN
from compositors import GCCompositor
from datasets.utils import windowing

def combinations(n):
    combs = np.array(np.meshgrid(np.arange(n), np.arange(n))).T.reshape(-1, 2)
    combs = np.delete(combs, np.where(combs[:, 0] >= combs[:, 1])[0], axis=0)
    return combs


X = Bike_Total_Rents().torch()

lags = 5

model_a = Shallow_CNN_RNN(batch_size=50, epochs=None, ts_length=lags, hidden_states=10)
model_b = Shallow_FCN(batch_size=50, ts_length=lags, epochs=None)
model_c = AS_LSTM_01(batch_size=50, epochs=None, ts_length=lags, hidden_states=10)
model_d = AS_LSTM_02(batch_size=50, epochs=None, ts_length=lags, hidden_states=10)

model_a.load_state_dict(torch.load("models/bike_total_rents_rnn.pth"))
model_b.load_state_dict(torch.load("models/bike_total_rents_cnn.pth"))
model_c.load_state_dict(torch.load("models/bike_total_rents_as01.pth"))
model_d.load_state_dict(torch.load("models/bike_total_rents_as02.pth"))

names = ["RNN", "CNN", "AS01", "AS02"]
models = [model_a, model_b, model_c, model_d]

threshold_min = 0
threshold_max = 1
step_size = 0.01

thresholds = np.arange(threshold_min, threshold_max, step_size)

for [idx1, idx2] in combinations(len(models)):
    [_, _], [_, _], x_val, x_test = windowing(X, train_input_width=lags, val_input_width=lags*lags, use_torch=True)
    errors = []
    for thresh in thresholds:
        comp = GCCompositor([models[idx1], models[idx2]], lags, threshold=thresh)
        _, error = comp.run(x_val, x_test)
        errors.append(error)

    plt.figure()
    plt.plot(thresholds, errors, "b-")
    plt.title("{} x {}: bike_total_rents".format(names[idx1], names[idx2]))
    plt.savefig("plots/exp_threshold_grid_search/{}x{}_BTR.pdf".format(names[idx1], names[idx2]))


