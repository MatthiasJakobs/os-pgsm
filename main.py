import numpy as np 
import torch
import torch.nn as nn
import torch.optim as optim
from tsx.attribution import Grad_CAM
from tsx.models.forecaster import *
from tsx.datasets.ucr import load_itapowdem
from tsx.visualizations import plot_cam
from tsx.metrics import smape, mae
from tsx.utils import to_numpy
import matplotlib.pyplot as plt 
from tsx.distances import dtw
from tqdm import tqdm, trange

from viz import plot_test_preds, plot_cams, plot_compositor_results
from datasets import Jena_Climate, Bike_Total_Rents, Bike_Temperature, Bike_Registered
from datasets.utils import windowing
from compositors import *


def evaluate_test(model, x_test, reuse_predictions=False, lags=5):
    predictions = np.zeros_like(x_test)

    x = x_test[:lag]
    predictions[:lag] = x

    for x_i in range(lag, len(x_test)):
        if reuse_predictions:
            x = torch.from_numpy(predictions[x_i-lag:x_i]).unsqueeze(0)
        else:
            x = x_test[x_i-lag:x_i].unsqueeze(0)

        predictions[x_i] = np.squeeze(model.predict(x.unsqueeze(0)))
        
    error = smape(x_test.numpy(), predictions)
    return predictions, error

exps = [
    # {"name": "bike_total_rents", "ds": Bike_Total_Rents},
    # {"name": "bike_registered", "ds": Bike_Registered},
    {"name": "bike_temperature", "ds": Bike_Temperature},
    # {"name": "jena", "ds": Jena_Climate},
]

for experiment in exps:

    print(experiment)
    X = experiment["ds"]().torch()
    name = experiment["name"]

    lag = 5
    [x_train, x_val], [y_train, y_val], x_val_big, x_test = windowing(X, train_input_width=lag, val_input_width=lag*lag, use_torch=True)

    # TODO: Otherwise, evaluation is to long
    x_test = x_test[:200]

    epochs = 35
    batch_size = 300
    # TODO: Store model
    model_a = Shallow_CNN_RNN(batch_size=50, epochs=None, ts_length=lag, hidden_states=10)
    model_b = Shallow_FCN(batch_size=50, ts_length=lag, epochs=None)
    model_c = AS_LSTM_01(batch_size=50, epochs=None, ts_length=lag, hidden_states=10)
    model_d = AS_LSTM_02(batch_size=50, epochs=None, ts_length=lag, hidden_states=10)

    model_a.load_state_dict(torch.load("models/bike_total_rents_rnn.pth"))
    model_b.load_state_dict(torch.load("models/bike_total_rents_cnn.pth"))
    model_c.load_state_dict(torch.load("models/bike_total_rents_as01.pth"))
    model_d.load_state_dict(torch.load("models/bike_total_rents_as02.pth"))

    preds_a, smape_a = evaluate_test(model_a, x_test, reuse_predictions=False, lags=lag)
    preds_b, smape_b = evaluate_test(model_b, x_test, reuse_predictions=False, lags=lag)
    preds_c, smape_c = evaluate_test(model_c, x_test, reuse_predictions=False, lags=lag)
    preds_d, smape_d = evaluate_test(model_d, x_test, reuse_predictions=False, lags=lag)

    models = [model_a, model_b, model_c, model_d]

    print("RNN", smape_a)
    print("CNN", smape_b)
    print("AS_01", smape_c)
    print("AS_02", smape_d)
    baseline_comp = BaselineCompositor(models, 5)
    preds_base, smape_base = baseline_comp.run(x_val_big, x_test)
    print("Baseline", smape_base)
    plot_compositor_results(baseline_comp, preds_base, x_test, ["RNN", "CNN", "AS01", "AS02"], name, "Baseline")

    our_comp = GCCompositor(models, 5)
    preds_o, smape_o = our_comp.run(x_val_big, x_test)
    print("Our", smape_o)

    #plot_cams(our_comp, x_val_big, ["RNN", "CNN", "AS01", "AS02"], name, "our")
    plot_compositor_results(our_comp, preds_o, x_test, ["RNN", "CNN", "AS01", "AS02"], name, "our method")

    our_comp = GC_EvenCompositor(models, 5)
    preds_even, smape_even = our_comp.run(x_val_big, x_test)
    print("Our_even", smape_even)

    #plot_cams(our_comp, x_val_big, ["RNN", "CNN", "AS01", "AS02"], name, "our_even")
    plot_compositor_results(our_comp, preds_even, x_test, ["RNN", "CNN", "AS01", "AS02"], name, "our_even")

    plot_test_preds([preds_a, preds_b, preds_base, preds_o, preds_even], ["RNN", "CNN", "AS01", "AS02", "Baseline", "Our", "Our_even"], [smape_a, smape_b, smape_c, smape_d, smape_base, smape_o, smape_even], x_test, "{}_test_eval".format(name), first_n=len(x_test)-1)