import torch
import numpy as np
import matplotlib.pyplot as plt

from experiments import *
from compositors import *
from datasets.utils import windowing
from viz import plot_compositor_results
from viz import model_colors
from tsx.visualizations import calc_optimal_grid
from tsx.metrics import smape

def viz_rocs(comp, subset_indices=None):
    fig = plt.figure(figsize=(10,6))

    if subset_indices is None:
        subset_indices = list(range(12))

    rows, cols = calc_optimal_grid(len(subset_indices))

    for n, i in enumerate(subset_indices):
        ax = fig.add_subplot(rows, cols, n+1)
        rocs = comp.rocs[i]
        ax.set_title(list(single_models.keys())[i])
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])
        if len(rocs) == 0:
            continue
        mu = torch.mean(torch.cat(rocs))
        s = torch.std(torch.cat(rocs))
        max_length = max([len(w) for w in rocs])

        ax.set_xlabel("$t$")
        ax.set_ylabel("$y$")

        for r in rocs:
            ax.plot(r, color=model_colors[i], alpha=0.5)

    fig.tight_layout()
    fig.savefig("plots/explainability/abnormal_viz_rocs.png")
    plt.close(fig)

def train_comp(models, comp, x_val, X_test, lag):
    if len(x_val.shape) == 1:
        preds = comp.run(x_val, X_test, big_lag=lag_mapping[str(lag)])
    else:
        preds = comp.run(x_val, X_test)

    return comp, preds

def get_comp_abnormal(seed=0, lag=5):
    torch.manual_seed(seed)
    np.random.seed(seed)

    models = []
    d_name = "AbnormalHeartbeat"
    ds = implemented_datasets[d_name]['ds']()
    X = ds.torch()
    [x_train, x_val_small], [_, _], x_val_big, X_val, X_test = windowing(X, train_input_width=lag, val_input_width=lag_mapping[str(lag)], use_torch=True)

    single_names = []
    for m_name in single_models.keys():
        if m_name not in ['lstm_a', 'adaptive_mixture']:
            single_names.append(m_name)
            models.append(load_model(m_name, d_name, lag, lag))

    comp1 = GC_Large(models, lag, lag_mapping[str(lag)])
    comp1, preds = train_comp(models, comp1, x_val_big, X_test, lag)

    return {
        "comp": comp1,
        "lag": lag,
        "x_val_big": x_val_big,
        "X_val": X_val,
        "X_test": X_test,
    }

def plot_viz_rocs():
    res = get_comp_abnormal(lag=10)
    comp1 = res['comp']
    viz_rocs(comp1)

def plot_compositor_selection_11():

    res = get_comp_abnormal()
    comp1 = res['comp']
    X_test = res['X_test']
    X_val = res['X_val']
    lag = res['lag']

    x_input = X_val[-lag:].unsqueeze(0).unsqueeze(0).float()
    y_output = X_test[0]

    #viz_rocs(comp1, subset_indices=[0, 5, 7, 8])
    #plot_compositor_results(comp1, [100, 150], preds, X_test, single_names, d_name, "GC_Large", lag)
    #plot_compositor_selection(comp1, x_input, y_output)

    # Plot shows: Why was which model chosen to predict point x
    best_forecaster, r = comp1.find_best_forecaster(x_input, return_closest_roc=True)
    best_prediction = None

    for i in range(len(comp1.models)):
        prediction = comp1.models[i].predict(x_input)
        l = smape(np.concatenate([np.squeeze(x_input), np.expand_dims(prediction, 0)]), np.concatenate([np.squeeze(x_input), np.expand_dims(y_output.numpy(), 0)]))
        if i == best_forecaster:
            best_prediction = np.squeeze(prediction)
            print("best forecaster ({}):".format(i), l)
        else:
            print("model {}:".format(i), l)

    x = np.squeeze(x_input)
    fig = plt.figure(figsize=(10,6))
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.plot(np.arange(len(x)), x, "k")
    ax1.plot([len(x) - 1, len(x)], [x[-1], y_output], "g-.")
    ax1.plot([len(x) - 1, len(x)], [x[-1], best_prediction], "r--")
    ax1.set_title("Prediction and Ground truth for forecaster #11")
    ax1.set_ylabel("$y$")
    ax1.set_xlabel("$t$")

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(r)
    ax2.set_title("Closest time-series in ROC for forecaster #11")
    ax2.set_ylabel("$y$")
    ax2.set_xlabel("$t$")

    fig.tight_layout()
    fig.savefig("plots/explainability/compositor_selection.pdf")
    plt.close(fig)

plot_viz_rocs()
# plot_compositor_results(comp1, [100, 150], preds, X_test, single_names, d_name, "GC_Large", lag)
#plot_compositor_selection_11()