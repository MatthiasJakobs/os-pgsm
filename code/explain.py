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

def plot_compositor_test(subset, limit_models=None):
    res = get_comp_abnormal()
    comp1 = res['comp']
    #single_names = res['single_names']
    single_names = ["$C_{" + str(w) + "}$" for w in range(12)]
    preds = res['preds']
    X_test = res['X_test']
    lag = res['lag']
    plot_compositor_results(comp1, subset, preds, X_test, single_names, "AbnormalHeartbeat", "GC_Large", lag, limit_models=limit_models)

def viz_roc_change(rocs1, rocs2, idx, name="roc_change"):
    fig = plt.figure(figsize=(10,3))
    max_length_1 = max([len(w) for w in rocs1])
    max_length_2 = max([len(w) for w in rocs2])
    max_length = max(max_length_1, max_length_2)

    ax1 = fig.add_subplot(1,2,1)
    ax1.set_title(["$R^{" + str(w) + "}$ before drift" for w in range(12)][idx])
    ax1.get_xaxis().set_ticks(np.arange(max_length))

    for r in rocs1[idx]:
        ax1.plot(r, color=model_colors[idx], alpha=0.5)

    ax2 = fig.add_subplot(1,2,2)
    ax2.set_title(["$R^{" + str(w) + "}$ after drift" for w in range(12)][idx])
    ax2.get_xaxis().set_ticks(np.arange(max_length))

    for r in rocs1[idx]:
        ax2.plot(r, color=model_colors[idx], alpha=0.5)
    for r in rocs2[idx]:
        ax2.plot(r, color=model_colors[idx], alpha=0.5)

    fig.tight_layout()
    fig.savefig("plots/explainability/{}.pdf".format(name))
    plt.close(fig)

def viz_rocs(comp, subset_indices=None, name="abnormal_viz_rocs"):
    fig = plt.figure(figsize=(10,4))

    if subset_indices is None:
        subset_indices = list(range(12))
        rows = 2
        cols = 6
    else:
        rows, cols = calc_optimal_grid(len(subset_indices))

    for n, i in enumerate(subset_indices):
        ax = fig.add_subplot(rows, cols, n+1)
        rocs = comp.rocs[i]
        #ax.set_title(list(single_models.keys())[i])
        #ax.set_title(["Model #" + str(i+1) for i in range(len(single_models.keys()))][i])
        ax.set_title(["$C_{" + str(w) + "}$" for w in range(12)][i])
        #ax.get_xaxis().set_ticks([])
        #ax.get_yaxis().set_ticks([])
        if len(rocs) == 0:
            ax.get_xaxis().set_ticks([])
            ax.get_yaxis().set_ticks([])
            continue
        mu = torch.mean(torch.cat(rocs))
        s = torch.std(torch.cat(rocs))
        max_length = max([len(w) for w in rocs])

        # ax.set_xlabel("$t$")
        # ax.set_ylabel("$y$")
        ax.get_xaxis().set_ticks(np.arange(max_length))

        for r in rocs:
            ax.plot(r, color=model_colors[i], alpha=0.5)

    fig.tight_layout()
    fig.savefig("plots/explainability/{}.pdf".format(name))
    plt.close(fig)

def train_comp(models, comp, x_val, X_test, lag):
    if len(x_val.shape) == 1:
        preds = comp.run(x_val, X_test, big_lag=lag_mapping[str(lag)])
    else:
        preds = comp.run(x_val, X_test)

    return comp, preds

def get_comp_cc(seed=0, lag=5):
    torch.manual_seed(seed)
    np.random.seed(seed)

    models = []
    d_name = "CloudCoverage"
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


def get_comp_abnormal(seed=0, lag=5, G=GC_Large):
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

    comp1 = G(models, lag, lag_mapping[str(lag)])
    if isinstance(comp1, BaseAdaptive):
        comp1, preds = train_comp(models, comp1, X_val, X_test, lag)
    else:
        comp1, preds = train_comp(models, comp1, x_val_big, X_test, lag)

    return {
        "comp": comp1,
        "lag": lag,
        "x_val_big": x_val_big,
        "X_val": X_val,
        "X_test": X_test,
        "preds": preds,
        "single_names": single_names
    }

def plot_viz_rocs():
    res = get_comp_abnormal(lag=10)
    comp1 = res['comp']
    viz_rocs(comp1)

def plot_rocs_change():
    res = get_comp_abnormal(lag=10, G=GC_Large_Adaptive_Hoeffding)
    comp1 = res['comp']
    limit_rocs = 7
    initial_roc = comp1.roc_history[0]
    change_roc = comp1.roc_history[1]
    viz_roc_change(initial_roc, change_roc, limit_rocs)
    #comp1.rocs = change_roc
    comp1.rocs = change_roc
    viz_rocs(comp1, name="after_drift")

def plot_compositor_selection_cc(offset=0, seed=0, lag=5, best=11, name="compositor_selection_cc"):

    res = get_comp_cc(seed=seed, lag=lag)
    comp1 = res['comp']
    X_test = res['X_test']
    X_val = res['X_val']
    lag = res['lag']

    x_input = X_test[offset:offset+lag].unsqueeze(0).unsqueeze(0).float()
    y_output = X_test[offset+lag]

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
    fig = plt.figure(figsize=(10,4))
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.plot(offset + np.arange(len(x)), x, "k")
    ax1.plot(offset + np.array([len(x) - 1, len(x)]), [x[-1], y_output], "g-.", label="Ground truth")
    ax1.plot(offset + np.array([len(x) - 1, len(x)]), [x[-1], best_prediction], "r--", label="Prediction")
    ax1.legend()
    ax1.set_title("Prediction using model $C_{" + str(best) + "}$ and ground truth")
    ax1.set_ylabel("$y$")
    ax1.set_xlabel("$t$")

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(r)
    ax2.get_xaxis().set_ticks(np.arange(len(r)))
    ax2.set_title("Closest time-series in ROC of $C_{" + str(best) + "}$")
    ax2.set_ylabel("$y$")
    ax2.set_xlabel("$t$")

    fig.tight_layout()
    fig.savefig("plots/explainability/{}.pdf".format(name))
    plt.close(fig)

def plot_compositor_selection_11(offset=0, seed=0, lag=5, best=11, name="compositor_selection"):

    res = get_comp_abnormal(seed=seed, lag=lag)
    comp1 = res['comp']
    X_test = res['X_test']
    X_val = res['X_val']
    lag = res['lag']

    x_input = X_val[-lag:].unsqueeze(0).unsqueeze(0).float()
    y_output = X_test[0]

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
    ax1.set_title("Prediction and Ground truth for forecaster #{}".format(best))
    ax1.set_ylabel("$y$")
    ax1.set_xlabel("$t$")

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(r)
    ax2.set_title("Closest time-series in ROC for forecaster #{}".format(best))
    ax2.set_ylabel("$y$")
    ax2.set_xlabel("$t$")

    fig.tight_layout()
    fig.savefig("plots/explainability/{}.pdf".format(name))
    plt.close(fig)

#plot_rocs_change()
#plot_compositor_selection_cc(offset=3, best=11, name="compositor_cc_offset3")
#plot_compositor_selection_cc(offset=4, best=11, name="compositor_cc_offset4")
#plot_compositor_test([100, 150], limit_models=[0, 8])