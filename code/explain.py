import torch
import numpy as np
import matplotlib.pyplot as plt

from experiments import *
from compositors import *
from datasets.utils import windowing
from tsx.visualizations import calc_optimal_grid
from tsx.metrics import smape

background_colors = ["cornflowerblue", "violet", "moccasin", "palegreen", "limegreen", "teal", "lime", "orange", "mediumorchid", "yellow", "lightgray", "darkturquoise"]
model_colors = [
    "lightcoral",
    "firebrick",
    "darkred",
    "lightgreen",
    "limegreen",
    "darkgreen",
    "skyblue",
    "royalblue",
    "navy",
    "orchid",
    "mediumpurple",
    "purple",
]

def plot_compositor_results(comp, x_range, preds, x_test, model_names, ds_name, comp_name, lag, limit_models=None):
    from_x = x_range[0]
    to_x = x_range[1]

    if limit_models is None:
        limit_models = list(range(12))

    x_test = x_test[lag:]
    preds = preds[lag:]

    x_test = x_test[from_x:to_x]
    preds = preds[from_x:to_x]
    ranking = comp.test_forecasters[from_x:to_x]

    background_colors = model_colors
    bg_legends = [False] * len(model_names)
    plt.figure(figsize=(10, 4))
    for i in range(to_x-from_x):
        if ranking[i] in limit_models:
            bg_color = background_colors[ranking[i]]
            if not bg_legends[ranking[i]]:
                plt.axvspan(i-0.5, i+0.5, facecolor=bg_color, alpha=0.3, label=model_names[ranking[i]])
                bg_legends[ranking[i]] = True
            else:
                plt.axvspan(i-0.5, i+0.5, facecolor=bg_color, alpha=0.3)

    plt.plot(x_test, color="dimgray", label="$x_{test}$")
    plt.plot(preds, color="crimson", label="prediction")
    plt.xlim((- 0.5, (to_x-from_x) - 0.5))
    plt.xlabel("$t$")
    plt.ylabel("$y$")
    plt.xticks(np.arange(0, to_x-from_x, step=5), list(range(from_x, to_x, 5)), rotation=70)
    #plt.title("{} on {}".format(comp_name, ds_name))
    plt.legend()
    plt.tight_layout()
    plt.savefig("plots/fig_4_{}_{}.pdf".format(comp_name, ds_name))

def viz_roc_change(rocs1, rocs2, idx, name="fig_1_roc_change"):
    fig = plt.figure(figsize=(10,3))
    max_length_1 = max([len(w) for w in rocs1])
    max_length_2 = max([len(w) for w in rocs2])
    max_length = max(max_length_1, max_length_2)

    ax1 = fig.add_subplot(1,2,1)
    ax1.set_title(["$RoC^{" + str(w) + "}$ before drift" for w in range(12)][idx])
    ax1.get_xaxis().set_ticks(np.arange(max_length))

    for r in rocs1[idx]:
        ax1.plot(r, color=model_colors[idx], alpha=0.5)

    ax2 = fig.add_subplot(1,2,2)
    ax2.set_title(["$RoC^{" + str(w) + "}$ after drift" for w in range(12)][idx])
    ax2.get_xaxis().set_ticks(np.arange(max_length))

    for r in rocs1[idx]:
        ax2.plot(r, color=model_colors[idx], alpha=0.5)
    for r in rocs2[idx]:
        ax2.plot(r, color=model_colors[idx], alpha=0.5)

    fig.tight_layout()
    fig.savefig("plots/{}.pdf".format(name))
    plt.close(fig)

def viz_rocs(comp, subset_indices=None, name="abnormal_viz_rocs"):

    if subset_indices is None:
        fig = plt.figure(figsize=(10,4))
        subset_indices = list(range(12))
        rows = 2
        cols = 6
    else:
        fig = plt.figure(figsize=(10,2))
        rows, cols = calc_optimal_grid(len(subset_indices))
        rows = 1
        cols = len(subset_indices)

    for n, i in enumerate(subset_indices):
        ax = fig.add_subplot(rows, cols, n+1)
        rocs = comp.rocs[i]
        ax.set_title(["$C_{" + str(w) + "}$" for w in range(12)][i])
        if len(rocs) == 0:
            ax.get_xaxis().set_ticks([])
            ax.get_yaxis().set_ticks([])
            continue
        mu = torch.mean(torch.cat(rocs))
        s = torch.std(torch.cat(rocs))
        max_length = max([len(w) for w in rocs])

        ax.get_xaxis().set_ticks(np.arange(max_length))

        for r in rocs:
            ax.plot(r, color=model_colors[i], alpha=0.5)

    fig.tight_layout()
    fig.savefig("plots/{}.pdf".format(name))
    plt.close(fig)

'''
    Helper function to train the compositor 
'''
def train_comp(models, comp, x_val, X_test, lag):
    if len(x_val.shape) == 1:
        preds = comp.run(x_val, X_test, big_lag=lag_mapping[str(lag)])
    else:
        preds = comp.run(x_val, X_test)

    return comp, preds

'''
    Helper function for plots on Cloud Coverage Dataset
'''
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

    comp1 = OS_PGSM_St(models, lag, lag_mapping[str(lag)])
    comp1, preds = train_comp(models, comp1, x_val_big, X_test, lag)

    return {
        "comp": comp1,
        "lag": lag,
        "x_val_big": x_val_big,
        "X_val": X_val,
        "X_test": X_test,
    }


'''
    Helper function for plots on Abnormal Dataset
'''
def get_comp_abnormal(seed=0, lag=5, G=OS_PGSM_St):
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

'''
    Plot showing the Region of Competences on the Abnormal Dataset (Figure 3)
'''
def plot_viz_rocs(subset_indices=None, name="abnormal_viz_rocs"):
    res = get_comp_abnormal(lag=10)
    comp1 = res['comp']
    viz_rocs(comp1, subset_indices=subset_indices, name=name)

'''
    Plot showing how the Region of Competence changes after drift is detected (Figure 1)
'''
def plot_rocs_change():
    res = get_comp_abnormal(lag=10, G=OS_PGSM)
    comp1 = res['comp']
    limit_rocs = 7
    initial_roc = comp1.roc_history[0]
    change_roc = comp1.roc_history[1]
    viz_roc_change(initial_roc, change_roc, limit_rocs)
    #comp1.rocs = change_roc
    comp1.rocs = change_roc
    #viz_rocs(comp1, name="after_drift")

'''
    Plot visualizing why a certain predictor was chosen, and what its prediction was (Figure 2)
'''
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
    fig = plt.figure(figsize=(10,3))
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
    fig.savefig("plots/{}.pdf".format(name))
    plt.close(fig)

'''
    Plot showing where the compositor chose which single model to predict the next datapoint (Figure 4)
'''
def plot_compositor_test(subset, limit_models=None):
    res = get_comp_abnormal()
    comp1 = res['comp']
    #single_names = res['single_names']
    single_names = ["$C_{" + str(w) + "}$" for w in range(12)]
    preds = res['preds']
    X_test = res['X_test']
    lag = res['lag']
    plot_compositor_results(comp1, subset, preds, X_test, single_names, "AbnormalHeartbeat", "OS_PGSM_St", lag, limit_models=limit_models)


if __name__ == "__main__":
    plot_rocs_change() # Figure 1
    plot_compositor_selection_cc(offset=4, best=11, name="fig_2_compositor_cc_offset4") # Figure 2
    plot_viz_rocs(subset_indices=[0, 1, 2, 4, 5, 7, 8, 10], name="fig_3_abnormal_viz_rocs") # Figure 3
    plot_compositor_test([100, 150], limit_models=[0, 8]) # Figure 4
    plot_viz_rocs(subset_indices=None, name="supp_abnormal_complete_rocs") # Figure supplementary material