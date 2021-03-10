import matplotlib.pyplot as plt
import numpy as np

from datasets.utils import *
from tsx.utils import to_numpy
from tsx.visualizations import plot_cam

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
    plt.savefig("plots/poc_train_logs.pdf")

def plot_train_log(log, save_path, best_epoch=None):
    plt.figure(figsize=(10,6))
    for c in log.columns:
        if c.startswith("val"):
            linestyle="--"
        else:
            linestyle="-"
        plt.plot(log[c], linestyle, label=c)
        if best_epoch is not None:
            plt.vlines(best_epoch, 0, 1, colors="black")

    plt.xlabel("Epoch")
    plt.legend()
    plt.savefig(save_path)

def plot_test_preds(preds, labels, scores, x_test, name, first_n=30):
    plt.figure(figsize=(10, 6))
    plt.plot(x_test[:first_n], color="black", label="x_test")
    colors = ["red", "green", "blue", "orange", "purple"]
    for i, p in enumerate(preds):
        plt.plot(preds[i][:first_n], color=colors[i], label="{}: {:.5} sMAPE".format(labels[i], scores[i]))

    plt.legend()
    plt.tight_layout()
    plt.savefig("plots/{}.pdf".format(name))

def plot_compositor_results(comp, preds, x_test, model_names, ds_name, comp_name):
    ranking = comp.test_forecasters
    x_test = x_test[5:]
    preds = preds[5:]
    assert len(np.unique(ranking)) <= 4

    background_colors = ["teal", "lime", "orange", "mediumorchid"]
    bg_legends = [False, False, False, False]
    plt.figure()
    for i in range(len(x_test)):
        bg_color = background_colors[ranking[i]]
        if not bg_legends[ranking[i]]:
            plt.axvspan(i-0.5, i+0.5, facecolor=bg_color, alpha=0.3, label=model_names[ranking[i]])
            bg_legends[ranking[i]] = True
        else:
            plt.axvspan(i-0.5, i+0.5, facecolor=bg_color, alpha=0.3)

    plt.plot(x_test, color="dimgray", label="$x_{test}$")
    plt.plot(preds, color="crimson", label="prediction")
    plt.xlim((-0.5, len(x_test) - 0.5))
    plt.xlabel("$t$")
    plt.ylabel("$y$")
    plt.title("{} on {}".format(comp_name, ds_name))
    plt.legend()
    plt.tight_layout()
    plt.savefig("plots/test_{}_{}.pdf".format(comp_name, ds_name))


def plot_cams(compositor, x_val, model_names, ds_name, comp_name):

    nr_rows = 2
    nr_cols = 2
    nr_plots = nr_rows * nr_cols

    subset_inds = np.random.permutation(compositor.cams.shape[1])[:100]
    subset_x = x_val[subset_inds]

    for m in range(len(compositor.models)):
        subset_cams = compositor.cams[m][subset_inds] 

        xs = []
        cs = []

        i = 0
        nonzero = 0
        while nonzero < nr_plots:
            x = to_numpy(sliding_split(subset_x[i], compositor.lag)[0])[0]
            c = to_numpy(subset_cams[i])[0]
            i += 1
            if np.sum(c) == 0:
                continue
            c = c / np.max(c)

            xs.append(np.expand_dims(x, axis=0))
            cs.append(np.expand_dims(c, axis=0))

            nonzero += 1

        plot_cam(np.concatenate(xs, axis=0), np.concatenate(cs, axis=0), title="{} - {} ({})".format(comp_name, model_names[m], ds_name), save_to="plots/cams_{}_{}_{}.pdf".format(comp_name, model_names[m], ds_name))