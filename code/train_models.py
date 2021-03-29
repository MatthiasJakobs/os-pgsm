import torch 
import numpy as np
import argparse

from datasets.utils import windowing, _apply_window, train_test_split
from viz import plot_train_log
from experiments import single_models, implemented_datasets, lag_mapping
from datasets import M4_Hourly, M4_Daily, M4_Monthly, M4_Weekly, M4_Quaterly, M4_Yearly
from os.path import exists


def train_model(model, x_train, y_train, x_val, y_val, lag, model_name, ds_name, nr_filters, batch_size, epochs, hidden_states, learning_rate, save_plot=False, verbose=True):
        save_path = "models/{}/{}_lag{}.pth".format(model_name, ds_name, lag)
        if not exists(save_path):
            model_instances = []
            model_logs = []
            model_scores = []
            model_bests = []
            for i in range(5):
                try:
                    if "lstm" in model_name or "adaptive_mixture" in model_name:
                        m = model(lag, batch_size=batch_size, nr_filters=nr_filters, epochs=epochs, ts_length=x_train.shape[-1], hidden_states=hidden_states, learning_rate=learning_rate)
                    else:
                        m = model(batch_size=batch_size, nr_filters=nr_filters, epochs=epochs, ts_length=x_train.shape[-1], hidden_states=hidden_states, learning_rate=learning_rate)
                except TypeError:
                    # TODO: FCN does not accept "hidden_states" variable. Maybe pass configs as dictionary? 
                    m = model(batch_size=batch_size, nr_filters=nr_filters, epochs=epochs, ts_length=x_train.shape[-1], learning_rate=learning_rate)

                logs, best = m.fit(x_train, y_train, X_val=x_val, y_val=y_val, model_save_path=save_path, verbose=verbose)
                model_instances.append(m)
                model_logs.append(logs)
                model_scores.append(logs["val_mse"][best])
                model_bests.append(best)

            # find model which is nearest to average performance
            model_scores = np.array(model_scores)
            perf_average = np.mean(model_scores)
            rel_scores = np.abs(model_scores-perf_average)
            nearest_model_idx = np.argmin(rel_scores)

            logs = model_logs[nearest_model_idx]
            best = model_bests[nearest_model_idx]

            model_instances[nearest_model_idx].save(save_path)

            if save_plot:
                plot_train_log(logs, "plots/train_{}_{}.pdf".format(ds_name, model_name), best_epoch=best)
        else:
            print("{} already trained, skip".format(save_path))

def train_m4_subset(lag=5):
    name_list = ['hourly', 'weekly', 'quaterly', 'daily', 'monthly']
    ds_list = [ M4_Hourly(), M4_Weekly(), M4_Quaterly(), M4_Daily(), M4_Monthly()]

    for name, ds in zip(name_list, ds_list):
        indices = list(range(20))
        for idx in indices:
            padded_idx = ds.train_data.columns[idx]
            ds_train, _ = ds.get(padded_idx)
            ds_train, ds_val = train_test_split(ds_train, split_percentages=(2.0/3.0, 1.0/3.0))

            if len(ds_train) <= lag or len(ds_val) <= lag:
                print(padded_idx, "is to short, skipping")
                continue

            x_train, y_train = _apply_window(ds_train, lag)
            x_val, y_val = _apply_window(ds_val, lag)

            for m_name, model_obj in single_models.items():
                # if "adaptive" in m_name or "lstm" in m_name:
                #     continue
                model = model_obj["obj"]
                nr_filters = model_obj["nr_filters"]
                hidden_states = model_obj["hidden_states"]
                batch_size = 500
                epochs = 3000
                lr = 1e-3
                train_model(model, x_train, y_train, x_val, y_val, lag, m_name, "m4_{}_{}".format(name, padded_idx), nr_filters, batch_size, epochs, hidden_states, lr, save_plot=False, verbose=False)


def main(lag, ds_name=None, model_name=None):

    if model_name is not None:
        use_models = [model_name]
    else:
        use_models = single_models.keys()

    if ds_name is not None:
        if ds_name == "M4":
            train_m4_subset(lag=lag)
            return 
        else:
            use_ds = [ds_name]
    else:
        use_ds = implemented_datasets.keys()

    for n_ds in use_ds:
        ds = implemented_datasets[n_ds]
        X = ds["ds"]().torch()

        epochs = ds["epochs"]
        batch_size = ds["batch_size"]
        lr = ds["lr"]

        [x_train, x_val_small], [y_train, y_val], _, _, _ = windowing(X, train_input_width=lag, val_input_width=lag_mapping[str(lag)], use_torch=True)

        for m_name in use_models:
            model_obj = single_models[m_name]
            model = model_obj["obj"]
            nr_filters = model_obj["nr_filters"]
            hidden_states = model_obj["hidden_states"]
            train_model(model, x_train, y_train, x_val_small, y_val, lag, m_name, n_ds, nr_filters, batch_size, epochs, hidden_states, lr, save_plot=False, verbose=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", action="store", help="choose dataset to use for training", type=str)
    parser.add_argument("--model", action="store", help="choose model to use for training", type=str)
    parser.add_argument("--lag", action="store", help="choose lag to use for training", default=5, type=int)
    args = parser.parse_args()
    main(args.lag, args.dataset, model_name=args.model)

