import torch 
import numpy as np
import argparse

from datasets import Jena_Climate, Bike_Total_Rents, Bike_Registered, Bike_Temperature, M4_Hourly, M4_Daily, M4_Monthly, M4_Weekly, M4_Quaterly, M4_Yearly
from datasets.utils import windowing, _apply_window, train_test_split
from viz import plot_train_log
from tsx.models.forecaster import Shallow_CNN_RNN, Shallow_FCN, AS_LSTM_01, AS_LSTM_02

implemented_datasets = {
    # "jena": {
    #     "ds": Jena_Climate,
    #     "lag": 5,
    #     "epochs": 200,
    #     "batch_size": 300,
    #     "lr": 1e-3,
    #     "hidden_states": 10,
    # },
    "bike_total_rents": {
        "ds": Bike_Total_Rents,
        "lag": 5,
        "epochs": 1500,
        "batch_size": 50,
        "lr": 1e-4,
        "hidden_states": 10,
    },
    "bike_registered": {
        "ds": Bike_Registered,
        "lag": 5,
        "epochs": 1500,
        "batch_size": 50,
        "lr": 1e-4,
        "hidden_states": 10,
    },
    "bike_temperature": {
        "ds": Bike_Temperature,
        "lag": 5,
        "epochs": 2500,
        "batch_size": 50,
        "lr": 1e-4,
        "hidden_states": 10,
    },
}

models = {
    "rnn": Shallow_CNN_RNN,
    "cnn": Shallow_FCN,
    "as01": AS_LSTM_01,
    "as02": AS_LSTM_02
}

def train_model(model, x_train, y_train, x_val, y_val, model_name, ds_name, batch_size, epochs, hidden_states, learning_rate, save_plot=False, verbose=True):
        save_path = "models/{}_{}.pth".format(ds_name, model_name)
        model_instances = []
        model_logs = []
        model_scores = []
        model_bests = []
        for i in range(5):
            try:
                m = model(batch_size=batch_size, epochs=epochs, ts_length=x_train.shape[-1], hidden_states=hidden_states, learning_rate=learning_rate)
            except TypeError:
                # TODO: FCN does not accept "hidden_states" variable. Maybe pass configs as dictionary? 
                m = model(batch_size=batch_size, epochs=epochs, ts_length=x_train.shape[-1], learning_rate=learning_rate)

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

def train_m4_subset(lag=5):
    name_list = ['hourly', 'weekly', 'quaterly', 'daily', 'monthly']
    ds_list = [ M4_Hourly(), M4_Weekly(), M4_Quaterly(), M4_Daily(), M4_Monthly()]

    for name, ds in zip(name_list, ds_list):
        indices = list(range(20))
        for idx in indices:
            padded_idx = ds.train_data.columns[idx]
            print(padded_idx)
            ds_train, _ = ds.get(padded_idx)
            ds_train, ds_val = train_test_split(ds_train, split_percentages=(2.0/3.0, 1.0/3.0))

            x_train, y_train = _apply_window(ds_train, lag)
            x_val, y_val = _apply_window(ds_val, lag)

            for m_name, model in models.items():
                train_model(model, x_train, y_train, x_val, y_val, m_name, "m4_{}_{}".format(name, padded_idx), 500, 3000, 10, 1e-3, save_plot=False, verbose=False)


def main(ds_name=None, model_name=None):

    if model_name is not None:
        use_models = [model_name]
    else:
        use_models = models.keys()

    if ds_name is not None:
        if ds_name == "M4":
            train_m4_subset()
            return 
        else:
            use_ds = [ds_name]
    else:
        use_ds = implemented_datasets.keys()

    for n_ds in use_ds:
        ds = implemented_datasets[n_ds]
        X = ds["ds"]().torch()

        lag = ds["lag"]
        epochs = ds["epochs"]
        batch_size = ds["batch_size"]
        lr = ds["lr"]
        hidden_states = ds["hidden_states"]

        [x_train, x_val_small], [y_train, y_val], _, _, _ = windowing(X, train_input_width=lag, val_input_width=lag*lag, use_torch=True)

        for m_name in use_models:
            model = models[m_name]
            train_model(model, x_train, y_train, x_val_small, y_val, m_name, n_ds, batch_size, epochs, hidden_states, lr, save_plot=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", action="store", help="choose dataset to use for training", type=str)
    parser.add_argument("--model", action="store", help="choose model to use for training", type=str)
    args = parser.parse_args()
    main(args.dataset, model_name=args.model)

