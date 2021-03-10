import torch 
import argparse
from datasets import Jena_Climate, Bike_Total_Rents, Bike_Registered, Bike_Temperature
from datasets.utils import windowing
from viz import plot_train_log
from tsx.models.forecaster import Shallow_CNN_RNN, Shallow_FCN, AS_LSTM_01, AS_LSTM_02

implemented_datasets = {
    "jena": {
        "ds": Jena_Climate,
        "lag": 5,
        "epochs": 200,
        "batch_size": 300,
        "lr": 1e-3,
        "hidden_states": 10,
    },
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

def main(ds_name=None, model_name=None):

    if model_name is not None:
        use_models = [model_name]
    else:
        use_models = models.keys()

    if ds_name is not None:
        use_ds = [ds_name]
    else:
        use_ds = implemented_datasets.keys()

    for n_ds in use_ds:
        ds = implemented_datasets[n_ds]
        X = ds["ds"]().torch()

        lag = ds["lag"]
        [x_train, x_val], [y_train, y_val], x_val_big, x_test = windowing(X, train_input_width=lag, val_input_width=lag*lag, use_torch=True)

        epochs = ds["epochs"]
        batch_size = ds["batch_size"]
        lr = ds["lr"]
        hidden_states = ds["hidden_states"]

        for m_name in use_models:
            try:
                model = models[m_name](batch_size=batch_size, epochs=epochs, ts_length=x_train.shape[-1], hidden_states=hidden_states, learning_rate=lr)
            except TypeError:
                # TODO: FCN does not accept "hidden_states" variable. Maybe pass configs as dictionary? 
                model = models[m_name](batch_size=batch_size, epochs=epochs, ts_length=x_train.shape[-1], learning_rate=lr)
            logs, best = model.fit(x_train, y_train, X_val=x_val, y_val=y_val, model_save_path="models/{}_{}.pth".format(n_ds, m_name))
            plot_train_log(logs, "plots/train_{}_{}.pdf".format(n_ds, m_name), best_epoch=best)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", action="store", help="choose dataset to use for training", type=str)
    parser.add_argument("--model", action="store", help="choose model to use for training", type=str)
    args = parser.parse_args()
    main(args.dataset, model_name=args.model)

