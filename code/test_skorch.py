import torch
import time
import skorch
import numpy as np
import time
import pickle
import warnings
import matplotlib.pyplot as plt
from os.path import exists
from datasets.utils import windowing, simple_windowing, _apply_window
from functools import lru_cache
from itertools import product
from experiments import single_models, implemented_datasets
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from utils import calculate_single_seed
from datasets.monash_forecasting import load_dataset, _get_ds_names
from single_models import BaselineLastValue
from torch.optim.lr_scheduler import ReduceLROnPlateau
from skorch.callbacks import LRScheduler, EarlyStopping
from skorch.dataset import Dataset
from skorch.helper import predefined_split

warnings.filterwarnings("ignore")

#ds_names =  ["electricity_hourly"]
ds_names =  _get_ds_names()
ds_indices = list(range(5))
single_model_list = single_models.items()

all_configs = product(ds_names, ds_indices, single_model_list)

#@lru_cache(maxsize=1)
def load_data(ds_name, ds_index):
    X = torch.from_numpy(load_dataset(ds_name, ds_index)).float()

    [x_train, x_val], [y_train, y_val], _, _, X_test = windowing(X, train_input_width=5, val_input_width=25, use_torch=True)
    x_test, y_test = _apply_window(X_test, 5, offset=0, label_width=1, use_torch=True)

    return x_train, y_train, x_val, y_val, x_test, y_test

def test_performance():
    #for ds_name, ds_index, (model_name, model_obj) in all_configs:
    for ds_name in ds_names:
        fig, ax = plt.subplots(1,5, figsize=(14, 2))
        for col, ds_index in enumerate(ds_indices):
            ax[col].axes.get_xaxis().set_visible(False)
            ax[col].axes.get_yaxis().set_visible(False)
            for i, (model_name, model_obj) in enumerate(single_model_list):
                save_path = f"results/{ds_name}/{ds_index}_{model_name}.pth"
                if not exists(save_path):
                    continue
                X_train, y_train, X_val, y_val, X_test, y_test = load_data(ds_name, ds_index)

                seed = calculate_single_seed(model_name, ds_name, 5)

                torch.manual_seed(seed)
                np.random.seed(seed)

                nr_filters = model_obj["nr_filters"]
                hidden_states = model_obj["hidden_states"]
                model = skorch.NeuralNetRegressor(
                        model_obj["obj"], 
                        max_epochs=2000, 
                        lr=0.0001,
                        #train_split=predefined_split(val_ds),
                        module__nr_filters=nr_filters, 
                        module__hidden_states=hidden_states, 
                        module__ts_length=X_test.shape[-1], 
                        callbacks=[('early_stoping', EarlyStopping(patience=100, threshold=1e-4)), ('lr_scheduler', LRScheduler(policy=ReduceLROnPlateau, patience=20, factor=0.3))])

                model.initialize()
                model.load_params(f_params=save_path)
                
                preds = model.predict(X_test)
                if np.any(np.isnan(preds)):
                    print(f"NaN in {ds_name} {ds_index} {model_name}")
                    exit()
                loss = mean_squared_error(preds.squeeze(), y_test.squeeze())

                #print(i, row, col, ds_name, ds_index, model_name, loss)

                ax[col].bar(x=i, height=loss)

            # Reset color map
            #plt.gca().set_prop_cycle(None)
            baseline = BaselineLastValue()
            baseline_preds = baseline.predict(X_test)
            baseline_loss = mean_squared_error(baseline_preds.squeeze(), y_test.squeeze())
            ax[col].axhline(y=baseline_loss, xmin=-0.8, xmax=33.8, color="black")

        plt.suptitle(ds_name)
        plt.tight_layout()
        plt.savefig(f"plots/skorch_{ds_name}.png")
        plt.close()

def train():

    for ds_name, ds_index, (model_name, model_obj) in all_configs:
        if "lstm" in model_name or "adaptive" in model_name:
            continue

        save_path = f"results/{ds_name}/{ds_index}_{model_name}.pth"
        if exists(save_path):
            continue
        X_train, y_train, X_val, y_val, X_test, y_test = load_data(ds_name, ds_index)
        X = torch.cat([X_train, X_val])
        y = torch.cat([y_train, y_val])

        seed = calculate_single_seed(model_name, ds_name, 5)

        torch.manual_seed(seed)
        np.random.seed(seed)

        params = {
            "lr": [0.01, 0.001, 0.0001]
        }

        nr_filters = model_obj["nr_filters"]
        hidden_states = model_obj["hidden_states"]
        model = skorch.NeuralNetRegressor(
                model_obj["obj"], 
                max_epochs=2000, 
                #train_split=predefined_split(val_ds),
                module__nr_filters=nr_filters, 
                module__hidden_states=hidden_states, 
                module__ts_length=X_train.shape[-1], 
                callbacks=[('early_stoping', EarlyStopping(patience=100, threshold=1e-4)), ('lr_scheduler', LRScheduler(policy=ReduceLROnPlateau, patience=20, factor=0.3))]
        )

        gs = GridSearchCV(model, params, refit=True, cv=10, scoring="neg_mean_squared_error")

        model.set_params(verbose=False)

        before = time.time()
        gs.fit(X, y)
        after = time.time()
        print(after-before, ds_name, ds_index, model_name)
        best_estimator = gs.best_estimator_
        best_estimator.save_params(f_params=save_path)
                
if __name__ == "__main__":
    #train()
    test_performance()
