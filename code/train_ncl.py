import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import copy

from typing import List
from seedpy import fixedseed
from experiments import single_models
from datasets.dataloading import implemented_datasets
from utils import ncl_seed
from os.path import exists
from train_single_models import load_data
from ncl import NegCorLearning

def main():
    repeats = 5
    batch_size = 100

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Use device {device}')

    for (ds_name, ds_index) in implemented_datasets:
        save_path = f"models/{ds_name}/{ds_index}_ncl.pth"
        if exists(save_path):
            continue

        X_train, y_train, X_val, y_val, X_test, y_test = load_data(ds_name, ds_index)

        with fixedseed(torch, seed=ncl_seed(ds_name, ds_index)):
            losses = np.zeros((repeats))
            models = []
            for i in range(repeats):
                used_single_models = []
                for m_name, model_obj in single_models.items():
                    m = model_obj["obj"]
                    nr_filters = model_obj["nr_filters"]
                    hidden_states = model_obj["hidden_states"]
                    try:
                        used_single_models.append(m(nr_filters=nr_filters, ts_length=X_train.shape[-1], hidden_states=hidden_states, batch_size=batch_size, device=device))
                    except TypeError:
                        used_single_models.append(m(nr_filters=nr_filters, ts_length=X_train.shape[-1], batch_size=batch_size, device=device))

                model = NegCorLearning(used_single_models, {}, device=device)
                model.batch_size = batch_size
                val_loss, best_model = model.fit(X_train, y_train, X_val, y_val, verbose=False)

                print(ds_name, ds_index, i, val_loss)

                losses[i] = val_loss
                models.append(best_model)

            # Choose the most average model
            average_performance = np.mean(losses)
            rel_scores = np.abs(losses-average_performance)
            nearest_model_idx = np.argmin(rel_scores)
            print(ds_name, ds_index, "chose", nearest_model_idx)
            print("-"*30)

            torch.save(models[nearest_model_idx], save_path)

if __name__ == "__main__":
    main()
