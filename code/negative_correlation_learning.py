import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import copy

from typing import List
from seedpy import fixedseed
from experiments import single_models, implemented_datasets
from utils import ncl_seed
from os.path import exists
from train_single_models import load_data

class NegCorLearning(nn.Module):

    def __init__(self, models, lamb=0.9):
        super(NegCorLearning, self).__init__()
        self.models = models
        self.nr_models = len(self.models)
        self.learning_rate = 1e-3
        self.batch_size = 500
        self.optimizer = torch.optim.Adam
        self.lamb = lamb
        self.epochs = 500
        self.early_stopping = 25

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (1/self.nr_models) * np.sum(np.array([m.predict(X) for m in self.models]), axis=0)

    def forward(self, X: torch.tensor) -> List[torch.tensor]:
        return [m(X).unsqueeze(0) for m in self.models]

    def div_loss(self, y, predictions, ensemble_prediction):
        first_agg = 0
        second_agg = 0

        for p in predictions:
            p = p.squeeze(0)
            first_agg += 0.5 * (p-y)**2
            second_agg += 0.5 * (p-ensemble_prediction)**2

        return torch.sum((1/self.nr_models) * first_agg - self.lamb * (1/self.nr_models) * second_agg)

    def fit(self, X_train, y_train, X_val, y_val, verbose=False):

        train_ds = torch.utils.data.TensorDataset(X_train, y_train)
        train_dl = torch.utils.data.DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
        val_ds = torch.utils.data.TensorDataset(X_val, y_val)
        val_dl = torch.utils.data.DataLoader(val_ds, batch_size=self.batch_size, shuffle=False)

        optim = self.optimizer([{'params': m.parameters()} for m in self.models], lr=self.learning_rate, weight_decay=1e-1)
        val_losses = []
        train_losses = []

        best_val_epoch = 0
        best_val_loss = 1e9
        best_val_model = None

        for epoch in range(self.epochs):
            epoch_loss = 0.0
            epoch_val_loss = 0.0
            for X, y in train_dl:
                self.train()
                optim.zero_grad()

                predictions = self.forward(X)
                ensemble_prediction = torch.mean(torch.cat(predictions, axis=0), axis=0)
                loss = self.div_loss(y, predictions, ensemble_prediction)

                loss.backward()
                epoch_loss += loss.item()
                optim.step()

            for X, y in val_dl:
                self.eval()

                with torch.no_grad():
                    predictions = self.forward(X)
                    ensemble_prediction = torch.mean(torch.cat(predictions, axis=0), axis=0)
                    loss = self.div_loss(y, predictions, ensemble_prediction).item()


                epoch_val_loss += loss
                if loss < best_val_loss:
                    best_val_epoch = epoch
                    best_val_loss = loss
                    #torch.save(self.models, model_save_path)
                    best_val_model = copy.deepcopy(self.models)
                else:
                    if (epoch - best_val_epoch) >= self.early_stopping:
                        #print(f"Early stopping at epoch {epoch}, reverting to {best_val_epoch}")
                        return best_val_loss, best_val_model

            train_losses.append(epoch_loss)
            val_losses.append(epoch_val_loss)
            if verbose:
                print(epoch, epoch_loss, epoch_val_loss)

        return best_val_loss, best_val_model

def main():
    repeats = 5
    batch_size = 100

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
                        used_single_models.append(m(nr_filters=nr_filters, ts_length=X_train.shape[-1], hidden_states=hidden_states, batch_size=batch_size))
                    except TypeError:
                        used_single_models.append(m(nr_filters=nr_filters, ts_length=X_train.shape[-1], batch_size=batch_size))

                model = NegCorLearning(used_single_models)
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
