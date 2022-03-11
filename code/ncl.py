import numpy as np
import torch
import torch.nn as nn
import copy

from typing import List
from seedpy import fixedseed
from datasets.utils import sliding_split
from utils import mse

class NegCorLearning(nn.Module):

    def __init__(self, models, config, lamb=0.9, device='cpu'):
        super(NegCorLearning, self).__init__()
        self.models = models
        self.config = config
        self.nr_models = len(self.models)
        self.learning_rate = 1e-3
        self.batch_size = 500
        self.optimizer = torch.optim.Adam
        self.lamb = lamb
        self.epochs = 500
        self.early_stopping = 25
        self.device = device

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (1/self.nr_models) * np.sum(np.array([m.predict(X) for m in self.models]), axis=0)

    def forward(self, X: torch.tensor) -> List[torch.tensor]:
        return [m(X) for m in self.models]

    def run(self, X_val, X_test):
        X_test_small, y_test_small = sliding_split(X_test, 5, use_torch=True)
        with fixedseed(torch, seed=0):
            preds = self.predict(X_test_small.unsqueeze(1))
            preds = np.concatenate([np.squeeze(X_test_small[0]), preds])
            return preds

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
                X = X.to(self.device)
                y = y.to(self.device)
                self.train()
                optim.zero_grad()

                predictions = self.forward(X)
                ensemble_prediction = torch.mean(torch.cat(predictions, axis=0), axis=0)
                loss = self.div_loss(y, predictions, ensemble_prediction)

                loss.backward()
                epoch_loss += loss.item()
                optim.step()

            y_val_pred = self.predict(X_val.to(self.device))
            val_mse = mse(torch.from_numpy(y_val_pred).reshape(-1), y_val.reshape(-1))

            if val_mse < best_val_loss:
                best_val_epoch = epoch
                best_val_loss = val_mse
                best_val_model = copy.deepcopy(self.models)
            else:
                if (epoch - best_val_epoch) >= self.early_stopping:
                    print(f"Early stopping at epoch {epoch}, reverting to {best_val_epoch}")
                    return best_val_loss.item(), best_val_model

            train_losses.append(epoch_loss)
            val_losses.append(epoch_val_loss)
            if verbose:
                print(epoch, epoch_loss, val_mse)

        return best_val_loss, best_val_model