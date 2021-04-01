import torch
import torch.nn as nn
import pandas as pd

from tsx.metrics import smape, mae
from collections import OrderedDict

class BaseForecaster:

    def fit(self, X, y):
        raise NotImplementedError()

    def transform(self, X):
        raise NotImplementedError()

    def forecast_single(self, X):
        raise NotImplementedError()

    def forecast_multi(self, X):
        raise NotImplementedError()

    def save(self):
        raise NotImplementedError()

    def load(self):
        raise NotImplementedError()

    def inform(self, string):
        if self.verbose:
            print(string)

    def preprocessing(self, X_train, y_train, X_test=None, y_test=None):
        raise NotImplementedError()

class BasePyTorchForecaster(nn.Module, BaseForecaster):

    def __init__(self, optimizer=torch.optim.Adam, batch_size=5, learning_rate=1e-3, verbose=False, epochs=5):
        super(BasePyTorchForecaster, self).__init__()
        self.classifier = False
        self.forecaster = True
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.verbose = verbose
        self.epochs = epochs
        self.fitted = False

    def save(self, path):
        torch.save(self.state_dict(), path)

    def fit(self, X_train, y_train, X_val=None, y_val=None, losses=None, model_save_path=None, verbose=True):
        if losses is None:
            losses = OrderedDict(mse=torch.nn.MSELoss(), smape=smape, mae=mae)

        # Expects X, y to be Pytorch tensors 
        X_train, y_train, X_val, y_val = self.preprocessing(X_train, y_train, X_test=X_val, y_test=y_val)

        ds = torch.utils.data.TensorDataset(X_train, y_train)
        dl = torch.utils.data.DataLoader(ds, batch_size=self.batch_size, shuffle=True)

        # Convention: Use first metric in `losses` for training
        loss_fn = losses[list(losses.keys())[0]]
        optim = self.optimizer(self.parameters(), lr=self.learning_rate, weight_decay=1e-1)

        log_cols = ["train_" + k for k in losses.keys()]
        if X_val is not None and y_val is not None:
            log_cols += ["val_" + k for k in losses.keys()]

        if model_save_path is not None:
            best_validation = 1e12
            best_epoch = 0

        logs = pd.DataFrame(columns=log_cols)
        for epoch in range(self.epochs):
            train_predictions = []
            train_labels = []
            print_epoch = epoch + 1
            epoch_loss = 0.0
            for i, (X, y) in enumerate(dl):
                self.train()
                optim.zero_grad()
                prediction = self.forward(X)
                loss = loss_fn(prediction, y)
                loss.backward()
                epoch_loss += loss.item()
                optim.step()
                train_predictions.append(prediction.detach())
                train_labels.append(y.clone().detach())

            with torch.no_grad():
                self.eval()
                log_values = []
                # Evaluate train
                train_predictions = torch.cat(train_predictions, dim=0)
                train_labels = torch.cat(train_labels, dim=0)
                for k, L in losses.items():
                    log_values.append(L(train_predictions, train_labels).item())

                # Evaluate val (if present)
                if X_val is not None and y_val is not None:
                    val_prediction = self.forward(X_val)
                    for k, L in losses.items():
                        loss_value = L(val_prediction, y_val).item()

                        if k == "mse" and model_save_path is not None and epoch > 0:
                            if loss_value <= best_validation:
                                best_epoch = epoch
                                best_validation = loss_value
                                torch.save(self.state_dict(), model_save_path)

                        log_values.append(loss_value)
                logs.loc[epoch] = log_values
                if verbose:
                    print(epoch, *zip(log_cols, [format(v, '.5f') for v in log_values]))

        self.fitted = True
        if model_save_path is not None:
            print("{} was best epoch with val_mse {}".format(best_epoch, best_validation))
            return logs, best_epoch

        return logs
