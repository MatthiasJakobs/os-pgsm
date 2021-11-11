import numpy as np
import torch
import torch.nn as nn
import pandas as pd

from utils import smape, mae
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

class BaselineLastValue(nn.Module):
    def __init__(self, **kwargs):
        super(BaselineLastValue, self).__init__()

    def forecast_point(self, x):
        x = x.squeeze()
        if len(x.shape) == 1:
            return x[-1]
        if len(x.shape) == 2:
            return x[:, -1]
        raise Exception(f"Input shape {x.shape} not supported in baseline model")

    def forward(self, x):
        return self.forecast_point(x)

    def predict(self, x):
        with torch.no_grad():
            if isinstance(x, np.ndarray):
                x = torch.from_numpy(x)
            return self.forecast_point(x).numpy()

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


# Inspired by residual block found in ResNets
class ResidualBlock(nn.Module):

    def __init__(self, input_filters, output_filters):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv1d(input_filters, output_filters, 3, padding=1)
        self.conv2 = nn.Conv1d(output_filters, output_filters, 3, padding=1)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.batchnorm_1 = nn.BatchNorm1d(output_filters)
        self.batchnorm_2 = nn.BatchNorm1d(output_filters)

    def forward(self, x):
        z = self.conv1(x)
        z = self.batchnorm_1(z)
        z = self.relu1(z)
        z = self.conv2(z)
        z = self.batchnorm_2(z)
        x = x + z
        x = self.relu2(x)
        return x

class Shallow_FCN(BasePyTorchForecaster):

    def __init__(self, ts_length, hidden_states=0, nr_filters=32, **kwargs):
        super().__init__(**kwargs)

        self.feature_extractor = nn.Sequential(
            nn.Conv1d(1, nr_filters, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(nr_filters)
            # nn.Conv1d(int(nr_filters/2), nr_filters, 3, padding=1),
            # nn.ReLU()
        )

        self.flatten = nn.Flatten()

        self.forecaster = nn.Sequential(
            nn.Dropout(0.9),
            nn.Linear(ts_length * nr_filters, 1)
        )

    def predict(self, x):
        with torch.no_grad():
            prediction = self.forward(x)
            return prediction.squeeze().numpy()

    def forward(self, x, return_intermediate=False):

        feats = self.feature_extractor(x)
        flatted = self.flatten(feats)
        prediction = self.forecaster(flatted)

        if return_intermediate:
            to_return = {}
            to_return['feats'] = feats
            to_return['logits'] = prediction
            to_return['output'] = prediction
            return to_return
        else:
            return prediction

    def reset_gradients(self):
        self.feature_extractor.zero_grad()
        self.forecaster.zero_grad()

    def preprocessing(self, X_train, y_train, X_test=None, y_test=None):
        return X_train, y_train, X_test, y_test

class OneResidualFCN(Shallow_FCN):

    def __init__(self, ts_length, nr_filters=32, **kwargs):
        super().__init__(ts_length, nr_filters=nr_filters, **kwargs)

        self.feature_extractor = nn.Sequential(
            ResidualBlock(1, nr_filters)
        )

        self.flatten = nn.Flatten()

        self.forecaster = nn.Sequential(
            nn.Dropout(0.9),
            nn.Linear(ts_length * nr_filters, 100),
            nn.Dropout(0.9),
            nn.Linear(100, 1)
        )

class TwoResidualFCN(Shallow_FCN):

    def __init__(self, ts_length, nr_filters=32, **kwargs):
        super().__init__(ts_length, nr_filters=nr_filters, **kwargs)

        self.feature_extractor = nn.Sequential(
            ResidualBlock(1, nr_filters),
            ResidualBlock(nr_filters, nr_filters)
        )

        self.flatten = nn.Flatten()

        self.forecaster = nn.Sequential(
            nn.Dropout(0.9),
            nn.Linear(ts_length * nr_filters, 100),
            nn.Dropout(0.9),
            nn.Linear(100, 1)
        )

# class OneResidualShallow(OneResidualFCN):

#     def __init__(self, ts_length, nr_filters=32, **kwargs):
#         super().__init__(ts_length, nr_filters=nr_filters, **kwargs)

#         self.forecaster = nn.Sequential(
#             nn.Dropout(0.9),
#             nn.Linear(ts_length * nr_filters, 1)
#         )

# class TwoResidualShallow(TwoResidualFCN):

#     def __init__(self, ts_length, nr_filters=32, **kwargs):
#         super().__init__(ts_length, nr_filters=nr_filters,**kwargs)

#         self.forecaster = nn.Sequential(
#             nn.Dropout(0.9),
#             nn.Linear(ts_length * nr_filters, 1),
#         )

class TwoResidualFCN(Shallow_FCN):

    def __init__(self, ts_length, nr_filters=32, **kwargs):
        super().__init__(ts_length, nr_filters=nr_filters, **kwargs)

        self.feature_extractor = nn.Sequential(
            ResidualBlock(1, nr_filters),
            ResidualBlock(nr_filters, nr_filters)
        )

        self.flatten = nn.Flatten()

        self.forecaster = nn.Sequential(
            nn.Dropout(0.9),
            nn.Linear(ts_length * nr_filters, 100),
            nn.Dropout(0.9),
            nn.Linear(100, 1)
        )


class Shallow_CNN_RNN(BasePyTorchForecaster):

    # This version is STATELESS, i.e., it will reset the hidden state before every batch
    # See https://discuss.pytorch.org/t/lstm-hidden-state-logic/48101/4
    # I could have used a simpler version of not giving the LSTM any input state to begin with, but now, this is easily adaptable to statefull
    def __init__(self, nr_filters=64, ts_length=23, output_size=1, hidden_states=100, **kwargs):
        super().__init__(**kwargs)

        self.cnn_filters = nr_filters
        self.output_size = output_size
        self.hidden_states = hidden_states

        self.feature_extractor = nn.Sequential(
            nn.Conv1d(1, int(self.cnn_filters/2), 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(int(self.cnn_filters/2)),
            nn.Conv1d(int(self.cnn_filters/2), self.cnn_filters, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(self.cnn_filters)
        )

        self.lstm = nn.LSTM(self.cnn_filters, self.hidden_states, dropout=0.9)
        self.reset_hidden_states()

        self.dense = nn.Linear(ts_length, output_size)

    def reset_gradients(self):
        self.feature_extractor.zero_grad()
        self.reset_hidden_states()
        self.dense.zero_grad()
        
    def reset_hidden_states(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size

        self.lstm_hidden_cell = (
            torch.zeros(1,batch_size, self.hidden_states), 
            torch.zeros(1,batch_size, self.hidden_states)
        )

    def forward(self, x, return_intermediate=False):

        feats = self.feature_extractor(x)
        batch_size, nr_filters, seq_length = feats.shape
        self.reset_hidden_states(batch_size=batch_size) # STATELESS
        lstm_out, self.lstm_hidden_cell = self.lstm(feats.view(seq_length, batch_size, nr_filters), self.lstm_hidden_cell)
        prediction = self.dense(lstm_out[..., -1].view(batch_size, -1))

        if return_intermediate:
            output = {}
            output['feats'] = feats
            output['prediction'] = prediction
            output['logits'] = prediction
            return output

        return prediction

    def preprocessing(self, X_train, y_train, X_test=None, y_test=None):
        return X_train, y_train, X_test, y_test

    def predict(self, x):
        with torch.no_grad():
            prediction = self.forward(x)
            return prediction.squeeze().numpy()

class AS_LSTM_02(Shallow_CNN_RNN):

    def __init__(self, nr_filters=32, ts_length=23, output_size=1, hidden_states=10, **kwargs):
        super().__init__(**kwargs)

        self.cnn_filters = nr_filters
        self.output_size = output_size
        self.hidden_states = hidden_states

        self.feature_extractor = nn.Sequential(
            nn.Conv1d(1, self.cnn_filters, 1, padding=0),
            nn.ReLU(),
            nn.MaxPool1d(1)
        )

        self.lstm = nn.LSTM(self.cnn_filters, self.hidden_states, dropout=0.9)
        self.reset_hidden_states()

        self.dense = nn.Linear(ts_length, output_size)

class AS_LSTM_03(Shallow_CNN_RNN):

    def __init__(self, nr_filters=32, ts_length=23, output_size=1, hidden_states=10, **kwargs):
        super().__init__(**kwargs)

        self.cnn_filters = nr_filters
        self.output_size = output_size
        self.hidden_states = hidden_states

        self.feature_extractor = nn.Sequential(
            nn.Conv1d(1, self.cnn_filters, 1, padding=0),
            nn.ReLU(),
            nn.BatchNorm1d(self.cnn_filters),
            nn.Conv1d(self.cnn_filters, self.cnn_filters, 1, padding=0),
            nn.ReLU(),
            nn.BatchNorm1d(self.cnn_filters),
            nn.Conv1d(self.cnn_filters, self.cnn_filters, 1, padding=0),
            nn.ReLU(),
            nn.BatchNorm1d(self.cnn_filters),
        )

        self.lstm = nn.LSTM(self.cnn_filters, self.hidden_states, dropout=0.9)
        self.reset_hidden_states()

        self.dense = nn.Linear(ts_length, output_size)

class AS_LSTM_01(Shallow_CNN_RNN):

    def __init__(self, nr_filters=32, ts_length=23, output_size=1, hidden_states=10, **kwargs):
        super().__init__(**kwargs)

        self.cnn_filters = nr_filters
        self.output_size = output_size
        self.hidden_states = hidden_states

        self.feature_extractor = nn.Sequential(
            nn.Conv1d(1, self.cnn_filters, 1, padding=0),
            nn.ReLU(),
            nn.BatchNorm1d(self.cnn_filters),
            nn.Conv1d(self.cnn_filters, self.cnn_filters, 1, padding=0),
            nn.ReLU(),
            nn.BatchNorm1d(self.cnn_filters),
        )

        self.lstm = nn.LSTM(self.cnn_filters, self.hidden_states, dropout=0.9)
        self.reset_hidden_states()

        self.dense = nn.Linear(ts_length, output_size)

class Simple_LSTM(BasePyTorchForecaster):

    def __init__(self, lag=5, nr_filters=64, ts_length=23, output_size=1, hidden_states=100, **kwargs):
        super().__init__(**kwargs)

        self.output_size = output_size
        self.hidden_states = hidden_states

        self.lstm = nn.LSTM(1, self.hidden_states, dropout=0.9)
        self.reset_hidden_states()

        self.dense = nn.Linear(lag, output_size)

    def reset_gradients(self):
        self.reset_hidden_states()
        self.dense.zero_grad()
        
    def reset_hidden_states(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size

        self.lstm_hidden_cell = (
            torch.zeros(1,batch_size, self.hidden_states), 
            torch.zeros(1,batch_size, self.hidden_states)
        )

    def forward(self, x):
        batch_size, nr_filters, seq_length = x.shape
        self.reset_hidden_states(batch_size=batch_size) # STATELESS
        lstm_out, self.lstm_hidden_cell = self.lstm(x.view(seq_length, batch_size, nr_filters), self.lstm_hidden_cell)
        prediction = self.dense(lstm_out[..., -1].view(batch_size, -1))

        return prediction

    def preprocessing(self, X_train, y_train, X_test=None, y_test=None):
        return X_train, y_train, X_test, y_test

    def predict(self, x):
        with torch.no_grad():
            prediction = self.forward(x)
            return prediction.squeeze().numpy()

