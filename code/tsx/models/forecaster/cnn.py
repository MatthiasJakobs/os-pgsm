import numpy as np
import torch
import torch.nn as nn
import pandas as pd

from .base import BasePyTorchForecaster

class Shallow_FCN(BasePyTorchForecaster):

    def __init__(self, ts_length, nr_filters=32, **kwargs):
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