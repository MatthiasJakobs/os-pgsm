import torch
import torch.nn as nn

from .base import BasePyTorchForecaster

class Simple_LSTM(BasePyTorchForecaster):

    def __init__(self, lag, nr_filters=64, ts_length=23, output_size=1, hidden_states=100, **kwargs):
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

