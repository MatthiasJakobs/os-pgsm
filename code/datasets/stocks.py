import pandas as pd
from .base import BaseDataset

class BaseStock(BaseDataset):

    def load_ds(self, path):
        df = pd.read_csv(path, delimiter=",")
        x = df['Close'].to_numpy()
        return x

class NASDAQ(BaseStock):

    def load_ds(self):
        return super().load_ds("code/datasets/Processed_NASDAQ.csv")

class DJI(BaseStock):

    def load_ds(self):
        return super().load_ds("code/datasets/Processed_DJI.csv")

class RUSSEL(BaseStock):

    def load_ds(self):
        return super().load_ds("code/datasets/Processed_RUSSEL.csv")

class NYSE(BaseStock):

    def load_ds(self):
        return super().load_ds("code/datasets/Processed_NYSE.csv")