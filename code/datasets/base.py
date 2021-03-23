import torch
import pandas as pd
import numpy as np

class BaseDataset:

    def __init__(self, normalize=True):
        self.X = self.load_ds()
        if normalize:
            self.normalize()

    def load_ds(self):
        raise NotImplementedError()

    def torch(self):
        return torch.from_numpy(self.X).float()

    def normalize(self):
        mean = np.mean(self.X)
        std = np.std(self.X)
        self.X = (self.X - mean) / std