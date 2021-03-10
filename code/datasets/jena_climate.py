import numpy as np
import pandas as pd
import torch

# based on https://www.tensorflow.org/tutorials/structured_data/time_series
class Jena_Climate:

    # TODO: If we use other columns: needs more cleanup! See link above
    def __init__(self, path="code/datasets/jena_climate_2009_2016.csv", cols=['T (degC)'], normalize=True, subsample=None):
        self.df = pd.read_csv(path)[5::6]
        self.df = self.df[cols]

        self.means = self.df.mean()
        self.stds = self.df.std()

        if normalize:
            self.df = (self.df - self.means) / self.stds

        if subsample is not None:
            self.df = self.df.head(subsample)

    def torch(self):
        pth = torch.from_numpy(self.df.to_numpy()).squeeze().float()
        return pth
