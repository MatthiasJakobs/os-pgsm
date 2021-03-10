import pandas as pd
import numpy as np
import torch

class Bike_Total_Rents:
    def __init__(self, path="code/datasets/bike-total-rents.csv", normalize=True):
        self.df = pd.read_csv(path)

        self.means = self.df.mean()
        self.stds = self.df.std()

        if normalize:
            self.df = (self.df - self.means) / self.stds

    def torch(self):
        pth = torch.from_numpy(self.df.to_numpy()).squeeze().float()
        return pth

class Bike_Registered:
    def __init__(self, path="code/datasets/bike_sharing_registered_counts_ts.csv", normalize=True):
        self.df = pd.read_csv(path)

        self.means = self.df.mean()
        self.stds = self.df.std()

        if normalize:
            self.df = (self.df - self.means) / self.stds

    def torch(self):
        pth = torch.from_numpy(self.df.to_numpy()).squeeze().float()
        return pth

class Bike_Temperature:
    def __init__(self, path="code/datasets/normalized-temperature-bikesharing.csv", normalize=False):
        self.df = pd.read_csv(path)

        self.means = self.df.mean()
        self.stds = self.df.std()

        if normalize:
            self.df = (self.df - self.means) / self.stds

    def torch(self):
        pth = torch.from_numpy(self.df.to_numpy()).squeeze().float()
        return pth


