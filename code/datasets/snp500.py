import pandas as pd
import numpy as np
import torch

# data from https://datahub.io/core/s-and-p-500#data
class SNP500:
    def __init__(self, path="code/datasets/snp500.csv", normalize=True):
        column = "SP500"
        self.df = pd.read_csv(path)

        self.df = self.df[column]
        self.df = self.df.loc[1069:]

        self.means = self.df.mean()
        self.stds = self.df.std()

        if normalize:
            self.df = (self.df - self.means) / self.stds

    def torch(self):
        pth = torch.from_numpy(self.df.to_numpy()).squeeze().float()
        return pth