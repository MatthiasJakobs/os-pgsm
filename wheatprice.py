import numpy as np 
import torch
import pandas as pd

class WheatPrice:

    def __init__(self, path="wheatprice.csv", normalize=True):
        self.df = pd.read_csv(path, delimiter=",", header=0)

        if normalize:
            self.df['price'] = (self.df['price'] - self.df['price'].mean()) / self.df['price'].std()

    def torch(self):
        pth = torch.from_numpy(self.df['price'].to_numpy()).float()
        return pth
