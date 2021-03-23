import pandas as pd
import numpy as np
import torch

from .base import BaseDataset

class BaseEnergy(BaseDataset):

    def load_ds(self, column_name, length=1500):
        df = pd.read_csv("code/datasets/energydata_complete.csv")
        x = df[column_name].to_numpy()
        return x[:length]

class Energy_RH1(BaseEnergy):

    def load_ds(self):
        return super().load_ds("RH_1")

class Energy_RH2(BaseEnergy):

    def load_ds(self):
        return super().load_ds("RH_2")

class Energy_TH4(BaseEnergy):

    def load_ds(self):
        return super().load_ds("TH4")

class Energy_TH5(BaseEnergy):

    def load_ds(self):
        return super().load_ds("TH5")