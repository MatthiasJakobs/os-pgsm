import pandas as pd
import numpy as np

from .base import BaseDataset

class CloudCoverage(BaseDataset):

    def load_ds(self):
        df = pd.read_csv("code/datasets/cloud.csv")
        x = df["CLOUDCOVER"][5:]
        x[x == -1] = np.nan
        x = x.fillna(method="ffill")
        return x.to_numpy()