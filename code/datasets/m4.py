import numpy as np
import pandas as pd
import torch

from os.path import exists, join

class M4_Base:

    def __init__(self, subset_name, path, normalize=True):
        self.root_dir = path
        self.train_path = join(self.root_dir, "Train")
        self.test_path = join(self.root_dir, "Test")

        self.train_path = join(self.train_path, "{}-train.csv".format(subset_name))
        self.test_path = join(self.test_path, "{}-test.csv".format(subset_name))

        self.train_data = pd.read_csv(self.train_path).set_index("V1").T
        self.test_data = pd.read_csv(self.test_path).set_index("V1").T

        self.normalize = normalize

    # taken from https://davistownsend.github.io/blog/Parallel_ts_fc_Dask/
    def get(self, key, remove_outlier=False, interpolate_missing=False, data_type="torch"):
        subsets = [self.train_data[key], self.test_data[key]]

        for t_i in range(2):

            ts = subsets[t_i]

            # Find start and endpoint of ts
            start = ts[ts.notna()].index[0]
            stop = ts[ts.notna()].index[-1]
            ts = ts.loc[start:stop]

            if remove_outlier:
                #sets outliers to NaN defined as a point that is 3 deviations away from rolling mean + 3*rolling standard deviation
                ts = ts.where(~(abs(ts) > (ts.rolling(center=False,window=3).mean() + (ts.rolling(center=False,window=3,min_periods = 3).std() * 3))),np.nan)

            if interpolate_missing:
                #linearly interpolate any NaN values
                ts = ts.interpolate()

            ts = ts.fillna(0)

            if self.normalize:
                mean = ts.mean()
                std = ts.std()
                ts = (ts - mean) / std

            if data_type == "torch":
                ts = torch.from_numpy(ts.to_numpy())
            elif data_type == "numpy":
                ts = ts.to_numpy()
            elif data_type == "pandas":
                pass
            else:
                raise NotImplementedError("Datatype {} is unknown".format(data_type))

            subsets[t_i] = ts

        
        return subsets[0], subsets[1]



class M4_Hourly(M4_Base):

    def __init__(self, path='/data/M4'):
        super().__init__(path=path, subset_name="Hourly")

class M4_Monthly(M4_Base):

    def __init__(self, path='/data/M4'):
        super().__init__(path=path, subset_name="Monthly")

class M4_Yearly(M4_Base):

    def __init__(self, path='/data/M4'):
        super().__init__(path=path, subset_name="Yearly")

class M4_Daily(M4_Base):

    def __init__(self, path='/data/M4'):
        super().__init__(path=path, subset_name="Daily")

class M4_Weekly(M4_Base):

    def __init__(self, path='/data/M4'):
        super().__init__(path=path, subset_name="Weekly")

class M4_Quaterly(M4_Base):

    def __init__(self, path='/data/M4'):
        super().__init__(path=path, subset_name="Quarterly")