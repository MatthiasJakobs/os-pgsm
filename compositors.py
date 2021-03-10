import torch
import numpy as np

from tqdm import trange
from utils import simple_gradcam, stupid_gradcam
from datasets.utils import sliding_split, equal_split
from tsx.metrics import smape
from tsx.attribution import Grad_CAM
from tsx.distances import dtw
from sklearn.neighbors import KNeighborsClassifier

class BaseCompositor:

    def __init__(self, models, lag):
        self.models = models
        # Assume identical lag for all modules no. Easily changable
        self.lag = lag

    def calculate_rocs(self):
        raise NotImplementedError()

    def find_best_forecaster(self, x):
        raise NotImplementedError()

    def ranking(self):
        raise NotImplementedError()

    def run(self, x_val, x_test, reuse_prediction=False):
        self.val_losses = self.evaluate_on_validation(x_val)
        self.ranking = self.ranking()
        self.rocs = self.calculate_rocs(x_val)
        return self.forecast_on_test(x_test, reuse_prediction)

    # TODO: Make faster
    def forecast_on_test(self, x_test, reuse_prediction=False):
        self.test_forecasters = []
        predictions = np.zeros_like(x_test)

        x = x_test[:self.lag]
        predictions[:self.lag] = x

        for x_i in trange(self.lag, len(x_test)):
            if reuse_prediction:
                x = torch.from_numpy(predictions[x_i-self.lag:x_i]).unsqueeze(0)
            else:
                x = x_test[x_i-self.lag:x_i].unsqueeze(0)

            best_model = self.find_best_forecaster(x)
            self.test_forecasters.append(best_model)
            predictions[x_i] = self.models[best_model].predict(x.unsqueeze(0))

        error = smape(predictions, x_test.numpy())

        return np.array(predictions), error

    def ranking(self):
        if len(self.models) > 2:
            nr_models = len(self.models)
            nr_entries = len(self.val_losses[0])
            rankings = np.zeros(nr_entries)
            for i in range(nr_entries):
                best_model = None
                best_value = 1e9
                for m_idx in range(nr_models):
                    loss_m_i = self.val_losses[m_idx][i]
                    if loss_m_i < best_value:
                        best_value = loss_m_i
                        best_model = m_idx

                rankings[i] = best_model

            return rankings.astype(np.int)

        rankings = (self.val_losses[0] > self.val_losses[1]).astype(np.int)
        return rankings


class GCCompositor(BaseCompositor):

    def __init__(self, models, lag, threshold=0.5):
        super().__init__(models, lag)
        self.threshold = threshold

    def evaluate_on_validation(self, x_val):
        cams = np.squeeze(np.zeros((len(self.models), x_val.shape[0], x_val.shape[1]-self.lag, self.lag)))
        losses = np.squeeze(np.zeros((len(self.models), x_val.shape[0])))
        for o, x in enumerate(x_val):
            X, y = sliding_split(x, self.lag, use_torch=True)
            for n_m, m in enumerate(self.models):
                for idx in range(len(X)):
                    test = X[idx].unsqueeze(0)
                    res = m.forward(test.unsqueeze(1), return_intermediate=True)
                    logits = res['logits'].squeeze()
                    feats = res['feats']
                    l = smape(logits, y[idx])
                    r = stupid_gradcam(l, feats)
                    cams[n_m, o, idx] = r
                    losses[n_m,o] += l.detach().item()

        self.cams = cams
        return losses

    def calculate_rocs(self, x_val): #x_val_big

        def split_array_at_zero(arr):
            indices = np.where(arr != 0)[0]
            splits = []
            i = 0
            while i+1 < len(indices):
                start = i
                stop = start
                j = i+1
                while j < len(indices):
                    if indices[j] - indices[stop] == 1:
                        stop = j
                        j += 1
                    else:
                        break

                if start != stop:
                    splits.append((indices[start], indices[stop]))
                    i = stop
                else:
                    i += 1

            return splits

        assert len(self.ranking) == len(x_val)
        rocs = []
        for _ in range(len(self.models)):
            rocs.append([])

        for i, rank in enumerate(self.ranking):
            model = self.models[rank]
            cams = self.cams[rank][i]
            x = x_val[i]
            for offset, cam in enumerate(cams):
                max_r = np.max(cam) 
                if max_r == 0:
                    continue
                normalized = cam / max_r
                after_threshold = normalized * (normalized > self.threshold)

                if len(np.nonzero(after_threshold)[0]) > 0:
                    indidces = split_array_at_zero(after_threshold)
                    for (f, t) in indidces:
                        if t-f > 2:
                            rocs[rank].append(x[f+offset:(t+offset+1)])

        return rocs

    def find_best_forecaster(self, x):
        best_model = -1
        best_indice = None
        smallest_distance = 1e8

        for i, m in enumerate(self.models):
            # (x_1, x_2, x_3)
            # (1,   0.5, 0.2) => (x_1, x_2)

            x = x.squeeze()
            for r in self.rocs[i]:
                distance = dtw(r, x)
                if distance < smallest_distance:
                    best_model = i
                    smallest_distance = distance

        return best_model

class BaselineCompositor(BaseCompositor):

    def evaluate_on_validation(self, x_val):
        X, y = equal_split(x_val, self.lag, use_torch=True)
        losses = torch.zeros((len(self.models), len(X)))
        for n_m, m in enumerate(self.models):
            losses[n_m] = smape(m.forward(X.unsqueeze(1)).squeeze().detach(), y, axis=1)

        return losses.numpy()

    def calculate_rocs(self, x_val):
        X, y = equal_split(x_val, self.lag, use_torch=True)
        knn = KNeighborsClassifier(n_neighbors=3, metric=dtw)
        knn.fit(X, self.ranking)
        return knn

    def find_best_forecaster(self, x):
        return self.rocs.predict(x)[0]

class GC_EvenCompositor(GCCompositor):
    
    def evaluate_on_validation(self, x_val):
        cams = np.squeeze(np.zeros((len(self.models), x_val.shape[0], self.lag, self.lag)))
        losses = np.squeeze(np.zeros((len(self.models), x_val.shape[0])))
        for o, x in enumerate(x_val):
            X, y = equal_split(x, self.lag, use_torch=True)
            for n_m, m in enumerate(self.models):
                res = m.forward(X.unsqueeze(1), return_intermediate=True)
                logits = res['logits'].squeeze()
                feats = res['feats']
                l = smape(logits, y)
                r = simple_gradcam(l, feats)
                cams[n_m, o] = r
                losses[n_m,o] += l.detach().item()

        self.cams = cams
        return losses
