import torch
import numpy as np
import time

from tqdm import trange, tqdm
from utils import simple_gradcam, stupid_gradcam
from datasets.utils import sliding_split, equal_split
from tsx.metrics import smape
from tsx.attribution import Grad_CAM
from tsx.distances import dtw
from sklearn.neighbors import KNeighborsClassifier

class BaseCompositor:

    def __init__(self, models, lag, big_lag):
        self.models = models
        # Assume identical lag for all modules no. Easily changable
        self.lag = lag
        self.big_lag = big_lag

    def calculate_rocs(self):
        raise NotImplementedError()

    def find_best_forecaster(self, x):
        raise NotImplementedError()

    def run(self, x_val, x_test, reuse_prediction=False):
        self.val_losses = self.evaluate_on_validation(x_val)
        self.ranking = self.compute_ranking()
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

    def compute_ranking(self):
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

class GC_Large(BaseCompositor):

    def __init__(self, models, lag, big_lag, threshold=0.5):
        super().__init__(models, lag, big_lag)
        self.threshold = threshold

    def evaluate_on_validation(self, x_val):
        cams = np.zeros((len(self.models), x_val.shape[0], x_val.shape[1]-self.lag, self.lag))
        losses = np.zeros((len(self.models), x_val.shape[0]))
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

    def calculate_rocs(self, x_val): 

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

class BaseAdaptive(BaseCompositor):

    def __init__(self, models, lag, big_lag, val_selection="sliding"):
        super().__init__(models, lag, big_lag)
        self.val_selection = val_selection

    def rebuild(self, x_val):
        self.val_losses = self.evaluate_on_validation(x_val)
        self.ranking = self.compute_ranking()
        self.rocs = self.calculate_rocs(x_val)

    def detect_concept_drift(self, residuals, ts_length):
        raise NotImplementedError()

    def sliding_val(self, val_start, val_stop):
        return val_start+1, val_stop+1

    def stationary_val(self, val_start, val_stop):
        return val_start, val_stop+1

    def run(self, X_val, X_test, threshold=0.1, big_lag=25, verbose=True, dry_run=False):
        self.drifts_detected = []
        val_start = 0
        val_stop = len(X_val) + self.lag
        X_complete = torch.cat([X_val, X_test])
        current_X = X_complete[val_start:val_stop]

        means = []
        residuals = []
        predictions = []
        offset = 1

        if self.val_selection == "sliding":
            new_val = self.sliding_val
        elif self.val_selection == "stationary":
            new_val = self.stationary_val
        else:
            raise NotImplementedError("Unknown val selection method {}".format(val_selection))

        # Initial creation of ROCs
        x_val, _ = equal_split(current_X, big_lag, use_torch=True)
        self.rebuild(x_val)
        self.test_forecasters = []

        means.append(torch.mean(current_X).numpy())

        if verbose:
            used_range = trange(self.lag, len(X_test))
        else:
            used_range = range(self.lag, len(X_test))

        for target_idx in used_range: 
            f_test = (target_idx-self.lag)
            t_test = (target_idx)
            x = X_test[f_test:t_test] 

            val_start, val_stop = new_val(val_start, val_stop)
            current_X = X_complete[val_start:val_stop]
            means.append(torch.mean(current_X).numpy())

            residuals.append(means[-1]-means[-2])

            if len(residuals) > 1: #and len(X_complete) > (val_stop+offset):
                if self.detect_concept_drift(residuals, len(current_X)):
                    self.drifts_detected.append(target_idx)
                    val_start = val_stop - len(X_val) - self.lag
                    current_X = X_complete[val_start:val_stop]
                    x_val, _ = equal_split(current_X, big_lag, use_torch=True)
                    residuals = []
                    means = [torch.mean(current_X).numpy()]
                    if not dry_run:
                        self.rebuild(x_val)

            best_model = self.find_best_forecaster(x)
            self.test_forecasters.append(best_model)
            predictions.append(self.models[best_model].predict(x.unsqueeze(0).unsqueeze(0)))

        return np.concatenate([X_test[:self.lag].numpy(), np.array(predictions)])

class GC_Large_Adaptive_Periodic(GC_Large, BaseAdaptive):

    def __init__(self, models, lag, big_lag, threshold=0.5, periodicity=None, val_selection="sliding"):
        super().__init__(models, lag, big_lag, threshold=threshold)
        self.periodicity = periodicity
        self.val_selection = val_selection

    def run(self, X_val, X_test, **kwargs):
        if self.periodicity is None:
            self.periodicity = int(len(X_test) / 10.0)

        return BaseAdaptive.run(self, X_val, X_test, **kwargs)

    def detect_concept_drift(self, residuals, x_len):
        return len(residuals) >= self.periodicity

class GC_Large_Adaptive_PageHinkley(GC_Large, BaseAdaptive):

    def __init__(self, models, lag, big_lag, threshold=0.5, lamb=0.2, delta=0.01):
        super().__init__(models, lag, big_lag, threshold=threshold)
        self.lamb = lamb
        self.delta = delta

    def detect_concept_drift(self, residuals):
        sr = np.zeros_like(residuals)
        m_t = np.zeros_like(residuals)
        M_T = 1
        for i in range(1, len(residuals)):
            x_i = residuals[:i]
            mean_upto_t = np.mean(x_i)

            sr[i] = sr[i-1] + residuals[i]
            m_t[i] = m_t[i-1] + residuals[i] + sr[i] - self.delta
            M_T = min(M_T, m_t[i])

        if m_t[-1] - M_T >= self.lamb:
            return True
        else:
            return False

class GC_Large_Adaptive_Hoeffding(GC_Large, BaseAdaptive):
    def __init__(self, models, lag, big_lag, threshold=0.5, val_selection="sliding", delta=0.95):
        super().__init__(models, lag, big_lag, threshold=threshold)
        self.delta = delta
        self.val_selection = val_selection

    def detect_concept_drift(self, residuals, ts_length):
        #ts_length = len(residuals)+1
        residuals = np.array(residuals)

        # Empirical range of residuals
        R = 1
        #R = np.max(np.abs(residuals)) # R = 1 

        epsilon = np.sqrt((R**2)*np.log(1/self.delta) / (2*ts_length))

        if np.abs(residuals[-1]) <= np.abs(epsilon):
            return False
        else:
            return True

class Baseline(BaseCompositor):

    def evaluate_on_validation(self, x_val):
        X, y = equal_split(x_val, self.lag, use_torch=True)
        losses = torch.zeros((len(self.models), len(X)))
        for n_m, m in enumerate(self.models):
            pred = m.forward(X.unsqueeze(1)).squeeze().detach()
            losses[n_m] = smape(pred, y, axis=1)

        return losses.numpy()

    def calculate_rocs(self, x_val):
        X, y = equal_split(x_val, self.lag, use_torch=True)
        knn = KNeighborsClassifier(n_neighbors=3, metric=dtw)
        knn.fit(X, self.ranking)
        return knn

    def find_best_forecaster(self, x):
        return self.rocs.predict(x)[0]

class GC_Small(GC_Large):
    
    def evaluate_on_validation(self, x_val):
        cams = np.zeros((len(self.models), x_val.shape[0], int(self.big_lag/self.lag), self.lag))
        losses = np.zeros((len(self.models), x_val.shape[0]))
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

class GC_Large_Euclidian(GC_Large):

    def calculate_rocs(self, x_val): 

        assert len(self.ranking) == len(x_val)
        rocs = []
        for _ in range(len(self.models)):
            rocs.append([])

        for i, rank in enumerate(self.ranking):
            model = self.models[rank]
            cams = self.cams[rank][i]
            #x = x_val[i]
            for offset, cam in enumerate(cams):
                x = x_val[i][offset:(self.lag+offset)]
                max_r = np.max(cam) 
                if max_r == 0:
                    continue
                normalized = cam / max_r
                mask = (normalized > self.threshold).astype(np.float)

                if np.sum(mask) > 1:
                    rocs[rank].append(x * mask)

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
                mask = (r != 0).float()
                distance = torch.cdist(r.unsqueeze(0).float(), (x*mask).unsqueeze(0), 2).squeeze().item()
                if distance < smallest_distance:
                    best_model = i
                    smallest_distance = distance

        return best_model

class GC_Small_Euclidian(GC_Large_Euclidian, GC_Small):

    def __init__(self, models, lag, big_lag, threshold=0.5):
        super().__init__(models, lag, big_lag, threshold=threshold)