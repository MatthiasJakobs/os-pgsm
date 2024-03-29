import torch
import numpy as np
import time

from tqdm import trange
from utils import gradcam, smape, dtw, mape
#from datasets.utils import sliding_split, equal_split
from sklearn.neighbors import KNeighborsClassifier

from seedpy import fixedseed

def sliding_split(x, lag, use_torch=False):
    assert len(x.shape) == 1

    X = np.zeros((len(x)-lag, lag))
    y = np.zeros(len(X))

    for i in range(len(X)):
        X[i] = x[i:(i+lag)]
        y[i] = x[(i+lag)]

    if use_torch:
        return torch.from_numpy(X).float(), torch.from_numpy(y).float()

    return X, y

def equal_split(x, lag, use_torch=False):
    if len(x.shape) == 2:
        x = x.reshape(-1)

    X = torch.zeros((int(len(x)/lag), lag))
    y = torch.zeros(int((len(x)/lag)))

    for i, idx in enumerate(range(0, len(x)-lag, lag)):
        X[i] = x[idx:idx+lag]
        y[i] = x[idx+lag]

    if not use_torch:
        return X.numpy(), y.numpy()
    else:
        return X.float(), y.float()

class BaseCompositor:

    def __init__(self, models, lag, big_lag):
        self.models = models
        # Assume identical lag for all modules no. Easily changable
        self.lag = lag
        self.big_lag = big_lag

    def calculate_rocs(self, x, losses, best_model):
        raise NotImplementedError()

    def find_best_forecaster(self, x):
        raise NotImplementedError()

    def split_val(self, X):
        raise NotImplementedError()

    def run(self, X_val, X_test, reuse_prediction=False, verbose=True, report_runtime=False, random_state=0):
        with fixedseed(torch, seed=random_state):
            self.rocs = [ [] for _ in range(len(self.models))]
            x_c, y_c = self.split_val(X_val)
            for x, y in zip(x_c, y_c):
                losses, cams = self.evaluate_on_validation(x, y) # Return-shapes: n_models, (n_models, blag-lag, lag)
                best_model = self.compute_ranking(losses) # Return: int [0, n_models]
                rocs_i = self.calculate_rocs(x, cams, best_model) # Return: List of vectors
                if rocs_i is not None:
                    self.rocs[best_model].extend(rocs_i)

            if report_runtime:
                before_total = time.time()
                preds, runtimes_per_decision = self.forecast_on_test(X_test, reuse_prediction, report_runtime=True)
                after_total = time.time()
                runtime_total = after_total - before_total

                return preds, runtime_total, runtimes_per_decision

            preds = self.forecast_on_test(X_test, reuse_prediction)
        return preds

    # TODO: Make faster
    def forecast_on_test(self, x_test, reuse_prediction=False, report_runtime=False):
        self.test_forecasters = []
        predictions = np.zeros_like(x_test)

        x = x_test[:self.lag]
        predictions[:self.lag] = x

        runtimes = []

        for x_i in range(self.lag, len(x_test)):
            if reuse_prediction:
                x = torch.from_numpy(predictions[x_i-self.lag:x_i]).unsqueeze(0)
            else:
                x = x_test[x_i-self.lag:x_i].unsqueeze(0)

            before_rt = time.time()
            best_model = self.find_best_forecaster(x)

            self.test_forecasters.append(best_model)
            predictions[x_i] = self.models[best_model].predict(x.unsqueeze(0))
            after_rt = time.time()
            runtimes.append(after_rt - before_rt)

        if report_runtime:
            return np.array(predictions), runtimes
        else:
            return np.array(predictions)

    def compute_ranking(self, losses):
        assert len(losses) == len(self.models)
        return np.argmin(losses)

class OS_PGSM_St(BaseCompositor):

    def __init__(self, models, lag, big_lag, threshold=0.5):
        super().__init__(models, lag, big_lag)
        self.threshold = threshold

    def split_val(self, X):
        return equal_split(X, self.big_lag, use_torch=True)

    def small_split(self, X):
        return sliding_split(X, self.lag, use_torch=True)

    def evaluate_on_validation(self, x_val, y_val):
        # Assume: x_val.shape == (n, blag)
        #         y_val.shape == (n, 1)
        
        losses = np.zeros((len(self.models)))#, x_val.shape[0]-self.lag))
        #cams = np.zeros((len(self.models), x_val.shape[0]-self.lag, self.lag))#x_val.shape[1]-self.lag, self.lag))
        all_cams = []
        X, y = self.small_split(x_val)
        nonzero_idcs = np.where(np.all(X.numpy() != 0, axis=1))[0]
        X = X[nonzero_idcs]
        y = y[nonzero_idcs]
        for n_m, m in enumerate(self.models):
            cams = []
            for idx in range(len(X)):
                test = X[idx].unsqueeze(0)
                res = m.forward(test.unsqueeze(1), return_intermediate=True)
                logits = res['logits'].squeeze()
                feats = res['feats']
                #l = mape(logits, y[idx])
                #l = torch.nn.MSELoss()(logits, y[idx])
                #l = smape(logits, y[idx])
                l = torch.nn.MSELoss()(logits, y[idx])
                r = gradcam(l, feats)
                cams.append(np.expand_dims(r, 0))
                losses[n_m] += l.detach().item()

            all_cams.append(np.concatenate(cams, axis=0))

        return losses, all_cams

    def calculate_rocs(self, x, cams, best_model): 
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
        # x.shape == (n, blag)

        cams = cams[best_model] # after shape: (blag-lag,lag)
        rocs = []

        # print(cams)
        # return
        for offset, cam in enumerate(cams):
            max_r = np.max(cam) 
            if max_r == 0:
                continue
            normalized = cam / max_r
            after_threshold = normalized * (normalized > self.threshold)
            if len(np.nonzero(after_threshold)[0]) > 0:
                indidces = split_array_at_zero(after_threshold)
                for (f, t) in indidces:
                    if t-f >= 2:
                        rocs.append(x[f+offset:(t+offset+1)])

        return rocs

    def find_best_forecaster(self, x, return_closest_roc=False):
        best_model = -1
        closest_roc = None
        best_indice = None
        smallest_distance = 1e8

        for i, m in enumerate(self.models):

            x = x.squeeze()
            for r in self.rocs[i]:
                distance = dtw(r, x)
                if distance < smallest_distance:
                    best_model = i
                    smallest_distance = distance
                    closest_roc = r

        if return_closest_roc:
            return best_model, closest_roc
        else:
            return best_model

class BaseAdaptive(BaseCompositor):

    def __init__(self, models, lag, big_lag, val_selection="sliding"):
        super().__init__(models, lag, big_lag)
        self.val_selection = val_selection
        self.roc_history = []

    def rebuild(self, X):
        self.rocs = [ [] for _ in range(len(self.models))]
        #x_c, y_c = sliding_split(X, self.big_lag, use_torch=True)
        x_c, y_c = self.split_val(X)
        for x, y in zip(x_c, y_c):
            losses, cams = self.evaluate_on_validation(x, y) # Return-shapes: n_models, (n_models, blag-lag, lag)
            best_model = self.compute_ranking(losses) # Return: int [0, n_models]
            rocs_i = self.calculate_rocs(x, cams, best_model) # Return: List of vectors
            if rocs_i is not None:
                self.rocs[best_model].extend(rocs_i)

    def detect_concept_drift(self, residuals, ts_length):
        raise NotImplementedError()

    def sliding_val(self, val_start, val_stop):
        return val_start+1, val_stop+1

    def stationary_val(self, val_start, val_stop):
        return val_start, val_stop+1

    def run(self, X_val, X_test, threshold=0.1, big_lag=25, verbose=True, dry_run=False, report_runtime=False, random_state=0):
        with fixedseed(torch, seed=random_state):
            self.drifts_detected = []
            val_start = 0
            val_stop = len(X_val) + self.lag
            X_complete = torch.cat([X_val, X_test])
            current_val = X_complete[val_start:val_stop]

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
            #x_val, _ = equal_split(current_val, big_lag, use_torch=True)
            self.rebuild(current_val)
            self.test_forecasters = []

            means.append(torch.mean(current_val).numpy())

            if verbose:
                used_range = trange(self.lag, len(X_test))
            else:
                used_range = range(self.lag, len(X_test))

            runtimes = []
            before_total = time.time()
            for target_idx in used_range: 
                f_test = (target_idx-self.lag)
                t_test = (target_idx)
                x = X_test[f_test:t_test] 

                val_start, val_stop = new_val(val_start, val_stop)
                current_val = X_complete[val_start:val_stop]
                means.append(torch.mean(current_val).numpy())

                residuals.append(means[-1]-means[-2])
                #print(f_test, t_test, residuals[-1])

                if len(residuals) > 1: #and len(X_complete) > (val_stop+offset):
                    if self.detect_concept_drift(residuals, len(current_val)):
                        self.drifts_detected.append(target_idx)
                        val_start = val_stop - len(X_val) - self.lag
                        current_val = X_complete[val_start:val_stop]
                        #x_val, _ = equal_split(current_val, big_lag, use_torch=True)
                        residuals = []
                        means = [torch.mean(current_val).numpy()]
                        self.roc_history.append(self.rocs)
                        if not dry_run:
                            self.rebuild(current_val)

                before_single = time.time()
                best_model = self.find_best_forecaster(x)
                self.test_forecasters.append(best_model)
                predictions.append(self.models[best_model].predict(x.unsqueeze(0).unsqueeze(0)))
                after_single = time.time()
                runtimes.append(after_single-before_single)

            after_total = time.time()
            total_runtime = after_total - before_total

            if report_runtime:
                return np.concatenate([X_test[:self.lag].numpy(), np.array(predictions)]), total_runtime, runtimes
            else:
                return np.concatenate([X_test[:self.lag].numpy(), np.array(predictions)])

class OS_PGSM_Per(OS_PGSM_St, BaseAdaptive):

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

class OS_PGSM(OS_PGSM_St, BaseAdaptive):
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

class KNN_ROC(BaseCompositor):

    def __init__(self, models, lag, big_lag):
        super().__init__(models, lag, big_lag)
        self.knn = KNeighborsClassifier(n_neighbors=3, metric=dtw)
        self.knn_x = []
        self.knn_y = []

    def split_val(self, X):
        return equal_split(X, self.lag, use_torch=True)

    def evaluate_on_validation(self, x, y):
        losses = torch.zeros((len(self.models)))
        for n_m, m in enumerate(self.models):
            pred = m.forward(x.unsqueeze(0).unsqueeze(0)).squeeze().detach()
            losses[n_m] = smape(pred, y)

        return losses.numpy(), None

    def calculate_rocs(self, x, cams, best_model):
        self.knn_x.append(x)
        self.knn_y.append(best_model)
        return None

    def forecast_on_test(self, x_test, reuse_prediction=False, report_runtime=False):
        x = torch.cat([x.unsqueeze(0) for x in self.knn_x], axis=0).numpy() 
        self.knn.fit(x, np.array(self.knn_y))
        return super().forecast_on_test(x_test, reuse_prediction=reuse_prediction, report_runtime=report_runtime)

    def find_best_forecaster(self, x):
        return self.knn.predict(x)[0]

class OS_PGSM_Int(OS_PGSM_St):

    def small_split(self, X):
        return equal_split(X, self.lag, use_torch=True)

class OS_PGSM_Euc(OS_PGSM_St):

    def calculate_rocs(self, x_val, cams, best_model): 

        cams = cams[best_model]
        rocs = []

        for offset, cam in enumerate(cams):
            x = x_val[offset:(self.lag+offset)]
            max_r = np.max(cam) 
            if max_r == 0:
                continue
            normalized = cam / max_r
            mask = (normalized > self.threshold).astype(np.float)

            if np.sum(mask) > 1:
                rocs.append(x * mask)

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

class OS_PGSM_Int_Euc(OS_PGSM_Euc, OS_PGSM_Int):

    def __init__(self, models, lag, big_lag, threshold=0.5):
        super().__init__(models, lag, big_lag, threshold=threshold)