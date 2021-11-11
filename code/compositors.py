from os import close
import torch
import numpy as np
import time

from utils import gradcam
from datasets.utils import sliding_split, roc_matrix, roc_mean
from utils import smape, dtw, mse, mape, mae, msle, mbe
from tslearn.clustering import TimeSeriesKMeans
from tslearn.utils import to_time_series_dataset
from seedpy import fixedseed

class OS_PGSM:

    def __init__(self, models, config, random_state=0):
        self.models = models
        self.config = config
        # Assume identical lag for all modules no. Easily changable
        self.lag = config.get("k", 5)
        self.big_lag = config.get("big_lag", 25)
        self.topm = config.get("topm", 1)
        self.nr_clusters_single = config.get("nr_clusters_single", 1) # Default value: No clustering
        self.threshold = config.get("smoothing_threshold", 0.5)
        self.nr_clusters_ensemble = config.get("nr_clusters_ensemble", 1) # Default value: No clustering
        self.n_omega = config.get("n_omega", self.lag)
        self.z = config.get("z", 1)
        self.invert_relu = config.get("invert_relu", False)
        self.roc_take_only_best = config.get("roc_take_only_best", True)
        self.small_z = config.get("small_z", 1)
        self.delta = config.get("delta", 0.05)
        self.roc_mean = config.get("roc_mean", False)
        self.rng = np.random.RandomState(random_state)
        self.random_state = random_state
        self.concept_drift_detection = config.get("concept_drift_detection", None)
        self.drift_type = config.get("drift_type", "ospgsm")
        self.invert_loss = config.get("invert_loss", False)

        if self.topm != 1 and self.nr_clusters_ensemble != 1:
            assert self.nr_clusters_ensemble < self.topm

    def ensemble_predict(self, x, subset=None):
        if subset is None:
            predictions = [m.predict(x) for m in self.models]
        else:
            predictions = [self.models[i].predict(x) for i in subset]

        return np.mean(predictions)

    def shrink_rocs(self):
        # Make RoCs more concise by considering cluster centers instead of all RoCs
        if self.nr_clusters_single > 1:
            for i, single_roc in enumerate(self.rocs): 

                # Skip clustering if there would be no shrinking anyway
                if len(single_roc) <= self.nr_clusters_single:
                    continue

                tslearn_formatted = to_time_series_dataset(single_roc)
                km = TimeSeriesKMeans(n_clusters=self.nr_clusters_single, metric="dtw", random_state=self.rng)
                km.fit(tslearn_formatted)

                # Choose cluster centers as new RoCs
                new_roc = km.cluster_centers_.squeeze()
                self.rocs[i] = []
                for roc in new_roc:
                    self.rocs[i].append(torch.tensor(roc).float())

    def rebuild_rocs(self, X):
        self.rocs = [ [] for _ in range(len(self.models))]

        x_c, y_c = self.split_n_omega(X)
        # Create RoCs
        for x, y in zip(x_c, y_c):
            losses, cams = self.evaluate_on_validation(x, y) # Return-shapes: n_models, (n_models, blag-lag, lag)
            best_model = self.compute_ranking(losses) # Return: int [0, n_models]
            all_rocs = self.calculate_rocs(x, cams, best_model) # Return: List of vectors
            if self.roc_take_only_best:
                rocs_i = all_rocs[best_model]
                if rocs_i is not None:
                    self.rocs[best_model].extend(rocs_i)
            else:
                for i, rocs_i in enumerate(all_rocs):
                    if rocs_i is not None:
                        self.rocs[i].extend(rocs_i)


    def detect_concept_drift(self, residuals, ts_length, test_length, R=1):
        if self.concept_drift_detection is None:
            raise RuntimeError("Concept drift should be detected even though config does not specify method", self.concept_drift_detection)

        if self.concept_drift_detection == "periodic":
            return len(residuals) >= int(test_length / 10.0)
        elif self.concept_drift_detection == "hoeffding":
            residuals = np.array(residuals)

            # Empirical range of residuals
            #R = 1
            #R = np.max(np.abs(residuals)) # R = 1 

            epsilon = np.sqrt((R**2)*np.log(1/self.delta) / (2*ts_length))

            if np.abs(residuals[-1]) <= np.abs(epsilon):
                return False
            else:
                return True

    def adaptive_online_roc_rebuild(self, X_val, X_test):
        # Adaptive method from OS-PGSM
        self.test_forecasters = []
        self.drifts_detected = []
        val_start = 0
        val_stop = len(X_val) + self.lag
        X_complete = torch.cat([X_val, X_test])
        current_val = X_complete[val_start:val_stop]

        residuals = []
        predictions = []
        means = [torch.mean(current_val).numpy()]

        for target_idx in range(self.lag, len(X_test)):
            f_test = (target_idx-self.lag)
            t_test = (target_idx)
            x = X_test[f_test:t_test] 

            # TODO: Only sliding val, since default paramter
            val_start += 1
            val_stop += 1

            current_val = X_complete[val_start:val_stop]
            means.append(torch.mean(current_val).numpy())

            residuals.append(means[-1]-means[-2])

            if len(residuals) > 1: 
                if self.detect_concept_drift(residuals, len(current_val), len(X_test)):
                    self.drifts_detected.append(target_idx)
                    val_start = val_stop - len(X_val) - self.lag
                    current_val = X_complete[val_start:val_stop]
                    residuals = []
                    means = [torch.mean(current_val).numpy()]
                    self.rebuild_rocs(current_val)

            best_model = self.find_best_forecaster(x)
            self.test_forecasters.append(best_model)
            predictions.append(self.ensemble_predict(x.unsqueeze(0).unsqueeze(0), subset=best_model))

        return np.concatenate([X_test[:self.lag].numpy(), np.array(predictions)])

    def adaptive_monitor_min_distance(self, X_val, X_test):
        self.test_forecasters = []
        self.drifts_detected = []

        # Save all residuals to compute empirical mean
        all_residuals = []

        residuals = []
        predictions = []
        x = X_test[:self.lag]

        # Get min distance to current input pattern
        best_models, closest_rocs = self.find_best_forecaster(x, return_closest_roc=True)

        # TODO: Is this the closest one?
        initial_min = dtw(closest_rocs[0], x)

        model_subset = self.reduce_best_m(best_models, closest_rocs)

        predictions.append(self.ensemble_predict(x.unsqueeze(0).unsqueeze(0), subset=model_subset))

        # For each new pattern: 
        for target_idx in range(self.lag+1, len(X_test)):
            f_test = (target_idx-self.lag)
            t_test = (target_idx)
            x = X_test[f_test:t_test] 

            #best_models, closest_rocs = self.find_best_forecaster(x, return_closest_roc=True)

            # calc min distance and compute residuals
            new_min = dtw(closest_rocs[0], x)
            residuals.append(new_min - initial_min)
            all_residuals.append(new_min - initial_min)

            empirical_range = np.abs(np.min(all_residuals) - np.max(all_residuals))

            # pass residuals into drift detection
            if len(residuals) > 1 and self.detect_concept_drift(residuals, len(residuals), len(X_test), R=empirical_range):
                # if drift: recreate ensemble and set new min distance for comparison
                #print("Drift detected at iteration", target_idx, residuals)
                self.drifts_detected.append(target_idx)

                best_models, closest_rocs = self.find_best_forecaster(x, return_closest_roc=True)
                model_subset = self.reduce_best_m(best_models, closest_rocs)
                initial_min = new_min
                residuals = []

            predictions.append(self.ensemble_predict(x.unsqueeze(0).unsqueeze(0), subset=model_subset))

        return np.concatenate([X_test[:self.lag].numpy(), np.array(predictions)])

    # TODO: No runtime reports
    def run(self, X_val, X_test, reuse_prediction=False):
        with fixedseed(torch, seed=self.random_state):
            self.rebuild_rocs(X_val)
            self.shrink_rocs()        

            if self.concept_drift_detection is None:
                return self.forecast_on_test(X_test, reuse_prediction=reuse_prediction)

            if self.drift_type == "ospgsm":
                forecast = self.adaptive_online_roc_rebuild(X_val, X_test)
            elif self.drift_type == "min_distance_change":
                forecast = self.adaptive_monitor_min_distance(X_val, X_test)
            else:
                raise NotImplementedError(f"Drift type {self.drift_type} not implemented")

            return forecast


    def reduce_best_m(self, best_models, clostest_rocs):
        if self.nr_clusters_ensemble == 1:
            return best_models

        # Edge case where there were not enough topm models to allow for further shrinkage
        if self.nr_clusters_ensemble > len(best_models):
            print(f"WARNING: No clustering since nr_cluster_ensemble={self.nr_clusters_ensemble} > len(best_models)={len(best_models)}")
            return best_models

        # Cluster into the desired number of left-over models.
        tslearn_formatted = to_time_series_dataset(clostest_rocs)
        km = TimeSeriesKMeans(n_clusters=self.nr_clusters_ensemble, metric="dtw", random_state=self.rng)
        C = km.fit_predict(tslearn_formatted)
        C_count = np.bincount(C)

        # Final model selection
        G = []

        for p in range(len(C_count)):
            if C_count[p] == 1:
                G.append(p)
                continue
            
            # Under all cluster members, find the one maximizing distance to current point
            cluster_member_indices = np.where(C == p)[0]
            # Since the best_models (and closest_rocs) are sorted by distance to x (ascending), 
            # choosing the last one will always maximize distance
            if len(cluster_member_indices) > 0:
                G.append(cluster_member_indices[-1])

        if len(G) > 0:
            return G
        else:
            return best_models
        
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

            # Find top-m model who contain the smallest distances to x in their RoC
            best_models, closest_rocs = self.find_best_forecaster(x, return_closest_roc=True)

            # Further reduce number of best models by clustering
            best_models = self.reduce_best_m(best_models, closest_rocs)

            self.test_forecasters.append(best_models)
            for i in range(len(best_models)):
                predictions[x_i] += self.models[best_models[i]].predict(x.unsqueeze(0))

            predictions[x_i] = predictions[x_i] / len(best_models)

            after_rt = time.time()
            runtimes.append(after_rt - before_rt)

        if report_runtime:
            return np.array(predictions), runtimes
        else:
            return np.array(predictions)

    def compute_ranking(self, losses):
        assert len(losses) == len(self.models)
        return np.argmin(losses)

    def split_n_omega(self, X):
        if self.roc_mean:
            return sliding_split(X, self.n_omega+1, z=self.z, use_torch=True)
        else:
            return sliding_split(X, self.n_omega, z=self.z, use_torch=True)

    def small_split(self, X):
        return sliding_split(X, self.lag, z=self.small_z, use_torch=True)

    def evaluate_on_validation(self, x_val, y_val):
        losses = np.zeros((len(self.models)))

        if self.roc_mean:
            all_cams = np.zeros((len(self.models), self.n_omega))
        else:
            all_cams = []

        X, y = self.small_split(x_val)
        for n_m, m in enumerate(self.models):
            cams = []
            for idx in range(len(X)):
                test = X[idx].unsqueeze(0)
                res = m.forward(test.unsqueeze(1), return_intermediate=True)
                logits = res['logits'].squeeze()
                feats = res['feats']
                #l = smape(logits, y[idx])
                l = mse(logits, y[idx])
                r = gradcam(l, feats)
                cams.append(np.expand_dims(r, 0))
                losses[n_m] += l.detach().item()
            cams = np.concatenate(cams, axis=0)

            if self.roc_mean:
                all_cams[n_m] = roc_mean(roc_matrix(cams, z=1))
            else:
                all_cams.append(cams)

        if not self.roc_mean:
            all_cams = np.array(all_cams)

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

        all_rocs = []
        for i in range(len(cams)):
            rocs = []
            cams_i = cams[i] 
            #cams_i = cams[best_model] 

            if len(cams_i.shape) == 1:
                cams_i = np.expand_dims(cams_i, 0)

            for offset, cam in enumerate(cams_i):
                # Normalize CAMs
                if self.invert_relu:
                    after_threshold = (cam == 0).astype(np.int8)
                    condition = np.sum(after_threshold) > 0
                else:
                    max_r = np.max(cam)
                    if max_r == 0:
                        continue
                    normalized = cam / max_r

                    # Extract all subseries divided by zeros
                    after_threshold = normalized * (normalized > self.threshold)
                    condition = len(np.nonzero(after_threshold)[0]) > 0

                if condition:
                    indidces = split_array_at_zero(after_threshold)
                    for (f, t) in indidces:
                        if t-f >= 2:
                            rocs.append(x[f+offset:(t+offset+1)])

            all_rocs.append(rocs)
        
        return all_rocs

    def find_best_forecaster(self, x, return_closest_roc=False):
        model_distances = np.ones(len(self.models)) * 1e10
        closest_rocs_agg = [None]*len(self.models)

        for i, m in enumerate(self.models):

            x = x.squeeze()
            for r in self.rocs[i]:
                distance = dtw(r, x)
                if distance < model_distances[i]:
                    model_distances[i] = distance
                    closest_rocs_agg[i] = r

        top_models = np.argsort(model_distances)[:self.topm]
        closest_rocs = []
        for i in top_models:
            if closest_rocs_agg[i] is not None:
                closest_rocs.append(closest_rocs_agg[i])

        # There might be more desired models than rocs available, so we need to reduce top models accordingly
        top_models = top_models[:len(closest_rocs)]

        if return_closest_roc:
            return top_models, closest_rocs

        return top_models

class Inv_OS_PGSM(OS_PGSM):

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

        cams = cams[best_model] 
        rocs = []

        if len(cams.shape) == 1:
            cams = np.expand_dims(cams, 0)

        for offset, cam in enumerate(cams):
            after_threshold = (cam == 0).astype(np.int8)
            if np.sum(after_threshold) > 0:
                indidces = split_array_at_zero(after_threshold)
                for (f, t) in indidces:
                    if t-f >= 2:
                        rocs.append(x[f+offset:(t+offset+1)])
        
        return rocs

