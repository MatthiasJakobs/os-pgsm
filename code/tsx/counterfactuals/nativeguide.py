# Implements Delaney et al. (2020): Instance-Based Counterfactual Explanations for Time Series Classification

import torch
import warnings
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

from tsx.utils import to_numpy
from tsx.distances import dtw, euclidian

class NativeGuide:

    # TODO: Should work with non-classifiers as well
    def __init__(self, model, X, y, distance='euclidian', batch_size=100):
        self.supported_distances = {
            'euclidian': euclidian,
            'dtw': dtw
        }

        self.distance = distance
        try:
            self.d = self.supported_distances[self.distance]
        except:
            raise ValueError("Unsupported distance meassure '{}' for NativeGuide".format(distance))

        self.batch_size = batch_size

        self.model = model
        self.X = np.squeeze(to_numpy(X))
        self.y = to_numpy(y)

        self.knn = KNeighborsClassifier(metric=self.d)
        self.knn.fit(self.X, self.y)

    # Generates n counterfactual examples for each datapoint in x
    # x.shape == (batch_size, ...)
    # TODO: 100 is just used for testing. Needs to depend on n 
    def generate(self, x, y, n=1, steps=1000):
        x = np.squeeze(to_numpy(x))
        y = to_numpy(y)
        dists, inds = self.knn.kneighbors(x, min(100, len(self.X)), True)

        # TODO: I think it always finds itself first. This could cause trouble, however, if x is not in self.x
        dists = dists[:, 1:]
        inds = inds[:, 1:]

        found_labels = self.y[inds]

        counterfactuals = []
        for i in range(len(x)):
            mask = found_labels[i] != y[i]
            guide_indices = inds[i][mask]

            if len(guide_indices) == 0:
                raise Exception("No guides found")
        
            if len(guide_indices) < n:
                warnings.warn("Merely found {} guides".format(len(guide_indices)), RuntimeWarning)

            guides = self.X[guide_indices[:n]]

            local_counterfactuals = []
            for guide in guides:
                if self.distance == 'euclidian':
                    betas = np.linspace(0, 1, num=steps)
                    candidates = np.array([beta * x[i] + (1-beta) * guide for beta in betas])
                    batches_needed = int(np.ceil(len(candidates) / self.batch_size))
                    predictions = []

                    for b_idx in range(batches_needed):
                        from_idx = b_idx * self.batch_size
                        to_idx = (b_idx+1) * self.batch_size
                        predictions.extend(self.model.predict(candidates[from_idx:to_idx]))

                    predictions = np.array(predictions)
                    hits = predictions != y[i]
                    local_counterfactuals = [(betas[idx], candidates[idx]) for idx, hit in enumerate(hits) if hit]
                    
                else:
                    raise NotImplementedError("Only euclidian supported right now")

            counterfactuals.append(local_counterfactuals[-n:])
        
        return counterfactuals
