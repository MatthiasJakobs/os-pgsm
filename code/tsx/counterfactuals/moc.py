import numpy as np

from itertools import permutations
from random import shuffle
from tsx.utils import NSGA2
from tsx.distances import euclidian, dtw
from tsx.utils import to_numpy, sigmoid
from sklearn.neighbors import KNeighborsClassifier


# Implements the approach by Dandl et al. (2020), but for time series data
# https://arxiv.org/abs/2004.11165
class MOC(NSGA2):

    # TODO: Only for time series data of same length
    def __init__(self, model, X, y, **kwargs):
        self.model = model
        self.X = to_numpy(X)
        self.y = to_numpy(y)
        self.nr_classes = len(np.unique(y))
        self.knn = KNeighborsClassifier(n_neighbors=3, algorithm="ball_tree", metric=dtw, n_jobs=4)
        self.knn.fit(self.X, self.y)

        self.mutation_rate = 0.5 # how many individuals to mutate (on average)
        self.mutation_strength = 0.4 # how many points in each individual to mutate (on average)

        super().__init__(**kwargs)

    def generate(self, x_star, target=None):
        x_star = to_numpy(x_star)
        if target is None:
            # untargeted attack
            # TODO
            raise NotImplementedError()
            
        criterias = [
            self.generate_obj_1(self.nr_classes, target, True), #TODO
            self.generate_obj_2(x_star, dtw),
            self.generate_obj_3(x_star),
            self.generate_obj_4()
        ]

        self.set_criterias(criterias)
        fitness, xs = self.run(guide=x_star)
        true_counterfactual_indices = fitness[:, 0] == 0
        return fitness[true_counterfactual_indices], xs[true_counterfactual_indices]

    def _random_individuals(self, n, guide=None):
        # assumption: zero mean ts
        ind_length = len(self.X[0])

        # just copy initial point
        if guide is not None:
            if guide.ndim == 1:
                return self.mutation(np.tile(np.expand_dims(guide, 0), (n, 1)), override_mutation_rate=1.0)
            if guide.ndim == 2 and guide.shape[0] == 1:
                return self.mutation(np.tile(guide, (n, 1)), override_mutation_rate=1.0)
            raise Exception("guide has unsuported shape {}".format(guide.shape))

        # cumsum
        return np.cumsum(np.random.normal(size=(n, ind_length)), axis=1)

        # naive random 
        # return np.random.rand(n * ind_length).reshape(n, -1)

    def recombination(self, x):
        # 1 point crossover (naive)
        recomb_combinations = list(permutations(np.arange(len(x)), 2))
        shuffle(recomb_combinations)
        recomb_combinations = recomb_combinations[:self.offspring_size]
        to_return = np.zeros((self.offspring_size, len(x[0])))
        for i, (s, t) in enumerate(recomb_combinations):
            crossover_point = np.random.randint(1, len(x[0]))
            to_return[i][:crossover_point] = x[s][:crossover_point]
            to_return[i][crossover_point:] = x[t][crossover_point:]

        return to_return 

    def mutation(self, x, override_mutation_rate=None):
        ind_length = len(self.X[0])
        assert self.parent_size == self.offspring_size

        # mask to choose individuals to mutate
        if override_mutation_rate is not None:
            rate_mask = np.random.binomial(1, p=override_mutation_rate, size=len(x))
        else:
            rate_mask = np.random.binomial(1, p=self.mutation_rate, size=len(x))

        for idx in rate_mask.nonzero()[0]:
            # mask to choose features to mutate
            strength_mask = np.random.binomial(1, p=self.mutation_strength, size=ind_length)
            x[idx] = x[idx] + strength_mask * np.random.normal(size=ind_length) * 0.6

        return x

    # prediction close to desired outcome
    def generate_obj_1(self, nr_classes, out_class_index, targeted):
        threshold = 1.0 / nr_classes

        def obj_1(x):
            predictions = self.model.proba(x)

            # weird edge case for binary sklearn classifier without proba
            if self.nr_classes == 2:
                predictions_class_one = np.expand_dims(sigmoid(predictions), 1)
                predictions = np.concatenate((1-predictions_class_one, predictions_class_one), axis=1)
                assert np.max(predictions) <= 1 and np.min(predictions) >= 0, print(predictions_class_one)

            predictions = predictions[:, out_class_index]

            mask = predictions > threshold
            inv_mask = np.logical_not(mask)

            predictions[mask] = 0
            predictions[inv_mask] = np.abs(predictions[inv_mask]-threshold)

            return predictions.squeeze()

        return obj_1

    # distance close to initial point
    # x_star is initial point
    def generate_obj_2(self, x_star, distance):

        def obj_2(x):
            n_elements = x.shape[0]
            distances = np.zeros(n_elements)
            for i in range(n_elements):
                distances[i] = distance(np.expand_dims(x[i], 0), x_star)

            return distances

        return obj_2

    # differs from x_star only in few dimensions
    def generate_obj_3(self, x_star, threshold=0.01):

        def obj_3(x):
            # l_0 norm
            # TODO: Use np.isclose ?
            difference = np.abs(x-x_star)
            difference[difference < threshold] = 0.0
            difference[difference >= threshold] = 1.0
            return np.sum(difference, axis=1)

        return obj_3

    # plausible datapoint
    # use mean of DTW between k nearest neighbours (analog to Dandl paper, but with DTW)
    def generate_obj_4(self, k=3):

        def obj_4(x):
            neighbors_distance, _ = self.knn.kneighbors(x, k, True)
            return np.mean(neighbors_distance, axis=1)

        return obj_4
        
