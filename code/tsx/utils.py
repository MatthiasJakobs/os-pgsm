import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from tqdm import trange
from itertools import combinations


class NSGA2:
    # based on implementation from https://github.com/haris989/NSGA-II/blob/master/NSGA%20II.py
    # NSGA2 is configured to always minimize (with 0 being optimal), so make sure to configure your criteria functions accordingly

    def __init__(self, parent_size=10, offspring_size=10, dimensions=3, generations=10, log_generations=False):
        self.parent_size = parent_size
        self.offspring_size = offspring_size
        self.dimensions = dimensions
        self.generations = generations
        self.log_generations = log_generations

    def set_criterias(self, criterias):
        self.criterias = criterias

    def _random_individuals(self, n):
        return np.random.binomial(1, p=0.5, size=3*n).reshape(n, 3)

    def _apply_functions(self, X):
        result = np.zeros((len(X), len(self.criterias)))
        for i, f in enumerate(self.criterias):
            result[:, i] = f(X)

        return result

    # input: (batch_size, len(self.criterias))
    def fast_non_dominated_sort(self, individual_performs):
        fronts = []
        indices = np.arange(len(individual_performs))
        while len(indices) != 0:
            permutations = list(combinations(np.arange(len(indices)), 2))
            domination_count = np.zeros(len(indices))
            dominates = [[] for _ in range(len(indices))]
            for (ix1, ix2) in permutations:
                if self._dominates(individual_performs[ix1], individual_performs[ix2]):
                    domination_count[ix2] += 1
                    dominates[ix1].append(ix2)
                if self._dominates(individual_performs[ix2], individual_performs[ix1]):
                    domination_count[ix1] += 1
                    dominates[ix2].append(ix1)

            domination_mask = domination_count == 0
            inv_domination_mask = np.logical_not(domination_mask)
            non_dominated = np.where(domination_mask)[0]
            for ix in non_dominated:
                for i in dominates[ix]:
                    domination_count[i] -= 1
            if np.all(inv_domination_mask):
                fronts.append(indices[inv_domination_mask])
                return fronts
            fronts.append(indices[non_dominated])
            indices = indices[inv_domination_mask]
        return fronts

    def recombination(self, x):
        return self._random_individuals(len(x))

    def mutation(self, x):
        return self._random_individuals(len(x))

    def run(self, guide=None):
        parents = self._random_individuals(self.parent_size, guide=guide)
        offspring = self._random_individuals(self.offspring_size, guide=guide)

        mean_metrics = np.zeros((self.generations, len(self.criterias)))
        var_metrics = np.zeros((self.generations, len(self.criterias)))

        range_generations = range(self.generations)
        if not self.log_generations:
            range_generations = trange(self.generations)

        for g in range_generations:
            offspring = self.recombination(parents.copy())
            offspring = self.mutation(offspring)
            population = np.concatenate((parents, offspring))

            population = np.unique(population, axis=0)
            assert len(population) >= self.parent_size

            evaluation = self._apply_functions(population)
            fronts = self.fast_non_dominated_sort(evaluation)

            parent_indices = []
            for i, front in enumerate(fronts):

                if (len(front) + len(parent_indices)) <= self.parent_size:
                    # take entire front
                    parent_indices.extend(list(front))
                else:
                    # do selection
                    cd = self.crowding_distance(evaluation[front])

                    # randomized argsort descending
                    perm = np.random.permutation(len(cd))
                    sorted_indices = np.argsort(cd[perm])
                    sorted_indices = np.flip(perm[sorted_indices])
                    sorted_indices = sorted_indices[:(self.parent_size - len(parent_indices))]
                    parent_indices.extend(list(front[sorted_indices]))

                    # because of duplicate removal: maybe needs another round
                    if len(parent_indices) == self.parent_size:
                        break
                    if len(parent_indices) > self.parent_size:
                        raise RuntimeError("Cannot have more parents than specified")

            parents = population[parent_indices]
            evaluation = evaluation[parent_indices]

            mean_metrics[g] = np.mean(evaluation, axis=0)
            var_metrics[g] = np.var(evaluation, axis=0)

            if self.log_generations:
                print("GENERATION {}".format(g+1))
                print(evaluation)
                print("-"*20)
        return evaluation, parents

    def crowding_distance(self, individuals):
        n_ind, n_obj = individuals.shape

        eps = 1e-9 # to prevent possible division by zero
        distances = np.zeros(n_ind)

        for c in range(n_obj):

            sorted_indices = np.argsort(individuals[:, c])
            distances[sorted_indices[0]] += np.inf
            distances[sorted_indices[-1]] += np.inf
            normalization = np.max(individuals[:, c]) - np.min(individuals[:, c])
            normalization = eps if normalization == 0 else normalization

            for j in range(1, n_ind-1):
                dist = individuals[sorted_indices[j+1]][c] - individuals[sorted_indices[j-1]][c]
                dist /= normalization
                distances[sorted_indices[j]] += dist

        return distances
        
    def _dominates(self, a, b):
        return np.all(a <= b) and np.any(a < b)

def to_numpy(x):
    if isinstance(x, type(torch.zeros(1))):
        if x.requires_grad:
            return x.detach().numpy()
        else:
            return x.numpy()
    if isinstance(x, type(pd.Series(data=[1,2]))):
        return x.to_numpy()
    if isinstance(x, type(np.zeros(1))):
        return x

    raise ValueError("Input of type {} cannot be converted to numpy array".format(type(x)))

def sigmoid(x):
    return 1/(1 + np.exp(-x)) 

def prepare_for_pytorch(x, batch=True, channel=True):
    if isinstance(x, type(np.zeros(0))):
        x = torch.from_numpy(x)
    
    # Missing batch and channel information
    if batch and len(x.shape) == 1:
        x = x.unsqueeze(0)

    if channel and len(x.shape) == 2:
        x = x.unsqueeze(1)

    return x
