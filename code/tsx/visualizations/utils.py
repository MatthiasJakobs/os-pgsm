import numpy as np

def calc_optimal_grid(n):
    # Try to distribute n images as best as possible
    # For now: simple overestimation
    sides = int(np.ceil(np.sqrt(n)))
    return sides, sides
