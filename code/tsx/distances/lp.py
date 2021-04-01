import numpy as np

# l1 (Manhattan) norm
def manhattan(s, t):
    return np.linalg.norm(s-t, 1)

# l2 (Euclidian) norm
def euclidian(s, t):
    return np.linalg.norm(s-t, 2)

# linf norm
def linf(s, t):
    return np.linalg.norm(s-t, np.inf)