import numpy as np
from fastdtw import fastdtw

def dtw(s, t):
    return fastdtw(s, t)[0]
