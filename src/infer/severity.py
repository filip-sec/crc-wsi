import numpy as np
from typing import Sequence

def compute_severity(probs: np.ndarray, weights: Sequence[float] = (0, 1, 2)) -> np.ndarray:
    """
    probs: array (N,3) with columns [p0,p1,p2] = [NNEO, LGD, HGD]
    returns: severity score = w1*p(LGD) + w2*p(HGD)
    """
    assert probs.ndim == 2 and probs.shape[1] == 3, "probs must be (N,3)"
    _, w1, w2 = weights
    return probs[:, 1] * w1 + probs[:, 2] * w2
