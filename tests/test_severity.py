import numpy as np
from src.infer.severity import compute_severity

def test_monotone_severity():
    a = np.array([[0.8, 0.1, 0.1],
                  [0.1, 0.6, 0.3]])  # second row has higher LGD+HGD
    s = compute_severity(a)
    assert s[1] > s[0]
