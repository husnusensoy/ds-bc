import numpy as np
from functools import partial


def minkowski(x: np.ndarray, y: np.ndarray, p: int) -> float:
    """Calculate generalized metric between vector x and vector y

    :param x:
    :param y:
    :param p:
    :return: distance
    """

    return np.power(np.power(np.abs(x - y), p).sum(), 1 / p)


l2 = euc = partial(minkowski, p=2)
l1 = partial(minkowski, p=1)
