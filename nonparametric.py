import numpy as np
from typing import Callable, Optional
from metric import euc


def weighted_mode(clazzes: np.ndarray, weight: np.ndarray) -> int:
    ws = {}
    for c, w in zip(clazzes, weight):
        ws[c] = ws.get(c, 0) + w

    return max(ws.items(), key=lambda t: t[1])[0]


def majority(clazzes: np.ndarray, distance: np.ndarray):
    return weighted_mode(clazzes, np.ones(len(distance)))


def exponential(clazzes: np.ndarray, distance: np.ndarray):
    return weighted_mode(clazzes, np.exp(-distance))


def inverse(clazzes: np.ndarray, distance: np.ndarray):
    return weighted_mode(clazzes, 1 / distance)


def knn(x: np.ndarray, y: np.ndarray, k: int,
        dist_fn: Callable[[np.ndarray, np.ndarray], float] = euc, x_space: Optional[np.ndarray] = None,
        weight_fn: Optional[Callable[[np.ndarray], np.ndarray]] = majority,
        verbose: bool = True) -> np.ndarray:
    """



    :param x: Feature matrix
    :param y: Class of instance
    :param k: Number of neighbour
    :param dist_fn: Distance calculation function
    :param x_space: Feature matrix to be used for prediction. Use None to use x.
    :param weight_fn: Weight contribution of each instance in decision-making.
    :param verbose: verbosity
    :return: Predicted class of each instance
    """
    x_test = x if x_space is None else x_space

    m1 = x_test.shape[0]
    m2 = x.shape[0]
    dist = np.empty((m1, m2))

    for i in range(m1):
        for j in range(m2):
            dist[i][j] = dist_fn(x_test[i, :], x[j, :])

    y_pred = []

    for i in range(m1):
        idx = np.argsort(dist[i, :])[:k]

        y_pred.append(weight_fn(y[idx], dist[i, idx]))

    return y_pred
