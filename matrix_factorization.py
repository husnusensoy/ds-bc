import numpy as np
import pandas as pd

r = np.array([[7, 6, 7, 4, 5, 4],
              [6, 7, np.nan, 4, 3, 4],
              [np.nan, 3, 3, 1, 1, np.nan],
              [1, 2, 3, 3, 3, 4],
              [1, np.nan, 1, 2, 3, 3]])

d = 2

U = np.random.rand(r.shape[0], d)
V = np.random.rand(d, r.shape[1])

U, V

from tqdm import trange

d = 2

U = np.random.rand(r.shape[0], d)
V = np.random.rand(d, r.shape[1])

lamb = 0.
alpha = 0.001

row, col = np.nonzero(~np.isnan(r))

print(f"% fill on rating matrix: {len(row) / (r.shape[0] * r.shape[1]):.4f}")

# L2 Regularized
with trange(1000) as epochs:
    for _ in epochs:
        total_e = 0
        for i, j in zip(row, col):
            # Prediction of r_ij
            y_pred = np.dot(U[i, :], V[:, j])
            e = r[i][j] - y_pred

            U[i, :] += (2 * e * V[:, j] - 2 * lamb * U[i, :]) * alpha
            V[:, j] += (2 * e * U[i, :] - 2 * lamb * V[:, j]) * alpha

            total_e += e ** 2

        epochs.set_description(f'Total Square Error: {total_e:.2f}')

# L1 Regularized
from tqdm import trange

d = 2

U = np.random.rand(r.shape[0], d)
V = np.random.rand(d, r.shape[1])

lamb = 1.
alpha = 0.001

with trange(1000) as epochs:
    for _ in epochs:
        total_e = 0
        for i, j in zip(row, col):
            # Prediction of r_ij
            y_pred = np.dot(U[i, :], V[:, j])
            e = r[i][j] - y_pred

            U[i, :] += (2 * e * V[:, j] - lamb * U[i, :] * np.sign(U[i, :])) * alpha
            V[:, j] += (2 * e * U[i, :] - lamb * V[:, j] * np.sign(V[:, j])) * alpha

            total_e += e ** 2

        epochs.set_description(f'Total Square Error: {total_e:.2f}')
