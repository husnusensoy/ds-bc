import pandas as pd
import numpy as np
from reco.collob import UserBased
from rich.console import Console

cons = Console()

df = pd.read_csv('https://files.grouplens.org/datasets/movielens/ml-100k/u.data', delimiter=r'\t',
                 names=['user_id', 'item_id', 'rating', 'timestamp'])

r = df.pivot(index='user_id', columns='item_id', values='rating').values

irow, jcol = np.where(~np.isnan(r))

cons.print(f"{len(irow)} entries available")

idx = np.random.choice(np.arange(100_000), 1000, replace=False)
test_irow = irow[idx]
test_jcol = jcol[idx]

r_copy = r.copy()

for i in test_irow:
    for j in test_jcol:
        r_copy[i][j] = np.nan

user = UserBased(beta=3, idf=True)

sim = user.fit(r_copy)

# cons.print(sim)

# u0 = user.predict(r, 0, 3)

# cons.print(u0)

err = []
for u, j in zip(test_irow, test_jcol):
    y_pred = user.predict1(r_copy, u, j)
    y = r[u, j]

    err.append((y_pred - y) ** 2)

cons.print(f"RMSE: {np.sqrt(np.nanmean(np.array(err)))}")
