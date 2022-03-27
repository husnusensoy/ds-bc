import numpy as np
from reco.collob import UserBased
from rich.console import Console

cons = Console()

r = np.array([[7, 6, 7, 4, 5, 4],
              [6, 7, np.nan, 4, 3, 4],
              [np.nan, 3, 3, 1, 1, np.nan],
              [1, 2, 3, 3, 3, 4],
              [1, np.nan, 1, 2, 3, 3]])

user = UserBased(True, beta=3,idf=True)

sim = user.fit(r)

cons.print(sim)

for u in range(r.shape[0]):
    u0 = user.predict(r, u, 3)

    cons.print(u0)
