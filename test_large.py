import pandas as pd
from reco.collob import UserBased
from rich.console import Console

cons = Console()

df = pd.read_csv('https://files.grouplens.org/datasets/movielens/ml-100k/u.data', delimiter=r'\t',
                 names=['user_id', 'item_id', 'rating', 'timestamp'])

r = df.pivot(index='user_id', columns='item_id', values='rating').values

user = UserBased(beta=3, idf=True)

sim = user.fit(r)

cons.print(sim)

u0 = user.predict(r, 0, 3)

cons.print(u0)
