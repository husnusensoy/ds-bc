import numpy as np
from tqdm import tqdm, trange
from ..helper import ProbabilityDist, CondProbabilityDist, FrequencyDist, CondFrequencyDist
from typing import List
from itertools import product


class NaiveBayes:
    add_k: int
    prior: List[ProbabilityDist]
    conditional: CondProbabilityDist
    rating_list: List

    def __init__(self, add_k: int = 1):
        self.add_k = add_k

    def fit(self, r: np.ndarray):
        m, n = r.shape
        self.rating_list = list(np.unique(r[~np.isnan(r)]))

        row, col = np.nonzero(~np.isnan(r))

        prior = [FrequencyDist() for _ in range(n)]
        cond = CondFrequencyDist()

        for i, j in tqdm(zip(row, col)):
            prior[j][r[i][j]] += 1

            for i2, j2 in zip(row[(row == i) & (col != j)], col[(row == i) & (col != j)]):
                cond[(j2, r[i2][j2])][(j, r[i][j])] += 1

        self.prior = [ProbabilityDist(freq, self.rating_list, self.add_k) for freq in prior]
        self.conditional = CondProbabilityDist(cond, list(product(range(n), self.rating_list)), self.add_k)

    def predict(self, r: np.array, u: int, top_k: int = 3) -> np.ndarray:
        """

        :param r: Rating matrix
        :param u: User u
        :param top_k: Top k neighbourhood
        :return: Calculated Rating of each item
        """

        _, n = r.shape

        score = np.zeros(n)

        for j in trange(n):
            score[j] = self.predict1(r, u, j, top_k)

        return score

    def predict1(self, r: np.array, u: int, j: int) -> float:
        """

        :param r: Rating matrix
        :param u:  User id
        :param j:  Item id
        :return: rating predicted
        """
        idxs = np.nonzero(~np.isnan(r[u, :]))[0]

        rating, probability = max([(k, self.prior[j][k] * np.array(
            [self.conditional[(idx, r[u, idx])][j, k] for idx in idxs if idx != j]).prod()) for k in
                                   self.rating_list], key=lambda x: x[1])

        return rating
