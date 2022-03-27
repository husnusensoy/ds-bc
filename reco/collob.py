import numpy as np
from tqdm import trange


class UserBased:
    mu: np.ndarray
    sim: np.ndarray

    def __init__(self, zero_mean: bool = True, beta: int = 1, idf: bool = False, verbosity: int = 0):
        """

        :param zero_mean:
        :param beta: Discounting parameter
        :param idf: Enable inverse document frequency management
        """
        self.zero_mean = zero_mean
        self.beta = beta
        self.idf = idf
        self.verbosity = verbosity

    def fit(self, r: np.ndarray):
        m, n = r.shape
        if self.zero_mean:
            self.mu = np.nanmean(r, axis=1)
        else:
            self.mu = np.zeros(m)

        self.sim = np.zeros((m, m))

        if self.idf:
            idf = np.log(1 + m / (~np.isnan(r)).sum(axis=0))
        else:
            idf = np.ones(n)

        if self.verbosity > 0:
            print(idf)

        for i in trange(m):
            for j in range(m):
                mask = ~np.isnan(r[i, :]) & ~np.isnan(r[j, :])

                si = r[i, mask] - self.mu[i]
                sj = r[j, mask] - self.mu[j]

                self.sim[i][j] = (si * sj * idf[mask]).sum() / (
                        np.sqrt((idf[mask] * (si ** 2)).sum()) * np.sqrt((idf[mask] * (sj ** 2)).sum()))

                total_intersection = mask.sum()

                self.sim[i][j] *= min(total_intersection, self.beta) / self.beta

        return self.sim

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

    def predict1(self, r: np.array, u: int, j: int, top_k: int = 3) -> float:
        _, n = r.shape

        users_rated_j = np.nonzero(~np.isnan(r[:, j]))[0]

        topk_users = users_rated_j[self.sim[u, users_rated_j].argsort()[::-1][:top_k]]

        mean_centered_topk_user_rate = r[topk_users, j] - self.mu[topk_users]

        w = self.sim[u, topk_users]

        return np.dot(mean_centered_topk_user_rate, w) / np.abs(w).sum() + self.mu[u]
