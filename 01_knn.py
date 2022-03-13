import numpy as np
import streamlit as st
from sklearn.datasets import make_blobs
import pandas as pd
import plotly.express as px

from typing import Callable, Optional
from math import sqrt
from collections import Counter
from sklearn.metrics import accuracy_score


def mode(x: np.ndarray) -> int:
    return Counter(x).most_common(1)[0][0]


def weighted_mode(clazzes: np.ndarray, weight: np.ndarray) -> int:
    ws = {}
    for c, w in zip(clazzes, weight):
        ws[c] = ws.get(c, 0) + w

    return max(ws.items(), key=lambda t: t[1])[0]


def knn(x: np.ndarray, y: np.ndarray, k: int,
        dist_fn: Callable[[np.ndarray, np.ndarray], float], x_space: Optional[np.ndarray] = None,
        weight_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        verbose: bool = True) -> np.ndarray:
    """



    :param x: Feature matrix
    :param y: Class of instance
    :param k: Number of neighbour
    :param dist_fn: Distance calculation function
    :param x_space: Feature matrix to be used for prediction. Use None to use x. For future use
    :param weight_fn: Weight contribution of each instance in decision-making.  For future use
    :param verbose: verbosity
    :return: Predicted class of each instance
    """
    m = x.shape[0]
    dist = np.empty((m, m))

    for i in range(m):
        for j in range(m):
            if i < j:
                dist[i][j] = dist_fn(x[i, :], x[j, :])
            elif i > j:
                dist[i][j] = dist[j][i]
            else:
                dist[i][j] = 0

    if verbose:
        st.write(dist)

    y_pred = []

    for i in range(m):
        idx = np.argsort(dist[i, :])

        y_pred.append(weighted_mode(y[idx[1:k + 1]], np.exp(-dist[i, 1:k + 1])))

    if verbose:
        st.subheader("Tracing for element 0")
        idx0 = np.argsort(dist[0, :])
        st.write(idx0)
        st.write(idx0[1:k + 1])
        st.write(y[idx0[1:k + 1]])
        st.write(mode(y[idx0[1:k + 1]]))

        st.write(weighted_mode(y[idx0[1:k + 1]], np.exp(-dist[0, 1:k + 1])))

    return y_pred


# Callable[[np.ndarray, np.ndarray], float]
def euc(x1: np.ndarray, x2: np.ndarray) -> float:
    return sqrt(((x1 - x2) ** 2).sum())


# l2
# euclidean
# norm-2
def euc2(x1: np.ndarray, x2: np.ndarray) -> float:
    return np.linalg.norm(x1 - x2)


def l1(x1: np.ndarray, x2: np.ndarray) -> float:
    return np.sum(np.abs(x1 - x2))


def main(verbose: bool = True):
    st.header("Let's Generate a Blob")

    m = st.slider("Number of samples", 10, 10_000, value=500)
    sd = st.slider("Noise", 0., 10., value=2.)
    n_center = st.slider("Number of Centers", 2, 5, value=4)
    x, y = make_blobs(n_samples=m, centers=n_center, n_features=2, random_state=42, cluster_std=sd)

    df = pd.DataFrame(dict(x1=x[:, 0], x2=x[:, 1], y=y))

    if verbose:
        st.dataframe(df)

    fig = px.scatter(df, x="x1", y="x2", color="y")

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Algorithm")
    st.markdown("""
    * A new instance is similar to *k* **nearest** neighbour
    * Strategies
      * Majority Voting
      * **Weighted** Voting
        * Note that majority voting is a spacial case for weighted voting.
    """)

    st.write("Play with slider to get different accuracy values")
    y_pred = knn(x, y, k=st.slider("Neighbour Count", 1, 5, value=3), dist_fn=euc2, verbose=verbose)

    if verbose:
        st.write(y_pred)
        st.write(y)

    st.write(f"Accuracy: {accuracy_score(y, y_pred):.4f}")

    if st.checkbox("Search for k"):
        st.subheader("Search for best k")
        st.write("Generate accuracy values for different k(s) and plot")
        max_k = st.slider("Max k", 1, 20, value=5)
        accuracy = search_k(max_k, x, y)

        st.dataframe(accuracy)

        fig = px.line(accuracy, x="k", y="accuracy")

        st.plotly_chart(fig, use_container_width=True)


@st.cache(suppress_st_warning=True)
def search_k(max_k: int, x: np.ndarray, y: np.ndarray) -> pd.DataFrame:
    acc = []
    my_bar = st.progress(0)
    for _k in range(1, max_k + 1):
        y_pred = knn(x, y, k=_k, dist_fn=l1, verbose=False)

        acc.append(accuracy_score(y, y_pred))

        my_bar.progress(_k / max_k)

    accuracy = pd.DataFrame(dict(k=list(range(1, max_k + 1)), accuracy=acc))
    return accuracy


def counter_demo():
    mock = [4, 3, 3, 2, 2, 2, 1, 1, 1, 1]
    c = Counter(mock)
    st.write(c.most_common(2)[0][0])


if __name__ == '__main__':
    v = st.sidebar.checkbox("Verbose")
    main(verbose=v)

    if v:
        counter_demo()
