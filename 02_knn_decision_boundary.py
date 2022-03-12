import numpy as np
import streamlit as st
from sklearn.datasets import make_blobs
import pandas as pd
import plotly.express as px

from typing import Callable
from math import sqrt
from collections import Counter
from sklearn.metrics import accuracy_score
from nonparametric import knn, majority, exponential, inverse
from metric import l1, l2, euc


def main(verbose: bool = True):
    st.header("Let's Generata a Blob")

    m = st.slider("Number of samples", 10, 10_000, value=500)
    sd = st.slider("Noise", 0., 10., value=2.)
    n_center = st.slider("Number of Centers", 2, 5, value=4)
    x, y = make_blobs(n_samples=m, centers=n_center, n_features=2, random_state=42, cluster_std=sd)

    df = pd.DataFrame(dict(x1=x[:, 0], x2=x[:, 1], y=y))

    if verbose:
        st.dataframe(df)

    fig = px.scatter(df, x="x1", y="x2", color="y")

    st.plotly_chart(fig, use_container_width=True)

    st.header("Try several parameters to test decision boundary")

    num = 40

    x1 = np.linspace(np.min(x[:, 0]), np.max(x[:, 0]), num)
    x2 = np.linspace(np.min(x[:, 1]), np.max(x[:, 1]), num)

    x1v, x2v = np.meshgrid(x1, x2)

    xspace = np.empty((num * num, 2))
    xspace[:, 0] = np.ravel(x1v)
    xspace[:, 1] = np.ravel(x2v)

    k = st.slider("k", 1, 100, value=3)

    dist_fn_map = {"l1": l1, "l2": l2, "euclidean": euc}
    weight_fn_map = {"majority": majority, "expo": exponential, "inv": inverse}

    dist_fn_str = st.selectbox("Metric", list(dist_fn_map.keys()))
    weight_fn_str = st.selectbox("Weighting Strategy", list(weight_fn_map.keys()))

    y_pred = knn(x, y, k=k, dist_fn=dist_fn_map[dist_fn_str], x_space=xspace, weight_fn=weight_fn_map[weight_fn_str])

    df = pd.DataFrame(dict(x1=xspace[:, 0], x2=xspace[:, 1], y=y_pred))

    fig = px.scatter(df, x="x1", y="x2", color="y")

    st.plotly_chart(fig, use_container_width=True)


if __name__ == '__main__':
    main(verbose=st.sidebar.checkbox("Verbose"))
