import numpy as np
import pandas as pd
import streamlit as st
from sklearn.datasets import make_classification

import plotly.express as px
from sklearn.linear_model import LogisticRegression


def main():
    X, y = make_classification(n_samples=st.slider("n", 10, 1000, value=100), n_features=2, n_classes=2,
                               n_informative=2,
                               n_redundant=0, n_clusters_per_class=1,
                               random_state=42)

    df = pd.DataFrame(dict(x1=X[:, 0], x2=X[:, 1], y=y))

    fig = px.scatter(df, x="x1", y="x2", color=y)

    st.plotly_chart(fig, use_container_width=True)

    lr = LogisticRegression()
    lr.fit(X, y)

    y_pred = lr.predict(X)

    st.write(y_pred)

    st.write(f"Total estimation error: {sum(y1 != y2 for y1, y2 in zip(y_pred, y)) / len(y)}")

    num = 40

    x1 = np.linspace(np.min(X[:, 0]), np.max(X[:, 0]), num)
    x2 = np.linspace(np.min(X[:, 1]), np.max(X[:, 1]), num)

    x1v, x2v = np.meshgrid(x1, x2)

    xspace = np.empty((num * num, 2))
    xspace[:, 0] = np.ravel(x1v)
    xspace[:, 1] = np.ravel(x2v)

    color_space = lr.predict(xspace)

    df = pd.DataFrame(dict(x1=xspace[:, 0], x2=xspace[:, 1], y=color_space))

    st.dataframe(df)

    fig = px.scatter(df, x="x1", y="x2", color="y")

    st.plotly_chart(fig, use_container_width=True)


if __name__ == '__main__':
    main()
