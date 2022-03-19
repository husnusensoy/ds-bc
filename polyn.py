import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV


def glm(x: np.ndarray, y: np.ndarray, fig, max_iter=100):
    beta = np.random.rand(x.shape[1])

    print(beta)

    alpha = 0.0001
    my_bar = st.progress(0)
    for it in range(max_iter):

        grad = np.zeros(beta.shape)
        total_err = 0
        for i in range(x.shape[0]):
            y_pred = np.dot(beta, x[i, :])

            err = y[i, 0] - y_pred
            total_err += err ** 2

            grad += -2 * err * x[i, :]

            beta = beta - alpha * grad

        print(f"({it}) beta: {beta}, gradient: {np.linalg.norm(grad)}")
        my_bar.progress((it + 1) / max_iter)

    st.write(f"Final Training Loss: {((x @ beta - y) ** 2).mean():.4f}")

    fig.add_trace(
        go.Scatter(
            x=x[:, 1],
            y=x @ beta,
            mode="lines",
            name="Multivariate Regresssion",
        )
    )

    return np.linalg.norm(grad), beta


def main(verbose: bool = False):
    st.header("Generalized Linear Models")
    st.subheader("Multivariate Model")

    # x = np.linspace(-1, 1, N).reshape((N, 1))

    x = (np.random.rand(160) * 2 - 1).reshape((160, 1))
    x_test = (np.random.rand(40) * 2 - 1).reshape((40, 1))

    # st.write(x)
    # st.write(x_test)

    n = 160
    # x = (np.random.rand(n) * 2 - 1).reshape((n, 1))

    d = st.slider("Max term degree", 1, 10, value=3)

    z = np.zeros((n, d + 1))
    z[:, 0] = 1

    for i in range(d):
        z[:, i + 1] = x[:, 0] ** (i + 1)

    y = (x - 0.1) ** 5 + np.random.normal(scale=st.slider("Noise", 0., 10., value=0.15), size=n)

    fig = px.scatter(pd.DataFrame(dict(x=x[:, 0], y=y[:, 0])), x="x", y="y")

    g, beta = glm(z, y, fig, max_iter=st.slider("Number of GD passes", 100, 10_000, value=100))

    z_test = np.zeros((40, d + 1))
    z_test[:, 0] = 1

    for i in range(d):
        z_test[:, i + 1] = x_test[:, 0] ** (i + 1)

    y_test_pred = z_test @ beta
    y_test = (x_test - 0.1) ** 5

    st.write(f"Final Test Loss: {((y_test_pred - y_test) ** 2).mean():.4f}")

    if g > 1:
        st.write(f"Not converged (Gradient Norm: {g:.4f}) increase number of gd passes")

    st.plotly_chart(fig, use_container_width=True)

    st.write(z)


if __name__ == '__main__':
    main(verbose=False)
