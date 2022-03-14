import numpy as np
import streamlit as st
from sklearn.datasets import make_regression
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from typing import Callable
from math import sqrt
from collections import Counter
from sklearn.metrics import accuracy_score
import plotly.graph_objects as go


def mode(x: np.ndarray) -> int:
    return Counter(x).most_common(1)[0][0]


def weighted_mode(clazzes: np.ndarray, weight: np.ndarray) -> int:
    ws = {}
    for c, w in zip(clazzes, weight):
        ws[c] = ws.get(c, 0) + w

    return max(ws.items(), key=lambda t: t[1])[0]


def knn(x: np.ndarray, y: np.ndarray, xspace: np.ndarray, k: int, dist_fn: Callable[[np.ndarray, np.ndarray], float],
        verbose: bool = True) -> np.ndarray:
    """


    :param x: Feature matrix
    :param y: Class of instance
    :param xspace: Data grid of space
    :param k: Number of neighbour
    :param dist_fn: Distance calculation function
    :param verbose: verbosity
    :return: Predicted class of each instance
    """
    m1 = xspace.shape[0]
    st.write(m1)
    m2 = x.shape[0]
    dist = np.empty((m1, m2))

    for i in range(m1):
        for j in range(m2):
            dist[i][j] = dist_fn(xspace[i, :], x[j, :])

    if verbose:
        st.write(dist)

    y_pred = []

    for i in range(m1):
        idx = np.argsort(dist[i, :])

        y_pred.append(mode(y[idx[:k]]))

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


def regression_models(verbose: True):
    st.header("Let's Generate a Blob")
    n = st.slider("Number of Samples", 10, 1000, value=100, help="Number of samples generated")
    x, y = make_reg(n, verbose)

    m1, m2, m3, m4 = "Model 1 (Bias only Model)", "Model 2 (Least Square)", "Model 3 (L2 Regularized/Ridge)", "Model 4 (L1 Regularized/Lasso)"
    section = st.sidebar.selectbox("Linear Regression",
                                   [m1, m2, m3, m4])

    if section == m1:
        st.subheader("Model 1")
        st.markdown(r"""
        Our first model has only 1 parameter $\beta_0$. Model can be defined as 
        """)
        st.latex(r"y = \beta_0")

        st.markdown(r"Given that we use quadratic loss our loss function can be written as ")
        st.latex(r"L(\beta_0) = \sum^{N}_{i=1}{(y_i - \beta_0)^2}")
        st.markdown(
            r"Given that $L$ is convex wrt $\beta_0$, $\beta_0^{*}$ to minimize L can be found by setting derivative to $0$")
        st.latex(
            r"\frac{dL}{d\beta_0} =  -2\sum^{N}_{i=1}{(y_i - \beta_0)} = 0 \Rightarrow \sum^{N}_{i=1}{y_i} = \sum^{N}_{i=1}{\beta_0}")
        st.latex(r"\Rightarrow \beta_0^{*} = \frac{1}{N}\sum^{N}_{i=1}{y_i}")

        st.markdown(r"In other words $\beta_0^{*}$ is simply the average of $y$ values,")
        st.markdown(f"which is {np.mean(y):.4f} for the given dataset")
        # if st.checkbox("Convexity of different loss functions"):
        #
        #     loss0, loss, loss2, loss3 = [], [], [], []
        #
        #     for a in np.linspace(-100, 100, 100):
        #         loss0.append(np.abs(y - a).sum())
        #         loss.append(np.power((y - a), 2).sum())
        #         loss2.append(np.power((y - a), 2).mean())
        #         loss3.append(np.power((y - a), 4).mean())
        #
        #     l0 = pd.DataFrame({"a": np.linspace(-100, 100, 100), "loss": loss0, "Type": "Total Absolute Error"})
        #     l1 = pd.DataFrame({"a": np.linspace(-100, 100, 100), "loss": loss, "Type": "Total Square Error"})
        #     l2 = pd.DataFrame({"a": np.linspace(-100, 100, 100), "loss": loss2, "Type": "Mean Square Error"})
        #     l3 = pd.DataFrame({"a": np.linspace(-100, 100, 100), "loss": loss3, "Type": "Total Power 4 Error"})
        #
        #     l = pd.concat([l0, l1, l2, l3])
        #
        #     if verbose:
        #         st.dataframe(l)
        #
        #     st.markdown("Click on any convex loss **Type** to enable/disable it")
        #     fig = px.line(l, x="a", y="loss", color="Type")
        #
        #     st.plotly_chart(fig, use_container_width=True)
    elif section == m2:

        st.subheader("Model 2")
        st.markdown(r"""
                Our second model has 2 parametes $\beta_0$ and $\beta_1$. Model can be defined as 
                """)
        st.latex(r"y = \beta_0 + \beta_1 x = {\beta} ^T x")

        st.markdown(r"Given that we use quadratic loss our loss function can be written as ")

        st.latex(r"L(\beta_0, \beta_1) = \sum_{i=1}^{N}{(y_i - (\beta_0 + \beta_1 x_i))^2 }")

        st.markdown(
            r"Given that $L$ is convex wrt both $\beta_0$ and $\beta_1$, "
            r"we can use Gradient Descent to find  $\beta_0^{*}$ and $\beta_1^{*}$ by using partial derivatives")
        st.latex(
            r"\frac{\partial L}{\partial \beta_0} =  -2\sum^{N}_{i=1}{(y_i - \beta_0 - \beta_1 x_i )}")
        st.latex(
            r"\frac{\partial L}{\partial \beta_1} =  -2\sum^{N}_{i=1}{(y_i - \beta_0 - \beta_1 x_i )x_i}")

        beta = ls(x[:, 0], y, verbose=verbose)
        st.latex(fr"\beta_0={beta[0]:.4f}, \beta_1={beta[1]:.4f}")

    elif section == m3:

        st.subheader("Model 3 (L2 Regularized)")
        st.markdown(r"""
                        Our third model has also 2 parameters $\beta_0$ and $\beta_1$.
                        """)
        st.latex(r"y = \beta_0 + \beta_1 x = {\beta} ^T x")

        st.markdown(
            r"But this time we have a regularization term in loss function to prevent growing $\beta_0$ and $\beta_1$ values")
        st.latex(
            r"L(\beta_0, \beta_1) = \sum_{i=1}^{N}{(y_i - (\beta_0 + \beta_1 x_i))^2 } + \lambda (\beta_0^2 + \beta_1^2), \lambda > 0")

        st.markdown(
            r"Given that $L$ is convex wrt both $\beta_0$ and $\beta_1$ (Regularization term is convex and sum of two convex functions is still convex), "
            r"we can use Gradient Descent to find  $\beta_0^{*}$ and $\beta_1^{*}$ by using partial derivatives")
        st.latex(
            r"\frac{\partial L}{\partial \beta_0} =  -2\sum^{N}_{i=1}{(y_i - \beta_0 - \beta_1 x_i )} + 2 \lambda \beta_0")
        st.latex(
            r"\frac{\partial L}{\partial \beta_1} =  -2\sum^{N}_{i=1}{(y_i - \beta_0 - \beta_1 x_i )x_i} + 2 \lambda \beta_1")

        lam1 = st.slider("Regularization Multiplier for L2 (lambda)", 0.001, 10., value=0.1)
        beta = ls_l2(x[:, 0], y, lam1)
        st.latex(fr"\beta_0={beta[0]:.4f}, \beta_1={beta[1]:.4f}")

    elif section == m4:
        st.subheader("Model 3 (L1 Regularized)")
        st.markdown(r"""
                                Our fourth model has also 2 parameters $\beta_0$ and $\beta_1$.
                                """)
        st.latex(r"y = \beta_0 + \beta_1 x = {\beta} ^T x")
        st.markdown(
            r"Now we have a L1 regularization term in loss function to prevent growing $\beta_0$ and $\beta_1$ values (Difference between L1 and L2 will be clear later)")

        st.latex(
            r"L(\beta_0, \beta_1) = \sum_{i=1}^{N}{(y_i - (\beta_0 + \beta_1 x_i))^2 } + \lambda (|\beta_0| + |\beta_1|), \lambda > 0")

        st.markdown(
            r"Given that $L$ is convex wrt both $\beta_0$ and $\beta_1$ (Regularization term is almost convex and sum of two convex functions is still convex), "
            r"we can use Gradient Descent to find  $\beta_0^{*}$ and $\beta_1^{*}$ by using partial derivatives")

        st.latex(
            r"\frac{\partial L}{\partial \beta_0} =  -2\sum^{N}_{i=1}{(y_i - \beta_0 - \beta_1 x_i )} + \lambda \frac{\beta_0}{|\beta_0|}")
        st.latex(
            r"\frac{\partial L}{\partial \beta_1} =  -2\sum^{N}_{i=1}{(y_i - \beta_0 - \beta_1 x_i )x_i} + \lambda \frac{\beta_1}{|\beta_1|}")

        lam2 = st.slider("Regularization Multiplier for L1(lambda)", 0.001, 10., value=0.1)
        beta = ls_l1(x[:, 0], y, lam2)

        st.latex(fr"\beta_0={beta[0]:.4f}, \beta_1={beta[1]:.4f}")

        st.header("All Models")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x[:, 0], y=y, mode='markers', name='data points'))
        fig.add_trace(
            go.Scatter(x=x[:, 0], y=np.full(x.shape[0], fill_value=np.mean(y)), mode='lines', name='bias only'))

        beta = ls(x[:, 0], y, verbose=verbose)
        fig.add_trace(go.Scatter(x=x[:, 0], y=beta[0] + beta[1] * x[:, 0], mode='lines', name='least square'))
        beta = ls_l2(x[:, 0], y, 100)
        fig.add_trace(
            go.Scatter(x=x[:, 0], y=beta[0] + beta[1] * x[:, 0], mode='lines', name='regression + L2 (Lambda = 100)'))
        beta = ls_l1(x[:, 0], y, 1000)
        fig.add_trace(
            go.Scatter(x=x[:, 0], y=beta[0] + beta[1] * x[:, 0], mode='lines', name='regression + L1  (Lambda = 1000)'))

        st.plotly_chart(fig, use_container_width=True)


def convexity(verbose: bool = False):
    st.header("Let's Generate a Regression Dataset")
    n = st.slider("Number of Samples", 10, 1000, value=100, help="Number of samples generated")
    x, y = make_reg(n, verbose)
    m, n = 50, 50
    b0 = np.linspace(-25, 25, m)
    b1 = np.linspace(-25, 25., n)

    loss = np.empty((m, n))

    l = st.selectbox("Loss", ["Mean Square", "Mean Square + L2", "Mean Square + L1"])

    if l == "Mean Square":
        beta = ls(x[:, 0], y, verbose=verbose)
        for i, _b0 in enumerate(b0):
            for j, _b1 in enumerate(b1):
                loss[i][j] = ((y - (_b1 * x + _b0)) ** 2).mean()

    elif l == "Mean Square + L2":
        lam = st.slider("Lambda", 0., 10., value=1.)
        for i, _b0 in enumerate(b0):
            for j, _b1 in enumerate(b1):
                loss[i][j] = np.sqrt(np.power((y - _b1 * x - _b0), 2).mean()) + lam * np.linalg.norm(
                    np.array([_b0, _b1]))
    elif l == "Mean Square + L1":
        lam = st.slider("Lambda", 0., 10., value=1.)
        for i, _b0 in enumerate(b0):
            for j, _b1 in enumerate(b1):
                loss[i][j] = np.sqrt(np.power((y - _b1 * x - _b0), 2).mean()) + lam * np.abs(np.array([_b0, _b1])).sum()

    # FIX: Axis naming

    fig =go.Figure(data=go.Contour(
        z=loss,
        x=b0,
        y=b1
    ))

    # fig.update_layout(height=800)
    st.plotly_chart(fig, use_container_width=True)


def main(verbose: bool = True):
    section = st.sidebar.radio("Section", ["Regression Models", "Convexity", "Polynomial Features"])

    if section == "Regression Models":
        regression_models(verbose)
    elif section == "Polynomial Features":
        st.header("Let's Generate a Non Linear Regression Data")
        n = st.slider("Number of Samples", 10, 1000, value=100, help="Number of samples generated")
        x, y = make_nl_reg(n, verbose)

        st.markdown(f"We have already defined a dependency between $y$ and $x$ in the following form")

        st.latex(r"y = 4x^3 + 3x^2 + 2x + 1 + N(0, \sigma)")

        st.markdown("One can obviously conclude that $y$ is not a linear function of $x$. "
                    "On the other hand, instead of defining problem as an uni-variate regression, we can define in the following form as linear function")

        st.latex(r"y = \beta_3 z_3  +  \beta_2 z_2 + \beta_1 z_1  +\beta_0, z_1 = x, z_2 = x^2, z_3 = x^3")

        st.markdown("Efficient solution of this for requires to take partial derivatives wrt vector $z$ "
                    "as opposed to partial derivative wrt single variable $x$. Wait till next week :wink:")
    else:
        convexity()


def make_reg(n: int = 100, verbose: bool = False):
    rng = np.random.RandomState(0)
    x, y = make_regression(n, 1, random_state=rng,
                           noise=st.slider("Noise", 1., 100., value=10., help="Perturbation around mean"))
    if st.checkbox("Outlier"):
        x = np.concatenate((x, np.array([[-2], [0], [2], [-2], [0], [2]])))
        y = np.concatenate((y, np.array([300, 300, 300, 300, 300, 300])))
    df = pd.DataFrame(dict(x=x[:, 0], y=y))
    if verbose:
        st.dataframe(df)

    if st.checkbox("Plotly OLS fit"):
        fig = px.scatter(df, x="x", y="y", trendline="ols")
    else:
        fig = px.scatter(df, x="x", y="y")
    st.plotly_chart(fig, use_container_width=True)
    return x, y


def make_nl_reg(n: int = 100, verbose: bool = False):
    rng = np.random.RandomState(0)

    x = np.random.randn(n)
    y = 4 * x ** 3 + 3 * x ** 2 + 2 * x + 1 + np.random.normal(
        scale=st.slider("Noise", 0.1, 3., value=.5, help="Perturbation around mean"), size=n)

    df = pd.DataFrame(dict(x=x, y=y))
    if verbose:
        st.dataframe(df)

    fig = px.scatter(df, x="x", y="y")

    st.plotly_chart(fig, use_container_width=True)

    return x, y


def ls_l1(x, y, lam, alpha=0.0001) -> np.ndarray:
    print("starting sgd")
    beta = np.random.random(2)

    for i in range(1000):
        y_pred: np.ndarray = beta[0] + beta[1] * x

        if beta[0] >= 0:
            g_b0 = -2 * (y - y_pred).sum() + lam
        else:
            g_b0 = -2 * (y - y_pred).sum() - lam

        if beta[1] >= 0:
            g_b1 = -2 * (x * (y - y_pred)).sum() + lam
        else:
            g_b1 = -2 * (x * (y - y_pred)).sum() - lam

        print(f"({i}) beta: {beta}, gradient: {g_b0} {g_b1}")

        beta_prev = np.copy(beta)

        beta[0] = beta[0] - alpha * g_b0
        beta[1] = beta[1] - alpha * g_b1

        if np.linalg.norm(beta - beta_prev) < 0.000001:
            print(f"I do early stoping at iteration {i}")
            break

    return beta


def ls_l2(x, y, lam, alpha=0.0001) -> np.ndarray:
    print("starting sgd")
    beta = np.random.random(2)

    for i in range(1000):
        y_pred: np.ndarray = beta[0] + beta[1] * x

        g_b0 = -2 * (y - y_pred).sum() + 2 * lam * beta[0]
        g_b1 = -2 * (x * (y - y_pred)).sum() + 2 * lam * beta[1]

        print(f"({i}) beta: {beta}, gradient: {g_b0} {g_b1}")

        beta_prev = np.copy(beta)

        beta[0] = beta[0] - alpha * g_b0
        beta[1] = beta[1] - alpha * g_b1

        if np.linalg.norm(beta - beta_prev) < 0.000001:
            print(f"I do early stoping at iteration {i}")
            break

    return beta


def ls(x, y, alpha=0.001, verbose=False) -> np.ndarray:
    beta = np.random.random(2)
    if verbose:
        st.write(beta)

    print("starting sgd")
    for i in range(100):
        y_pred: np.ndarray = beta[0] + beta[1] * x

        g_b0 = -2 * (y - y_pred).sum()
        g_b1 = -2 * (x * (y - y_pred)).sum()

        print(f"({i}) beta: {beta}, gradient: {g_b0} {g_b1}")

        beta_prev = np.copy(beta)

        beta[0] = beta[0] - alpha * g_b0
        beta[1] = beta[1] - alpha * g_b1

        if np.linalg.norm(beta - beta_prev) < 0.001:
            print(f"I do early stoping at iteration {i}")
            break

    return beta


if __name__ == '__main__':
    main(verbose=st.sidebar.checkbox("Verbose"))
