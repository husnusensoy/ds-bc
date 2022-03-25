import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from collections import Counter
import plotly.figure_factory as ff

from scipy.stats import poisson
from sklearn.metrics import confusion_matrix


def grouping_fn(math: float) -> int:
    if math <= 49:
        return 0
    elif 49 < math <= 59:
        return 1
    elif 59 < math:
        return 2


@st.cache
def get_data():
    df = pd.read_csv("https://stats.idre.ucla.edu/stat/data/poisson_sim.csv")

    df["math_group"] = df.math.apply(grouping_fn)

    return df


from typing import Union


def error(y: np.ndarray, y_pred: Union[float, np.ndarray]) -> float:
    return np.sqrt(((y - y_pred) ** 2).mean())


def accuracy(y: np.ndarray, y_pred: Union[float, np.ndarray]) -> float:
    if type(y_pred) == int:
        return np.sum(y == y_pred) / len(y)
    else:
        return np.count_nonzero(y == y_pred) / len(y)


def error_log(y: np.ndarray, y_pred: Union[float, np.ndarray], logger) -> float:
    mean_square_error = error(df.num_awards.values, y_pred)
    acc = accuracy(df.num_awards.values, y_pred)

    logger(f"Average square error: {mean_square_error:.4f}")
    logger(f"Accuracy: {acc:.4f}")


st.header("Dataset")
df = get_data()
st.dataframe(df.head())
# df.to_csv(index=False)

st.header("Model 1")
st.write(
    "Let's start with a baseline model by setting mode of `num_awards` as the most popular (mode choice which is 0"
)

# mode = Counter(df.num_awards)
# print(mode.most_common(1))


error_log(df.num_awards.values, 0, st.write)

st.header("Model 2")
st.write(
    "Assume that `num_awards` follows a Poisson distribution, given the data using MLE"
)

st.latex(r"f(k; \lambda)=\frac{\lambda^k e^{-\lambda}}{k!}")

st.latex("\lambda_{MLE} =" + f"{df.num_awards.mean()}")

probability = pd.DataFrame(dict(num_awards=range(10), p=[poisson.pmf(k, df.num_awards.mean()) for k in range(10)]))

st.write("At this time 1 is the single value to choose")

# print(df.num_awards.mean())
st.dataframe(probability)

error_log(df.num_awards.values, 1, st.write)

st.subheader("Confusion Matrix")
st.write(
    confusion_matrix(
        df.num_awards.values, np.full(len(df.num_awards.values), fill_value=1)
    )
)

st.header("Model 3")
st.write("Let's model each program separately as a Poisson process by following Bayes Rule")

st.latex(r"P(n_{awards}=k, P=p) =  P(P=p) \times  P(n_{awards}=k|P=p)")

st.subheader("Estimate for program prior")
priors = df.groupby("prog")[["id"]].count()

priors = (
    (priors / priors.sum())
        .reset_index()
        .set_index("prog")
        .rename(columns={"id": "p"})
        .to_dict("index")
)

# priors = priors.copy().reset_index()

st.write(priors)

st.subheader(r"Estimate for Poisson $\lambda$")
st.markdown(
    "Average number of `num_awards`is the Maximum Likelihood Estimate (MLE) for Poisson distribution parameter per program"
)
poisson_lambda = (
    df.groupby(["prog"])
        .mean()["num_awards"]
        .reset_index()
        .set_index("prog")
        .rename(columns={"num_awards": "lam"})
        .to_dict("index")
)

st.write(poisson_lambda)

y_pred = []
for rec in df.itertuples():
    y_pred.append(
        max(
            (
                (
                    k,
                    priors[rec.prog]["p"]
                    * poisson.pmf(k, poisson_lambda[rec.prog]["lam"]),
                )
                for k in range(10)
            ),
            key=lambda x: (x[1], x[0]),
        )[0]
    )

# st.write(y_pred)

error_log(df.num_awards.values, y_pred, st.write)

st.subheader("Confusion Matrix")

st.write(confusion_matrix(df.num_awards.values, y_pred))

st.header("Model 4")

st.write("Let's build a one final model by adding math score into equation by using Naive Bayesian")
st.latex(
    r"P(n_{awards}=k, P=p, M=m) = \\"
    r"P(n_{awards}=k) \times  \\ P(p=p|n_{awards}=k)\times \\ P(M=m|n_{awards}=k)"
)

st.markdown(
    "Note that we choose $p$ & $m$ to be conditionally independent given $n_{awards}$"
)

st.subheader("Prior parameter")
lam = df.mean()['num_awards']

st.markdown("MLE estimate for Poisson $P(n_{awards})$ is" + f" {lam}")


st.subheader("Conditional math distribution parameters")

d = df.groupby(['math', 'num_awards']).size().unstack([1], fill_value=0).reset_index()

full = pd.DataFrame(dict(math=range(100)))
math = full.merge(d, on="math", how="left").fillna(0).set_index("math")

st.markdown(r"#### Counts on math  $\times num_{awards}$ matrix")
st.write(math)
math = math + 1
st.markdown(r"#### Add 1 smoothed probability estimates")
st.write("In order to prevent 0 probability for some math scores use +1 smoothing (better ideas are possible)")
math = math / math.sum(axis=0)

st.dataframe(math)

st.subheader("Conditional program distribution parameters")
d = df.groupby(['prog', 'num_awards']).size().unstack([1], fill_value=0).reset_index()

prog = pd.DataFrame(dict(prog=range(1, 4)))
prog = prog.merge(d, on="prog", how="left").fillna(0).set_index("prog") + 1

prog = prog / prog.sum(axis=0)

st.dataframe(prog)

y_pred = []
for rec in df.itertuples():
    k, p = max([(k, poisson.pmf(k=k, mu=lam) * math.loc[rec.math][k] * prog.loc[rec.prog][k] ) for k in range(7)], key=lambda x: x[1])

    y_pred.append(k)

error_log(df.num_awards, y_pred, st.write)
st.write(
    confusion_matrix(
        df.num_awards.values, y_pred
    )
)