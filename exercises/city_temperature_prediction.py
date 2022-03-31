import IMLearn.learners.regressors.linear_regression
from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
pio.templates.default = "simple_white"


def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """
    df = pd.read_csv(filename, parse_dates=["Date"])
    df = df.dropna()
    df = df[df["Temp"] >= -12]
    df["DayOfYear"] = df.Date.dt.day_of_year
    return df


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of city temperature dataset
    df = load_data("../datasets/City_Temperature.csv")

    # Question 2 - Exploring data for specific country
    israel_df = df[df.Country == "Israel"]
    israel_df = israel_df.astype({"Year": str})
    px.scatter(israel_df, x="DayOfYear", y="Temp", color="Year",
               title="Temperature as a Function of DayOfYear").show()
    px.bar(israel_df.groupby('Month').agg(std=('Temp', 'std')),
           y="std", title="Standard Deviation of Temperature by Month").show()

    # Question 3 - Exploring differences between countries
    px.line(df.groupby(["Country", "Month"]).agg(std=("Temp", "std"),
                                           mean=("Temp", "mean")).reset_index(),
            x="Month", y="mean", color="Country", error_y="std", error_y_minus=None,
            title="Average Monthly Temperature").show()

    # Question 4 - Fitting model for different values of `k`
    train_x, train_y, test_x, test_y =\
        split_train_test(israel_df["DayOfYear"], israel_df["Temp"])
    losses = []
    for k in range(1, 11):
        model = PolynomialFitting(k)
        model.fit(train_x.to_numpy(), train_y.to_numpy())
        loss = round(model.loss(test_x.to_numpy(), test_y.to_numpy()), 2)
        losses.append(loss)
        print(f"k: {k}, loss: {loss}")
    results = pd.DataFrame(zip(range(1, 11), losses), columns=["Degree", "Loss"])
    px.bar(results, x="Degree", y="Loss", title="Loss as Function of Degree").show()

    # Question 5 - Evaluating fitted model on different countries
    model = PolynomialFitting(5)
    train_x = israel_df["DayOfYear"]
    train_y = israel_df["Temp"]
    model.fit(train_x.to_numpy(), train_y.to_numpy())
    errors = []
    countries = ["The Netherlands", "Jordan", "South Africa"]
    for country in countries:
        test_x = df[df.Country == country].DayOfYear
        test_y = df[df.Country == country].Temp
        errors.append(model.loss(test_x, test_y))
    results = pd.DataFrame(zip(countries, errors), columns=["Country", "Loss"])
    px.bar(results, x="Country", y="Loss", title="Loss as Function of Country").show()
