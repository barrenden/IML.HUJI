from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression

from typing import NoReturn, Tuple
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
pio.templates.default = "simple_white"


def load_data(filename: str) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load house prices dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """
    df = pd.read_csv(filename)
    df = df.fillna(0)
    # removing sample errors
    df = df[(df["price"] > 0) & (df["bedrooms"] > 0) & (df["bathrooms"] > 0)
            & (df["floors"] > 0) &
            (df["sqft_lot"] > 0) & (df["sqft_living"]) > 0 &
            (df["date"] != "0") & (df["yr_built"] > 0)]

    # dropping irrelevant columns
    df.drop(["lat", "long", "id"], axis=1, inplace=True)

    # extracting months from dates
    df["date"] = pd.to_datetime(df["date"])
    df["date"] = df["date"].dt.month
    df.rename(columns={"date": "month"}, inplace=True)

    # calculating yard size instead of whole lot size
    df["sqft_yard"] = df["sqft_lot"] - df["sqft_living"]
    df["sqft_yard15"] = df["sqft_living15"] - df["sqft_lot15"]
    df.drop(["sqft_lot", "sqft_lot15"], axis=1, inplace=True)

    # replacing renovation year with max between renovation year and year built
    df.loc[df["yr_renovated"] == 0, "yr_renovated"] = df["yr_built"]
    df["yr_renovated"] = df[["yr_built", "yr_renovated"]].max(axis=1)

    # treating zipcodes as categorical columns
    df = pd.concat([df, pd.get_dummies(df["zipcode"])], axis=1)

    return df.loc[:, df.columns != "price"], df["price"]


def feature_evaluation(X: pd.DataFrame, y: pd.Series, output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """
    raise NotImplementedError()


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of housing prices dataset
    raise NotImplementedError()

    # Question 2 - Feature evaluation with respect to response
    raise NotImplementedError()

    # Question 3 - Split samples into training- and testing sets.
    raise NotImplementedError()

    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)
    raise NotImplementedError()
