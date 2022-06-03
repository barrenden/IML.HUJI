from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn import datasets
from IMLearn.metrics import mean_square_error
from IMLearn.utils import split_train_test
from IMLearn.model_selection import cross_validate
from IMLearn.learners.regressors import PolynomialFitting, LinearRegression, \
    RidgeRegression
from sklearn.linear_model import Lasso

from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def f(x):
    """
    function for polynomial fitting model.
    """
    return (x + 3) * (x + 2) * (x + 1) * (x - 1) * (x - 2)


def select_polynomial_degree(n_samples: int = 100, noise: float = 5):
    """
    Simulate data from a polynomial model and use cross-validation to select
     the best fitting degree

    Parameters
    ----------
    n_samples: int, default=100
        Number of samples to generate

    noise: float, default = 5
        Noise level to simulate in responses
    """
    # Question 1 - Generate dataset for model f(x)=(x+3)(x+2)(x+1)(x-1)(x-2)
    # + eps for eps Gaussian noise and split into training- and testing
    # portions
    # x = np.linspace(-1.2, 2, num=n_samples)
    X = pd.DataFrame(np.linspace(-1.2, 2, num=n_samples), columns=["x"])
    y = f(X["x"])
    noisy_y = y + np.random.normal(loc=0, scale=noise, size=n_samples)
    train_X, train_y, test_X, test_y = split_train_test(X, pd.Series(noisy_y),
                                                        2 / 3)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=X["x"], y=y,
                             name="Real Data",
                             mode="markers",
                             marker=dict(color="Blue")))
    fig.add_trace(go.Scatter(x=train_X["x"], y=train_y,
                             name="Train Data",
                             mode="markers",
                             marker=dict(color="Red")))
    fig.add_trace(go.Scatter(x=test_X["x"], y=test_y,
                             name="Test Data",
                             mode="markers",
                             marker=dict(color="Green")))
    fig.update_layout(title="Real, Train and Test Data", showlegend=True)
    fig.show()

    # Question 2 - Perform CV for polynomial fitting with degrees 0,1,...,10
    k_values = list(range(11))
    train_errors = []
    validation_errors = []
    for k in k_values:
        estimator = PolynomialFitting(k)
        errors = cross_validate(estimator, np.array(train_X["x"]),
                                np.array(train_y),
                                mean_square_error)
        train_errors.append(errors[0])
        validation_errors.append(errors[1])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=k_values, y=train_errors,
                             name="Train Errors",
                             mode="markers+lines",
                             marker=dict(color="Blue")))
    fig.add_trace(go.Scatter(x=k_values, y=validation_errors,
                             name="Validation Errors",
                             mode="markers+lines",
                             marker=dict(color="Red")))
    fig.update_layout(title="Train and Validation Errors as a Function"
                            " of Polynomial Degree", showlegend=True)
    fig.update_xaxes(title="Polynomial Degree")
    fig.update_yaxes(title="Error")
    fig.show()

    # Question 3 - Using best value of k,
    # fit a k-degree polynomial model and report test error
    best_k = np.argmin(np.array(validation_errors))
    model = PolynomialFitting(int(best_k))
    model.fit(np.array(train_X["x"]), np.array(train_y))
    test_loss = model.loss(np.array(test_X["x"]), np.array(test_y))
    validation_error = np.min(np.array(validation_errors))
    print(f"Best polynomial degree found in k-folds cross-validation is:"
          f" {best_k}.\n"
          f"Loss on test set was: {round(test_loss, 2)}.\n"
          f"Validation error was: {round(validation_error, 2)}")


def select_regularization_parameter(n_samples: int = 50,
                                    n_evaluations: int = 500):
    """
    Using sklearn's diabetes dataset use cross-validation to select the best
     fitting regularization parameter
    values for Ridge and Lasso regressions

    Parameters
    ----------
    n_samples: int, default=50
        Number of samples to generate

    n_evaluations: int, default = 500
        Number of regularization parameter values to evaluate for each of the
         algorithms
    """
    # Question 6 - Load diabetes dataset and split
    # into training and testing portions
    X, y = datasets.load_diabetes(return_X_y=True)
    train_X = X[:n_samples]
    train_y = y[:n_samples]
    test_X = X[n_samples:]
    test_y = y[n_samples:]

    # Question 7 - Perform CV for different values of the
    # regularization parameter for Ridge and Lasso regressions
    ridge_train = []
    ridge_validation = []
    lasso_train = []
    lasso_validation = []
    possible_lambdas = np.linspace(0, 2.5, num=n_evaluations)
    for lam in possible_lambdas:
        ridge_errors = cross_validate(RidgeRegression(lam), train_X,
                                      train_y, mean_square_error)
        ridge_train.append(ridge_errors[0])
        ridge_validation.append(ridge_errors[1])
        lasso_errors = cross_validate(Lasso(lam), train_X, train_y,
                                      mean_square_error)
        lasso_train.append(lasso_errors[0])
        lasso_validation.append(lasso_errors[1])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=possible_lambdas, y=ridge_train,
                             name=f"Ridge Train",
                             mode="markers",
                             marker=dict(color="Blue")))
    fig.add_trace(go.Scatter(x=possible_lambdas, y=ridge_validation,
                             name=f"Ridge Validation",
                             mode="markers",
                             marker=dict(color="Red")))
    fig.add_trace(go.Scatter(x=possible_lambdas, y=lasso_train,
                             name=f"Lasso Train",
                             mode="markers",
                             marker=dict(color="Green")))
    fig.add_trace(go.Scatter(x=possible_lambdas, y=lasso_validation,
                             name=f"Lasso Validation",
                             mode="markers",
                             marker=dict(color="Yellow")))
    fig.update_layout(title=f"Ridge and Lasso Train and Validation Errors"
                            f" for 0 < lambda < 2.5",
                      showlegend=True)
    fig.update_xaxes(title="lambda")
    fig.update_yaxes(title='MSE')
    fig.show()

    # Question 8 - Compare best Ridge model,
    # best Lasso model and Least Squares model
    best_lambda_ridge = possible_lambdas[np.argmin(np.array(ridge_validation))]
    best_lambda_lasso = possible_lambdas[np.argmin(np.array(lasso_validation))]
    ridge_model = RidgeRegression(float(best_lambda_ridge))
    ridge_model.fit(train_X, train_y)
    ridge_loss = ridge_model.loss(test_X, test_y)
    lasso_model = Lasso(float(best_lambda_lasso))
    lasso_model.fit(train_X, train_y)
    lasso_predictions = lasso_model.predict(test_X)
    lasso_loss = mean_square_error(test_y, lasso_predictions)
    least_squares = LinearRegression()
    least_squares.fit(train_X, train_y)
    least_squares_loss = least_squares.loss(test_X, test_y)
    print(f"Ridge Test Error for Best Lambda ({round(best_lambda_ridge, 2)}):"
          f" {round(ridge_loss, 2)}\n"
          f"Lasso Test Error for Best Lambda ({round(best_lambda_lasso, 2)}):"
          f" {round(lasso_loss, 2)}\n"
          f"Least Squares Test Error: {round(least_squares_loss, 2)}")


if __name__ == '__main__':
    np.random.seed(0)
    select_polynomial_degree()
    select_polynomial_degree(noise=0)
    select_polynomial_degree(1500, noise=10)
    select_regularization_parameter()
