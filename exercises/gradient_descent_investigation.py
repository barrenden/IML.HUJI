import numpy as np
import pandas as pd
from typing import Tuple, List, Callable, Type

from IMLearn import BaseModule
from IMLearn.desent_methods import GradientDescent, FixedLR, ExponentialLR
from IMLearn.desent_methods.modules import L1, L2
from IMLearn.learners.classifiers.logistic_regression import LogisticRegression
from IMLearn.utils import split_train_test
from plotly.subplots import make_subplots
from sklearn.metrics import roc_curve, RocCurveDisplay, auc
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from IMLearn.model_selection import cross_validate
from IMLearn.metrics.loss_functions import misclassification_error


def plot_descent_path(module: Type[BaseModule],
                      descent_path: np.ndarray,
                      title: str = "",
                      xrange=(-1.5, 1.5),
                      yrange=(-1.5, 1.5)) -> go.Figure:
    """
    Plot the descent path of the gradient descent algorithm

    Parameters:
    -----------
    module: Type[BaseModule]
        Module type for which descent path is plotted

    descent_path: np.ndarray of shape (n_iterations, 2)
        Set of locations if 2D parameter space being the regularization path

    title: str, default=""
        Setting details to add to plot title

    xrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    yrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    Return:
    -------
    fig: go.Figure
        Plotly figure showing module's value in a grid of [xrange]x[yrange] over which regularization path is shown

    Example:
    --------
    fig = plot_descent_path(IMLearn.desent_methods.modules.L1, np.ndarray([[1,1],[0,0]]))
    fig.show()
    """

    def predict_(w):
        return np.array([module(weights=wi).compute_output() for wi in w])

    from utils import decision_surface
    return go.Figure([decision_surface(predict_, xrange=xrange, yrange=yrange,
                                       density=70, showscale=False),
                      go.Scatter(x=descent_path[:, 0], y=descent_path[:, 1],
                                 mode="markers+lines", marker_color="black")],
                     layout=go.Layout(xaxis=dict(range=xrange),
                                      yaxis=dict(range=yrange),
                                      title=f"GD Descent Path {title}"))


def get_gd_state_recorder_callback() -> Tuple[
    Callable[[], None], List[np.ndarray], List[np.ndarray]]:
    """
    Callback generator for the GradientDescent class, recording the objective's value and parameters at each iteration

    Return:
    -------
    callback: Callable[[], None]
        Callback function to be passed to the GradientDescent class, recoding the objective's value and parameters
        at each iteration of the algorithm

    values: List[np.ndarray]
        Recorded objective values

    weights: List[np.ndarray]
        Recorded parameters
    """
    values_ = []
    weights_ = []

    def callback(**kwargs):
        values_.append(kwargs["val"])
        weights_.append(kwargs["weights"])

    return callback, values_, weights_


def compare_fixed_learning_rates(
        init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
        etas: Tuple[float] = (1, .1, .01, .001)):
    for eta in etas:
        l1 = L1(weights=init)
        l2 = L2(weights=init)
        callback, values, weights = get_gd_state_recorder_callback()
        gd = GradientDescent(learning_rate=FixedLR(eta), callback=callback)
        gd.fit(l1, X=None, y=None)
        fig = plot_descent_path(L1, np.array(weights), title=f"L1, eta={eta}")
        fig.show()
        iteration_numbers = list(range(0, len(values)))
        convergence_fig = go.Figure()
        convergence_fig.update_layout(
            title=f"Convergence Rate for L1 and L2 norm, eta={eta}")
        convergence_fig.update_xaxes(title="Iteration Number")
        convergence_fig.update_yaxes(title="Norm")
        convergence_fig.add_trace(go.Scatter(x=iteration_numbers, y=values,
                                             mode='lines+markers', name="L1"))
        print(f"minimal loss for L1, eta={eta} is {min(values)}")
        callback, values, weights = get_gd_state_recorder_callback()
        gd = GradientDescent(learning_rate=FixedLR(eta), callback=callback)
        gd.fit(l2, X=None, y=None)
        fig = plot_descent_path(L2, np.array(weights), title=f"L2, eta={eta}")
        fig.show()
        iteration_numbers = list(range(0, len(values)))
        convergence_fig.add_trace(go.Scatter(x=iteration_numbers, y=values,
                                             mode='lines+markers', name="L2"))
        convergence_fig.show()
        print(f"minimal loss for L2, eta={eta} is {min(values)}\n\n")


def compare_exponential_decay_rates(
        init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
        eta: float = .1,
        gammas: Tuple[float] = (.9, .95, .99, 1)):
    # Optimize the L1 objective using different decay-rate values of the
    # exponentially decaying learning rate

    # Plot algorithm's convergence for the different values of gamma
    fig = go.Figure()
    minimal_norm = np.inf
    for gamma in gammas:
        l1 = L1(weights=init)
        callback, values, weights = get_gd_state_recorder_callback()
        gd = GradientDescent(learning_rate=ExponentialLR(eta, gamma),
                             callback=callback)
        gd.fit(l1, X=None, y=None)
        if min(values) < minimal_norm:
            minimal_norm = min(values)
        iteration_numbers = list(range(0, len(values)))
        fig.add_trace(
            go.Scatter(x=iteration_numbers, y=values, name=f"gamma={gamma}"))
    fig.update_layout(title="Convergence Rate as a Function of Iteration"
                            " Number for Different Decay Rates")
    fig.update_xaxes(title="Iteration Number")
    fig.update_yaxes(title="Norm")
    fig.show()
    print(f"Lowest L1 norm achieved was {minimal_norm}")

    # Plot descent path for gamma=0.95
    l1 = L1(weights=init)
    l2 = L2(weights=init)
    callback, values, weights = get_gd_state_recorder_callback()
    gd = GradientDescent(learning_rate=ExponentialLR(eta, 0.95),
                         callback=callback)
    gd.fit(l1, X=None, y=None)
    fig = plot_descent_path(L1, np.array(weights), title=f"L1,"
                                                         f" eta={eta},"
                                                         f" gamma=0.95")
    fig.show()
    callback, values, weights = get_gd_state_recorder_callback()
    gd = GradientDescent(learning_rate=ExponentialLR(eta, 0.95),
                         callback=callback)
    gd.fit(l2, X=None, y=None)
    fig = plot_descent_path(L2, np.array(weights), title=f"L2,"
                                                         f" eta={eta},"
                                                         f" gamma=0.95")
    fig.show()


def load_data(path: str = "../datasets/SAheart.data",
              train_portion: float = .8) -> \
        Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Load South-Africa Heart Disease dataset and randomly split into a train- and test portion

    Parameters:
    -----------
    path: str, default= "../datasets/SAheart.data"
        Path to dataset

    train_portion: float, default=0.8
        Portion of dataset to use as a training set

    Return:
    -------
    train_X : DataFrame of shape (ceil(train_proportion * n_samples), n_features)
        Design matrix of train set

    train_y : Series of shape (ceil(train_proportion * n_samples), )
        Responses of training samples

    test_X : DataFrame of shape (floor((1-train_proportion) * n_samples), n_features)
        Design matrix of test set

    test_y : Series of shape (floor((1-train_proportion) * n_samples), )
        Responses of test samples
    """
    df = pd.read_csv(path)
    df.famhist = (df.famhist == 'Present').astype(int)
    return split_train_test(df.drop(['chd', 'row.names'], axis=1), df.chd,
                            train_portion)


def fit_logistic_regression():
    # Load and split SA Heard Disease dataset
    X_train, y_train, X_test, y_test = load_data()

    # Plotting convergence rate of logistic regression over SA heart
    # disease data
    model = LogisticRegression(solver=GradientDescent(), )
    model.fit(np.array(X_train), np.array(y_train))
    probs = model.predict_proba(np.array(X_test))
    fpr, tpr, thresholds = roc_curve(y_test,
                                     probs)  # TODO: change to check all alphas in given range
    roc_auc = auc(fpr, tpr)
    display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc,
                              estimator_name=f'Logistic Regression')
    display.plot()
    plt.show()
    best_alpha = np.argmax(tpr - fpr)
    model = LogisticRegression(solver=GradientDescent(),
                               alpha=best_alpha / 100)
    model.fit(np.array(X_train), np.array(y_train))
    print(f"Using alpha={best_alpha / 100}, test error is"
          f" {model.loss(np.array(X_test), np.array(y_test))}")

    # Fitting l1- and l2-regularized logistic regression models,
    # using cross-validation to specify values
    # of regularization parameter
    # l1
    lambdas = np.linspace(0, 1, 5)
    train_scores, validation_scores = [], []
    for lam in lambdas:
        model = LogisticRegression(solver=GradientDescent(), alpha=0.5,
                                   penalty='l1', lam=lam)
        train_score, validation_score = cross_validate(model,
                                                       np.array(X_train),
                                                       np.array(y_train),
                                                       misclassification_error)
        train_scores.append(train_score)
        validation_scores.append(validation_score)
    best_lambda = np.argmin(validation_scores) / 20
    model = LogisticRegression(solver=GradientDescent(), alpha=0.5,
                               penalty='l1', lam=best_lambda)
    model.fit(np.array(X_train), np.array(y_train))
    print(f"Chosen lambda was {best_lambda}, model error was"
          f" {model.loss(np.array(X_test), np.array(y_test))}")
    # l2
    train_scores, validation_scores = [], []
    for lam in lambdas:
        model = LogisticRegression(solver=GradientDescent(), alpha=0.5,
                                   penalty='l2', lam=lam)
        train_score, validation_score = cross_validate(model,
                                                       np.array(X_train),
                                                       np.array(y_train),
                                                       misclassification_error)
        train_scores.append(train_score)
        validation_scores.append(validation_score)
    best_lambda = np.argmin(validation_scores) / 20
    model = LogisticRegression(solver=GradientDescent(), alpha=0.5,
                               penalty='l2', lam=best_lambda)
    model.fit(np.array(X_train), np.array(y_train))
    print(f"Chosen lambda was {best_lambda}, model error was"
          f" {model.loss(np.array(X_test), np.array(y_test))}")


if __name__ == '__main__':
    np.random.seed(0)
    compare_fixed_learning_rates()
    compare_exponential_decay_rates()
    fit_logistic_regression()
