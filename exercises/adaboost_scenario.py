import numpy as np
from typing import Tuple
from IMLearn.metalearners.adaboost import AdaBoost
from IMLearn.learners.classifiers import DecisionStump
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from IMLearn.metrics.loss_functions import accuracy


def generate_data(n: int, noise_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a dataset in R^2 of specified size

    Parameters
    ----------
    n: int
        Number of samples to generate

    noise_ratio: float
        Ratio of labels to invert

    Returns
    -------
    X: np.ndarray of shape (n_samples,2)
        Design matrix of samples

    y: np.ndarray of shape (n_samples,)
        Labels of samples
    """
    '''
    generate samples X with shape: (num_samples, 2) and labels y with shape (num_samples).
    num_samples: the number of samples to generate
    noise_ratio: invert the label for this ratio of the samples
    '''
    X, y = np.random.rand(n, 2) * 2 - 1, np.ones(n)
    y[np.sum(X ** 2, axis=1) < 0.5 ** 2] = -1
    y[np.random.choice(n, int(noise_ratio * n))] *= -1
    return X, y


def fit_and_evaluate_adaboost(noise, n_learners=250, train_size=5000, test_size=500):
    (train_X, train_y), (test_X, test_y) = generate_data(train_size, noise), generate_data(test_size, noise)

    # Question 1: Train- and test errors of AdaBoost in noiseless case
    adaboost_model = AdaBoost(DecisionStump, n_learners)
    adaboost_model.fit(train_X, train_y)
    train_errors = np.zeros(n_learners)
    test_errors = np.zeros(n_learners)
    num_learners = np.array(range(1, n_learners + 1))
    for i in num_learners:
        train_errors[i - 1] = adaboost_model.partial_loss(train_X, train_y, i)
        test_errors[i - 1] = adaboost_model.partial_loss(test_X, test_y, i)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=num_learners, y=train_errors, name="Train Errors", mode="lines"))
    fig.add_trace(go.Scatter(x=num_learners, y=test_errors, name="Test Errors", mode="lines"))
    fig.update_layout(title="Error as Function of Number of Learners")
    fig.update_xaxes(title="Number of Learners")
    fig.update_yaxes(title="Error")
    fig.show()

    # Question 2: Plotting decision surfaces
    T = [5, 50, 100, 250]
    lims = np.array([np.r_[train_X, test_X].min(axis=0), np.r_[train_X, test_X].max(axis=0)]).T + np.array([-.1, .1])
    fig = make_subplots(rows=2, cols=2,
                        subplot_titles=[f"Decision Surface for {t} Iterations" for t in T],
                        horizontal_spacing=0.01, vertical_spacing=.03)
    symbols = np.array(["", "circle", "x"])
    best_error = np.inf
    best_predictions = None
    best_num_iterations = 0
    for i, num_iterations in enumerate(T):
        func = lambda x: adaboost_model.partial_predict(x, num_iterations)
        fig.add_traces([decision_surface(func, lims[0],
                                         lims[1], showscale=False),
                        go.Scatter(x=test_X[:, 0], y=test_X[:, 1],
                                   mode="markers",
                                   showlegend=False,
                                   marker=dict(color=test_y.astype(int),
                                               symbol=symbols[test_y.astype(int)],
                                               colorscale=[custom[0],
                                                           custom[-1]],
                                               line=dict(color="black",
                                                         width=1)))],
                       rows=(i // 2) + 1, cols=(i % 2) + 1)
        fig.update_layout(margin=dict(t=100)) \
            .update_xaxes(visible=False).update_yaxes(visible=False)
        cur_error = adaboost_model.partial_loss(test_X, test_y, num_iterations)
        if cur_error < best_error:
            best_error = cur_error
            best_predictions = adaboost_model.partial_predict(test_X, num_iterations)
            best_num_iterations = num_iterations
    fig.show()

    # Question 3: Decision surface of best performing ensemble
    fig = go.Figure()
    func = lambda x: adaboost_model.partial_predict(x, best_num_iterations)
    fig.add_traces([decision_surface(func, lims[0],
                                     lims[1], showscale=False),
                    go.Scatter(x=test_X[:, 0], y=test_X[:, 1],
                               mode="markers",
                               showlegend=False,
                               marker=dict(color=test_y.astype(int),
                                           symbol=symbols[test_y.astype(int)],
                                           colorscale=[custom[0],
                                                       custom[-1]],
                                           line=dict(color="black",
                                                     width=1)))])

    fig.update_layout(title=f"Best Ensemble\n Size: {best_num_iterations},"
                            f" Accuracy: {accuracy(test_y, best_predictions)}")
    fig.show()

    # Question 4: Decision surface with weighted samples
    fig = go.Figure()
    normalized_D = adaboost_model.D_ / np.max(adaboost_model.D_) * 5
    fig.add_traces([decision_surface(adaboost_model.predict, lims[0],
                                     lims[1], showscale=False),
                    go.Scatter(x=train_X[:, 0], y=train_X[:, 1],
                               mode="markers",
                               showlegend=False,
                               marker=dict(color=train_y.astype(int),
                                           symbol=symbols[train_y.astype(int)],
                                           colorscale=[custom[0],
                                                       custom[-1]],
                                           line=dict(color="black",
                                                     width=1),
                                           size=normalized_D))])
    fig.update_layout(title=f"Train Samples Proportional to their Weight")
    fig.show()

if __name__ == '__main__':
    np.random.seed(0)
    fit_and_evaluate_adaboost(0)
    fit_and_evaluate_adaboost(0.4)