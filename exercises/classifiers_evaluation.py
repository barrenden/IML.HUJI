from IMLearn.learners.classifiers import Perceptron, LDA, GaussianNaiveBayes
from typing import Tuple
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from math import atan2, pi


def load_dataset(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load dataset for comparing the Gaussian Naive Bayes and LDA classifiers. File is assumed to be an
    ndarray of shape (n_samples, 3) where the first 2 columns represent features and the third column the class

    Parameters
    ----------
    filename: str
        Path to .npy data file

    Returns
    -------
    X: ndarray of shape (n_samples, 2)
        Design matrix to be used

    y: ndarray of shape (n_samples,)
        Class vector specifying for each sample its class

    """
    data = np.load(filename)
    return data[:, :2], data[:, 2].astype(int)


def run_perceptron():
    """
    Fit and plot fit progression of the Perceptron algorithm over both the linearly separable and inseparable datasets

    Create a line plot that shows the perceptron algorithm's training loss values (y-axis)
    as a function of the training iterations (x-axis).
    """
    for n, f in [("Linearly Separable", "linearly_separable.npy"),
                 ("Linearly Inseparable", "linearly_inseparable.npy")]:
        # Load dataset
        X, y = load_dataset(f"../datasets/{f}")

        # Fit Perceptron and record loss in each fit iteration
        losses = []

        def callback_function(perceptron, sample, response):
            losses.append(perceptron.loss(X, y))

        Perceptron(callback=callback_function).fit(X, y)

        # Plot figure of loss as function of fitting iteration
        df = pd.DataFrame(zip(list(range(len(losses))), losses),
                          columns=["Iteration Number", "Loss"])
        px.line(df, x="Iteration Number", y="Loss", title=n).show()


def get_ellipse(mu: np.ndarray, cov: np.ndarray):
    """
    Draw an ellipse centered at given location and according to specified covariance matrix

    Parameters
    ----------
    mu : ndarray of shape (2,)
        Center of ellipse

    cov: ndarray of shape (2,2)
        Covariance of Gaussian

    Returns
    -------
        scatter: A plotly trace object of the ellipse
    """
    l1, l2 = tuple(np.linalg.eigvalsh(cov)[::-1])
    theta = atan2(l1 - cov[0, 0], cov[0, 1]) if cov[0, 1] != 0 else (np.pi / 2 if cov[0, 0] < cov[1, 1] else 0)
    t = np.linspace(0, 2 * pi, 100)
    xs = (l1 * np.cos(theta) * np.cos(t)) - (l2 * np.sin(theta) * np.sin(t))
    ys = (l1 * np.sin(theta) * np.cos(t)) + (l2 * np.cos(theta) * np.sin(t))

    return go.Scatter(x=mu[0] + xs, y=mu[1] + ys, mode="lines", marker_color="black")


def compare_gaussian_classifiers():
    """
    Fit both Gaussian Naive Bayes and LDA classifiers on both gaussians1 and gaussians2 datasets
    """
    for f in ["gaussian1.npy", "gaussian2.npy"]:
        # Load dataset
        X, y = load_dataset(f"../datasets/{f}")

        # Fit models and predict over training set
        lda_model = LDA()
        lda_model.fit(X, y)
        lda_prediction = lda_model.predict(X)
        gaussian_naive_model = GaussianNaiveBayes()
        gaussian_naive_model.fit(X, y)
        gaussian_naive_prediction = gaussian_naive_model.predict(X)

        # Plot a figure with two suplots, showing the Gaussian Naive Bayes predictions on the left and LDA predictions
        # on the right. Plot title should specify dataset used and subplot titles should specify algorithm and accuracy
        # Create subplots
        from IMLearn.metrics import accuracy
        lda_accuracy = accuracy(y, lda_prediction)
        gaussian_naive_accuracy = accuracy(y, gaussian_naive_prediction)
        fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.01,
                            subplot_titles=[f"LDA Accuracy: {lda_accuracy}",
                                            f"Gaussian Naive Bayes Accuracy: {gaussian_naive_accuracy}"])

        # Add traces for data-points setting symbols and colors
        limits = np.array([X.min(axis=0) - 0.5, X.max(axis=0) + 0.5]).T
        symbols = np.array(["circle", "triangle-up", 'square'])
        for index, model in enumerate([lda_model, gaussian_naive_model]):
            fig.add_trace(decision_surface(model.predict, limits[0], limits[1],
                                           showscale=False), row=1, col=index + 1)
            graph = go.Scatter(x=X[:, 0], y=X[:, 1], mode="markers",
                               marker=dict(color=y, symbol=symbols[y],
                                           colorscale=[custom[0], custom[-1]],
                                           line=dict(color='black', width=1)),
                               showlegend=False)
            fig.add_trace(graph, row=1, col=index + 1)

        # Add `X` dots specifying fitted Gaussians' means
        for index, model in enumerate([lda_model, gaussian_naive_model]):
            traces = []
            points = []
            for mu in model.mu_:
                points.append(mu)
            points = np.array(points)
            centers = go.Scatter(x=points[:, 0], y=points[:, 1],
                                 mode='markers',
                                 marker={'color': 'black', 'symbol': 'x'},
                                 showlegend=False)
            traces.append(centers)
            fig.add_traces(traces, rows=1, cols=index+1)

        # Add ellipses depicting the covariances of the fitted Gaussians
        for index, model in enumerate([lda_model, gaussian_naive_model]):
            traces = []
            for class_index, mu in enumerate(model.mu_):
                if isinstance(model, LDA):
                    cov = model.cov_
                else:
                    cov = np.diag(model.vars_[class_index])
                ellipse = get_ellipse(mu, cov)
                ellipse.showlegend = False
                traces.append(ellipse)
            fig.add_traces(traces, rows=1, cols=index + 1)
        fig.update_layout(title=f"Dataset: {f.split('.')[0]}",
                          margin=dict(t=100))
        fig.show()

if __name__ == '__main__':
    np.random.seed(0)
    run_perceptron()
    compare_gaussian_classifiers()
