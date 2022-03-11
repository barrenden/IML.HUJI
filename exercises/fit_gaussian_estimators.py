from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
pio.templates.default = "simple_white"


def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model
    X = np.random.normal(10, 1, 1000)
    estimator = UnivariateGaussian(biased_var=False)
    estimator = estimator.fit(X)
    print(estimator.mu_, estimator.var_)
    raise NotImplementedError()

    # Question 2 - Empirically showing sample mean is consistent
    raise NotImplementedError()

    # Question 3 - Plotting Empirical PDF of fitted model
    raise NotImplementedError()


def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    raise NotImplementedError()

    # Question 5 - Likelihood evaluation
    raise NotImplementedError()

    # Question 6 - Maximum likelihood
    raise NotImplementedError()


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
