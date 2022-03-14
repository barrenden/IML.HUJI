import pandas as pd
import plotly.express

from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
pio.templates.default = "simple_white"


def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model
    X = np.random.normal(10, 1, 1000)
    estimator = UnivariateGaussian()
    estimator = estimator.fit(X)
    print(estimator.mu_, estimator.var_)

    # Question 2 - Empirically showing sample mean is consistent
    df = pd.DataFrame(columns=['Sample Size', 'Distance from Real Expectation'])
    for sample_size in range(10, 1001, 10):
        sample = X[:sample_size]
        model = UnivariateGaussian(biased_var=False)
        model = model.fit(sample)
        distance = abs(model.mu_ - 10)
        df = pd.concat([df, pd.DataFrame({'Sample Size': [sample_size],
                                          'Distance from Real Expectation': [distance]})])
    plotly.express.bar(df, x='Sample Size', y='Distance from Real Expectation',
                       title='Distance from Actual Expectation as a Function '
                             'of Sample Size').show()

    # Question 3 - Plotting Empirical PDF of fitted model
    pdf = estimator.pdf(X)
    df = pd.DataFrame(zip(X, pdf), columns=['Sample Value', 'Probability'])
    plotly.express.scatter(df, x='Sample Value', y='Probability',
                           title='PDF of Fitted Model').show()


def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    mu = np.array([0, 0, 4, 0])
    sigma = np.matrix([[1, 0.2, 0, 0.5],
                       [0.2, 2, 0, 0],
                       [0, 0, 1, 0],
                       [0.5, 0, 0, 1]])
    X = np.random.multivariate_normal(mu, sigma, 1000)
    estimator = MultivariateGaussian()
    estimator.fit(X)
    print(estimator.mu_)
    print(estimator.cov_)

    # Question 5 - Likelihood evaluation
    raise NotImplementedError()

    # Question 6 - Maximum likelihood
    raise NotImplementedError()


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
