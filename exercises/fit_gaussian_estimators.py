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
    estimator = UnivariateGaussian(biased_var=False)
    estimator = estimator.fit(X)
    print(estimator.mu_, estimator.var_)

    # Question 2 - Empirically showing sample mean is consistent
    df = pd.DataFrame(columns=['Sample Size', 'Distance from Real Expectation'])
    increasing_sample = X.copy()
    sample = np.array([])
    for i in range(1, 101):
        sample_to_add = np.random.choice(increasing_sample, 10)
        sample = np.append(sample, sample_to_add, 0)
        model = UnivariateGaussian(biased_var=False)
        model = model.fit(sample)
        distance = abs(model.mu_ - 10)
        sample_size = i * 10
        df = pd.concat([df, pd.DataFrame({'Sample Size': [sample_size],
                                          'Distance from Real Expectation': [distance]})])
    plotly.express.bar(df, x='Sample Size', y='Distance from Real Expectation',
                       title='Distance from Actual Expectation as a Function '
                             'of Sample Size').show()

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
