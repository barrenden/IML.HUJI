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
    #TODO find out why this result is weird
    df = pd.DataFrame(columns=['sample_size', 'distance'])
    for sample_size in range(10, 1001, 10):
        sample = np.random.choice(X, size=sample_size)
        model = UnivariateGaussian(biased_var=False)
        model = model.fit(sample)
        distance = abs(model.mu_ - 10)
        df = pd.concat([df, pd.DataFrame({'sample_size': [sample_size],
                                          'distance': [distance]})])
    plotly.express.bar(df, x='sample_size', y='distance',
                       title='distance from actual expectation as a function '
                             'of sample size').show()

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
