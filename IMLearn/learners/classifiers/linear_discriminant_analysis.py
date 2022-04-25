from typing import NoReturn
from ...base import BaseEstimator
import numpy as np
from numpy.linalg import det, inv, slogdet


class LDA(BaseEstimator):
    """
    Linear Discriminant Analysis (LDA) classifier

    Attributes
    ----------
    self.classes_ : np.ndarray of shape (n_classes,)
        The different labels classes. To be set in `LDA.fit`

    self.mu_ : np.ndarray of shape (n_classes,n_features)
        The estimated features means for each class. To be set in `LDA.fit`

    self.cov_ : np.ndarray of shape (n_features,n_features)
        The estimated features covariance. To be set in `LDA.fit`

    self._cov_inv : np.ndarray of shape (n_features,n_features)
        The inverse of the estimated features covariance. To be set in `LDA.fit`

    self.pi_: np.ndarray of shape (n_classes)
        The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
    """
    def __init__(self):
        """
        Instantiate an LDA classifier
        """
        super().__init__()
        self.classes_, self.mu_, self.cov_, self._cov_inv, self.pi_ = None, None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits an LDA model.
        Estimates gaussian for each label class - Different mean vector, same covariance
        matrix with dependent features.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        unique, counts = np.unique(y, return_counts=True)
        self.classes_ = unique
        m = X.shape[0]
        self.pi_ = counts / m
        num_classes = unique.shape[0]
        num_features = X.shape[1]
        mu = np.zeros([num_classes, num_features])
        for index, k in enumerate(unique):
            mu[index] = sum(X[np.where(y == k)]) / counts[np.where(unique == k)]
        self.mu_ = mu
        sigma = 0
        for i in range(len(X)):
            sigma += np.outer(X[i] - self.mu_[y[i]], X[i] - self.mu_[y[i]])
        sigma /= m - num_classes
        self.cov_ = sigma
        self._cov_inv = inv(self.cov_)

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """
        class_likelihoods = self.likelihood(X)
        return class_likelihoods.argmax(axis=1)

    def likelihood(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate the likelihood of a given data over the estimated model

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data to calculate its likelihood over the different classes.

        Returns
        -------
        likelihoods : np.ndarray of shape (n_samples, n_classes)
            The likelihood for each sample under each of the classes

        """
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling "
                             "`likelihood` function")
        m = X.shape[0]
        d = X.shape[1]
        constant = (-d / 2) * np.log(2 * np.pi) - 0.5 * slogdet(self.cov_)[0] * slogdet(self.cov_)[1]
        likelihoods = np.zeros([m, self.classes_.shape[0]])
        for sample_index, sample in enumerate(X):
            sample_likelihoods = np.zeros(self.classes_.shape[0])
            for class_index, k in enumerate(self.classes_):
                log_pi_k = np.log(self.pi_[class_index])
                centered_sample = sample - self.mu_[class_index]
                sample_calc = np.matmul(np.transpose(centered_sample),
                                        np.matmul(self._cov_inv, centered_sample))
                sample_likelihoods[class_index] = constant - 0.5 * sample_calc + log_pi_k
            likelihoods[sample_index] = sample_likelihoods
        return likelihoods


    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under misclassification loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """
        from ...metrics import misclassification_error
        return misclassification_error(y, self.predict(X))
