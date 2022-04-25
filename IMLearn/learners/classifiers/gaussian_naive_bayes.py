from typing import NoReturn
from ...base import BaseEstimator
from numpy.linalg import slogdet, inv
import numpy as np

class GaussianNaiveBayes(BaseEstimator):
    """
    Gaussian Naive-Bayes classifier
    """
    def __init__(self):
        """
        Instantiate a Gaussian Naive Bayes classifier

        Attributes
        ----------
        self.classes_ : np.ndarray of shape (n_classes,)
            The different labels classes. To be set in `GaussianNaiveBayes.fit`

        self.mu_ : np.ndarray of shape (n_classes,n_features)
            The estimated features means for each class. To be set in `GaussianNaiveBayes.fit`

        self.vars_ : np.ndarray of shape (n_classes, n_features)
            The estimated features variances for each class. To be set in `GaussianNaiveBayes.fit`

        self.pi_: np.ndarray of shape (n_classes)
            The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
        """
        super().__init__()
        self.classes_, self.mu_, self.vars_, self.pi_ = None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a gaussian naive bayes model

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
            mu[index] = sum(X[np.where(y == k)]) / counts[
                np.where(unique == k)]
        self.mu_ = mu
        sigma = np.zeros([num_classes, num_features])
        for class_index, k in enumerate(self.classes_):
            samples = X[np.where(y == k)]
            for sample in samples:
                centered_sample = sample - self.mu_[class_index]
                sigma[class_index] += np.diag(np.outer(centered_sample, centered_sample))
            sigma[class_index] /= (counts[class_index] - num_classes)
        self.vars_ = sigma

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
            raise ValueError("Estimator must first be fitted before calling `likelihood` function")
        m = X.shape[0]
        d = X.shape[1]
        constant = (-d / 2) * np.log(2 * np.pi)
        likelihoods = np.zeros([m, self.classes_.shape[0]])
        for sample_index, sample in enumerate(X):
            sample_likelihoods = np.zeros(self.classes_.shape[0])
            for class_index, k in enumerate(self.classes_):
                log_pi_k = np.log(self.pi_[class_index])
                log_det = np.log(np.prod(self.vars_[class_index]))
                centered_sample = sample - self.mu_[class_index]
                sample_calc = np.matmul(np.transpose(centered_sample),
                                        np.matmul(np.diag(1 / self.vars_[class_index]),
                                                  centered_sample))
                sample_likelihoods[
                    class_index] = constant - 0.5 * log_det - 0.5 * sample_calc + log_pi_k
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
