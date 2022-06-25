from collections import defaultdict

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_array


class TargetEncoder(BaseEstimator, TransformerMixin):
    """Sklearn transformer that encodes categorical features using target-encoding"""

    def __init__(
        self, alpha: float, handle_unknown: str = "nan", copy: bool = True
    ) -> None:
        """Creates an instance of target encoder transformer
        :param alpha: smoothing parameter
        :param handle_unknown: what to do with unknown categories
        :param copy: copy array during transformation
        """
        self.alpha = alpha
        self.y_mean = None
        self.mapping = defaultdict(dict)
        self.handle_unknown = handle_unknown
        self.copy = copy

    def fit(self, X, y=None, **fit_params) -> "TargetEncoder":
        """Fits target-encoder"""
        check_array(array=X, dtype="object", estimator="TargetEncoder")
        check_array(array=y, dtype="numeric", estimator="TargetEncoder")
        self.y_mean = np.mean(y)

        for i in range(X.shape[1]):
            x = X[:, i]
            for category in np.unique(x):
                mask = x == category
                y_mean = np.mean(x[mask])
                k_count = len(x[mask])
                self.mapping[i][category] = (
                    y_mean * k_count + self.y_mean * self.alpha
                ) / k_count + self.alpha

        return self

    def transform(self, X):
        """Transforms using fitted mappings"""
        check_array(array=X, dtype="object", estimator="TargetEncoder")
        X_ = X
        if self.copy:
            X_ = X.copy()

        for i in range(X.shape[1]):
            for category, mapping in self.mapping[i]:
                mask = X_[:, i] == category
                X_[mask, i] = mapping

        return X_
