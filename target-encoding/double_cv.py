import numpy as np
from sklearn.base import BaseEstimator
from sklearn.model_selection import GridSearchCV, cross_val_score


class DoubleSearchCV(BaseEstimator):
    """Performs double cross-validation and parameter optimization"""

    def __init__(
        self,
        estimator,
        param_grid,
        *,
        scoring=None,
        n_jobs=None,
        refit=True,
        inner_cv=None,
        outer_cv=None,
        verbose=0,
        pre_dispatch="2*n_jobs",
        error_score=np.nan,
        return_train_score=False
    ):
        """
        Creates instance of double cross-validation estimator
        :param estimator: algorithm to be cross-validated
        :param param_grid: grid of model parameters
        :param scoring: scoring function
        :param n_jobs: number of parallel tasks
        :param refit: refit the model
        :param inner_cv: inner cross validation object
        :param outer_cv: outer cross validation object
        :param verbose: verbosity level
        :param pre_dispatch: redispatch some number of tasks
        :param error_score: value to assign if error accuses
        :param return_train_score: include train score in result
        """
        self.best_score_ = None
        self.search = GridSearchCV(
            estimator,
            param_grid,
            scoring=scoring,
            n_jobs=n_jobs,
            refit=refit,
            cv=inner_cv,
            verbose=verbose,
            pre_dispatch=pre_dispatch,
            error_score=error_score,
            return_train_score=return_train_score,
        )
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.refit = refit
        self.inner_cv = inner_cv
        self.outer_cv = outer_cv
        self.verbose = verbose
        self.pre_dispatch = (pre_dispatch,)
        self.error_score = (error_score,)
        self.return_train_score = return_train_score

    def fit(self, X, y=None, **fit_params):
        scores = cross_val_score(
            self.search,
            X,
            y,
            cv=self.outer_cv,
            n_jobs=self.n_jobs,
            verbose=self.verbose,
            fit_params=fit_params,
        )
        self.best_score_ = scores.mean()

        return self
