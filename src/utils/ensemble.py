"""Soft-voting ensemble: averages the class probabilities of its member models.

Used by src/pipeline/train.py when a combination of models beats the single best model on the
validation fold. It is a plain scikit-learn-style estimator (fit / predict_proba / predict), so
the API, the model loader and MLflow treat it like any other model. Members must expose
predict_proba and be trained on the same label encoding (0..n_classes-1).
"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone


class SoftVotingEnsemble(ClassifierMixin, BaseEstimator):
    def __init__(self, members=None, names=None):
        self.members = members
        self.names = names

    def fit(self, X, y):
        self.members_ = [clone(m).fit(X, y) for m in self.members]
        self.classes_ = self.members_[0].classes_
        return self

    def predict_proba(self, X):
        return np.mean([m.predict_proba(X) for m in self.members_], axis=0)

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]
