"""
training.py
-----------
Thin wrapper around scikit-learn's fit() method.

This module was introduced early in the project to keep the training
step explicit and easy to swap out.  For new code, prefer calling
RandomForestModel.train(data) directly through the BaseModel interface.

Authors: Anubhav Khare, Syam Sundar Gujjaru
Course : PGDAI - Engineering and Evaluating Artificial Intelligence
CA     : CA 1, 2026
"""


def fit_classifier(classifier, X_train, y_train):
    """
    Fit a scikit-learn compatible classifier on the provided data.

    Args:
        classifier: Any object that exposes a fit(X, y) method -
                    typically a scikit-learn estimator.
        X_train:    Feature matrix for training.
        y_train:    Label array for training.

    Returns:
        The fitted classifier (same object that was passed in).
    """
    return classifier.fit(X_train, y_train)
