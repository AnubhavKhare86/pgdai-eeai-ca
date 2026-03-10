"""
dealing_data_imbalance.py
-------------------------
Standalone utility for splitting a dataset while ensuring that
rare-class samples never end up in the test set.

This module predates DataObject and is kept for backwards
compatibility with older notebook experiments.  New code should
use DataObject directly, which wraps the same logic.

Authors: Anubhav Khare, Syam Sundar Gujjaru
Course : PGDAI - Engineering and Evaluating Artificial Intelligence
CA     : CA 1, 2026
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def train_test_with_rare_in_train(
    X,
    y,
    base_test_fraction: float = 0.20,
    random_state: int = 0,
):
    """
    Split X and y into train/test sets, keeping rare-class rows in
    the training set only.

    Any class that has fewer than 3 samples is considered 'rare'.
    Those rows are excluded from the test set so the classifier is
    never asked to predict a class it has barely seen.  The test
    fraction is adjusted upward to compensate, keeping the absolute
    test-set size close to base_test_fraction * len(y).

    Args:
        X:                  Feature matrix (numpy array or sparse).
        y:                  Label array or pandas Series.
        base_test_fraction: Desired fraction of the *total* dataset
                            to use for testing.  Default is 0.20.
        random_state:       Random seed for reproducibility.

    Returns:
        Tuple of (X_train, X_test, y_train, y_test) as numpy arrays.
    """
    y_series = pd.Series(y)
    counts = y_series.value_counts()

    # Classes with at least 3 samples can safely appear in the test set.
    good_classes = counts[counts >= 3].index
    mask_good = y_series.isin(good_classes)

    X_good, y_good = X[mask_good],  y[mask_good]
    X_rare, y_rare = X[~mask_good], y[~mask_good]

    # Edge case: nothing is 'good' - return everything as training data.
    if X_good.shape[0] == 0:
        empty = np.empty((0,) + X.shape[1:]) if hasattr(X, 'shape') else []
        return X, empty, y, []

    # Scale the test fraction so the absolute test count stays consistent.
    adjusted = X.shape[0] * base_test_fraction / X_good.shape[0]
    adjusted = min(max(adjusted, 0.0), 1.0)

    X_train, X_test, y_train, y_test = train_test_split(
        X_good, y_good, test_size=adjusted, random_state=random_state
    )

    # Rare-class rows go into training only.
    X_train = np.concatenate((X_train, X_rare), axis=0)
    y_train = np.concatenate((y_train, y_rare), axis=0)

    return X_train, X_test, y_train, y_test
