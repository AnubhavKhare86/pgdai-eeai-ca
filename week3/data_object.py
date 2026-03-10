"""
data_object.py
--------------
DataObject is a lightweight container that holds the four arrays
needed to train and evaluate one model instance:

    X_train, X_test, y_train, y_test

It also handles the rare-class problem: classes with very few samples
cannot be stratified into both train and test sets.  The _split helper
keeps those rare-class rows in training only, then adjusts the test
fraction so the overall test-set size stays close to the configured
TEST_FRACTION.

A second construction path (via train_idx / test_idx) lets multiple
model instances share the exact same split - this is used by
ChainedClassifier so all three chain steps are evaluated on identical
test samples.

Authors: Anubhav Khare, Syam Sundar Gujjaru
Course : PGDAI - Engineering and Evaluating Artificial Intelligence
CA     : CA 1, 2026
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from week3.config import TEST_FRACTION, RANDOM_STATE, MIN_CLASS_SAMPLES


class DataObject:
    """
    Encapsulates train/test arrays for a single model instance.

    Two construction modes:

    1. Default (train_idx and test_idx are None):
       Performs rare-class-aware splitting via _split().

    2. Frozen split (train_idx and test_idx provided):
       Reuses pre-computed index arrays so multiple DataObjects share
       the same samples.  No rare-class reshuffling is done in this
       mode because the caller is responsible for the split.

    Attributes:
        X_train: Feature matrix for training.
        X_test:  Feature matrix for evaluation.
        y_train: Label array for training.
        y_test:  Label array for evaluation.
    """

    def __init__(
        self,
        X,
        y,
        test_fraction: float = TEST_FRACTION,
        random_state: int = RANDOM_STATE,
        train_idx=None,
        test_idx=None,
    ):
        """
        Initialise a DataObject.

        Args:
            X:             Full feature matrix (numpy array or sparse).
            y:             Full label array.
            test_fraction: Desired fraction of data for the test set.
                           Only used in the default (non-frozen) path.
            random_state:  Random seed for reproducibility.
            train_idx:     Pre-computed training indices.  Must be
                           provided together with test_idx.
            test_idx:      Pre-computed test indices.  Must be provided
                           together with train_idx.
        """
        if train_idx is not None and test_idx is not None:
            # Frozen-split path: just index into the arrays directly.
            self.X_train = X[train_idx]
            self.y_train = y[train_idx]
            self.X_test  = X[test_idx]
            self.y_test  = y[test_idx]
        else:
            # Default path: handle rare classes then split.
            self.X_train, self.X_test, self.y_train, self.y_test = self._split(
                X, y, test_fraction, random_state
            )

    @staticmethod
    def _split(X, y, test_fraction: float, random_state: int):
        """
        Split X and y into train/test sets while keeping rare-class
        rows in the training set only.

        Classes with fewer than MIN_CLASS_SAMPLES examples cannot be
        reliably split, so they are excluded from the test set.  The
        test fraction is then scaled up slightly so the absolute number
        of test samples stays close to the intended size.

        Args:
            X:             Feature matrix.
            y:             Label array.
            test_fraction: Target fraction for the test set.
            random_state:  Random seed.

        Returns:
            Tuple of (X_train, X_test, y_train, y_test).
        """
        y_series = pd.Series(y)
        counts = y_series.value_counts()

        # Separate rows whose class has enough samples from those that don't.
        good_mask = y_series.isin(counts[counts >= MIN_CLASS_SAMPLES].index).values
        X_good, y_good = X[good_mask],  y[good_mask]
        X_rare, y_rare = X[~good_mask], y[~good_mask]

        # Edge case: every class is rare - return everything as training data.
        if X_good.shape[0] == 0:
            return X, np.empty((0, X.shape[1])), y, np.array([])

        # Adjust the test fraction so the test set is still ~TEST_FRACTION
        # of the *total* dataset even though rare rows are excluded.
        adjusted = min(max(X.shape[0] * test_fraction / X_good.shape[0], 0.0), 1.0)

        X_tr, X_te, y_tr, y_te = train_test_split(
            X_good, y_good, test_size=adjusted, random_state=random_state
        )

        # Append rare-class rows to training only.
        X_tr = np.concatenate([X_tr, X_rare], axis=0)
        y_tr = np.concatenate([y_tr, y_rare], axis=0)

        return X_tr, X_te, y_tr, y_te
