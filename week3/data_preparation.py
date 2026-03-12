"""
data_preparation.py
-------------------
Thin utility wrapper around scikit-learn's train_test_split.

This module was written early in the project when the split logic
lived outside DataObject.  It is kept for compatibility with older
notebook cells.  For new code, use DataObject which handles rare-class
edge cases automatically.

Authors: Anubhav Khare, Syam Sundar Gujjaru
Course : PGDAI - Engineering and Evaluating Artificial Intelligence
CA     : CA 1, 2026
"""

import numpy as np
from sklearn.model_selection import train_test_split


def split_train_test(
    X,
    y,
    test_size: float = 0.2,
    random_state: int = 0,
):
    """
    Split feature matrix X and label array y into train and test sets.

    Accepts both numpy arrays and pandas Series for y; converts to a
    numpy array internally so the return type is always consistent.

    Args:
        X:            Feature matrix.
        y:            Label array or pandas Series.
        test_size:    Fraction of data to reserve for testing.
                      Default is 0.2 (20 %).
        random_state: Random seed for reproducibility.

    Returns:
        Tuple of (X_train, X_test, y_train, y_test) as numpy arrays.
    """
    try:
        y_arr = y.to_numpy()
    except AttributeError:
        y_arr = np.asarray(y)

    return train_test_split(
        X, y_arr,
        test_size=test_size,
        random_state=random_state,
    )
