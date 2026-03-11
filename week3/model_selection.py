"""
model_selection.py
------------------
Factory helper that constructs a configured RandomForestClassifier.

This thin wrapper exists so notebook experiments and older scripts
can create a classifier with the project's default hyperparameters
without importing config.py directly.  New code should prefer
RandomForestModel from random_forest_model.py, which integrates with
the BaseModel interface.

Authors: Anubhav Khare, Syam Sundar Gujjaru
Course : PGDAI - Engineering and Evaluating Artificial Intelligence
CA     : CA 1, 2026
"""

from sklearn.ensemble import RandomForestClassifier


def make_random_forest(
    n_estimators: int = 1000,
    random_state: int = 0,
) -> RandomForestClassifier:
    """
    Build and return a RandomForestClassifier with sensible defaults.

    Args:
        n_estimators: Number of trees in the forest.  1000 gives
                      stable out-of-bag estimates on the AppGallery
                      dataset without being too slow.
        random_state: Seed for the random number generator.  Fix this
                      to get reproducible results across runs.

    Returns:
        An unfitted RandomForestClassifier instance.
    """
    return RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
    )
