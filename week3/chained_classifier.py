"""
chained_classifier.py
---------------------
Design Decision 1: Chained Multi-Output classification.

The idea is to evaluate the same model type at three progressively
richer label levels on an identical test set:

  Step 1 - Type 2 only
  Step 2 - Type 2 + Type 3  (labels joined with '+')
  Step 3 - Type 2 + Type 3 + Type 4

Using a frozen train/test split across all three steps means the
accuracy numbers are directly comparable - they all reflect performance
on the same held-out rows.

The classifier accepts a model_factory callable rather than a concrete
class so it stays decoupled from any specific algorithm.  Only main.py
knows that the factory is RandomForestModel.

Authors: Anubhav Khare, Syam Sundar Gujjaru
Course : PGDAI - Engineering and Evaluating Artificial Intelligence
CA     : CA 1, 2026
"""

from typing import Callable, List, Tuple

import numpy as np
import pandas as pd

from week3.base_model import BaseModel
from week3.data_object import DataObject
from week3.config import TEST_FRACTION, RANDOM_STATE


class ChainedClassifier:
    """
    Evaluates a model at three chained label levels on a shared split.

    The three steps and their combined label columns are defined in
    CHAIN_STEPS.  Each step creates a fresh model instance via the
    factory, trains it, and prints results - all on the same frozen
    train/test indices so the results are directly comparable.

    Attributes:
        _model_factory:  Callable that returns a new BaseModel instance.
        _models:         List of trained model instances, one per step.
        _data_objects:   List of DataObjects, one per step.
    """

    # Each entry is (list_of_label_columns, display_label).
    # Labels are joined with '|' to match the CA specification format.
    CHAIN_STEPS: List[Tuple[list, str]] = [
        (['y2'],             'Step 1 - Type 2'),
        (['y2', 'y3'],       'Step 2 - Type 2 + Type 3'),
        (['y2', 'y3', 'y4'], 'Step 3 - Type 2 + Type 3 + Type 4'),
    ]

    def __init__(self, model_factory: Callable[..., BaseModel]):
        """
        Initialise the chained classifier.

        Args:
            model_factory: A callable (e.g. a class) that accepts a
                           'label' keyword argument and returns a new
                           BaseModel instance.
        """
        self._model_factory  = model_factory
        self._models: List[BaseModel]    = []
        self._data_objects: List[DataObject] = []

    def fit_evaluate(self, df: pd.DataFrame, X: np.ndarray) -> None:
        """
        Run all three chain steps and print results for each.

        Each step filters independently to rows that have all required
        label columns populated, then performs its own train/test split.
        This matches the CA specification where Step 1 uses all rows
        with a y2 label, Step 2 uses rows with y2+y3, and Step 3 uses
        rows with y2+y3+y4.

        Args:
            df: DataFrame with columns y2, y3, y4 (after preprocessing).
            X:  TF-IDF feature matrix aligned with df's rows.
        """
        for cols, label in self.CHAIN_STEPS:
            for col in cols:
                if col not in df.columns:
                    print(f"[skip] Missing column '{col}' - skipping {label}.")
                    continue

            # Filter to rows where all required label columns are non-empty.
            mask = df[cols].apply(
                lambda s: s.fillna('').astype(str).str.strip() != ''
            ).all(axis=1).to_numpy()

            if not mask.any():
                print(f'[skip] No rows with complete labels for {label}.')
                continue

            X_step  = X[mask]
            df_step = df.loc[mask].reset_index(drop=True)
            y       = self._compose_labels(df_step, cols)

            data  = DataObject(X_step, y, TEST_FRACTION, RANDOM_STATE)
            model: BaseModel = self._model_factory(label=label)
            model.train(data)
            model.predict(data)
            model.print_results(data)
            self._models.append(model)
            self._data_objects.append(data)

    @staticmethod
    def _compose_labels(df: pd.DataFrame, cols: list) -> np.ndarray:
        """
        Combine multiple label columns into a single compound label.

        Labels are joined with '|' so 'Problem/Fault' and 'Gallery-Use'
        become 'Problem/Fault|Gallery-Use'.  This matches the format
        expected by the CA specification.

        Args:
            df:   DataFrame containing the label columns.
            cols: List of column names to combine, in order.

        Returns:
            numpy array of compound label strings.
        """
        combined = df[cols[0]].fillna('').astype(str).str.strip()
        for col in cols[1:]:
            combined = combined + '|' + df[col].fillna('').astype(str).str.strip()
        return combined.to_numpy()
