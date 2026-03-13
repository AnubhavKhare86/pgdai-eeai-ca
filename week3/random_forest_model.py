"""
random_forest_model.py
----------------------
Concrete implementation of BaseModel using scikit-learn's
RandomForestClassifier.

This is the only file in the pipeline that knows about Random Forest
specifically.  Everything else works through the BaseModel interface,
so swapping this out for a different algorithm only requires writing a
new subclass - the rest of the pipeline stays untouched.

Authors: Anubhav Khare, Syam Sundar Gujjaru
Course : PGDAI - Engineering and Evaluating Artificial Intelligence
CA     : CA 1, 2026
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

from week3.base_model import BaseModel
from week3.data_object import DataObject
from week3.config import RF_N_ESTIMATORS, RANDOM_STATE


class RandomForestModel(BaseModel):
    """
    Random Forest classifier wrapped in the BaseModel interface.

    A label string can be passed at construction time so that the
    printed results header identifies which level or step this model
    instance belongs to (e.g. 'Level 0', 'Step 1 - Type 2').

    Attributes:
        label:   Human-readable name shown in print_results output.
        _clf:    The underlying RandomForestClassifier instance.
        _y_pred: Cached predictions from the last predict() call.
    """

    def __init__(self, label: str = ''):
        """
        Initialise the model with an optional descriptive label.

        Args:
            label: Short string identifying this model instance in
                   printed output.  Defaults to an empty string.
        """
        self.label  = label
        self._clf   = RandomForestClassifier(
            n_estimators=RF_N_ESTIMATORS,
            random_state=RANDOM_STATE,
        )
        self._y_pred = None

    def train(self, data: DataObject) -> None:
        """
        Fit the Random Forest on the training data.

        Args:
            data: DataObject providing X_train and y_train.
        """
        self._clf.fit(data.X_train, data.y_train)

    def predict(self, data: DataObject):
        """
        Run inference on the test data and cache the predictions.

        Args:
            data: DataObject providing X_test.

        Returns:
            numpy array of predicted class labels.
        """
        self._y_pred = self._clf.predict(data.X_test)
        return self._y_pred

    def print_results(self, data: DataObject) -> None:
        """
        Print accuracy and a per-class classification report.

        Calls predict() automatically if it has not been called yet,
        so this method can be used as a one-shot evaluation step.

        Args:
            data: DataObject providing X_test and y_test.
        """
        if self._y_pred is None:
            self.predict(data)

        acc = accuracy_score(data.y_test, self._y_pred)
        print(f'\n=== {self.label} ===')
        print(f'Accuracy : {acc:.4f}')
        print(classification_report(data.y_test, self._y_pred, zero_division=0))
