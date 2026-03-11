"""
base_model.py
-------------
Defines the abstract contract that every model in this pipeline must
satisfy.  By programming to this interface rather than to a concrete
class, the rest of the codebase (ChainedClassifier, HierarchicalClassifier)
stays decoupled from any specific algorithm.

Swapping RandomForest for a different classifier only requires writing
a new subclass - nothing else changes.

Authors: Anubhav Khare, Syam Sundar Gujjaru
Course : PGDAI - Engineering and Evaluating Artificial Intelligence
CA     : CA 1, 2026
"""

from abc import ABC, abstractmethod

from week3.data_object import DataObject


class BaseModel(ABC):
    """
    Abstract base class for all classification models in the pipeline.

    Subclasses must implement three methods that map cleanly onto the
    standard ML workflow: fit the model, generate predictions, then
    print a human-readable summary of the results.

    The DataObject parameter carries X_train, X_test, y_train, y_test
    so each method has everything it needs without relying on shared
    mutable state.
    """

    @abstractmethod
    def train(self, data: DataObject) -> None:
        """
        Fit the model on the training portion of data.

        Args:
            data: A DataObject whose X_train and y_train attributes
                  are used for fitting.
        """
        ...

    @abstractmethod
    def predict(self, data: DataObject):
        """
        Generate predictions for the test portion of data.

        Args:
            data: A DataObject whose X_test attribute is used for
                  inference.

        Returns:
            A numpy array of predicted class labels.
        """
        ...

    @abstractmethod
    def print_results(self, data: DataObject) -> None:
        """
        Print accuracy and a full classification report to stdout.

        Implementations should call predict() internally if it has not
        been called yet, so callers can invoke print_results() directly
        without a separate predict() call.

        Args:
            data: The same DataObject used for training and prediction,
                  needed to access y_test for the report.
        """
        ...
