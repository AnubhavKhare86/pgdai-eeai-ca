"""
testing_and_results.py
----------------------
Evaluation utilities that sit outside the BaseModel hierarchy.

These functions were written for early notebook experiments where the
model was a raw scikit-learn estimator rather than a BaseModel subclass.
They are kept for backwards compatibility.  New code should call
RandomForestModel.print_results() instead.

Authors: Anubhav Khare, Syam Sundar Gujjaru
Course : PGDAI - Engineering and Evaluating Artificial Intelligence
CA     : CA 1, 2026
"""

import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


def evaluate_classifier(classifier, X_test, y_test):
    """
    Evaluate a fitted classifier and print a full diagnostic report.

    Prints (in order):
      1. A probability DataFrame if the classifier supports predict_proba.
      2. The confusion matrix.
      3. A per-class classification report (precision, recall, F1).

    Args:
        classifier: A fitted scikit-learn estimator with a predict()
                    method.  predict_proba() is used if available.
        X_test:     Feature matrix for the test set.
        y_test:     True labels for the test set.

    Returns:
        A tuple of (y_pred, prob_df, accuracy) where:
          - y_pred   : numpy array of predicted labels.
          - prob_df  : pandas DataFrame of class probabilities, or None
                       if the classifier does not support predict_proba.
          - accuracy : float accuracy score on the test set.
    """
    y_pred  = classifier.predict(X_test)
    prob_df = None

    # Print class probabilities when the classifier supports them.
    if hasattr(classifier, 'predict_proba'):
        try:
            proba = classifier.predict_proba(X_test)
            prob_df = pd.DataFrame(proba)
            prob_df.columns = getattr(
                classifier, 'classes_',
                [f'class_{i}' for i in range(prob_df.shape[1])]
            )
            print(prob_df)
        except Exception:
            prob_df = None

    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    accuracy = accuracy_score(y_test, y_pred)
    return y_pred, prob_df, accuracy
