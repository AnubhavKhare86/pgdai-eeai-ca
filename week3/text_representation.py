"""
text_representation.py
-----------------------
Converts the cleaned text columns into a numeric feature matrix using
TF-IDF (Term Frequency - Inverse Document Frequency).

Both text columns ('ic' and 'ts') are vectorised with a *shared*
vocabulary: the vectoriser is fitted on the concatenation of both
columns so terms that appear in either column are included.  The two
resulting matrices are then concatenated column-wise, giving each row
a feature vector of length 2 * TFIDF_MAX_FEATURES.

Authors: Anubhav Khare, Syam Sundar Gujjaru
Course : PGDAI - Engineering and Evaluating Artificial Intelligence
CA     : CA 1, 2026
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from week3.config import TFIDF_MAX_FEATURES, TFIDF_MIN_DF, TFIDF_MAX_DF


def make_tfidf_features(
    df: pd.DataFrame,
    text_col_1: str,
    text_col_2: str,
    max_features: int = TFIDF_MAX_FEATURES,
    min_df: int = TFIDF_MIN_DF,
    max_df: float = TFIDF_MAX_DF,
):
    """
    Build a TF-IDF feature matrix from two text columns.

    The vectoriser is fitted on the union of both columns so the
    vocabulary reflects terms from both sources.  Each column is then
    transformed independently and the two dense arrays are concatenated
    horizontally.

    Args:
        df:           DataFrame containing the text columns.
        text_col_1:   Name of the first text column (typically 'ic').
        text_col_2:   Name of the second text column (typically 'ts').
        max_features: Maximum vocabulary size.  Terms beyond this limit
                      (ranked by corpus frequency) are discarded.
        min_df:       Minimum document frequency.  Terms that appear in
                      fewer documents are ignored.
        max_df:       Maximum document frequency as a fraction.  Terms
                      that appear in more than this fraction of documents
                      are treated as stop words.

    Returns:
        A tuple of:
          - X          : numpy array of shape (n_rows, 2 * max_features)
          - vectorizer : the fitted TfidfVectorizer instance, useful if
                         you need to transform new data later.
    """
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        min_df=min_df,
        max_df=max_df,
    )

    # Fit on the combined corpus so both columns share one vocabulary.
    col1_texts = df[text_col_1].fillna('').astype(str).tolist()
    col2_texts = df[text_col_2].fillna('').astype(str).tolist()
    vectorizer.fit(col1_texts + col2_texts)

    # Transform each column separately, then join them side by side.
    x1 = vectorizer.transform(col1_texts).toarray()
    x2 = vectorizer.transform(col2_texts).toarray()
    X  = np.concatenate((x1, x2), axis=1)

    return X, vectorizer
