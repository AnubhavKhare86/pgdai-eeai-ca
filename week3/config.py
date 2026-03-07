"""
config.py
---------
Central place for all tunable constants used across the pipeline.
Keeping them here means you only need to touch one file when
experimenting with different TF-IDF sizes, train/test ratios, or
Random Forest settings.

Authors: Anubhav Khare, Syam Sundar Gujjaru
Course : PGDAI - Engineering and Evaluating Artificial Intelligence
CA     : CA 1, 2026
"""

# Column names that hold the three label levels after preprocessing.
TARGET_COLS = ['Type 2', 'Type 3', 'Type 4']

# Names of the two cleaned text columns produced by noise_remover.py.
# 'ic' = Interaction Content,  'ts' = Ticket Summary.
TEXT_COL_1 = 'ic'
TEXT_COL_2 = 'ts'

# TF-IDF vocabulary size.  2000 keeps the feature matrix manageable
# while still capturing the most discriminative terms.
TFIDF_MAX_FEATURES = 2000

# Ignore terms that appear in fewer than MIN_DF documents - likely
# typos or one-off tokens that won't generalise to unseen data.
TFIDF_MIN_DF = 4

# Ignore terms that appear in more than MAX_DF fraction of all
# documents - they are so common they carry no class signal.
TFIDF_MAX_DF = 0.90

# Fraction of the dataset held out for testing.
TEST_FRACTION = 0.20

# Fixed seed so every run produces the same train/test split and the
# same Random Forest bootstrap samples - important for reproducibility.
RANDOM_STATE = 0

# Number of trees in the Random Forest.  1000 gives stable estimates
# without being prohibitively slow on the AppGallery dataset.
RF_N_ESTIMATORS = 1000

# Classes with fewer than this many samples are kept in training only.
# They are never placed in the test set to avoid zero-support warnings
# in the classification report.
MIN_CLASS_SAMPLES = 3
