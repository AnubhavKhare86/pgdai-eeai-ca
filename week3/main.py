"""
main.py
-------
Entry point for Design Decision 1: Chained Multi-Output classification.

Wires together the preprocessing, feature engineering, and modelling
components in the correct order.  The controller is the only module
that imports a concrete model class (RandomForestModel); everything
downstream works through the BaseModel interface.

Usage:
    python -m week3.main --csv week3/data/AppGallery.csv
    python -m week3.main --csv week3/data/AppGallery.csv --use-translation

Authors: Anubhav Khare, Syam Sundar Gujjaru
Course : PGDAI - Engineering and Evaluating Artificial Intelligence
CA     : CA 1, 2026
"""

import argparse
import os
import sys

import pandas as pd

from week3.data_selection import load_and_prepare
from week3.noise_remover import apply_noise_filters
from week3.text_representation import make_tfidf_features
from week3.chained_classifier import ChainedClassifier
from week3.random_forest_model import RandomForestModel
from week3.config import TEXT_COL_1, TEXT_COL_2


def maybe_translate(df: pd.DataFrame, use_translation: bool) -> pd.DataFrame:
    """
    Optionally translate non-English text to English.

    Translation is skipped when use_translation is False, which is the
    default.  If the translation module cannot be imported (e.g. stanza
    or transformers are not installed) a warning is printed and the
    original DataFrame is returned unchanged.

    Args:
        df:              DataFrame with a 'ts' column to translate.
        use_translation: When True, run the translation pipeline.

    Returns:
        The original DataFrame, or a copy with an added 'ts_en' column
        containing English translations of the 'ts' column.
    """
    if not use_translation:
        return df

    try:
        from week3.translation import trans_to_en
    except Exception as exc:
        print(f'[translation] Skipped - {exc}')
        return df

    df = df.copy()
    df['ts_en'] = trans_to_en(df['ts'].fillna('').tolist())
    return df


def main():
    """
    Parse command-line arguments and run the full DD1 pipeline.

    The pipeline runs in four stages:
      1. Load and prepare the raw CSV.
      2. Apply noise filters to the text columns.
      3. Optionally translate non-English text.
      4. Build TF-IDF features and run ChainedClassifier.
    """
    ap = argparse.ArgumentParser(
        description='DD1 - Chained Multi-Output email classification.'
    )
    ap.add_argument('--csv', required=True,
                    help='Path to the AppGallery CSV file.')
    ap.add_argument('--use-translation', action='store_true',
                    help='Translate non-English text to English before vectorising.')
    args = ap.parse_args()

    if not os.path.exists(args.csv):
        print(f'ERROR: File not found: {args.csv}')
        sys.exit(2)

    # Stage 1 - load and label-map the raw data.
    df = load_and_prepare(args.csv)
    print(f'[info] Loaded: {df.shape}')

    # Stage 2 - clean the text columns.
    df = apply_noise_filters(df)

    # Stage 3 - optional translation.
    df = maybe_translate(df, args.use_translation)

    # Decide which Ticket Summary column to use.
    text_col2 = 'ts_en' if 'ts_en' in df.columns else TEXT_COL_2

    for col in (TEXT_COL_1, text_col2):
        if col not in df.columns:
            print(f"ERROR: column '{col}' not found. Available: {sorted(df.columns)}")
            sys.exit(3)

    # Stage 4 - build features and run the chained classifier.
    X, _ = make_tfidf_features(df, TEXT_COL_1, text_col2)
    print(f'[info] Feature matrix: {X.shape}')

    chained = ChainedClassifier(model_factory=RandomForestModel)
    chained.fit_evaluate(df, X)


if __name__ == '__main__':
    main()
