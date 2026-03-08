"""
data_selection.py
-----------------
Responsible for reading the raw CSV and producing a clean DataFrame
that the rest of the pipeline can work with.

The function maps the original column names ('Type 2', 'Type 3',
'Type 4') to short aliases (y2, y3, y4) so downstream code never
has to deal with spaces in column names.  Rows that have no Type 2
label at all are dropped here - they cannot be used for training or
evaluation.

Authors: Anubhav Khare, Syam Sundar Gujjaru
Course : PGDAI - Engineering and Evaluating Artificial Intelligence
CA     : CA 1, 2026
"""

import pandas as pd


def load_and_prepare(csv_path: str) -> pd.DataFrame:
    """
    Load the AppGallery CSV and return a cleaned DataFrame.

    Steps performed:
      1. Read the CSV from disk.
      2. Cast 'Interaction content' and 'Ticket Summary' to unicode
         strings so NaN values become the literal string 'nan' rather
         than a float - easier to handle downstream.
      3. Map Type 2 / Type 3 / Type 4 columns to y2 / y3 / y4 and
         strip leading/trailing whitespace from every label.
      4. Copy 'Interaction content' into a convenience column 'x'.
      5. Drop any row where y2 is empty or missing - those rows have
         no usable target label for the first classification level.

    Args:
        csv_path: Absolute or relative path to the AppGallery CSV file.

    Returns:
        A pandas DataFrame with at minimum the columns:
        'y2', 'y3', 'y4', 'x', 'Interaction content', 'Ticket Summary'.
    """
    df = pd.read_csv(csv_path)

    # Force text columns to unicode so downstream string ops are safe.
    for col in ['Interaction content', 'Ticket Summary']:
        if col in df.columns:
            df[col] = df[col].astype('U')

    # Rename label columns to short aliases and strip whitespace.
    for src, dst in [('Type 2', 'y2'), ('Type 3', 'y3'), ('Type 4', 'y4')]:
        if src in df.columns:
            df[dst] = df[src].astype(str).str.strip()

    # Convenience alias for the main text feature.
    if 'Interaction content' in df.columns:
        df['x'] = df['Interaction content']

    # Keep only rows that have a valid Type 2 label.
    df = df.loc[df['y2'].notna() & (df['y2'] != '') & (df['y2'] != 'nan')]

    return df
