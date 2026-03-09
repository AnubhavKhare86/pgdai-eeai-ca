"""
noise_remover.py
----------------
Cleans the raw text columns before they are fed into the TF-IDF
vectoriser.  Two columns are processed:

  - 'Ticket Summary'  -> stored in 'ts'
  - 'Interaction content' -> stored in 'ic'

The cleaning removes email boilerplate (greetings, sign-offs, dates,
phone signatures), punctuation, digits, and other tokens that carry
no classification signal.  An optional filter also drops rows whose
Type 1 label is too rare to be useful.

Authors: Anubhav Khare, Syam Sundar Gujjaru
Course : PGDAI - Engineering and Evaluating Artificial Intelligence
CA     : CA 1, 2026
"""

import pandas as pd


# Regex patterns that appear in Ticket Summary lines but add no signal.
_NOISE_TS = (
    r"(sv\s*:)"
    r"|(wg\s*:)"
    r"|(ynt\s*:)"
    r"|(fw(d)?\s*:)"
    r"|(r\s*:)"
    r"|(re\s*:)"
    r"|(\[|\])"
    r"|(aspiegel support issue submit)"
    r"|(null)"
    r"|(nan)"
    r"|((bonus place my )?support.pt :)"
)

# Ordered list of regex patterns applied to Interaction Content.
# Each pattern targets a specific category of noise (dates, greetings,
# legal boilerplate, etc.).  They are applied sequentially so earlier
# removals don't interfere with later ones.
_NOISE_IC = [
    r"(from :)|(subject :)|(sent :)|(r\s*:)|(re\s*:)",
    r"(january|february|march|april|may|june|july|august|september|october|november|december)",
    r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)",
    r"(monday|tuesday|wednesday|thursday|friday|saturday|sunday)",
    r"\d{2}(:|.)\d{2}",
    r"(xxxxx@xxxx\.com)|(\*{5}([a-z]+))",
    r"dear ((customer)|(user))",
    r"\bdear\b",
    r"(hello|hallo|hi |hi there)",
    r"good morning",
    r"thank you for your patience( during (our)? investigation)?( and cooperation)?",
    r"thank you for contacting us",
    r"thank you for your availability",
    r"thank you for providing us this information",
    r"thank you for contacting",
    r"thank you for reaching us( back)?",
    r"thank you for patience",
    r"thank you for (your )?reply",
    r"thank you for (your )?response",
    r"thank you for (your )?cooperation",
    r"thank you for providing us with more information",
    r"thank you very kindly",
    r"thank you( very much)?",
    r"i would like to follow up on the case you raised on( the date)?",
    r"i will do my very best to assist you",
    r"in order to give you the best solution",
    r"could you please clarify your request with following information:",
    r"in this matter",
    r"we hope you('re| are) doing (fine|well)",
    r"we apologize for the inconvenience",
    r"sent from my huawei (cell )?phone",
    r"original message",
    r"customer support team",
    r"(aspiegel )?se is a company incorporated under the laws of ireland with its headquarters in dublin, ireland\.",
    r"(aspiegel )?se is the provider of huawei mobile services to huawei and honor device owners in",
    r"canada, australia, new zealand and other countries",
    r"\d+",
    r"[^0-9a-zA-Z]+",
    r"(\s\n^).*(\s\n$)",
]


def apply_noise_filters(df: pd.DataFrame, filter_y1_min: int = 10) -> pd.DataFrame:
    """
    Apply text-cleaning rules to 'Ticket Summary' and 'Interaction content'.

    The function works on a copy of the DataFrame so the original is
    never mutated.  After cleaning, whitespace is collapsed and the
    result is stripped.

    Args:
        df:            DataFrame produced by data_selection.load_and_prepare.
        filter_y1_min: Rows whose Type 1 label appears fewer than this
                       many times are removed.  Set to 0 to skip this
                       filter entirely.  Default is 10.

    Returns:
        A new DataFrame with additional columns 'ts' (cleaned Ticket
        Summary) and 'ic' (cleaned Interaction Content).
    """
    temp = df.copy()

    # Clean Ticket Summary -> 'ts'
    if 'Ticket Summary' in temp.columns:
        temp['ts'] = (
            temp['Ticket Summary']
            .str.lower()
            .replace(_NOISE_TS, ' ', regex=True)
            .replace(r'\s+', ' ', regex=True)
            .str.strip()
        )

    # Clean Interaction Content -> 'ic'
    if 'Interaction content' in temp.columns:
        temp['ic'] = temp['Interaction content'].str.lower()
        for pattern in _NOISE_IC:
            temp['ic'] = temp['ic'].replace(pattern, ' ', regex=True)
        temp['ic'] = temp['ic'].replace(r'\s+', ' ', regex=True).str.strip()

    # Optionally drop rows whose Type 1 label is too rare.
    if 'y1' in temp.columns and filter_y1_min > 0:
        counts = temp['y1'].value_counts()
        valid = counts[counts > filter_y1_min].index
        temp = temp.loc[temp['y1'].isin(valid)]

    return temp
