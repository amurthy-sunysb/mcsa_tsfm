"""
One function to combine the feature tables of the three phases (isa, isb, isc)
into a single table, whichever way the features were made: TSFresh, MOMENT or Mantis.
Not tied to a dataset.
"""

import pandas as pd

# The supported feature types. Each one is also the prefix of its feature columns after
# merging, e.g. 'moment_a_17' is feature 17 of phase A from MOMENT.
FEATURE_TYPES = ("tsfresh", "moment", "mantis")

# The columns that identify a window. The three phases are matched on them.
KEY_COLUMNS = ["file_name", "window_num"]

# Window metadata that can come with a phase's table. Every other column is a feature.
METADATA_COLUMNS = ["file_name", "window_num", "window_size", "overlap", "signal_type"]


def rename_and_merge(df_a: pd.DataFrame,
                     df_b: pd.DataFrame,
                     df_c: pd.DataFrame,
                     feature_type: str) -> pd.DataFrame:
    """
    Renames the feature columns of the three phase tables and merges them into one table.

    Every column that is not window metadata is a feature, and is renamed to
    '<feature_type>_<phase>_<original name>':
        TSFresh:  value__mean, ...     ->  tsfresh_a_value__mean, ...
        MOMENT:   0, 1, ..., 1023      ->  moment_a_0, ..., moment_a_1023
        Mantis:   0, 1, ..., 255       ->  mantis_a_0, ..., mantis_a_255
    The tables are then joined on file_name and window_num. The other metadata columns
    (window_size, overlap, signal_type) are dropped: they are the same for every window
    or only say which phase a table is.

    Args:
        df_a (pd.DataFrame): Features of phase A (isa), one row per window, with at least
            'file_name' and 'window_num' columns.
        df_b (pd.DataFrame): Features of phase B (isb), same format.
        df_c (pd.DataFrame): Features of phase C (isc), same format.
        feature_type (str): How the features were made: "tsfresh", "moment" or "mantis".

    Returns:
        pd.DataFrame: One row per window, in the same order as df_a: file_name, window_num,
        then the features of phase A, phase B and phase C.

    Raises:
        ValueError: If feature_type is not supported, a table has no 'file_name' or
            'window_num' column, or the three tables do not have exactly the same windows.
    """
    if feature_type not in FEATURE_TYPES:
        raise ValueError(f"feature_type must be one of {FEATURE_TYPES}, not '{feature_type}'.")

    renamed_tables = []
    for phase, df in zip(["a", "b", "c"], [df_a, df_b, df_c]):
        missing = [col for col in KEY_COLUMNS if col not in df.columns]
        if missing:
            raise ValueError(f"The table of phase {phase} has no {missing} column(s).")

        feature_columns = [col for col in df.columns if col not in METADATA_COLUMNS]
        new_names = {col: f"{feature_type}_{phase}_{col}" for col in feature_columns}
        renamed_tables.append(df[KEY_COLUMNS + feature_columns].rename(columns=new_names))

    # Join the phases window by window. validate="one_to_one" makes pandas raise an
    # error if a window appears more than once in a table.
    df_a_renamed, df_b_renamed, df_c_renamed = renamed_tables
    df_merged = df_a_renamed.merge(df_b_renamed, on=KEY_COLUMNS, how="inner", validate="one_to_one")
    df_merged = df_merged.merge(df_c_renamed, on=KEY_COLUMNS, how="inner", validate="one_to_one")

    # The three phases are cut from the same recordings, so they must have the same windows.
    # An inner merge would silently drop a window that is missing from one phase, so check.
    if not (len(df_merged) == len(df_a) == len(df_b) == len(df_c)):
        raise ValueError(f"The phases do not have the same windows: {len(df_a)}, {len(df_b)} and "
                         f"{len(df_c)} rows went in, but only {len(df_merged)} windows are in all three.")

    return df_merged
