"""
This file contains the functions needed for calculating TSFresh features on windowed
time series (one row per window, samples in 'sample_*' columns). Not tied to a dataset.
"""

from tsfresh import extract_features
from tsfresh.feature_extraction import MinimalFCParameters, ComprehensiveFCParameters
import pandas as pd

# Features that add no information for our windows, so they are dropped after extraction.
# All windows have the same length, which makes:
#   - value__length the same number for every window,
#   - value__sum_values equal to value__mean * length (redundant with the mean),
#   - value__variance equal to value__standard_deviation squared (redundant with the std).
IRRELEVANT_FEATURES = ['value__sum_values', 'value__variance', 'value__length']


def extract_tsfresh_features(df_phase: pd.DataFrame, phase_name: str):
    """
    Extracts TSFresh features for every window of one phase.

    Args:
        df_phase (pd.DataFrame): One row per window: the metadata columns plus the samples
            in 'sample_0', 'sample_1', ... columns (see windows_to_dataframe()).
        phase_name (str): Name of the phase, only used in the progress messages.

    Returns:
        pd.DataFrame: One row per window: the metadata columns followed by the TSFresh
        features, without the ones listed in IRRELEVANT_FEATURES.
    """
    print(f"Extracting TSFresh features for {phase_name}...")

    # Create a unique ID for each window (row)
    df_phase_copy = df_phase.reset_index().rename(columns={'index': 'id'})

    # Identify the sample columns dynamically
    sample_cols = [col for col in df_phase_copy.columns if col.startswith('sample_')]

    # Melt the DataFrame to long format (required by tsfresh)
    df_long = pd.melt(df_phase_copy,
                      id_vars=['id'],
                      value_vars=sample_cols,
                      var_name='time',
                      value_name='value')

    # Convert 'time' column from 'sample_X' to integer X
    df_long['time'] = df_long['time'].str.replace('sample_', '').astype(int)

    # Define custom feature parameters
    custom_fc_parameters = {
        'fft_coefficient': [{'coeff': i, 'attr': 'abs'} for i in range(1, 11)] + \
                           [{'coeff': i, 'attr': 'real'} for i in range(1, 11)],
        'abs_energy': None,
        'energy_ratio_by_chunks': [{'num_segments': 10, 'segment_focus': i} for i in range(10)],
        'fft_aggregated': [{'aggtype': 'centroid'}, {'aggtype': 'variance'}],
        #'spectral_entropy': None, # Removed as it's causing the AttributeError
        'permutation_entropy': [{'tau': 1, 'dimension': 8}]
    }

    # Combine MinimalFCParameters with custom parameters
    combined_fc_parameters = MinimalFCParameters()
    combined_fc_parameters.update(custom_fc_parameters)

    # Extract features
    features_df = extract_features(df_long,
                                   column_id='id',
                                   column_sort='time',
                                   column_value='value',
                                   default_fc_parameters=combined_fc_parameters,
                                   n_jobs=0, # Set n_jobs to 0 for single process as intended
                                   show_warnings=False)

    # Drop the features that add no information (see IRRELEVANT_FEATURES above)
    features_df = features_df.drop(columns=IRRELEVANT_FEATURES, errors='ignore')

    # Reset the index of features_df to make 'id' a regular column for merging
    # Explicitly name the new column 'id'
    features_df = features_df.reset_index(names=['id'])

    # Merge features back with original metadata (excluding sample columns)
    metadata_cols = [col for col in df_phase.columns if not col.startswith('sample_')]
    final_df = df_phase_copy[metadata_cols + ['id']].merge(features_df, on='id', how='left')
    final_df = final_df.drop(columns=['id']) # Drop the temporary id column

    print(f"Finished feature extraction for {phase_name}. Shape: {final_df.shape}")
    return final_df